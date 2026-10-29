"""Main request flow contracts, using the real graph with offline boundaries."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import src.agent.graph as graph_module
from src.agent.agents import knowledge, planner
from src.agent.routing.policy import rules_route


@pytest.fixture
def memory(monkeypatch):
    import config.settings as settings_module
    from src.agent import memory as memory_module
    settings = settings_module.get_settings()
    monkeypatch.setattr(settings, 'mem0_enabled', True)
    monkeypatch.setattr(settings, 'routing_provider', 'rules')
    manager = SimpleNamespace(
        search=AsyncMock(return_value=[]),
        filter_memories_for_current_user=Mock(return_value=([], {})),
        format_memories_for_context=Mock(return_value='fresh memory'),
    )
    monkeypatch.setattr(memory_module, 'get_mem0_manager', lambda: manager)
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize('question', ['1+1', '现在几点', '将 1 GiB 转为 MiB', '你好'])
async def test_self_contained_tools_and_greetings_skip_memory(memory, question):
    result = await graph_module.retrieve_mem0_memories_node({
        'messages': [HumanMessage(content=question)],
        'route_decision': rules_route(question).to_dict(), 'mem0_memories': 'stale',
    })
    assert result == {'mem0_memories': ''}
    memory.search.assert_not_called()


@pytest.mark.asyncio
async def test_greeting_prefix_does_not_hide_a_real_memory_question(memory):
    question = '你好，我之前关注哪个研究方向？'
    await graph_module.retrieve_mem0_memories_node({
        'messages': [HumanMessage(content=question)], 'route_decision': rules_route(question).to_dict(),
    })
    memory.search.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['disabled', 'empty', 'denied', 'error', 'timeout'])
async def test_memory_never_reuses_previous_turn_after_failure(monkeypatch, memory, failure):
    from config.settings import get_settings
    if failure == 'disabled': monkeypatch.setattr(get_settings(), 'mem0_enabled', False)
    if failure == 'denied': memory.search.return_value = [{'memory': 'restricted'}]
    if failure == 'error': memory.search.side_effect = RuntimeError('private error')
    if failure == 'timeout':
        async def slow(**kwargs): await asyncio.sleep(1)
        memory.search.side_effect = slow
        monkeypatch.setattr(graph_module, '_AUXILIARY_TIMEOUT_SECONDS', .01)
    result = await graph_module.retrieve_mem0_memories_node({
        'messages': [HumanMessage(content='我之前关注什么？')], 'mem0_memories': 'stale secret',
    })
    assert result == {'mem0_memories': ''}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['error', 'timeout', 'empty'])
async def test_summary_failure_retains_original_history(monkeypatch, failure):
    from config.settings import get_settings
    import src.models.llm as llm_module
    settings = get_settings()
    monkeypatch.setattr(settings, 'summary_threshold', 2)
    monkeypatch.setattr(settings, 'summary_keep_recent', 1)
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content='')))
    if failure == 'error': model.ainvoke.side_effect = RuntimeError('private error')
    if failure == 'timeout':
        async def slow(prompt): await asyncio.sleep(1)
        model.ainvoke.side_effect = slow
        monkeypatch.setattr(graph_module, '_AUXILIARY_TIMEOUT_SECONDS', .01)
    monkeypatch.setattr(llm_module, 'get_llm', lambda **kwargs: model)
    messages = [HumanMessage(content='old input', id='u1'), AIMessage(content='old answer', id='a1'),
                HumanMessage(content='current', id='u2')]
    state = {'messages': messages, 'summary': 'existing summary'}
    result = await graph_module.maybe_summarize_node(state)
    assert result == {}
    assert state['messages'] == messages and state['summary'] == 'existing summary'


@pytest.mark.asyncio
async def test_successful_summary_only_removes_summarized_messages(monkeypatch):
    from config.settings import get_settings
    import src.models.llm as llm_module
    monkeypatch.setattr(get_settings(), 'summary_threshold', 2)
    monkeypatch.setattr(get_settings(), 'summary_keep_recent', 1)
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content='new summary')))
    monkeypatch.setattr(llm_module, 'get_llm', lambda **kwargs: model)
    result = await graph_module.maybe_summarize_node({'messages': [
        HumanMessage(content='old', id='u1'), AIMessage(content='answer', id='a1'),
        HumanMessage(content='current', id='u2')], 'summary': 'previous summary'})
    assert result['summary'] == 'new summary'
    assert [message.id for message in result['messages']] == ['u1', 'a1']


@pytest.mark.asyncio
async def test_actual_main_graph_retrieval_outage_skips_generation(monkeypatch, memory):
    import src.models.llm as llm_module
    from config.settings import get_settings
    monkeypatch.setattr(get_settings(), 'mem0_enabled', False)
    model_factory = Mock(side_effect=AssertionError('Outage must not call generation'))
    monkeypatch.setattr(llm_module, 'get_llm', model_factory)
    monkeypatch.setattr(knowledge, '_retrieve_documents', AsyncMock(side_effect=RuntimeError('private://credentials')))
    graph = graph_module.create_multi_agent_graph().compile(checkpointer=MemorySaver())
    result = await graph.ainvoke({'messages': [HumanMessage(content='实验室制度是什么')],
                                  'user_id': 'student'}, {'configurable': {'thread_id': 'outage'}})
    assert result['retrieval_decision'] == 'error'
    assert result['generation_metrics']['llm_calls'] == 0
    assert result['retrieved_docs'] == [] and result['sources'] == ''
    assert '暂时不可用' in result['final_answer']
    assert 'private://' not in str(result)
    assert len([m for m in result['messages'] if isinstance(m, AIMessage)]) == 1
    model_factory.assert_not_called()


@pytest.mark.asyncio
async def test_empty_retrieval_and_backend_failure_remain_different():
    result = await knowledge.generation_agent_node({
        'messages': [HumanMessage(content='问题')], 'retrieval_decision': 'no_results'})
    assert '没有找到' in result['final_answer']
    assert '暂时不可用' not in result['final_answer']


@pytest.mark.asyncio
async def test_real_graph_routes_once_and_skips_memory_for_next_calculation(monkeypatch, memory):
    from src.agent import memory as memory_module
    memory.search.return_value = [{'memory': 'prefers concise answers'}]
    memory.filter_memories_for_current_user.return_value = (memory.search.return_value, {})
    original_planner = graph_module.planner_node
    planned = []
    async def record_plan(state):
        planned.append(state['messages'][-1].content)
        return await original_planner(state)
    monkeypatch.setattr(graph_module, 'planner_node', record_plan)
    async def retrieve(state):
        assert state['route_decision']['handler'] == 'knowledge_agent'
        assert state['mem0_memories'] == 'fresh memory'
        return {'retrieved_docs': [Document(page_content='evidence')], 'retrieval_decision': 'high'}
    async def generate(state):
        return {'final_answer': 'answer', 'sources': 'knowledge_base', 'used_agent': 'knowledge_agent'}
    async def operate(state):
        assert state['mem0_memories'] == '' and state['retrieved_docs'] == []
        assert state['research_trace'] == {} and state['evidence_package'] == {}
        return {'final_answer': '2', 'used_agent': 'operation_agent'}
    monkeypatch.setattr(graph_module, 'retrieval_agent_node', retrieve)
    monkeypatch.setattr(graph_module, 'generation_agent_node', generate)
    monkeypatch.setattr(graph_module, 'operation_agent_node', operate)
    graph = graph_module.create_multi_agent_graph().compile(checkpointer=MemorySaver())
    config = {'configurable': {'thread_id': 'two-turns'}}
    first = await graph.ainvoke({'messages': [HumanMessage(content='实验室制度是什么')],
                                 'research_trace': {'old': 'trace'}, 'evidence_package': {'old': 'doc'}}, config)
    assert first['final_answer'] == 'answer'
    second = await graph.ainvoke({'messages': [HumanMessage(content='1+1')]}, config)
    assert second['final_answer'] == '2' and second['sources'] == ''
    assert planned == ['实验室制度是什么', '1+1']
    memory.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_auxiliary_failure_does_not_stop_real_tool_branch(monkeypatch, memory):
    from config.settings import get_settings
    import src.models.llm as llm_module
    monkeypatch.setattr(get_settings(), 'summary_threshold', 2)
    monkeypatch.setattr(get_settings(), 'summary_keep_recent', 1)
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError('summary unavailable')))
    monkeypatch.setattr(llm_module, 'get_llm', lambda **kwargs: model)
    async def operate(state):
        assert state['summary'] == 'existing summary'
        assert [m.content for m in state['messages']] == ['old', 'previous', '1+1']
        return {'final_answer': '2', 'used_agent': 'operation_agent'}
    monkeypatch.setattr(graph_module, 'operation_agent_node', operate)
    graph = graph_module.create_multi_agent_graph().compile()
    result = await graph.ainvoke({'messages': [HumanMessage(content='old'), AIMessage(content='previous'),
        HumanMessage(content='1+1')], 'summary': 'existing summary'})
    assert result['final_answer'] == '2'
    memory.search.assert_not_called()

@pytest.mark.asyncio
async def test_evidence_gate_blocks_error_and_empty_without_generation():
    from src.agent.graph import evidence_gate_node, route_after_evidence_gate
    error = await evidence_gate_node({'retrieval_decision': 'error', 'retrieved_docs': []})
    assert error['evidence_gate']['reason'] == 'retrieval_failed'
    assert route_after_evidence_gate(error) == 'finalize_response'
    empty = await evidence_gate_node({'retrieval_decision': 'no_results', 'retrieved_docs': []})
    assert empty['evidence_gate']['reason'] == 'no_authorized_evidence'
    assert route_after_evidence_gate(empty) == 'finalize_response'


@pytest.mark.asyncio
async def test_evidence_gate_approves_active_authorized_documents():
    from src.agent.graph import evidence_gate_node, route_after_evidence_gate
    doc = Document(page_content='公开证据', metadata={'confidentiality': 'internal', 'visibility': 'public', 'ingestion_state': 'active'})
    result = await evidence_gate_node({'retrieval_decision': 'high', 'retrieved_docs': [doc], 'user_context': {
        'username': 'student', 'role': 'student', 'is_active': True,
    }})
    assert result['evidence_gate']['status'] == 'approved'
    assert result['retrieved_docs'] == [doc]
    assert route_after_evidence_gate(result) == 'generation_agent'
