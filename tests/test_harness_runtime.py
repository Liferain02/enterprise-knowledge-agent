import asyncio
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from src.agent.harness.runtime import BoundedToolAgent, BudgetExceeded, ExecutionRun


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0
        self.inputs = []

    def bind_tools(self, tools):
        self.tool_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.inputs.append(list(messages))
        return next(self.replies)


def budget(**overrides):
    values = dict(model_limit=4, tool_limit=6, deadline_seconds=3, tool_timeout=1, input_limit=60000, output_limit=16000)
    return ExecutionRun(**(values | overrides))


def request(run):
    return {'configurable': {'execution_run': run}}


def call(name='echo', args=None, id='t1'):
    return {'name': name, 'args': args or {'text': 'secret experiment'}, 'id': id, 'type': 'tool_call'}


def echo(text: str) -> str:
    """Return provided text."""
    return text


@pytest.mark.asyncio
async def test_real_tool_loop_preserves_protocol_and_redacts_report():
    model = ScriptedModel([AIMessage(content='', tool_calls=[call()]), AIMessage(content='done')])
    run = budget()
    agent = BoundedToolAgent(model, [StructuredTool.from_function(echo)], 'Only echo')
    result = await agent.ainvoke({'messages': [HumanMessage(content='please echo')]}, request(run))
    assert model.inputs[1][-1].tool_call_id == 't1'
    assert model.inputs[1][-1].content == 'secret experiment'
    assert result['harness_report']['tool_calls'] == 1
    assert result['harness_report']['model_calls'] == 2
    assert 'secret experiment' not in str(result['harness_report'])
    assert [event['sequence'] for event in run.events] == list(range(len(run.events)))


@pytest.mark.asyncio
@pytest.mark.parametrize('calls,limit,error', [
    ([call(), call(id='t2')], 1, BudgetExceeded),
    ([call(), call(name='not_allowed', id='t2')], 5, ValueError),
    ([call(), call(id='t1')], 5, ValueError),
])
async def test_rejected_batch_executes_no_tools(calls, limit, error):
    invoked = []
    def record(text: str) -> str:
        """Record input."""
        invoked.append(text)
        return text
    tool = StructuredTool.from_function(record, name='echo')
    agent = BoundedToolAgent(ScriptedModel([AIMessage(content='', tool_calls=calls)]), [tool], '')
    with pytest.raises(error):
        await agent.ainvoke({'messages': []}, request(budget(tool_limit=limit)))
    assert invoked == []


@pytest.mark.asyncio
async def test_model_call_limit_stops_infinite_tool_loop():
    model = ScriptedModel([AIMessage(content='', tool_calls=[call()]), AIMessage(content='', tool_calls=[call(id='t2')])])
    agent = BoundedToolAgent(model, [StructuredTool.from_function(echo)], '')
    run = budget(model_limit=2)
    with pytest.raises(BudgetExceeded, match='model_calls'):
        await agent.ainvoke({'messages': []}, request(run))
    assert model.calls == 2 and run.tool_calls == 2


@pytest.mark.asyncio
async def test_async_tool_timeout_stops_task():
    async def slow(text: str) -> str:
        """Simulate a slow tool."""
        await asyncio.sleep(1)
        return text
    tool = StructuredTool.from_function(coroutine=slow, name='echo')
    agent = BoundedToolAgent(ScriptedModel([AIMessage(content='', tool_calls=[call()])]), [tool], '')
    with pytest.raises(asyncio.TimeoutError):
        await agent.ainvoke({'messages': []}, request(budget(tool_timeout=.01)))


@pytest.mark.asyncio
async def test_oversized_source_is_not_silently_truncated():
    model = ScriptedModel([])
    agent = BoundedToolAgent(model, [], '')
    with pytest.raises(BudgetExceeded, match='input_characters'):
        await agent.ainvoke({'messages': [HumanMessage(content='x'*100)]}, request(budget(input_limit=10)))
    assert model.calls == 0


@pytest.mark.asyncio
async def test_json_tool_error_is_preserved_for_model():
    def invalid(text: str) -> str:
        """Return a business error."""
        return '{"ok":false,"error":"missing data"}'
    model = ScriptedModel([AIMessage(content='', tool_calls=[call()]), AIMessage(content='请补充数据')])
    agent = BoundedToolAgent(model, [StructuredTool.from_function(invalid, name='echo')], '')
    result = await agent.ainvoke({'messages': []}, request(budget()))
    assert any(event['kind'] == 'tool_failed' for event in result['harness_report']['events'])
    assert 'missing data' in model.inputs[1][-1].content


def test_calculator_cannot_execute_python_objects():
    from src.agent.skills.calculator.scripts.tools import calculator
    assert calculator('sqrt(16)+2**3').endswith('12.0')
    for value in ["().__class__.__base__.__subclasses__()", '[0]*100000000', '2**10000000', "__import__('os')"]:
        assert calculator(value).startswith('计算错误:')


def test_skill_loading_does_not_execute_zero_argument_tool(tmp_path):
    from src.agent.skills.skill_loader import SkillLoader
    directory = tmp_path / 'safe'
    (directory/'scripts').mkdir(parents=True)
    (directory/'Skill.md').write_text('---\nname: safe\ntools:\n  - module: scripts.tools\n    names: [side_effect]\n---\nSafe')
    (directory/'scripts/tools.py').write_text('def side_effect() -> str:\n    """Never call during discovery."""\n    raise RuntimeError("executed")\n')
    skill = SkillLoader(str(tmp_path)).load_skill('safe')
    assert [tool.name for tool in skill.tools] == ['side_effect']


@pytest.mark.asyncio
async def test_invalid_arguments_reject_entire_batch_before_execution():
    invoked = []
    def record(text: str) -> str:
        """Record input."""
        invoked.append(text)
        return text
    calls = [call(), call(args={'wrong_field': 'invalid'}, id='t2')]
    agent = BoundedToolAgent(ScriptedModel([AIMessage(content='', tool_calls=calls)]),
                             [StructuredTool.from_function(record, name='echo')], '')
    with pytest.raises(ValueError):
        await agent.ainvoke({'messages': []}, request(budget()))
    assert invoked == []


@pytest.mark.asyncio
async def test_final_output_budget_is_enforced():
    agent = BoundedToolAgent(ScriptedModel([AIMessage(content='x' * 100)]), [], '')
    with pytest.raises(BudgetExceeded, match='output_characters'):
        await agent.ainvoke({'messages': []}, request(budget(output_limit=10)))


@pytest.mark.asyncio
async def test_run_report_is_saved_by_parent_graph_checkpoint(monkeypatch):
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    import src.agent.agents.operation as operation
    tool = StructuredTool.from_function(echo)
    model = ScriptedModel([AIMessage(content='', tool_calls=[call()]), AIMessage(content='done')])
    monkeypatch.setattr(operation, 'get_selected_tools', lambda names: [tool])
    monkeypatch.setattr(operation, '_get_operation_agent', lambda tools: BoundedToolAgent(model, tools, ''))
    class State(TypedDict, total=False):
        messages: list
        route_decision: dict
        harness_report: dict
        final_answer: str
        used_agent: str
    builder = StateGraph(State)
    builder.add_node('operation', operation.operation_agent_node)
    builder.add_edge(START, 'operation')
    builder.add_edge('operation', END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {'configurable': {'thread_id': 'harness-checkpoint-test'}}
    result = await graph.ainvoke({'messages': [HumanMessage(content='calculate')],
        'route_decision': {'handler': 'operation_agent', 'skills': ['calculator']}}, config)
    saved = await graph.aget_state(config)
    assert saved.values['harness_report'] == result['harness_report']
    assert saved.values['harness_report']['status'] == 'completed'
    assert saved.values['harness_report']['tool_calls'] == 1


def test_routing_fixture_acceptance_contract():
    import json
    from pathlib import Path
    from src.agent.routing.policy import rules_route
    path = Path(__file__).parent/'eval/fixtures/research_routing_v1.jsonl'
    for line in path.read_text().splitlines():
        case = json.loads(line)
        decision = rules_route(case['query'], case.get('previous'))
        assert decision.handler == case['expected'], case['id']
        assert set(case.get('required_skills', ())).issubset(decision.skills), case['id']
