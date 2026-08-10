"""主 Agent 图的架构契约测试。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.agents import planner as planner_module
from src.agent.graph import (
    create_multi_agent_graph,
    finalize_response_node,
    maybe_summarize_node,
    retrieve_mem0_memories_node,
    save_to_mem0_node,
)
from src.rag.evaluation.retrieval_grader import is_meaningful_retrieval_query
from src.rag.storage.version_manager import DocumentVersionManager
from src.api.services.session_service import _generate_title
from tests.eval.run_rag_eval import _average_precision, ndcg_at_k


@pytest.mark.asyncio
async def test_complex_knowledge_query_uses_rag_expansion_without_planner_llm(monkeypatch):
    """对比/列举类问题必须进入 RAG，不再进入不可用的 Send 分支。"""

    def _unexpected_llm():
        raise AssertionError("规则已识别的复杂知识查询不应调用 Planner LLM")

    monkeypatch.setattr(planner_module, "get_llm", _unexpected_llm)

    state = await planner_module.planner_node(
        {"messages": [HumanMessage(content="对比 RDMA 和 TCP 的差异")]}
    )

    assert state["is_complex"] is True
    assert state["needs_expansion"] is True
    assert state["plan_steps"] == []
    assert planner_module.route_from_planner(state) == "retrieval_agent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("你好", "general_agent"),
        ("现在几点", "operation_agent"),
        ("1+1", "operation_agent"),
        ("帮我计算 123 * 456 = ?", "operation_agent"),
        ("实验室组会多久一次", "retrieval_agent"),
        ("有哪些 RDMA 实验要求", "retrieval_agent"),
    ],
)
async def test_product_queries_have_one_deterministic_route(query, expected_route):
    state = await planner_module.planner_node(
        {"messages": [HumanMessage(content=query)]}
    )
    assert planner_module.route_from_planner(state) == expected_route


def test_version_sources_hide_server_paths_and_keep_documents_separate():
    docs = [
        Document(
            page_content="RDMA 规范",
            metadata={
                "source": "/srv/private/operations/RDMA实验规范.md",
                "version": "1.0",
            },
        ),
        Document(
            page_content="集群规范",
            metadata={
                "source": "/srv/private/operations/集群使用说明.md",
                "version": "1.0",
            },
        ),
    ]

    manager = DocumentVersionManager.__new__(DocumentVersionManager)
    rendered = manager.format_version_source(docs)

    assert "/srv/private" not in rendered
    assert "RDMA实验规范" in rendered
    assert "集群使用说明" in rendered
    assert rendered.count("1 个文档片段") == 2


def test_graph_contains_single_router_and_common_finalizer():
    graph = create_multi_agent_graph().compile().get_graph().draw_mermaid()

    assert "planner" in graph
    assert "retrieval_agent" in graph
    assert "finalize_response" in graph
    assert "execute_plan" not in graph
    assert "supervisor" not in graph
    assert "knowledge_step_node" not in graph
    assert "save_to_mem0" not in graph


@pytest.mark.asyncio
async def test_finalizer_adds_exactly_one_visible_ai_message():
    first = await finalize_response_node(
        {
            "messages": [HumanMessage(content="组会多久一次？")],
            "final_answer": "每周一次。",
        }
    )
    assert len(first["messages"]) == 1
    assert isinstance(first["messages"][0], AIMessage)
    assert first["messages"][0].content == "每周一次。"

    second = await finalize_response_node(
        {
            "messages": [
                HumanMessage(content="组会多久一次？"),
                AIMessage(content="每周一次。"),
            ],
            "final_answer": "每周一次。",
        }
    )
    assert "messages" not in second


@pytest.mark.asyncio
async def test_summary_threshold_ignores_internal_tool_messages(monkeypatch):
    settings = SimpleNamespace(summary_threshold=2, summary_keep_recent=1)
    monkeypatch.setattr("config.settings.get_settings", lambda: settings)

    state = {
        "messages": [
            HumanMessage(content="运行计算"),
            ToolMessage(content="内部工具输出一", tool_call_id="1"),
            ToolMessage(content="内部工具输出二", tool_call_id="2"),
            AIMessage(content="计算完成"),
        ],
        "summary": "",
    }

    assert await maybe_summarize_node(state) == {}


@pytest.mark.asyncio
async def test_mem0_uses_one_user_level_search(monkeypatch):
    manager = SimpleNamespace(
        search=AsyncMock(return_value=[]),
        format_memories_for_context=lambda memories, max_chars: "",
    )
    settings = SimpleNamespace(mem0_enabled=True, mem0_max_context_chars=500)
    monkeypatch.setattr("config.settings.get_settings", lambda: settings)
    monkeypatch.setattr("src.agent.memory.get_mem0_manager", lambda: manager)

    await retrieve_mem0_memories_node(
        {
            "messages": [HumanMessage(content="我之前关注哪个研究方向？")],
            "session_id": "session-1",
            "user_id": "alice",
        }
    )

    manager.search.assert_awaited_once_with(
        query="我之前关注哪个研究方向？",
        user_id="alice",
        session_id=None,
        limit=5,
    )


@pytest.mark.asyncio
async def test_mem0_writes_once_and_excludes_tool_messages(monkeypatch):
    manager = SimpleNamespace(add_conversation=AsyncMock(return_value={"success": True}))
    settings = SimpleNamespace(mem0_enabled=True)
    monkeypatch.setattr("config.settings.get_settings", lambda: settings)
    monkeypatch.setattr("src.agent.memory.get_mem0_manager", lambda: manager)

    await save_to_mem0_node(
        {
            "messages": [
                HumanMessage(content="记住我研究 RDMA"),
                ToolMessage(content="内部检索结果", tool_call_id="1"),
                AIMessage(content="好的，我记住了。"),
            ],
            "session_id": "session-1",
            "user_id": "alice",
        }
    )
    # 保存是后台任务；让事件循环执行已调度任务，但不让用户链路等待其耗时。
    await asyncio.sleep(0)

    manager.add_conversation.assert_awaited_once_with(
        messages=[
            {"role": "user", "content": "记住我研究 RDMA"},
            {"role": "assistant", "content": "好的，我记住了。"},
        ],
        user_id="alice",
        session_id="session-1",
    )


@pytest.mark.asyncio
async def test_slow_mem0_write_does_not_block_graph_completion(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_add(**_kwargs):
        started.set()
        await release.wait()

    manager = SimpleNamespace(add_conversation=AsyncMock(side_effect=_slow_add))
    settings = SimpleNamespace(mem0_enabled=True)
    monkeypatch.setattr("config.settings.get_settings", lambda: settings)
    monkeypatch.setattr("src.agent.memory.get_mem0_manager", lambda: manager)

    result = await asyncio.wait_for(
        save_to_mem0_node(
            {
                "messages": [
                    HumanMessage(content="记住这个偏好"),
                    AIMessage(content="已记录"),
                ],
                "session_id": "session-2",
                "user_id": "alice",
            }
        ),
        timeout=0.2,
    )

    assert result == {}
    await asyncio.wait_for(started.wait(), timeout=0.2)
    release.set()
    await asyncio.sleep(0)


def test_document_metrics_do_not_double_count_chunks_from_same_source():
    docs = [
        Document(page_content="片段一", metadata={"source": "实验规范.md"}),
        Document(page_content="片段二", metadata={"source": "实验规范.md"}),
        Document(page_content="其他", metadata={"source": "其他资料.md"}),
    ]

    assert ndcg_at_k(docs, {"实验规范"}, 3) == 1.0
    assert _average_precision(docs, {"实验规范"}) == 1.0


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("？!@#$%^&*(){}[]|", False),
        ("a" * 500, False),
        ("RDMA RDMA RDMA", True),
        ("\u200b\u200c组会多久召开一次？", True),
    ],
)
def test_retrieval_query_validity_guard(query, expected):
    assert is_meaningful_retrieval_query(query) is expected


def test_session_title_is_deterministic_and_bounded():
    assert _generate_title("  实验室组会多久召开一次？  ") == "实验室组会多久召开一次？"
    assert _generate_title("这是一个超过二十个字符并且不需要调用模型生成的会话标题") == "这是一个超过二十个字符并且不需要调用模型..."
