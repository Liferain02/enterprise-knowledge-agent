"""聊天流只向用户发送最终回答，不泄露图内模型中间输出。"""

import importlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.rag.retrieval.acl_filter import UserContext


chat_module = importlib.import_module("src.api.services.chat_service")


class _FakeGraph:
    def __init__(self, events, state):
        self.events = events
        self.state = state

    async def astream_events(self, *_args, **_kwargs):
        for event in self.events:
            yield event

    async def aget_state(self, _config):
        return SimpleNamespace(values=self.state)


def _model_event(content: str, *, visible: bool = False):
    return {
        "event": "on_chat_model_stream",
        "tags": ["user_visible_answer"] if visible else [],
        "metadata": {"langgraph_node": "generation_agent" if visible else "planner"},
        "data": {"chunk": SimpleNamespace(content=content)},
    }


def _decode_sse(chunks):
    return [json.loads(chunk.removeprefix("data: ").strip()) for chunk in chunks]


def test_source_cards_are_reauthorized_for_current_user():
    user = UserContext(
        user_id="u1",
        username="普通员工",
        role="employee",
        department="dev",
        department_name="研发部",
        department_path="/研发部",
    )
    allowed = Document(
        page_content="公开实验结论",
        metadata={"title": "公开资料", "source": "公开.md", "confidentiality": "internal"},
    )
    denied = Document(
        page_content="高管薪酬",
        metadata={"title": "受限资料", "source": "薪酬.md", "confidentiality": "confidential"},
    )

    cards = chat_module.ChatService()._build_source_cards(
        [denied, allowed], user_context=user
    )

    assert [card["title"] for card in cards] == ["公开资料"]


@pytest.mark.asyncio
async def test_stream_excludes_internal_model_tokens(monkeypatch):
    graph = _FakeGraph(
        [
            _model_event('{"is_complex": true}'),
            _model_event("可信回答", visible=True),
            _model_event("内部记忆摘要"),
        ],
        {
            "final_answer": "可信回答",
            "retrieved_docs": [
                Document(
                    page_content="实验结果显示吞吐量提升。",
                    metadata={"title": "RDMA 实验记录", "source": "rdma.md"},
                )
            ],
            "used_agent": "knowledge_agent",
            "version_source": "",
        },
    )

    async def _get_graph():
        return graph

    saved = []
    monkeypatch.setattr(chat_module, "get_agent_graph_async", _get_graph)
    monkeypatch.setattr(chat_module.session_service, "ensure_session_exists", lambda *_: None)
    monkeypatch.setattr(
        chat_module.session_service,
        "get_session",
        lambda *_: {"message_count": 1},
    )
    monkeypatch.setattr(
        chat_module.session_service,
        "save_message",
        lambda *args, **kwargs: saved.append((args, kwargs)),
    )
    monkeypatch.setattr(
        chat_module.ChatService,
        "_schedule_memory_save",
        lambda *_args, **_kwargs: None,
    )

    chunks = [
        chunk
        async for chunk in chat_module.ChatService().achat_stream(
            "总结最近的实验结论", "session-1", username="alice"
        )
    ]
    events = _decode_sse(chunks)
    streamed_text = "".join(
        event["data"] for event in events if event["type"] == "llm_token"
    )

    assert streamed_text == "可信回答"
    assert all("is_complex" not in chunk and "内部记忆摘要" not in chunk for chunk in chunks)
    sources_event = next(event for event in events if event["type"] == "sources")
    assert sources_event["data"][0]["title"] == "RDMA 实验记录"
    assert saved[-1][0][3] == "可信回答"


@pytest.mark.asyncio
async def test_stream_falls_back_to_final_graph_answer(monkeypatch):
    graph = _FakeGraph(
        [_model_event("内部工具选择")],
        {
            "final_answer": "工具执行完成后的回答",
            "retrieved_docs": [],
            "used_agent": "operation_agent",
            "version_source": "",
        },
    )

    async def _get_graph():
        return graph

    monkeypatch.setattr(chat_module, "get_agent_graph_async", _get_graph)
    monkeypatch.setattr(chat_module.session_service, "ensure_session_exists", lambda *_: None)
    monkeypatch.setattr(
        chat_module.session_service,
        "get_session",
        lambda *_: {"message_count": 1},
    )
    monkeypatch.setattr(chat_module.session_service, "save_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_module.ChatService,
        "_schedule_memory_save",
        lambda *_args, **_kwargs: None,
    )

    chunks = [
        chunk
        async for chunk in chat_module.ChatService().achat_stream(
            "现在几点", "session-2", username="alice"
        )
    ]
    events = _decode_sse(chunks)
    streamed_text = "".join(
        event["data"] for event in events if event["type"] == "llm_token"
    )

    assert streamed_text == "工具执行完成后的回答"
    assert "内部工具选择" not in "".join(chunks)
