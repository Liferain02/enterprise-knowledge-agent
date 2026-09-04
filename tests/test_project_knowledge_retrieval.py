"""P3 项目知识检索适配与自我强化隔离测试。"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.rag.retrieval.project_knowledge import (
    merge_project_knowledge,
    retrieve_project_knowledge,
)


def _record(status="active", origin="project_knowledge"):
    return {
        "id": "record-1",
        "project_id": "project-1",
        "statement": "RDMA 实验吞吐量为 91 Gbps。",
        "status": status,
        "version": 1,
        "research_run_id": "run-1",
        "claim_id": "C1",
        "source_ids": ["raw-1"],
        "knowledge_origin": origin,
        "sources": [{
            "source_id": "raw-1",
            "title": "原始实验记录",
            "excerpt": "RDMA 实验吞吐量为 91 Gbps。",
            "locator": "page=2",
        }],
    }


def test_active_project_knowledge_keeps_full_provenance(monkeypatch):
    service = MagicMock()
    service.list_knowledge_records.return_value = [_record()]
    monkeypatch.setattr(
        "src.api.services.research_service.research_service", service,
    )
    results = retrieve_project_knowledge(
        "RDMA 吞吐量", "project-1", {"username": "alice", "role": "student"},
    )

    assert len(results) == 1
    document, score = results[0]
    assert score > 0
    assert document.metadata["knowledge_record_id"] == "record-1"
    assert document.metadata["research_run_id"] == "run-1"
    assert document.metadata["claim_id"] == "C1"
    assert document.metadata["root_source_ids"] == ["raw-1"]
    assert document.metadata["knowledge_origin"] == "project_knowledge"
    assert "原始实验记录" in document.page_content
    service.list_knowledge_records.assert_called_once_with(
        "project-1", {"username": "alice", "role": "student"}, status="active",
    )


@pytest.mark.parametrize("status", ["revoked", "superseded"])
def test_inactive_project_knowledge_is_never_retrieved(monkeypatch, status):
    service = MagicMock()
    service.list_knowledge_records.return_value = [_record(status=status)]
    monkeypatch.setattr(
        "src.api.services.research_service.research_service", service,
    )
    assert retrieve_project_knowledge(
        "RDMA 吞吐量", "project-1", {"username": "alice", "role": "student"},
    ) == []


def test_derived_only_project_knowledge_is_never_retrieved(monkeypatch):
    service = MagicMock()
    service.list_knowledge_records.return_value = [_record(origin="derived_only")]
    monkeypatch.setattr(
        "src.api.services.research_service.research_service", service,
    )
    assert retrieve_project_knowledge(
        "RDMA 吞吐量", "project-1", {"username": "alice", "role": "student"},
    ) == []


def test_missing_project_or_acl_user_fails_closed(monkeypatch):
    service = MagicMock()
    monkeypatch.setattr(
        "src.api.services.research_service.research_service", service,
    )
    assert retrieve_project_knowledge("RDMA", "", {"username": "alice"}) == []
    assert retrieve_project_knowledge("RDMA", "project-1", None) == []
    service.list_knowledge_records.assert_not_called()


def test_merge_preserves_raw_documents_and_adds_bounded_knowledge(monkeypatch):
    service = MagicMock()
    service.list_knowledge_records.return_value = [_record()]
    monkeypatch.setattr(
        "src.api.services.research_service.research_service", service,
    )
    raw = Document(page_content="原始 RDMA 资料", metadata={"source": "raw.md"})
    merged = merge_project_knowledge(
        [(raw, 0.88)], "RDMA 吞吐量", "project-1",
        {"username": "alice", "role": "student"}, top_k=1, limit=1,
    )
    assert merged[0][0] is raw
    assert merged[1][0].metadata["knowledge_record_id"] == "record-1"


@pytest.mark.asyncio
async def test_default_retrieval_path_does_not_read_project_knowledge(monkeypatch):
    import config.settings as settings_module
    import src.rag.retrieval.retriever as retriever_module
    from src.agent.agents import knowledge as knowledge_module

    raw = Document(page_content="原始资料", metadata={"source": "raw.md"})
    manager = MagicMock()
    manager.search_with_rerank.return_value = [(raw, 0.8)]
    monkeypatch.setattr(retriever_module, "get_retriever_manager", lambda: manager)
    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: SimpleNamespace(
            crag_enabled=False,
            query_expand_enabled=False,
            project_knowledge_retrieval_enabled=False,
            project_knowledge_retrieval_top_k=2,
        ),
    )
    adapter = MagicMock(side_effect=AssertionError("默认路径不应读取项目知识"))
    monkeypatch.setattr(
        "src.rag.retrieval.project_knowledge.merge_project_knowledge", adapter,
    )
    results, _grade, _history = await knowledge_module._retrieve_documents(
        "实验室制度", 5, False, None, project_id="project-1",
    )
    assert results == [(raw, 0.8)]
    adapter.assert_not_called()
