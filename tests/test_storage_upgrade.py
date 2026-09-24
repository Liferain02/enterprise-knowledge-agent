"""Storage contracts: ACL filters, idempotence, cache isolation and fail-open."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.documents import Document


def test_qdrant_roundtrip_filter_pagination_and_delete():
    from qdrant_client import QdrantClient
    from src.rag.storage.qdrant_store import QdrantStoreManager, to_filter
    embedding = Mock()
    embedding.embed_documents.side_effect = lambda texts: [[1.0, float(i % 2)] for i, _ in enumerate(texts)]
    embedding.embed_query.return_value = [1.0, 0.0]
    manager = QdrantStoreManager("test", client=QdrantClient(":memory:"), embeddings=embedding)
    docs = [Document(page_content=f"text{i}", metadata={"source": f"s{i}", "visibility": "public" if i % 2 == 0 else "restricted"}) for i in range(6)]
    manager.add_documents(docs, ids=[str(i) for i in range(6)])
    assert manager.count() == 6
    manager.add_documents(docs, ids=[str(i) for i in range(6)])
    assert manager.count() == 6
    found = manager.similarity_search("q", 6, {"$and": [{"visibility": {"$in": ["public"]}}, {"source": {"$ne": "s0"}}]})
    assert {d.metadata["source"] for d in found} == {"s2", "s4"}
    first, second = manager.list_documents(3), manager.list_documents(3, 3)
    assert set(first["ids"]).isdisjoint(second["ids"])
    assert manager.get_document_ids_by_source("s2") == ["2"]
    old_revision = manager.revision()
    manager.delete_documents_by_source("s2")
    assert manager.count() == 5 and manager.revision() != old_revision
    with pytest.raises(ValueError):
        to_filter({"visibility": {"$unknown": "public"}})


def test_embedding_batch_order_deduplication_and_cache(monkeypatch):
    import src.models.embeddings as module
    from src.rag.cache import cache
    import dashscope
    values = {}
    monkeypatch.setattr(cache, "get", values.get)
    monkeypatch.setattr(cache, "set", lambda k, v, ttl: values.__setitem__(k, v))
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(
        embedding_cache_enabled=True, embedding_batch_size=2, embedding_cache_ttl=60))
    call = Mock(side_effect=lambda **kw: SimpleNamespace(status_code=200, output={
        "embeddings": [{"text_index": i, "embedding": [float(ord(text[0])), 1.0]}
                       for i, text in reversed(list(enumerate(kw["input"])))]}))
    monkeypatch.setattr(dashscope.TextEmbedding, "call", call)
    embedding = module.DashScopeEmbeddings("model-a", "key")
    assert embedding.embed_documents(["a", "b", "a", "c"]) == [[97., 1.], [98., 1.], [97., 1.], [99., 1.]]
    assert call.call_count == 2
    embedding.embed_documents(["a", "b"])
    assert call.call_count == 2
    embedding.embed_query("a")  # Query/document cache domains are distinct.
    assert call.call_count == 3
    module.DashScopeEmbeddings("model-b", "key").embed_query("a")
    assert call.call_count == 4


def test_redis_outage_opens_circuit(monkeypatch):
    import src.rag.cache as module
    cache = module.RedisCache()
    broken = Mock()
    broken.get.side_effect = ConnectionError("offline")
    cache._client = broken
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(rag_cache_enabled=True, redis_host="localhost"))
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert broken.get.call_count == 1
    assert cache.errors == 1


def test_rerank_cache_uses_current_metadata(monkeypatch):
    import dashscope
    from src.rag.cache import cache
    from src.rag.retrieval.reranker import QwenReranker
    values = {}
    monkeypatch.setattr(cache, "get", values.get)
    monkeypatch.setattr(cache, "set", lambda k, v, ttl: values.__setitem__(k, v))
    call = Mock(return_value=SimpleNamespace(status_code=200, output={"results": [{"index": 0, "relevance_score": .9}]}))
    monkeypatch.setattr(dashscope.TextReRank, "call", call)
    ranker = QwenReranker(api_key="test")
    ranker.rerank("q", [Document(page_content="same", metadata={"source": "old", "visibility": "public"})])
    current = Document(page_content="same", metadata={"source": "new", "visibility": "restricted"})
    assert ranker.rerank("q", [current])[0][0] is current
    assert call.call_count == 1


def test_incremental_ingestion_keeps_old_chunks_on_failure():
    from scripts.sync_knowledge_corpus import sync
    manager = Mock()
    manager.get_document_ids_by_source.return_value = ["old"]
    manager.add_documents.side_effect = RuntimeError("backend offline")
    with pytest.raises(RuntimeError):
        sync([(Path("source.md"), [Document(page_content="new")], ["new-id"])], manager)
    manager.delete_documents_by_ids.assert_not_called()


def test_sql_binding_preserves_literals():
    from src.storage.relational import bind_sql, Row
    assert bind_sql("SELECT '?' WHERE title LIKE ? AND note='50%'") == "SELECT '?' WHERE title LIKE %s AND note='50%%'"
    row = Row(["id", "content"], [1, "你好"])
    assert row[0] == row["id"] == 1
    assert dict(row) == {"id": 1, "content": "你好"}


@pytest.fixture
def mysql_backend(monkeypatch):
    if os.getenv("RUN_MYSQL_TESTS") != "1":
        pytest.skip("Requires local Docker MySQL and eka_verify schemas")
    from scripts.start_infrastructure import infrastructure_env
    from config.settings import get_settings
    from src.storage.relational import SCHEMA, MySQLConnection
    settings = get_settings()
    for key, value in {"database_provider": "mysql", "mysql_database_prefix": "eka_verify",
                       "mysql_password": infrastructure_env()["MYSQL_PASSWORD"]}.items():
        monkeypatch.setattr(settings, key, value)
    for ns, tables in SCHEMA.items():
        with MySQLConnection(ns) as conn:
            for table in tables.values():
                conn.raw.cursor().execute(table["create_sql"])
    return settings


def test_mysql_sessions_unicode_duplicate_and_cascade(mysql_backend):
    from src.api.repositories.dao.session_dao import SessionDAO, MessageDAO
    sessions, messages = SessionDAO(), MessageDAO()
    sid = "test-" + uuid4().hex
    try:
        sessions.create_session(sid)
        sessions.create_session(sid)
        text = "中文🚀 O'Reilly 100% ?"
        messages.save(sid, "user", text)
        assert messages.get_by_session(sid)[0]["content"] == text
        sessions.rename_session_id(sid, sid + "-new")
        assert messages.get_by_session(sid + "-new")[0]["content"] == text
        assert messages.get_by_session(sid) == []
    finally:
        sessions.delete(sid + "-new")
        sessions.delete(sid)


def test_mysql_queue_two_consumers_claim_once(mysql_backend):
    from concurrent.futures import ThreadPoolExecutor
    from src.rag.ingestion.job_queue import IngestionJobQueue
    from src.storage.relational import MySQLConnection
    queues = [IngestionJobQueue(), IngestionJobQueue()]
    job_id = queues[0].enqueue("test.md", "test", {})
    try:
        with ThreadPoolExecutor(2) as pool:
            result = list(pool.map(lambda queue: queue.dequeue(), queues))
        assert sum(item is not None and item.job_id == job_id for item in result) == 1
        queues[0].complete(job_id, {"chunks": 2})
        assert queues[0].get_job(job_id).result == {"chunks": 2}
    finally:
        with MySQLConnection("ingestion") as conn:
            conn.execute("DELETE FROM ingestion_jobs WHERE job_id=?", (job_id,))


def test_mysql_catalog_nested_audit_and_rollback(mysql_backend):
    from src.api import database
    from src.storage.relational import MySQLConnection
    database.init_database()
    doc_id = "test-" + uuid4().hex
    try:
        database.create_document_meta(doc_id, "测试", "test", "test")
        assert database.get_document(doc_id)["title"] == "测试"
        with pytest.raises(RuntimeError):
            with database._db_cursor() as cur:
                cur.execute("UPDATE documents SET title=? WHERE id=?", ("rollback", doc_id))
                raise RuntimeError("rollback")
        assert database.get_document(doc_id)["title"] == "测试"
    finally:
        with MySQLConnection("catalog") as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            conn.execute("DELETE FROM audit_logs WHERE resource_id=?", (doc_id,))


def test_mysql_research_project_acl_and_foreign_keys(mysql_backend):
    from src.api.services.research_service import ResearchService
    from src.storage.relational import MySQLConnection
    service = ResearchService()
    user = {"username": "test-pi", "role": "pi"}
    project = service.create_project({"title": "测试-" + uuid4().hex, "visibility": "restricted"}, user)
    try:
        assert service.get_project(project["id"], user)["title"] == project["title"]
        with pytest.raises(PermissionError):
            service.get_project(project["id"], {"username": "outsider", "role": "student"})
        service.create_experiment(project["id"], {"title": "实验", "hypothesis": "中文长文" * 100}, user)
    finally:
        with MySQLConnection("research") as conn:
            conn.execute("DELETE FROM research_projects WHERE id=?", (project["id"],))
            assert conn.execute("SELECT COUNT(*) FROM research_experiments WHERE project_id=?", (project["id"],)).fetchone()[0] == 0
