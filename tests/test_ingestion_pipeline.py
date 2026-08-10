"""异步入库队列和文件提交测试。"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.api.services.knowledge_service import KnowledgeService
from src.rag.ingestion.job_queue import IngestionJobQueue, JobStatus
from src.rag.ingestion.worker import IngestionWorker


def test_job_queue_records_result_and_finds_hash(tmp_path):
    queue = IngestionJobQueue(str(tmp_path / "queue.db"))
    job_id = queue.enqueue(
        file_path=str(tmp_path / "paper.md"),
        category="paper_note",
        metadata={"doc_type": "paper_note"},
        file_hash="abc123",
        original_filename="paper.md",
    )

    job = queue.dequeue()
    assert job is not None
    assert job.job_id == job_id
    assert job.status == JobStatus.PENDING

    queue.complete(job_id, {"stored_chunks": 3, "elapsed_seconds": 0.2})
    completed = queue.find_by_file_hash("abc123")

    assert completed is not None
    assert completed.status == JobStatus.COMPLETED
    assert completed.result["stored_chunks"] == 3
    assert queue.get_stats()["completed"] == 1


def test_enqueue_document_deduplicates_same_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    service = KnowledgeService()
    service._job_queue = IngestionJobQueue(str(tmp_path / "queue.db"))

    first = service.enqueue_document_from_file(
        b"# RDMA notes",
        "RDMA notes.md",
        category="paper_note",
        uploaded_by="tester",
    )
    second = service.enqueue_document_from_file(
        b"# RDMA notes",
        "copy.md",
        category="paper_note",
        uploaded_by="tester",
    )

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["job"]["job_id"] == first["job"]["job_id"]
    assert Path("data/uploads").exists()


def test_upload_metadata_tags_are_chroma_compatible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    service = KnowledgeService()
    service._job_queue = IngestionJobQueue(str(tmp_path / "queue.db"))

    result = service.enqueue_document_from_file(
        b"# NUMA notes",
        "numa.md",
        metadata={"tags": ["NUMA", "RDMA"]},
    )

    job = service.job_queue.get_job(result["job"]["job_id"])
    assert job.metadata["tags"] == "NUMA,RDMA"


def test_worker_replaces_same_source_chunks():
    worker = IngestionWorker(queue=MagicMock())
    manager = MagicMock()
    manager.get_document_ids_by_source.return_value = ["old-chunk"]
    manager.add_documents.return_value = ["chunk-1", "chunk-2"]
    docs = [
        MagicMock(page_content="chunk one", metadata={}),
        MagicMock(page_content="chunk two", metadata={}),
    ]

    import src.rag.storage.vectorstore as vectorstore_module

    original = vectorstore_module.get_vectorstore_manager
    vectorstore_module.get_vectorstore_manager = MagicMock(return_value=manager)
    try:
        stored = worker._embed_and_store(
            docs,
            {"source": "notes.md", "file_hash": "abc123"},
        )
    finally:
        vectorstore_module.get_vectorstore_manager = original

    assert stored == 2
    manager.get_document_ids_by_source.assert_called_once_with("notes.md")
    manager.delete_documents_by_ids.assert_called_once_with(["old-chunk"])
    _, kwargs = manager.add_documents.call_args
    assert len(kwargs["ids"]) == 2
    assert docs[0].metadata["chunk_index"] == 0


def test_worker_converts_structured_metadata_to_scalars():
    worker = IngestionWorker(queue=MagicMock())
    manager = MagicMock()
    manager.get_document_ids_by_source.return_value = []
    manager.add_documents.return_value = ["chunk-1"]
    docs = [MagicMock(page_content="chunk one", metadata={})]

    import src.rag.storage.vectorstore as vectorstore_module

    original = vectorstore_module.get_vectorstore_manager
    vectorstore_module.get_vectorstore_manager = MagicMock(return_value=manager)
    try:
        worker._embed_and_store(docs, {"source": "notes.md", "tags": ["RDMA", "NUMA"]})
    finally:
        vectorstore_module.get_vectorstore_manager = original

    assert docs[0].metadata["tags"] == "RDMA,NUMA"


def test_worker_preserves_old_chunks_when_new_write_fails():
    worker = IngestionWorker(queue=MagicMock())
    manager = MagicMock()
    manager.get_document_ids_by_source.return_value = ["old-chunk"]
    manager.add_documents.side_effect = RuntimeError("embedding failed")
    docs = [MagicMock(page_content="chunk one", metadata={})]

    import src.rag.storage.vectorstore as vectorstore_module

    original = vectorstore_module.get_vectorstore_manager
    vectorstore_module.get_vectorstore_manager = MagicMock(return_value=manager)
    try:
        with pytest.raises(RuntimeError, match="embedding failed"):
            worker._embed_and_store(docs, {"source": "notes.md"})
    finally:
        vectorstore_module.get_vectorstore_manager = original

    manager.delete_documents_by_ids.assert_not_called()


def test_worker_completes_job_with_result(tmp_path):
    queue = IngestionJobQueue(str(tmp_path / "queue.db"))
    job_id = queue.enqueue(
        file_path=str(tmp_path / "notes.md"),
        category="paper_note",
        metadata={"source": "notes.md", "file_hash": "hash"},
    )
    job = queue.dequeue()
    worker = IngestionWorker(queue=queue)
    worker._load_and_chunk = MagicMock(return_value=[MagicMock()])
    worker._embed_and_store = MagicMock(return_value=2)

    worker._process_job(job, max_retries=0)

    completed = queue.get_job(job_id)
    assert completed.status == JobStatus.COMPLETED
    assert completed.result["stored_chunks"] == 2


def test_queue_recovers_stale_running_job(tmp_path):
    queue = IngestionJobQueue(str(tmp_path / "queue.db"))
    job_id = queue.enqueue(
        file_path=str(tmp_path / "notes.md"),
        category="paper_note",
        metadata={},
    )
    queue.dequeue()

    assert queue.recover_stale_jobs(stale_after_seconds=0) == 1
    assert queue.get_job(job_id).status == JobStatus.RETRYING
