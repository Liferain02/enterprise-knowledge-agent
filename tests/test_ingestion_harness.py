"""Real loader + shared splitter + in-memory Qdrant + transactional metadata."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from src.rag.ingestion.job_queue import IngestionJobQueue, JobStatus
from src.rag.ingestion.worker import IngestionWorker
from src.rag.storage.qdrant_store import QdrantStoreManager
from src.rag.storage.version_manager import DocumentVersion, DocumentVersionManager, VersionDB


def version(identifier='old', number='1.0'):
    return DocumentVersion(identifier, 'paper', number, '2026-09-01', None, 'active', None,
                           'manual', '', 'tester', 1.0)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    import src.rag.storage.vectorstore as vectors
    import src.rag.storage.version_manager as versions
    embedding = Mock()
    embedding.embed_documents.side_effect = lambda texts: [[1., 0.] for text in texts]
    embedding.embed_query.return_value = [1., 0.]
    manager = QdrantStoreManager('ingestion-contract', client=QdrantClient(':memory:'), embeddings=embedding)
    vm = DocumentVersionManager(VersionDB(str(tmp_path/'versions.db')))
    vm.db.publish_version(version())
    manager.add_documents([Document(page_content='old content', metadata={'source': 'paper.txt'})], ids=['old-chunk'])
    monkeypatch.setattr(vectors, 'get_vectorstore_manager', lambda: manager)
    monkeypatch.setattr(versions, 'get_version_manager', lambda: vm)
    path = tmp_path/'paper.txt'
    path.write_text('科研实验' * 700, encoding='utf-8')
    queue = IngestionJobQueue(str(tmp_path/'queue.db'))
    queue.enqueue(str(path), 'paper_note', {'source': 'paper.txt', 'doc_id': 'paper', 'version': '2.0',
        'file_hash': 'new-content', 'visibility': 'restricted'})
    yield IngestionWorker(queue), queue.dequeue(), manager, vm, embedding
    manager.client.close()


def test_upload_chunks_verifies_publishes_then_removes_old_and_retry_is_idempotent(pipeline):
    worker, job, manager, vm, embedding = pipeline
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.COMPLETED
    rows = manager.list_documents(20)
    assert len(rows['ids']) >= 3 and 'old-chunk' not in rows['ids']
    docs = manager.similarity_search('科研', 20, {'visibility': 'restricted'})
    assert len(docs) == len(rows['ids'])
    assert all(len(doc.page_content) <= 1200 for doc in docs)
    assert {doc.metadata['source'] for doc in docs} == {'paper.txt'}
    assert all(doc.metadata['ingestion_signature'] == 'split1200-overlap150-v1' for doc in docs)
    current = vm.db.get_current_version('paper')
    assert current.version == '2.0' and current.status == 'active'
    old = next(row for row in vm.db.get_versions('paper') if row.id == 'old')
    assert old.status == 'superseded' and old.superseded_by == current.id
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.COMPLETED
    assert set(manager.list_documents(20)['ids']) == set(rows['ids'])
    assert len(vm.db.get_versions('paper')) == 2


def test_embedding_failure_keeps_active_metadata_and_old_vectors(pipeline):
    worker, job, manager, vm, embedding = pipeline
    embedding.embed_documents.side_effect = RuntimeError('offline embedding')
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.FAILED
    assert vm.db.get_current_version('paper').id == 'old'
    assert manager.get_document_ids_by_source('paper.txt') == ['old-chunk']


def test_incomplete_vector_ack_does_not_publish_or_delete(pipeline, monkeypatch):
    worker, job, manager, vm, _ = pipeline
    original = manager.add_documents
    def incomplete(docs, ids):
        return original(docs[:1], ids[:1])
    monkeypatch.setattr(manager, 'add_documents', incomplete)
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.FAILED
    assert vm.db.get_current_version('paper').id == 'old'
    assert 'old-chunk' in manager.get_document_ids_by_source('paper.txt')


def test_missing_persisted_vectors_does_not_publish(pipeline, monkeypatch):
    worker, job, manager, vm, _ = pipeline
    monkeypatch.setattr(manager, 'add_documents', lambda docs, ids: ids)
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.FAILED
    assert vm.db.get_current_version('paper').id == 'old'
    assert manager.get_document_ids_by_source('paper.txt') == ['old-chunk']


def test_cleanup_failure_can_retry_without_duplicate_version(pipeline, monkeypatch):
    worker, job, manager, vm, _ = pipeline
    original = manager.delete_documents_by_ids
    monkeypatch.setattr(manager, 'delete_documents_by_ids', Mock(side_effect=RuntimeError('delete unavailable')))
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.FAILED
    assert vm.db.get_current_version('paper').version == '2.0'
    assert 'old-chunk' in manager.get_document_ids_by_source('paper.txt')
    monkeypatch.setattr(manager, 'delete_documents_by_ids', original)
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.COMPLETED
    assert len(vm.db.get_versions('paper')) == 2
    assert 'old-chunk' not in manager.get_document_ids_by_source('paper.txt')


def test_empty_upload_is_failed_not_zero_chunk_success(pipeline):
    from pathlib import Path
    worker, job, manager, vm, _ = pipeline
    Path(job.file_path).write_text('  \n')
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.FAILED
    assert vm.db.get_current_version('paper').id == 'old'


def test_sqlite_publish_rolls_back_insert_when_superseding_fails(tmp_path):
    import sqlite3
    db = VersionDB(str(tmp_path/'versions.db'))
    db.publish_version(version())
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON document_versions BEGIN SELECT RAISE(ABORT, 'injected'); END")
    with pytest.raises(sqlite3.IntegrityError, match='injected'):
        db.publish_version(version('new', '2.0'))
    assert [row.id for row in db.get_versions('paper')] == ['old']
    assert db.get_current_version('paper').status == 'active'


def test_superseded_retry_cannot_reactivate_old_version(tmp_path):
    db = VersionDB(str(tmp_path/'versions.db'))
    db.publish_version(version())
    db.publish_version(version('new', '2.0'))
    with pytest.raises(ValueError):
        db.publish_version(version())
    assert db.get_current_version('paper').id == 'new'


def test_cleanup_failure_retires_old_generation_before_delete(pipeline, monkeypatch):
    worker, job, manager, vm, _ = pipeline
    monkeypatch.setattr(manager, 'delete_documents_by_ids', Mock(side_effect=RuntimeError('delete unavailable')))
    worker._process_job(job, max_retries=0)
    assert worker.queue.get_job(job.job_id).status == JobStatus.FAILED
    visible = manager.similarity_search('科研', 20, {'$or': [{'ingestion_state': 'active'}, {'ingestion_state': {'$exists': False}}]})
    assert all(doc.metadata.get('ingestion_version_id') == vm.db.get_current_version('paper').id for doc in visible)
    assert manager.get_document_ids_by_source('paper.txt') != ['old-chunk']
