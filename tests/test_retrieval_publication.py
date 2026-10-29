"""Publication filters must hold for both sides of hybrid search, before top-k."""
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from src.rag.retrieval.hybrid_retriever import HybridRetrieverManager


def corpus():
    return [
        Document(page_content='latency', metadata={
            'source': name, 'project_name': project, **({'ingestion_state': state} if state else {}),
        }) for name, state, project in [
            ('staged', 'staging', 'A'), ('retired', 'retired', 'A'),
            ('unknown', 'unexpected', 'A'), ('other-project', 'active', 'B'),
            ('published', 'active', 'A'), ('legacy', None, 'A'),
        ]
    ]


@pytest.mark.parametrize('scored', [False, True])
def test_bm25_excludes_unpublished_chunks_before_topk(scored):
    manager = HybridRetrieverManager(enable_vector=False, enable_bm25=True)
    manager._documents = corpus()
    manager._bm25_index = Mock()
    manager._bm25_index.get_scores.return_value = [6., 5., 4., 3., 2., 1.]
    manager._bm25_backend = 'rank_bm25'
    search = manager.search_with_scores if scored else manager.search
    result = search('latency', k=2, base_filter={'project_name': 'A'})
    docs = [row[0] for row in result] if scored else result
    assert {doc.metadata['source'] for doc in docs} == {'published', 'legacy'}


def test_vector_candidates_rechecked_if_backend_ignores_filter():
    manager = HybridRetrieverManager(enable_vector=True, enable_bm25=False)
    manager._vectorstore = Mock()
    manager._vectorstore.similarity_search_with_score.return_value = [(doc, .1) for doc in corpus()]
    result = manager.search_with_scores('latency', k=2, base_filter={'project_name': 'A'})
    assert {doc.metadata['source'] for doc, _, _ in result} == {'published', 'legacy'}


def test_caller_cannot_override_publication_gate():
    manager = HybridRetrieverManager(enable_vector=False, enable_bm25=True)
    manager._documents = corpus()
    manager._bm25_index = Mock()
    manager._bm25_index.get_scores.return_value = [6., 5., 4., 3., 2., 1.]
    manager._bm25_backend = 'rank_bm25'
    assert manager.search('latency', base_filter={'ingestion_state': 'staging'}) == []
