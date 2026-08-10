from src.rag.retrieval.reranker import QwenReranker, RerankerManager


def test_qwen_provider_uses_existing_qwen_reranker():
    manager = RerankerManager(
        reranker_model="gte-rerank-v2",
        provider="qwen",
        top_n=3,
        score_threshold=0.1,
    )

    assert isinstance(manager.reranker, QwenReranker)
    assert manager.reranker.model == "gte-rerank-v2"
    assert manager.reranker.top_n == 3
