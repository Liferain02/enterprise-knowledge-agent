from types import SimpleNamespace

import src.models.embeddings as embeddings_module


def test_embedding_provider_does_not_follow_chat_provider(monkeypatch):
    settings = SimpleNamespace(
        llm_provider="openai",
        embedding_provider="qwen",
        embedding_model="text-embedding-v2",
        dashscope_api_key="qwen-key",
        openai_api_key="deepseek-key",
        openai_base_url="https://api.deepseek.com",
    )
    monkeypatch.setattr(embeddings_module, "get_settings", lambda: settings)
    embeddings_module.reset_embeddings()

    instance = embeddings_module.get_embeddings()

    assert isinstance(instance, embeddings_module.DashScopeEmbeddings)
    assert instance.model == "text-embedding-v2"
    assert instance.api_key == "qwen-key"
    embeddings_module.reset_embeddings()
