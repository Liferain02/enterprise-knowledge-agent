"""
向量化嵌入模块
支持 OpenAI Embeddings 和 阿里千问 Embeddings
"""
import os
import math
import time
from typing import Optional, List, Union
from langchain_core.embeddings import Embeddings
from config.settings import get_settings


class DashScopeEmbeddings(Embeddings):
    """阿里千问 Embeddings 实现"""

    def __init__(
        self,
        model: str = "text-embedding-v2",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch cache misses, deduplicate inputs, preserve exact input order."""
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "query")[0]

    @staticmethod
    def _valid_vector(value):
        return (isinstance(value, list) and bool(value)
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in value))

    def _embed(self, texts, purpose):
        from dashscope import TextEmbedding
        from src.rag.cache import cache, cache_key

        settings = get_settings()
        use_cache = getattr(settings, "embedding_cache_enabled", False)
        batch_size = getattr(settings, "embedding_batch_size", 10)
        keys = {text: cache_key("embedding", ["dashscope-native-v1", self.model, purpose, text])
                for text in texts}
        values, missing = {}, []
        for text, key in keys.items():
            hit = cache.get(key) if use_cache else None
            if self._valid_vector(hit):
                values[text] = hit
            else:
                missing.append(text)
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            for attempt in range(4):
                response = TextEmbedding.call(model=self.model, input=batch, api_key=self.api_key)
                if response.status_code == 200:
                    break
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f"Embedding API error: status={response.status_code}, code={response.code}")
                time.sleep(2 ** attempt)
            rows = response.output["embeddings"]
            ordered = {row["text_index"]: row["embedding"] for row in rows}
            if (set(ordered) != set(range(len(batch))) or len(rows) != len(batch)
                    or not all(self._valid_vector(v) for v in ordered.values())):
                raise ValueError("Invalid/incomplete embedding response")
            for index, text in enumerate(batch):
                values[text] = ordered[index]
                if use_cache:
                    cache.set(keys[text], ordered[index], settings.embedding_cache_ttl)
        return [values[text] for text in texts]


class LocalEmbeddings(Embeddings):
    """Pinned BGE-M3 dense embeddings, local inference and Redis cache."""
    def __init__(self, settings):
        import threading
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _embed(self, texts, purpose):
        from src.rag.cache import cache, cache_key
        settings = self.settings
        keys = {t: cache_key("embedding", ["bge-m3-dense-normalized-v1",
                                          settings.local_embedding_revision, purpose, t]) for t in texts}
        values, missing = {}, []
        for text, key in keys.items():
            value = cache.get(key) if settings.embedding_cache_enabled else None
            if DashScopeEmbeddings._valid_vector(value) and len(value) == 1024:
                values[text] = value
            else:
                missing.append(text)
        if missing:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    path = settings.project_root / settings.local_embedding_path
                    if (path / "REVISION").read_text().strip() != settings.local_embedding_revision:
                        raise ValueError("Embedding snapshot revision does not match configuration")
                    torch.set_num_threads(settings.local_embedding_threads)
                    self._model = SentenceTransformer(str(path), device=settings.local_embedding_device,
                                                      local_files_only=True, trust_remote_code=False)
                encoded = self._model.encode(missing, batch_size=settings.embedding_batch_size,
                                             normalize_embeddings=True, show_progress_bar=False).tolist()
            for text, vector in zip(missing, encoded):
                values[text] = vector
                if settings.embedding_cache_enabled:
                    cache.set(keys[text], vector, settings.embedding_cache_ttl)
        return [values[t] for t in texts]

    def embed_documents(self, texts):
        return self._embed(texts, "document")

    def embed_query(self, text):
        return self._embed([text], "query")[0]


# 全局 Embeddings 实例
_embeddings_instance: Optional[Embeddings] = None


def get_embeddings(
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    provider: Optional[str] = None,
    **kwargs
) -> Embeddings:
    """
    获取向量化嵌入实例（单例模式）
    
    Args:
        model: 嵌入模型名称
        dimensions: 嵌入向量维度 (Qwen 不支持)
        provider: Embedding 提供商 "qwen" 或 "openai"
        **kwargs: 其他参数
    
    Returns:
        Embeddings 实例
    """
    global _embeddings_instance
    
    settings = get_settings()
    
    # Embedding 不跟随对话 LLM 切换，避免 DeepSeek 等无 embedding
    # 端点的 OpenAI 兼容服务导致知识检索整体失效。
    embedding_provider = provider or settings.embedding_provider
    
    # 如果已有实例，直接返回
    if _embeddings_instance is not None:
        return _embeddings_instance
    
    # 根据提供商创建不同的 Embeddings 实例
    if embedding_provider == "local":
        _embeddings_instance = LocalEmbeddings(settings)
    elif embedding_provider == "qwen":
        # 使用 Qwen 的 dashscope SDK
        # 设置代理（dashscope 使用 HTTPX，需要设置环境变量）
        http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
        if http_proxy:
            os.environ['DASHSCOPE_SDK_HTTP_PROXY'] = http_proxy
            os.environ['DASHSCOPE_SDK_HTTPS_PROXY'] = http_proxy

        _embeddings_instance = DashScopeEmbeddings(
            model=model or settings.embedding_model,
            api_key=settings.dashscope_api_key
        )
    else:
        # 使用 OpenAI 兼容的 API
        from langchain_openai import OpenAIEmbeddings

        http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
        https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")

        model_kwargs = {}
        if http_proxy:
            model_kwargs["http_proxy"] = http_proxy
        if https_proxy:
            model_kwargs["https_proxy"] = https_proxy

        _embeddings_instance = OpenAIEmbeddings(
            model=model or settings.embedding_model,
            dimensions=dimensions,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            **model_kwargs,
            **kwargs
        )
    
    return _embeddings_instance


def reset_embeddings():
    """重置 Embeddings 实例"""
    global _embeddings_instance
    _embeddings_instance = None


class EmbeddingManager:
    """嵌入管理器类"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._embeddings: Optional[Embeddings] = None
    
    @property
    def embeddings(self) -> Embeddings:
        """获取 Embeddings 实例"""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档列表进行向量化"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行向量化"""
        return self.embeddings.embed_query(text)
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        test_embedding = self.embed_query("test")
        return len(test_embedding)
