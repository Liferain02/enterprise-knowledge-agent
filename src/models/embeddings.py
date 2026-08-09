"""
向量化嵌入模块
支持 OpenAI Embeddings 和 阿里千问 Embeddings
"""
import os
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
        """对文档列表进行向量化"""
        try:
            import dashscope
            from dashscope import TextEmbedding

            if self.api_key:
                dashscope.api_key = self.api_key

            embeddings = []
            for text in texts:
                response = TextEmbedding.call(
                    model=self.model,
                    input=text
                )
                if response.status_code == 200:
                    embedding = response.output['embeddings'][0]['embedding']
                    embeddings.append(embedding)
                else:
                    raise Exception(f"Embedding API error: {response.message}")

            return embeddings
        except Exception as e:
            raise Exception(f"DashScope embedding error: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行向量化"""
        try:
            import dashscope
            from dashscope import TextEmbedding

            if self.api_key:
                dashscope.api_key = self.api_key

            response = TextEmbedding.call(
                model=self.model,
                input=text
            )

            if response.status_code == 200:
                return response.output['embeddings'][0]['embedding']
            else:
                raise Exception(f"Embedding API error: {response.message}")
        except Exception as e:
            raise Exception(f"DashScope embedding error: {str(e)}")


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
    if embedding_provider == "qwen":
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
