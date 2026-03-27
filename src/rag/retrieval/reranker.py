"""
Reranker 模块 - 文档重排序
默认使用阿里百炼 qwen3-rerank，也支持 BAAI BGE 本地模型
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from config.settings import get_settings


class RerankerManager:
    """Reranker 管理器"""

    def __init__(
        self,
        reranker_model: str = "gte-rerank-v2",
        provider: str = "qwen",  # "qwen" 或 "baai"
        top_n: int = 3,
        score_threshold: float = 0.3
    ):
        """
        初始化 Reranker

        Args:
            reranker_model: Reranker 模型名称
            provider: 提供商 "qwen"(阿里百炼) 或 "baai"(BAAI)
            top_n: 重排序后返回的 top n 结果
            score_threshold: 分数阈值，低于此分数的结果将被过滤
        """
        self.settings = get_settings()
        self.reranker_model = reranker_model
        self.provider = provider
        self.top_n = top_n
        self.score_threshold = score_threshold
        self._reranker = None

    @property
    def reranker(self):
        """获取 Reranker 实例"""
        if self._reranker is None:
            if self.provider == "qwen":
                self._reranker = self._create_qwen_reranker()
            else:
                self._reranker = self._create_baai_reranker()
        return self._reranker

    def _create_qwen_reranker(self):
        """创建阿里百炼 Reranker"""
        try:
            from langchain_community.chat_models import ChatOpenAI
            from langchain_community.retrievers import BingVectorStore
        except ImportError:
            pass

        # 使用阿里百炼的 Reranker API
        # 注意：需要安装 dashscope 并配置 API Key
        return QwenReranker(
            model=self.reranker_model,
            api_key=self.settings.dashscope_api_key,
            top_n=self.top_n,
            score_threshold=self.score_threshold
        )

    def _create_baai_reranker(self):
        """创建 BAAI BGE Reranker"""
        return BGEHReranker(
            model=self.reranker_model,
            top_n=self.top_n,
            score_threshold=self.score_threshold
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_n: 返回 top n 结果，默认使用初始化时的 top_n

        Returns:
            重排序后的文档列表（包含原始文档和分数）
        """
        if not documents:
            return []

        top_n = top_n or self.top_n

        # 调用 Reranker
        try:
            results = self.reranker.rerank(query, documents, top_n)
            return results
        except Exception as e:
            print(f"Reranker 错误: {e}")
            # 如果 Reranker 失败，使用均匀分布（保持原始顺序）
            return [(doc, 1.0 / len(documents)) for doc in documents[:top_n]]

    def get_compression_retriever(
        self,
        base_retriever: BaseRetriever,
        llm=None
    ) -> ContextualCompressionRetriever:
        """
        获取压缩检索器（使用 Reranker 作为压缩器）

        Args:
            base_retriever: 基础检索器
            llm: LLM 实例（可选，用于 LLMChainExtractor）

        Returns:
            压缩后的检索器
        """
        # 使用 Reranker 作为文档压缩器
        compressor = DocumentRerankerCompressor(
            reranker_manager=self
        )

        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )


class QwenReranker:
    """阿里百炼 Reranker 实现"""

    def __init__(
        self,
        model: str = "gte-rerank-v2",
        api_key: str = None,
        top_n: int = 3,
        score_threshold: float = 0.3
    ):
        self.model = model
        self.api_key = api_key
        self.top_n = top_n
        self.score_threshold = score_threshold

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = None
    ) -> List[tuple[Document, float]]:
        """使用阿里百炼 Reranker API 进行重排序"""
        import os

        top_n = top_n or self.top_n

        # 设置代理
        http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
        if http_proxy:
            os.environ['DASHSCOPE_SDK_HTTP_PROXY'] = http_proxy
            os.environ['DASHSCOPE_SDK_HTTPS_PROXY'] = http_proxy

        try:
            import dashscope
            from dashscope import TextReRank

            # 每次调用都设置 API Key，确保使用正确的配置
            if self.api_key:
                dashscope.api_key = self.api_key

            # 准备文档列表（需要字符串格式）
            doc_texts = [doc.page_content for doc in documents]

            # 调用 TextReRank API
            response = TextReRank.call(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=min(top_n, len(doc_texts)),
                return_documents=True
            )

            if response.status_code == 200:
                results = []
                for item in response.output['results']:
                    doc_index = item['index']
                    score = item['relevance_score']
                    if score >= self.score_threshold:
                        results.append((documents[doc_index], score))

                # 按分数降序排序
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_n]
            else:
                print(f"Reranker API 错误: {response.message}")
                # 使用均匀分布
                return [(doc, 1.0 / len(documents)) for doc in documents[:top_n]]

        except ImportError:
            print("dashscope 未安装，使用备用排序")
            # 使用均匀分布
            return [(doc, 1.0 / len(documents)) for doc in documents[:top_n]]
        except Exception as e:
            print(f"Reranker 错误: {e}")
            # 使用均匀分布
            return [(doc, 1.0 / len(documents)) for doc in documents[:top_n]]


class BGEHReranker:
    """BAAI BGE Reranker 实现（本地模型）"""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 3,
        score_threshold: float = 0.3
    ):
        self.model = model
        self.top_n = top_n
        self.score_threshold = score_threshold
        self._model = None

    @property
    def model_instance(self):
        """获取模型实例"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model)
            except ImportError:
                raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = None
    ) -> List[tuple[Document, float]]:
        """使用 BGE Reranker 模型进行重排序"""
        top_n = top_n or self.top_n

        try:
            # 准备 query-document 对
            pairs = [(query, doc.page_content) for doc in documents]

            # 获取分数
            scores = self.model_instance.predict(pairs)

            # 组合结果并按分数降序排序
            results = []
            for i, doc in enumerate(documents):
                score = float(scores[i])
                if score >= self.score_threshold:
                    results.append((doc, score))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_n]

        except Exception as e:
            print(f"BGE Reranker 错误: {e}")
            # 如果 Reranker 失败，使用均匀分布（每个文档分数相同，保持原始顺序）
            return [(doc, 1.0 / len(documents)) for doc in documents[:top_n]]


class DocumentRerankerCompressor:
    """文档压缩器 - 使用 Reranker"""

    def __init__(self, reranker_manager: RerankerManager):
        self.reranker_manager = reranker_manager

    def compress_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """压缩文档 - 重排序并返回 top n"""
        results = self.reranker_manager.rerank(query, documents)
        return [doc for doc, score in results]


# 全局 Reranker 实例
_reranker_manager: Optional[RerankerManager] = None


def get_reranker_manager(
    reranker_model: str = None,
    provider: str = "qwen",
    top_n: int = None,
    score_threshold: float = None
) -> RerankerManager:
    """
    获取 Reranker 管理器实例

    Args:
        reranker_model: Reranker 模型名称
        provider: 提供商
        top_n: 返回结果数
        score_threshold: 分数阈值

    Returns:
        RerankerManager 实例
    """
    global _reranker_manager

    settings = get_settings()

    if _reranker_manager is None:
        _reranker_manager = RerankerManager(
            reranker_model=reranker_model or getattr(settings, 'reranker_model', 'gte-rerank-v2'),
            provider=provider or getattr(settings, 'reranker_provider', 'qwen'),
            top_n=top_n or getattr(settings, 'reranker_top_n', 3),
            score_threshold=score_threshold or getattr(settings, 'reranker_threshold', 0.3)
        )

    return _reranker_manager


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = None
) -> List[tuple[Document, float]]:
    """
    对文档进行重排序的便捷函数

    Args:
        query: 查询文本
        documents: 待重排序的文档
        top_n: 返回 top n 结果

    Returns:
        重排序后的文档列表
    """
    manager = get_reranker_manager()
    return manager.rerank(query, documents, top_n)
