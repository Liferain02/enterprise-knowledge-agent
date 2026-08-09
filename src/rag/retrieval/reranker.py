"""
Reranker 模块 - Tier 1 开源复用
支持三种后端（按优先级）：
1. FlashRank（Tier 1，5M 参数本地模型，<10ms 延迟，即开即用）
2. Cohere（精度最高，需要 API Key）
3. BGE（本地，需要 GPU）

默认优先尝试 FlashRank，无需 API Key。
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)


class RerankerManager:
    """
    Reranker 管理器 - Tier 1 开源优先

    后端优先级：
    1. FlashRank（Tier 1，5M 参数，<10ms，即开即用）
    2. Cohere（需要 API Key）
    3. BGE（本地，需要 GPU）
    4. 均匀分布（所有后端失败时的降级）
    """

    def __init__(
        self,
        reranker_model: str = "gte-rerank-v2",
        provider: str = "flashrank",  # 默认改为 flashrank
        top_n: int = 3,
        score_threshold: float = 0.3
    ):
        self.settings = get_settings()
        self.reranker_model = reranker_model
        self.provider = provider
        self.top_n = top_n
        self.score_threshold = score_threshold
        self._reranker = None

    @property
    def reranker(self):
        """获取 Reranker 实例（延迟加载）"""
        if self._reranker is None:
            self._reranker = self._init_reranker()
        return self._reranker

    def _init_reranker(self):
        """按优先级初始化 Reranker 后端"""
        # 1. FlashRank（Tier 1，无 API Key 要求，内联实现）
        if self.provider in ("flashrank", "auto"):
            try:
                r = FlashRankReranker()
                logger.info("[Reranker] 使用 FlashRank 后端")
                return r
            except Exception as e:
                logger.warning(f"FlashRank 初始化失败: {e}")

        # 阿里百炼（配置为 qwen 时使用已有的 TextReRank 实现）
        if self.provider == "qwen":
            logger.info("[Reranker] 使用 Qwen 后端")
            return QwenReranker(
                model=self.reranker_model,
                api_key=self.settings.dashscope_api_key,
                top_n=self.top_n,
                score_threshold=self.score_threshold,
            )

        # 2. Cohere（需要 API Key）
        if self.provider in ("cohere", "auto"):
            try:
                r = self._create_cohere_reranker()
                if r:
                    logger.info("[Reranker] 使用 Cohere 后端")
                    return r
            except Exception as e:
                logger.warning(f"Cohere 初始化失败: {e}")

        # 3. BGE 本地
        if self.provider in ("baai", "bge", "auto"):
            try:
                r = self._create_baai_reranker()
                logger.info("[Reranker] 使用 BGE 后端")
                return r
            except Exception as e:
                logger.warning(f"BGE 初始化失败: {e}")

        # 4. 降级：返回 None，走均匀分布
        logger.warning("[Reranker] 所有后端失败，使用均匀分布降级")
        return None

    def _create_cohere_reranker(self):
        """创建 Cohere Reranker（需要 API Key）"""
        try:
            from llama_index_postprocessor_cohere_rerank import CohereRerank
            api_key = self.settings.dashscope_api_key  # Cohere 也用此字段（需用户配置）
            return CohereRerank(api_key=api_key, top_n=self.top_n)
        except ImportError:
            return None

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
        """对文档进行重排序。后端优先级：FlashRank > Cohere > BGE > 均匀分布。"""
        if not documents:
            return []

        top_n = top_n or self.top_n

        if self.reranker is not None:
            try:
                return self.reranker.rerank(query, documents, top_n)
            except Exception as e:
                print(f"Reranker 错误: {e}")

        # 降级：均匀分布
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


class FlashRankReranker:
    """
    FlashRank 本地轻量级重排序

    使用 MS MARCO 数据集上微调的 MiniLM 模型（22M 参数，约 80MB），
    延迟 < 10ms，支持中文。
    """

    def __init__(
        self,
        model_name: str = "Ms-MiniLM-L6-v2",
        top_n: int = 3,
    ):
        self.model_name = model_name
        self.top_n = top_n
        self._client = None

    @property
    def client(self):
        """延迟加载 FlashRank 客户端"""
        if self._client is None:
            try:
                from flashrank import Ranker
                self._client = Ranker(model_name=self.model_name)
            except ImportError:
                raise ImportError(
                    "请安装 flashrank: pip install flashrank"
                )
        return self._client

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = None,
    ) -> List[tuple[Document, float]]:
        """
        使用 FlashRank 对文档重排序

        Args:
            query: 查询文本
            documents: 待重排序的文档
            top_n: 返回 top n 结果

        Returns:
            [(doc, score), ...] 按分数降序
        """
        top_n = top_n or self.top_n

        if not documents:
            return []

        try:
            # 构造 FlashRank 输入格式
            flashrank_docs = [
                {"id": i, "text": doc.page_content}
                for i, doc in enumerate(documents)
            ]

            # 重排序
            results = self.client.rerank(query, flashrank_docs)

            # 转换回 (doc, score) 格式
            reranked = []
            for item in results[:top_n]:
                doc_idx = item["id"]
                score = item["score"]
                reranked.append((documents[doc_idx], float(score)))

            return reranked

        except Exception as e:
            logger.warning(f"FlashRank rerank 失败: {e}")
            return [(doc, 1.0 / len(documents)) for doc in documents[:top_n]]


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
    provider: str = None,
    top_n: int = None,
    score_threshold: float = None
) -> RerankerManager:
    """
    获取 Reranker 管理器实例。

    后端优先级（settings.reranker_provider）：
    - "flashrank"（默认，Tier 1，无需 API Key）
    - "cohere"（需要 API Key）
    - "baai"（本地 BGE，需要 GPU）
    - "qwen"（阿里百炼，需要 API Key）
    """
    global _reranker_manager

    settings = get_settings()
    effective_provider = provider or getattr(settings, "reranker_provider", "flashrank")

    if _reranker_manager is None:
        _reranker_manager = RerankerManager(
            reranker_model=reranker_model or getattr(settings, "reranker_model", "gte-rerank-v2"),
            provider=effective_provider,
            top_n=top_n or getattr(settings, "reranker_top_n", 3),
            score_threshold=score_threshold or getattr(settings, "reranker_threshold", 0.3)
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
