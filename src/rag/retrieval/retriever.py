"""
检索器模块
支持基础向量检索、混合检索（BM25 + 向量）和带 Reranker 的重排序检索
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..storage.vectorstore import get_vectorstore, VectorStoreManager
from .reranker import get_reranker_manager
from config.settings import get_settings


class RetrieverManager:
    """检索器管理器"""

    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        top_k: int = 5,
        similarity_threshold: float = None,
        use_reranker: bool = True,
        reranker_top_n: int = 3,
        use_hybrid: bool = None  # 新增：是否使用混合检索
    ):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.top_k = top_k or self.settings.retrieval_top_k
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else self.settings.similarity_threshold
        self.use_reranker = use_reranker
        self.reranker_top_n = reranker_top_n
        # 混合检索配置
        self.use_hybrid = use_hybrid if use_hybrid is not None else self.settings.hybrid_search_enabled
        self.hybrid_vector_weight = self.settings.hybrid_vector_weight
        self.hybrid_bm25_weight = self.settings.hybrid_bm25_weight
        
        self._retriever = None
        self._reranker_manager = None
        self._hybrid_manager = None

    @property
    def retriever(self):
        """获取检索器实例"""
        if self._retriever is None:
            vectorstore = get_vectorstore(self.collection_name)
            # 使用更宽松的检索配置
            self._retriever = vectorstore.as_retriever(
                search_type="similarity",  # 使用简单的相似度检索
                search_kwargs={
                    "k": self.top_k
                }
            )
        return self._retriever

    @property
    def reranker_manager(self):
        """获取 Reranker 管理器"""
        if self._reranker_manager is None and self.use_reranker:
            self._reranker_manager = get_reranker_manager()
        return self._reranker_manager
    
    @property
    def hybrid_manager(self):
        """获取混合检索管理器"""
        if self._hybrid_manager is None and self.use_hybrid:
            from .hybrid_retriever import get_hybrid_retriever_manager
            self._hybrid_manager = get_hybrid_retriever_manager(
                collection_name=self.collection_name,
                top_k=self.top_k,
                vector_weight=self.hybrid_vector_weight,
                bm25_weight=self.hybrid_bm25_weight
            )
        return self._hybrid_manager

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        搜索文档
        支持混合检索（BM25 + 向量）和基础向量检索
        """
        k = k or self.top_k
        
        # 如果启用混合检索，使用混合检索
        if self.use_hybrid and self.hybrid_manager:
            return self.hybrid_manager.search(query, k=k)
        
        # 否则使用基础向量检索
        vectorstore = get_vectorstore(self.collection_name)
        return vectorstore.similarity_search(
            query,
            k=k,
            filter=filter
        )
    
    def search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[tuple[Document, float]]:
        """带分数的搜索"""
        k = k or self.top_k
        
        # 如果启用混合检索
        if self.use_hybrid and self.hybrid_manager:
            results = self.hybrid_manager.search_with_scores(query, k=k)
            # 转换为 (doc, score) 格式
            return [(doc, score) for doc, score, _ in results]
        
        # 否则使用基础向量检索
        vectorstore = get_vectorstore(self.collection_name)
        return vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )

    def search_with_rerank(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[tuple[Document, float]]:
        """
        带重排序的搜索
        先检索更多候选文档，然后用 Reranker 重排序
        """
        k = k or self.top_k

        # 1. 先检索更多候选文档（通常是 top_n 的 2-3 倍）
        candidate_k = k * 3

        # 使用 search_with_score 获取带分数的结果
        scored_candidates = self.search_with_score(query, k=candidate_k, filter=filter)
        candidates = [doc for doc, score in scored_candidates]

        if not candidates:
            return []

        # 2. 使用 Reranker 重排序
        if self.use_reranker and self.reranker_manager:
            results = self.reranker_manager.rerank(query, candidates, top_n=k)
            return results
        else:
            # 如果没有 Reranker，返回原始排序
            return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(candidates[:k])]

    def format_search_results(
        self,
        results: List[Document],
        include_metadata: bool = True
    ) -> str:
        """格式化搜索结果为文本"""
        if not results:
            return "未找到相关内容"

        formatted_parts = []

        for i, doc in enumerate(results, 1):
            part = f"【文档 {i}】\n"
            part += f"内容: {doc.page_content}\n"

            if include_metadata and doc.metadata:
                metadata_str = ", ".join(
                    f"{k}: {v}" for k, v in doc.metadata.items()
                )
                part += f"元数据: {metadata_str}\n"

            formatted_parts.append(part)

        return "\n".join(formatted_parts)

    def format_results_with_score(
        self,
        results: List[tuple[Document, float]],
        include_metadata: bool = True
    ) -> str:
        """格式化带分数的搜索结果"""
        if not results:
            return "未找到相关内容"

        formatted_parts = []

        for i, (doc, score) in enumerate(results, 1):
            part = f"【文档 {i}】(相关性: {score:.4f})\n"
            part += f"内容: {doc.page_content}\n"

            if include_metadata and doc.metadata:
                metadata_str = ", ".join(
                    f"{k}: {v}" for k, v in doc.metadata.items()
                )
                part += f"元数据: {metadata_str}\n"

            formatted_parts.append(part)

        return "\n".join(formatted_parts)


# 全局实例
_retriever_manager: Optional[RetrieverManager] = None


def get_retriever_manager() -> RetrieverManager:
    """获取检索器管理器实例"""
    global _retriever_manager
    if _retriever_manager is None:
        _retriever_manager = RetrieverManager()
    return _retriever_manager


def retrieve_documents(
    query: str,
    k: Optional[int] = None
) -> List[Document]:
    """检索文档的便捷函数"""
    manager = get_retriever_manager()
    return manager.search(query, k=k)


def format_retrieved_context(
    query: str,
    k: Optional[int] = None
) -> str:
    """检索并格式化上下文的便捷函数"""
    manager = get_retriever_manager()
    results = manager.search(query, k=k)
    return manager.format_search_results(results)


def retrieve_documents_with_rerank(
    query: str,
    k: Optional[int] = None
) -> List[tuple[Document, float]]:
    """
    检索文档并重排序的便捷函数

    Args:
        query: 查询文本
        k: 返回结果数

    Returns:
        重排序后的文档列表（包含分数）
    """
    manager = get_retriever_manager()
    return manager.search_with_rerank(query, k=k)


def format_retrieved_context_with_rerank(
    query: str,
    k: Optional[int] = None
) -> str:
    """
    检索并格式化上下文的便捷函数（带重排序）

    Args:
        query: 查询文本
        k: 返回结果数

    Returns:
        格式化后的上下文字符串
    """
    manager = get_retriever_manager()
    results = manager.search_with_rerank(query, k=k)
    return manager.format_results_with_score(results)

