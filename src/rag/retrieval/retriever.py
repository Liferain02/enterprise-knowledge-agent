"""
检索器模块
支持基础向量检索、混合检索（BM25 + 向量）和带 Reranker 的重排序检索。
集成 ACL 权限过滤：在 Chroma filter 层面实现"检索前过滤"。
"""
import time
import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..storage.vectorstore import get_vectorstore, VectorStoreManager
from .reranker import get_reranker_manager
from config.settings import get_settings
from .acl_filter import build_acl_filter, UserContext, check_doc_access

logger = logging.getLogger(__name__)


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

    def _build_filter(
        self,
        base_filter: Optional[Dict] = None,
        user: Optional[UserContext] = None,
        include_expired: bool = False,
    ) -> Optional[Dict]:
        """
        合并 Chroma filter：基础 filter + ACL filter。
        ACL filter 通过 build_acl_filter() 构建，在检索前完成权限过滤。
        """
        acl_filter = build_acl_filter(user=user, include_expired=include_expired)

        if base_filter is None and acl_filter is None:
            return None
        if base_filter is None:
            return acl_filter
        if acl_filter is None:
            return base_filter

        # AND 组合：同时满足基础过滤和 ACL 权限
        return {"$and": [base_filter, acl_filter]}

    def search_with_acl(
        self,
        query: str,
        k: Optional[int] = None,
        user: Optional[UserContext] = None,
        base_filter: Optional[Dict] = None,
        include_expired: bool = False,
    ) -> List[Document]:
        """
        带 ACL 权限过滤的检索。
        检索前先通过 build_acl_filter() 过滤，仅返回用户有权限看到的文档。

        Args:
            query: 查询文本
            k: 返回数量
            user: 当前用户上下文（用于构建 ACL filter）
            base_filter: 额外的 Chroma filter（如分类过滤）
            include_expired: 是否包含过期文档

        Returns:
            用户有权限访问的文档列表
        """
        k = k or self.top_k

        # 合并 ACL filter 和 base_filter
        final_filter = self._build_filter(
            base_filter=base_filter,
            user=user,
            include_expired=include_expired,
        )

        # ACL 审计日志
        if user:
            logger.debug(
                f"[Retriever] user={user.username}({user.role}), "
                f"filter_keys={list(final_filter.keys()) if final_filter else None}"
            )

        # 检索
        if self.use_hybrid and self.hybrid_manager:
            return self.hybrid_manager.search(query, k=k)

        vectorstore = get_vectorstore(self.collection_name)
        return vectorstore.similarity_search(query, k=k, filter=final_filter)

    def search_with_score_acl(
        self,
        query: str,
        k: Optional[int] = None,
        user: Optional[UserContext] = None,
        base_filter: Optional[Dict] = None,
        include_expired: bool = False,
    ) -> List[tuple[Document, float]]:
        """
        带 ACL 权限过滤和分数的检索。

        Returns:
            (文档, 相关分) 列表，仅包含用户有权限的文档
        """
        k = k or self.top_k
        final_filter = self._build_filter(
            base_filter=base_filter,
            user=user,
            include_expired=include_expired,
        )

        if self.use_hybrid and self.hybrid_manager:
            results = self.hybrid_manager.search_with_scores(query, k=k)
            return [(doc, score) for doc, score, _ in results]

        vectorstore = get_vectorstore(self.collection_name)
        return vectorstore.similarity_search_with_score(query, k=k, filter=final_filter)

    def search_with_rerank(
        self,
        query: str,
        k: Optional[int] = None,
        user: Optional[UserContext] = None,
        base_filter: Optional[Dict] = None,
        include_expired: bool = False,
    ) -> List[tuple[Document, float]]:
        """
        带重排序的检索（集成 ACL 权限过滤）。

        Args:
            user: 当前用户上下文（用于 ACL 过滤）
            base_filter: 额外的 Chroma filter
            include_expired: 是否包含过期文档
        """
        k = k or self.top_k

        # 1. 带 ACL 过滤检索候选文档
        candidate_k = k * 3
        scored_candidates = self.search_with_score_acl(
            query=query,
            k=candidate_k,
            user=user,
            base_filter=base_filter,
            include_expired=include_expired,
        )
        candidates = [doc for doc, score in scored_candidates]

        if not candidates:
            return []

        # 2. Reranker 重排序
        if self.use_reranker and self.reranker_manager:
            results = self.reranker_manager.rerank(query, candidates, top_n=k)
            return results
        else:
            return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(candidates[:k])]

    def filter_results_by_acl(
        self,
        results: List[tuple[Document, float]],
        user: Optional[UserContext],
    ) -> List[tuple[Document, float]]:
        """
        在已有检索结果上做二次 ACL 过滤。
        用于回答后验证：防止 ACL filter 绕过。

        Returns:
            仅保留用户有权限访问的文档
        """
        if not results or not user:
            return results

        filtered = []
        for doc, score in results:
            if check_doc_access(doc.metadata or {}, user):
                filtered.append((doc, score))
            else:
                logger.warning(
                    f"[Retriever] ACL 二次验证过滤："
                    f"用户 {user.username} 无权访问 {doc.metadata.get('source', '?')}"
                )

        return filtered

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

