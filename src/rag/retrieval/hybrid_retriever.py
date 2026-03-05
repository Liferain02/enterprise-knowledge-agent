"""
混合检索模块 - 结合 BM25（关键词检索）+ 向量检索
支持多种混合检索策略
"""
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from ..storage.vectorstore import get_vectorstore
from config.settings import get_settings


class HybridRetrieverManager:
    """混合检索管理器 - 结合 BM25 和向量检索"""

    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        top_k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        enable_bm25: bool = True,
        enable_vector: bool = True
    ):
        """
        初始化混合检索管理器

        Args:
            collection_name: 向量数据库集合名
            top_k: 返回结果数
            vector_weight: 向量检索权重 (0-1)
            bm25_weight: BM25 检索权重 (0-1)
            enable_bm25: 是否启用 BM25
            enable_vector: 是否启用向量检索
        """
        self.settings = get_settings()
        self.collection_name = collection_name
        self.top_k = top_k or self.settings.retrieval_top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_bm25 = enable_bm25
        self.enable_vector = enable_vector

        self._vector_retriever = None
        self._bm25_retriever = None
        self._ensemble_retriever = None

        # 用于 BM25 的文档集合
        self._documents: List[Document] = []

    @property
    def vector_retriever(self) -> BaseRetriever:
        """获取向量检索器"""
        if self._vector_retriever is None:
            vectorstore = get_vectorstore(self.collection_name)
            self._vector_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k * 2}  # 检索更多候选
            )
        return self._vector_retriever

    def set_documents(self, documents: List[Document]):
        """
        设置 BM25 使用的文档集合

        Args:
            documents: 文档列表
        """
        self._documents = documents
        if self.enable_bm25 and documents:
            self._bm25_retriever = BM25Retriever.from_documents(
                documents,
                k=self.top_k * 2  # 检索更多候选
            )

    @property
    def bm25_retriever(self) -> Optional[BM25Retriever]:
        """获取 BM25 检索器"""
        if self._bm25_retriever is None and self.enable_bm25 and self._documents:
            self._bm25_retriever = BM25Retriever.from_documents(
                self._documents,
                k=self.top_k * 2
            )
        return self._bm25_retriever

    @property
    def ensemble_retriever(self) -> Optional[EnsembleRetriever]:
        """获取集成检索器"""
        if self._ensemble_retriever is not None:
            return self._ensemble_retriever

        retrievers = []
        weights = []

        if self.enable_vector:
            retrievers.append(self.vector_retriever)
            weights.append(self.vector_weight)

        if self.enable_bm25 and self.bm25_retriever:
            retrievers.append(self.bm25_retriever)
            weights.append(self.bm25_weight)

        if not retrievers:
            return None

        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        self._ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )

        return self._ensemble_retriever

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        混合搜索

        Args:
            query: 查询文本
            k: 返回结果数
            filter: 元数据过滤条件

        Returns:
            检索到的文档列表
        """
        k = k or self.top_k

        # 如果有集成检索器，使用集成检索
        if self.ensemble_retriever:
            return self.ensemble_retriever.invoke(query)[:k]

        # 否则回退到向量检索
        return self.vector_retriever.invoke(query)[:k]

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float, str]]:
        """
        带分数的混合搜索（包含来源信息）

        Args:
            query: 查询文本
            k: 返回结果数
            filter: 元数据过滤条件

        Returns:
            (文档, 分数, 来源类型) 元组列表
        """
        k = k or self.top_k
        results = []

        # 向量检索
        if self.enable_vector:
            try:
                vector_docs = self.vector_retriever.invoke(query)
                for doc in vector_docs[:k]:
                    results.append((doc, 1.0, "vector"))
            except Exception as e:
                print(f"向量检索错误: {e}")

        # BM25 检索
        if self.enable_bm25 and self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
                for doc in bm25_docs[:k]:
                    # 检查是否已存在
                    doc_exists = any(
                        doc.page_content == existing[0].page_content
                        for existing in results
                    )
                    if not doc_exists:
                        results.append((doc, 1.0, "bm25"))
            except Exception as e:
                print(f"BM25 检索错误: {e}")

        # 按来源优先级和分数排序
        # 权重: 向量 > BM25
        def sort_key(item):
            doc, score, source = item
            source_priority = 1.0 if source == "vector" else 0.5
            return source_priority * score

        results.sort(key=sort_key, reverse=True)

        return results[:k]

    def format_results(
        self,
        results: List[Tuple[Document, float, str]],
        include_metadata: bool = True
    ) -> str:
        """格式化混合检索结果"""
        if not results:
            return "未找到相关内容"

        formatted_parts = []

        for i, (doc, score, source) in enumerate(results, 1):
            source_label = "向量检索" if source == "vector" else "关键词检索"
            part = f"【文档 {i}】({source_label}, 相关度: {score:.4f})\n"
            part += f"内容: {doc.page_content[:300]}...\n" if len(doc.page_content) > 300 else f"内容: {doc.page_content}\n"

            if include_metadata and doc.metadata:
                metadata_str = ", ".join(
                    f"{k}: {v}" for k, v in doc.metadata.items()
                )
                part += f"元数据: {metadata_str}\n"

            formatted_parts.append(part)

        return "\n".join(formatted_parts)


# 全局实例
_hybrid_retriever_manager: Optional[HybridRetrieverManager] = None


def get_hybrid_retriever_manager(
    collection_name: str = "enterprise_knowledge",
    top_k: int = None,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> HybridRetrieverManager:
    """获取混合检索管理器实例"""
    global _hybrid_retriever_manager

    settings = get_settings()

    if _hybrid_retriever_manager is None:
        _hybrid_retriever_manager = HybridRetrieverManager(
            collection_name=collection_name,
            top_k=top_k or settings.retrieval_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )

    return _hybrid_retriever_manager


def hybrid_search(
    query: str,
    k: int = None,
    documents: List[Document] = None
) -> List[Document]:
    """
    混合搜索的便捷函数

    Args:
        query: 查询文本
        k: 返回结果数
        documents: BM25 使用的文档集合

    Returns:
        检索到的文档列表
    """
    manager = get_hybrid_retriever_manager()

    if documents:
        manager.set_documents(documents)

    return manager.search(query, k=k)


def hybrid_search_with_scores(
    query: str,
    k: int = None,
    documents: List[Document] = None
) -> List[Tuple[Document, float, str]]:
    """
    带分数的混合搜索

    Args:
        query: 查询文本
        k: 返回结果数
        documents: BM25 使用的文档集合

    Returns:
        (文档, 分数, 来源类型) 元组列表
    """
    manager = get_hybrid_retriever_manager()

    if documents:
        manager.set_documents(documents)

    return manager.search_with_scores(query, k=k)
