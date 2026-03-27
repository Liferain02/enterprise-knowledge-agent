"""
混合检索模块 - 结合 BM25（关键词检索）+ 向量检索
支持多种混合检索策略
"""
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import jieba
from rank_bm25 import BM25Okapi
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
        self._vectorstore = None

        # 预构建 BM25 索引（避免每次 search 重建）
        self._tokenized_corpus: List[List[str]] = []
        self._bm25_index: Optional[Any] = None

    @property
    def vectorstore(self):
        """获取向量存储实例"""
        if self._vectorstore is None:
            self._vectorstore = get_vectorstore(self.collection_name)
        return self._vectorstore

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
        设置 BM25 使用的文档集合，并预构建索引

        Args:
            documents: 文档列表
        """
        self._documents = documents
        if self.enable_bm25 and documents:
            self._bm25_retriever = BM25Retriever.from_documents(
                documents,
                k=self.top_k * 2  # 检索更多候选
            )
            # 预构建 BM25 索引（避免每次 search_with_score 重建）
            texts = [doc.page_content for doc in documents]
            self._tokenized_corpus = [
                [w.lower() for w in jieba.cut(text) if w.strip()]
                for text in texts
            ]
            self._bm25_index = BM25Okapi(self._tokenized_corpus)

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
                # 使用 similarity_search_with_score 获取实际分数
                vector_results = self.vectorstore.similarity_search_with_score(
                    query, k=self.top_k * 2
                )
                vector_docs = [doc for doc, _ in vector_results]
                vector_scores = [score for _, score in vector_results]

                # ChromaDB 返回的是余弦距离（0-2，越小越相似）
                # 转换为相似度分数：1 - (distance / 2)，范围 0-1
                # 距离 0 → 相似度 1，距离 2 → 相似度 0
                vector_similarities = [1.0 - (s / 2.0) for s in vector_scores]

                # 对向量相似度分数进行 Min-Max 归一化到 0-1
                vector_scores_normalized = self._min_max_normalize(vector_similarities)

                for i, doc in enumerate(vector_docs[:k]):
                    # 将原始分数和归一化分数都存入 metadata
                    doc.metadata = doc.metadata or {}
                    doc.metadata["score"] = vector_similarities[i]
                    doc.metadata["score_normalized"] = vector_scores_normalized[i]
                    results.append((doc, vector_scores_normalized[i], "vector"))
            except Exception as e:
                print(f"向量检索错误: {e}")

        # BM25 检索
        if self.enable_bm25 and self.bm25_retriever:
            try:
                # 获取文档列表
                bm25_docs = self.bm25_retriever.invoke(query)
                if bm25_docs and self._documents:
                    # 复用预构建的 BM25 索引和 tokenized 语料
                    texts = [doc.page_content for doc in self._documents]
                    tokenized_corpus = self._tokenized_corpus
                    bm25 = self._bm25_index

                    # 对查询分词并计算分数
                    query_tokens = [w.lower() for w in jieba.cut(query) if w.strip()]
                    all_scores = bm25.get_scores(query_tokens)

                    # 检查是否有非零分数，如果没有则使用关键词匹配
                    if all(s == 0 for s in all_scores):
                        # BM25 分数为 0（常见词/英文），使用关键词匹配
                        # 简单的词匹配：统计查询词在文档中出现的次数
                        for i, text in enumerate(texts):
                            text_lower = text.lower()
                            match_count = sum(1 for token in query_tokens if token in text_lower)
                            # 基于匹配比例给分
                            all_scores[i] = match_count / max(len(query_tokens), 1)

                    # 获取返回文档的分数
                    doc_texts = {doc.page_content: i for i, doc in enumerate(self._documents)}
                    bm25_scores = []
                    for doc in bm25_docs:
                        idx = doc_texts.get(doc.page_content)
                        if idx is not None:
                            bm25_scores.append(all_scores[idx])
                        else:
                            bm25_scores.append(0.0)

                    # 对 BM25 分数进行 Min-Max 归一化到 0-1
                    if max(bm25_scores) > 0:
                        bm25_scores_normalized = self._min_max_normalize(bm25_scores)
                    else:
                        # 如果所有分数都是0，使用均匀分布
                        bm25_scores_normalized = [1.0 / len(bm25_scores)] * len(bm25_scores)

                    for i, doc in enumerate(bm25_docs[:k]):
                        # 检查是否已存在（基于内容去重）
                        doc_exists = any(
                            doc.page_content == existing[0].page_content
                            for existing in results
                        )
                        if not doc_exists:
                            # 将原始分数和归一化分数都存入 metadata
                            doc.metadata = doc.metadata or {}
                            doc.metadata["score"] = bm25_scores[i]
                            doc.metadata["score_normalized"] = bm25_scores_normalized[i]
                            results.append((doc, bm25_scores_normalized[i], "bm25"))
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

    def _get_vector_scores(self, documents: List[Document]) -> List[float]:
        """
        获取向量检索的相似度分数（已归一化到 0-1）

        Args:
            documents: 文档列表

        Returns:
            归一化后的分数列表
        """
        if not documents:
            return []

        # 向量检索通常返回余弦相似度，已在 0-1 范围
        # 但为了安全起见，也做一次 Min-Max 归一化
        scores = []
        for doc in documents:
            # 从元数据中获取相似度分数（如果存在）
            score = doc.metadata.get("score") if doc.metadata else None
            if score is None:
                # 默认分数为 1.0（如果无法获取）
                score = 1.0
            scores.append(score)

        return self._min_max_normalize(scores)

    def _get_bm25_scores(self, documents: List[Document]) -> List[float]:
        """
        获取 BM25 检索的分数并进行归一化

        Args:
            documents: 文档列表

        Returns:
            归一化后的分数列表
        """
        if not documents:
            return []

        # 从元数据中获取 BM25 分数
        scores = []
        for doc in documents:
            score = doc.metadata.get("score") if doc.metadata else None
            if score is None:
                # 如果无法获取分数，默认设为较低值
                score = 0.0
            scores.append(score)

        # 对 BM25 分数进行 Min-Max 归一化到 0-1
        return self._min_max_normalize(scores)

    def _min_max_normalize(self, scores: List[float]) -> List[float]:
        """
        Min-Max 归一化：将分数映射到 0-1 范围

        Args:
            scores: 原始分数列表

        Returns:
            归一化后的分数列表
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # 如果所有分数相同，避免除以零
        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

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
