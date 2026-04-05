"""
混合检索模块 - Tier 1 开源复用
结合 bm25s（BM25）+ 向量检索，支持多种融合策略。

bm25s: Rust+Python 实现，比 rank_bm25 快 10 倍，内置中文分词（jieba）。

参考：
- https://github.com/xhluca/bm25s
"""
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import jieba
import logging
from rank_bm25 import BM25Okapi
from ..storage.vectorstore import get_vectorstore
from config.settings import get_settings

logger = logging.getLogger(__name__)


# RRF 融合常数（k 越大，各方法贡献越均衡）
RRF_K = 60

# jieba 用户词典全局加载（只加载一次）
_JIEBA_DICT_LOADED = False


def _ensure_jieba_dict():
    """确保 jieba 用户词典只加载一次"""
    global _JIEBA_DICT_LOADED
    if not _JIEBA_DICT_LOADED:
        import os as _os
        dict_path = _os.path.join(
            _os.path.dirname(__file__),
            "..", "..", "..",  # src/rag/retrieval/ -> project root
            "data", "knowledge", "dicts", "jieba_user_dict.txt"
        )
        dict_path = _os.path.normpath(dict_path)
        if _os.path.exists(dict_path):
            import jieba as _jieba
            _jieba.load_userdict(dict_path)
            _JIEBA_DICT_LOADED = True


class HybridRetrieverManager:
    """混合检索管理器 - 结合 BM25 和向量检索"""

    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        top_k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        enable_bm25: bool = True,
        enable_vector: bool = True,
        fusion_strategy: str = "rrf",
    ):
        """
        初始化混合检索管理器

        Args:
            collection_name: 向量数据库集合名
            top_k: 返回结果数
            vector_weight: 向量检索权重 (0-1)，仅用于 score fusion
            bm25_weight: BM25 检索权重 (0-1)，仅用于 score fusion
            enable_bm25: 是否启用 BM25
            enable_vector: 是否启用向量检索
            fusion_strategy: 融合策略: "rrf"（推荐）或 "score"
        """
        self.settings = get_settings()
        self.collection_name = collection_name
        self.top_k = top_k or self.settings.retrieval_top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_bm25 = enable_bm25
        self.enable_vector = enable_vector
        self.fusion_strategy = fusion_strategy

        self._vector_retriever = None
        self._bm25_retriever = None
        self._ensemble_retriever = None

        # 用于 BM25 的文档集合
        self._documents: List[Document] = []
        self._vectorstore = None

        # 预构建 BM25 索引（避免每次 search 重建）
        self._tokenized_corpus: List[List[str]] = []
        self._bm25_index: Optional[Any] = None

        # 加载 jieba 用户词典（只需加载一次）
        _ensure_jieba_dict()

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
        设置 BM25 使用的文档集合，并预构建索引。

        优先级：bm25s（Tier 1，Rust 实现，10x faster）> rank_bm25（备用）
        """
        self._documents = documents
        if not documents:
            return

        if self.enable_bm25:
            texts = [doc.page_content for doc in documents]
            tokens = [self._tokenize(text) for text in texts]

            # Tier 1: bm25s（Rust+Python，极速）
            try:
                import bm25s
                self._bm25_index = bm25s.BM25()
                self._bm25_index.index(tokens)
                self._bm25_backend = "bm25s"
                logger.info(f"混合检索 BM25 引擎: bm25s ({len(documents)} 篇文档)")
            except ImportError:
                # 备用：rank_bm25
                self._bm25_index = BM25Okapi(tokens)
                self._bm25_backend = "rank_bm25"
                logger.info(f"混合检索 BM25 引擎: rank_bm25 ({len(documents)} 篇文档)")

    def _tokenize(self, text: str) -> List[str]:
        """分词（统一使用 jieba）"""
        return [w.lower() for w in jieba.cut(text) if w.strip()]

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
        混合搜索（不带分数，用于不需要分数的场景）。

        底层调用 search_with_scores()，丢弃分数后返回文档列表。
        与 search_with_scores() 共用相同的向量 + BM25 混合逻辑。

        Args:
            query: 查询文本
            k: 返回结果数
            filter: 元数据过滤条件

        Returns:
            检索到的文档列表
        """
        k = k or self.top_k

        if self.enable_bm25:
            results_with_scores = self.search_with_scores(query, k=k, filter=filter)
            return [doc for doc, _, _ in results_with_scores]

        # 纯向量检索
        return self.vector_retriever.invoke(query)[:k]

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float, str]]:
        """
        带分数的混合搜索

        融合策略：
        - RRF（Reciprocal Rank Fusion）：按排名融合，不依赖分数绝对值
          RRF_score(d) = Σ weight_i / (k + rank_i(d))
          其中 k=60（融合常数），weight_i 来自配置权重
        - 优势：不受两路分数范围差异影响，BM25 和向量各凭排名贡献

        Args:
            query: 查询文本
            k: 返回结果数
            filter: 元数据过滤条件

        Returns:
            (文档, 融合分数, 主要来源) 元组列表，按融合分降序
        """
        k = k or self.top_k
        candidate_k = k * 2  # 两路各取 2k，留足融合余量

        # ── 自适应权重：根据查询长度动态调整 BM25 vs 向量权重 ─────────────
        # 短查询（≤8字）：BM25 更擅长精确关键词匹配，提升其权重
        # 长查询（>8字）：向量语义检索更强，降低 BM25 权重
        _ensure_jieba_dict()  # 安全加载用户词典
        query_len = len(query.strip())
        if query_len <= 4:
            # 极短查询（≤4字）：BM25 占主导
            vec_w = 0.3
            bm_w = 0.7
        elif query_len <= 8:
            # 短查询（5-8字）：BM25 稍强
            vec_w = 0.4
            bm_w = 0.6
        else:
            # 正常查询（>8字）：保持配置权重
            vec_w = self.vector_weight
            bm_w = self.bm25_weight

        # ── 第一路：向量检索（返回 rank）────────────────────────────────
        vector_ranked: Dict[str, Tuple[Document, int]] = {}  # key → (doc, rank)
        if self.enable_vector:
            try:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=candidate_k
                )
                for rank, (doc, raw_score) in enumerate(results, 1):
                    key = hash(doc.page_content)
                    if key not in vector_ranked:
                        vector_ranked[key] = (doc, rank)
            except Exception as e:
                print(f"向量检索错误: {e}")

        # ── 第二路：BM25 检索（返回 rank）─────────────────────────────
        bm25_ranked: Dict[str, Tuple[Document, int]] = {}  # key → (doc, rank)
        if self.enable_bm25 and self._bm25_index is not None and self._documents:
            try:
                tokens = self._tokenize(query)
                all_scores: List[float] = []

                if getattr(self, "_bm25_backend", None) == "bm25s":
                    # Tier 1: bm25s（Rust 实现，极速）
                    import bm25s
                    import numpy as np
                    results = self._bm25_index.retrieve(
                        [tokens],
                        corpus=None,
                        k=min(candidate_k * 2, len(self._documents)),
                        sorted=True,
                        return_as="tuple",
                        show_progress=False,
                    )
                    if results is not None and hasattr(results, "documents"):
                        doc_indices = np.asarray(results.documents[0])
                        doc_scores = np.asarray(results.scores[0])
                        all_scores = [0.0] * len(self._documents)
                        for doc_idx, score in zip(doc_indices, doc_scores):
                            if 0 <= doc_idx < len(self._documents):
                                all_scores[int(doc_idx)] = float(score)
                else:
                    # 备用：rank_bm25
                    all_scores = self._bm25_index.get_scores(tokens)

                if all(s == 0 for s in all_scores):
                    for i, text in enumerate(self._documents):
                        t_lower = text.page_content.lower()
                        all_scores[i] = sum(
                            1 for t in tokens if t in t_lower
                        ) / max(len(tokens), 1)

                scored = [(all_scores[i], doc) for i, doc in enumerate(self._documents) if all_scores[i] > 0]
                scored.sort(key=lambda x: x[0], reverse=True)

                for rank, (_, doc) in enumerate(scored[:candidate_k], 1):
                    key = hash(doc.page_content)
                    if key not in bm25_ranked:
                        bm25_ranked[key] = (doc, rank)
            except Exception as e:
                print(f"BM25 检索错误: {e}")

        # ── 融合 ──────────────────────────────────────────────────────
        all_keys = set(vector_ranked) | set(bm25_ranked)
        if not all_keys:
            return []

        if self.fusion_strategy == "rrf":
            # RRF：按排名融合，与分数范围无关
            w_vec = vec_w
            w_bm = bm_w

            fused = []
            for key in all_keys:
                vec_score = 0.0
                bm_score = 0.0
                if key in vector_ranked:
                    _, rank = vector_ranked[key]
                    vec_score = w_vec / (RRF_K + rank)
                if key in bm25_ranked:
                    _, rank = bm25_ranked[key]
                    bm_score = w_bm / (RRF_K + rank)
                fused.append((key, vec_score + bm_score))
        else:
            # Score-level fusion（降级兼容）
            fused = self._score_fusion(all_keys, vector_ranked, bm25_ranked, vec_w, bm_w)

        fused.sort(key=lambda x: x[1], reverse=True)

        # ── 组装返回 ──────────────────────────────────────────────────
        results = []
        for key, score in fused[:k]:
            doc = (
                vector_ranked.get(key, bm25_ranked.get(key, (None, None)))[0]
                if key in vector_ranked or key in bm25_ranked
                else None
            )
            if doc is None:
                continue
            has_vec = key in vector_ranked
            has_bm = key in bm25_ranked
            source = "vector+bm25" if (has_vec and has_bm) else ("vector" if has_vec else "bm25")
            results.append((doc, score, source))

        return results

    def _score_fusion(
        self,
        all_keys: set,
        vector_ranked: Dict,
        bm25_ranked: Dict,
        vec_weight: float = 0.5,
        bm_weight: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        降级：Score-level 融合（仅在 fusion_strategy != "rrf" 时使用）

        对两路分数分别做 Min-Max 归一化后加权求和。
        注意：此方法不如 RRF 稳健，仅作降级兼容。
        """
        def minmax_normalize(
            ranked: Dict[str, Tuple[Document, int]]
        ) -> Dict[str, float]:
            if not ranked:
                return {}
            # 将 rank (1-based, lower=better) 转为相似分 (0-1, higher=better)
            max_rank = max(rank for _, rank in ranked.values())
            return {key: (max_rank - rank + 1) / max_rank for key, (_, rank) in ranked.items()}

        vec_norm = minmax_normalize(vector_ranked)
        bm_norm = minmax_normalize(bm25_ranked)

        fused = []
        for key in all_keys:
            vs = vec_norm.get(key, 0.0)
            bs = bm_norm.get(key, 0.0)
            score = (1 - bm_weight) * vs + bm_weight * bs
            fused.append((key, score))
        return fused

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
    bm25_weight: float = 0.5,
    fusion_strategy: str = "rrf",
) -> HybridRetrieverManager:
    """获取混合检索管理器实例

    Args:
        fusion_strategy: 融合策略，"rrf"（推荐）或 "score"
    """
    global _hybrid_retriever_manager

    settings = get_settings()

    if _hybrid_retriever_manager is None:
        _hybrid_retriever_manager = HybridRetrieverManager(
            collection_name=collection_name,
            top_k=top_k or settings.retrieval_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            fusion_strategy=fusion_strategy,
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
