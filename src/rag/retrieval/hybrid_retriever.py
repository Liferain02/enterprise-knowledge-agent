"""
混合检索模块 - Tier 1 开源复用
结合 bm25s（BM25）+ 向量检索，支持多种融合策略。
同时集成 ACL 权限过滤：在检索前完成权限控制。

bm25s: Rust+Python 实现，比 rank_bm25 快 10 倍，内置中文分词（jieba）。

参考：
- https://github.com/xhluca/bm25s
"""
import hashlib
import json
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
import jieba
import logging
from rank_bm25 import BM25Okapi
from ..storage.vectorstore import get_vectorstore
from config.settings import get_settings
from .acl_filter import build_acl_filter, UserContext, check_doc_access

logger = logging.getLogger(__name__)


# RRF 融合常数（k 越大，各方法贡献越均衡）
RRF_K = 60

# jieba 用户词典全局加载（只加载一次）
_JIEBA_DICT_LOADED = False


def _document_identity(doc: Document) -> str:
    """返回跨进程稳定、且不会合并不同来源同文内容的 Chunk 身份。"""
    metadata = doc.metadata or {}
    identity = {
        "source": metadata.get("doc_id") or metadata.get("source") or "",
        "version": (
            metadata.get("document_version")
            or metadata.get("version_id")
            or metadata.get("version")
            or ""
        ),
        "chunk_id": metadata.get("chunk_id") or "",
        "chunk_hash": metadata.get("chunk_hash") or "",
    }
    # 旧文档可能没有 chunk_id/chunk_hash；内容摘要只作为稳定兜底，并与
    # 来源、版本共同参与计算，避免不同文档的相同段落被错误合并。
    if not identity["chunk_id"] and not identity["chunk_hash"]:
        identity["content_sha256"] = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        collection_name: str = "lab_knowledge",
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

    def _ensure_bm25_index(self) -> None:
        """首次检索时从当前 Chroma snapshot 构建 BM25，避免名义 Hybrid。"""
        if not self.enable_bm25 or self._bm25_index is not None:
            return
        try:
            from ..storage.vectorstore import get_vectorstore_manager

            manager = get_vectorstore_manager(self.collection_name)
            count = manager.raw_collection.count()
            if count <= 0:
                return
            payload = manager.list_documents(limit=count)
            documents = [
                Document(page_content=text or "", metadata=metadata or {})
                for text, metadata in zip(
                    payload.get("documents") or [],
                    payload.get("metadatas") or [],
                )
                if text
            ]
            self.set_documents(documents)
        except Exception as exc:
            logger.warning("[Hybrid] BM25 语料加载失败，退化为向量检索: %s", exc)

    def invalidate_bm25_index(self) -> None:
        """文档入库或删除后使内存 BM25 snapshot 失效。"""
        self._documents = []
        self._tokenized_corpus = []
        self._bm25_index = None

    def _tokenize(self, text: str) -> List[str]:
        """分词（统一使用 jieba）"""
        return [w.lower() for w in jieba.cut(text) if w.strip()]

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        user: Optional[UserContext] = None,
    ) -> List[Document]:
        """
        搜索文档（集成 ACL 权限过滤）。

        底层调用 search_with_scores()，丢弃分数后返回文档列表。
        与 search_with_scores() 共用相同的向量 + BM25 混合逻辑。

        Args:
            query: 查询文本
            k: 返回结果数
            user: 当前用户上下文（用于 ACL 过滤）

        Returns:
            用户有权限访问的文档列表
        """
        k = k or self.top_k

        results_with_scores = self.search_with_scores(query, k=k, user=user)
        return [doc for doc, _, _ in results_with_scores]

    def _build_acl_filter(self, user: Optional[UserContext]) -> Optional[Dict[str, Any]]:
        """构建 ACL filter"""
        return build_acl_filter(user=user, include_expired=False) if user else None

    def _acl_filter_results(
        self,
        results: List[Tuple[Document, float, str]],
        user: Optional[UserContext],
    ) -> List[Tuple[Document, float, str]]:
        """对混合检索结果做二次 ACL 过滤（防止 filter 绕过）"""
        if not user:
            return results
        filtered = []
        for doc, score, source in results:
            if check_doc_access(doc.metadata or {}, user):
                filtered.append((doc, score, source))
            else:
                logger.debug(
                    f"[Hybrid-ACL] 过滤无权限文档: source={doc.metadata.get('source', '?')}"
                )
        return filtered

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        user: Optional[UserContext] = None,
    ) -> List[Tuple[Document, float, str]]:
        """
        带分数的混合搜索（集成 ACL 权限过滤）。

        融合策略：
        - RRF（Reciprocal Rank Fusion）：按排名融合，不依赖分数绝对值
          RRF_score(d) = Σ weight_i / (k + rank_i(d))
          其中 k=60（融合常数），weight_i 来自配置权重
        - 优势：不受两路分数范围差异影响，BM25 和向量各凭排名贡献

        Args:
            query: 查询文本
            k: 返回结果数
            user: 当前用户上下文（用于 ACL 过滤）

        Returns:
            (文档, 融合分数, 主要来源) 元组列表，按融合分降序，
            仅包含用户有权限访问的文档
        """
        k = k or self.top_k
        candidate_k = k * 2  # 两路各取 2k，留足融合余量
        self._ensure_bm25_index()

        # 固定使用配置权重。字符长度阈值没有独立消融依据，尤其中文字符数
        # 不能稳定代表查询意图；默认 0.5/0.5 便于复现和解释。
        _ensure_jieba_dict()
        vec_w = self.vector_weight
        bm_w = self.bm25_weight

        # ── 第一路：向量检索（返回 rank，带 ACL 过滤）───────────────
        vector_ranked: Dict[str, Tuple[Document, int]] = {}  # key → (doc, rank)
        if self.enable_vector:
            try:
                final_filter = self._build_acl_filter(user)
                # Chroma 只能预过滤密级/可见性；适度 over-fetch 后在进入候选
                # 排名之前完成部门、角色和日期检查，避免无权限结果占满 top-k。
                vector_fetch_k = candidate_k * 4 if user else candidate_k
                results = self.vectorstore.similarity_search_with_score(
                    query, k=vector_fetch_k, filter=final_filter
                )
                authorized_rank = 0
                for doc, _raw_score in results:
                    if user and not check_doc_access(doc.metadata or {}, user):
                        continue
                    authorized_rank += 1
                    key = _document_identity(doc)
                    if key not in vector_ranked:
                        vector_ranked[key] = (doc, authorized_rank)
                    if authorized_rank >= candidate_k:
                        break
            except Exception as e:
                logger.warning(f"[Hybrid] 向量检索错误: {e}")

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

                scored = [
                    (all_scores[i], doc)
                    for i, doc in enumerate(self._documents)
                    if all_scores[i] > 0
                    and (not user or check_doc_access(doc.metadata or {}, user))
                ]
                scored.sort(key=lambda x: x[0], reverse=True)

                for rank, (_, doc) in enumerate(scored[:candidate_k], 1):
                    key = _document_identity(doc)
                    if key not in bm25_ranked:
                        bm25_ranked[key] = (doc, rank)
            except Exception as e:
                logger.warning(f"[Hybrid] BM25 检索错误: {e}")

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
        for key, score in fused:
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

        # ── ACL 二次过滤（防止 filter 绕过）──────────────────────
        results = self._acl_filter_results(results, user)

        return results[:k]

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
            score = vec_weight * vs + bm_weight * bs
            fused.append((key, score))
        return fused

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
    collection_name: str = "lab_knowledge",
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
    documents: List[Document] = None,
    user: Optional[UserContext] = None,
) -> List[Document]:
    """
    混合搜索的便捷函数（支持 ACL 过滤）

    Args:
        query: 查询文本
        k: 返回结果数
        documents: BM25 使用的文档集合
        user: 当前用户上下文（用于 ACL 过滤）

    Returns:
        用户有权限访问的文档列表
    """
    manager = get_hybrid_retriever_manager()

    if documents:
        manager.set_documents(documents)

    return manager.search(query, k=k, user=user)


def hybrid_search_with_scores(
    query: str,
    k: int = None,
    documents: List[Document] = None,
    user: Optional[UserContext] = None,
) -> List[Tuple[Document, float, str]]:
    """
    带分数的混合搜索（支持 ACL 过滤）

    Args:
        query: 查询文本
        k: 返回结果数
        documents: BM25 使用的文档集合
        user: 当前用户上下文（用于 ACL 过滤）

    Returns:
        (文档, 分数, 来源类型) 元组列表，仅包含用户有权限的文档
    """
    manager = get_hybrid_retriever_manager()

    if documents:
        manager.set_documents(documents)

    return manager.search_with_scores(query, k=k, user=user)
