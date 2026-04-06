"""
检索评估指标模块
==================
提供检索质量的定量评估指标，包括：
- Recall@K / Precision@K / F1@K
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Hit@K
- Coverage Rate
- BM25 Score
- RRF Score

Ground Truth 格式:
    每个查询关联一个相关文档 ID 列表。
    评估时以文档 metadata["source"] 作为文档唯一标识。

Usage:
    from src.rag.evaluation.retrieval_metrics import (
        recall_at_k, precision_at_k, mrr, ndcg_at_k,
        RetrievalMetricsEngine, GROUND_TRUTH_DATASET
    )
"""

from __future__ import annotations

import math
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from langchain_core.documents import Document

# ============================================================
# Ground Truth 数据集
# ============================================================

# 格式: { query: {"relevant": List[doc_id], "category": str, "note": str} }
GROUND_TRUTH_DATASET: Dict[str, Dict[str, Any]] = {
    # ----- 核心知识库查询（员工手册相关）-----
    "公司年假政策是什么": {
        "relevant": ["员工手册.pdf"],
        "category": "normal",
        "note": "员工手册明确说明年假政策（满1年10天，满3年15天）",
    },
    "年假多少天": {
        "relevant": ["员工手册.pdf", "新政策.pdf"],
        "category": "normal",
        "note": "员工手册提供精确天数；新政策.pdf是2026年最新权威版本；旧政策.pdf为过期版本，不应作为答案",
    },
    "年假怎么算": {
        "relevant": ["员工手册.pdf"],
        "category": "normal",
        "note": "员工手册提供年假计算规则",
    },
    "公司年假": {
        "relevant": ["员工手册.pdf"],
        "category": "normal",
        "note": "直接命中员工手册",
    },
    # ----- 病假相关 -----
    "病假怎么扣": {
        "relevant": ["HR制度.pdf"],
        "category": "normal",
        "note": "HR制度.pdf 明确病假扣除规则（3天额度，80%发放）",
    },
    "病假制度": {
        "relevant": ["HR制度.pdf"],
        "category": "normal",
        "note": "HR制度.pdf 详细描述病假额度",
    },
    # ----- 请假流程 -----
    "怎么请假": {
        "relevant": ["OA操作指南.pdf", "员工手册.pdf"],
        "category": "normal",
        "note": "OA操作指南.pdf 提供完整请假流程",
    },
    "请假流程是什么": {
        "relevant": ["OA操作指南.pdf"],
        "category": "normal",
        "note": "OA操作指南.pdf 明确请假步骤",
    },
    "如何申请年假": {
        "relevant": ["员工手册.pdf", "OA操作指南.pdf"],
        "category": "normal",
        "note": "员工手册说明年假申请条件，OA指南说明流程",
    },
    "请假步骤是怎样的": {
        "relevant": ["OA操作指南.pdf"],
        "category": "normal",
        "note": "OA操作指南.pdf",
    },
    # ----- 调休相关 -----
    "调休规定": {
        "relevant": ["加班管理.pdf"],
        "category": "normal",
        "note": "加班管理.pdf 说明调休折算规则（1:1，30天内）",
    },
    "加班怎么调休": {
        "relevant": ["加班管理.pdf"],
        "category": "normal",
        "note": "加班管理.pdf 说明调休申请规则",
    },
    # ----- 福利相关 -----
    "公司有哪些福利": {
        "relevant": ["薪酬福利手册.pdf", "HR制度.pdf", "加班管理.pdf", "员工手册.pdf"],
        "category": "enumerate",
        "note": "列举类查询，多个文档各有部分答案",
    },
    "员工福利有什么": {
        "relevant": ["薪酬福利手册.pdf", "HR制度.pdf", "加班管理.pdf", "员工手册.pdf"],
        "category": "enumerate",
        "note": "多文档联合覆盖",
    },
    "福利都包括什么": {
        "relevant": ["薪酬福利手册.pdf", "HR制度.pdf"],
        "category": "enumerate",
        "note": "薪酬福利和HR制度各覆盖一部分",
    },
    # ----- 对比查询 -----
    "年假和病假的区别": {
        "relevant": ["员工手册.pdf", "HR制度.pdf"],
        "category": "contrast",
        "note": "对比类：需同时检索年假和病假两个文档",
    },
    "年假与病假的区别": {
        "relevant": ["员工手册.pdf", "HR制度.pdf"],
        "category": "contrast",
        "note": "对比类：'与'连接词",
    },
    "年假跟病假有什么不同": {
        "relevant": ["员工手册.pdf", "HR制度.pdf"],
        "category": "contrast",
        "note": "对比类：'跟'连接词",
    },
    "请假类型包含哪些": {
        "relevant": ["OA操作指南.pdf", "员工手册.pdf", "HR制度.pdf", "加班管理.pdf"],
        "category": "enumerate",
        "note": "OA指南列举了事假/病假/年假/调休四种",
    },
    # ----- 薪酬相关 -----
    "五险一金": {
        "relevant": ["薪酬福利手册.pdf"],
        "category": "normal",
        "note": "薪酬福利手册.pdf 明确说明五险一金内容",
    },
    "住房公积金缴存比例": {
        "relevant": ["薪酬福利手册.pdf"],
        "category": "normal",
        "note": "薪酬福利手册.pdf 明确12%缴存比例",
    },
    # ----- 极短查询 -----
    "年假": {
        "relevant": ["员工手册.pdf"],
        "category": "short",
        "note": "单关键词，命中员工手册",
    },
    "请假": {
        "relevant": ["OA操作指南.pdf", "员工手册.pdf"],
        "category": "short",
        "note": "单关键词，多文档相关",
    },
    # ----- 复杂/注入类 -----
    "完全不存在的XYZABC内容": {
        "relevant": [],
        "category": "adversarial_nonexistent",
        "note": "无答案查询，预期 NO_RESULTS",
    },
    "公司决定取消年假制度": {
        "relevant": [],
        "category": "adversarial_contradict",
        "note": "诱导性错误信息，正确系统不应检索到任何文档",
    },
}


# ============================================================
# 核心评估指标
# ============================================================

def _doc_id(doc: Document) -> str:
    """从 Document 中提取唯一标识（优先用 source）"""
    return doc.metadata.get("source", "") if doc.metadata else ""


def _doc_ids(results: List[Tuple[Document, float]]) -> List[str]:
    """将检索结果转换为 doc_id 列表"""
    return [_doc_id(doc) for doc, _ in results]


def _normalize_relevance(rel_str: str) -> float:
    """将相关性标签转为数值"""
    mapping = {"完全相关": 3.0, "高": 3.0, "中": 2.0, "低": 1.0, "无关": 0.0, "low": 1.0, "medium": 2.0, "high": 3.0}
    return mapping.get(rel_str, 1.0)


# ---------- Recall@K ----------

def recall_at_k(
    retrieved: List[Tuple[Document, float]],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Recall@K = |relevant_in_topk| / |all_relevant|
    当 relevant_ids 为空时，返回 1.0（无相关信息≠错误）
    """
    if not relevant_ids:
        return 1.0
    topk_ids = set(_doc_ids(retrieved[:k]))
    relevant_set = set(relevant_ids)
    return len(topk_ids & relevant_set) / len(relevant_set)


# ---------- Precision@K ----------

def precision_at_k(
    retrieved: List[Tuple[Document, float]],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Precision@K = |relevant_in_topk| / k
    """
    if k <= 0:
        return 0.0
    topk_ids = set(_doc_ids(retrieved[:k]))
    relevant_set = set(relevant_ids)
    return len(topk_ids & relevant_set) / k


# ---------- F1@K ----------

def f1_at_k(
    retrieved: List[Tuple[Document, float]],
    relevant_ids: List[str],
    k: int,
) -> float:
    """F1@K = 2 * P * R / (P + R)"""
    p = precision_at_k(retrieved, relevant_ids, k)
    r = recall_at_k(retrieved, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------- MRR (Mean Reciprocal Rank) ----------

def mrr(retrieved: List[Tuple[Document, float]], relevant_ids: List[str]) -> float:
    """
    MRR = mean(1 / first_relevant_rank)
    如果没有任何相关文档命中，返回 0
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for i, (doc, _) in enumerate(retrieved, 1):
        if _doc_id(doc) in relevant_set:
            return 1.0 / i
    return 0.0


# ---------- NDCG@K (Normalized Discounted Cumulative Gain) ----------

def _dcg_at_k(gains: List[float], k: int) -> float:
    """DCG@K = sum(g_i / log2(i+1)) for i=1..k (1-indexed)"""
    dcg = 0.0
    for i, g in enumerate(gains[:k], 1):
        dcg += g / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    retrieved: List[Tuple[Document, float]],
    relevant_ids: List[str],
    relevance_scores: Optional[List[float]] = None,
    k: int = 5,
) -> float:
    """
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved: 检索结果列表
        relevant_ids: 正确答案 doc_id 列表
        relevance_scores: 可选，每个检索文档的相关度分数列表（长度与 retrieved 一致）
                         默认为二值相关（1.0 或 0.0）
        k: 截断位置
    """
    if k <= 0:
        return 0.0

    # 构建 gains 列表
    relevant_set = set(relevant_ids)
    if relevance_scores is not None:
        gains = relevance_scores[:k]
    else:
        gains = [1.0 if _doc_id(doc) in relevant_set else 0.0 for doc, _ in retrieved[:k]]

    # IDCG：理想情况下，所有相关文档排在最前面，按相关度降序排列
    num_relevant = len(relevant_ids)
    if relevance_scores is not None:
        # graded relevance: ideal = 所有 relevance_scores 降序排列
        sorted_scores = sorted(relevance_scores, reverse=True)
        ideal_gains = sorted_scores[:k] + [0.0] * max(0, k - len(sorted_scores))
    else:
        # binary relevance: 前 num_relevant 个为 1.0
        ideal_gains = [1.0] * num_relevant + [0.0] * (k - num_relevant)
    idcg = _dcg_at_k(ideal_gains, k)

    if idcg == 0:
        return 1.0 if not relevant_ids else 0.0

    dcg = _dcg_at_k(gains, k)
    return dcg / idcg


# ---------- MAP (Mean Average Precision) ----------

def _average_precision(
    retrieved: List[Tuple[Document, float]],
    relevant_ids: List[str],
) -> float:
    """AP = sum(P@i * rel_i) / |relevant|"""
    if not relevant_ids:
        return 1.0
    relevant_set = set(relevant_ids)
    ap = 0.0
    hits = 0
    for i, (doc, _) in enumerate(retrieved, 1):
        if _doc_id(doc) in relevant_set:
            hits += 1
            ap += hits / i
    return ap / len(relevant_set)


def map_score(retrieved_list: List[List[Tuple[Document, float]]], relevant_ids_list: List[List[str]]) -> float:
    """
    MAP = mean(AP) over all queries
    用于批量评估
    """
    if not retrieved_list or len(retrieved_list) != len(relevant_ids_list):
        return 0.0
    aps = [_average_precision(r, rel) for r, rel in zip(retrieved_list, relevant_ids_list)]
    return sum(aps) / len(aps)


# ---------- Hit@K ----------

def hit_at_k(retrieved: List[Tuple[Document, float]], relevant_ids: List[str], k: int) -> float:
    """Hit@K = 1 if any relevant doc in top-k, else 0"""
    if not relevant_ids:
        return 1.0
    topk_ids = set(_doc_ids(retrieved[:k]))
    return 1.0 if topk_ids & set(relevant_ids) else 0.0


# ---------- Coverage Rate ----------

def coverage_at_k(
    retrieved: List[Tuple[Document, float]],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Coverage@K = 命中的相关文档数 / 总相关文档数（与 Recall@K 相同）
    也可理解为"需要多少篇才能覆盖全部答案"
    """
    return recall_at_k(retrieved, relevant_ids, k)


# ============================================================
# BM25 指标（使用 rank_bm25）
# ============================================================

def compute_bm25_scores(
    query: str,
    corpus: List[Document],
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    对语料库中每篇文档计算 BM25 分数并排序。

    Returns:
        按 BM25 分数降序排列的 (Document, score) 列表
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        # Fallback: 基于词重叠的简单评分
        tokens_q = _tokenize(query)
        scored = []
        for doc in corpus:
            tokens_d = _tokenize(doc.page_content)
            score = len(set(tokens_q) & set(tokens_d))
            scored.append((doc, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    tokenized_corpus = [_tokenize(doc.page_content) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = _tokenize(query)
    raw_scores = bm25.get_scores(query_tokens)

    scored = [(doc, float(raw_scores[i])) for i, doc in enumerate(corpus)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _tokenize(text: str) -> List[str]:
    """简单分词：中文按字符 n-gram，英文按空格+小写"""
    if not text:
        return []
    # 英文转小写并按空格分词
    text = text.lower()
    en_tokens = re.findall(r'[a-z0-9]+', text)
    # 简单中文字符分词（1-gram + 2-gram）
    zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
    zh_tokens = zh_chars
    zh_bigrams = [text[i:i+2] for i in range(len(text)-1) if re.match(r'[\u4e00-\u9fff]{2}', text[i:i+2])]
    return en_tokens + zh_tokens + zh_bigrams


# ============================================================
# RRF (Reciprocal Rank Fusion) 分数
# ============================================================

RRF_K = 60  # 标准 RRF 常数


def rrf_score(rank: int, k: int = RRF_K) -> float:
    """RRF 分数 = 1 / (k + rank)"""
    return 1.0 / (k + rank)


def fused_rrf_score(
    bm25_rank: int,
    vector_rank: int,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    k: int = RRF_K,
) -> float:
    """
    加权 RRF 融合分数

    Args:
        bm25_rank: BM25 检索中的排名（从1开始，0表示未命中）
        vector_rank: 向量检索中的排名（从1开始，0表示未命中）
        bm25_weight: BM25 权重
        vector_weight: 向量权重
        k: RRF 常数
    """
    rrf_b = rrf_score(bm25_rank, k) if bm25_rank > 0 else 0.0
    rrf_v = rrf_score(vector_rank, k) if vector_rank > 0 else 0.0
    return bm25_weight * rrf_b + vector_weight * rrf_v


# ============================================================
# 综合评估引擎
# ============================================================

@dataclass
class RetrievalMetricsResult:
    """单次检索评估结果"""
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]

    # 核心指标
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0

    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    f1_at_1: float = 0.0
    f1_at_3: float = 0.0
    f1_at_5: float = 0.0

    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    map_score: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0

    # 对抗类查询额外信息
    is_adversarial: bool = False
    category: str = "normal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_ids": self.retrieved_ids[:10],
            "relevant_ids": self.relevant_ids,
            "category": self.category,
            "is_adversarial": self.is_adversarial,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "precision_at_1": round(self.precision_at_1, 4),
            "precision_at_3": round(self.precision_at_3, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "f1_at_1": round(self.f1_at_1, 4),
            "f1_at_3": round(self.f1_at_3, 4),
            "f1_at_5": round(self.f1_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_3": round(self.ndcg_at_3, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "map": round(self.map_score, 4),
            "hit_at_1": round(self.hit_at_1, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_5": round(self.hit_at_5, 4),
        }


@dataclass
class AggregatedMetrics:
    """聚合评估结果"""
    stage: str = ""
    description: str = ""
    num_queries: int = 0
    num_adversarial: int = 0

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0

    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    f1_at_1: float = 0.0
    f1_at_3: float = 0.0
    f1_at_5: float = 0.0

    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    map_score: float = 0.0

    hit_rate: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0

    details: List[RetrievalMetricsResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "description": self.description,
            "num_queries": self.num_queries,
            "num_adversarial": self.num_adversarial,
            "metrics": {
                "recall_at_1": round(self.recall_at_1, 4),
                "recall_at_3": round(self.recall_at_3, 4),
                "recall_at_5": round(self.recall_at_5, 4),
                "recall_at_10": round(self.recall_at_10, 4),
                "precision_at_1": round(self.precision_at_1, 4),
                "precision_at_3": round(self.precision_at_3, 4),
                "precision_at_5": round(self.precision_at_5, 4),
                "f1_at_1": round(self.f1_at_1, 4),
                "f1_at_3": round(self.f1_at_3, 4),
                "f1_at_5": round(self.f1_at_5, 4),
                "mrr": round(self.mrr, 4),
                "ndcg_at_3": round(self.ndcg_at_3, 4),
                "ndcg_at_5": round(self.ndcg_at_5, 4),
                "map": round(self.map_score, 4),
                "hit_at_1": round(self.hit_at_1, 4),
                "hit_at_3": round(self.hit_at_3, 4),
                "hit_at_5": round(self.hit_at_5, 4),
            },
            "hit_rate": round(self.hit_rate, 4),
            "details": [d.to_dict() for d in self.details],
        }

    def __str__(self) -> str:
        return (
            f"Recall@1/3/5/10: {self.recall_at_1:.2%} / {self.recall_at_3:.2%} / "
            f"{self.recall_at_5:.2%} / {self.recall_at_10:.2%}\n"
            f"Precision@1/3/5: {self.precision_at_1:.2%} / {self.precision_at_3:.2%} / "
            f"{self.precision_at_5:.2%}\n"
            f"MRR: {self.mrr:.4f}  |  NDCG@5: {self.ndcg_at_5:.4f}  |  MAP: {self.map_score:.4f}\n"
            f"Hit@1/3/5: {self.hit_at_1:.2%} / {self.hit_at_3:.2%} / {self.hit_at_5:.2%}"
        )


class RetrievalMetricsEngine:
    """
    检索指标评估引擎

    用法:
        engine = RetrievalMetricsEngine()
        result = engine.evaluate_query(
            query="公司年假政策是什么",
            retrieved=[(doc, 0.95), ...],  # pipeline 返回格式
            top_k=10,
        )
        agg = engine.aggregate(all_results)
    """

    def __init__(self, ground_truth: Optional[Dict[str, Dict[str, Any]]] = None):
        self.ground_truth = ground_truth or GROUND_TRUTH_DATASET

    def evaluate_query(
        self,
        query: str,
        retrieved: List[Tuple[Document, float]],
        top_k: int = 10,
    ) -> RetrievalMetricsResult:
        """评估单个查询的检索质量"""
        gt = self.ground_truth.get(query, {"relevant": [], "category": "unknown"})
        relevant_ids: List[str] = gt.get("relevant", [])
        category: str = gt.get("category", "unknown")
        is_adversarial = category.startswith("adversarial")

        r = RetrievalMetricsResult(
            query=query,
            retrieved_ids=_doc_ids(retrieved),
            relevant_ids=relevant_ids,
            category=category,
            is_adversarial=is_adversarial,
        )

        # Recall@K
        r.recall_at_1 = recall_at_k(retrieved, relevant_ids, 1)
        r.recall_at_3 = recall_at_k(retrieved, relevant_ids, 3)
        r.recall_at_5 = recall_at_k(retrieved, relevant_ids, 5)
        r.recall_at_10 = recall_at_k(retrieved, relevant_ids, min(10, top_k))

        # Precision@K
        r.precision_at_1 = precision_at_k(retrieved, relevant_ids, 1)
        r.precision_at_3 = precision_at_k(retrieved, relevant_ids, 3)
        r.precision_at_5 = precision_at_k(retrieved, relevant_ids, 5)

        # F1@K
        r.f1_at_1 = f1_at_k(retrieved, relevant_ids, 1)
        r.f1_at_3 = f1_at_k(retrieved, relevant_ids, 3)
        r.f1_at_5 = f1_at_k(retrieved, relevant_ids, 5)

        # MRR / NDCG / MAP
        r.mrr = mrr(retrieved, relevant_ids)
        r.ndcg_at_3 = ndcg_at_k(retrieved, relevant_ids, k=3)
        r.ndcg_at_5 = ndcg_at_k(retrieved, relevant_ids, k=5)
        r.map_score = _average_precision(retrieved, relevant_ids)

        # Hit@K
        r.hit_at_1 = hit_at_k(retrieved, relevant_ids, 1)
        r.hit_at_3 = hit_at_k(retrieved, relevant_ids, 3)
        r.hit_at_5 = hit_at_k(retrieved, relevant_ids, 5)

        return r

    def aggregate(self, results: List[RetrievalMetricsResult]) -> AggregatedMetrics:
        """聚合多个查询的评估结果"""
        agg = AggregatedMetrics()
        agg.num_queries = len(results)
        agg.num_adversarial = sum(1 for r in results if r.is_adversarial)
        agg.details = results

        n = len(results)
        if n == 0:
            return agg

        def _mean(values: List[float]) -> float:
            return sum(values) / n

        agg.recall_at_1 = _mean([r.recall_at_1 for r in results])
        agg.recall_at_3 = _mean([r.recall_at_3 for r in results])
        agg.recall_at_5 = _mean([r.recall_at_5 for r in results])
        agg.recall_at_10 = _mean([r.recall_at_10 for r in results])

        agg.precision_at_1 = _mean([r.precision_at_1 for r in results])
        agg.precision_at_3 = _mean([r.precision_at_3 for r in results])
        agg.precision_at_5 = _mean([r.precision_at_5 for r in results])

        agg.f1_at_1 = _mean([r.f1_at_1 for r in results])
        agg.f1_at_3 = _mean([r.f1_at_3 for r in results])
        agg.f1_at_5 = _mean([r.f1_at_5 for r in results])

        agg.mrr = _mean([r.mrr for r in results])
        agg.ndcg_at_3 = _mean([r.ndcg_at_3 for r in results])
        agg.ndcg_at_5 = _mean([r.ndcg_at_5 for r in results])
        agg.map_score = _mean([r.map_score for r in results])

        agg.hit_at_1 = _mean([r.hit_at_1 for r in results])
        agg.hit_at_3 = _mean([r.hit_at_3 for r in results])
        agg.hit_at_5 = _mean([r.hit_at_5 for r in results])
        agg.hit_rate = agg.hit_at_5  # Hit Rate 通常指 Hit@5

        return agg

    def evaluate_stage(
        self,
        retrieved_map: Dict[str, List[Tuple[Document, float]]],
        stage_name: str,
        description: str = "",
    ) -> AggregatedMetrics:
        """
        评估整个阶段（e.g., "B-1 基线", "C-1 CRAG")

        Args:
            retrieved_map: {query -> [(doc, score), ...]} 检索结果
            stage_name: 阶段名称
            description: 阶段描述
        """
        results = []
        for query, retrieved in retrieved_map.items():
            r = self.evaluate_query(query, retrieved)
            results.append(r)

        agg = self.aggregate(results)
        agg.stage = stage_name
        agg.description = description
        return agg
