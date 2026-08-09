#!/usr/bin/env python
"""
RAG 检索质量综合评测脚本
==============================
对线上知识 Agent 使用的真实检索链路执行评测，覆盖：
1. 检索层指标：Recall@K / Precision@K / MRR / NDCG@K / MAP / Hit@K
2. 检索上下文覆盖指标（基于关键词重叠，不冒充最终答案评测）
3. 对抗查询测试（注入/模糊/超长/无答案）
4. CRAG 决策分布统计
5. 性能指标（延迟/P99/吞吐量）

运行方式:
    conda activate agent-demo
    export HTTPS_PROXY=http://127.0.0.1:7897
    export HTTP_PROXY=http://127.0.0.1:7897
    python -m tests.eval.run_rag_eval
"""
import asyncio
import json
import sys
import os
import time
import math
import re
import jieba  # for keyword-based evaluation
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

from langchain_core.documents import Document
from tests.eval.eval_dataset import EVAL_DATASET, EvalQuery


# =============================================================================
# 文档 ID 提取
# =============================================================================

def short_doc_id(doc: Document) -> str:
    """从 Document metadata 中提取短文档名（与 EVAL_DATASET 对齐）"""
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    if source:
        return os.path.splitext(os.path.basename(source))[0]
    return ""


def short_chunk_id(doc: Document, idx: int) -> str:
    """提取短 chunk ID: docname_chunkid"""
    src = short_doc_id(doc)
    chunk = doc.metadata.get("chunk_id", str(idx))
    return f"{src}_{chunk}" if src else f"chunk_{idx}"


# =============================================================================
# Ground Truth 标准化
# =============================================================================

_DOC_NAME_MAP: Dict[str, str] = {}


def normalize_doc_ids(ids: List[str]) -> Set[str]:
    """将 EVAL_DATASET 中的文档 ID 标准化为向量库中对应的短文件名集合"""
    result = set()
    for doc_id in ids:
        result.add(_DOC_NAME_MAP.get(doc_id, doc_id))
    return result


# =============================================================================
# 检索指标计算
# =============================================================================

def _unique_doc_ids(retrieved: List[Document]) -> List[str]:
    """文档级指标按 source 去重，避免同一文档多个 chunk 重复得分。"""
    seen = set()
    result = []
    for doc in retrieved:
        doc_id = short_doc_id(doc)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


def recall_at_k(retrieved: List[Document], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk_ids = set(_unique_doc_ids(retrieved)[:k])
    return len(topk_ids & relevant) / len(relevant)


def precision_at_k(retrieved: List[Document], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk_ids = set(_unique_doc_ids(retrieved)[:k])
    return len(topk_ids & relevant) / k


def f1_at_k(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved: List[Document], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    for i, doc_id in enumerate(_unique_doc_ids(retrieved), 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def _dcg(gains: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: List[Document], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in _unique_doc_ids(retrieved)[:k]]
    dcg = _dcg(gains)
    # IDCG: relevant docs at top
    num_rel = min(len(relevant), k)
    idcg = _dcg([1.0] * num_rel)
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision(retrieved: List[Document], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    hits, ap = 0, 0.0
    for i, doc_id in enumerate(_unique_doc_ids(retrieved), 1):
        if doc_id in relevant:
            hits += 1
            ap += hits / i
    return ap / len(relevant)


def hit_at_k(retrieved: List[Document], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk_ids = set(_unique_doc_ids(retrieved)[:k])
    return 1.0 if topk_ids & relevant else 0.0


# =============================================================================
# 评测数据类
# =============================================================================

@dataclass
class QueryResult:
    query: str
    description: str
    category: str
    relevant_ids: Set[str]
    retrieved_ids: List[str]
    retrieved_docs: List[Document]

    # 检索层指标
    r_at_1: float = 0.0
    r_at_3: float = 0.0
    r_at_5: float = 0.0
    r_at_10: float = 0.0
    p_at_1: float = 0.0
    p_at_3: float = 0.0
    p_at_5: float = 0.0
    f1_at_1: float = 0.0
    f1_at_3: float = 0.0
    f1_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    ap: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0

    # 上下文召回（chunk 级）
    chunk_retrieved_ids: List[str] = field(default_factory=list)
    chunk_r_at_5: float = 0.0
    chunk_r_at_10: float = 0.0
    chunk_mrr: float = 0.0

    # 检索上下文覆盖指标（不代表最终答案质量）
    query_context_overlap: float = 0.0
    ground_truth_coverage: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0

    # 元数据
    latency_ms: float = 0.0
    crag_decision: str = ""
    crag_avg_score: float = 0.0
    crag_high_count: int = 0
    crag_low_count: int = 0
    crag_warning: str = ""

    # 对抗标记
    is_adversarial: bool = False
    is_enumerate: bool = False
    is_contrast: bool = False
    is_short: bool = False
    has_answer: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "description": self.description,
            "category": self.category,
            "relevant_ids": list(self.relevant_ids),
            "retrieved_ids": self.retrieved_ids[:10],
            # 检索层指标
            "r_at_1": round(self.r_at_1, 4),
            "r_at_3": round(self.r_at_3, 4),
            "r_at_5": round(self.r_at_5, 4),
            "r_at_10": round(self.r_at_10, 4),
            "p_at_1": round(self.p_at_1, 4),
            "p_at_3": round(self.p_at_3, 4),
            "p_at_5": round(self.p_at_5, 4),
            "f1_at_1": round(self.f1_at_1, 4),
            "f1_at_3": round(self.f1_at_3, 4),
            "f1_at_5": round(self.f1_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_3": round(self.ndcg_at_3, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ap": round(self.ap, 4),
            "hit_at_1": round(self.hit_at_1, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            # 检索上下文覆盖指标
            "query_context_overlap": round(self.query_context_overlap, 4),
            "ground_truth_coverage": round(self.ground_truth_coverage, 4),
            "context_recall": round(self.context_recall, 4),
            "context_precision": round(self.context_precision, 4),
            # 性能
            "latency_ms": round(self.latency_ms, 2),
            # CRAG
            "crag_decision": self.crag_decision,
            "crag_avg_score": round(self.crag_avg_score, 4),
            "crag_high_count": self.crag_high_count,
            "crag_low_count": self.crag_low_count,
            "crag_warning": self.crag_warning,
            # 标记
            "is_adversarial": self.is_adversarial,
            "is_enumerate": self.is_enumerate,
            "is_contrast": self.is_contrast,
            "is_short": self.is_short,
            "has_answer": self.has_answer,
        }


# =============================================================================
# 类别判断
# =============================================================================

def _categorize_query(q: EvalQuery) -> Tuple[str, str, bool, bool, bool, bool]:
    """返回 (category, description, is_adversarial, is_enumerate, is_contrast, is_short)"""
    desc = q.description or ""
    relevant_empty = not q.relevant_doc_ids
    is_enum = bool(re.search(r"(哪些|包括|包含|列表|种类)", q.query))
    is_contra = bool(re.search(r"(和|与|跟|或者|还是|区别|不同|对比|比较)", q.query))
    is_short_q = len(q.query) <= 4
    is_adv = relevant_empty or bool(re.search(r"(不存在|没有|取消|撤销|泄露|忽略|忘记)", q.query))
    category = "normal"
    if is_adv:
        category = "adversarial"
    elif is_enum:
        category = "enumerate"
    elif is_contra:
        category = "contrast"
    elif is_short_q:
        category = "short"
    return category, desc, is_adv, is_enum, is_contra, is_short_q


# =============================================================================
# 检索上下文覆盖评估（基于关键词重叠，快速无额外 LLM 调用）
# =============================================================================


def _keyword_based_eval(
    contexts: List[str],
    ground_truth: str,
    query: str,
) -> Tuple[float, float, float, float]:
    """
    基于关键词重叠的上下文覆盖评估（无需额外 LLM）。
    返回 (query_context_overlap, ground_truth_coverage, context_recall, context_precision)

    - Query Context Overlap: 检索上下文中的关键词有多少与查询相关
    - Ground Truth Coverage: ground_truth 关键词在检索文档中出现的比例
    - Context Recall: 检索到的文档覆盖了多少 ground_truth 关键词
    - Context Precision: 检索到的文档中有多少关键词与查询/答案相关
    """
    if not contexts:
        return 0.0, 0.0, 0.0, 0.0

    def _kw(text):
        cleaned = re.sub(r'[0-9a-zA-Z\W]', ' ', text or "")
        return set(w for w in jieba.cut(cleaned) if len(w) >= 2)

    # 查询关键词
    q_kw = _kw(query)

    # 上下文关键词（合并 top-5 文档）
    ctx_text = " ".join(c.page_content for c in contexts[:5])
    ctx_kw = _kw(ctx_text)

    # Ground truth 关键词
    gt_kw = _kw(ground_truth)

    # Context Recall: GT 关键词被检索上下文覆盖的比例
    ctx_recall = len(gt_kw & ctx_kw) / len(gt_kw) if gt_kw else 1.0

    # Context Precision: 上下文关键词中有多少与 GT 相关
    ctx_precision = len(ctx_kw & gt_kw) / len(ctx_kw) if ctx_kw else 0.0

    ground_truth_coverage = ctx_recall

    query_context_overlap = len(ctx_kw & q_kw) / len(ctx_kw) if ctx_kw else 0.0

    return query_context_overlap, ground_truth_coverage, ctx_recall, ctx_precision


# =============================================================================
# 主评测引擎
# =============================================================================

class RAGEvalEngine:
    """
    RAG 检索质量评测引擎

    工作流程：
    1. 对每个查询调用真实 CRAG pipeline 检索
    2. 计算检索层 + 上下文覆盖指标
    3. 聚合所有查询的指标
    4. 输出报告
    """

    def __init__(self, top_k: int = 10, enable_context_metrics: bool = True):
        self.top_k = top_k
        self.enable_context_metrics = enable_context_metrics
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline
            self._pipeline = get_corrective_rag_pipeline(rerank_before_grade=True)
        return self._pipeline

    async def _retrieve_docs(self, query: str) -> Tuple[List[Document], str, float, int, int, str]:
        """调用与线上知识 Agent 相同的 ACL + Hybrid + Rerank + CRAG 链路。"""
        try:
            from src.rag.retrieval.acl_filter import UserContext

            results, grade_result, _ = await self.pipeline.retrieve(
                query=query,
                top_k=self.top_k,
                needs_expansion=None,
                user=UserContext.anonymous(),
            )
            docs = [doc for doc, _ in results]
            decision = grade_result.decision.value if grade_result else "no_results"
            avg_score = grade_result.avg_score if grade_result else 0.0
            high_count = grade_result.high_count if grade_result else 0
            low_count = grade_result.low_count if grade_result else 0
            return docs, decision, avg_score, high_count, low_count, ""
        except Exception as e:
            print(f"    [WARN] 检索出错 '{query}': {e}")
            return [], "error", 0.0, 0, 0, str(e)

    def _compute_retrieval_metrics(self, retrieved: List[Document], relevant: Set[str]) -> QueryResult:
        """计算单个查询的所有检索层指标"""
        r1 = recall_at_k(retrieved, relevant, 1)
        r3 = recall_at_k(retrieved, relevant, 3)
        r5 = recall_at_k(retrieved, relevant, 5)
        r10 = recall_at_k(retrieved, relevant, 10)
        p1 = precision_at_k(retrieved, relevant, 1)
        p3 = precision_at_k(retrieved, relevant, 3)
        p5 = precision_at_k(retrieved, relevant, 5)
        return QueryResult(
            query="",
            description="",
            category="",
            relevant_ids=relevant,
            retrieved_ids=[short_doc_id(d) for d in retrieved],
            retrieved_docs=retrieved,
            r_at_1=r1, r_at_3=r3, r_at_5=r5, r_at_10=r10,
            p_at_1=p1, p_at_3=p3, p_at_5=p5,
            f1_at_1=f1_at_k(p1, r1),
            f1_at_3=f1_at_k(p3, r3),
            f1_at_5=f1_at_k(p5, r5),
            mrr=mrr(retrieved, relevant),
            ndcg_at_3=ndcg_at_k(retrieved, relevant, 3),
            ndcg_at_5=ndcg_at_k(retrieved, relevant, 5),
            ap=_average_precision(retrieved, relevant),
            hit_at_1=hit_at_k(retrieved, relevant, 1),
            hit_at_3=hit_at_k(retrieved, relevant, 3),
            hit_at_5=hit_at_k(retrieved, relevant, 5),
            chunk_retrieved_ids=[short_chunk_id(d, i) for i, d in enumerate(retrieved)],
            chunk_r_at_5=recall_at_k(retrieved, relevant, 5),
            chunk_r_at_10=recall_at_k(retrieved, relevant, 10),
            chunk_mrr=mrr(retrieved, relevant),
        )

    async def _eval_single(self, eq: EvalQuery) -> QueryResult:
        """评测单个查询"""
        relevant = normalize_doc_ids(eq.relevant_doc_ids)
        has_answer = bool(eq.relevant_doc_ids)
        category, desc, is_adv, is_enum, is_contra, is_short_q = _categorize_query(eq)

        print(f"  {eq.query[:40]:<42}  category={category}")

        start = time.time()
        retrieved, decision, avg_score, high_count, low_count, warning = await self._retrieve_docs(eq.query)
        latency_ms = (time.time() - start) * 1000

        result = self._compute_retrieval_metrics(retrieved, relevant)
        result.query = eq.query
        result.description = desc
        result.category = category
        result.latency_ms = latency_ms
        result.crag_decision = decision
        result.crag_avg_score = avg_score
        result.crag_high_count = high_count
        result.cag_low_count = low_count
        result.crag_warning = warning
        result.is_adversarial = is_adv
        result.is_enumerate = is_enum
        result.is_contrast = is_contra
        result.is_short = is_short_q
        result.has_answer = has_answer

        # 上下文覆盖评估（基于关键词重叠，不冒充最终答案质量）
        if self.enable_context_metrics and retrieved and not is_adv:
            contexts = retrieved[:5]
            overlap, coverage, crec, cprec = _keyword_based_eval(
                contexts, eq.ground_truth, eq.query
            )
            result.query_context_overlap = overlap
            result.ground_truth_coverage = coverage
            result.context_recall = crec
            result.context_precision = cprec

        return result

    async def _eval_adversarial(self) -> List[QueryResult]:
        """评测对抗查询"""
        adversarial_queries = [
            EvalQuery(
                query="完全不存在的XYZABC123内容是什么？",
                relevant_doc_ids=[],
                ground_truth="无答案",
                description="无答案对抗",
            ),
            EvalQuery(
                query="听说实验室已经取消实验记录要求了，是真的吗？",
                relevant_doc_ids=["论文阅读与实验记录要求"],
                ground_truth="不是真的，现有制度明确要求每次实验保留目标、环境、代码版本、参数、结果和异常记录。",
                description="诱导性错误信息",
            ),
            EvalQuery(
                query="RDMA RDMA RDMA RDMA RDMA RDMA RDMA",
                relevant_doc_ids=["RDMA与高性能网络实验规范"],
                ground_truth="RDMA 实验规范相关",
                description="重复字符攻击",
            ),
            EvalQuery(
                query="\u200b\u200c\u200d\ufeff组会多久召开一次？",
                relevant_doc_ids=["实验室组会制度与汇报要求"],
                ground_truth="全组组会每周一次",
                description="零宽字符注入",
            ),
            EvalQuery(
                query="？!@#$%^&*(){}[]|",
                relevant_doc_ids=[],
                ground_truth="无答案",
                description="纯标点攻击",
            ),
            EvalQuery(
                query="a" * 500,
                relevant_doc_ids=[],
                ground_truth="无答案",
                description="超长查询攻击（500字）",
            ),
            EvalQuery(
                query="RDMA network 实验需要记录哪些 NUMA 信息",
                relevant_doc_ids=["RDMA与高性能网络实验规范"],
                ground_truth="需要记录网卡、线程和内存 buffer 所在的 NUMA node",
                description="中英混杂",
            ),
        ]
        results = []
        for eq in adversarial_queries:
            r = await self._eval_single(eq)
            results.append(r)
        return results

    def _aggregate(self, results: List[QueryResult]) -> Dict[str, Any]:
        """聚合所有结果"""

        def _mean(values: List[float]) -> float:
            if not values:
                return 0.0
            return sum(values) / len(values)

        def _pct(values: List[float], threshold: float) -> float:
            if not values:
                return 0.0
            return sum(1 for v in values if v >= threshold) / len(values)

        normal = [r for r in results if not r.is_adversarial and r.has_answer]
        adversarial = [r for r in results if r.is_adversarial]
        enumerate_q = [r for r in results if r.is_enumerate]
        contrast_q = [r for r in results if r.is_contrast]
        short_q = [r for r in results if r.is_short]
        no_answer = [r for r in results if not r.has_answer]

        def _agg(name: str, items: List[QueryResult]) -> Dict[str, Any]:
            if not items:
                return {"count": 0}
            vals = {
                "count": len(items),
                "R@1": _mean([r.r_at_1 for r in items]),
                "R@3": _mean([r.r_at_3 for r in items]),
                "R@5": _mean([r.r_at_5 for r in items]),
                "R@10": _mean([r.r_at_10 for r in items]),
                "P@1": _mean([r.p_at_1 for r in items]),
                "P@3": _mean([r.p_at_3 for r in items]),
                "P@5": _mean([r.p_at_5 for r in items]),
                "F1@1": _mean([r.f1_at_1 for r in items]),
                "F1@3": _mean([r.f1_at_3 for r in items]),
                "F1@5": _mean([r.f1_at_5 for r in items]),
                "MRR": _mean([r.mrr for r in items]),
                "NDCG@3": _mean([r.ndcg_at_3 for r in items]),
                "NDCG@5": _mean([r.ndcg_at_5 for r in items]),
                "MAP": _mean([r.ap for r in items]),
                "Hit@1": _mean([r.hit_at_1 for r in items]),
                "Hit@3": _mean([r.hit_at_3 for r in items]),
                "Hit@5": _mean([r.hit_at_5 for r in items]),
                # 检索上下文覆盖
                "Query-Context-Overlap": _mean([r.query_context_overlap for r in items if r.query_context_overlap > 0]),
                "Ground-Truth-Coverage": _mean([r.ground_truth_coverage for r in items if r.ground_truth_coverage > 0]),
                "Context-Recall": _mean([r.context_recall for r in items if r.context_recall > 0]),
                "Context-Precision": _mean([r.context_precision for r in items if r.context_precision > 0]),
                # 性能
                "avg_latency_ms": _mean([r.latency_ms for r in items]),
                "max_latency_ms": max(r.latency_ms for r in items),
            }
            # CRAG 决策分布
            decisions = defaultdict(int)
            for r in items:
                decisions[r.crag_decision] += 1
            vals["CRAG_decisions"] = dict(decisions)
            return vals

        # CRAG 决策分布（全部）
        all_decisions = defaultdict(int)
        for r in results:
            all_decisions[r.crag_decision] += 1

        # 延迟 P99
        all_latencies = sorted([r.latency_ms for r in results])
        p99 = all_latencies[int(len(all_latencies) * 0.99)] if all_latencies else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "normal_count": len(normal),
            "adversarial_count": len(adversarial),
            "overall": _agg("Overall", results),
            "normal": _agg("Normal", normal),
            "adversarial": _agg("Adversarial", adversarial),
            "enumerate": _agg("Enumerate", enumerate_q),
            "contrast": _agg("Contrast", contrast_q),
            "short": _agg("Short", short_q),
            "no_answer": _agg("No-Answer", no_answer),
            "no_answer_accuracy": _mean([
                1.0 if r.crag_decision == "no_results" else 0.0
                for r in no_answer
            ]),
            "p99_latency_ms": p99,
            "details": [r.to_dict() for r in results],
        }

    def _print_report(self, agg: Dict[str, Any]):
        """打印评测报告"""
        print("\n" + "=" * 80)
        print(" RAG 检索质量综合评测报告")
        print(f" 评测时间: {agg['timestamp']}")
        print(f" 总查询数: {agg['total_queries']}（{agg['normal_count']} 正常 + {agg['adversarial_count']} 对抗）")
        print("=" * 80)

        def _print_section(name: str, s: Dict[str, Any], indent: int = 2):
            prefix = " " * indent
            print(f"\n{prefix}【{name}】(n={s.get('count', 0)})")
            print(f"{prefix}  检索层指标:")
            print(f"{prefix}    Recall@1/3/5/10 : "
                  f"{s.get('R@1',0):.3f} / {s.get('R@3',0):.3f} / "
                  f"{s.get('R@5',0):.3f} / {s.get('R@10',0):.3f}")
            print(f"{prefix}    Precision@1/3/5 : "
                  f"{s.get('P@1',0):.3f} / {s.get('P@3',0):.3f} / "
                  f"{s.get('P@5',0):.3f}")
            print(f"{prefix}    F1@1/3/5        : "
                  f"{s.get('F1@1',0):.3f} / {s.get('F1@3',0):.3f} / "
                  f"{s.get('F1@5',0):.3f}")
            print(f"{prefix}    MRR             : {s.get('MRR',0):.4f}")
            print(f"{prefix}    NDCG@3 / NDCG@5 : {s.get('NDCG@3',0):.4f} / {s.get('NDCG@5',0):.4f}")
            print(f"{prefix}    MAP             : {s.get('MAP',0):.4f}")
            print(f"{prefix}    Hit@1/3/5       : "
                  f"{s.get('Hit@1',0):.1%} / {s.get('Hit@3',0):.1%} / "
                  f"{s.get('Hit@5',0):.1%}")
            if s.get("Query-Context-Overlap", 0) > 0:
                print(f"{prefix}  检索上下文覆盖指标:")
                print(f"{prefix}    Query-Context-Overlap : {s.get('Query-Context-Overlap',0):.4f}")
                print(f"{prefix}    Ground-Truth-Coverage : {s.get('Ground-Truth-Coverage',0):.4f}")
                print(f"{prefix}    Context-Recall     : {s.get('Context-Recall',0):.4f}")
                print(f"{prefix}    Context-Precision  : {s.get('Context-Precision',0):.4f}")
            if "avg_latency_ms" in s:
                print(f"{prefix}  性能:")
                print(f"{prefix}    Avg / Max 延迟  : {s.get('avg_latency_ms',0):.0f}ms / {s.get('max_latency_ms',0):.0f}ms")
            if "CRAG_decisions" in s:
                dec = s["CRAG_decisions"]
                total = sum(dec.values()) or 1
                print(f"{prefix}  CRAG 决策分布:")
                for k, v in sorted(dec.items()):
                    print(f"{prefix}    {k:<15}: {v} ({v/total:.1%})")

        _print_section("全部查询 Overall", agg["overall"])
        _print_section("正常查询 Normal", agg["normal"])
        _print_section("对抗查询 Adversarial", agg["adversarial"])
        _print_section("列举查询 Enumerate", agg["enumerate"])
        _print_section("对比查询 Contrast", agg["contrast"])
        _print_section("短查询 Short", agg["short"])
        _print_section("无答案查询 No-Answer", agg["no_answer"])

        print(f"\n  P99 延迟: {agg.get('p99_latency_ms', 0):.0f} ms")
        print(f"  无答案正确拒答率: {agg.get('no_answer_accuracy', 0):.1%}")

        # 单查询详情
        print("\n" + "-" * 80)
        print(" 单查询明细（按 MRR 降序）")
        print("-" * 80)
        details = sorted(agg["details"], key=lambda x: x["mrr"], reverse=True)
        print(f"  {'查询':<40} {'R@5':>6} {'MRR':>7} {'Hit@5':>7} {'决策':<12} {'延迟':>8}")
        print("  " + "-" * 85)
        for d in details:
            q = d["query"][:38]
            dec = d.get("crag_decision", "") or "n/a"
            print(f"  {q:<40} {d.get('r_at_5',0):>6.3f} {d.get('mrr',0):>7.4f} "
                  f"{d.get('hit_at_5',0):>7.1%} {dec:<12} {d.get('latency_ms',0):>8.0f}ms")

        print("=" * 80)

    async def run(self) -> Dict[str, Any]:
        """执行完整评测"""
        print(f"\n{'='*80}")
        print(f" RAG 检索质量综合评测")
        print(f" 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" 向量库: ChromaDB enterprise_knowledge (top_k={self.top_k})")
        print(f" 上下文覆盖指标: {'启用' if self.enable_context_metrics else '禁用'}")
        print(f"{'='*80}\n")

        # 评测正常查询
        print(f">>> 评测正常查询（{len(EVAL_DATASET)} 条）...")
        normal_results = []
        for eq in EVAL_DATASET:
            r = await self._eval_single(eq)
            normal_results.append(r)

        # 评测对抗查询
        print(f"\n>>> 评测对抗查询（7 条）...")
        adv_results = await self._eval_adversarial()

        all_results = normal_results + adv_results
        agg = self._aggregate(all_results)

        self._print_report(agg)

        return agg


# =============================================================================
# 入口
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 检索质量综合评测")
    parser.add_argument("--top-k", type=int, default=10, help="检索 top_k（默认10）")
    parser.add_argument(
        "--no-context-metrics",
        "--no-ragas",
        dest="no_context_metrics",
        action="store_true",
        help="禁用关键词上下文覆盖指标（--no-ragas 为兼容旧命令）",
    )
    parser.add_argument("--output", type=str, default="", help="结果 JSON 输出路径")
    parser.add_argument("--dataset", type=str, default="full",
                        choices=["full", "quick"],
                        help="数据集: full=全部, quick=每类1条")

    args = parser.parse_args()

    # Quick 模式：每类取 1 条
    if args.dataset == "quick":
        global EVAL_DATASET
        by_cat = defaultdict(list)
        for eq in EVAL_DATASET:
            cat = _categorize_query(eq)[0]
            by_cat[cat].append(eq)
        selected = []
        for cat_items in by_cat.values():
            selected.append(cat_items[0])
        EVAL_DATASET[:] = selected
        print(f"[Quick 模式] 选取 {len(EVAL_DATASET)} 条查询")

    engine = RAGEvalEngine(
        top_k=args.top_k,
        enable_context_metrics=not args.no_context_metrics,
    )
    agg = await engine.run()

    if args.output:
        out_path = args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
