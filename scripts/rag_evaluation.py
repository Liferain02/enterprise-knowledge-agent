#!/usr/bin/env python
"""
RAG 完整评估体系
===================
评估分为三个层次：
1. 检索层评估（Retrieval）：评估检索质量
2. 端到端评估（E2E-RAG）：评估生成质量
3. Chunk 级细粒度评估（Chunk-level）：更真实的召回场景

运行方式:
    conda activate agent-demo
    export HTTPS_PROXY=http://127.0.0.1:7897
    export HTTP_PROXY=http://127.0.0.1:7897
    python scripts/rag_evaluation.py
"""
import asyncio
import json
import sys
import os
import time
import re
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

from langchain_core.documents import Document
from tests.eval_dataset import EVAL_DATASET, EvalQuery


# =============================================================================
# 评估指标定义
# =============================================================================

@dataclass
class RetrievalMetrics:
    """检索层评估指标"""
    # 召回率
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    # 精确率
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    # F1
    f1_at_1: float = 0.0
    f1_at_3: float = 0.0
    f1_at_5: float = 0.0
    # 排序质量
    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    map: float = 0.0
    # Hit Rate
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    # 覆盖率
    coverage_at_5: float = 0.0  # 相关文档被召回到Top5的比例

    def to_dict(self) -> Dict[str, float]:
        return {
            "R@1": self.recall_at_1, "R@3": self.recall_at_3,
            "R@5": self.recall_at_5, "R@10": self.recall_at_10,
            "P@1": self.precision_at_1, "P@3": self.precision_at_3,
            "P@5": self.precision_at_5,
            "F1@1": self.f1_at_1, "F1@3": self.f1_at_3, "F1@5": self.f1_at_5,
            "MRR": self.mrr, "NDCG@3": self.ndcg_at_3, "NDCG@5": self.ndcg_at_5,
            "MAP": self.map,
            "Hit@1": self.hit_at_1, "Hit@3": self.hit_at_3, "Hit@5": self.hit_at_5,
        }


@dataclass
class ChunkLevelMetrics:
    """Chunk 级细粒度评估指标（更真实的召回场景）"""
    # 与检索层相同的多 K 召回率
    chunk_recall_at_5: float = 0.0
    chunk_recall_at_10: float = 0.0
    chunk_recall_at_20: float = 0.0
    # Chunk 粒度的 MRR
    chunk_mrr: float = 0.0
    # 上下文完整性（检索到的 chunk 覆盖了多少 ground truth 关键词）
    context_completeness: float = 0.0
    # 冗余率（检索到的 chunk 中不相关比例）
    redundancy_rate: float = 0.0
    # 关键信息覆盖率
    key_info_coverage: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "Chunk-R@5": self.chunk_recall_at_5,
            "Chunk-R@10": self.chunk_recall_at_10,
            "Chunk-R@20": self.chunk_recall_at_20,
            "Chunk-MRR": self.chunk_mrr,
            "Context-Complete": self.context_completeness,
            "Redundancy": self.redundancy_rate,
            "KeyInfo-Coverage": self.key_info_coverage,
        }


@dataclass
class E2ERAGMetrics:
    """端到端 RAG 生成质量评估"""
    # 答案正确性
    answer_accuracy: float = 0.0
    partial_correct: float = 0.0
    incorrect: float = 0.0
    # 关键词匹配
    keyword_match_rate: float = 0.0
    avg_keyword_matches: float = 0.0
    # 幻觉检测
    hallucination_rate: float = 0.0
    # 检索召回对生成的影响
    retrieval_e2e_recall: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "Answer-Accuracy": self.answer_accuracy,
            "Partial-Correct": self.partial_correct,
            "Incorrect": self.incorrect,
            "Keyword-Match-Rate": self.keyword_match_rate,
            "Avg-Keyword-Matches": self.avg_keyword_matches,
            "Hallucination-Rate": self.hallucination_rate,
            "Retrieval-E2E-Recall": self.retrieval_e2e_recall,
        }


# =============================================================================
# 工具函数
# =============================================================================

def extract_doc_id(doc: Document, index: int) -> str:
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    if source:
        return os.path.splitext(os.path.basename(source))[0]
    return f"doc_{index}"


def extract_chunk_id(doc: Document, index: int) -> str:
    """提取 chunk ID（包含文档来源）"""
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    chunk_id = doc.metadata.get("chunk_id", f"chunk_{index}")
    if source:
        doc_name = os.path.splitext(os.path.basename(source))[0]
        return f"{doc_name}_{chunk_id}"
    return f"chunk_{index}"


# 检索层指标计算
def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len([d for d in retrieved_ids[:k] if d in set(relevant_ids)]) / k


def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in set(relevant_ids):
            return 1.0 / i
    return 0.0


def calculate_dcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in set(relevant_ids):
            dcg += 1.0 / (i ** 2 - i + 1) ** 0.5
    return dcg


def calculate_ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = calculate_dcg(retrieved_ids, relevant_ids, k)
    ideal = list(relevant_ids) + [d for d in retrieved_ids if d not in set(relevant_ids)]
    idcg = calculate_dcg(ideal, relevant_ids, k)
    return dcg / idcg if idcg else 0.0


def calculate_map(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    ap = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            num_hits += 1
            ap += num_hits / i
    return ap / len(relevant_ids) if relevant_ids else 0.0


def evaluate_retrieval(retrieved_docs: List[Document], eval_query: EvalQuery) -> Tuple[RetrievalMetrics, List[str]]:
    """计算检索层所有指标"""
    retrieved_ids = [extract_doc_id(doc, i) for i, doc in enumerate(retrieved_docs)]
    metrics = RetrievalMetrics()

    for k in [1, 3, 5, 10]:
        recall = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
        precision = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
        if k <= 5:
            metrics.recall_at_1 if k == 1 else None
            setattr(metrics, f"recall_at_{k}", recall)
            setattr(metrics, f"precision_at_{k}", precision)
            setattr(metrics, f"f1_at_{k}", calculate_f1(precision, recall))
            setattr(metrics, f"hit_at_{k}", 1.0 if recall > 0 else 0.0)
        elif k == 10:
            setattr(metrics, f"recall_at_{k}", recall)

    metrics.mrr = calculate_mrr(retrieved_ids, eval_query.relevant_doc_ids)
    metrics.ndcg_at_3 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, 3)
    metrics.ndcg_at_5 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, 5)
    metrics.map = calculate_map(retrieved_ids, eval_query.relevant_doc_ids)
    metrics.hit_at_1 = 1.0 if metrics.recall_at_1 > 0 else 0.0
    metrics.hit_at_3 = 1.0 if metrics.recall_at_3 > 0 else 0.0
    metrics.hit_at_5 = 1.0 if metrics.recall_at_5 > 0 else 0.0

    return metrics, retrieved_ids


def evaluate_chunk_level(retrieved_docs: List[Document], eval_query: EvalQuery,
                         all_chunks: Dict[str, List[str]]) -> ChunkLevelMetrics:
    """
    Chunk 级细粒度评估
    - 每个文档可能有多个 chunk，相关文档的所有 chunk 都应该被召回到
    - 更真实的召回场景：知识库中有大量 chunk，需要更精细的分块策略
    """
    metrics = ChunkLevelMetrics()

    # 获取该查询相关文档的所有 chunk
    relevant_chunks = set()
    for doc_id in eval_query.relevant_doc_ids:
        if doc_id in all_chunks:
            relevant_chunks.update(all_chunks[doc_id])

    # 如果没有 chunk 级别数据，使用文档级别
    if not relevant_chunks:
        retrieved_ids = [extract_doc_id(doc, i) for i, doc in enumerate(retrieved_docs)]
        metrics.chunk_recall_at_5 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, 5)
        metrics.chunk_recall_at_10 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, 10)
        metrics.chunk_recall_at_20 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, 20)
        metrics.chunk_mrr = calculate_mrr(retrieved_ids, eval_query.relevant_doc_ids)
        return metrics

    # Chunk 级别召回率
    retrieved_chunk_ids = [extract_chunk_id(doc, i) for i, doc in enumerate(retrieved_docs)]
    metrics.chunk_recall_at_5 = calculate_recall(retrieved_chunk_ids, list(relevant_chunks), 5)
    metrics.chunk_recall_at_10 = calculate_recall(retrieved_chunk_ids, list(relevant_chunks), 10)
    metrics.chunk_recall_at_20 = calculate_recall(retrieved_chunk_ids, list(relevant_chunks), 20)
    metrics.chunk_mrr = calculate_mrr(retrieved_chunk_ids, list(relevant_chunks))

    # 上下文完整性：检索到的 chunk 覆盖了多少关键词
    gt_keywords = extract_keywords(eval_query.ground_truth)
    retrieved_texts = " ".join([doc.page_content for doc in retrieved_docs[:10]])
    matched_kw = sum(1 for kw in gt_keywords if kw in retrieved_texts)
    metrics.context_completeness = matched_kw / len(gt_keywords) if gt_keywords else 0.0

    # 冗余率：Top5 中不相关 chunk 的比例
    relevant_sources = set(eval_query.relevant_doc_ids)
    irrelevant_count = sum(1 for doc in retrieved_docs[:5]
                           if extract_doc_id(doc, 0) not in relevant_sources)
    metrics.redundancy_rate = irrelevant_count / 5.0

    # 关键信息覆盖率
    key_infos = extract_key_info(eval_query.ground_truth)
    matched_info = sum(1 for info in key_infos if info in retrieved_texts)
    metrics.key_info_coverage = matched_info / len(key_infos) if key_infos else 0.0

    return metrics


def extract_keywords(text: str) -> List[str]:
    """从 ground truth 提取关键词"""
    # 去除标点，提取中文词组和英文词
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
    words = text.split()
    # 过滤掉单字
    keywords = [w for w in words if len(w) >= 2]
    return keywords


def extract_key_info(text: str) -> List[str]:
    """从 ground truth 提取关键信息片段（数字+单位、专有名词等）"""
    patterns = [
        r'\d+[年月日天小时分秒个%]+\w*',  # 数字+单位
        r'[\u4e00-\u9fff]+\w*[\u4e00-\u9fff]+',  # 连续中文词
        r'\w+[\u4e00-\u9fff]+',  # 英文+中文
    ]
    info = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        info.extend(matches)
    return list(set(info))


def evaluate_e2e(answer: str, eval_query: EvalQuery) -> Tuple[E2ERAGMetrics, Dict]:
    """评估端到端 RAG 生成质量"""
    metrics = E2ERAGMetrics()

    # 关键词匹配
    keywords = extract_keywords(eval_query.ground_truth)
    keyword_matches = sum(1 for kw in keywords if kw in answer)
    metrics.avg_keyword_matches = keyword_matches / len(keywords) if keywords else 0.0
    metrics.keyword_match_rate = keyword_matches / len(keywords) if keywords else 0.0

    # 答案正确性判断
    gt_main = eval_query.ground_truth.split("：")[0][:15]
    matched_main = gt_main in answer
    matched_keywords = keyword_matches >= max(2, len(keywords) * 0.3)

    if matched_main or matched_keywords:
        metrics.answer_accuracy = 1.0
        metrics.partial_correct = 0.0
        metrics.incorrect = 0.0
    elif keyword_matches >= 1:
        metrics.answer_accuracy = 0.0
        metrics.partial_correct = 1.0
        metrics.incorrect = 0.0
    else:
        metrics.answer_accuracy = 0.0
        metrics.partial_correct = 0.0
        metrics.incorrect = 1.0

    # 幻觉检测：答案中是否包含不确定的表达
    uncertain_phrases = ["不确定", "不知道", "无法确定", "可能", "也许", "应该"]
    metrics.hallucination_rate = 1.0 if any(p in answer for p in uncertain_phrases) else 0.0

    return metrics, {"matched_main": matched_main, "keyword_matches": keyword_matches}


def average_retrieval_metrics(metrics_list: List[RetrievalMetrics]) -> RetrievalMetrics:
    if not metrics_list:
        return RetrievalMetrics()
    avg = RetrievalMetrics()
    n = len(metrics_list)
    for attr in ['recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
                 'precision_at_1', 'precision_at_3', 'precision_at_5',
                 'f1_at_1', 'f1_at_3', 'f1_at_5',
                 'mrr', 'ndcg_at_3', 'ndcg_at_5', 'map',
                 'hit_at_1', 'hit_at_3', 'hit_at_5']:
        setattr(avg, attr, sum(getattr(m, attr) for m in metrics_list) / n)
    return avg


def average_chunk_metrics(metrics_list: List[ChunkLevelMetrics]) -> ChunkLevelMetrics:
    if not metrics_list:
        return ChunkLevelMetrics()
    avg = ChunkLevelMetrics()
    n = len(metrics_list)
    for attr in ['chunk_recall_at_5', 'chunk_recall_at_10', 'chunk_recall_at_20',
                 'chunk_mrr', 'context_completeness', 'redundancy_rate', 'key_info_coverage']:
        setattr(avg, attr, sum(getattr(m, attr) for m in metrics_list) / n)
    return avg


def average_e2e_metrics(metrics_list: List[E2ERAGMetrics]) -> E2ERAGMetrics:
    if not metrics_list:
        return E2ERAGMetrics()
    avg = E2ERAGMetrics()
    n = len(metrics_list)
    for attr in ['answer_accuracy', 'partial_correct', 'incorrect',
                 'keyword_match_rate', 'avg_keyword_matches',
                 'hallucination_rate', 'retrieval_e2e_recall']:
        setattr(avg, attr, sum(getattr(m, attr) for m in metrics_list) / n)
    return avg


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


# =============================================================================
# 评估 Stage 1: 检索层评估（完整指标）
# =============================================================================

async def eval_retrieval(test_cases: List[EvalQuery], strategy_name: str,
                          search_func) -> Dict[str, Any]:
    """检索层评估 - 计算完整指标"""
    print(f"\n{'='*70}")
    print(f"【检索层评估】{strategy_name}")
    print(f"{'='*70}")

    all_metrics = []
    details = []
    total_latency = 0.0

    for eq in test_cases:
        start = time.time()
        docs = search_func(eq.query)
        latency = (time.time() - start) * 1000
        total_latency += latency

        metrics, ret_ids = evaluate_retrieval(docs, eq)
        all_metrics.append(metrics)

        details.append({
            "query": eq.query,
            "relevant": eq.relevant_doc_ids,
            "retrieved_doc_ids": ret_ids[:10],
            "metrics": asdict(metrics),
            "latency_ms": round(latency, 1),
        })

        # 打印关键指标
        status = "+" if metrics.hit_at_5 > 0 else "-"
        print(f"  {status} {eq.query[:40]:<42} "
              f"R@5={metrics.recall_at_5:.2f} P@5={metrics.precision_at_5:.2f} "
              f"F1@5={metrics.f1_at_5:.2f} MRR={metrics.mrr:.2f} MAP={metrics.map:.2f} "
              f"H@5={metrics.hit_at_5:.0f} {latency:.0f}ms")

    avg = average_retrieval_metrics(all_metrics)
    hit_rate = sum(1 for m in all_metrics if m.hit_at_5 > 0) / len(all_metrics)

    result = {
        "stage": strategy_name,
        "description": f"检索层评估 - {strategy_name}",
        "metrics": asdict(avg),
        "hit_rate": hit_rate,
        "details": details,
        "total_queries": len(test_cases),
        "avg_latency_ms": total_latency / len(test_cases),
    }

    print(f"\n  汇总:")
    print(f"    Hit@5={hit_rate:.1%} | 平均延迟: {result['avg_latency_ms']:.0f}ms")
    print(f"    完整指标:")
    print(f"      R@1={avg.recall_at_1:.3f} R@3={avg.recall_at_3:.3f} R@5={avg.recall_at_5:.3f} R@10={avg.recall_at_10:.3f}")
    print(f"      P@1={avg.precision_at_1:.3f} P@3={avg.precision_at_3:.3f} P@5={avg.precision_at_5:.3f}")
    print(f"      F1@1={avg.f1_at_1:.3f} F1@3={avg.f1_at_3:.3f} F1@5={avg.f1_at_5:.3f}")
    print(f"      MRR={avg.mrr:.3f} NDCG@3={avg.ndcg_at_3:.3f} NDCG@5={avg.ndcg_at_5:.3f} MAP={avg.map:.3f}")

    return result


# =============================================================================
# 评估 Stage 2: Chunk 级细粒度评估
# =============================================================================

async def eval_chunk_level(test_cases: List[EvalQuery],
                            hybrid_search_func) -> Dict[str, Any]:
    """Chunk 级细粒度评估"""
    print(f"\n{'='*70}")
    print(f"【Chunk 级细粒度评估】")
    print(f"{'='*70}")

    # 构建 chunk 到文档的映射（简化版本：按 chunk 内容去重）
    from src.rag.storage.vectorstore import get_vectorstore
    vs = get_vectorstore("enterprise_knowledge")
    all_results = vs.get(limit=9999)
    all_docs = all_results.get("documents") or []
    all_metadatas = all_results.get("metadatas") or []

    # 按文档分组 chunks
    doc_chunks: Dict[str, List[str]] = defaultdict(list)
    for i, (doc, meta) in enumerate(zip(all_docs, all_metadatas)):
        doc_obj = Document(page_content=doc, metadata=meta or {})
        chunk_id = extract_chunk_id(doc_obj, i)
        doc_chunks[os.path.splitext(os.path.basename(meta.get("source", "") if meta else ""))[0]].append(chunk_id)

    all_metrics = []
    details = []
    total_latency = 0.0

    for eq in test_cases:
        start = time.time()
        docs = hybrid_search_func(eq.query)
        latency = (time.time() - start) * 1000
        total_latency += latency

        metrics = evaluate_chunk_level(docs, eq, doc_chunks)
        all_metrics.append(metrics)

        details.append({
            "query": eq.query,
            "relevant_chunks": list({
                chunk for doc in eq.relevant_doc_ids
                for chunk in doc_chunks.get(doc, [])
            }),
            "metrics": asdict(metrics),
            "latency_ms": round(latency, 1),
        })

        print(f"  {'+' if metrics.chunk_recall_at_5 > 0 else '-'} {eq.query[:40]:<42} "
              f"Chunk-R@5={metrics.chunk_recall_at_5:.2f} "
              f"Context={metrics.context_completeness:.2f} "
              f"Redundancy={metrics.redundancy_rate:.2f} {latency:.0f}ms")

    avg = average_chunk_metrics(all_metrics)

    result = {
        "stage": "Chunk-Level",
        "description": "Chunk 级细粒度评估",
        "metrics": asdict(avg),
        "details": details,
        "total_queries": len(test_cases),
        "avg_latency_ms": total_latency / len(test_cases),
    }

    print(f"\n  汇总:")
    print(f"    Chunk-R@5={avg.chunk_recall_at_5:.3f} Chunk-R@10={avg.chunk_recall_at_10:.3f} "
          f"Chunk-R@20={avg.chunk_recall_at_20:.3f}")
    print(f"    Chunk-MRR={avg.chunk_mrr:.3f}")
    print(f"    Context-Completeness={avg.context_completeness:.3f} "
          f"Redundancy-Rate={avg.redundancy_rate:.3f}")
    print(f"    KeyInfo-Coverage={avg.key_info_coverage:.3f}")

    return result


# =============================================================================
# 评估 Stage 3: 端到端 RAG 评估
# =============================================================================

async def eval_e2e_rag(test_cases: List[EvalQuery],
                       max_cases: int = 10) -> Dict[str, Any]:
    """端到端 RAG 生成质量评估"""
    print(f"\n{'='*70}")
    print(f"【端到端 RAG 评估】（限制 {max_cases} 条，控制 token 消耗）")
    print(f"{'='*70}")

    from src.agent.graph import arun_agent

    all_metrics = []
    details = []
    total_latency = 0.0
    errors = 0

    for eq in test_cases[:max_cases]:
        start = time.time()
        try:
            result = await arun_agent(eq.query, session_id=f"e2e-test-{hash(eq.query)}")
            answer = result.get("final_answer", "")
            latency = (time.time() - start) * 1000
            total_latency += latency

            metrics, extra = evaluate_e2e(answer, eq)
            all_metrics.append(metrics)

            details.append({
                "query": eq.query,
                "ground_truth": eq.ground_truth,
                "answer": answer[:500] + "..." if len(answer) > 500 else answer,
                "metrics": asdict(metrics),
                "extra": extra,
                "latency_ms": round(latency, 1),
            })

            verdict = "correct" if metrics.answer_accuracy > 0 else "partial" if metrics.partial_correct > 0 else "wrong"
            print(f"  [{verdict.upper():<8}] {eq.query[:35]:<37} "
                  f"KW-match={extra['keyword_matches']} {latency:.0f}ms")
        except Exception as e:
            errors += 1
            total_latency += time.time() - start
            print(f"  [ERROR] {eq.query[:40]:<42} - {e}")
            details.append({
                "query": eq.query,
                "error": str(e),
                "latency_ms": 0,
            })

    avg = average_e2e_metrics(all_metrics)

    result = {
        "stage": "E2E-RAG",
        "description": "端到端 RAG 生成质量评估",
        "metrics": asdict(avg),
        "details": details,
        "total_queries": len(test_cases[:max_cases]),
        "errors": errors,
        "avg_latency_ms": total_latency / len(test_cases[:max_cases]) if test_cases[:max_cases] else 0,
    }

    print(f"\n  汇总:")
    print(f"    答案正确率: {avg.answer_accuracy:.1%} | 部分正确: {avg.partial_correct:.1%} | 错误: {avg.incorrect:.1%}")
    print(f"    关键词匹配率: {avg.keyword_match_rate:.1%} | 平均匹配数: {avg.avg_keyword_matches:.1f}")
    print(f"    幻觉率: {avg.hallucination_rate:.1%} | 异常: {errors}")

    return result


# =============================================================================
# 主函数
# =============================================================================

async def run_full_evaluation():
    start_time = time.time()

    print(f"\n{'#'*70}")
    print(f"# RAG 完整评估体系")
    print(f"# 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 数据集: {len(EVAL_DATASET)} 条查询")
    print(f"#{'#'*70}")

    all_results = {}

    # ========== Stage 1: 检索层评估 ==========
    from src.rag.retrieval.retriever import get_retriever_manager
    from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
    from src.rag.storage.vectorstore import get_vectorstore

    retriever = get_retriever_manager()

    # B-1: 基线向量检索
    all_results["B1_baseline"] = await eval_retrieval(
        EVAL_DATASET, "B-1 基线向量检索",
        lambda q: retriever.search(q, k=10)
    )

    # B-2: 混合检索
    hybrid = get_hybrid_retriever_manager(
        collection_name="enterprise_knowledge",
        top_k=10,
        vector_weight=0.5,
        bm25_weight=0.5,
    )
    vs = get_vectorstore("enterprise_knowledge")
    all_results_raw = vs.get(limit=9999)
    all_docs = all_results_raw.get("documents") or []
    all_metadatas = all_results_raw.get("metadatas") or []
    docs_for_bm25 = [Document(page_content=c, metadata=m or {}) for c, m in zip(all_docs, all_metadatas)]
    hybrid.set_documents(docs_for_bm25)

    all_results["B2_hybrid"] = await eval_retrieval(
        EVAL_DATASET, "B-2 BM25+向量混合检索",
        lambda q: hybrid.search(q, k=10)
    )

    # ========== Stage 2: Chunk 级细粒度评估 ==========
    all_results["ChunkLevel"] = await eval_chunk_level(
        EVAL_DATASET, lambda q: hybrid.search(q, k=20)
    )

    # ========== Stage 3: 端到端 RAG ==========
    all_results["E2E_RAG"] = await eval_e2e_rag(EVAL_DATASET, max_cases=10)

    # ========== 汇总报告 ==========
    total_time = time.time() - start_time

    print(f"\n\n{'#'*70}")
    print(f"# 完整评估汇总")
    print(f"# 总耗时: {total_time:.0f}秒")
    print(f"#{'#'*70}")

    # 检索层对比表
    print(f"\n【检索层评估汇总】")
    print(f"{'阶段':<30} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'P@5':>6} {'F1@5':>6} {'MRR':>6} {'MAP':>6} {'NDCG@5':>7} {'Hit@5':>7}")
    print("-" * 105)
    for key in ["B1_baseline", "B2_hybrid"]:
        if key in all_results and "metrics" in all_results[key]:
            m = all_results[key]["metrics"]
            label = key.replace("B1_baseline", "B-1 基线向量").replace("B2_hybrid", "B-2 混合检索")
            print(f"{label:<30} {m['recall_at_1']:>6.3f} {m['recall_at_5']:>6.3f} "
                  f"{m['recall_at_10']:>6.3f} {m['precision_at_5']:>6.3f} {m['f1_at_5']:>6.3f} "
                  f"{m['mrr']:>6.3f} {m['map']:>6.3f} {m['ndcg_at_5']:>7.3f} {m['hit_at_5']:>7.1%}")

    # Chunk 级对比
    if "ChunkLevel" in all_results:
        cl = all_results["ChunkLevel"]["metrics"]
        print(f"\n【Chunk 级细粒度评估】")
        print(f"  Chunk-R@5={cl['chunk_recall_at_5']:.3f} Chunk-R@10={cl['chunk_recall_at_10']:.3f} "
              f"Chunk-R@20={cl['chunk_recall_at_20']:.3f}")
        print(f"  Chunk-MRR={cl['chunk_mrr']:.3f} Context-Completeness={cl['context_completeness']:.3f} "
              f"Redundancy={cl['redundancy_rate']:.3f}")

    # E2E RAG
    if "E2E_RAG" in all_results:
        e2e = all_results["E2E_RAG"]["metrics"]
        print(f"\n【端到端 RAG 评估】")
        print(f"  答案正确率={e2e['answer_accuracy']:.1%} 部分正确={e2e['partial_correct']:.1%} "
              f"错误={e2e['incorrect']:.1%}")
        print(f"  关键词匹配率={e2e['keyword_match_rate']:.1%} 平均匹配={e2e['avg_keyword_matches']:.1f}")
        print(f"  幻觉率={e2e['hallucination_rate']:.1%}")

    # ========== 保存结果 ==========
    output_path = os.path.join(os.path.dirname(__file__), "rag_full_evaluation_results.json")
    serializable_results = make_serializable(all_results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"\n  结果已保存: {output_path}")
    print(f"\n{'#'*70}")
    print(f"# 评估完成，总耗时 {total_time:.0f}秒")
    print(f"{'#'*70}")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_full_evaluation())
