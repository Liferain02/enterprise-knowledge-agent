#!/usr/bin/env python
"""
RAG 检索全流程分级优化测试
==============================
Stage 1: 基线向量检索（B-1, 30条, 无LLM）
Stage 2: BM25+向量混合检索（B-2, 30条, 无LLM）
Stage 3: 分块策略测试（C, 无LLM）
Stage 4: 精排过滤率（B-6, 30条, 仅embedding无LLM）
Stage 5: 多阶段+Rerank（B-3, 10条, 精简LLM）
Stage 6: CRAG正确性反馈（B-4, 10条, 精简LLM）
Stage 7: Query Expansion（B-5, 复杂查询子集, 精简LLM）

运行方式:
    conda activate agent-demo
    export HTTPS_PROXY=http://127.0.0.1:7897
    export HTTP_PROXY=http://127.0.0.1:7897
    python scripts/run_rag_benchmark.py
==============================
"""
import asyncio
import json
import sys
import os
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 保留代理（用于嵌入查询和LLM API调用）
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

from langchain_core.documents import Document
from tests.eval_dataset import EVAL_DATASET, EvalQuery


# =============================================================================
# 评估指标
# =============================================================================

@dataclass
class RetrievalMetrics:
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "Recall@1": self.recall_at_1,
            "Recall@3": self.recall_at_3,
            "Recall@5": self.recall_at_5,
            "Precision@1": self.precision_at_1,
            "Precision@3": self.precision_at_3,
            "Precision@5": self.precision_at_5,
            "MRR": self.mrr,
            "NDCG@3": self.ndcg_at_3,
            "NDCG@5": self.ndcg_at_5,
        }


def extract_doc_id(doc: Document, index: int) -> str:
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    if source:
        return os.path.splitext(os.path.basename(source))[0]
    return f"doc_{index}"


def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len([d for d in retrieved_ids[:k] if d in set(relevant_ids)]) / k


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


def evaluate_retrieval(retrieved_docs: List[Document], eval_query: EvalQuery) -> Tuple[RetrievalMetrics, List[str]]:
    retrieved_ids = [extract_doc_id(doc, i) for i, doc in enumerate(retrieved_docs)]
    metrics = RetrievalMetrics()
    for k in [1, 3, 5]:
        setattr(metrics, f"recall_at_{k}", calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k))
        setattr(metrics, f"precision_at_{k}", calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k))
    metrics.mrr = calculate_mrr(retrieved_ids, eval_query.relevant_doc_ids)
    metrics.ndcg_at_3 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, 3)
    metrics.ndcg_at_5 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, 5)
    return metrics, retrieved_ids


def average_metrics(metrics_list: List[RetrievalMetrics]) -> RetrievalMetrics:
    if not metrics_list:
        return RetrievalMetrics()
    avg = RetrievalMetrics()
    n = len(metrics_list)
    for attr in ['recall_at_1', 'recall_at_3', 'recall_at_5',
                 'precision_at_1', 'precision_at_3', 'precision_at_5',
                 'mrr', 'ndcg_at_3', 'ndcg_at_5']:
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
# Stage 1: B-1 基线向量检索
# =============================================================================

async def stage1_baseline_vector(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("【Stage 1】基线 - 纯向量检索（仅 embedding，无 LLM）")
    print(f"{'='*70}")

    from src.rag.retrieval.retriever import get_retriever_manager
    retriever = get_retriever_manager()

    all_metrics = []
    details = []
    total_latency = 0.0

    for eq in test_cases:
        start = time.time()
        docs = retriever.search(eq.query, k=10)
        latency = (time.time() - start) * 1000
        total_latency += latency

        metrics, ret_ids = evaluate_retrieval(docs, eq)
        all_metrics.append(metrics)

        is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])
        details.append({
            "query": eq.query,
            "relevant": eq.relevant_doc_ids,
            "retrieved_top5": ret_ids[:5],
            "hit": is_hit,
            "recall@5": metrics.recall_at_5,
            "mrr": metrics.mrr,
            "latency_ms": round(latency, 1),
        })
        print(f"  {'+' if is_hit else '-'} {eq.query[:44]:<46} R@5={metrics.recall_at_5:.2f} MRR={metrics.mrr:.2f} {latency:.0f}ms")

    avg = average_metrics(all_metrics)
    result = {
        "stage": "B-1",
        "description": "基线向量检索",
        "metrics": asdict(avg),
        "details": details,
        "total_queries": len(test_cases),
        "hit_count": sum(1 for d in details if d["hit"]),
        "avg_latency_ms": total_latency / len(test_cases),
    }
    hit_rate = result["hit_count"] / len(test_cases)
    print(f"\n  汇总: Hit@5 = {result['hit_count']}/{len(test_cases)} ({hit_rate:.1%}) | "
          f"平均延迟: {result['avg_latency_ms']:.0f}ms")
    return result


# =============================================================================
# Stage 2: B-2 混合检索
# =============================================================================

async def stage2_hybrid_retrieval(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("【Stage 2】BM25 + 向量混合检索（无 LLM）")
    print(f"{'='*70}")

    from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
    from src.rag.storage.vectorstore import get_vectorstore

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

    all_metrics = []
    details = []
    total_latency = 0.0

    for eq in test_cases:
        start = time.time()
        docs = hybrid.search(eq.query, k=10)
        latency = (time.time() - start) * 1000
        total_latency += latency

        metrics, ret_ids = evaluate_retrieval(docs, eq)
        all_metrics.append(metrics)

        is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])
        details.append({
            "query": eq.query,
            "relevant": eq.relevant_doc_ids,
            "retrieved_top5": ret_ids[:5],
            "hit": is_hit,
            "recall@5": metrics.recall_at_5,
            "mrr": metrics.mrr,
            "latency_ms": round(latency, 1),
        })
        print(f"  {'+' if is_hit else '-'} {eq.query[:44]:<46} R@5={metrics.recall_at_5:.2f} MRR={metrics.mrr:.2f} {latency:.0f}ms")

    avg = average_metrics(all_metrics)
    result = {
        "stage": "B-2",
        "description": "BM25+向量混合检索",
        "metrics": asdict(avg),
        "details": details,
        "total_queries": len(test_cases),
        "hit_count": sum(1 for d in details if d["hit"]),
        "avg_latency_ms": total_latency / len(test_cases),
    }
    hit_rate = result["hit_count"] / len(test_cases)
    print(f"\n  汇总: Hit@5 = {result['hit_count']}/{len(test_cases)} ({hit_rate:.1%}) | "
          f"平均延迟: {result['avg_latency_ms']:.0f}ms")
    return result


# =============================================================================
# Stage 3: C 分块策略
# =============================================================================

def stage3_chunking() -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("【Stage 3】文档分块策略测试（无 LLM）")
    print(f"{'='*70}")

    from src.rag.processing.document_loader import (
        get_document_loader_manager, estimate_tokens, split_sentences)
    from src.rag.processing.chunker import SemanticChunker, HybridChunker

    loader_manager = get_document_loader_manager()
    test_md = "/share/home/lifr/workspace/code/enterprise-knowledge-agent/data/knowledge/员工手册.md"
    docs = loader_manager.load_file(test_md)

    result = {"stage": "C", "description": "分块策略测试", "strategies": {}}

    for strategy in ["recursive", "semantic", "hybrid", "markdown"]:
        print(f"\n  -- 分块策略: {strategy} --")
        try:
            chunks = loader_manager.split_documents(docs, splitter_type=strategy)
            tokens_list = [estimate_tokens(c.page_content) for c in chunks]
            avg_t = sum(tokens_list) / len(tokens_list)
            min_t, max_t = min(tokens_list), max(tokens_list)
            over_limit = sum(1 for t in tokens_list if t > 800)
            over_ratio = over_limit / len(tokens_list)

            print(f"    分块数量: {len(chunks)}")
            print(f"    Token范围: {min_t:.0f}~{max_t:.0f}, 平均: {avg_t:.0f}")
            print(f"    超过800token的块: {over_limit}/{len(chunks)} ({over_ratio:.1%})")

            result["strategies"][strategy] = {
                "chunk_count": len(chunks),
                "avg_tokens": avg_t,
                "min_tokens": min_t,
                "max_tokens": max_t,
                "over_limit_count": over_limit,
                "over_limit_ratio": over_ratio,
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            result["strategies"][strategy] = {"error": str(e)}

    # 句子分割验证
    print(f"\n  -- 句子分割验证 --")
    sentence_tests = [
        ("省略号", "用户手册……请仔细阅读以下内容。", 1),
        ("分号列表", "第一步：提交申请；第二步：等待审批；第三步：领取结果。", 3),
        ("圆圈序号", "① 第一条。② 第二条。③ 第三条。", 3),
        ("数字列表", "1. 第一项。2. 第二项。3. 第三项。", 3),
    ]
    sentence_ok = 0
    for name, text, expected in sentence_tests:
        sents = split_sentences(text)
        ok = len(sents) == expected
        sentence_ok += ok
        print(f"    {'PASS' if ok else 'FAIL'} {name}: 期望{expected}句, 实际{len(sents)}句")
    result["sentence_split"] = {
        "total": len(sentence_tests),
        "passed": sentence_ok,
        "rate": sentence_ok / len(sentence_tests),
    }

    # Token 估算精度
    print(f"\n  -- Token 估算精度 --")
    for text in ["这是一个中文句子。", "This is an English sentence. " * 5,
                 "公司年假政策：工作满1年可休5天。报销流程需要发票和出差报告。"]:
        tokens = estimate_tokens(text)
        ratio = len(text) / max(tokens, 1)
        print(f"    {len(text)}字符 -> {tokens}token, 比例~{ratio:.1f}字符/token")

    return result


# =============================================================================
# Stage 4: B-6 精排过滤率（向量+BM25候选，仅统计，无LLM rerank）
# =============================================================================

async def stage4_candidate_filter(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("【Stage 4】候选文档过滤率分析（向量+BM25候选分布，无LLM）")
    print(f"{'='*70}")

    from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
    from src.rag.storage.vectorstore import get_vectorstore

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

    source_counts = {"vector": 0, "bm25": 0, "vector+bm25": 0}
    hit_at_5_counts = []
    details = []

    for eq in test_cases:
        # Retrieve large k to analyze source distribution
        docs_k10 = hybrid.search(eq.query, k=10)
        docs_k20 = hybrid.search(eq.query, k=20)

        # Count source distribution in top-10
        source_dist = {"vector": 0, "bm25": 0, "vector+bm25": 0}
        for doc in docs_k10:
            tag = doc.metadata.get("source_tag", "vector")
            source_dist[tag] = source_dist.get(tag, 0) + 1

        for tag, cnt in source_dist.items():
            source_counts[tag] += cnt

        is_hit_5 = any(extract_doc_id(d, i) in eq.relevant_doc_ids for i, d in enumerate(docs_k10[:5]))
        is_hit_10 = any(extract_doc_id(d, i) in eq.relevant_doc_ids for i, d in enumerate(docs_k10[:10]))
        hit_at_5_counts.append(1 if is_hit_5 else 0)

        # Vector similarity distribution
        vec_scores = [doc.metadata.get("vector_score", 0) for doc in docs_k10]
        bm25_scores = [doc.metadata.get("bm25_score", 0) for doc in docs_k10]

        details.append({
            "query": eq.query,
            "hit@5": is_hit_5,
            "hit@10": is_hit_10,
            "source_dist": source_dist,
            "top_score": vec_scores[0] if vec_scores else 0,
        })
        print(f"  {'+' if is_hit_5 else '-'} {eq.query[:44]:<46} "
              f"hit@5={is_hit_5} hit@10={is_hit_10} "
              f"vec={source_dist.get('vector',0)} bm25={source_dist.get('bm25',0)} both={source_dist.get('vector+bm25',0)}")

    total_sources = sum(source_counts.values())
    result = {
        "stage": "B-6",
        "description": "候选过滤率分析",
        "source_distribution": {k: v / total_sources if total_sources else 0 for k, v in source_counts.items()},
        "total_sources": source_counts,
        "hit@5_rate": sum(hit_at_5_counts) / len(hit_at_5_counts),
        "details": details,
    }
    print(f"\n  汇总:")
    print(f"    Top10来源分布: vector={result['source_distribution']['vector']:.1%} "
          f"bm25={result['source_distribution']['bm25']:.1%} "
          f"both={result['source_distribution']['vector+bm25']:.1%}")
    print(f"    Hit@5 = {sum(hit_at_5_counts)}/{len(hit_at_5_counts)} ({result['hit@5_rate']:.1%})")
    return result


# =============================================================================
# Stage 5: B-3 多阶段 + Rerank 精排
# =============================================================================

async def stage5_rerank(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("【Stage 5】BM25+向量 -> Qwen-Rerank 精排（需 LLM reranker）")
    print(f"  -> 使用精简数据集 {len(test_cases)} 条，控制 LLM token 消耗")
    print(f"{'='*70}")

    from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
    from src.rag.retrieval.reranker import get_reranker_manager
    from src.rag.storage.vectorstore import get_vectorstore

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
    reranker = get_reranker_manager()

    all_metrics = []
    details = []
    total_latency = 0.0
    total_candidates = 0
    total_after = 0

    for eq in test_cases:
        start = time.time()

        candidates = hybrid.search(eq.query, k=15)
        total_candidates += len(candidates)

        reranked = reranker.rerank(eq.query, candidates, top_n=5)
        docs = [doc for doc, score in reranked]
        total_after += len(docs)

        latency = (time.time() - start) * 1000
        total_latency += latency

        metrics, ret_ids = evaluate_retrieval(docs, eq)
        all_metrics.append(metrics)

        is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])
        rerank_scores = [f"{score:.3f}" for _, score in reranked[:5]]

        details.append({
            "query": eq.query,
            "relevant": eq.relevant_doc_ids,
            "retrieved_top5": ret_ids[:5],
            "hit": is_hit,
            "recall@5": metrics.recall_at_5,
            "mrr": metrics.mrr,
            "candidates": len(candidates),
            "after_rerank": len(docs),
            "rerank_scores": rerank_scores,
            "latency_ms": round(latency, 1),
        })
        print(f"  {'+' if is_hit else '-'} {eq.query[:44]:<46} "
              f"R@5={metrics.recall_at_5:.2f} MRR={metrics.mrr:.2f} "
              f"cand={len(candidates)}->{len(docs)} "
              f"scores={rerank_scores} {latency:.0f}ms")

    avg = average_metrics(all_metrics)
    filter_rate = (total_candidates - total_after) / total_candidates * 100 if total_candidates else 0
    result = {
        "stage": "B-3",
        "description": "多阶段+Rerank精排",
        "metrics": asdict(avg),
        "details": details,
        "total_queries": len(test_cases),
        "hit_count": sum(1 for d in details if d["hit"]),
        "avg_latency_ms": total_latency / len(test_cases),
        "total_candidates": total_candidates,
        "total_after_rerank": total_after,
        "filter_rate": filter_rate,
    }
    print(f"\n  汇总: Hit@5 = {result['hit_count']}/{len(test_cases)} "
          f"({result['hit_count']/len(test_cases):.1%}) | 平均延迟: {result['avg_latency_ms']:.0f}ms")
    print(f"  精排过滤率: {filter_rate:.1f}% ({total_candidates}候选 -> {total_after}精排)")
    return result


# =============================================================================
# Stage 6: B-4 CRAG 正确性反馈
# =============================================================================

async def stage6_crag(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("【Stage 6】Corrective RAG 评估（需 LLM）")
    print(f"  -> 使用精简数据集 {len(test_cases)} 条，控制 LLM token 消耗")
    print(f"{'='*70}")

    from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

    reset_crags()
    pipeline = get_corrective_rag_pipeline()

    crag_decisions = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NO_RESULTS": 0}
    rewrite_count = 0
    all_high_ratios = []
    all_avg_scores = []
    details = []
    total_latency = 0.0

    for eq in test_cases:
        start = time.time()
        try:
            results, grade_result, history = await pipeline.retrieve(eq.query, top_k=5)
            latency = (time.time() - start) * 1000
            total_latency += latency

            decision = grade_result.decision.value.upper()
            crag_decisions[decision] = crag_decisions.get(decision, 0) + 1

            high_ratio = grade_result.high_count / grade_result.total_docs if grade_result.total_docs else 0
            all_high_ratios.append(high_ratio)
            all_avg_scores.append(grade_result.avg_score)

            docs = [doc for doc, score in results]
            metrics, ret_ids = evaluate_retrieval(docs, eq)
            is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])

            rewrites = len(history) - 1
            if rewrites > 0:
                rewrite_count += 1

            details.append({
                "query": eq.query,
                "decision": decision,
                "high_count": grade_result.high_count,
                "total_docs": grade_result.total_docs,
                "high_ratio": high_ratio,
                "avg_score": grade_result.avg_score,
                "rewrite_history": history,
                "rewrite_count": rewrites,
                "recall@5": metrics.recall_at_5,
                "mrr": metrics.mrr,
                "hit": is_hit,
                "latency_ms": round(latency, 1),
            })
            print(f"  {decision:<12} high={grade_result.high_count}/{grade_result.total_docs} "
                  f"avg={grade_result.avg_score:.2f} rewrite={rewrites}x "
                  f"{'+' if is_hit else '-'} {eq.query[:36]:<38} "
                  f"R@5={metrics.recall_at_5:.2f} {latency:.0f}ms")
        except Exception as e:
            print(f"  ERROR: {eq.query[:44]} - {e}")
            details.append({"query": eq.query, "decision": "ERROR", "error": str(e)})

    total = len(test_cases)
    avg_high_ratio = sum(all_high_ratios) / len(all_high_ratios) if all_high_ratios else 0
    avg_avg_score = sum(all_avg_scores) / len(all_avg_scores) if all_avg_scores else 0

    result = {
        "stage": "B-4",
        "description": "CRAG正确性反馈",
        "total_queries": total,
        "crag_decisions": crag_decisions,
        "avg_high_ratio": avg_high_ratio,
        "avg_avg_score": avg_avg_score,
        "rewrite_count": rewrite_count,
        "avg_latency_ms": total_latency / total if total else 0,
        "details": details,
    }
    print(f"\n  CRAG决策: HIGH={crag_decisions['HIGH']} MEDIUM={crag_decisions['MEDIUM']} "
          f"LOW={crag_decisions['LOW']} NO_RESULTS={crag_decisions['NO_RESULTS']}")
    print(f"  HIGH比例: {crag_decisions['HIGH']/total:.1%} | 平均高相关文档: {avg_high_ratio:.1%}")
    print(f"  触发查询重写: {rewrite_count}/{total} | 平均延迟: {result['avg_latency_ms']:.0f}ms")
    return result


# =============================================================================
# Stage 7: B-5 Query Expansion
# =============================================================================

async def stage7_query_expansion(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("【Stage 7】Query Expansion 复杂查询分解")
    print(f"{'='*70}")

    from src.rag.retrieval.query_expander import (
        expand_query, RuleBasedDecomposer, reset_query_expander, ExpandStrategy)

    reset_query_expander()

    # 筛选复杂查询
    complex_kws = ["和", "与", "对比", "比较", "哪个", "还是", "哪些", "以及", "或者"]
    complex_cases = [eq for eq in test_cases if any(kw in eq.query for kw in complex_kws)]
    if not complex_cases:
        complex_cases = test_cases[:8]

    rule_ok = 0
    llm_ok = 0
    expand_latencies = []
    details = []

    for eq in complex_cases:
        start = time.time()

        # Rule-based decomposition (no LLM)
        rule_result = RuleBasedDecomposer.decompose(eq.query)
        rule_sub_queries = [sq.text for sq in rule_result]
        rule_strategy = "rule" if rule_sub_queries else "none"
        if rule_sub_queries:
            rule_ok += 1

        # LLM-based expansion (with token control - only on complex cases)
        try:
            exp_result = await expand_query(eq.query, strategy=ExpandStrategy.HYBRID)
            llm_sub_queries = [sq.text for sq in exp_result.sub_queries]
            llm_strategy = exp_result.strategy.value
            used_llm = exp_result.used_llm
            if llm_sub_queries:
                llm_ok += 1
        except Exception as e:
            llm_sub_queries = []
            llm_strategy = "error"
            used_llm = False
            print(f"    LLM expansion failed: {e}")

        latency = (time.time() - start) * 1000
        expand_latencies.append(latency)

        details.append({
            "query": eq.query,
            "rule_sub_queries": rule_sub_queries,
            "llm_sub_queries": llm_sub_queries,
            "rule_strategy": rule_strategy,
            "llm_strategy": llm_strategy,
            "used_llm": used_llm,
            "latency_ms": round(latency, 1),
        })

        print(f"  原查询: {eq.query}")
        print(f"    规则分解({len(rule_sub_queries)}个): {rule_sub_queries}")
        print(f"    LLM分解({len(llm_sub_queries)}个): {llm_sub_queries}")
        print(f"    策略: rule={rule_strategy} llm={llm_strategy} 耗时={latency:.0f}ms")

    avg_latency = sum(expand_latencies) / len(expand_latencies) if expand_latencies else 0
    result = {
        "stage": "B-5",
        "description": "Query Expansion查询扩展",
        "total_queries": len(complex_cases),
        "rule_decomposed": rule_ok,
        "llm_decomposed": llm_ok,
        "rule_rate": rule_ok / len(complex_cases) if complex_cases else 0,
        "llm_rate": llm_ok / len(complex_cases) if complex_cases else 0,
        "avg_latency_ms": avg_latency,
        "details": details,
    }
    print(f"\n  汇总:")
    print(f"    规则分解率: {rule_ok}/{len(complex_cases)} ({result['rule_rate']:.1%})")
    print(f"    LLM分解率: {llm_ok}/{len(complex_cases)} ({result['llm_rate']:.1%})")
    print(f"    平均分解耗时: {avg_latency:.0f}ms")
    return result


# =============================================================================
# 主函数
# =============================================================================

async def run_benchmark():
    start_time = time.time()

    print(f"\n{'#'*70}")
    print(f"# RAG 检索全流程分级优化测试报告")
    print(f"# 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 测试数据集: {len(EVAL_DATASET)} 条查询")
    print(f"# LLM消耗控制: B-3/B-4/B-5 仅使用前10条精简集")
    print(f"#{'#'*70}")

    all_results = {}

    # ========== Stage 1: B-1 基线向量检索（无LLM，30条）==========
    all_results["stage1_baseline"] = await stage1_baseline_vector(EVAL_DATASET)

    # ========== Stage 2: B-2 混合检索（无LLM，30条）==========
    all_results["stage2_hybrid"] = await stage2_hybrid_retrieval(EVAL_DATASET)

    # ========== Stage 3: C 分块策略（无LLM）==========
    all_results["stage3_chunking"] = stage3_chunking()

    # ========== Stage 4: B-6 候选过滤率（无LLM，30条）==========
    all_results["stage4_filter"] = await stage4_candidate_filter(EVAL_DATASET)

    # ========== Stage 5: B-3 Rerank（精简10条，LLM reranker）==========
    small_set = EVAL_DATASET[:10]
    all_results["stage5_rerank"] = await stage5_rerank(small_set)

    # ========== Stage 6: B-4 CRAG（精简10条，LLM）==========
    all_results["stage6_crag"] = await stage6_crag(small_set)

    # ========== Stage 7: B-5 Query Expansion（复杂查询子集，LLM）==========
    all_results["stage7_query_expansion"] = await stage7_query_expansion(small_set)

    # ========== 汇总报告 ==========
    total_time = time.time() - start_time
    print(f"\n\n{'#'*70}")
    print(f"# 测试结果汇总")
    print(f"# 总耗时: {total_time:.0f}秒")
    print(f"#{'#'*70}")

    # 指标对比表
    print(f"\n{'阶段':<32} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG@3':>8} {'NDCG@5':>8}")
    print("-" * 80)
    stage_labels = [
        ("stage1_baseline", "① B-1 基线-向量检索"),
        ("stage2_hybrid", "② B-2 混合-BM25+向量"),
        ("stage5_rerank", "③ B-3 多阶段+Rerank"),
    ]
    for key, label in stage_labels:
        if key in all_results and "metrics" in all_results[key]:
            m = all_results[key]["metrics"]
            print(f"{label:<32} {m['recall_at_1']:>6.2f} {m['recall_at_3']:>6.2f} "
                  f"{m['recall_at_5']:>6.2f} {m['mrr']:>6.2f} "
                  f"{m['ndcg_at_3']:>8.2f} {m['ndcg_at_5']:>8.2f}")

    # 提升幅度
    b1 = all_results.get("stage1_baseline", {}).get("metrics", {})
    b2 = all_results.get("stage2_hybrid", {}).get("metrics", {})
    b3 = all_results.get("stage5_rerank", {}).get("metrics", {})

    print(f"\n{'='*70}")
    print("【逐级优化提升效果】")
    print(f"{'='*70}")

    def pct(new, old):
        return f"{(new/old-1)*100:+.1f}%" if old > 0 else "N/A"

    print(f"\n  阶段对比（以R@5为基准）:")
    print(f"    B-1 基线向量:      R@5={b1.get('recall_at_5',0):.3f}  MRR={b1.get('mrr',0):.3f}")
    print(f"    B-2 BM25+向量:     R@5={b2.get('recall_at_5',0):.3f}  MRR={b2.get('mrr',0):.3f}  "
          f"vs B-1: R@5 {pct(b2.get('recall_at_5',0), b1.get('recall_at_5',0))}")
    print(f"    B-3 +Rerank精排:  R@5={b3.get('recall_at_5',0):.3f}  MRR={b3.get('mrr',0):.3f}  "
          f"vs B-2: R@5 {pct(b3.get('recall_at_5',0), b2.get('recall_at_5',0))}")

    # 精排过滤率
    filt = all_results.get("stage4_filter", {})
    print(f"\n【候选过滤率（B-6）】")
    sd = filt.get("source_distribution", {})
    print(f"  Top10来源: vector={sd.get('vector',0):.1%}  bm25={sd.get('bm25',0):.1%}  "
          f"vector+bm25={sd.get('vector+bm25',0):.1%}")
    print(f"  候选Hit@5: {filt.get('hit@5_rate',0):.1%}")

    if "stage5_rerank" in all_results:
        rr = all_results["stage5_rerank"]
        print(f"\n【精排过滤率（B-6 with rerank）】")
        print(f"  精排过滤: {rr.get('filter_rate',0):.1f}% ({rr.get('total_candidates',0)}候选 -> "
              f"{rr.get('total_after_rerank',0)}精排)")

    # CRAG评估
    if "stage6_crag" in all_results:
        crag = all_results["stage6_crag"]
        total = crag["total_queries"]
        dec = crag["crag_decisions"]
        print(f"\n【CRAG评估（B-4）】")
        print(f"  HIGH={dec['HIGH']} ({dec['HIGH']/total:.0%})  "
              f"MEDIUM={dec['MEDIUM']} ({dec['MEDIUM']/total:.0%})  "
              f"LOW={dec['LOW']} ({dec['LOW']/total:.0%})  "
              f"NO_RESULTS={dec['NO_RESULTS']}")
        print(f"  平均高相关文档比例: {crag['avg_high_ratio']:.1%}")
        print(f"  触发查询重写: {crag['rewrite_count']}/{total}")
        print(f"  平均延迟: {crag['avg_latency_ms']:.0f}ms")

    # Query Expansion
    if "stage7_query_expansion" in all_results:
        qe = all_results["stage7_query_expansion"]
        print(f"\n【Query Expansion（B-5）】")
        print(f"  规则分解率: {qe['rule_decomposed']}/{qe['total_queries']} ({qe['rule_rate']:.1%})")
        print(f"  LLM分解率: {qe['llm_decomposed']}/{qe['total_queries']} ({qe['llm_rate']:.1%})")
        print(f"  平均分解耗时: {qe['avg_latency_ms']:.0f}ms")

    # 分块策略
    if "stage3_chunking" in all_results:
        chunking = all_results["stage3_chunking"]
        print(f"\n【分块策略（C）】")
        for strat, stats in chunking.get("strategies", {}).items():
            if "error" in stats:
                print(f"  {strat}: ERROR - {stats['error']}")
            else:
                print(f"  {strat}: {stats['chunk_count']}块, "
                      f"Token={stats['min_tokens']:.0f}~{stats['max_tokens']:.0f}, "
                      f"均值={stats['avg_tokens']:.0f}, "
                      f"超限={stats['over_limit_count']}块({stats['over_limit_ratio']:.1%})")
        ss = chunking.get("sentence_split", {})
        print(f"  句子分割: {ss.get('passed',0)}/{ss.get('total',0)} 通过 ({ss.get('rate',0):.1%})")

    # 保存JSON
    output_path = os.path.join(os.path.dirname(__file__), "rag_benchmark_results.json")
    serializable_results = make_serializable(all_results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(os.path.dirname(__file__), "rag_benchmark_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"RAG 检索全流程分级优化测试报告\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time:.0f}秒\n")
        f.write(f"数据集: {len(EVAL_DATASET)} 条查询\n\n")
        for key, label in stage_labels:
            if key in all_results and "metrics" in all_results[key]:
                m = all_results[key]["metrics"]
                f.write(f"{label}: R@5={m['recall_at_5']:.3f} MRR={m['mrr']:.3f}\n")

    print(f"\n  结果JSON: {output_path}")
    print(f"  报告TXT: {report_path}")
    print(f"\n{'#'*70}")
    print(f"# 测试完成，总耗时 {total_time:.0f}秒")
    print(f"{'#'*70}")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_benchmark())
