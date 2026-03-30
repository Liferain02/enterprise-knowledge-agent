#!/usr/bin/env python
"""
RAG 链路逐级优化测试 - 统一执行脚本
============================================================
Stage B-1: 基线向量检索（仅 embedding，无 LLM）
Stage B-2: BM25 + 向量混合检索（无 LLM）
Stage B-6: 精排过滤率统计（无 LLM）
Stage C:   分块策略测试（无 LLM）
Stage B-3: 多阶段 + Rerank（需 LLM，用精简数据集）
Stage B-4/5: CRAG + Query Expansion（需 LLM，用精简数据集）

运行方式:
    conda activate agent-demo
    export HTTPS_PROXY=http://127.0.0.1:7897
    export HTTP_PROXY=http://127.0.0.1:7897
    python scripts/test_rag_retrieval.py
============================================================
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

# 清除代理环境变量（直连）
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        os.environ.pop(k, None)

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


# =============================================================================
# 辅助函数
# =============================================================================

def extract_doc_id(doc: Document, index: int) -> str:
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    if source:
        return os.path.splitext(os.path.basename(source))[0]
    return f"doc_{index}"


def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_k & relevant) / len(relevant)


def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return len([d for d in retrieved_k if d in relevant]) / k


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def calculate_dcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    relevant = set(relevant_ids)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant:
            dcg += 1.0 / (i ** 2 - i + 1) ** 0.5
    return dcg


def calculate_ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = calculate_dcg(retrieved_ids, relevant_ids, k)
    ideal_ids = list(relevant_ids) + [id for id in retrieved_ids if id not in relevant_ids]
    idcg = calculate_dcg(ideal_ids, relevant_ids, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(retrieved_docs: List[Document], eval_query: EvalQuery) -> Tuple[RetrievalMetrics, List[str]]:
    retrieved_ids = [extract_doc_id(doc, i) for i, doc in enumerate(retrieved_docs)]
    metrics = RetrievalMetrics()

    for k in [1, 3, 5]:
        if k == 1:
            metrics.recall_at_1 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
            metrics.precision_at_1 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 3:
            metrics.recall_at_3 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
            metrics.precision_at_3 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
            metrics.ndcg_at_3 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 5:
            metrics.recall_at_5 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
            metrics.precision_at_5 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
            metrics.ndcg_at_5 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, k)

    metrics.mrr = calculate_mrr(retrieved_ids, eval_query.relevant_doc_ids)
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


def print_metrics_table(results: Dict[str, Dict]) -> None:
    """打印阶段对比表格"""
    print(f"\n{'阶段':<30} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG@3':>8} {'NDCG@5':>8}")
    print("-" * 80)
    for key, label in [
        ("baseline", "① 基线-向量检索"),
        ("hybrid", "② 混合-BM25+向量"),
        ("reranker", "③ 多阶段+Rerank精排"),
    ]:
        if key not in results:
            continue
        m = results[key]["metrics"]
        print(f"{label:<30} {m['recall_at_1']:>6.2f} {m['recall_at_3']:>6.2f} "
              f"{m['recall_at_5']:>6.2f} {m['mrr']:>6.2f} {m['ndcg_at_3']:>8.2f} "
              f"{m['ndcg_at_5']:>8.2f}")


# =============================================================================
# Stage B-1: 基线向量检索
# =============================================================================

async def test_baseline_vector(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("Stage B-1: 基线 - 纯向量检索（仅 embedding，无 LLM）")
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
            "latency_ms": round(latency, 1),
        })
        print(f"  {'✓' if is_hit else '✗'} {eq.query[:40]:<42} R@5={metrics.recall_at_5:.2f} lat={latency:.0f}ms")

    avg = average_metrics(all_metrics)
    result = {
        "metrics": asdict(avg),
        "details": details,
        "total_queries": len(test_cases),
        "hit_count": sum(1 for d in details if d["hit"]),
        "avg_latency_ms": total_latency / len(test_cases) if test_cases else 0,
    }

    print(f"\n  汇总: Hit@5 = {result['hit_count']}/{len(test_cases)} = {result['hit_count']/len(test_cases):.1%}")
    return result


# =============================================================================
# Stage B-2: 混合检索
# =============================================================================

async def test_hybrid_retrieval(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("Stage B-2: BM25 + 向量混合检索（无 LLM）")
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
            "latency_ms": round(latency, 1),
        })
        print(f"  {'✓' if is_hit else '✗'} {eq.query[:40]:<42} R@5={metrics.recall_at_5:.2f} lat={latency:.0f}ms")

    avg = average_metrics(all_metrics)
    result = {
        "metrics": asdict(avg),
        "details": details,
        "total_queries": len(test_cases),
        "hit_count": sum(1 for d in details if d["hit"]),
        "avg_latency_ms": total_latency / len(test_cases) if test_cases else 0,
    }

    print(f"\n  汇总: Hit@5 = {result['hit_count']}/{len(test_cases)} = {result['hit_count']/len(test_cases):.1%}")
    return result


# =============================================================================
# Stage B-3: 多阶段 + Rerank
# =============================================================================

async def test_multistage_rerank(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("Stage B-3: BM25+向量 → Qwen3-Rerank 精排（需 LLM）")
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
    total_before = 0
    total_after = 0

    for eq in test_cases:
        start = time.time()

        # Stage 1: 混合检索候选
        candidates = hybrid.search(eq.query, k=15)
        total_before += len(candidates)

        # Stage 2: Rerank 精排
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
            "candidates_before": len(candidates),
            "candidates_after": len(docs),
            "rerank_scores": rerank_scores,
            "latency_ms": round(latency, 1),
        })
        print(f"  {'✓' if is_hit else '✗'} {eq.query[:40]:<42} "
              f"R@5={metrics.recall_at_5:.2f} "
              f"cand={len(candidates)}→{len(docs)} "
              f"scores={rerank_scores} lat={latency:.0f}ms")

    avg = average_metrics(all_metrics)
    filter_rate = (total_before - total_after) / total_before * 100 if total_before else 0
    result = {
        "metrics": asdict(avg),
        "details": details,
        "total_queries": len(test_cases),
        "hit_count": sum(1 for d in details if d["hit"]),
        "avg_latency_ms": total_latency / len(test_cases) if test_cases else 0,
        "total_before": total_before,
        "total_after": total_after,
        "filter_rate": filter_rate,
    }

    print(f"\n  汇总: Hit@5 = {result['hit_count']}/{len(test_cases)} = {result['hit_count']/len(test_cases):.1%}")
    print(f"  过滤率: {filter_rate:.1f}% ({total_before}候选 → {total_after}精排)")
    return result


# =============================================================================
# Stage B-6: 精排过滤率统计
# =============================================================================

async def test_filter_rate(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("Stage B-6: 精排阶段文档过滤率统计（无 LLM）")
    print(f"{'='*70}")

    from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
    from src.rag.retrieval.reranker import get_reranker_manager
    from src.rag.storage.vectorstore import get_vectorstore

    hybrid = get_hybrid_retriever_manager(collection_name="enterprise_knowledge", top_k=10)
    vs = get_vectorstore("enterprise_knowledge")
    all_results_raw = vs.get(limit=9999)
    all_docs = all_results_raw.get("documents") or []
    all_metadatas = all_results_raw.get("metadatas") or []
    docs_for_bm25 = [Document(page_content=c, metadata=m or {}) for c, m in zip(all_docs, all_metadatas)]
    hybrid.set_documents(docs_for_bm25)
    reranker = get_reranker_manager()

    total_candidates = 0
    total_after_rerank = 0
    total_filtered_by_threshold = 0
    rerank_score_list = []
    details = []

    for eq in test_cases:
        candidates = hybrid.search(eq.query, k=30)
        total_candidates += len(candidates)

        # Rerank top10
        reranked_top = reranker.rerank(eq.query, candidates, top_n=10)
        total_after_rerank += len(reranked_top)

        # Rerank top30 for threshold analysis
        reranked_full = reranker.rerank(eq.query, candidates, top_n=30)
        below_threshold = sum(1 for _, score in reranked_full if score < reranker.reranker.score_threshold)
        total_filtered_by_threshold += below_threshold
        rerank_score_list.extend([score for _, score in reranked_top])

        is_hit_before = any(extract_doc_id(d, i) in eq.relevant_doc_ids for i, d in enumerate(candidates[:5]))
        is_hit_after = any(extract_doc_id(d, i) in eq.relevant_doc_ids for i, d in enumerate([doc for doc, _ in reranked_top[:5]]))

        details.append({
            "query": eq.query,
            "candidates": len(candidates),
            "after_topn": len(reranked_top),
            "below_threshold": below_threshold,
            "hit_before": is_hit_before,
            "hit_after": is_hit_after,
            "scores": [f"{s:.3f}" for _, s in reranked_top[:5]],
        })

        print(f"  {eq.query[:40]:<42} cand={len(candidates):2d}→{len(reranked_top):2d} "
              f"below_thresh={below_threshold} scores={[f'{s:.2f}' for _,s in reranked_top[:3]]}")

    topn_filter_rate = (total_candidates - total_after_rerank) / total_candidates * 100 if total_candidates else 0
    threshold_filter_rate = total_filtered_by_threshold / total_candidates * 100 if total_candidates else 0
    total_filter_rate = (total_candidates - total_after_rerank) / total_candidates * 100 if total_candidates else 0

    avg_score = sum(rerank_score_list) / len(rerank_score_list) if rerank_score_list else 0

    result = {
        "total_candidates": total_candidates,
        "total_after_rerank": total_after_rerank,
        "total_filtered_by_threshold": total_filtered_by_threshold,
        "topn_filter_rate": topn_filter_rate,
        "threshold_filter_rate": threshold_filter_rate,
        "total_filter_rate": total_filter_rate,
        "avg_rerank_score": avg_score,
        "min_rerank_score": min(rerank_score_list) if rerank_score_list else 0,
        "max_rerank_score": max(rerank_score_list) if rerank_score_list else 0,
        "details": details,
    }

    print(f"\n  总候选文档数: {total_candidates}")
    print(f"  精排后文档数: {total_after_rerank}")
    print(f"  被 top_n 过滤: {topn_filter_rate:.1f}%")
    print(f"  被 threshold 过滤: {threshold_filter_rate:.1f}%")
    print(f"  Rerank 分数: min={result['min_rerank_score']:.3f} avg={avg_score:.3f} max={result['max_rerank_score']:.3f}")
    return result


# =============================================================================
# Stage C: 分块策略测试
# =============================================================================

def test_chunking_strategies() -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("Stage C: 文档分块策略测试（无 LLM）")
    print(f"{'='*70}")

    from src.rag.processing.document_loader import (
        get_document_loader_manager,
        estimate_tokens,
        split_sentences,
        OPTIMIZED_SEPARATORS,
    )
    from src.rag.processing.chunker import SemanticChunker, HybridChunker

    loader_manager = get_document_loader_manager()
    test_md = "/share/home/lifr/workspace/code/enterprise-knowledge-agent/data/knowledge/员工手册.md"
    docs = loader_manager.load_file(test_md)

    result = {"strategies": {}}

    for strategy in ["recursive", "semantic", "hybrid", "markdown"]:
        print(f"\n--- 分块策略: {strategy} ---")
        try:
            chunks = loader_manager.split_documents(docs, splitter_type=strategy)
            total_chars = sum(len(c.page_content) for c in chunks)
            total_tokens = sum(estimate_tokens(c.page_content) for c in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            min_tokens = min(estimate_tokens(c.page_content) for c in chunks) if chunks else 0
            max_tokens = max(estimate_tokens(c.page_content) for c in chunks) if chunks else 0

            # Token 限制合规检查
            max_allowed = 800  # 默认 max_tokens
            over_limit = sum(1 for c in chunks if estimate_tokens(c.page_content) > max_allowed)
            over_ratio = over_limit / len(chunks) if chunks else 0

            print(f"  分块数量: {len(chunks)}")
            print(f"  Token范围: {min_tokens:.0f} ~ {max_tokens:.0f}, 平均: {avg_tokens:.0f}")
            print(f"  超过{max_allowed}token的块: {over_limit}/{len(chunks)} ({over_ratio:.1%})")

            result["strategies"][strategy] = {
                "chunk_count": len(chunks),
                "total_tokens": total_tokens,
                "avg_tokens": avg_tokens,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "over_limit_count": over_limit,
                "over_limit_ratio": over_ratio,
            }
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            result["strategies"][strategy] = {"error": str(e)}

    # 句子分割验证
    print(f"\n--- 句子分割验证 ---")
    sentence_tests = [
        ("省略号", "用户手册……请仔细阅读以下内容。", 1),
        ("分号", "第一步：提交申请；第二步：等待审批；第三步：领取结果。", 3),
        ("圆圈序号", "① 第一条。② 第二条。③ 第三条。", 3),
        ("数字列表", "1. 第一项。2. 第二项。3. 第三项。", 3),
    ]
    sentence_ok = 0
    for name, text, expected in sentence_tests:
        sents = split_sentences(text)
        ok = len(sents) == expected
        sentence_ok += ok
        print(f"  {'✓' if ok else '✗'} {name}: 期望{expected}句, 实际{len(sents)}句 {sents}")
    result["sentence_split"] = {
        "total": len(sentence_tests),
        "passed": sentence_ok,
        "rate": sentence_ok / len(sentence_tests),
    }
    print(f"\n  句子分割: {sentence_ok}/{len(sentence_tests)} 通过")

    # Token 估算精度
    print(f"\n--- Token 估算精度 ---")
    token_tests = [
        "这是一个中文句子。",           # 纯中文
        "This is an English sentence. " * 5,  # 纯英文
        "公司年假政策：工作满1年可休5天。报销流程需要发票和出差报告。",  # 混合
    ]
    for text in token_tests:
        tokens = estimate_tokens(text)
        chars = len(text)
        ratio = chars / max(tokens, 1)
        print(f"  {chars}字符 → {tokens}token, 比例≈{ratio:.1f}字符/token")

    return result


# =============================================================================
# Stage B-4: CRAG 评估（精简数据集）
# =============================================================================

async def test_crag(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("Stage B-4: Corrective RAG 评估（需 LLM，精简数据集）")
    print(f"{'='*70}")

    from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

    reset_crags()
    pipeline = get_corrective_rag_pipeline()

    crag_decisions = {"HIGH": 0, "LOW": 0, "NO_RESULTS": 0}
    rewrite_triggered = 0
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

            rewrite_count = len(history) - 1
            if rewrite_count > 0:
                rewrite_triggered += 1

            details.append({
                "query": eq.query,
                "decision": decision,
                "high_count": grade_result.high_count,
                "total_docs": grade_result.total_docs,
                "high_ratio": high_ratio,
                "avg_score": grade_result.avg_score,
                "rewrite_history": history,
                "rewrite_count": rewrite_count,
                "recall@5": metrics.recall_at_5,
                "hit": is_hit,
                "latency_ms": round(latency, 1),
            })

            print(f"  {decision:<12} high={grade_result.high_count}/{grade_result.total_docs} "
                  f"avg={grade_result.avg_score:.2f} "
                  f"rewrite={rewrite_count}x "
                  f"{'✓' if is_hit else '✗'} {eq.query[:35]:<37} "
                  f"R@5={metrics.recall_at_5:.2f} lat={latency:.0f}ms")
        except Exception as e:
            print(f"  ERROR: {eq.query[:40]} - {e}")
            details.append({"query": eq.query, "decision": "ERROR", "error": str(e)})

    avg_high_ratio = sum(all_high_ratios) / len(all_high_ratios) if all_high_ratios else 0
    avg_avg_score = sum(all_avg_scores) / len(all_avg_scores) if all_avg_scores else 0
    total = len(test_cases)

    result = {
        "total_queries": total,
        "crag_decisions": crag_decisions,
        "avg_high_ratio": avg_high_ratio,
        "avg_avg_score": avg_avg_score,
        "rewrite_triggered_count": rewrite_triggered,
        "avg_latency_ms": total_latency / total if total else 0,
        "details": details,
    }

    print(f"\n  CRAG 决策分布: HIGH={crag_decisions['HIGH']} LOW={crag_decisions['LOW']} NO_RESULTS={crag_decisions['NO_RESULTS']}")
    print(f"  HIGH 比例: {crag_decisions['HIGH']/total:.1%}")
    print(f"  平均高相关文档比例: {avg_high_ratio:.1%}")
    print(f"  平均相关分: {avg_avg_score:.3f}")
    print(f"  触发查询重写次数: {rewrite_triggered}/{total}")
    print(f"  平均延迟: {result['avg_latency_ms']:.0f}ms")
    return result


# =============================================================================
# Stage B-5: Query Expansion（精简数据集）
# =============================================================================

async def test_query_expansion(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("Stage B-5: Query Expansion 复杂查询分解（需 LLM，精简数据集）")
    print(f"{'='*70}")

    from src.rag.retrieval.query_expander import (
        expand_query,
        decompose_and_retrieve,
        ExpandStrategy,
        RuleBasedDecomposer,
        reset_query_expander,
    )

    reset_query_expander()
    complex_cases = [eq for eq in test_cases if any(
        kw in eq.query for kw in ["和", "与", "对比", "比较", "哪个", "还是", "哪些"]
    )]

    if not complex_cases:
        complex_cases = test_cases[:6]

    decomp_count = 0
    expand_latencies = []
    details = []

    for eq in complex_cases:
        start = time.time()

        rule_result = RuleBasedDecomposer.decompose(eq.query)
        rule_sub_queries = [sq.text for sq in rule_result]

        try:
            exp_result = await expand_query(eq.query, strategy=ExpandStrategy.HYBRID)
            llm_sub_queries = [sq.text for sq in exp_result.sub_queries]
            strategy = exp_result.strategy.value
            used_llm = exp_result.used_llm
        except Exception as e:
            llm_sub_queries = []
            strategy = "error"
            used_llm = False
            print(f"  LLM expansion failed: {e}")

        latency = (time.time() - start) * 1000
        expand_latencies.append(latency)

        if rule_sub_queries or llm_sub_queries:
            decomp_count += 1

        details.append({
            "query": eq.query,
            "rule_sub_queries": rule_sub_queries,
            "llm_sub_queries": llm_sub_queries,
            "strategy": strategy,
            "used_llm": used_llm,
            "latency_ms": round(latency, 1),
        })

        print(f"  原查询: {eq.query}")
        print(f"    规则分解: {rule_sub_queries}")
        print(f"    LLM 分解: {llm_sub_queries}")
        print(f"    策略={strategy} 耗时={latency:.0f}ms")

    avg_latency = sum(expand_latencies) / len(expand_latencies) if expand_latencies else 0
    result = {
        "total_queries": len(complex_cases),
        "decomposed_count": decomp_count,
        "decompose_rate": decomp_count / len(complex_cases) if complex_cases else 0,
        "avg_expand_latency_ms": avg_latency,
        "details": details,
    }
    print(f"\n  分解率: {decomp_count}/{len(complex_cases)} = {result['decompose_rate']:.1%}")
    print(f"  平均分解耗时: {avg_latency:.0f}ms")
    return result


# =============================================================================
# 主函数
# =============================================================================

async def run_all_tests():
    print(f"\n{'#'*70}")
    print(f"# RAG 链路逐级优化测试")
    print(f"# 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 测试样例: {len(EVAL_DATASET)} 条")
    print(f"#{'#'*70}")

    all_results = {}

    # ========== Stage B-1: 基线向量检索（无 LLM）==========
    all_results["baseline"] = await test_baseline_vector(EVAL_DATASET)

    # ========== Stage B-2: 混合检索（无 LLM）==========
    all_results["hybrid"] = await test_hybrid_retrieval(EVAL_DATASET)

    # ========== Stage C: 分块策略（无 LLM）==========
    all_results["chunking"] = test_chunking_strategies()

    # ========== Stage B-6: 过滤率统计（无 LLM）==========
    all_results["filter_rate"] = await test_filter_rate(EVAL_DATASET)

    # ========== Stage B-3: 多阶段 + Rerank（需 LLM，用精简集）==========
    # 用 10 条避免过多 LLM 调用
    small_set = EVAL_DATASET[:10]
    all_results["reranker"] = await test_multistage_rerank(small_set)

    # ========== Stage B-4: CRAG（需 LLM，用精简集）==========
    all_results["crag"] = await test_crag(small_set)

    # ========== Stage B-5: Query Expansion（需 LLM，用精简集）==========
    all_results["query_expansion"] = await test_query_expansion(EVAL_DATASET)

    # ========== 汇总报告 ==========
    print(f"\n\n{'#'*70}")
    print(f"# 测试结果汇总")
    print(f"#{'#'*70}")

    print_metrics_table(all_results)

    # 提升幅度
    baseline_r5 = all_results["baseline"]["metrics"]["recall_at_5"]
    hybrid_r5 = all_results["hybrid"]["metrics"]["recall_at_5"]
    reranker_r5 = all_results["reranker"]["metrics"]["recall_at_5"]
    baseline_mrr = all_results["baseline"]["metrics"]["mrr"]
    reranker_mrr = all_results["reranker"]["metrics"]["mrr"]

    print(f"\n【提升效果】")
    if baseline_r5 > 0:
        print(f"  混合检索 vs 基线: R@5 提升 {(hybrid_r5/baseline_r5 - 1)*100:+.1f}% ({baseline_r5:.3f} → {hybrid_r5:.3f})")
    if baseline_r5 > 0 and reranker_r5 > 0:
        print(f"  Rerank  vs 基线: R@5 提升 {(reranker_r5/baseline_r5 - 1)*100:+.1f}% ({baseline_r5:.3f} → {reranker_r5:.3f})")
    if baseline_mrr > 0 and reranker_mrr > 0:
        print(f"  Rerank  vs 基线: MRR  提升 {(reranker_mrr/baseline_mrr - 1)*100:+.1f}% ({baseline_mrr:.3f} → {reranker_mrr:.3f})")

    # 过滤率
    filt = all_results["filter_rate"]
    print(f"\n【精排过滤率】")
    print(f"  top_n 过滤: {filt['topn_filter_rate']:.1f}%")
    print(f"  threshold 过滤: {filt['threshold_filter_rate']:.1f}%")
    print(f"  Rerank 分数: min={filt['min_rerank_score']:.3f} avg={filt['avg_rerank_score']:.3f} max={filt['max_rerank_score']:.3f}")

    # CRAG
    if "crag" in all_results:
        crag = all_results["crag"]
        total = crag["total_queries"]
        print(f"\n【CRAG 评估】")
        print(f"  HIGH={crag['crag_decisions']['HIGH']}/{total} ({crag['crag_decisions']['HIGH']/total:.1%})")
        print(f"  LOW={crag['crag_decisions']['LOW']}/{total} ({crag['crag_decisions']['LOW']/total:.1%})")
        print(f"  NO_RESULTS={crag['crag_decisions']['NO_RESULTS']}/{total}")
        print(f"  触发重写: {crag['rewrite_triggered_count']}/{total}")
        print(f"  平均延迟: {crag['avg_latency_ms']:.0f}ms")

    # Query Expansion
    if "query_expansion" in all_results:
        qe = all_results["query_expansion"]
        print(f"\n【Query Expansion】")
        print(f"  分解率: {qe['decomposed_count']}/{qe['total_queries']} = {qe['decompose_rate']:.1%}")
        print(f"  平均分解耗时: {qe['avg_expand_latency_ms']:.0f}ms")

    # 分块
    if "chunking" in all_results:
        chunking = all_results["chunking"]
        print(f"\n【分块策略】")
        for strat, stats in chunking["strategies"].items():
            if "error" in stats:
                print(f"  {strat}: ❌ {stats['error']}")
            else:
                print(f"  {strat}: {stats['chunk_count']}块, Token范围{stats['min_tokens']:.0f}~{stats['max_tokens']:.0f}, 均值{stats['avg_tokens']:.0f}, 超限{stats['over_limit_count']}块({stats['over_limit_ratio']:.1%})")
        ss = chunking.get("sentence_split", {})
        print(f"  句子分割: {ss.get('passed',0)}/{ss.get('total',0)} 通过 ({ss.get('rate',0):.1%})")

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "rag_test_results.json")
    # Convert non-serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(all_results), f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {output_path}")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_all_tests())
