#!/usr/bin/env python
"""
CRAG + Query Expansion 快速测试 (Stage 6 & 7)
只测试前5条，避免过多 LLM 调用
"""
import asyncio
import sys
import os
import time
import json
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

from langchain_core.documents import Document
from scripts.eval_dataset import EVAL_DATASET


def extract_doc_id(doc, index):
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    if source:
        return os.path.splitext(os.path.basename(source))[0]
    return f"doc_{index}"


def calc_recall(ret, rel, k):
    if not rel: return 0.0
    return len(set(ret[:k]) & set(rel)) / len(rel)


def calc_mrr(ret, rel):
    for i, d in enumerate(ret, 1):
        if d in set(rel): return 1.0 / i
    return 0.0


def make_serializable(obj):
    if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [make_serializable(v) for v in obj]
    if hasattr(obj, 'item'): return obj.item()
    return obj


async def test_crag(test_cases):
    print(f"\n{'='*70}")
    print("【Stage 6】Corrective RAG 评估（需 LLM）")
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
            ret_ids = [extract_doc_id(d, i) for i, d in enumerate(docs)]
            is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])
            rec5 = calc_recall(ret_ids, eq.relevant_doc_ids, 5)
            mrr = calc_mrr(ret_ids, eq.relevant_doc_ids)

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
                "rewrite_count": rewrites,
                "recall@5": rec5,
                "mrr": mrr,
                "hit": is_hit,
                "latency_ms": round(latency, 1),
            })
            print(f"  {decision:<12} high={grade_result.high_count}/{grade_result.total_docs} "
                  f"avg={grade_result.avg_score:.2f} rewrite={rewrites}x "
                  f"{'+' if is_hit else '-'} {eq.query[:36]:<38} "
                  f"R@5={rec5:.2f} {latency:.0f}ms")
        except Exception as e:
            print(f"  ERROR: {eq.query[:44]} -> {e}")
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


async def test_query_expansion(test_cases):
    print(f"\n{'='*70}")
    print("【Stage 7】Query Expansion 复杂查询分解")
    print(f"{'='*70}")

    from src.rag.retrieval.query_expander import (
        expand_query, RuleBasedDecomposer, reset_query_expander, ExpandStrategy)

    reset_query_expander()

    complex_kws = ["和", "与", "对比", "比较", "哪个", "还是", "哪些", "以及", "或者"]
    complex_cases = [eq for eq in test_cases if any(kw in eq.query for kw in complex_kws)]
    if not complex_cases:
        complex_cases = test_cases[:5]

    rule_ok = 0
    llm_ok = 0
    expand_latencies = []
    details = []

    for eq in complex_cases:
        start = time.time()

        rule_result = RuleBasedDecomposer.decompose(eq.query)
        rule_sub_queries = [sq.text for sq in rule_result]
        rule_strategy = "rule" if rule_sub_queries else "none"
        if rule_sub_queries:
            rule_ok += 1

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


async def main():
    print(f"\n{'#'*70}")
    print(f"# CRAG + Query Expansion 快速测试")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#{'#'*70}")

    # 仅测试前5条（极少量 LLM 消耗）
    test_set = EVAL_DATASET[:5]
    print(f"# 测试集: {len(test_set)} 条\n")

    results = {}

    # Stage 6: CRAG
    results["stage6_crag"] = await test_crag(test_set)

    # Stage 7: Query Expansion
    results["stage7_query_expansion"] = await test_query_expansion(test_set)

    # 保存
    out = os.path.join(os.path.dirname(__file__), "rag_crag_qe_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(make_serializable(results), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
