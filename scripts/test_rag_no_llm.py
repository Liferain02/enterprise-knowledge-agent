#!/usr/bin/env python
"""
Stage B-1 + B-2 + C + B-6: 无 LLM 调用的链路测试
"""
import asyncio
import sys
import os
import time
import json
from dataclasses import asdict
from datetime import datetime
from typing import List, Dict, Any, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        os.environ.pop(k, None)

from langchain_core.documents import Document
from scripts.eval_dataset import EVAL_DATASET, EvalQuery

def extract_doc_id(doc, index):
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    if source:
        import os as _os
        return _os.path.splitext(_os.path.basename(source))[0]
    return f"doc_{index}"

def calc_recall(ret, rel, k):
    if not rel: return 0.0
    return len(set(ret[:k]) & set(rel)) / len(rel)

def calc_precision(ret, rel, k):
    if k == 0: return 0.0
    return len([d for d in ret[:k] if d in set(rel)]) / k

def calc_mrr(ret, rel):
    for i, d in enumerate(ret, 1):
        if d in set(rel): return 1.0 / i
    return 0.0

def calc_dcg(ret, rel, k):
    dcg = 0.0
    for i, d in enumerate(ret[:k], 1):
        if d in set(rel): dcg += 1.0 / (i**2-i+1)**0.5
    return dcg

def calc_ndcg(ret, rel, k):
    if not rel: return 0.0
    dcg = calc_dcg(ret, rel, k)
    ideal = list(rel) + [d for d in ret if d not in set(rel)]
    idcg = calc_dcg(ideal, rel, k)
    return dcg / idcg if idcg else 0.0

def evaluate(ret_docs, eq):
    ret_ids = [extract_doc_id(d, i) for i, d in enumerate(ret_docs)]
    rec1 = calc_recall(ret_ids, eq.relevant_doc_ids, 1)
    rec3 = calc_recall(ret_ids, eq.relevant_doc_ids, 3)
    rec5 = calc_recall(ret_ids, eq.relevant_doc_ids, 5)
    prec1 = calc_precision(ret_ids, eq.relevant_doc_ids, 1)
    prec3 = calc_precision(ret_ids, eq.relevant_doc_ids, 3)
    prec5 = calc_precision(ret_ids, eq.relevant_doc_ids, 5)
    mrr = calc_mrr(ret_ids, eq.relevant_doc_ids)
    ndcg3 = calc_ndcg(ret_ids, eq.relevant_doc_ids, 3)
    ndcg5 = calc_ndcg(ret_ids, eq.relevant_doc_ids, 5)
    metrics = dict(recall_at_1=rec1, recall_at_3=rec3, recall_at_5=rec5,
                   precision_at_1=prec1, precision_at_3=prec3, precision_at_5=prec5,
                   mrr=mrr, ndcg_at_3=ndcg3, ndcg_at_5=ndcg5)
    return metrics, ret_ids

def avg_metric_list(metrics_list):
    if not metrics_list: return dict(recall_at_1=0, recall_at_3=0, recall_at_5=0,
                                     precision_at_1=0, precision_at_3=0, precision_at_5=0,
                                     mrr=0, ndcg_at_3=0, ndcg_at_5=0)
    n = len(metrics_list)
    keys = ['recall_at_1','recall_at_3','recall_at_5','precision_at_1','precision_at_3','precision_at_5','mrr','ndcg_at_3','ndcg_at_5']
    return {k: sum(m[k] for m in metrics_list) / n for k in keys}

# ============================================================
# B-1: 基线向量检索
# ============================================================
async def run_baseline(test_cases):
    print(f"\n{'='*70}")
    print("Stage B-1: 基线 - 纯向量检索（仅 embedding，无 LLM）")
    print(f"{'='*70}")
    from src.rag.retrieval.retriever import get_retriever_manager
    retriever = get_retriever_manager()
    all_m = []
    details = []
    total_lat = 0.0
    for eq in test_cases:
        t0 = time.time()
        docs = retriever.search(eq.query, k=10)
        lat = (time.time()-t0)*1000
        total_lat += lat
        m, ret_ids = evaluate(docs, eq)
        all_m.append(m)
        hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])
        details.append({"query":eq.query, "relevant":eq.relevant_doc_ids,
                        "retrieved_top5":ret_ids[:5], "hit":hit,
                        "recall@5":m['recall_at_5'], "latency_ms":round(lat,1)})
        print(f"  {'✓' if hit else '✗'} {eq.query[:42]:<44} R@5={m['recall_at_5']:.2f} lat={lat:.0f}ms")
    avg = avg_metric_list(all_m)
    hit_cnt = sum(1 for d in details if d['hit'])
    print(f"\n  汇总: Hit@5 = {hit_cnt}/{len(test_cases)} = {hit_cnt/len(test_cases):.1%}")
    return {"metrics":avg, "details":details, "total_queries":len(test_cases),
            "hit_count":hit_cnt, "avg_latency_ms": total_lat/len(test_cases) if test_cases else 0}

# ============================================================
# B-2: 混合检索
# ============================================================
async def run_hybrid(test_cases):
    print(f"\n{'='*70}")
    print("Stage B-2: BM25 + 向量混合检索（无 LLM）")
    print(f"{'='*70}")
    from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
    from src.rag.storage.vectorstore import get_vectorstore
    hybrid = get_hybrid_retriever_manager(collection_name="enterprise_knowledge",
                                          top_k=10, vector_weight=0.5, bm25_weight=0.5)
    vs = get_vectorstore("enterprise_knowledge")
    raw = vs.get(limit=9999)
    docs_bm25 = [Document(page_content=c, metadata=m or {})
                  for c, m in zip(raw.get("documents") or [], raw.get("metadatas") or [])]
    hybrid.set_documents(docs_bm25)
    all_m = []
    details = []
    total_lat = 0.0
    for eq in test_cases:
        t0 = time.time()
        docs = hybrid.search(eq.query, k=10)
        lat = (time.time()-t0)*1000
        total_lat += lat
        m, ret_ids = evaluate(docs, eq)
        all_m.append(m)
        hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])
        details.append({"query":eq.query, "relevant":eq.relevant_doc_ids,
                        "retrieved_top5":ret_ids[:5], "hit":hit,
                        "recall@5":m['recall_at_5'], "latency_ms":round(lat,1)})
        print(f"  {'✓' if hit else '✗'} {eq.query[:42]:<44} R@5={m['recall_at_5']:.2f} lat={lat:.0f}ms")
    avg = avg_metric_list(all_m)
    hit_cnt = sum(1 for d in details if d['hit'])
    print(f"\n  汇总: Hit@5 = {hit_cnt}/{len(test_cases)} = {hit_cnt/len(test_cases):.1%}")
    return {"metrics":avg, "details":details, "total_queries":len(test_cases),
            "hit_count":hit_cnt, "avg_latency_ms":total_lat/len(test_cases) if test_cases else 0}

# ============================================================
# B-6: 过滤率统计
# ============================================================
async def run_filter_rate(test_cases):
    print(f"\n{'='*70}")
    print("Stage B-6: 精排阶段文档过滤率统计（无 LLM）")
    print(f"{'='*70}")
    from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
    from src.rag.retrieval.reranker import get_reranker_manager
    from src.rag.storage.vectorstore import get_vectorstore
    hybrid = get_hybrid_retriever_manager(collection_name="enterprise_knowledge", top_k=10)
    vs = get_vectorstore("enterprise_knowledge")
    raw = vs.get(limit=9999)
    docs_bm25 = [Document(page_content=c, metadata=m or {})
                  for c, m in zip(raw.get("documents") or [], raw.get("metadatas") or [])]
    hybrid.set_documents(docs_bm25)
    reranker = get_reranker_manager()
    total_cand = 0
    total_after = 0
    total_below_thresh = 0
    score_list = []
    details = []
    for eq in test_cases:
        cands = hybrid.search(eq.query, k=30)
        total_cand += len(cands)
        reranked = reranker.rerank(eq.query, cands, top_n=10)
        total_after += len(reranked)
        reranked_full = reranker.rerank(eq.query, cands, top_n=30)
        below = sum(1 for _, s in reranked_full if s < reranker.reranker.score_threshold)
        total_below_thresh += below
        score_list.extend([s for _, s in reranked])
        hit_b = any(extract_doc_id(d,i) in eq.relevant_doc_ids for i,d in enumerate(cands[:5]))
        hit_a = any(extract_doc_id(d,i) in eq.relevant_doc_ids for i,d in enumerate([d for d,_ in reranked[:5]]))
        details.append({"query":eq.query, "candidates":len(cands), "after_topn":len(reranked),
                         "below_thresh":below, "hit_before":hit_b, "hit_after":hit_a,
                         "scores":[f"{s:.3f}" for _,s in reranked[:5]]})
        print(f"  {eq.query[:42]:<44} cand={len(cands):2d}→{len(reranked)} "
              f"below_thresh={below} scores={[f'{s:.2f}' for _,s in reranked[:3]]}")
    topn_f = (total_cand-total_after)/total_cand*100 if total_cand else 0
    thresh_f = total_below_thresh/total_cand*100 if total_cand else 0
    avg_score = sum(score_list)/len(score_list) if score_list else 0
    print(f"\n  总候选: {total_cand}, 精排后: {total_after}")
    print(f"  top_n过滤: {topn_f:.1f}%  |  threshold过滤: {thresh_f:.1f}%")
    print(f"  Rerank分数: min={min(score_list):.3f} avg={avg_score:.3f} max={max(score_list):.3f}")
    return {"total_candidates":total_cand, "total_after_rerank":total_after,
            "total_filtered_by_threshold":total_below_thresh,
            "topn_filter_rate":topn_f, "threshold_filter_rate":thresh_f,
            "avg_rerank_score":avg_score,
            "min_rerank_score":min(score_list) if score_list else 0,
            "max_rerank_score":max(score_list) if score_list else 0,
            "details":details}

# ============================================================
# C: 分块策略
# ============================================================
def run_chunking():
    print(f"\n{'='*70}")
    print("Stage C: 文档分块策略测试（无 LLM）")
    print(f"{'='*70}")
    from src.rag.processing.document_loader import (
        get_document_loader_manager, estimate_tokens, split_sentences)
    from src.rag.processing.chunker import SemanticChunker, HybridChunker
    lm = get_document_loader_manager()
    test_md = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge", "员工手册.md")
    docs = lm.load_file(test_md)
    result = {"strategies": {}}
    for strategy in ["recursive", "semantic", "hybrid", "markdown"]:
        print(f"\n--- 分块策略: {strategy} ---")
        try:
            chunks = lm.split_documents(docs, splitter_type=strategy)
            toks = [estimate_tokens(c.page_content) for c in chunks]
            avg_t = sum(toks)/len(toks)
            min_t, max_t = min(toks), max(toks)
            over_limit = sum(1 for t in toks if t > 800)
            over_r = over_limit/len(toks) if toks else 0
            print(f"  分块数量: {len(chunks)}")
            print(f"  Token范围: {min_t:.0f}~{max_t:.0f}, 平均: {avg_t:.0f}")
            print(f"  超过800token的块: {over_limit}/{len(chunks)} ({over_r:.1%})")
            result["strategies"][strategy] = {
                "chunk_count":len(chunks), "avg_tokens":avg_t,
                "min_tokens":min_t, "max_tokens":max_t,
                "over_limit_count":over_limit, "over_limit_ratio":over_r}
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            result["strategies"][strategy] = {"error": str(e)}
    print(f"\n--- 句子分割验证 ---")
    stests = [
        ("省略号", "用户手册……请仔细阅读以下内容。", 1),
        ("分号", "第一步；第二步；第三步。", 3),
        ("圆圈序号", "① 第一条。② 第二条。③ 第三条。", 3),
        ("数字列表", "1. 第一项。2. 第二项。3. 第三项。", 3),
    ]
    sok = 0
    for name, text, exp in stests:
        sents = split_sentences(text)
        ok = len(sents) == exp
        sok += ok
        print(f"  {'✓' if ok else '✗'} {name}: 期望{exp}句, 实际{len(sents)}句 {sents}")
    result["sentence_split"] = {"total":len(stests), "passed":sok, "rate":sok/len(stests)}
    print(f"\n  句子分割: {sok}/{len(stests)} 通过")
    print(f"\n--- Token 估算精度 ---")
    for text in ["这是一个中文句子。", "This is an English sentence. " * 5,
                 "公司年假政策：工作满1年可休5天。报销流程需要发票和出差报告。"]:
        toks = estimate_tokens(text)
        ratio = len(text)/max(toks,1)
        print(f"  {len(text)}字符 → {toks}token, 比例≈{ratio:.1f}字符/token")
    return result

# ============================================================
# 主函数
# ============================================================
async def main():
    print(f"\n{'#'*70}")
    print(f"# RAG 链路测试 - 无 LLM 阶段")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 测试样例: {len(EVAL_DATASET)} 条")
    print(f"#{'#'*70}")

    all_results = {}

    # B-1
    all_results["baseline"] = await run_baseline(EVAL_DATASET)
    # B-2
    all_results["hybrid"] = await run_hybrid(EVAL_DATASET)
    # B-6
    all_results["filter_rate"] = await run_filter_rate(EVAL_DATASET)
    # C
    all_results["chunking"] = run_chunking()

    # 汇总
    print(f"\n\n{'#'*70}")
    print(f"# 测试结果汇总")
    print(f"#{'#'*70}")

    # 指标对比表
    print(f"\n{'阶段':<30} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG@3':>8} {'NDCG@5':>8}")
    print("-" * 80)
    for key, label in [("baseline","① 基线-向量检索"), ("hybrid","② 混合-BM25+向量")]:
        m = all_results[key]["metrics"]
        print(f"{label:<30} {m['recall_at_1']:>6.2f} {m['recall_at_3']:>6.2f} "
              f"{m['recall_at_5']:>6.2f} {m['mrr']:>6.2f} {m['ndcg_at_3']:>8.2f} {m['ndcg_at_5']:>8.2f}")

    # 提升
    b_r5 = all_results["baseline"]["metrics"]["recall_at_5"]
    h_r5 = all_results["hybrid"]["metrics"]["recall_at_5"]
    b_mrr = all_results["baseline"]["metrics"]["mrr"]
    h_mrr = all_results["hybrid"]["metrics"]["mrr"]
    print(f"\n【提升效果】")
    if b_r5 > 0:
        print(f"  混合 vs 基线: R@5 提升 {(h_r5/b_r5-1)*100:+.1f}% ({b_r5:.3f} → {h_r5:.3f})")
        print(f"  混合 vs 基线: MRR  提升 {(h_mrr/b_mrr-1)*100:+.1f}% ({b_mrr:.3f} → {h_mrr:.3f})")

    # 过滤率
    f = all_results["filter_rate"]
    print(f"\n【精排过滤率】")
    print(f"  top_n过滤: {f['topn_filter_rate']:.1f}%")
    print(f"  threshold过滤: {f['threshold_filter_rate']:.1f}%")
    print(f"  Rerank分数: min={f['min_rerank_score']:.3f} avg={f['avg_rerank_score']:.3f} max={f['max_rerank_score']:.3f}")

    # 分块
    ch = all_results["chunking"]
    print(f"\n【分块策略】")
    for strat, stats in ch["strategies"].items():
        if "error" in stats:
            print(f"  {strat}: ❌ {stats['error']}")
        else:
            print(f"  {strat}: {stats['chunk_count']}块, Token范围{stats['min_tokens']:.0f}~{stats['max_tokens']:.0f}, "
                  f"均值{stats['avg_tokens']:.0f}, 超限{stats['over_limit_count']}块({stats['over_limit_ratio']:.1%})")
    ss = ch.get("sentence_split", {})
    print(f"  句子分割: {ss.get('passed',0)}/{ss.get('total',0)} 通过 ({ss.get('rate',0):.1%})")

    # 保存
    def ms(obj):
        if isinstance(obj, dict): return {k: ms(v) for k, v in obj.items()}
        if isinstance(obj, list): return [ms(v) for v in obj]
        if hasattr(obj, 'item'): return obj.item()
        return obj
    out = os.path.join(os.path.dirname(__file__), "rag_test_nollm.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(ms(all_results), fp, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")
    return all_results

if __name__ == "__main__":
    asyncio.run(main())
