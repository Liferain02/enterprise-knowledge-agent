#!/usr/bin/env python
"""RAG 检索快速测试"""
import sys, os

for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
from src.rag.retrieval.reranker import get_reranker_manager
from src.rag.storage.vectorstore import get_vectorstore

QUICK_CASES = [
    ("公司的年假政策是什么？", "员工手册"),
    ("试用期多长时间？", "员工手册"),
    ("工资什么时候发放？", "员工手册"),
    ("社会招聘的流程是什么？", "招聘管理制度"),
    ("内部推荐有奖励吗？", "招聘管理制度"),
    ("绩效考核等级有哪些？", "绩效考核制度"),
    ("新员工培训多长时间？", "培训发展体系"),
    ("差旅费报销标准是多少？", "财务报销制度"),
    ("上班时间是几点？", "行政办公管理制度"),
    ("信息密级如何划分？", "信息安全管理制度"),
]

def main():
    print("=" * 60)
    print("RAG 快速测试")
    print("=" * 60)

    hybrid = get_hybrid_retriever_manager("enterprise_knowledge", top_k=10)
    reranker = get_reranker_manager()
    vs = get_vectorstore("enterprise_knowledge")
    all_docs = vs.get()["documents"]
    all_metas = vs.get()["metadatas"]
    docs_for_bm25 = [Document(page_content=c, metadata=m or {}) for c, m in zip(all_docs, all_metas)]
    hybrid.set_documents(docs_for_bm25)

    hits = 0
    total_cand = 0
    total_after = 0
    print(f"\n[多阶段检索] {len(QUICK_CASES)} 条查询 ...")
    for query, target_doc in QUICK_CASES:
        candidates = hybrid.search_with_scores(query, k=10)
        reranked = reranker.rerank(query, [d for d, _, _ in candidates], top_n=5)
        total_cand += len(candidates)
        total_after += len(reranked)
        ret_ids = [d.metadata.get("source", "") for d, _ in reranked] if reranked else []
        hit = any(target_doc in rid for rid in ret_ids)
        if hit:
            hits += 1
        print(f"  {'✓' if hit else '✗'} {query[:30]}")

    r5 = hits / len(QUICK_CASES)
    filter_rate = (total_cand - total_after) / total_cand * 100 if total_cand else 0
    print(f"\n{'=' * 60}")
    print(f"R@5 = {r5:.1%}  ({hits}/{len(QUICK_CASES)})")
    print(f"精排过滤率 = {filter_rate:.0f}%  (候选{total_cand}→精排{total_after})")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
