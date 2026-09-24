#!/usr/bin/env python3
"""Small real-corpus retrieval/cache smoke benchmark, not a quality evaluation."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from src.rag.retrieval.retriever import get_retriever_manager
    from src.rag.retrieval.acl_filter import UserContext
    from src.rag.cache import cache
    from src.api.services.knowledge_service import KnowledgeService
    from src.rag.storage.vectorstore import get_vectorstore_manager
    queries = [
        ("ibv_reg_mr 注册内存时有哪些访问权限标志？", "ibv_reg_mr"),
        ("How does rdma_resolve_addr resolve source and destination addresses?", "rdma_resolve_addr"),
        ("ibv_post_send 发布发送请求失败时 bad_wr 表示什么？", "ibv_post_send"),
        ("How does ibv_poll_cq return work completions?", "ibv_poll_cq"),
        ("rping 怎么测试 RDMA 连接？", "rping"),
    ]
    user = UserContext.anonymous()
    retriever = get_retriever_manager()
    report = {"scope": "retrieval/cache smoke only; not an end-to-end RAG quality evaluation", "queries": []}
    for query, expected in queries:
        attempts = []
        for _ in range(2):
            hits, misses = cache.hits, cache.misses
            start = time.perf_counter()
            found = retriever.search_with_rerank(query, k=3, user=user)
            attempts.append({"seconds": round(time.perf_counter() - start, 4),
                             "cache_hits": cache.hits - hits, "cache_misses": cache.misses - misses,
                             "sources": [Path(doc.metadata["source"]).name for doc, _ in found]})
        matched = any(expected in name for name in attempts[0]["sources"])
        stable = attempts[0]["sources"] == attempts[1]["sources"]
        report["queries"].append({"query": query, "expected_source": expected,
                                  "expected_in_top3": matched, "repeat_stable": stable, "attempts": attempts})
        print(json.dumps(report["queries"][-1], ensure_ascii=False), flush=True)
    report["visible_documents"] = len(KnowledgeService().list_documents(user_context=user))
    report["collection"] = get_vectorstore_manager().get_collection_info()
    report["cache_errors"] = cache.errors
    (ROOT / ".run/storage-benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "queries"}, ensure_ascii=False))
    if not all(row["expected_in_top3"] and row["repeat_stable"] for row in report["queries"]):
        raise SystemExit("One or more retrieval smoke checks failed; inspect the report")


if __name__ == "__main__":
    main()
