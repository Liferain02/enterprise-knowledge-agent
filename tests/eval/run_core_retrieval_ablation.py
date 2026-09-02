#!/usr/bin/env python
"""对同一冻结 Retrieval Dev 集运行核心检索消融。

方案依次增加 Vector、BM25、RRF Hybrid、Qwen Rerank 与完整 CRAG。报告同时
区分“全部标注目标”和“当前 Chroma snapshot 中确实存在的目标”，避免把未入库
文档误判成算法失败。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.rag.evaluation.retrieval_grader import CorrectiveRAGPipeline
from src.rag.retrieval.acl_filter import UserContext, check_doc_access
from src.rag.retrieval.hybrid_retriever import HybridRetrieverManager
from src.rag.retrieval.query_expander import RuleBasedDecomposer
from src.rag.retrieval.reranker import get_reranker_manager
from src.rag.retrieval.retriever import RetrieverManager
from src.rag.storage.vectorstore import get_vectorstore_manager
from tests.eval.deep_research_retrieval_dev_dataset import RETRIEVAL_DEV_DATASET


VARIANTS = ("vector", "bm25", "hybrid", "hybrid_rerank", "hybrid_rerank_crag")
NO_ANSWER_QUERIES = (
    "实验室食堂周末几点开门？",
    "完全不存在的 XYZABC123 制度是什么？",
    "实验室是否规定每个人必须饲养一只企鹅？",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "核心检索消融结果.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "核心检索消融报告.md"


def _title(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(metadata.get("title") or Path(str(metadata.get("source") or "")).stem)


def _matches(actual: str, expected: str) -> bool:
    return expected.casefold() in actual.casefold()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


class AblationRunner:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.user = UserContext.anonymous()
        settings = get_settings()
        self.vector = RetrieverManager(use_hybrid=False, use_reranker=False, top_k=top_k)
        self.bm25 = HybridRetrieverManager(
            top_k=top_k,
            enable_vector=False,
            enable_bm25=True,
            vector_weight=settings.hybrid_vector_weight,
            bm25_weight=settings.hybrid_bm25_weight,
        )
        self.hybrid = HybridRetrieverManager(
            top_k=top_k,
            enable_vector=True,
            enable_bm25=True,
            vector_weight=settings.hybrid_vector_weight,
            bm25_weight=settings.hybrid_bm25_weight,
        )
        self.reranker = get_reranker_manager()
        self.crag = CorrectiveRAGPipeline(max_retries=2, rerank_before_grade=True)
        self.crag._retriever_manager = RetrieverManager(
            top_k=top_k,
            use_hybrid=True,
            use_reranker=False,
        )

        payload = get_vectorstore_manager().list_documents(limit=10000)
        self.corpus = [
            Document(page_content=text or "", metadata=metadata or {})
            for text, metadata in zip(
                payload.get("documents") or [], payload.get("metadatas") or []
            )
            if text
        ]
        self.corpus_titles = sorted({_title(doc) for doc in self.corpus})

    def _eligible_expected(self, expected: list[str]) -> list[str]:
        return [
            item
            for item in expected
            if any(
                _matches(_title(doc), item)
                and check_doc_access(doc.metadata or {}, self.user)
                for doc in self.corpus
            )
        ]

    async def retrieve(self, variant: str, query: str) -> tuple[list[Document], dict[str, Any]]:
        started = time.perf_counter()
        extra: dict[str, Any] = {}
        if variant == "vector":
            results = self.vector.search_with_score_acl(query, k=self.top_k, user=self.user)
        elif variant == "bm25":
            results = [
                (doc, score)
                for doc, score, _source in self.bm25.search_with_scores(
                    query, k=self.top_k, user=self.user
                )
            ]
        elif variant == "hybrid":
            results = [
                (doc, score)
                for doc, score, _source in self.hybrid.search_with_scores(
                    query, k=self.top_k, user=self.user
                )
            ]
        elif variant == "hybrid_rerank":
            candidates = self.hybrid.search_with_scores(
                query, k=self.top_k * 3, user=self.user
            )
            results = self.reranker.rerank(
                query, [doc for doc, _score, _source in candidates], top_n=self.top_k
            )
        elif variant == "hybrid_rerank_crag":
            results, grade, history = await self.crag.retrieve(
                query=query,
                top_k=self.top_k,
                needs_expansion=RuleBasedDecomposer.needs_expansion(query),
                user=self.user,
            )
            extra = {
                "crag_decision": getattr(getattr(grade, "decision", None), "value", "no_results"),
                "rewrite_history": list(history),
            }
        else:
            raise ValueError(f"未知方案: {variant}")
        extra["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return [doc for doc, _score in results], extra

    async def run(self, variants: list[str], limit: int = 0) -> dict[str, Any]:
        cases = RETRIEVAL_DEV_DATASET[:limit or None]
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case.case_id} {case.category}", flush=True)
            eligible = self._eligible_expected(case.relevant_doc_ids)
            variant_rows = {}
            for variant in variants:
                docs, extra = await self.retrieve(variant, case.query)
                titles = list(dict.fromkeys(_title(doc) for doc in docs))
                ranks = {
                    expected: next(
                        (rank for rank, title in enumerate(titles, 1) if _matches(title, expected)),
                        None,
                    )
                    for expected in case.relevant_doc_ids
                }
                eligible_ranks = [ranks[item] for item in eligible if ranks[item] is not None]
                all_ranks = [rank for rank in ranks.values() if rank is not None]
                variant_rows[variant] = {
                    "titles": titles,
                    "ranks": ranks,
                    "recall_all": len(all_ranks) / len(case.relevant_doc_ids),
                    "recall_indexed": (
                        len(eligible_ranks) / len(eligible) if eligible else None
                    ),
                    "hit": int(bool(all_ranks)),
                    "mrr": 0.0 if not all_ranks else 1.0 / min(all_ranks),
                    **extra,
                }
            rows.append({
                "case_id": case.case_id,
                "category": case.category,
                "query": case.query,
                "expected": case.relevant_doc_ids,
                "indexed_and_allowed_expected": eligible,
                "variants": variant_rows,
            })

        no_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for query in NO_ANSWER_QUERIES:
            for variant in variants:
                docs, extra = await self.retrieve(variant, query)
                no_answer[variant].append({
                    "query": query,
                    "returned_documents": len(docs),
                    "empty": not docs,
                    **extra,
                })

        aggregate = {}
        for variant in variants:
            items = [row["variants"][variant] for row in rows]
            indexed = [item["recall_indexed"] for item in items if item["recall_indexed"] is not None]
            latencies = [float(item["latency_ms"]) for item in items]
            aggregate[variant] = {
                "cases": len(items),
                "recall_all": statistics.mean(item["recall_all"] for item in items),
                "recall_indexed": statistics.mean(indexed) if indexed else 0.0,
                "hit_at_k": statistics.mean(item["hit"] for item in items),
                "mrr": statistics.mean(item["mrr"] for item in items),
                "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
                "latency_p95_ms": _percentile(latencies, 0.95),
                "no_answer_empty_rate": statistics.mean(
                    int(item["empty"]) for item in no_answer[variant]
                ),
            }

        settings = get_settings()
        return {
            "dataset": "deep_research_retrieval_development",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "variants": variants,
            "case_count": len(rows),
            "configuration": {
                "top_k": self.top_k,
                "embedding_model": settings.embedding_model,
                "reranker_provider": settings.reranker_provider,
                "reranker_model": settings.reranker_model,
                "crag_max_retries": 2,
            },
            "corpus": {"chunks": len(self.corpus), "titles": self.corpus_titles},
            "aggregate": aggregate,
            "no_answer": dict(no_answer),
            "results": rows,
        }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 核心检索消融报告", "",
        f"- 运行时间：{payload['generated_at']}",
        f"- 样本：{payload['case_count']} 条；语料：{payload['corpus']['chunks']} chunks",
        "- Recall(indexed) 只统计当前语料中存在且当前用户可访问的标注目标。",
        "- No-answer empty 只表示检索/CRAG 是否返回空，不等价于最终回答人工正确率。",
        "",
        "| 方案 | Recall(all) | Recall(indexed) | Hit@K | MRR | P50/P95(ms) | No-answer empty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["variants"]:
        item = payload["aggregate"][variant]
        lines.append(
            f"| {variant} | {item['recall_all']:.3f} | {item['recall_indexed']:.3f} | "
            f"{item['hit_at_k']:.3f} | {item['mrr']:.3f} | "
            f"{item['latency_p50_ms']:.0f}/{item['latency_p95_ms']:.0f} | "
            f"{item['no_answer_empty_rate']:.3f} |"
        )
    vector = payload["aggregate"].get("vector")
    hybrid = payload["aggregate"].get("hybrid")
    rerank = payload["aggregate"].get("hybrid_rerank")
    crag = payload["aggregate"].get("hybrid_rerank_crag")
    lines.extend(["", "## 结论", ""])
    if vector and hybrid:
        lines.append(
            f"- RRF Hybrid 相对纯 Vector 的 indexed recall 变化："
            f"{vector['recall_indexed']:.3f} → {hybrid['recall_indexed']:.3f}。"
        )
    if hybrid and rerank:
        lines.append(
            f"- Rerank 相对 Hybrid：indexed recall {hybrid['recall_indexed']:.3f} → "
            f"{rerank['recall_indexed']:.3f}，MRR {hybrid['mrr']:.3f} → {rerank['mrr']:.3f}。"
        )
    if rerank and crag:
        latency_multiple = (
            crag["latency_p50_ms"] / rerank["latency_p50_ms"]
            if rerank["latency_p50_ms"] else 0.0
        )
        lines.append(
            f"- CRAG 相对 Rerank：indexed recall {rerank['recall_indexed']:.3f} → "
            f"{crag['recall_indexed']:.3f}，P50 延迟约 {latency_multiple:.1f} 倍；"
            f"其无答案检索为空率为 {crag['no_answer_empty_rate']:.3f}。"
        )

    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("results", []):
        category_rows[row["category"]].append(row)
    if category_rows:
        lines.extend(["", "## 分类别 indexed recall", ""])
        lines.append("| 类别 | " + " | ".join(payload["variants"]) + " |")
        lines.append("|---|" + "---:|" * len(payload["variants"]))
        for category, rows in sorted(category_rows.items()):
            values = []
            for variant in payload["variants"]:
                recalls = [
                    row["variants"][variant]["recall_indexed"]
                    for row in rows
                    if row["variants"][variant]["recall_indexed"] is not None
                ]
                values.append(f"{statistics.mean(recalls):.3f}" if recalls else "—")
            lines.append(f"| {category} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="核心检索消融")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--render-existing", type=Path)
    args = parser.parse_args()
    if args.render_existing:
        payload = json.loads(args.render_existing.read_text(encoding="utf-8"))
        args.report.write_text(render_report(payload), encoding="utf-8")
        print(render_report(payload))
        return
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    invalid = [item for item in variants if item not in VARIANTS]
    if invalid:
        raise SystemExit(f"未知方案: {invalid}")

    settings = get_settings()
    for key, value in (("HTTP_PROXY", settings.http_proxy), ("HTTPS_PROXY", settings.https_proxy)):
        if value:
            os.environ[key] = value
            os.environ[key.lower()] = value

    payload = await AblationRunner(top_k=args.top_k).run(variants, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(render_report(payload))


if __name__ == "__main__":
    asyncio.run(main())
