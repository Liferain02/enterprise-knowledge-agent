#!/usr/bin/env python
"""秋招版 Retrieval Benchmark。

只评估 Retrieval，不调用答案生成。四个 Variant 与任务书一一对应：
Dense、Hybrid+RRF、Hybrid+Rerank、Current Final（选择性 Standalone）。
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
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.agent.agents.knowledge import _retrieve_documents
from src.rag.retrieval.acl_filter import UserContext, check_doc_access
from src.rag.retrieval.hybrid_retriever import HybridRetrieverManager
from src.rag.retrieval.query_expander import RuleBasedDecomposer
from src.rag.retrieval.retriever import RetrieverManager
from src.rag.retrieval.reranker import get_reranker_manager
from src.rag.storage.vectorstore import get_vectorstore_manager
from tests.eval.秋招版检索评测数据集 import RETRIEVAL_BENCHMARK_CASES, RetrievalBenchmarkCase


VARIANTS = ("dense", "hybrid_rrf", "hybrid_rerank", "current_final")
VARIANT_LABELS = {
    "dense": "A Dense Only",
    "hybrid_rrf": "B Hybrid + RRF",
    "hybrid_rerank": "C Hybrid + Rerank",
    "current_final": "D Current Final",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "秋招版检索评测结果.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "秋招版检索评测报告.md"


def _title(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(metadata.get("title") or Path(str(metadata.get("source") or "")).stem)


def _matches(actual: str, expected: str) -> bool:
    return expected.casefold() in actual.casefold() or actual.casefold() in expected.casefold()


def _unique_titles(docs: Iterable[Document]) -> list[str]:
    result: list[str] = []
    for doc in docs:
        title = _title(doc)
        if title and title not in result:
            result.append(title)
    return result


def _gold_ranks(titles: list[str], gold: tuple[str, ...]) -> dict[str, int | None]:
    return {
        expected: next(
            (rank for rank, title in enumerate(titles, 1) if _matches(title, expected)),
            None,
        )
        for expected in gold
    }


def _ndcg_at_5(titles: list[str], gold: tuple[str, ...]) -> float:
    if not gold:
        return 0.0
    ranks = _gold_ranks(titles[:5], gold)
    dcg = sum(1.0 / __import__("math").log2(rank + 1) for rank in ranks.values() if rank)
    ideal = min(len(gold), 5)
    idcg = sum(1.0 / __import__("math").log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def _metrics(titles: list[str], case: RetrievalBenchmarkCase) -> dict[str, Any]:
    ranks = _gold_ranks(titles, case.gold_doc_ids)
    found = [rank for rank in ranks.values() if rank is not None]
    top1 = int(bool(found) and min(found) == 1)
    hit5 = int(bool(found))
    coverage = sum(rank is not None and rank <= 5 for rank in ranks.values()) / len(case.gold_doc_ids)
    return {
        "hit_at_1": top1,
        "hit_at_5": hit5,
        "mrr_at_5": 1.0 / min(found) if found else 0.0,
        "ndcg_at_5": _ndcg_at_5(titles, case.gold_doc_ids),
        "coverage_at_5": coverage,
        "first_gold_rank": min(found) if found else None,
        "gold_ranks": ranks,
        "distractor_top1": int(bool(titles) and any(_matches(titles[0], item) for item in case.distractor_ids)),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


class BenchmarkRunner:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.user = UserContext.anonymous()
        settings = get_settings()
        self.dense = RetrieverManager(top_k=top_k, use_hybrid=False, use_reranker=False)
        self.hybrid = HybridRetrieverManager(
            top_k=top_k,
            enable_vector=True,
            enable_bm25=True,
            vector_weight=settings.hybrid_vector_weight,
            bm25_weight=settings.hybrid_bm25_weight,
        )
        self.reranker = get_reranker_manager()

    async def retrieve(self, variant: str, query: str) -> tuple[list[Document], float, dict[str, Any]]:
        started = time.perf_counter()
        details: dict[str, Any] = {"rewrite_history": [query]}
        if variant == "dense":
            scored = self.dense.search_with_score_acl(query, k=self.top_k, user=self.user)
            docs = [doc for doc, _score in scored]
        elif variant == "hybrid_rrf":
            scored = self.hybrid.search_with_scores(query, k=self.top_k, user=self.user)
            docs = [doc for doc, _score, _source in scored]
        elif variant == "hybrid_rerank":
            candidates = self.hybrid.search_with_scores(query, k=self.top_k * 3, user=self.user)
            scored = self.reranker.rerank(query, [doc for doc, _score, _source in candidates], top_n=self.top_k)
            docs = [doc for doc, _score in scored]
        elif variant == "current_final":
            results, _grade, history = await _retrieve_documents(
                query,
                self.top_k,
                RuleBasedDecomposer.needs_expansion(query),
                self.user,
            )
            docs = [doc for doc, _score in results]
            details["rewrite_history"] = list(history)
        else:
            raise ValueError(f"未知 Variant: {variant}")
        latency_ms = (time.perf_counter() - started) * 1000
        details["acl_violation"] = sum(
            not check_doc_access(doc.metadata or {}, self.user) for doc in docs
        )
        return docs, latency_ms, details

    async def run(self, variants: tuple[str, ...] = VARIANTS) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(RETRIEVAL_BENCHMARK_CASES, 1):
            print(f"[{index}/{len(RETRIEVAL_BENCHMARK_CASES)}] {case.case_id}", flush=True)
            variant_rows: dict[str, Any] = {}
            for variant in variants:
                docs, latency_ms, details = await self.retrieve(variant, case.query)
                titles = _unique_titles(docs)
                variant_rows[variant] = {
                    "query": case.query,
                    "titles": titles,
                    "latency_ms": round(latency_ms, 3),
                    **_metrics(titles, case),
                    **details,
                }
                for paraphrase in case.paraphrase_queries:
                    p_docs, p_latency, p_details = await self.retrieve(variant, paraphrase)
                    p_titles = _unique_titles(p_docs)
                    variant_rows[variant].setdefault("paraphrases", []).append({
                        "query": paraphrase,
                        "titles": p_titles,
                        "latency_ms": round(p_latency, 3),
                        **_metrics(p_titles, case),
                        **p_details,
                    })
            rows.append({
                "case_id": case.case_id,
                "category": case.category,
                "gold_level": case.gold_level,
                "gold_doc_ids": list(case.gold_doc_ids),
                "acceptable_sources": list(case.acceptable_sources),
                "distractor_ids": list(case.distractor_ids),
                "paraphrase_queries": list(case.paraphrase_queries),
                "variants": variant_rows,
            })

        aggregate: dict[str, Any] = {}
        for variant in variants:
            primary = [row["variants"][variant] for row in rows]
            paraphrases = [p for item in primary for p in item.get("paraphrases", [])]
            category_metrics: dict[str, Any] = {}
            for category in sorted({row["category"] for row in rows}):
                category_rows = [row["variants"][variant] for row in rows if row["category"] == category]
                category_metrics[category] = self._aggregate_rows(category_rows)
            drift_rows = [item for item in primary if item.get("paraphrases")]
            top1_drift = []
            paraphrase_regressions = []
            for item in drift_rows:
                base_top1 = item["titles"][0] if item["titles"] else ""
                for paraphrase in item["paraphrases"]:
                    top1_drift.append(int(bool(base_top1) and bool(paraphrase["titles"]) and base_top1 != paraphrase["titles"][0]))
                    paraphrase_regressions.append(int(item["hit_at_5"] and not paraphrase["hit_at_5"]))
            aggregate[variant] = {
                **self._aggregate_rows(primary),
                "category_metrics": category_metrics,
                "paraphrase_count": len(paraphrases),
                "top1_drift_rate": sum(top1_drift) / len(top1_drift) if top1_drift else 0.0,
                "paraphrase_regression_rate": sum(paraphrase_regressions) / len(paraphrase_regressions) if paraphrase_regressions else 0.0,
                "paraphrase_metrics": self._aggregate_rows(paraphrases) if paraphrases else {},
            }

        settings = get_settings()
        return {
            "dataset": "lab_retrieval_benchmark_v1",
            "sample_count": len(rows),
            "variant_labels": VARIANT_LABELS,
            "configuration": {
                "top_k": self.top_k,
                "gold_level": "document",
                "hybrid_search_enabled": settings.hybrid_search_enabled,
                "reranker_enabled": settings.reranker_enabled,
                "reranker_provider": settings.reranker_provider,
                "reranker_model": settings.reranker_model,
                "crag_enabled": settings.crag_enabled,
            },
            "corpus": self._corpus_summary(),
            "aggregate": aggregate,
            "results": rows,
        }

    @staticmethod
    def _aggregate_rows(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"count": 0}
        latencies = [float(item.get("latency_ms", 0.0)) for item in items]
        return {
            "count": len(items),
            "hit_at_1": statistics.mean(item["hit_at_1"] for item in items),
            "hit_at_5": statistics.mean(item["hit_at_5"] for item in items),
            "mrr_at_5": statistics.mean(item["mrr_at_5"] for item in items),
            "ndcg_at_5": statistics.mean(item["ndcg_at_5"] for item in items),
            "coverage_at_5": statistics.mean(item["coverage_at_5"] for item in items),
            "distractor_top1_rate": statistics.mean(item["distractor_top1"] for item in items),
            "acl_violation": sum(item.get("acl_violation", 0) for item in items),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": _percentile(latencies, 0.95),
        }

    def _corpus_summary(self) -> dict[str, Any]:
        payload = get_vectorstore_manager().list_documents(limit=10000)
        docs = [
            Document(page_content=text or "", metadata=metadata or {})
            for text, metadata in zip(payload.get("documents") or [], payload.get("metadatas") or [])
            if text
        ]
        return {"chunks": len(docs), "documents": len(_unique_titles(docs)), "titles": _unique_titles(docs)}


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 秋招版检索评测报告", "",
        f"- 数据集：{payload['dataset']}，{payload['sample_count']} 条文档级 Gold；不伪造 chunk Gold。",
        f"- 语料：{payload['corpus']['documents']} 个文档，{payload['corpus']['chunks']} 个 chunks。",
        "- ACL violation 按返回结果逐条调用统一 check_doc_access；paraphrase 只用于稳定性，不改变 Gold。", "",
        "| Variant | Hit@1 | Hit@5 | MRR@5 | NDCG@5 | Coverage@5 | Top1 Drift | Regression | ACL | P50/P95(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["aggregate"]:
        item = payload["aggregate"][variant]
        lines.append(
            f"| {payload['variant_labels'][variant]} | {item['hit_at_1']:.3f} | {item['hit_at_5']:.3f} | "
            f"{item['mrr_at_5']:.3f} | {item['ndcg_at_5']:.3f} | {item['coverage_at_5']:.3f} | "
            f"{item['top1_drift_rate']:.3f} | {item['paraphrase_regression_rate']:.3f} | "
            f"{item['acl_violation']} | {item['latency_p50_ms']:.0f}/{item['latency_p95_ms']:.0f} |"
        )
    lines.extend(["", "## 分类别结果", ""])
    categories = sorted({category for item in payload["aggregate"].values() for category in item["category_metrics"]})
    lines.append("| 类别 | " + " | ".join(payload["variant_labels"][variant] + " MRR" for variant in payload["aggregate"]) + " |")
    lines.append("|---|" + "---:|" * len(payload["aggregate"]))
    for category in categories:
        values = [f"{payload['aggregate'][variant]['category_metrics'][category]['mrr_at_5']:.3f}" for variant in payload["aggregate"]]
        lines.append(f"| {category} | " + " | ".join(values) + " |")
    lines.extend(["", "## 结论", "", "- Dense 与 Hybrid + RRF 的差异用于说明精确技术词和多来源查询是否需要词法信号；结论只适用于本次冻结语料。", "- Reranker 只有在 Hit@1/MRR 稳定提升且延迟可接受时才值得保留；若结果未提升，不因模型名义上更复杂而默认上线。", "- Current Final 按规则选择性触发 Standalone Rewrite/Query Expansion；paraphrase 指标用于观察稳定性，不改变主查询 Gold。", "- 本报告只记录当前索引快照结果，不将一次运行外推为线上质量保证。"])
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="秋招版 Retrieval Benchmark")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    invalid = set(variants) - set(VARIANTS)
    if invalid:
        raise SystemExit(f"未知 Variant: {sorted(invalid)}")
    settings = get_settings()
    for key, value in (("HTTP_PROXY", settings.http_proxy), ("HTTPS_PROXY", settings.https_proxy)):
        if value:
            os.environ[key] = value
            os.environ[key.lower()] = value
    payload = await BenchmarkRunner(args.top_k).run(variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(render_report(payload))


if __name__ == "__main__":
    asyncio.run(main())
