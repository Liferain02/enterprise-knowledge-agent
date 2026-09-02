#!/usr/bin/env python
"""运行 Deep Research V2 retrieval-only Development Benchmark。

比较对象：
- normal：当前 normal 使用的原问题 + Query Expansion + Hybrid/Rerank；
- deep_researcher：Researcher 实际生成的有界检索查询 + 相同 Hybrid/Rerank。

该脚本不调用 Analyst、Reviewer 或最终 Generation，从而把检索失败与答案失败分开。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agents.research_team import MAX_SUBQUESTIONS, _plan_subquestions
from src.agent.agents.knowledge import _retrieve_documents
from src.rag.retrieval.acl_filter import UserContext, check_doc_access
from src.rag.retrieval.query_expander import RuleBasedDecomposer
from src.rag.retrieval.retriever import get_retriever_manager
from src.rag.storage.vectorstore import get_vectorstore_manager
from tests.eval.deep_research_retrieval_dev_dataset import (
    RETRIEVAL_DEV_DATASET,
    RetrievalDevQuery,
)


VARIANTS = ("normal", "deep_researcher")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "深度研究V2检索开发结果.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "深度研究V2检索开发报告.md"


def _title(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("title") or Path(str(metadata.get("source") or "")).stem)


def _matches(actual_title: str, expected_id: str) -> bool:
    return expected_id.casefold() in actual_title.casefold()


def _matched_expected(titles: Iterable[str], expected: list[str]) -> dict[str, int | None]:
    title_list = list(titles)
    ranks: dict[str, int | None] = {}
    for expected_id in expected:
        ranks[expected_id] = next(
            (rank for rank, title in enumerate(title_list, 1) if _matches(title, expected_id)),
            None,
        )
    return ranks


def _load_corpus() -> tuple[list[dict[str, Any]], list[str]]:
    payload = get_vectorstore_manager().list_documents(limit=10000)
    rows: list[dict[str, Any]] = []
    titles: list[str] = []
    for metadata, text in zip(payload.get("metadatas") or [], payload.get("documents") or []):
        title = str(metadata.get("title") or Path(str(metadata.get("source") or "")).stem)
        rows.append({"title": title, "metadata": metadata, "text": text})
        if title not in titles:
            titles.append(title)
    return rows, titles


def _expected_in_corpus(expected_id: str, corpus_titles: list[str]) -> bool:
    return any(_matches(title, expected_id) for title in corpus_titles)


def _expected_acl_allowed(expected_id: str, corpus_rows: list[dict[str, Any]], user: UserContext) -> bool:
    matched = [row for row in corpus_rows if _matches(row["title"], expected_id)]
    return bool(matched) and any(check_doc_access(row["metadata"], user) for row in matched)


async def _pipeline_retrieve(query: str, user: UserContext, *, expansion: bool) -> dict[str, Any]:
    manager = get_retriever_manager()
    candidate_k = 15
    raw_candidates = manager.search_with_score_acl(
        query, k=candidate_k, user=user,
    )
    started = time.perf_counter()
    results, grade, history = await _retrieve_documents(query, 5, expansion, user)
    reranked = results
    final_titles = list(dict.fromkeys(_title(doc) for doc, _score in results))
    return {
        "query": query,
        "titles": final_titles,
        "scores": [float(score) for _doc, score in results],
        "raw_candidate_titles": list(dict.fromkeys(_title(doc) for doc, _score in raw_candidates)),
        "reranked_titles": list(dict.fromkeys(_title(doc) for doc, _score in reranked)),
        "rewrite_history": list(history),
        "decision": getattr(getattr(grade, "decision", None), "value", None)
        or ("high" if results else "no_results"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _dedupe_titles(query_runs: list[dict[str, Any]], limit: int = 12) -> list[str]:
    # Researcher 的 EvidencePackage 按各查询结果出现顺序去重；评测只关心文档级排名。
    titles: list[str] = []
    for run in query_runs:
        for title in run["titles"]:
            if title not in titles:
                titles.append(title)
            if len(titles) >= limit:
                return titles
    return titles


def _diagnose(
    case: RetrievalDevQuery,
    variant: str,
    ranks: dict[str, int | None],
    corpus_rows: list[dict[str, Any]],
    corpus_titles: list[str],
    user: UserContext,
    query_runs: list[dict[str, Any]],
    other_variant_titles: list[str] | None = None,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    all_queries = [run["query"] for run in query_runs]
    all_histories = [item for run in query_runs for item in run["rewrite_history"]]
    for expected_id, rank in ranks.items():
        if rank is not None:
            continue
        if not _expected_in_corpus(expected_id, corpus_titles):
            failure_type = "missing_document"
            reason = "目标资料不在当前向量库 snapshot 中"
        elif not _expected_acl_allowed(expected_id, corpus_rows, user):
            failure_type = "ACL_filtering"
            reason = "目标资料存在，但当前用户上下文无权访问"
        elif variant == "deep_researcher" and other_variant_titles and any(
            _matches(title, expected_id) for title in other_variant_titles
        ):
            failure_type = "query_planning_failure"
            reason = "normal 找到目标资料，但 Researcher 查询集合未保住召回"
        elif any(
            _matches(title, expected_id)
            for run in query_runs for title in run.get("raw_candidate_titles", [])
        ) and not any(
            _matches(title, expected_id)
            for run in query_runs for title in run.get("reranked_titles", [])
        ):
            failure_type = "rerank_failure"
            reason = "目标资料进入原始候选，但未进入 rerank 后的评估集合"
        elif any(
            _matches(title, expected_id)
            for run in query_runs for title in run.get("reranked_titles", [])
        ):
            failure_type = "unknown"
            reason = "目标资料通过 rerank，但未进入文档级最终排名；需检查合并与截断"
        elif case.category in ("exact_named_document", "named_entity_alias"):
            failure_type = "lexical/title_failure"
            reason = "目标资料已入库且可见，但命名/别名查询未召回到最终结果"
        else:
            failure_type = "semantic_retrieval_failure"
            reason = "目标资料已入库且可见，但语义或多证据查询未召回"
        # 只有 raw candidates 可观测时才严格归因 rerank_failure，避免从最终结果猜测。
        if any(expected_id.casefold() in item.casefold() for item in all_histories) and failure_type == "lexical/title_failure":
            reason += "；改写历史已保留名称，仍未命中"
        failures.append({"expected_doc": expected_id, "failure_type": failure_type, "reason": reason})
    return failures


def _metrics(titles: list[str], expected: list[str], latency_ms: float, query_count: int) -> dict[str, Any]:
    ranks = _matched_expected(titles, expected)
    hit_ranks = [rank for rank in ranks.values() if rank is not None]
    coverage = len(hit_ranks) / len(expected) if expected else 1.0
    first_rank = min(hit_ranks) if hit_ranks else None
    return {
        "hit_at_k": 1 if hit_ranks else 0,
        "recall_at_k": coverage,
        "expected_document_coverage": coverage,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "ranks": ranks,
        "query_count": query_count,
        "latency_ms": round(latency_ms, 3),
    }


async def _evaluate_case(
    case: RetrievalDevQuery,
    user: UserContext,
    corpus_rows: list[dict[str, Any]],
    corpus_titles: list[str],
) -> dict[str, Any]:
    normal_run = await _pipeline_retrieve(
        case.query,
        user,
        expansion=RuleBasedDecomposer.needs_expansion(case.query),
    )
    normal_titles = normal_run["titles"]
    normal_metrics = _metrics(normal_titles, case.relevant_doc_ids, normal_run["latency_ms"], 1)

    planning_started = time.perf_counter()
    subquestions, _usage = await _plan_subquestions(case.query)
    planning_latency = (time.perf_counter() - planning_started) * 1000
    deep_runs: list[dict[str, Any]] = []
    for subquestion in subquestions[:MAX_SUBQUESTIONS]:
        deep_runs.append(await _pipeline_retrieve(subquestion, user, expansion=False))
    deep_titles = _dedupe_titles(deep_runs)
    deep_latency = planning_latency + sum(run["latency_ms"] for run in deep_runs)
    deep_metrics = _metrics(deep_titles, case.relevant_doc_ids, deep_latency, len(deep_runs))

    normal_metrics["failures"] = _diagnose(
        case, "normal", normal_metrics["ranks"], corpus_rows, corpus_titles, user,
        [normal_run], deep_titles,
    )
    deep_metrics["failures"] = _diagnose(
        case, "deep_researcher", deep_metrics["ranks"], corpus_rows, corpus_titles, user,
        deep_runs, normal_titles,
    )
    return {
        "case": asdict(case),
        "variants": {
            "normal": {**normal_metrics, "actual_docs": normal_titles, "queries": [case.query], "query_runs": [normal_run]},
            "deep_researcher": {**deep_metrics, "actual_docs": deep_titles, "queries": subquestions, "query_runs": deep_runs},
        },
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [row["variants"][variant] for row in results]
        failures = Counter(
            failure["failure_type"]
            for row in rows for failure in row["failures"]
        )
        latencies = [float(row["latency_ms"]) for row in rows]
        aggregate[variant] = {
            "cases": len(rows),
            "hit_at_k": statistics.mean(row["hit_at_k"] for row in rows),
            "recall_at_k": statistics.mean(row["recall_at_k"] for row in rows),
            "mrr": statistics.mean(row["mrr"] for row in rows),
            "expected_document_coverage": statistics.mean(row["expected_document_coverage"] for row in rows),
            "avg_query_count": statistics.mean(row["query_count"] for row in rows),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "failure_types": dict(sorted(failures.items())),
        }
    return aggregate


def _aggregate_by_category(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["case"]["category"]].append(row)
    return {category: _aggregate(rows) for category, rows in sorted(grouped.items())}


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Deep Research V2 检索开发评测报告",
        "",
        f"- 运行时间：{payload['generated_at']}",
        f"- 查询数：{payload['case_count']}",
        f"- 当前知识库：{payload['corpus']['chunk_count']} chunks / {payload['corpus']['document_count']} documents",
        "- 本报告只评价 Retrieval，不评价最终答案。",
        "",
        "| 方案 | Hit@K | Recall@K | MRR | 文档覆盖 | 平均查询数 | P50/P95(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = payload["aggregate"][variant]
        lines.append(
            f"| {variant} | {item['hit_at_k']:.3f} | {item['recall_at_k']:.3f} | "
            f"{item['mrr']:.3f} | {item['expected_document_coverage']:.3f} | "
            f"{item['avg_query_count']:.2f} | {item['latency_p50_ms']:.0f}/{item['latency_p95_ms']:.0f} |"
        )
    lines.extend(["", "## 按类别", ""])
    for category, variants in payload["aggregate_by_category"].items():
        normal, deep = variants["normal"], variants["deep_researcher"]
        lines.append(
            f"- {category}：Recall normal/deep={normal['recall_at_k']:.3f}/{deep['recall_at_k']:.3f}；"
            f"MRR={normal['mrr']:.3f}/{deep['mrr']:.3f}。"
        )
    lines.extend(["", "## 失败归因", ""])
    for variant in VARIANTS:
        lines.append(f"- {variant}：`{json.dumps(payload['aggregate'][variant]['failure_types'], ensure_ascii=False)}`")
    lines.extend(["", "## 逐题失败", ""])
    for row in payload["results"]:
        failures = []
        for variant in VARIANTS:
            for failure in row["variants"][variant]["failures"]:
                failures.append(f"{variant}:{failure['expected_doc']}={failure['failure_type']}")
        if failures:
            lines.append(f"- {row['case']['case_id']}：" + "；".join(failures))
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--label", default="development")
    args = parser.parse_args()

    from config.settings import get_settings
    settings = get_settings()
    if settings.http_proxy:
        os.environ["HTTP_PROXY"] = settings.http_proxy
        os.environ["http_proxy"] = settings.http_proxy
    if settings.https_proxy:
        os.environ["HTTPS_PROXY"] = settings.https_proxy
        os.environ["https_proxy"] = settings.https_proxy

    corpus_rows, corpus_titles = _load_corpus()
    cases = RETRIEVAL_DEV_DATASET[:args.limit or None]
    user = UserContext.anonymous()
    started = time.perf_counter()
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.case_id} {case.category}", flush=True)
        results.append(await _evaluate_case(case, user, corpus_rows, corpus_titles))

    payload = {
        "dataset": "deep_research_retrieval_development",
        "label": args.label,
        "case_count": len(results),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_duration_seconds": round(time.perf_counter() - started, 3),
        "configuration": {
            "git_commit": os.popen("git rev-parse HEAD").read().strip(),
            "llm_provider": settings.llm_provider,
            "llm_model": settings.dashscope_model,
            "embedding_model": settings.embedding_model,
            "reranker_provider": settings.reranker_provider,
            "reranker_model": settings.reranker_model,
            "top_k": 5,
        },
        "corpus": {
            "chunk_count": len(corpus_rows),
            "document_count": len(corpus_titles),
            "titles": corpus_titles,
        },
        "aggregate": _aggregate(results),
        "aggregate_by_category": _aggregate_by_category(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(_render_report(payload), encoding="utf-8")
    print(_render_report(payload))


if __name__ == "__main__":
    asyncio.run(main())
