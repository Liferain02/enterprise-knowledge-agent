#!/usr/bin/env python
"""在冻结多轮数据集上比较原追问与 Standalone 多查询检索。

该脚本只评估检索，不调用回答模型。准入阈值写死在脚本中，避免看到结果后
移动门槛；结果与报告写入 gitignore 中的 data/，不污染可复现代码。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.rag.retrieval.acl_filter import UserContext, check_doc_access
from src.rag.retrieval.query_expander import (
    StandaloneQueryRewriter,
    multi_query_retrieve,
)
from src.rag.retrieval.retriever import get_retriever_manager
from tests.eval.multi_turn_coreference_dataset import (
    DATASET_NAME,
    DATASET_VERSION,
    MULTI_TURN_COREFERENCE_CASES,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "多轮指代改写消融结果.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "多轮指代改写消融报告.md"

# 预先声明的最小准入标准。任何一项失败都不默认接入。
GATE = {
    "trigger_f1_min": 1.0,
    "positive_hit_at_5_delta_min": 0.0,
    "positive_mrr_at_5_delta_min": 0.03,
    "positive_regression_rate_max": 0.0,
    "negative_changed_rate_max": 0.0,
    "acl_violations_max": 0,
    "max_query_variants": 2,
}


def _source(doc: Any) -> str:
    return Path(str((doc.metadata or {}).get("source") or "")).name


def _unique_sources(docs: list[Any], limit: int) -> list[str]:
    sources: list[str] = []
    for doc in docs:
        source = _source(doc)
        if source and source not in sources:
            sources.append(source)
        if len(sources) >= limit:
            break
    return sources


def _first_gold_rank(sources: list[str], gold: tuple[str, ...]) -> int | None:
    return next(
        (rank for rank, source in enumerate(sources, 1) if source in gold),
        None,
    )


def _ndcg_at_k(sources: list[str], gold: tuple[str, ...], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, source in enumerate(sources[:k], 1)
        if source in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _metrics(sources: list[str], gold: tuple[str, ...], k: int) -> dict[str, float | int | None]:
    rank = _first_gold_rank(sources[:k], gold)
    return {
        "first_gold_rank": rank,
        "hit_at_1": int(rank == 1),
        "hit_at_5": int(rank is not None),
        "mrr_at_5": 0.0 if rank is None else 1.0 / rank,
        "ndcg_at_5": _ndcg_at_k(sources, gold, k),
    }


def _mean(rows: list[dict[str, Any]], variant: str, metric: str) -> float:
    if not rows:
        return 0.0
    return statistics.mean(float(row[variant][metric]) for row in rows)


def _classification_counts(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(row["should_rewrite"] and row["triggered"] for row in rows)
    fp = sum(not row["should_rewrite"] and row["triggered"] for row in rows)
    fn = sum(row["should_rewrite"] and not row["triggered"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


async def run(top_k: int = 5) -> dict[str, Any]:
    user = UserContext.anonymous()
    retriever = get_retriever_manager()
    rows: list[dict[str, Any]] = []
    acl_violations = 0
    max_variants = 0

    for index, case in enumerate(MULTI_TURN_COREFERENCE_CASES, 1):
        print(f"[{index}/{len(MULTI_TURN_COREFERENCE_CASES)}] {case.case_id}", flush=True)
        rewrite = StandaloneQueryRewriter.rewrite(
            case.followup_query,
            recent_messages=[
                {"role": "user", "content": case.previous_user_query},
                {"role": "user", "content": case.followup_query},
            ],
        )
        max_variants = max(max_variants, len(rewrite.variants))

        started = time.perf_counter()
        baseline_results = retriever.search_with_rerank(
            case.followup_query, k=top_k, user=user
        )
        baseline_latency = (time.perf_counter() - started) * 1000
        baseline_docs = [doc for doc, _score in baseline_results]

        if rewrite.triggered:
            started = time.perf_counter()
            fused_results = await multi_query_retrieve(
                [variant.text for variant in rewrite.variants],
                top_k_per_query=top_k,
                use_reranker=True,
                user=user,
            )
            standalone_latency = (time.perf_counter() - started) * 1000
            standalone_docs = [doc for doc, _score, _source in fused_results[:top_k]]
        else:
            standalone_latency = baseline_latency
            standalone_docs = list(baseline_docs)

        for doc in baseline_docs + standalone_docs:
            if not check_doc_access(doc.metadata or {}, user):
                acl_violations += 1

        baseline_sources = _unique_sources(baseline_docs, top_k)
        standalone_sources = _unique_sources(standalone_docs, top_k)
        rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "previous_user_query": case.previous_user_query,
            "followup_query": case.followup_query,
            "should_rewrite": case.should_rewrite,
            "triggered": rewrite.triggered,
            "reason_code": rewrite.reason_code,
            "queries": [variant.text for variant in rewrite.variants],
            "gold_sources": list(case.gold_sources),
            "baseline": {
                "sources": baseline_sources,
                "latency_ms": round(baseline_latency, 3),
                **_metrics(baseline_sources, case.gold_sources, top_k),
            },
            "standalone": {
                "sources": standalone_sources,
                "latency_ms": round(standalone_latency, 3),
                **_metrics(standalone_sources, case.gold_sources, top_k),
            },
        })

    positives = [row for row in rows if row["should_rewrite"]]
    negatives = [row for row in rows if not row["should_rewrite"]]
    classification = _classification_counts(rows)
    positive_regressions = sum(
        row["baseline"]["hit_at_5"] > row["standalone"]["hit_at_5"]
        for row in positives
    )
    negative_changes = sum(
        row["baseline"]["sources"] != row["standalone"]["sources"]
        for row in negatives
    )
    aggregate = {
        "trigger": classification,
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "positive": {
            variant: {
                metric: _mean(positives, variant, metric)
                for metric in ("hit_at_1", "hit_at_5", "mrr_at_5", "ndcg_at_5", "latency_ms")
            }
            for variant in ("baseline", "standalone")
        },
        "all": {
            variant: {
                metric: _mean(rows, variant, metric)
                for metric in ("hit_at_1", "hit_at_5", "mrr_at_5", "ndcg_at_5", "latency_ms")
            }
            for variant in ("baseline", "standalone")
        },
        "positive_regression_rate": positive_regressions / len(positives),
        "negative_changed_rate": negative_changes / len(negatives),
        "acl_violations": acl_violations,
        "max_query_variants": max_variants,
    }
    deltas = {
        metric: aggregate["positive"]["standalone"][metric] - aggregate["positive"]["baseline"][metric]
        for metric in ("hit_at_1", "hit_at_5", "mrr_at_5", "ndcg_at_5", "latency_ms")
    }
    checks = {
        "trigger_f1": classification["f1"] >= GATE["trigger_f1_min"],
        "positive_hit_at_5": deltas["hit_at_5"] >= GATE["positive_hit_at_5_delta_min"],
        "positive_mrr_at_5": deltas["mrr_at_5"] >= GATE["positive_mrr_at_5_delta_min"],
        "positive_regression_rate": aggregate["positive_regression_rate"] <= GATE["positive_regression_rate_max"],
        "negative_changed_rate": aggregate["negative_changed_rate"] <= GATE["negative_changed_rate_max"],
        "acl_violations": acl_violations <= GATE["acl_violations_max"],
        "max_query_variants": max_variants <= GATE["max_query_variants"],
    }
    settings = get_settings()
    return {
        "dataset": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "configuration": {
            "top_k": top_k,
            "embedding_model": settings.embedding_model,
            "reranker_provider": settings.reranker_provider,
            "reranker_model": settings.reranker_model,
            "user_role": user.role,
        },
        "gate": GATE,
        "aggregate": aggregate,
        "positive_deltas": deltas,
        "checks": checks,
        "passed": all(checks.values()),
        "results": rows,
    }


def render_report(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    positive = aggregate["positive"]
    deltas = payload["positive_deltas"]
    lines = [
        "# 多轮指代改写检索消融报告", "",
        f"- 数据集：{payload['dataset']}（{payload['dataset_version']}），共 {len(payload['results'])} 条",
        f"- 结论：{'通过准入，可接入主链路' if payload['passed'] else '未通过准入，不应默认接入'}",
        "- Baseline 只检索用户当前追问；Standalone 同时检索原追问和带最近用户问题的独立查询，并以 RRF 合并。", "",
        "| 正例指标 | Baseline | Standalone | 变化 |", "|---|---:|---:|---:|",
    ]
    for metric in ("hit_at_1", "hit_at_5", "mrr_at_5", "ndcg_at_5", "latency_ms"):
        lines.append(
            f"| {metric} | {positive['baseline'][metric]:.3f} | "
            f"{positive['standalone'][metric]:.3f} | {deltas[metric]:+.3f} |"
        )
    trigger = aggregate["trigger"]
    lines.extend([
        "", "## 安全与触发", "",
        f"- 触发 Precision / Recall / F1：{trigger['precision']:.3f} / {trigger['recall']:.3f} / {trigger['f1']:.3f}",
        f"- 正例 Hit@5 回退率：{aggregate['positive_regression_rate']:.3f}",
        f"- 负例结果变化率：{aggregate['negative_changed_rate']:.3f}",
        f"- ACL 违规结果数：{aggregate['acl_violations']}",
        f"- 单请求最多查询数：{aggregate['max_query_variants']}",
        "", "## 准入检查", "",
    ])
    for name, passed in payload["checks"].items():
        lines.append(f"- {'通过' if passed else '失败'}：{name}")
    failed = [row for row in payload["results"] if row["standalone"]["hit_at_5"] == 0]
    if failed:
        lines.extend(["", "## Standalone 仍未命中的正例", ""])
        for row in failed:
            if row["should_rewrite"]:
                lines.append(f"- {row['case_id']}：{row['followup_query']}")
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="多轮 Standalone 检索消融")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    settings = get_settings()
    for key, value in (("HTTP_PROXY", settings.http_proxy), ("HTTPS_PROXY", settings.https_proxy)):
        if value:
            os.environ[key] = value
            os.environ[key.lower()] = value

    payload = await run(top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(render_report(payload))


if __name__ == "__main__":
    asyncio.run(main())
