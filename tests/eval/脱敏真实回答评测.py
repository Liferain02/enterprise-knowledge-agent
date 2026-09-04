"""P4 小规模真实回答评测：Normal、Deep 与选择性路由。

不改线上路由。每条样本只实际执行 Normal 与 Deep 各一次，Selective 复用其中一份结果。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.agent.agents.research_team import is_complex_research_task
from src.rag.retrieval.acl_filter import UserContext
from tests.eval.complex_research_dataset import COMPLEX_RESEARCH_DATASET, ComplexResearchQuery
from tests.eval.run_research_team_eval import aggregate, evaluate_case


OUTPUT = Path("data/脱敏真实回答评测结果.json")
REPORT = Path("data/脱敏真实回答评测报告.md")
CHECKPOINT = Path("data/脱敏真实回答评测断点.json")


def _normal_cases() -> list[ComplexResearchQuery]:
    return [
        ComplexResearchQuery("N01", "实验室组会每周什么时候举行？", "normal", ["实验室组会制度与汇报要求"], ["组会", "每周"]),
        ComplexResearchQuery("N02", "请解释一下 NUMA 是什么。", "normal", ["实验室研究方向与课题地图"], ["NUMA"]),
        ComplexResearchQuery("N03", "请列出新人入组第一天要做的事。", "normal", ["新生入组第一周任务清单"], ["新人", "第一天"]),
    ]


def selected_cases(limit: int) -> list[ComplexResearchQuery]:
    wanted = ["C01", "C17", "C33"]
    complex_cases = [case for case in COMPLEX_RESEARCH_DATASET if case.case_id in wanted]
    cases = complex_cases + _normal_cases()
    return cases[:limit] if limit > 0 else cases


def _variant_metrics(results: list[dict], key: str) -> dict:
    rows = [{"case": row["case"], "variants": {key: row["variants"][key]}} for row in results]
    metrics = aggregate(rows, [key])[key]
    metrics["label"] = {"A": "All Normal", "C": "All Deep", "S": "Selective Deep"}[key]
    return metrics


def _render(payload: dict) -> str:
    lines = [
        "# P4 脱敏真实回答评测报告", "",
        f"- 运行时间：{payload['generated_at']}",
        f"- 样本数：{payload['case_count']}",
        "- 说明：Selective Deep 复用同一条 Normal/Deep 实际运行结果，不重复调用模型。", "",
        "| 方案 | 成功 | 关键词覆盖 | 文档召回 | 引用支持 proxy | 前提总体 | ACL 泄漏 | P50/P95(ms) | 估计调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("A", "C", "S"):
        item = payload["aggregate"][key]
        premise = item.get("premise_overall_accuracy")
        lines.append(
            f"| {item['label']} | {item['success_cases']}/{item['cases']} | "
            f"{item['avg_keyword_correctness']:.3f} | {item['avg_retrieved_doc_recall']:.3f} | "
            f"{item['avg_citation_support_rate']:.3f} | "
            f"{'—' if premise is None else f'{premise:.3f}'} | {item['acl_leak_count']} | "
            f"{item['latency_p50_ms']:.0f}/{item['latency_p95_ms']:.0f} | "
            f"{item['avg_logical_api_calls_estimate']:.2f} |"
        )
    lines.extend(["", "## 解释", "", "自动指标是回归 proxy，不能替代人工评分。若 Deep 的复杂样本收益不稳定而延迟/调用显著增加，停止自动路由上线，保留显式 Deep。", ""])
    return "\n".join(lines)


async def run(limit: int = 6, checkpoint: Path = CHECKPOINT) -> dict:
    cases = selected_cases(limit)
    user = UserContext.anonymous()
    existing = {}
    if checkpoint.exists():
        existing = {row["case"]["case_id"]: row for row in json.loads(checkpoint.read_text(encoding="utf-8")).get("results", [])}
    results = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        if case.case_id in existing and all(k in existing[case.case_id].get("variants", {}) for k in ("A", "C")):
            row = existing[case.case_id]
            print(f"[{index}/{len(cases)}] {case.case_id} 从断点恢复", flush=True)
        else:
            print(f"[{index}/{len(cases)}] {case.case_id} {case.category}", flush=True)
            row = await evaluate_case(case, ("A", "C"), user)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"status": "in_progress", "results": results + [row]}, ensure_ascii=False, indent=2), encoding="utf-8")
        row["variants"]["S"] = row["variants"]["C"] if is_complex_research_task(case.query) else row["variants"]["A"]
        results.append(row)
    payload = {
        "dataset": "p4_deidentified_real_answer_representative",
        "case_count": len(results),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_duration_seconds": round(time.perf_counter() - started, 3),
        "variants": ["A", "C", "S"],
        "aggregate": {key: _variant_metrics(results, key) for key in ("A", "C", "S")},
        "results": results,
        "methodology_note": "A=All Normal，C=All Deep，S=确定性选择性；自动 proxy 不等同人工事实评分。",
    }
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = await run(args.limit, args.checkpoint)
    print(_render(payload))
    if args.write:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT.write_text(_render(payload) + "\n", encoding="utf-8")
        args.checkpoint.unlink(missing_ok=True)
        print(f"JSON: {OUTPUT}\nMarkdown: {REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
