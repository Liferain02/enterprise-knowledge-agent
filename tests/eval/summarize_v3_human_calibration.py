#!/usr/bin/env python
"""汇总 V3 的 20% 独立人工标签并校准 Qwen Claim Judge。"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_ROOT / "data" / "深度研究V3声明级因果消融结果.json"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "深度研究V3人工校准评分.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "深度研究V3人工校准汇总.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "深度研究V3人工校准报告.md"
EXPECTED_CASES = {"V301", "V306", "V311", "V316"}


def _prediction_map(results: Dict[str, Any]) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    for row in results.get("results") or []:
        case_id = row["case"]["case_id"]
        if case_id not in EXPECTED_CASES:
            continue
        for variant_name, variant in row["variants"].items():
            evaluation = variant.get("claim_evaluation") or {}
            for axis in (
                "response_to_ground_truth",
                "ground_truth_to_response",
                "response_to_context",
            ):
                for item in evaluation.get(axis) or []:
                    task_id = f"{case_id}:{variant_name}:{axis}:{item['item_id']}"
                    predictions[task_id] = (
                        "supported" if item["verdict"] == "supported" else "not_supported"
                    )
    return predictions


def _cohen_kappa(pairs: Iterable[tuple[str, str]]) -> float:
    items = list(pairs)
    if not items:
        return 0.0
    labels = ("supported", "not_supported")
    observed = sum(left == right for left, right in items) / len(items)
    left_counts = Counter(left for left, _ in items)
    right_counts = Counter(right for _, right in items)
    expected = sum(
        left_counts[label] / len(items) * right_counts[label] / len(items)
        for label in labels
    )
    return (observed - expected) / (1 - expected) if not math.isclose(expected, 1.0) else 1.0


def summarize(results: Dict[str, Any], labels: Dict[str, Any]) -> Dict[str, Any]:
    if not str(labels.get("reviewer_id") or "").strip():
        raise ValueError("缺少独立人工 reviewer_id")
    if labels.get("independence_attestation") is not True:
        raise ValueError("独立评分声明尚未确认")
    selected = set(labels.get("selected_case_ids") or [])
    if selected != EXPECTED_CASES or labels.get("calibration_fraction") != 0.20:
        raise ValueError("人工校准必须覆盖预先选定的 4/20 cases")

    predictions = _prediction_map(results)
    plan = results.get("human_calibration_plan") or {}
    planned_task_ids = set(plan.get("task_ids") or [])
    tasks = labels.get("tasks") or []
    task_ids = {task["task_id"] for task in tasks}
    if not tasks or not planned_task_ids or task_ids != planned_task_ids:
        raise ValueError("人工任务与冻结的分层抽样计划不完整或不一致")
    if not task_ids <= set(predictions):
        raise ValueError("人工任务包含不存在的 Qwen Judge 项目")

    pairs: list[tuple[str, str]] = []
    by_axis: Dict[str, list[tuple[str, str]]] = defaultdict(list)
    confusion = Counter()
    for task in tasks:
        human = task.get("human_verdict")
        if human not in {"supported", "not_supported"}:
            raise ValueError(f"{task['task_id']} 尚未填写合法 human_verdict")
        predicted = predictions[task["task_id"]]
        pairs.append((predicted, human))
        by_axis[task["axis"]].append((predicted, human))
        confusion[f"qwen_{predicted}__human_{human}"] += 1

    agreement = sum(left == right for left, right in pairs) / len(pairs)
    kappa = _cohen_kappa(pairs)
    calibration_passed = agreement >= 0.80 and kappa >= 0.60
    preliminary = results.get("decision") or {}
    quality_passed = preliminary.get("quality_gate_preliminary") is True
    cost_passed = preliminary.get("cost_gate_preliminary") is True
    keep_gate = calibration_passed and quality_passed and cost_passed
    return {
        "status": "independent_human_calibration_complete",
        "reviewer_id": labels["reviewer_id"],
        "selected_cases": sorted(selected),
        "calibration_fraction": 0.20,
        "labels": len(pairs),
        "agreement": round(agreement, 6),
        "cohen_kappa": round(kappa, 6),
        "agreement_by_axis": {
            axis: round(sum(a == b for a, b in items) / len(items), 6)
            for axis, items in by_axis.items()
        },
        "confusion": dict(confusion),
        "thresholds": {"agreement": 0.80, "cohen_kappa": 0.60},
        "qwen_judge_calibration_passed": calibration_passed,
        "quality_gate_passed": quality_passed,
        "cost_gate_passed": cost_passed,
        "actionability_gate_final_decision": (
            "keep_actionability_gate" if keep_gate else "revert_actionability_gate"
        ),
    }


def _render_report(summary: Dict[str, Any]) -> str:
    return "\n".join([
        "# 深度研究 V3 人工校准报告", "",
        f"- 状态：{summary['status']}",
        f"- 校准样本：{len(summary['selected_cases'])}/20 cases，{summary['labels']} 个蕴含标签",
        f"- Qwen/人工一致率：{summary['agreement']:.3f}",
        f"- Cohen's kappa：{summary['cohen_kappa']:.3f}",
        f"- Judge 校准门槛：{'通过' if summary['qwen_judge_calibration_passed'] else '未通过'}",
        f"- 质量门槛：{'通过' if summary['quality_gate_passed'] else '未通过'}",
        f"- 成本门槛：{'通过' if summary['cost_gate_passed'] else '未通过'}",
        f"- 最终决策：`{summary['actionability_gate_final_decision']}`", "",
        "只有 Judge 校准、质量不下降和成本明确降低三项同时通过，才保留 actionability gate。",
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 V3 独立人工 Judge 校准")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    summary = summarize(results, labels)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
