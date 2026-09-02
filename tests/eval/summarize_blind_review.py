#!/usr/bin/env python
"""在独立评分锁定后解盲并汇总；不会修改冻结 V2 产物。"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.eval.blind_review_template import CandidateScore
from tests.eval.deep_research_v2_blind_dataset import BLIND_RESEARCH_V2_DATASET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES = PROJECT_ROOT / "data" / "深度研究V2人工盲评评分.json"
DEFAULT_MAPPING = PROJECT_ROOT / "data" / "深度研究V2盲评密封映射.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "深度研究V2人工盲评汇总.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "深度研究V2人工盲评汇总报告.md"
MODEL_SCORES = PROJECT_ROOT / "data" / "深度研究V2模型模拟盲评评分.json"
MODEL_OUTPUT = PROJECT_ROOT / "data" / "深度研究V2模型模拟盲评汇总.json"
MODEL_REPORT = PROJECT_ROOT / "data" / "深度研究V2模型模拟盲评汇总报告.md"


def _validate(scores: dict[str, Any]) -> None:
    expected_ids = {case.case_id for case in BLIND_RESEARCH_V2_DATASET}
    if set(scores) != expected_ids:
        missing = sorted(expected_ids - set(scores))
        extra = sorted(set(scores) - expected_ids)
        raise ValueError(f"评分 case 不完整：missing={missing}, extra={extra}")
    for case_id, row in scores.items():
        if set(row.get("candidates", {})) != {"候选甲", "候选乙"}:
            raise ValueError(f"{case_id} 必须同时评分候选甲和候选乙")
        for alias, values in row["candidates"].items():
            CandidateScore(
                correctness=values.get("correctness"),
                completeness=values.get("completeness"),
                evidence=values.get("evidence"),
            ).validate()
        if row.get("winner") not in ("候选甲", "候选乙", "平局"):
            raise ValueError(f"{case_id} winner 必须为候选甲、候选乙或平局")
        if not str(row.get("reason", "")).strip():
            raise ValueError(f"{case_id} 缺少简短理由")


def summarize(
    scores: dict[str, Any],
    mapping: dict[str, Any],
    *,
    review_kind: str = "human",
) -> dict[str, Any]:
    _validate(scores)
    expected_ids = {case.case_id for case in BLIND_RESEARCH_V2_DATASET}
    if set(mapping) != expected_ids:
        raise ValueError("密封映射与冻结数据集不一致")
    category_by_id = {case.case_id: case.category for case in BLIND_RESEARCH_V2_DATASET}
    totals = defaultdict(list)
    wins = Counter()
    category_totals: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    resolved = []
    for case_id, row in scores.items():
        alias_to_variant = mapping[case_id]
        variant_scores = {}
        for alias, values in row["candidates"].items():
            score = CandidateScore(**values)
            variant = alias_to_variant[alias]
            totals[variant].append(score.total)
            category_totals[category_by_id[case_id]][variant].append(score.total)
            variant_scores[variant] = {**values, "total": score.total}
        winner = row["winner"]
        winning_variant = "tie" if winner == "平局" else alias_to_variant[winner]
        wins[winning_variant] += 1
        resolved.append({
            "case_id": case_id,
            "category": category_by_id[case_id],
            "scores": variant_scores,
            "winner": winning_variant,
            "reason": row["reason"],
        })
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": (
            "independent_human_review_complete"
            if review_kind == "human"
            else "model_simulated_blind_review_complete"
        ),
        "review_kind": review_kind,
        "cases": len(resolved),
        "average_total": {
            variant: sum(values) / len(values) for variant, values in totals.items()
        },
        "wins": dict(wins),
        "category_average_total": {
            category: {
                variant: sum(values) / len(values)
                for variant, values in variants.items()
            }
            for category, variants in category_totals.items()
        },
        "results": resolved,
    }


def load_review(
    scores_path: Path,
    mapping_path: Path,
    *,
    review_kind: str = "human",
) -> dict[str, Any]:
    """评分完整锁定后才读取密封映射。"""

    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    _validate(scores)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return summarize(scores, mapping, review_kind=review_kind)


def render(payload: dict[str, Any]) -> str:
    is_human = payload.get("review_kind", "human") == "human"
    title = "深度研究 V2 独立人工盲评汇总" if is_human else "深度研究 V2 模型模拟盲评汇总"
    lines = [
        f"# {title}", "",
        f"- 评审性质：{'独立人工评分' if is_human else '模型模拟评分，不替代独立人工评分'}",
        f"- 完成时间：{payload['generated_at']}",
        f"- 样本数：{payload['cases']}",
        f"- 胜场：{json.dumps(payload['wins'], ensure_ascii=False)}", "",
        "| 方案 | 平均总分（满分 6） |", "|---|---:|",
    ]
    for variant, value in sorted(payload["average_total"].items()):
        lines.append(f"| {variant} | {value:.3f} |")
    lines.extend(["", "## 分类别平均总分", ""])
    for category, variants in sorted(payload["category_average_total"].items()):
        lines.append(
            f"- {category}：" + "；".join(
                f"{variant}={value:.3f}" for variant, value in sorted(variants.items())
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 Deep Research V2 盲评")
    parser.add_argument("--review-kind", choices=("human", "model"), default="human")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--create-template", action="store_true")
    args = parser.parse_args()
    is_human = args.review_kind == "human"
    args.scores = args.scores or (DEFAULT_SCORES if is_human else MODEL_SCORES)
    args.output = args.output or (DEFAULT_OUTPUT if is_human else MODEL_OUTPUT)
    args.report = args.report or (DEFAULT_REPORT if is_human else MODEL_REPORT)
    if args.create_template:
        template = {
            case.case_id: {
                "candidates": {
                    "候选甲": {"correctness": None, "completeness": None, "evidence": None},
                    "候选乙": {"correctness": None, "completeness": None, "evidence": None},
                },
                "winner": "",
                "reason": "",
            }
            for case in BLIND_RESEARCH_V2_DATASET
        }
        args.scores.write_text(
            json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已创建盲评评分模板：{args.scores}")
        return
    try:
        payload = load_review(args.scores, args.mapping, review_kind=args.review_kind)
    except ValueError as exc:
        raise SystemExit(f"盲评未就绪：{exc}") from None
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render(payload), encoding="utf-8")
    print(render(payload))


if __name__ == "__main__":
    main()
