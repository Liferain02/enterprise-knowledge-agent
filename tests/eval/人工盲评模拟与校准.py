"""对 P4 脱敏回答执行规则化的人工盲评模拟。

该脚本不调用 LLM。评分来自冻结关键点、引用和前提判断的可解释规则，
并显式标记为 simulated_human，不能替代真人评分。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


INPUT = Path("data/脱敏真实回答评测结果.json")
OUTPUT = Path("data/人工盲评模拟结果.json")
REPORT = Path("data/人工盲评模拟报告.md")


def _contains(answer: str, value: str) -> bool:
    return str(value or "").lower().replace(" ", "") in str(answer or "").lower().replace(" ", "")


def _score(answer: str, case: dict) -> dict:
    """按冻结关键点、引用和前提信号给出可复核的 0-2 评分。"""
    answer = answer or ""
    keywords = case.get("expected_keywords") or []
    keyword_rate = sum(_contains(answer, item) for item in keywords) / len(keywords) if keywords else 0.0
    correctness = 2 if keyword_rate >= 0.8 else 1 if keyword_rate >= 0.5 else 0

    expected_points = case.get("expected_key_points") or []
    point_rate = sum(_contains(answer, item) for item in expected_points) / len(expected_points) if expected_points else keyword_rate
    completeness = 2 if point_rate >= 0.67 else 1 if point_rate >= 0.34 else 0

    citations = len(re.findall(r"\[(?:文档\s*\d+|S\s*\d+)\]", answer))
    evidence = 2 if citations >= max(2, case.get("min_sources", 1)) else 1 if citations else 0
    if case.get("premise_expectation") == "false" and re.search(r"前提.{0,20}(不成立|不准确|不支持)|不能证明", answer):
        correctness = max(correctness, 2)
    if case.get("premise_expectation") == "supported" and re.search(r"前提.{0,20}(成立|支持|确认)", answer):
        correctness = max(correctness, 2)
    return {
        "correctness": correctness,
        "completeness": completeness,
        "evidence_basis": evidence,
        "total": correctness + completeness + evidence,
        "keyword_rate": round(keyword_rate, 4),
        "point_rate": round(point_rate, 4),
    }


def _build_blind_packet(payload: dict) -> list[dict]:
    rng = random.Random("p4-simulated-human-review-2026-09-04")
    packet = []
    for row in payload.get("results", []):
        case = row["case"]
        variants = ["A", "C"]
        rng.shuffle(variants)
        packet.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "question": case["query"],
            "expected_keywords": case.get("expected_keywords", []),
            "expected_key_points": case.get("expected_key_points", []),
            "candidate_order": ["候选甲", "候选乙"],
            "candidates": {
                "候选甲": {"answer": row["variants"][variants[0]]["answer"], "hidden_variant": variants[0]},
                "候选乙": {"answer": row["variants"][variants[1]]["answer"], "hidden_variant": variants[1]},
            },
        })
    return packet


def _review_packet(packet: list[dict]) -> list[dict]:
    reviewed = []
    for item in packet:
        scores = {}
        for alias, candidate in item["candidates"].items():
            scores[alias] = _score(candidate["answer"], item)
        winner = max(scores, key=lambda alias: scores[alias]["total"])
        if scores["候选甲"]["total"] == scores["候选乙"]["total"]:
            winner = "平局"
        reviewed.append({**item, "scores": scores, "winner": winner})
    return reviewed


def evaluate() -> dict:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    reviewed = _review_packet(_build_blind_packet(payload))
    by_variant = {"A": [], "C": []}
    for item in reviewed:
        for alias, candidate in item["candidates"].items():
            by_variant[candidate["hidden_variant"]].append(item["scores"][alias])
    aggregates = {}
    for variant, rows in by_variant.items():
        aggregates[variant] = {
            "cases": len(rows),
            "avg_total": round(sum(item["total"] for item in rows) / len(rows), 4) if rows else 0.0,
            "avg_correctness": round(sum(item["correctness"] for item in rows) / len(rows), 4) if rows else 0.0,
            "avg_completeness": round(sum(item["completeness"] for item in rows) / len(rows), 4) if rows else 0.0,
            "avg_evidence_basis": round(sum(item["evidence_basis"] for item in rows) / len(rows), 4) if rows else 0.0,
        }
    complex_ids = {"C01", "C17", "C33"}
    complex_scores = {"A": [], "C": []}
    for item in reviewed:
        if item["case_id"] not in complex_ids:
            continue
        for alias, candidate in item["candidates"].items():
            complex_scores[candidate["hidden_variant"]].append(item["scores"][alias])
    complex_agg = {
        variant: round(sum(item["total"] for item in rows) / len(rows), 4) if rows else 0.0
        for variant, rows in complex_scores.items()
    }
    delta = complex_agg["C"] - complex_agg["A"]
    evidence_delta = aggregates["C"]["avg_evidence_basis"] - aggregates["A"]["avg_evidence_basis"]
    public_reviews = []
    for item in reviewed:
        public_candidates = {
            alias: {"answer": candidate["answer"]}
            for alias, candidate in item["candidates"].items()
        }
        public_reviews.append({
            "case_id": item["case_id"],
            "category": item["category"],
            "question": item["question"],
            "expected_keywords": item["expected_keywords"],
            "expected_key_points": item["expected_key_points"],
            "candidate_order": item["candidate_order"],
            "candidates": public_candidates,
            "scores": item["scores"],
            "winner": item["winner"],
        })
    return {
        "dataset": "p4_simulated_blind_review",
        "review_kind": "simulated_human",
        "independent_human": False,
        "case_count": len(reviewed),
        "aggregate": aggregates,
        "complex_avg_total": complex_agg,
        "complex_total_delta_deep_minus_normal": round(delta, 4),
        "evidence_basis_delta_deep_minus_normal": round(evidence_delta, 4),
        "gate": {
            "passed": delta >= 0.5 and evidence_delta >= -0.2,
            "decision": "进入独立真人盲评" if delta >= 0.5 and evidence_delta >= -0.2 else "继续停止自动路由上线，保留显式 Deep",
        },
        "reviews": public_reviews,
    }


def _render(payload: dict) -> str:
    lines = [
        "# P4 人工盲评模拟报告", "",
        "- 评审性质：Codex 规则化模拟，`review_kind=simulated_human`。",
        "- 这不是独立真人评测，不能作为面试中的人工实验结论。", "",
        "| 方案 | 样本 | 总分 | 正确性 | 完整性 | 证据依据 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("A", "Normal"), ("C", "Deep")):
        item = payload["aggregate"][key]
        lines.append(f"| {label} | {item['cases']} | {item['avg_total']:.3f} | {item['avg_correctness']:.3f} | {item['avg_completeness']:.3f} | {item['avg_evidence_basis']:.3f} |")
    lines.extend([
        "", "## 复杂样本", "",
        f"Deep - Normal 平均总分：{payload['complex_total_delta_deep_minus_normal']:.3f}",
        f"Deep - Normal 证据依据：{payload['evidence_basis_delta_deep_minus_normal']:.3f}",
        f"门槛结论：{payload['gate']['decision']}", "",
        "模拟结果不能替代真人评分；下一步只有在实验室成员完成独立评分后，才可更新最终结论。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = evaluate()
    print(_render(payload))
    if args.write:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT.write_text(_render(payload), encoding="utf-8")
        print(f"JSON: {OUTPUT}\nMarkdown: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
