"""P4 选择性 Deep 路由的冻结评测，不调用模型、不改变线上路由。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.agent.agents.research_team import is_complex_research_task
from tests.eval.complex_research_dataset import COMPLEX_RESEARCH_DATASET


# 普通问题只标注为“不需要 Deep”，用于防止用全复杂数据集虚高 Recall。
NORMAL_CASES = [
    ("N01", "实验室组会每周什么时候举行？"),
    ("N02", "RDMA 和 TCP 有什么区别？"),
    ("N03", "帮我计算 123 * 456。"),
    ("N04", "请解释一下 NUMA 是什么。"),
    ("N05", "如何配置 Python 3.11 环境？"),
    ("N06", "你好，请介绍一下实验室。"),
    ("N07", "集群长任务需要注意什么？"),
    ("N08", "什么是 reranker？"),
    ("N09", "请列出新人入组第一天要做的事。"),
    ("N10", "查看最近的项目资料。"),
]


def _classify(question: str, strategy: str) -> bool:
    if strategy == "all_normal":
        return False
    if strategy == "all_deep":
        return True
    return is_complex_research_task(question)


def _metrics(rows: list[dict], strategy: str) -> dict:
    selected = [row for row in rows if row[strategy]]
    true_positive = sum(row["needs_deep"] and row[strategy] for row in rows)
    actual_positive = sum(row["needs_deep"] for row in rows)
    predicted_positive = len(selected)
    true_negative = sum((not row["needs_deep"]) and not row[strategy] for row in rows)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # 这是相对成本代理：不等同于实际 token 或账单。
    calls = sum(8 if row[strategy] else 2 for row in rows)
    return {
        "routing_precision": round(precision, 4),
        "routing_recall": round(recall, 4),
        "routing_f1": round(f1, 4),
        "over_route_count": sum((not row["needs_deep"]) and row[strategy] for row in rows),
        "under_route_count": sum(row["needs_deep"] and not row[strategy] for row in rows),
        "true_negative_count": true_negative,
        "estimated_logical_calls": calls,
    }


def evaluate() -> dict:
    rows = []
    for case in COMPLEX_RESEARCH_DATASET:
        rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "needs_deep": True,
            "question": case.query,
        })
    for case_id, query in NORMAL_CASES:
        rows.append({"case_id": case_id, "category": "normal", "needs_deep": False, "question": query})
    strategies = ("all_normal", "all_deep", "selective_deep")
    for row in rows:
        for strategy in strategies:
            row[strategy] = _classify(row["question"], strategy)
    metrics = {strategy: _metrics(rows, strategy) for strategy in strategies}
    selective = metrics["selective_deep"]
    passed = (
        selective["routing_recall"] >= 0.95
        and selective["routing_f1"] >= 0.90
        and selective["estimated_logical_calls"] < metrics["all_deep"]["estimated_logical_calls"]
    )
    return {
        "dataset": "p4_selective_routing_frozen",
        "case_count": len(rows),
        "complex_case_count": len(COMPLEX_RESEARCH_DATASET),
        "normal_case_count": len(NORMAL_CASES),
        "strategies": metrics,
        "gate": {
            "recall_threshold": 0.95,
            "f1_threshold": 0.90,
            "selective_gate_passed": passed,
            "decision": "进入脱敏真实回答评测" if passed else "继续保持显式 Deep，不上线自动路由",
        },
        "cases": rows,
    }


def main() -> int:
    payload = evaluate()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if "--write" in sys.argv:
        path = Path("data/选择性路由冻结评测结果.json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
