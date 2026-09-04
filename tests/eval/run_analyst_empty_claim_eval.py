#!/usr/bin/env python
"""Analyst 空 Claim 门禁的离线 before/after 评测。

该脚本只调用确定性 Reviewer 门禁，不访问模型、数据库或网络。Before 复现
``ca52b7c`` 之前的规则，After 使用当前规则；样本固定在本文件，避免为了评测
重新生成 V3 Gold/Judge。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agents import research_team as team
from src.rag.retrieval.acl_filter import UserContext, check_doc_access
from tests.eval.run_research_team_eval import _lexical_support


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    package: team.EvidencePackage
    analysis: team.AnalysisReport
    expected_claim_ids: tuple[str, ...] = ()


def _evidence(*source_ids: str) -> list[team.EvidenceItem]:
    return [
        team.EvidenceItem(
            source_id=source_id,
            subquestion="实验核验",
            title=f"实验记录 {source_id}",
            source=f"record-{source_id}.md",
            excerpt=(
                "RDMA 吞吐量为 91 Gbps；实验记录了 NUMA 绑定、驱动版本和消息大小。"
            ),
            metadata={"visibility": "public", "confidentiality": "internal"},
        )
        for source_id in source_ids
    ]


def _claim(claim_id: str, source_ids: Iterable[str], text: str) -> team.Claim:
    return team.Claim(
        claim_id=claim_id,
        text=text,
        claim_type="fact",
        source_ids=list(source_ids),
    )


def _cases() -> list[EvalCase]:
    def case(case_id: str, claims: list[team.Claim], sources: tuple[str, ...] = ("S1",)):
        return EvalCase(
            case_id=case_id,
            package=team.EvidencePackage(
                original_question="请总结 RDMA 实验结果和复现要求",
                evidences=_evidence(*sources),
            ),
            analysis=team.AnalysisReport(claims=claims),
            expected_claim_ids=tuple(claim.claim_id for claim in claims),
        )

    return [
        # 已知 failure：Before 会把“有 Evidence、无 Claim”交给模型 PASS。
        case("known_empty_claim_failure", []),
        # 6 条正常结构化回归样本，覆盖单声明、多声明和跨来源绑定。
        case("normal_fact", [_claim("C1", ["S1"], "RDMA 吞吐量为 91 Gbps")]),
        case(
            "normal_two_claims",
            [
                _claim("C1", ["S1"], "RDMA 吞吐量为 91 Gbps"),
                _claim("C2", ["S1"], "实验记录了 NUMA 绑定"),
            ],
        ),
        case(
            "normal_cross_source",
            [
                _claim("C1", ["S1"], "RDMA 吞吐量为 91 Gbps"),
                _claim("C2", ["S2"], "实验记录了 NUMA 绑定"),
            ],
            sources=("S1", "S2"),
        ),
        case("normal_versioned_fact", [_claim("C1", ["S1"], "记录了驱动版本")]),
        case("normal_reproduction_fact", [_claim("C1", ["S1"], "记录了消息大小")]),
        case("normal_environment_fact", [_claim("C1", ["S1"], "实验记录了 NUMA 绑定")]),
    ]


def _before_deterministic_review_issues(
    package: team.EvidencePackage,
    analysis: team.AnalysisReport,
) -> list[team.ReviewItem]:
    """复现 ca52b7c 之前的规则，保持与生产实现同一协议。"""
    valid_ids = {item.source_id for item in package.evidences}
    issues: list[team.ReviewItem] = []
    for claim in analysis.claims:
        invalid = [source_id for source_id in claim.source_ids if source_id not in valid_ids]
        if invalid:
            issues.append(team.ReviewItem(
                claim=claim.text,
                source_ids=claim.source_ids,
                supported=False,
                issue_type="invalid_source",
                revision_instruction=f"删除不存在的引用：{', '.join(invalid)}",
            ))
        elif not claim.source_ids:
            issues.append(team.ReviewItem(
                claim=claim.text,
                source_ids=[],
                supported=False,
                issue_type="citation_gap",
                revision_instruction="绑定支持该声明的 source_id；若无证据则删除或降级为局限性。",
            ))
    return issues


def _run_variant(
    cases: list[EvalCase],
    issue_fn: Callable[[team.EvidencePackage, team.AnalysisReport], list[team.ReviewItem]],
) -> dict:
    user = UserContext(
        user_id="eval-user",
        username="eval-user",
        role="student",
        department="",
        department_name="",
        department_path="",
    )
    rows = []
    latencies = []
    for item in cases:
        started = time.perf_counter()
        issues = issue_fn(item.package, item.analysis)
        latencies.append((time.perf_counter() - started) * 1000)
        empty_failure = bool(item.package.evidences and not item.analysis.claims and not issues)
        revision_triggered = bool(issues)
        acl_violations = sum(
            not check_doc_access(evidence.metadata, user)
            for evidence in item.package.evidences
        )
        rows.append({
            "case_id": item.case_id,
            "issue_types": [issue.issue_type for issue in issues],
            "empty_claim_failure": empty_failure,
            "revision_triggered": revision_triggered,
            "acl_violations": acl_violations,
            "claim_count": len(item.analysis.claims),
        })

    normal_rows = [row for row in rows if row["case_id"] != "known_empty_claim_failure"]
    normal_regression_count = sum(
        row["revision_triggered"] for row in normal_rows
        if row["case_id"].startswith("normal_")
    )
    revision_count = sum(row["revision_triggered"] for row in rows)
    return {
        "empty_claim_failure_count": sum(row["empty_claim_failure"] for row in rows),
        "revision_trigger_rate": revision_count / len(rows) if rows else 0.0,
        "revision_trigger_count": revision_count,
        "normal_regression_count": normal_regression_count,
        "acl_violation": sum(row["acl_violations"] for row in rows),
        # 该离线门禁不调用模型，因此真实 token 和逻辑调用均为 0；
        # Revision 触发次数单独报告，作为潜在成本增量的可解释上界。
        "input_tokens": 0,
        "output_tokens": 0,
        "logical_calls": 0,
        "revision_calls_upper_bound": revision_count,
        "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_ms": (
            max(latencies) if len(latencies) < 2
            else statistics.quantiles(latencies, n=20, method="inclusive")[18]
        ),
        "cases": rows,
    }


def _claim_metrics(cases: list[EvalCase]) -> dict:
    """对正常 fixture 的结构化声明绑定做协议级 precision/recall/F1。"""
    expected = {f"{item.case_id}:{claim_id}" for item in cases for claim_id in item.expected_claim_ids}
    actual = {
        f"{item.case_id}:{claim.claim_id}"
        for item in cases
        for claim in item.analysis.claims
        if claim.claim_id and claim.source_ids
        and all(source_id in {e.source_id for e in item.package.evidences} for source_id in claim.source_ids)
    }
    true_positive = len(expected & actual)
    precision = true_positive / len(actual) if actual else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"claim_precision": precision, "claim_recall": recall, "claim_f1": f1}


def _faithfulness(cases: list[EvalCase]) -> float:
    """复用现有声明-证据词汇支持 proxy，避免改变统计口径。"""
    checked = 0
    supported = 0
    for item in cases:
        evidence = {evidence.source_id: evidence.excerpt for evidence in item.package.evidences}
        for claim in item.analysis.claims:
            checked += 1
            if claim.source_ids and any(
                source_id in evidence and _lexical_support(claim.text, evidence[source_id])
                for source_id in claim.source_ids
            ):
                supported += 1
    return supported / checked if checked else 0.0


def evaluate() -> dict:
    cases = _cases()
    normal_cases = [item for item in cases if item.case_id.startswith("normal_")]
    before = _run_variant(cases, _before_deterministic_review_issues)
    after = _run_variant(cases, team._deterministic_review_issues)
    before.update(_claim_metrics(normal_cases))
    after.update(_claim_metrics(normal_cases))
    before["faithfulness"] = _faithfulness(cases)
    after["faithfulness"] = before["faithfulness"]
    return {
        "dataset": "analyst_empty_claim_gate_frozen_v1",
        "before_ref": "ca52b7c^",
        "after_ref": "ca52b7c",
        "sample_count": len(cases),
        "normal_regression_definition": "normal_* 样本的确定性复核结果发生变化",
        "variants": {"before": before, "after": after},
        "decision": {
            "target_failure_fixed": before["empty_claim_failure_count"] > after["empty_claim_failure_count"],
            "faithfulness_delta_after_minus_before": after["faithfulness"] - before["faithfulness"],
            "within_gate": (
                after["empty_claim_failure_count"] == 0
                and after["acl_violation"] == 0
                and after["normal_regression_count"] == 0
                and after["faithfulness"] - before["faithfulness"] >= -0.02
            ),
        },
    }


def main() -> int:
    result = evaluate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"]["within_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
