from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agent.agents import research_team as team
from src.rag.retrieval.acl_filter import UserContext
from tests.eval.claim_level_evaluator import (
    ClaimJudgement,
    ClaimVerdict,
    ExtractedClaim,
    score_claim_judgement,
)
from tests.eval.deep_research_v3_claim_dataset import V3_CLAIM_EVAL_DATASET
from tests.eval import run_v3_claim_causal_eval as runner
from tests.eval.summarize_v3_human_calibration import summarize


def test_v3_dataset_has_frozen_ground_truth_and_known_sources():
    known_sources = {
        "RDMA与高性能网络实验规范",
        "实验室常见问题FAQ",
        "论文阅读与实验记录要求",
        "实验室组会制度与汇报要求",
        "论文投稿与对外汇报流程",
        "设备预约与共享资源使用流程",
        "实验室安全与值班制度",
        "高性能计算集群使用说明",
        "新生入组第一周任务清单",
        "全局变量实现2",
        "分布式NUMA研究计划",
        "实验室成员与分工说明",
        "实验室例会纪要_2026-04-15",
        "分布式NUMA课题组会模板",
    }
    claim_ids = [
        claim.claim_id for case in V3_CLAIM_EVAL_DATASET for claim in case.atomic_claims
    ]

    assert len(V3_CLAIM_EVAL_DATASET) == 20
    assert len(claim_ids) == len(set(claim_ids))
    assert all(set(case.relevant_doc_ids) <= known_sources for case in V3_CLAIM_EVAL_DATASET)
    assert all(
        set(claim.source_doc_ids) <= known_sources
        for case in V3_CLAIM_EVAL_DATASET for claim in case.atomic_claims
    )


def test_claim_scoring_is_ragchecker_style_and_missing_judge_item_fails_closed():
    case = V3_CLAIM_EVAL_DATASET[0]
    response_claims = [
        ExtractedClaim(claim_id="R01", text="相同 commit 不足以复现"),
        ExtractedClaim(claim_id="R02", text="所有机器必然得到完全相同结果"),
    ]
    judgement = ClaimJudgement(
        response_to_ground_truth=[
            ClaimVerdict(item_id="R01", verdict="supported"),
            # R02 故意漏掉，应按 not_enough_information 计为不支持。
        ],
        ground_truth_to_response=[
            ClaimVerdict(item_id=case.atomic_claims[0].claim_id, verdict="supported"),
        ],
        response_to_context=[
            ClaimVerdict(item_id="R01", verdict="supported"),
            ClaimVerdict(item_id="R02", verdict="contradicted"),
        ],
    )

    result = score_claim_judgement(response_claims, case.atomic_claims, judgement)

    assert result["claim_precision"] == 0.5
    assert result["claim_recall"] == 0.25
    assert result["claim_f1"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["faithfulness"] == 0.5
    assert result["response_to_ground_truth"][1]["verdict"] == "not_enough_information"


@pytest.mark.asyncio
async def test_deep_causal_pair_shares_upstream_and_only_full_runs_empty_revision(monkeypatch):
    package = team.EvidencePackage(
        original_question="问题",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="问题", title="资料", source="资料.md", excerpt="证据",
        )],
    )
    before = team.ReviewReport(decision="REVISE", overall_instruction="笼统润色")
    after = team.ReviewReport(decision="PASS")

    async def research(state):
        return {
            "evidence_package": package.model_dump(),
            "research_team_metrics": {"llm_calls": 1, "retrieval_calls": 1, "input_tokens": 10, "output_tokens": 2, "elapsed_ms": 10},
            "research_trace": {"stages": {}},
            "research_revision_count": 0,
        }

    async def analyst(state):
        return {"analysis_report": team.AnalysisReport(claims=[team.Claim(text="证据", source_ids=["S1"])]).model_dump()}

    async def reviewer(state):
        return {
            "review_report": after.model_dump(),
            "research_trace": {"stages": {"reviewer": {
                "review_report_before_actionability_gate": before.model_dump(),
            }}},
        }

    async def revision(state):
        metrics = dict(state["research_team_metrics"])
        metrics.update({"llm_calls": 2, "input_tokens": 30, "output_tokens": 7, "elapsed_ms": 30})
        return {"research_revision_count": 1, "research_team_metrics": metrics}

    async def generation(state):
        metrics = dict(state["research_team_metrics"])
        metrics.update({
            "llm_calls": int(metrics.get("llm_calls", 0)) + 1,
            "input_tokens": int(metrics.get("input_tokens", 0)) + 5,
            "output_tokens": int(metrics.get("output_tokens", 0)) + 2,
            "elapsed_ms": int(metrics.get("elapsed_ms", 0)) + 5,
        })
        return {
            "final_answer": "修订答案" if state.get("research_revision_count") else "门槛答案",
            "research_team_metrics": metrics,
        }

    revision_mock = AsyncMock(side_effect=revision)
    monkeypatch.setattr(runner, "research_agent_node", research)
    monkeypatch.setattr(runner, "analyst_agent_node", analyst)
    monkeypatch.setattr(runner, "reviewer_agent_node", reviewer)
    monkeypatch.setattr(runner, "research_revision_node", revision_mock)
    monkeypatch.setattr(runner, "deep_research_generation_node", generation)

    gated, full = await runner._run_deep_pair(V3_CLAIM_EVAL_DATASET[0], UserContext.anonymous())

    assert revision_mock.await_count == 1
    assert gated["answer"] == "门槛答案"
    assert gated["revision_count"] == 0
    assert full["answer"] == "修订答案"
    assert full["revision_count"] == 1
    assert full["input_tokens"] > gated["input_tokens"]
    assert full["causal_path_identical"] is False


def _calibration_fixture(human_verdict="supported"):
    rows = []
    tasks = []
    for case_id in ("V301", "V306", "V311", "V316"):
        task_id = f"{case_id}:normal:response_to_ground_truth:R01"
        rows.append({
            "case": {"case_id": case_id},
            "variants": {"normal": {"claim_evaluation": {
                "response_to_ground_truth": [{"item_id": "R01", "verdict": "supported"}],
                "ground_truth_to_response": [],
                "response_to_context": [],
            }}},
        })
        tasks.append({
            "task_id": task_id,
            "axis": "response_to_ground_truth",
            "human_verdict": human_verdict,
        })
    results = {
        "results": rows,
        "decision": {"quality_gate_preliminary": True, "cost_gate_preliminary": True},
    }
    labels = {
        "reviewer_id": "independent-reviewer",
        "independence_attestation": True,
        "selected_case_ids": ["V301", "V306", "V311", "V316"],
        "calibration_fraction": 0.20,
        "tasks": tasks,
    }
    return results, labels


def test_human_calibration_requires_attestation_and_controls_final_gate_decision():
    results, labels = _calibration_fixture()
    summary = summarize(results, labels)

    assert summary["agreement"] == 1.0
    assert summary["qwen_judge_calibration_passed"] is True
    assert summary["actionability_gate_final_decision"] == "keep_actionability_gate"

    labels["independence_attestation"] = False
    with pytest.raises(ValueError, match="独立评分声明"):
        summarize(results, labels)
