from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock

import pytest

from src.agent.agents import research_team as team
from src.rag.retrieval.acl_filter import UserContext
from tests.eval.claim_level_evaluator import (
    CLAIM_JUDGE_BATCH_SIZE,
    ClaimExtraction,
    ClaimJudgement,
    ClaimVerdict,
    ExtractedClaim,
    FlatClaimJudgement,
    judge_claims,
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


def test_v3_freeze_validation_separates_generation_from_judge(tmp_path, monkeypatch):
    manifest_path = tmp_path / "freeze.json"
    frozen = {
        "dataset_sha256": "dataset-v1",
        "corpus_snapshot_sha256": "corpus-v1",
        "implementation_snapshot_sha256": "generation-v1",
        "variants": list(runner.VARIANTS),
        "implementation_files": {
            "src/agent/agents/research_team.py": "research-v1",
            "tests/eval/claim_level_evaluator.py": "judge-v1",
        },
    }
    current = {
        **frozen,
        "implementation_snapshot_sha256": "generation-v2",
        "implementation_files": {
            **frozen["implementation_files"],
            "src/agent/agents/research_team.py": "research-v2",
        },
    }
    manifest_path.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(runner, "FREEZE_MANIFEST", manifest_path)
    monkeypatch.setattr(runner, "_freeze_payload", lambda: current)

    assert runner.ensure_freeze_manifest("judge") == frozen
    with pytest.raises(RuntimeError, match="implementation_snapshot_sha256"):
        runner.ensure_freeze_manifest("generation")

    current["implementation_files"]["tests/eval/claim_level_evaluator.py"] = "judge-v2"
    with pytest.raises(RuntimeError, match="Claim Judge"):
        runner.ensure_freeze_manifest("judge")


def test_human_calibration_tasks_are_deterministically_stratified():
    rows = []
    selected = {"V301", "V306", "V311", "V316"}
    for case in V3_CLAIM_EVAL_DATASET:
        if case.case_id not in selected:
            continue
        variants = {}
        for variant in runner.VARIANTS:
            variants[variant] = {
                "answer": "回答",
                "retrieved_contexts": [{"context_id": "N1", "text": "证据"}],
                "claim_evaluation": {
                    "response_claims": [
                        {"claim_id": "R1", "text": "回答声明一"},
                        {"claim_id": "R2", "text": "回答声明二"},
                    ],
                    "response_to_ground_truth": [
                        {"item_id": "R1", "verdict": "supported"},
                        {"item_id": "R2", "verdict": "supported"},
                    ],
                    "ground_truth_to_response": [{
                        "item_id": case.atomic_claims[0].claim_id,
                        "verdict": "supported",
                    }],
                    "response_to_context": [
                        {"item_id": "R1", "verdict": "supported"},
                        {"item_id": "R2", "verdict": "supported"},
                    ],
                },
            }
        rows.append({"case": {"case_id": case.case_id}, "variants": variants})

    first = runner._human_task_payload({"results": rows})
    second = runner._human_task_payload({"results": rows})

    assert first["selected_task_count"] == 36
    assert first["candidate_task_count"] == 60
    assert first["task_selection_sha256"] == second["task_selection_sha256"]
    assert {(
        task["case_id"], task["variant"], task["axis"]
    ) for task in first["tasks"]} == {
        (case_id, variant, axis)
        for case_id in selected
        for variant in runner.VARIANTS
        for axis in (
            "response_to_ground_truth",
            "ground_truth_to_response",
            "response_to_context",
        )
    }


def test_qwen_claim_content_alias_is_normalized_to_text():
    claim = ExtractedClaim.model_validate({"claim_id": "R1", "content": "可验证声明"})

    assert claim.text == "可验证声明"


def test_claim_extraction_keeps_protocol_limit_at_wire_boundary():
    result = ClaimExtraction.model_validate([f"声明 {index}" for index in range(25)])

    assert len(result.claims) == 24
    assert result.claims[0].text == "声明 0"
    assert result.claims[-1].text == "声明 23"


def test_qwen_verdict_aliases_are_normalized():
    verdict = ClaimVerdict.model_validate({
        "item_id": "R1",
        "label": "supported",
        "rationale": "参考文本明确支持。",
    })

    assert verdict.verdict == "supported"
    assert verdict.reason == "参考文本明确支持。"


def test_flat_qwen_judgement_accepts_root_list_and_aliases():
    result = FlatClaimJudgement.model_validate([
        {
            "item_id": "RG:R01", "label": "supported", "explanation": "明确支持",
            "reference_ids": [1],
        },
    ])

    assert result.items[0].task_id == "RG:R01"
    assert result.items[0].verdict == "supported"
    assert result.items[0].reason == "明确支持"
    assert result.items[0].reference_ids == ["1"]


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
async def test_claim_checker_batches_each_axis_without_losing_items(monkeypatch):
    response_claims = [
        ExtractedClaim(claim_id=f"R{index:02d}", text=f"声明 {index}")
        for index in range(1, CLAIM_JUDGE_BATCH_SIZE + 2)
    ]

    async def fake_invoke(_schema, prompt, **_kwargs):
        task_ids = re.findall(r'"task_id":\s*"([^"]+)"', prompt)
        return FlatClaimJudgement(items=[
            {"task_id": task_id, "verdict": "supported"} for task_id in task_ids
        ]), {"input_tokens": 10, "output_tokens": 2}

    invoke_mock = AsyncMock(side_effect=fake_invoke)
    monkeypatch.setattr("tests.eval.claim_level_evaluator._invoke_structured", invoke_mock)

    judgement, usage = await judge_claims(
        question="问题",
        answer="回答",
        response_claims=response_claims,
        ground_truth_answer="标准答案",
        gold_claims=[V3_CLAIM_EVAL_DATASET[0].atomic_claims[0]],
        retrieved_contexts=[{"context_id": "N01", "title": "资料", "text": "证据"}],
    )

    assert invoke_mock.await_count == 5  # RG 两批、GR 一批、RC 两批
    assert len(judgement.response_to_ground_truth) == len(response_claims)
    assert len(judgement.ground_truth_to_response) == 1
    assert len(judgement.response_to_context) == len(response_claims)
    assert usage == {"input_tokens": 50, "output_tokens": 10, "calls": 5}


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
        "human_calibration_plan": {"task_ids": [task["task_id"] for task in tasks]},
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
