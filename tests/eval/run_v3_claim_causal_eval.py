#!/usr/bin/env python
"""V3 多 Agent 因果消融与声明级评测。

三组方案：
- normal：产品默认 Query Expansion + RAG；
- deep_gated：Reviewer 空 REVISE 归一化为 PASS；
- deep_full：从完全相同的上游状态分叉，恢复门槛前 ReviewReport 并执行旧修订。

这不是 Blind Holdout。标准答案在运行前写定，作用是诊断 actionability gate 的
质量/成本因果效应，并为 20% 独立人工 Judge 校准生成任务。
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from config.settings import get_settings
from src.agent.agents.research_team import (
    EvidencePackage,
    ReviewReport,
    analyst_agent_node,
    deep_research_generation_node,
    research_agent_node,
    research_revision_node,
    reviewer_agent_node,
)
from src.rag.retrieval.acl_filter import UserContext, check_doc_access
from src.rag.retrieval.hybrid_retriever import _document_identity
from tests.eval.claim_level_evaluator import evaluate_claims
from tests.eval.deep_research_v3_claim_dataset import (
    ClaimEvalV3Query,
    V3_CLAIM_EVAL_DATASET,
)
from tests.eval.run_research_team_eval import _run_single_agent


OUTPUT = PROJECT_ROOT / "data" / "深度研究V3声明级因果消融结果.json"
CHECKPOINT = PROJECT_ROOT / "data" / "深度研究V3声明级因果消融断点.json"
REPORT = PROJECT_ROOT / "data" / "深度研究V3声明级因果消融报告.md"
FREEZE_MANIFEST = PROJECT_ROOT / "data" / "深度研究V3评测冻结清单.json"
HUMAN_TASKS = PROJECT_ROOT / "data" / "深度研究V3人工校准任务.json"
HUMAN_LABELS = PROJECT_ROOT / "data" / "深度研究V3人工校准评分.json"

VARIANTS = ("normal", "deep_gated", "deep_full")
CALIBRATION_CASES = ("V301", "V306", "V311", "V316")
QUALITY_TOLERANCE = 0.02
COST_REDUCTION_GATE = 0.05


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _corpus_manifest() -> Dict[str, str]:
    root = PROJECT_ROOT / "data" / "knowledge"
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _dataset_source() -> Path:
    return PROJECT_ROOT / "tests" / "eval" / "deep_research_v3_claim_dataset.py"


def _freeze_payload() -> Dict[str, Any]:
    settings = get_settings()
    corpus = _corpus_manifest()
    implementation_paths = (
        PROJECT_ROOT / "src" / "agent" / "agents" / "research_team.py",
        PROJECT_ROOT / "tests" / "eval" / "claim_level_evaluator.py",
        PROJECT_ROOT / "tests" / "eval" / "deep_research_v3_claim_dataset.py",
        PROJECT_ROOT / "tests" / "eval" / "run_v3_claim_causal_eval.py",
    )
    implementation = {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in implementation_paths
    }
    return {
        "status": "frozen_before_v3_generation",
        "dataset_identity": "v3_claim_causal_development_eval",
        "case_count": len(V3_CLAIM_EVAL_DATASET),
        "dataset_sha256": _sha256_file(_dataset_source()),
        "corpus_files": corpus,
        "corpus_snapshot_sha256": _sha256_bytes(
            json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ),
        "implementation_files": implementation,
        "implementation_snapshot_sha256": _sha256_bytes(
            json.dumps(implementation, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ),
        "variants": list(VARIANTS),
        "quality_tolerance": QUALITY_TOLERANCE,
        "cost_reduction_gate": COST_REDUCTION_GATE,
        "calibration_case_ids": list(CALIBRATION_CASES),
        "calibration_fraction": len(CALIBRATION_CASES) / len(V3_CLAIM_EVAL_DATASET),
        "models": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.dashscope_model,
            "embedding_model": settings.embedding_model,
            "reranker_provider": settings.reranker_provider,
            "reranker_model": settings.reranker_model,
        },
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "disclaimer": (
            "标准答案在生成前冻结；这是声明级开发评测，不是 blind holdout，"
            "不能用于新的盲测泛化声明。"
        ),
    }


def ensure_freeze_manifest() -> Dict[str, Any]:
    current = _freeze_payload()
    if FREEZE_MANIFEST.exists():
        frozen = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
        for field in (
            "dataset_sha256", "corpus_snapshot_sha256",
            "implementation_snapshot_sha256", "variants",
        ):
            if frozen.get(field) != current.get(field):
                raise RuntimeError(f"V3 冻结后 {field} 已变化，拒绝继续混用旧结果")
        return frozen
    FREEZE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_MANIFEST.write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return current


def _context_snapshot_hash(contexts: Sequence[Dict[str, Any]]) -> str:
    canonical = [
        {"context_id": item["context_id"], "stable_chunk_id": item["stable_chunk_id"], "text": item["text"]}
        for item in contexts
    ]
    return _sha256_bytes(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def _normal_contexts(
    docs: Sequence[Document], user: UserContext,
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for index, doc in enumerate(docs, 1):
        metadata = dict(doc.metadata or {})
        if not check_doc_access(metadata, user):
            raise RuntimeError("Normal context freeze 检测到未授权文档")
        contexts.append({
            "context_id": f"N{index:02d}",
            "stable_chunk_id": _document_identity(doc),
            "title": str(metadata.get("title") or Path(str(metadata.get("source") or "")).stem),
            "source": Path(str(metadata.get("source") or "")).name,
            "text": doc.page_content,
            "metadata": _json_safe({
                key: value for key, value in metadata.items()
                if key not in {"page_content", "embedding"}
            }),
            "acl_checked": True,
        })
    return contexts


def _deep_contexts(package: EvidencePackage, user: UserContext) -> List[Dict[str, Any]]:
    if not package.acl_checked:
        raise RuntimeError("Deep EvidencePackage 未通过 ACL 检查")
    contexts: List[Dict[str, Any]] = []
    for item in package.evidences:
        metadata = dict(item.metadata or {})
        if not check_doc_access(metadata, user):
            raise RuntimeError("Deep context freeze 检测到未授权文档")
        synthetic = Document(page_content=item.excerpt, metadata={
            **metadata, "source": item.source, "title": item.title,
        })
        contexts.append({
            "context_id": item.source_id,
            "stable_chunk_id": _document_identity(synthetic),
            "title": item.title,
            "source": Path(item.source).name,
            "text": item.excerpt,
            "metadata": _json_safe(metadata),
            "acl_checked": True,
        })
    return contexts


def _variant_payload(
    *,
    variant: str,
    answer: str,
    contexts: Sequence[Dict[str, Any]],
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    logical_calls: int,
    revision_count: int = 0,
    trace: Dict[str, Any] | None = None,
    causal_path_identical: bool = False,
) -> Dict[str, Any]:
    return {
        "variant": variant,
        "answer": answer,
        "retrieved_contexts": list(contexts),
        "context_snapshot_sha256": _context_snapshot_hash(contexts),
        "all_contexts_acl_checked": all(item.get("acl_checked") for item in contexts),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "logical_api_calls": logical_calls,
        "revision_count": revision_count,
        "causal_path_identical": causal_path_identical,
        "research_trace": trace or {},
    }


async def _run_normal(case: ClaimEvalV3Query, user: UserContext) -> Dict[str, Any]:
    started = time.perf_counter()
    answer, docs, input_tokens, output_tokens, calls = await _run_single_agent(
        case, user, expansion=True,
    )
    contexts = _normal_contexts(docs, user)
    return _variant_payload(
        variant="normal",
        answer=answer,
        contexts=contexts,
        latency_ms=int((time.perf_counter() - started) * 1000),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        logical_calls=calls,
    )


async def _run_deep_pair(
    case: ClaimEvalV3Query, user: UserContext,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """共享上游状态，只在 actionability gate 真正改变路由时分叉。"""

    shared: Dict[str, Any] = {
        "messages": [HumanMessage(content=case.query)],
        "user_context": user,
        "retrieval_top_k": 5,
    }
    shared.update(await research_agent_node(shared))
    shared.update(await analyst_agent_node(shared))
    shared.update(await reviewer_agent_node(shared))

    reviewer_trace = (
        (shared.get("research_trace") or {}).get("stages") or {}
    ).get("reviewer") or {}
    before_payload = reviewer_trace.get("review_report_before_actionability_gate") or {}
    before_report = ReviewReport.model_validate(before_payload)
    gated_report = ReviewReport.model_validate(shared["review_report"])
    gate_changed_route = before_report.decision == "REVISE" and gated_report.decision == "PASS"

    gated_state = copy.deepcopy(shared)
    if gated_report.decision != "PASS":
        gated_state.update(await research_revision_node(gated_state))
    gated_state.update(await deep_research_generation_node(gated_state))
    gated_metrics = gated_state.get("research_team_metrics") or {}
    gated_package = EvidencePackage.model_validate(gated_state["evidence_package"])
    gated = _variant_payload(
        variant="deep_gated",
        answer=gated_state["final_answer"],
        contexts=_deep_contexts(gated_package, user),
        latency_ms=int(gated_metrics.get("elapsed_ms", 0)),
        input_tokens=int(gated_metrics.get("input_tokens", 0)),
        output_tokens=int(gated_metrics.get("output_tokens", 0)),
        logical_calls=(
            int(gated_metrics.get("llm_calls", 0))
            + int(gated_metrics.get("retrieval_calls", 0))
        ),
        revision_count=int(gated_state.get("research_revision_count", 0)),
        trace=gated_state.get("research_trace") or {},
    )

    if not gate_changed_route:
        full = copy.deepcopy(gated)
        full["variant"] = "deep_full"
        full["causal_path_identical"] = True
        return gated, full

    full_state = copy.deepcopy(shared)
    full_state["review_report"] = before_report.model_dump()
    full_state.update(await research_revision_node(full_state))
    full_state.update(await deep_research_generation_node(full_state))
    full_metrics = full_state.get("research_team_metrics") or {}
    full_package = EvidencePackage.model_validate(full_state["evidence_package"])
    full = _variant_payload(
        variant="deep_full",
        answer=full_state["final_answer"],
        contexts=_deep_contexts(full_package, user),
        latency_ms=int(full_metrics.get("elapsed_ms", 0)),
        input_tokens=int(full_metrics.get("input_tokens", 0)),
        output_tokens=int(full_metrics.get("output_tokens", 0)),
        logical_calls=(
            int(full_metrics.get("llm_calls", 0))
            + int(full_metrics.get("retrieval_calls", 0))
        ),
        revision_count=int(full_state.get("research_revision_count", 0)),
        trace=full_state.get("research_trace") or {},
    )
    return gated, full


async def generate_case(case: ClaimEvalV3Query, user: UserContext) -> Dict[str, Any]:
    normal = await _run_normal(case, user)
    gated, full = await _run_deep_pair(case, user)
    return {
        "case": asdict(case),
        "variants": {"normal": normal, "deep_gated": gated, "deep_full": full},
    }


def _base_payload(freeze: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "dataset": "v3_claim_causal_development_eval",
        "status": "answers_generated_claim_judge_pending",
        "case_count": len(V3_CLAIM_EVAL_DATASET),
        "completed_cases": len(results),
        "variants": list(VARIANTS),
        "freeze_manifest_sha256": _sha256_file(FREEZE_MANIFEST),
        "dataset_sha256": freeze["dataset_sha256"],
        "corpus_snapshot_sha256": freeze["corpus_snapshot_sha256"],
        "results": results,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _load_checkpoint(freeze: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not CHECKPOINT.exists():
        return []
    payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if payload.get("dataset_sha256") != freeze["dataset_sha256"]:
        raise RuntimeError("V3 checkpoint 与冻结数据集不一致")
    return list(payload.get("results") or [])


async def run_generation(limit: int | None = None) -> Dict[str, Any]:
    freeze = ensure_freeze_manifest()
    if OUTPUT.exists():
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if len(payload.get("results") or []) == len(V3_CLAIM_EVAL_DATASET):
            return payload
    results = _load_checkpoint(freeze)
    completed = {row["case"]["case_id"] for row in results}
    user = UserContext.anonymous()
    pending = [case for case in V3_CLAIM_EVAL_DATASET if case.case_id not in completed]
    if limit is not None:
        pending = pending[:limit]
    for case in pending:
        print(f"[generate {len(results)+1}/{len(V3_CLAIM_EVAL_DATASET)}] {case.case_id}", flush=True)
        results.append(await generate_case(case, user))
        payload = _base_payload(freeze, results)
        CHECKPOINT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    payload = _base_payload(freeze, results)
    if len(results) == len(V3_CLAIM_EVAL_DATASET):
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return payload


async def run_judge(limit: int | None = None) -> Dict[str, Any]:
    ensure_freeze_manifest()
    if not OUTPUT.exists():
        raise RuntimeError("必须先完成 20 条三变体答案生成")
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    completed = 0
    for row in payload["results"]:
        case = next(item for item in V3_CLAIM_EVAL_DATASET if item.case_id == row["case"]["case_id"])
        for variant_name in VARIANTS:
            variant = row["variants"][variant_name]
            if variant.get("claim_evaluation"):
                continue
            if limit is not None and completed >= limit:
                OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return payload
            print(f"[judge] {case.case_id} {variant_name}", flush=True)
            variant["claim_evaluation"] = await evaluate_claims(
                question=case.query,
                answer=variant["answer"],
                ground_truth_answer=case.ground_truth_answer,
                gold_claims=case.atomic_claims,
                retrieved_contexts=variant["retrieved_contexts"],
            )
            completed += 1
            payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["status"] = "qwen_claim_judge_complete_human_calibration_pending"
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.fmean(items) if items else 0.0


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))])


def aggregate(payload: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for variant_name in VARIANTS:
        rows = [row["variants"][variant_name] for row in payload["results"]]
        judged = [row["claim_evaluation"] for row in rows if row.get("claim_evaluation")]
        result[variant_name] = {
            "cases": len(rows),
            "avg_claim_precision": _mean(item["claim_precision"] for item in judged),
            "avg_claim_recall": _mean(item["claim_recall"] for item in judged),
            "avg_claim_f1": _mean(item["claim_f1"] for item in judged),
            "avg_faithfulness": _mean(item["faithfulness"] for item in judged),
            "avg_latency_ms": _mean(row["latency_ms"] for row in rows),
            "p50_latency_ms": _percentile([row["latency_ms"] for row in rows], 0.50),
            "p95_latency_ms": _percentile([row["latency_ms"] for row in rows], 0.95),
            "avg_input_tokens": _mean(row["input_tokens"] for row in rows),
            "avg_output_tokens": _mean(row["output_tokens"] for row in rows),
            "avg_logical_calls": _mean(row["logical_api_calls"] for row in rows),
            "total_input_tokens": sum(row["input_tokens"] for row in rows),
            "total_output_tokens": sum(row["output_tokens"] for row in rows),
            "all_contexts_acl_checked": all(row["all_contexts_acl_checked"] for row in rows),
        }
    affected = [
        row for row in payload["results"]
        if not row["variants"]["deep_full"]["causal_path_identical"]
    ]
    gated = result["deep_gated"]
    full = result["deep_full"]
    result["causal_gate"] = {
        "affected_cases": len(affected),
        "affected_case_ids": [row["case"]["case_id"] for row in affected],
        "claim_f1_delta_gated_minus_full": gated["avg_claim_f1"] - full["avg_claim_f1"],
        "faithfulness_delta_gated_minus_full": gated["avg_faithfulness"] - full["avg_faithfulness"],
        "latency_reduction": (
            (full["avg_latency_ms"] - gated["avg_latency_ms"]) / full["avg_latency_ms"]
            if full["avg_latency_ms"] else 0.0
        ),
        "input_token_reduction": (
            (full["avg_input_tokens"] - gated["avg_input_tokens"]) / full["avg_input_tokens"]
            if full["avg_input_tokens"] else 0.0
        ),
        "output_token_reduction": (
            (full["avg_output_tokens"] - gated["avg_output_tokens"]) / full["avg_output_tokens"]
            if full["avg_output_tokens"] else 0.0
        ),
    }
    return result


def _human_task_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    gold_by_case = {case.case_id: case for case in V3_CLAIM_EVAL_DATASET}
    for row in payload["results"]:
        case_id = row["case"]["case_id"]
        if case_id not in CALIBRATION_CASES:
            continue
        case = gold_by_case[case_id]
        gold_map = {claim.claim_id: claim.text for claim in case.atomic_claims}
        for variant_name in VARIANTS:
            variant = row["variants"][variant_name]
            evaluation = variant["claim_evaluation"]
            response_map = {
                claim["claim_id"]: claim["text"] for claim in evaluation["response_claims"]
            }
            axes = (
                ("response_to_ground_truth", case.ground_truth_answer, response_map),
                ("ground_truth_to_response", variant["answer"], gold_map),
                ("response_to_context", variant["retrieved_contexts"], response_map),
            )
            for axis, reference, claim_map in axes:
                for item in evaluation[axis]:
                    item_id = item["item_id"]
                    tasks.append({
                        "task_id": f"{case_id}:{variant_name}:{axis}:{item_id}",
                        "case_id": case_id,
                        "variant": variant_name,
                        "axis": axis,
                        "claim": claim_map[item_id],
                        "reference": reference,
                        "human_verdict": None,
                        "human_reason": "",
                    })
    return {
        "status": "pending_independent_human_labels",
        "calibration_fraction": 0.20,
        "selected_case_ids": list(CALIBRATION_CASES),
        "reviewer_id": "",
        "independence_attestation": False,
        "instructions": (
            "不要查看主结果中的 Qwen verdict。只判断 reference 是否足以支持 claim；"
            "human_verdict 只能填 supported 或 not_supported，并简述边界。完成后填写 reviewer_id，"
            "并将 independence_attestation 设为 true，确认评分前未查看 Qwen verdict。"
        ),
        "tasks": tasks,
    }


def write_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    if any(
        not row["variants"][name].get("claim_evaluation")
        for row in payload["results"] for name in VARIANTS
    ):
        raise RuntimeError("声明级 Judge 尚未完成，不能生成最终消融报告")
    summary = aggregate(payload)
    payload["aggregate"] = summary
    human_tasks = _human_task_payload(payload)
    if not HUMAN_TASKS.exists():
        HUMAN_TASKS.write_text(
            json.dumps(human_tasks, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    if not HUMAN_LABELS.exists():
        HUMAN_LABELS.write_text(
            json.dumps(human_tasks, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    gate = summary["causal_gate"]
    payload["decision"] = {
        "status": "pending_independent_human_calibration",
        "quality_gate_preliminary": (
            gate["claim_f1_delta_gated_minus_full"] >= -QUALITY_TOLERANCE
            and gate["faithfulness_delta_gated_minus_full"] >= -QUALITY_TOLERANCE
        ),
        "cost_gate_preliminary": (
            gate["affected_cases"] > 0
            and gate["latency_reduction"] >= COST_REDUCTION_GATE
            and gate["input_token_reduction"] >= COST_REDUCTION_GATE
        ),
        "actionability_gate_final_decision": "pending_human_calibration",
    }
    payload["status"] = "qwen_claim_judge_complete_human_calibration_pending"
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 深度研究 V3 声明级因果消融报告", "",
        "> 这是标准答案预先写定的 Development Eval，不是 Blind Holdout。", "",
        "| 方案 | Claim Precision | Recall | F1 | Faithfulness | P50/P95(ms) | 输入/输出 Token | 调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        item = summary[name]
        lines.append(
            f"| {name} | {item['avg_claim_precision']:.3f} | {item['avg_claim_recall']:.3f} | "
            f"{item['avg_claim_f1']:.3f} | {item['avg_faithfulness']:.3f} | "
            f"{item['p50_latency_ms']:.0f}/{item['p95_latency_ms']:.0f} | "
            f"{item['avg_input_tokens']:.0f}/{item['avg_output_tokens']:.0f} | "
            f"{item['avg_logical_calls']:.2f} |"
        )
    lines.extend([
        "", "## Actionability Gate 因果结果", "",
        f"- 真实分叉样本：{gate['affected_cases']} / 20（{', '.join(gate['affected_case_ids']) or '无'}）",
        f"- Claim F1 差值（gated-full）：{gate['claim_f1_delta_gated_minus_full']:+.3f}",
        f"- Faithfulness 差值：{gate['faithfulness_delta_gated_minus_full']:+.3f}",
        f"- 平均延迟降低：{gate['latency_reduction']:.1%}",
        f"- 平均 input token 降低：{gate['input_token_reduction']:.1%}",
        f"- 平均 output token 降低：{gate['output_token_reduction']:.1%}",
        "", "## 当前决策", "",
        "Qwen Judge 已完成但尚未通过 20% 独立人工标签校准，因此不能最终保留或否决优化。",
        f"人工任务：`{HUMAN_LABELS.relative_to(PROJECT_ROOT)}`。",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _configure_proxy() -> None:
    settings = get_settings()
    for key, value in (("HTTP_PROXY", settings.http_proxy), ("HTTPS_PROXY", settings.https_proxy)):
        if value:
            os.environ[key] = value
            os.environ[key.lower()] = value


async def main() -> None:
    parser = argparse.ArgumentParser(description="V3 声明级多 Agent 因果消融")
    parser.add_argument("--stage", choices=("generate", "judge", "report", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="本次最多处理的 case 或 Judge variant 数")
    args = parser.parse_args()
    _configure_proxy()
    payload: Dict[str, Any] | None = None
    if args.stage in ("generate", "all"):
        payload = await run_generation(args.limit)
        if len(payload.get("results") or []) < len(V3_CLAIM_EVAL_DATASET):
            print("生成阶段尚未完成，已保存断点。")
            return
    if args.stage in ("judge", "all"):
        payload = await run_judge(args.limit)
        if any(
            not row["variants"][name].get("claim_evaluation")
            for row in payload["results"] for name in VARIANTS
        ):
            print("Judge 阶段尚未完成，已保存断点。")
            return
    if args.stage in ("report", "all"):
        if payload is None:
            payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        write_report(payload)
        print(f"报告已写入 {REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
