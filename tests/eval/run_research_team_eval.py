#!/usr/bin/env python
"""复杂科研任务 A/B/C 对比评测。

A: Hybrid/CRAG + 单 Agent，不主动 Query Expansion
B: Hybrid/CRAG + Query Expansion + 单 Agent
C: 固定 Researcher + Analyst + Reviewer

主报告中的 token 为接口返回的可观测 usage；CRAG 内部 grader 调用无法全部取得
usage，因此同时报告 logical_api_calls_estimate，且不把它声明为账单值。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import jieba

# 允许既用 `python -m tests.eval...`，也可从仓库根目录直接执行本文件。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.agents.knowledge import _build_generation_prompt, _build_retrieval_context
from src.agent.agents.research_team import (
    AnalysisReport,
    EvidencePackage,
    ReviewReport,
    analyst_agent_node,
    research_agent_node,
    research_revision_node,
    research_team_finalizer_node,
    reviewer_agent_node,
)
from src.models.llm import get_llm
from src.rag.evaluation.conflict_detector import detect_document_conflicts
from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline
from src.rag.retrieval.acl_filter import UserContext, check_doc_access
from tests.eval.complex_research_dataset import (
    COMPLEX_RESEARCH_DATASET,
    ComplexResearchQuery,
)


VARIANT_LABELS = {
    "A": "Hybrid RAG + 单 Agent",
    "B": "Query Expansion + 单 Agent",
    "C": "Researcher + Analyst + Reviewer",
}


@dataclass
class VariantRun:
    variant: str
    answer: str
    document_titles: List[str]
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    logical_api_calls_estimate: int = 0
    keyword_correctness: float = 0.0
    citation_coverage: float = 0.0
    citation_support_rate: float = 0.0
    false_premise_accuracy: float = 0.0
    conflict_accuracy: float = 0.0
    acl_leak_count: int = 0
    retrieved_doc_recall: float = 0.0
    review_decision: str = ""
    revision_count: int = 0
    error: str = ""


def _title(doc: Document) -> str:
    metadata = doc.metadata or {}
    source = str(metadata.get("title") or metadata.get("source") or "")
    return Path(source).stem


def _matches_doc(actual: str, expected: str) -> bool:
    a = actual.lower().replace(" ", "")
    e = expected.lower().replace(" ", "")
    return e in a or a in e


def _doc_recall(docs: Sequence[Document], expected: Sequence[str]) -> float:
    if not expected:
        return 1.0
    actual = [_title(doc) for doc in docs]
    hits = sum(any(_matches_doc(item, target) for item in actual) for target in expected)
    return hits / len(expected)


def _keyword_correctness(answer: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer.lower()
    return sum(keyword.lower() in lowered for keyword in keywords) / len(keywords)


def _sentences(answer: str) -> List[str]:
    return [part.strip() for part in re.split(r"[。！？!?；;\n]+", answer) if len(part.strip()) >= 6]


_CITATION_RE = re.compile(r"\[(?:文档(\d+)|S(\d+))\]")


def _citation_coverage(answer: str) -> float:
    sentences = [
        sentence for sentence in _sentences(answer)
        if not sentence.startswith(">")
        and not any(label in sentence for label in ("证据局限", "资料冲突", "前提核验"))
    ]
    if not sentences:
        return 0.0
    return sum(bool(_CITATION_RE.search(sentence)) for sentence in sentences) / len(sentences)


def _tokens(text: str) -> set[str]:
    return {
        token.lower() for token in jieba.lcut(text)
        if len(token.strip()) >= 2 and not re.fullmatch(r"[\W_]+", token)
    }


def _lexical_support(claim: str, evidence: str) -> bool:
    claim_tokens = _tokens(_CITATION_RE.sub("", claim))
    if not claim_tokens:
        return False
    evidence_tokens = _tokens(evidence)
    return len(claim_tokens & evidence_tokens) / len(claim_tokens) >= 0.12


def _citation_support_single(answer: str, docs: Sequence[Document]) -> float:
    checked = 0
    supported = 0
    for sentence in _sentences(answer):
        matches = list(_CITATION_RE.finditer(sentence))
        for match in matches:
            index_text = match.group(1) or match.group(2)
            index = int(index_text) - 1
            checked += 1
            if 0 <= index < len(docs) and _lexical_support(sentence, docs[index].page_content):
                supported += 1
    return supported / checked if checked else 0.0


def _citation_support_team(analysis: AnalysisReport, package: EvidencePackage) -> float:
    evidence = {item.source_id: item.excerpt for item in package.evidences}
    if not analysis.claims:
        return 0.0
    supported = 0
    for claim in analysis.claims:
        if claim.source_ids and any(
            source_id in evidence and _lexical_support(claim.text, evidence[source_id])
            for source_id in claim.source_ids
        ):
            supported += 1
    return supported / len(analysis.claims)


def _false_premise_accuracy(answer: str, case: ComplexResearchQuery) -> float:
    if not case.should_flag_false_premise:
        return 1.0
    has_boundary = bool(re.search(r"前提.{0,12}(不成立|不准确|缺乏|没有|无法|未被)|"
                                  r"(没有|缺乏|无法).{0,10}(证据|证明)", answer))
    return float(has_boundary)


def _conflict_accuracy(answer: str, case: ComplexResearchQuery) -> float:
    if not case.should_handle_conflict:
        return 1.0
    return float(bool(re.search(r"冲突|矛盾|不一致|差异", answer)))


def _acl_leaks(docs: Sequence[Document], user: UserContext) -> int:
    return sum(not check_doc_access(doc.metadata or {}, user) for doc in docs)


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return 0, 0
    return (
        int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
    )


async def _run_single_agent(
    case: ComplexResearchQuery,
    user: UserContext,
    *,
    expansion: bool,
) -> tuple[str, List[Document], int, int, int]:
    pipeline = get_corrective_rag_pipeline()
    results, grade, _history = await pipeline.retrieve(
        query=case.query,
        top_k=5,
        needs_expansion=expansion,
        user=user,
    )
    docs = [doc for doc, _score in results if check_doc_access(doc.metadata or {}, user)]
    decision = getattr(getattr(grade, "decision", None), "value", "no_results")
    if decision == "no_results" or not docs:
        return "现有权限范围内没有找到足够证据，无法可靠回答。", docs, 0, 0, 1

    allowed_results = [(doc, score) for doc, score in results if doc in docs]
    context, _version = _build_retrieval_context(case.query, allowed_results, grade)
    prompt = _build_generation_prompt(
        query=case.query,
        retrieval_context=context,
        conflict_warnings=detect_document_conflicts(docs, case.query),
        summary="",
        mem0_memories="",
        user_context=user,
    )
    response = await get_llm(temperature=0.2).ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=case.query),
    ])
    input_tokens, output_tokens = _usage(response)
    # 1 次检索管线 + 1 次最终生成。检索管线内部 grader 次数另见报告声明。
    return str(response.content), docs, input_tokens, output_tokens, 2


async def _run_team(
    case: ComplexResearchQuery,
    user: UserContext,
) -> tuple[str, List[Document], Dict[str, Any]]:
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content=case.query)],
        "user_context": user,
        "retrieval_top_k": 5,
    }
    state.update(await research_agent_node(state))
    state.update(await analyst_agent_node(state))
    state.update(await reviewer_agent_node(state))
    review = ReviewReport.model_validate(state["review_report"])
    if review.decision != "PASS":
        state.update(await research_revision_node(state))
    state.update(await research_team_finalizer_node(state))
    return state["final_answer"], list(state.get("retrieved_docs") or []), state


def _score_run(
    run: VariantRun,
    case: ComplexResearchQuery,
    docs: Sequence[Document],
    user: UserContext,
    *,
    analysis: AnalysisReport | None = None,
    package: EvidencePackage | None = None,
) -> VariantRun:
    run.keyword_correctness = _keyword_correctness(run.answer, case.expected_keywords)
    run.citation_coverage = _citation_coverage(run.answer)
    run.citation_support_rate = (
        _citation_support_team(analysis, package)
        if analysis is not None and package is not None
        else _citation_support_single(run.answer, docs)
    )
    run.false_premise_accuracy = _false_premise_accuracy(run.answer, case)
    run.conflict_accuracy = _conflict_accuracy(run.answer, case)
    run.acl_leak_count = _acl_leaks(docs, user)
    run.retrieved_doc_recall = _doc_recall(docs, case.relevant_doc_ids)
    return run


async def evaluate_case(
    case: ComplexResearchQuery,
    variants: Sequence[str],
    user: UserContext,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"case": asdict(case), "variants": {}}
    for variant in variants:
        started = time.perf_counter()
        try:
            if variant in ("A", "B"):
                answer, docs, input_tokens, output_tokens, calls = await _run_single_agent(
                    case, user, expansion=variant == "B",
                )
                run = VariantRun(
                    variant=variant,
                    answer=answer,
                    document_titles=[_title(doc) for doc in docs],
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    logical_api_calls_estimate=calls,
                )
                _score_run(run, case, docs, user)
            else:
                answer, docs, state = await _run_team(case, user)
                metrics = state.get("research_team_metrics") or {}
                analysis = AnalysisReport.model_validate(state.get("analysis_report") or {})
                package = EvidencePackage.model_validate(state.get("evidence_package") or {})
                review = ReviewReport.model_validate(state.get("review_report") or {})
                run = VariantRun(
                    variant=variant,
                    answer=answer,
                    document_titles=[_title(doc) for doc in docs],
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    input_tokens=int(metrics.get("input_tokens", 0)),
                    output_tokens=int(metrics.get("output_tokens", 0)),
                    logical_api_calls_estimate=int(metrics.get("llm_calls", 0)) + int(metrics.get("retrieval_calls", 0)),
                    review_decision=review.decision,
                    revision_count=int(state.get("research_revision_count", 0)),
                )
                _score_run(run, case, docs, user, analysis=analysis, package=package)
        except Exception as exc:
            run = VariantRun(
                variant=variant,
                answer="",
                document_titles=[],
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
        result["variants"][variant] = asdict(run)
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def aggregate(results: Sequence[Dict[str, Any]], variants: Sequence[str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    metrics = (
        "keyword_correctness", "citation_coverage", "citation_support_rate",
        "false_premise_accuracy", "conflict_accuracy", "retrieved_doc_recall",
    )
    for variant in variants:
        rows = [row["variants"][variant] for row in results]
        valid = [row for row in rows if not row["error"]]
        summary: Dict[str, Any] = {
            "label": VARIANT_LABELS[variant],
            "cases": len(rows),
            "success_cases": len(valid),
            "error_cases": len(rows) - len(valid),
            "acl_leak_count": sum(row["acl_leak_count"] for row in valid),
            "avg_input_tokens": sum(row["input_tokens"] for row in valid) / len(valid) if valid else 0,
            "avg_output_tokens": sum(row["output_tokens"] for row in valid) / len(valid) if valid else 0,
            "avg_logical_api_calls_estimate": sum(row["logical_api_calls_estimate"] for row in valid) / len(valid) if valid else 0,
            "latency_p50_ms": _percentile([row["latency_ms"] for row in valid], 0.50),
            "latency_p95_ms": _percentile([row["latency_ms"] for row in valid], 0.95),
        }
        for metric in metrics:
            summary[f"avg_{metric}"] = sum(row[metric] for row in valid) / len(valid) if valid else 0
        output[variant] = summary
    return output


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# 复杂科研任务 A/B/C 评测报告",
        "",
        "> Token 为模型接口可观测 usage；logical API calls 是逻辑调用估计，未完整包含 CRAG 内部 grader，不能当作账单。",
        "",
        "| 方案 | 成功/总数 | 关键词正确性 | 引用覆盖 | 引用支持 | 错误前提 | 冲突处理 | ACL 泄漏 | P50(ms) | P95(ms) | 估计调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in payload["variants"]:
        item = payload["aggregate"][key]
        lines.append(
            f"| {item['label']} | {item['success_cases']}/{item['cases']} | "
            f"{item['avg_keyword_correctness']:.3f} | {item['avg_citation_coverage']:.3f} | "
            f"{item['avg_citation_support_rate']:.3f} | {item['avg_false_premise_accuracy']:.3f} | "
            f"{item['avg_conflict_accuracy']:.3f} | {item['acl_leak_count']} | "
            f"{item['latency_p50_ms']:.0f} | {item['latency_p95_ms']:.0f} | "
            f"{item['avg_logical_api_calls_estimate']:.2f} |"
        )
    lines.extend([
        "",
        "## 保留判据",
        "",
        "C 方案只应在复杂任务的正确性、声明级引用覆盖/支持、错误前提或冲突处理上取得明确收益，且 ACL 泄漏为 0；否则应简化或删除固定团队。",
    ])
    return "\n".join(lines) + "\n"


def _select_dataset(mode: str) -> List[ComplexResearchQuery]:
    if mode == "full":
        return list(COMPLEX_RESEARCH_DATASET)
    selected: List[ComplexResearchQuery] = []
    seen: Dict[str, int] = {}
    for case in COMPLEX_RESEARCH_DATASET:
        count = seen.get(case.category, 0)
        if count < 1:
            selected.append(case)
            seen[case.category] = count + 1
    return selected


async def main() -> None:
    parser = argparse.ArgumentParser(description="复杂科研任务固定 Research Team A/B/C 评测")
    parser.add_argument("--dataset", choices=["quick", "full"], default="quick")
    parser.add_argument("--variants", default="A,B,C", help="逗号分隔：A,B,C")
    parser.add_argument("--output", default="data/复杂科研评测结果.json")
    parser.add_argument("--report", default="data/复杂科研评测报告.md")
    parser.add_argument("--case-id", default="", help="只运行指定 case_id，例如 C10")
    parser.add_argument("--limit", type=int, default=0, help="只运行前 N 条，0 表示不限制")
    parser.add_argument("--dry-run", action="store_true", help="只校验数据集和路由，不调用模型")
    parser.add_argument("--merge-existing", default="", help="保留已有报告中本次未运行的方案")
    parser.add_argument("--checkpoint", default="", help="每条完成后保存断点 JSON，重跑时自动续跑")
    args = parser.parse_args()

    variants = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    invalid = [item for item in variants if item not in VARIANT_LABELS]
    if invalid:
        parser.error(f"未知方案: {invalid}")

    cases = _select_dataset(args.dataset)
    if args.case_id:
        cases = [case for case in COMPLEX_RESEARCH_DATASET if case.case_id == args.case_id]
        if not cases:
            parser.error(f"不存在 case_id={args.case_id}")
    if args.limit > 0:
        cases = cases[:args.limit]
    if args.dry_run:
        from src.agent.agents.research_team import is_complex_research_task
        not_routed = [case.case_id for case in cases if not is_complex_research_task(case.query)]
        print(json.dumps({
            "dataset": args.dataset,
            "case_count": len(cases),
            "variants": variants,
            "not_routed_to_team": not_routed,
        }, ensure_ascii=False, indent=2))
        if not_routed:
            raise SystemExit(1)
        return

    # 与 main.py 一致，让直接运行评测脚本时也读取 config/.env 中的代理。
    from config.settings import get_settings
    settings = get_settings()
    if settings.http_proxy:
        os.environ["HTTP_PROXY"] = settings.http_proxy
        os.environ["http_proxy"] = settings.http_proxy
    if settings.https_proxy:
        os.environ["HTTPS_PROXY"] = settings.https_proxy
        os.environ["https_proxy"] = settings.https_proxy

    user = UserContext.anonymous()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_by_id: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_by_id = {
            row["case"]["case_id"]: row for row in checkpoint_payload.get("results", [])
        }

    results: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        existing_case = checkpoint_by_id.get(case.case_id)
        if existing_case and all(variant in existing_case.get("variants", {}) for variant in variants):
            print(f"[{index}/{len(cases)}] {case.case_id} 从断点恢复", flush=True)
            results.append(existing_case)
        else:
            print(f"[{index}/{len(cases)}] {case.case_id} {case.description or case.category}", flush=True)
            row = await evaluate_case(case, variants, user)
            if existing_case:
                row["variants"] = {**existing_case.get("variants", {}), **row["variants"]}
            results.append(row)
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(json.dumps({
                    "dataset": args.dataset,
                    "variants": variants,
                    "results": results,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    report_variants = list(variants)
    if args.merge_existing:
        existing_path = Path(args.merge_existing)
        if existing_path.exists():
            existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
            existing_by_id = {
                row["case"]["case_id"]: row for row in existing_payload.get("results", [])
            }
            updated_by_id = {row["case"]["case_id"]: row for row in results}
            for row in results:
                old = existing_by_id.get(row["case"]["case_id"])
                if old:
                    row["variants"] = {**old.get("variants", {}), **row["variants"]}
            # 单 case/limit 复测时保留未运行的旧 case，按原顺序替换本次结果。
            if len(results) < len(existing_payload.get("results", [])):
                results = [
                    updated_by_id.get(old["case"]["case_id"], old)
                    for old in existing_payload.get("results", [])
                ]
            report_variants = list(dict.fromkeys(
                list(existing_payload.get("variants", [])) + report_variants
            ))

    payload = {
        "dataset": args.dataset,
        # merge-existing 可能把单用例复测合并回完整结果集，此时应报告
        # 实际落盘的用例数，而不是本次执行的子集大小。
        "case_count": len(results),
        "variants": report_variants,
        "methodology_note": "token usage 不完整包含 CRAG 内部 grader；logical API calls 为可比较估计而非账单。",
        "aggregate": aggregate(results, report_variants),
        "results": results,
    }
    output = Path(args.output)
    report = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"JSON: {output}\nMarkdown: {report}")


if __name__ == "__main__":
    asyncio.run(main())
