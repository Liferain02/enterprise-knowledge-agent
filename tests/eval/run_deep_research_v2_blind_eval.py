#!/usr/bin/env python
"""冻结后的 Deep Research Blind Holdout V2 一次性自动评测器。

数据集模块在 V2 实现冻结后才允许创建。运行器拒绝覆盖正式产物；它只生成
自动指标、label-blind 评分包和密封映射，不做人工评分、不作生产收益结论。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrieval.acl_filter import UserContext
from tests.eval.run_research_team_eval import aggregate, aggregate_by_category, evaluate_case


OUTPUT = PROJECT_ROOT / "data" / "深度研究V2盲测结果.json"
AUTO_REPORT = PROJECT_ROOT / "data" / "深度研究V2盲测自动报告.md"
BLIND_FORM = PROJECT_ROOT / "data" / "深度研究V2盲评表.md"
MAPPING = PROJECT_ROOT / "data" / "深度研究V2盲评密封映射.json"
CHECKPOINT = PROJECT_ROOT / "data" / "深度研究V2盲测断点.json"
GATE = PROJECT_ROOT / "data" / "深度研究V2生产门槛定义.json"
VARIANTS = ("B", "C")


_DEEP_FIXED_HEADINGS = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[1-7][.、]\s*)?"
    r"(?:研究问题|已有证据|关键事实|冲突/不确定项|推断|下一步研究建议|Sources)\s*[:：]?\s*$",
    re.IGNORECASE,
)


def normalize_for_blind_packet(answer: str) -> str:
    """只去除方案特有标题，不改事实、引用、句子顺序或结论。"""

    lines = []
    for line in (answer or "").splitlines():
        if _DEEP_FIXED_HEADINGS.match(line):
            continue
        # 其他 Markdown 标题只去掉展示标记，保留标题文本。
        lines.append(re.sub(r"^\s*#{1,6}\s+", "", line))
    return "\n".join(lines).strip()


def _render_blind_form(results: list[dict]) -> tuple[str, dict]:
    rng = random.Random("deep-research-v2-frozen-label-blind-2026-08-13")
    mapping: dict[str, dict[str, str]] = {}
    lines = [
        "# 深度研究 V2 独立人工盲评表",
        "",
        "> 这是 label-blind review，不是 double blind；答案仍可能因风格被识别。",
        "> 在评分完成并保存前，不要打开 `深度研究V2盲评密封映射.json`。",
        "> 候选已逐题固定随机化；展示中移除了方案名、Agent、trace、延迟、调用次数和固定 Research Brief 标题。",
        "> 标题归一化没有调用 LLM，也没有改变事实内容、引用、句子顺序或结论。",
        "",
        "每项 0～2 分：正确性、完整性、证据依据。另填写胜者（甲/乙/平局）和简短理由。",
        "",
    ]
    for row in results:
        case = row["case"]
        variants = list(VARIANTS)
        rng.shuffle(variants)
        aliases = {"候选甲": variants[0], "候选乙": variants[1]}
        mapping[case["case_id"]] = aliases
        lines.extend([
            f"## {case['case_id']} · {case['category']}", "",
            f"问题：{case['query']}", "",
            "预设关键点：" + "；".join(case["expected_key_points"]), "",
            "参考资料范围：" + "；".join(case["relevant_doc_ids"]), "",
        ])
        for alias in ("候选甲", "候选乙"):
            answer = row["variants"][aliases[alias]].get("answer") or "（无答案）"
            lines.extend([f"### {alias}", "", normalize_for_blind_packet(answer), ""])
        lines.extend([
            "| 候选 | 正确性(0-2) | 完整性(0-2) | 证据依据(0-2) | 总分(0-6) |",
            "|---|---:|---:|---:|---:|",
            "| 候选甲 |  |  |  |  |",
            "| 候选乙 |  |  |  |  |",
            "", "胜者（甲/乙/平局）：", "", "简短理由：", "",
        ])
    return "\n".join(lines) + "\n", mapping


def _render_report(payload: dict) -> str:
    lines = [
        "# 深度研究 V2 Blind Holdout 自动报告", "",
        f"- 运行时间：{payload['generated_at']}",
        f"- 冻结后新建样本：{payload['case_count']} 条",
        "- B = normal；C = 显式 Deep。", 
        "- 本报告不含独立人工评分，因此不能给出最终生产收益结论。", "",
        "| 方案 | 成功 | 关键点覆盖 | 引用覆盖 | 引用支持 proxy | 文档召回 | 前提总体 | ACL | P50/P95(ms) | 输入/输出 Token | 调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = payload["aggregate"][variant]
        premise = item["premise_overall_accuracy"]
        lines.append(
            f"| {variant} | {item['success_cases']}/{item['cases']} | "
            f"{item['avg_keyword_correctness']:.3f} | {item['avg_citation_coverage']:.3f} | "
            f"{item['avg_citation_support_rate']:.3f} | {item['avg_retrieved_doc_recall']:.3f} | "
            f"{('—' if premise is None else f'{premise:.3f}')} | {item['acl_leak_count']} | "
            f"{item['latency_p50_ms']:.0f}/{item['latency_p95_ms']:.0f} | "
            f"{item['avg_input_tokens']:.0f}/{item['avg_output_tokens']:.0f} | "
            f"{item['avg_logical_api_calls_estimate']:.2f} |"
        )
    lines.extend(["", "## 按类别", ""])
    for category, variants in payload["aggregate_by_category"].items():
        b, c = variants["B"], variants["C"]
        lines.append(
            f"- {category}：关键点 B/C={b['avg_keyword_correctness']:.3f}/{c['avg_keyword_correctness']:.3f}；"
            f"文档召回={b['avg_retrieved_doc_recall']:.3f}/{c['avg_retrieved_doc_recall']:.3f}；"
            f"引用支持 proxy={b['avg_citation_support_rate']:.3f}/{c['avg_citation_support_rate']:.3f}。"
        )
    deep = payload["aggregate"]["C"]
    lines.extend([
        "", "## Deep 阶段指标", "",
        f"- Researcher expected evidence coverage：{deep['avg_researcher_coverage']:.3f}",
        f"- Analyst claim coverage：{deep['avg_analyst_claim_coverage']:.3f}",
        f"- Reviewer drop rate：{deep['avg_reviewer_drop_rate']:.3f}",
        f"- Final validated-claim coverage proxy：{deep['avg_final_validated_claim_coverage']:.3f}",
        f"- Failure attribution：`{json.dumps(deep['failure_attribution'], ensure_ascii=False)}`",
        "", "## 当前状态", "",
        "自动评测已完成；独立人工 label-blind review 待完成。密封映射在人工评分前不得打开。",
    ])
    return "\n".join(lines) + "\n"


async def main() -> None:
    if not GATE.exists():
        raise SystemExit("缺少运行前冻结的 V2 gate definition")
    existing = [path for path in (OUTPUT, AUTO_REPORT, BLIND_FORM, MAPPING) if path.exists()]
    if existing:
        raise SystemExit("Blind V2 已运行或产物已存在，拒绝覆盖：" + ", ".join(map(str, existing)))

    # 延迟导入确保“实现冻结后才创建数据集”的流程可被文件时间与清单审计。
    from tests.eval.deep_research_v2_blind_dataset import BLIND_RESEARCH_V2_DATASET
    from config.settings import get_settings
    settings = get_settings()
    for key, value in (("HTTP_PROXY", settings.http_proxy), ("HTTPS_PROXY", settings.https_proxy)):
        if value:
            os.environ[key] = value
            os.environ[key.lower()] = value

    started = time.perf_counter()
    results: list[dict] = []
    user = UserContext.anonymous()
    for index, case in enumerate(BLIND_RESEARCH_V2_DATASET, 1):
        print(f"[{index}/{len(BLIND_RESEARCH_V2_DATASET)}] {case.case_id} {case.category}", flush=True)
        results.append(await evaluate_case(case, VARIANTS, user))
        CHECKPOINT.write_text(json.dumps({
            "dataset": "blind_holdout_v2_frozen",
            "status": "in_progress",
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "dataset": "blind_holdout_v2_frozen",
        "case_count": len(results),
        "variants": list(VARIANTS),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_duration_seconds": round(time.perf_counter() - started, 3),
        "configuration": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.dashscope_model,
            "embedding_model": settings.embedding_model,
            "reranker_provider": settings.reranker_provider,
            "reranker_model": settings.reranker_model,
        },
        "gate_definition_path": str(GATE.relative_to(PROJECT_ROOT)),
        "human_review_status": "pending_independent_review",
        "blindness": "label-blind but potentially style-identifiable",
        "aggregate": aggregate(results, VARIANTS),
        "aggregate_by_category": aggregate_by_category(results, VARIANTS),
        "results": results,
    }
    blind_form, mapping = _render_blind_form(results)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    AUTO_REPORT.write_text(_render_report(payload), encoding="utf-8")
    BLIND_FORM.write_text(blind_form, encoding="utf-8")
    MAPPING.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    CHECKPOINT.unlink(missing_ok=True)
    print(_render_report(payload))


if __name__ == "__main__":
    asyncio.run(main())
