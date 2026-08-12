#!/usr/bin/env python
"""冻结后的 Deep Research Blind Holdout 一次性运行器。

只比较 B（normal：Query Expansion + 单 Agent）与 C（显式 Deep Research）。
脚本拒绝覆盖已有结果，防止查看结果后重跑挑选。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrieval.acl_filter import UserContext
from tests.eval.deep_research_blind_dataset import BLIND_RESEARCH_DATASET
from tests.eval.run_research_team_eval import aggregate, aggregate_by_category, evaluate_case


OUTPUT = PROJECT_ROOT / "data" / "深度研究盲测结果.json"
AUTO_REPORT = PROJECT_ROOT / "data" / "深度研究盲测自动报告.md"
BLIND_FORM = PROJECT_ROOT / "data" / "深度研究盲评表.md"
MAPPING = PROJECT_ROOT / "data" / "深度研究盲评映射.json"
CHECKPOINT = PROJECT_ROOT / "data" / "深度研究盲测断点.json"
VARIANTS = ["B", "C"]


def _render_auto_report(payload: dict) -> str:
    lines = [
        "# Deep Research Blind Holdout 自动报告",
        "",
        f"- 运行时间：{payload['generated_at']}",
        f"- 冻结后新建样本：{payload['case_count']} 条",
        "- 比较：B = normal Query Expansion 单 Agent；C = 显式 Deep Research",
        "- 本报告只含自动指标，生产门禁还必须读取逐答案盲评。",
        "",
        "| 方案 | 成功 | 关键词覆盖 | 引用覆盖 | 引用支持 proxy | 文档召回 | 前提总体 | ACL 泄漏 | P50/P95(ms) | 平均调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = payload["aggregate"][variant]
        premise = item["premise_overall_accuracy"]
        premise_text = "—" if premise is None else f"{premise:.3f}"
        lines.append(
            f"| {variant} | {item['success_cases']}/{item['cases']} | "
            f"{item['avg_keyword_correctness']:.3f} | {item['avg_citation_coverage']:.3f} | "
            f"{item['avg_citation_support_rate']:.3f} | {item['avg_retrieved_doc_recall']:.3f} | "
            f"{premise_text} | {item['acl_leak_count']} | "
            f"{item['latency_p50_ms']:.0f}/{item['latency_p95_ms']:.0f} | "
            f"{item['avg_logical_api_calls_estimate']:.2f} |"
        )
    lines.extend(["", "## 按类别", ""])
    for category, values in payload["aggregate_by_category"].items():
        b, c = values["B"], values["C"]
        lines.append(
            f"- {category}：关键词 B/C={b['avg_keyword_correctness']:.3f}/{c['avg_keyword_correctness']:.3f}；"
            f"引用覆盖 B/C={b['avg_citation_coverage']:.3f}/{c['avg_citation_coverage']:.3f}；"
            f"召回 B/C={b['avg_retrieved_doc_recall']:.3f}/{c['avg_retrieved_doc_recall']:.3f}。"
        )
    lines.extend([
        "",
        "## 预先声明的生产门禁",
        "",
        "只有同时满足以下条件才开启显式 Deep Research：",
        "",
        "1. 18 条均成功，且 ACL 泄漏为 0；",
        "2. 盲评中至少一个明确类别（3 条）Deep 平均总分比 normal 高至少 1.0/6，且至少 2/3 题胜出；",
        "3. Deep 全集正确性和证据依据两个维度的平均分，任一不得比 normal 低 0.2 以上；",
        "4. 自动前提准确率不得低于 normal；",
        "5. 延迟与调用成本只报告，不用质量分掩盖，也不单独否决显式用户选择。",
        "",
        "未完成盲评前，不作上线结论。",
    ])
    return "\n".join(lines) + "\n"


def _render_blind_form(results: list[dict]) -> tuple[str, dict]:
    rng = random.Random("deep-research-holdout-frozen-2026-08-13")
    mapping: dict[str, dict[str, str]] = {}
    lines = [
        "# Deep Research 答案级盲评表",
        "",
        "> 评分前不要打开 `深度研究盲评映射.json`。候选顺序已逐题固定随机化。",
        "> 每个维度 0～2：正确性、完整性、证据依据。不要按文风、篇幅或是否像多 Agent 评分。",
        "",
        "- 正确性：0=关键结论错误；1=大体正确但有实质问题；2=关键结论正确。",
        "- 完整性：0=大部分关键点缺失；1=覆盖部分；2=覆盖全部或绝大多数预设关键点。",
        "- 证据依据：0=无依据/引用错；1=部分事实有依据；2=主要事实均有有效、就近依据且推断有标记。",
        "",
    ]
    for row in results:
        case = row["case"]
        variants = ["B", "C"]
        rng.shuffle(variants)
        aliases = {"候选甲": variants[0], "候选乙": variants[1]}
        mapping[case["case_id"]] = aliases
        lines.extend([
            f"## {case['case_id']} · {case['category']}",
            "",
            f"问题：{case['query']}",
            "",
            "预设关键点：" + "；".join(case["expected_key_points"]),
            "",
            "参考资料范围：" + "；".join(case["relevant_doc_ids"]),
            "",
        ])
        for alias in ("候选甲", "候选乙"):
            answer = row["variants"][aliases[alias]]["answer"] or "（无答案）"
            lines.extend([f"### {alias}", "", answer, ""])
        lines.extend([
            "| 候选 | 正确性(0-2) | 完整性(0-2) | 证据依据(0-2) | 总分(0-6) | 简短理由 |",
            "|---|---:|---:|---:|---:|---|",
            "| 候选甲 |  |  |  |  |  |",
            "| 候选乙 |  |  |  |  |  |",
            "",
        ])
    return "\n".join(lines) + "\n", mapping


async def main() -> None:
    existing = [path for path in (OUTPUT, AUTO_REPORT, BLIND_FORM, MAPPING) if path.exists()]
    if existing:
        raise SystemExit("Blind Holdout 已运行或已生成产物，拒绝覆盖：" + ", ".join(map(str, existing)))

    from config.settings import get_settings
    settings = get_settings()
    if settings.http_proxy:
        os.environ["HTTP_PROXY"] = settings.http_proxy
        os.environ["http_proxy"] = settings.http_proxy
    if settings.https_proxy:
        os.environ["HTTPS_PROXY"] = settings.https_proxy
        os.environ["https_proxy"] = settings.https_proxy

    started = time.perf_counter()
    user = UserContext.anonymous()
    results: list[dict] = []
    for index, case in enumerate(BLIND_RESEARCH_DATASET, 1):
        print(f"[{index}/18] {case.case_id} {case.category}", flush=True)
        row = await evaluate_case(case, VARIANTS, user)
        results.append(row)
        CHECKPOINT.write_text(json.dumps({
            "frozen_holdout": True,
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "dataset": "blind_holdout_frozen",
        "case_count": len(results),
        "variants": VARIANTS,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_duration_seconds": round(time.perf_counter() - started, 3),
        "configuration": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.dashscope_model,
            "embedding_model": settings.embedding_model,
            "reranker_provider": settings.reranker_provider,
            "reranker_model": settings.reranker_model,
        },
        "gate_definition_frozen_before_results": True,
        "aggregate": aggregate(results, VARIANTS),
        "aggregate_by_category": aggregate_by_category(results, VARIANTS),
        "results": results,
    }
    blind_form, mapping = _render_blind_form(results)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    AUTO_REPORT.write_text(_render_auto_report(payload), encoding="utf-8")
    BLIND_FORM.write_text(blind_form, encoding="utf-8")
    MAPPING.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_render_auto_report(payload))


if __name__ == "__main__":
    asyncio.run(main())
