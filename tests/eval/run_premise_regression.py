#!/usr/bin/env python
"""用冻结 V2 的六条前提题做已知缺陷回归；不覆盖、不重新解释盲测。"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrieval.acl_filter import UserContext
from tests.eval.deep_research_v2_blind_dataset import BLIND_RESEARCH_V2_DATASET
from tests.eval.run_research_team_eval import aggregate, evaluate_case


OUTPUT = PROJECT_ROOT / "data" / "深度研究V2前提判断回归结果.json"
REPORT = PROJECT_ROOT / "data" / "深度研究V2前提判断回归报告.md"


def _render(payload: dict) -> str:
    summary = payload["aggregate"]["C"]
    lines = [
        "# 深度研究 V2 前提判断回归报告",
        "",
        "> 这是针对已知缺陷的定向回归，不是新盲测，也不能替代独立人工盲评。",
        "",
        f"- 运行时间：{payload['generated_at']}",
        f"- 成功样本：{summary['success_cases']}/{summary['cases']}",
        f"- 错误前提识别：{summary['false_premise_detection_accuracy']:.3f}",
        f"- 成立前提确认：{summary['supported_premise_confirmation_accuracy']:.3f}",
        f"- 前提总体准确率：{summary['premise_overall_accuracy']:.3f}",
        f"- ACL 泄漏：{summary['acl_leak_count']}",
        f"- P50/P95：{summary['latency_p50_ms']:.0f}/{summary['latency_p95_ms']:.0f} ms",
        "",
        "| 样本 | 预期 | 准确 | Reviewer 结论 |",
        "|---|---|---:|---|",
    ]
    for row in payload["results"]:
        result = row["variants"]["C"]
        review = (
            (result.get("research_trace") or {}).get("stages", {}).get("reviewer", {})
            .get("review_report", {})
        )
        lines.append(
            f"| {row['case']['case_id']} | {row['case']['premise_expectation']} | "
            f"{result.get('premise_accuracy', 0):.0f} | "
            f"{review.get('premise_assessment', '未记录')} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    from config.settings import get_settings

    settings = get_settings()
    for key, value in (("HTTP_PROXY", settings.http_proxy), ("HTTPS_PROXY", settings.https_proxy)):
        if value:
            os.environ[key] = value
            os.environ[key.lower()] = value

    cases = [case for case in BLIND_RESEARCH_V2_DATASET if case.category == "premise"]
    started = time.perf_counter()
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        results.append(await evaluate_case(case, ("C",), UserContext.anonymous()))
    payload = {
        "dataset": "known_v2_premise_regression",
        "disclaimer": "已知缺陷定向回归，不是新盲测，不替代独立人工盲评",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_duration_seconds": round(time.perf_counter() - started, 3),
        "aggregate": aggregate(results, ("C",)),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(_render(payload), encoding="utf-8")
    print(_render(payload))


if __name__ == "__main__":
    asyncio.run(main())
