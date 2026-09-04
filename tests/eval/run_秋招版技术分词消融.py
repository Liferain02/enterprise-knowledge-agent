#!/usr/bin/env python
"""秋招版 technical-aware BM25 分词消融。

这是 P3 的隔离 POC：只替换 Hybrid + RRF 的 BM25 tokenizer，不改变线上
默认检索链路。三种 tokenizer 在同一冻结 Benchmark 上运行，结果用于判断
技术精确词是否值得继续投入。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path
from typing import Callable

import jieba

from tests.eval.run_秋招版检索评测 import BenchmarkRunner, DEFAULT_OUTPUT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POC_OUTPUT = PROJECT_ROOT / "data" / "秋招版技术分词消融结果.json"
DEFAULT_POC_REPORT = PROJECT_ROOT / "data" / "秋招版技术分词消融报告.md"
TOKENIZER_NAMES = ("current", "search", "search_technical")
TECHNICAL_CATEGORY = "technical_exact_token"
TECHNICAL_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_./-])[A-Za-z0-9_./-]+(?![A-Za-z0-9_./-])")


def _jieba_tokens(text: str) -> list[str]:
    return [token.lower() for token in jieba.cut(text) if token.strip()]


def _search_tokens(text: str) -> list[str]:
    return [token.lower() for token in jieba.cut_for_search(text) if token.strip()]


def _search_technical_tokens(text: str) -> list[str]:
    tokens = _search_tokens(text)
    for token in TECHNICAL_TOKEN_RE.findall(text):
        normalized = token.lower()
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "current": _jieba_tokens,
    "search": _search_tokens,
    "search_technical": _search_technical_tokens,
}


class TokenizerHybridRunner(BenchmarkRunner):
    def __init__(self, tokenizer: Callable[[str], list[str]], top_k: int = 5):
        super().__init__(top_k=top_k)
        self.hybrid._tokenize = tokenizer  # type: ignore[method-assign]
        # 强制 POC 使用新的 tokenizer 重建 BM25 snapshot。
        self.hybrid.invalidate_bm25_index()


async def run_poc(top_k: int = 5) -> dict:
    outputs = {}
    for name in TOKENIZER_NAMES:
        runner = TokenizerHybridRunner(TOKENIZERS[name], top_k=top_k)
        payload = await runner.run(("hybrid_rrf",))
        outputs[name] = payload
    baseline = outputs["current"]["aggregate"]["hybrid_rrf"]
    technical_baseline = baseline["category_metrics"][TECHNICAL_CATEGORY]
    comparison = {}
    for name, payload in outputs.items():
        aggregate = payload["aggregate"]["hybrid_rrf"]
        technical = aggregate["category_metrics"][TECHNICAL_CATEGORY]
        comparison[name] = {
            "overall": aggregate,
            "technical": technical,
            "technical_mrr_delta": technical["mrr_at_5"] - technical_baseline["mrr_at_5"],
            "overall_mrr_delta": aggregate["mrr_at_5"] - baseline["mrr_at_5"],
            "overall_hit_at_5_delta": aggregate["hit_at_5"] - baseline["hit_at_5"],
        }
    return {
        "dataset": "lab_retrieval_benchmark_v1",
        "top_k": top_k,
        "variants": comparison,
        "gate": {
            "technical_mrr_delta_min": 0.03,
            "overall_mrr_regression_max": 0.02,
            "acl_violation_required": 0,
            "latency_change": "需结合 P50/P95 判断，不以单次抖动做结论",
        },
        "decision": _decision(comparison),
    }


def _decision(comparison: dict) -> dict:
    baseline = comparison["current"]
    eligible = []
    for name, item in comparison.items():
        if name == "current":
            continue
        overall = item["overall"]
        if (
            item["technical_mrr_delta"] >= 0.03
            and item["overall_mrr_delta"] >= -0.02
            and overall["acl_violation"] == 0
        ):
            eligible.append(name)
    return {
        "eligible_for_online_poc": eligible,
        "recommendation": (
            "仅继续验证达到门槛的 tokenizer，暂不改变线上默认"
            if eligible
            else "无 tokenizer 同时达到预设门槛，保留 current tokenizer 并停止该方向"
        ),
        "baseline_mrr_at_5": baseline["overall"]["mrr_at_5"],
    }


def render_report(payload: dict) -> str:
    lines = [
        "# 秋招版技术分词消融报告",
        "",
        "- 目的：验证 technical exact token 子集是否需要 technical-aware BM25。",
        "- 范围：同一 28 条冻结集、Hybrid + RRF、文档级 Gold；只改评测 tokenizer。",
        "- 门槛：技术子集 MRR@5 至少提升 0.03；全量 MRR 回退不超过 0.02；ACL violation 必须为 0。",
        "",
        "| Tokenizer | 技术 MRR@5 | 技术 Delta | 全量 MRR@5 | 全量 Delta | Hit@5 | ACL | P50/P95(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in payload["variants"].items():
        overall = item["overall"]
        lines.append(
            f"| {name} | {item['technical']['mrr_at_5']:.3f} | {item['technical_mrr_delta']:+.3f} | "
            f"{overall['mrr_at_5']:.3f} | {item['overall_mrr_delta']:+.3f} | "
            f"{overall['hit_at_5']:.3f} | {overall['acl_violation']} | "
            f"{overall['latency_p50_ms']:.0f}/{overall['latency_p95_ms']:.0f} |"
        )
    lines.extend([
        "",
        "## 决策",
        "",
        f"- {payload['decision']['recommendation']}。",
        "- 该结果只代表当前 Chroma 快照，不能外推为线上质量保证。",
        "- 若要合入线上，必须先对通过门槛的方案做独立回归并重新评估延迟。",
    ])
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="技术精确词 BM25 tokenizer 消融")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_POC_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_POC_REPORT)
    args = parser.parse_args()
    payload = await run_poc(top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(render_report(payload))


if __name__ == "__main__":
    asyncio.run(main())
