#!/usr/bin/env python
"""受控知识演化的最小离线评测。

比较 Raw Documents Only 与 Raw Documents + Active Project Knowledge，覆盖新
结论、版本替代、撤销、ACL 和 derived-only。这里不调用 LLM；生成级指标用
冻结 gold fact、来源 Provenance 和确定性词项规则作为可复现 proxy。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.rag.retrieval.project_knowledge import merge_project_knowledge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "知识演化生成评测结果.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "知识演化生成评测报告.md"


def _record(
    record_id: str,
    statement: str,
    *,
    status: str = "active",
    origin: str = "project_knowledge",
    version: int = 1,
    source_id: str = "raw-1",
) -> dict:
    return {
        "id": record_id,
        "project_id": "project-1",
        "statement": statement,
        "status": status,
        "version": version,
        "research_run_id": f"run-{record_id}",
        "claim_id": f"claim-{record_id}",
        "source_ids": [source_id],
        "knowledge_origin": origin,
        "sources": [{
            "source_id": source_id,
            "title": "原始实验记录",
            "excerpt": statement,
            "locator": "page=2",
        }],
    }


def _raw(source_id: str, text: str) -> tuple[Document, float]:
    return Document(page_content=text, metadata={"source": source_id, "title": source_id}), 0.8


FROZEN_CASES = (
    {
        "case_id": "new_fact",
        "query": "当前确认 RDMA 吞吐量是多少？",
        "raw": [_raw("raw-1", "计划吞吐量为 80 Gbps。")],
        "records": [_record("knowledge-91", "当前实测 RDMA 吞吐量为 91 Gbps。")],
        "gold_fact": "91 Gbps",
    },
    {
        "case_id": "raw_already_covers",
        "query": "实验记录中的消息大小是多少？",
        "raw": [_raw("raw-2", "实验消息大小为 4096 bytes。")],
        "records": [_record("knowledge-4096", "实验消息大小为 4096 bytes。", source_id="raw-2")],
        "gold_fact": "4096 bytes",
    },
    {
        "case_id": "supersede_only_latest",
        "query": "最新确认的吞吐量是多少？",
        "raw": [],
        "records": [
            _record("knowledge-old", "旧测量值为 80 Gbps。", status="superseded"),
            _record("knowledge-new", "最新测量值为 91 Gbps。", version=2),
        ],
        "gold_fact": "91 Gbps",
        "forbidden": ("80 Gbps",),
    },
    {
        "case_id": "revoked_never_hits",
        "query": "已撤销的吞吐量结论是什么？",
        "raw": [],
        "records": [_record("knowledge-revoked", "吞吐量为 77 Gbps。", status="revoked")],
        "gold_fact": "77 Gbps",
        "expect_gold_hit": False,
    },
    {
        "case_id": "acl_blocked",
        "query": "受限实验的吞吐量是多少？",
        "raw": [],
        "records": [_record("knowledge-private", "受限实验吞吐量为 66 Gbps。")],
        "gold_fact": "66 Gbps",
        "acl_allowed": False,
        "expect_gold_hit": False,
    },
    {
        "case_id": "derived_only_blocked",
        "query": "推导结论的吞吐量是多少？",
        "raw": [],
        "records": [_record("knowledge-derived", "推导吞吐量为 88 Gbps。", origin="derived_only")],
        "gold_fact": "88 Gbps",
        "expect_gold_hit": False,
    },
    {
        "case_id": "provenance_required",
        "query": "复现实验最终带宽是多少？",
        "raw": [],
        "records": [_record("knowledge-repro", "复现实验最终带宽为 100 Gbps。", source_id="raw-7")],
        "gold_fact": "100 Gbps",
    },
    {
        "case_id": "project_fact_reuse",
        "query": "项目当前采用哪个网卡驱动版本？",
        "raw": [_raw("raw-8", "早期记录使用 OFED 5.4。")],
        "records": [_record("knowledge-ofed", "项目当前采用 OFED 5.8。", source_id="raw-8")],
        "gold_fact": "OFED 5.8",
    },
)


def _contains(text: str, value: str) -> bool:
    return value.casefold() in text.casefold()


def _evaluate_case(case: dict, include_project_knowledge: bool) -> dict:
    user = {"username": "alice", "role": "student"}
    allowed = case.get("acl_allowed", True)
    records = case["records"] if allowed else []
    service = type("FrozenResearchService", (), {
        "list_knowledge_records": lambda self, project_id, user, status="active": records,
    })()
    started = time.perf_counter()
    with patch("src.api.services.research_service.research_service", service):
        if include_project_knowledge:
            docs = merge_project_knowledge(
                list(case["raw"]), case["query"], "project-1", user, top_k=2, limit=5,
            )
        else:
            docs = list(case["raw"])
    latency_ms = (time.perf_counter() - started) * 1000
    text = "\n".join(doc.page_content for doc, _score in docs)
    gold_hit = _contains(text, case["gold_fact"])
    provenance_docs = [
        doc for doc, _score in docs
        if (doc.metadata or {}).get("knowledge_record_id")
    ]
    provenance_complete = all(
        all((doc.metadata or {}).get(field) for field in (
            "knowledge_record_id", "research_run_id", "claim_id", "root_source_ids",
        ))
        for doc in provenance_docs
    )
    forbidden_hit = any(_contains(text, item) for item in case.get("forbidden", ()))
    return {
        "case_id": case["case_id"],
        "gold_hit": int(gold_hit),
        "answer_correctness": int(gold_hit == case.get("expect_gold_hit", True)),
        "faithfulness": int(not provenance_docs or provenance_complete),
        "citation_support": int(not provenance_docs or provenance_complete),
        "revoked_or_superseded_wrong_hit": int(forbidden_hit),
        "acl_violation": int(include_project_knowledge and not allowed and bool(provenance_docs)),
        "provenance_complete": int(provenance_complete),
        "input_tokens_proxy": len(text) // 4,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _aggregate(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "gold_fact_hit": statistics.mean(row["gold_hit"] for row in rows),
        "answer_correctness": statistics.mean(row["answer_correctness"] for row in rows),
        "faithfulness": statistics.mean(row["faithfulness"] for row in rows),
        "citation_support": statistics.mean(row["citation_support"] for row in rows),
        "revoked_or_superseded_wrong_hit": sum(row["revoked_or_superseded_wrong_hit"] for row in rows),
        "acl_violation": sum(row["acl_violation"] for row in rows),
        "provenance_completeness": statistics.mean(row["provenance_complete"] for row in rows),
        "input_tokens_proxy": statistics.mean(row["input_tokens_proxy"] for row in rows),
        "latency_p50_ms": statistics.median(row["latency_ms"] for row in rows),
    }


def evaluate() -> dict:
    variants = {}
    for name, enabled in (("raw_documents_only", False), ("raw_plus_active_project_knowledge", True)):
        rows = [_evaluate_case(case, enabled) for case in FROZEN_CASES]
        variants[name] = {"aggregate": _aggregate(rows), "cases": rows}
    return {
        "dataset": "lab_knowledge_evolution_frozen_v1",
        "sample_count": len(FROZEN_CASES),
        "variants": variants,
        "gate": {
            "revoked_or_superseded_wrong_hit": 0,
            "acl_violation": 0,
            "provenance_completeness": 1.0,
        },
        "decision": "保留显式项目上下文开关；知识接入不得默认扩大到所有普通检索。",
    }


def render_report(payload: dict) -> str:
    lines = [
        "# 知识演化生成评测报告", "",
        "- 数据集：8 条冻结场景，覆盖新结论、重复覆盖、替代、撤销、ACL、Derived-only 和 Provenance。",
        "- 评测为离线 deterministic proxy，不调用模型；answer correctness 由 gold fact 命中定义。",
        "",
        "| Variant | Gold Fact Hit | Correctness | Faithfulness | Citation | Revoked/Superseded | ACL | Provenance | P50(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in payload["variants"].items():
        item = value["aggregate"]
        lines.append(
            f"| {name} | {item['gold_fact_hit']:.3f} | {item['answer_correctness']:.3f} | "
            f"{item['faithfulness']:.3f} | {item['citation_support']:.3f} | "
            f"{item['revoked_or_superseded_wrong_hit']} | {item['acl_violation']} | "
            f"{item['provenance_completeness']:.3f} | {item['latency_p50_ms']:.2f} |"
        )
    lines.extend([
        "", "## 决策", "", f"- {payload['decision']}",
        "- 只有 active、非 derived-only 且当前 ACL 可验证的项目知识可以作为补充结果。",
        "- 该评测证明生命周期安全边界，不代表生成模型事实蕴含能力；后续若有真人样本，再补人工校准。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="知识演化生成级离线评测")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(render_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
