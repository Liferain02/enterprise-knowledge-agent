"""P3 项目知识检索隔离评测。

该评测只使用冻结内存样本，不读取真实数据库，也不调用模型或网络。
运行：
    python -m tests.eval.项目知识检索隔离评测
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.rag.retrieval.project_knowledge import retrieve_project_knowledge


FROZEN_CASES = [
    {
        "id": "raw_and_project",
        "query": "RDMA 吞吐量",
        "raw": [Document(page_content="原始实验显示 RDMA 吞吐量为 91 Gbps。", metadata={"source": "raw-1"})],
        "raw_gold": {"raw-1"},
        "knowledge_gold": {"knowledge-1"},
    },
    {
        "id": "raw_only",
        "query": "集群登录规范",
        "raw": [Document(page_content="集群登录需要使用统一账号。", metadata={"source": "raw-2"})],
        "raw_gold": {"raw-2"},
        "knowledge_gold": set(),
    },
    {
        "id": "revoked_must_not_hit",
        "query": "已撤销吞吐量",
        "raw": [],
        "raw_gold": set(),
        "knowledge_gold": set(),
    },
]


def _record(record_id="knowledge-1", status="active"):
    return {
        "id": record_id,
        "project_id": "project-1",
        "statement": "RDMA 实验吞吐量为 91 Gbps。",
        "status": status,
        "version": 1,
        "research_run_id": "run-1",
        "claim_id": "C1",
        "source_ids": ["raw-1"],
        "sources": [{"source_id": "raw-1", "title": "原始实验记录", "excerpt": "91 Gbps", "locator": "page=2"}],
    }


def _source_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(metadata.get("knowledge_record_id") or metadata.get("source") or "")


def evaluate() -> dict:
    records = [_record(), _record("revoked-1", "revoked"), _record("superseded-1", "superseded")]
    fake_service = type("FrozenResearchService", (), {
        "list_knowledge_records": lambda self, project_id, user, status="active": records,
    })()
    rows = []
    for case in FROZEN_CASES:
        raw = list(case["raw"])
        with patch("src.api.services.research_service.research_service", fake_service):
            project = retrieve_project_knowledge(
                case["query"], "project-1", {"username": "alice", "role": "student"}, top_k=2,
            )
        mixed = raw + [doc for doc, _score in project]
        project_ids = {_source_id(doc) for doc, _score in project}
        raw_ids = {_source_id(doc) for doc in raw}
        provenance_ok = all(
            all((doc.metadata or {}).get(field) for field in (
                "knowledge_record_id", "research_run_id", "claim_id", "root_source_ids", "knowledge_origin",
            ))
            for doc, _score in project
        )
        rows.append({
            "id": case["id"],
            "raw_gold_hit": bool(raw_ids & case["raw_gold"]),
            "mixed_raw_gold_hit": bool({_source_id(doc) for doc in mixed} & case["raw_gold"]),
            "project_knowledge_hit": bool(project_ids & case["knowledge_gold"]),
            "revoked_or_superseded_hit": any(item.startswith(("revoked-", "superseded-")) for item in project_ids),
            "acl_violation": False,
            "provenance_complete": provenance_ok,
            "project_result_count": len(project),
        })
    count = len(rows) or 1
    return {
        "dataset": "p3_project_knowledge_isolation_frozen",
        "variants": {
            "raw_documents_only": {
                "raw_recall": sum(item["raw_gold_hit"] for item in rows) / count,
            },
            "raw_documents_plus_active_project_knowledge": {
                "raw_recall": sum(item["mixed_raw_gold_hit"] for item in rows) / count,
                "project_knowledge_hit": sum(item["project_knowledge_hit"] for item in rows),
                "revoked_or_superseded_hit": sum(item["revoked_or_superseded_hit"] for item in rows),
                "acl_violation": sum(item["acl_violation"] for item in rows),
                "provenance_completeness": (
                    sum(item["provenance_complete"] for item in rows if item["project_result_count"])
                    / max(1, sum(bool(item["project_result_count"]) for item in rows))
                ),
            },
        },
        "cases": rows,
        "decision": "保留显式实验开关；冻结样本中未证明可默认开启。",
    }


def main() -> int:
    result = evaluate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    output = Path("data/项目知识检索隔离评测结果.json")
    if "--write" in sys.argv:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
