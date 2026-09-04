"""项目知识的最小检索适配层。

项目知识已经经过 ResearchService 的生命周期和当前 ACL 校验，不写入向量库，
只在显式项目上下文中做有界的结构化补充检索。
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document


def _terms(value: str) -> set[str]:
    """提取中英文词和中文二元片段，避免依赖额外分词基础设施。"""
    text = str(value or "").lower()
    words = set(re.findall(r"[a-z0-9_]+", text))
    han = re.findall(r"[\u4e00-\u9fff]", text)
    words.update("".join(han[index:index + 2]) for index in range(len(han) - 1))
    return {item for item in words if item}


def _relevance(query: str, statement: str) -> float:
    query_terms = _terms(query)
    statement_terms = _terms(statement)
    if not query_terms or not statement_terms:
        return 0.0
    overlap = query_terms & statement_terms
    if not overlap:
        return 0.0
    # 结构化知识短且数量少，使用可解释的词项覆盖率，不调用 embedding。
    return min(0.99, len(overlap) / max(1, len(query_terms)))


def _record_to_document(record: dict[str, Any], score: float) -> Document:
    sources = list(record.get("sources") or [])
    root_source_ids = list(record.get("source_ids") or [])
    source_text = "\n".join(
        f"来源：{item.get('title', item.get('source_id', ''))}；"
        f"摘录：{item.get('excerpt', '')}；定位：{item.get('locator', '')}"
        for item in sources
    )
    metadata = {
        "source": f"project-knowledge:{record.get('id', '')}",
        "title": record.get("statement") or "项目可信知识",
        "doc_type": "project_knowledge",
        "project_id": record.get("project_id", ""),
        "knowledge_record_id": record.get("id", ""),
        "research_run_id": record.get("research_run_id", ""),
        "claim_id": record.get("claim_id", ""),
        "root_source_ids": root_source_ids,
        "knowledge_origin": "project_knowledge",
        "source_ids": root_source_ids,
        "version": record.get("version", 1),
        # 项目知识的 ACL 已由 ResearchService 按项目、Run、Evidence 三层复核；
        # 该标记供回答后的通用文档 ACL 过滤器识别，避免把项目成员误判为普通 project 文档。
        "project_knowledge_acl_verified": True,
        "visibility": "project",
    }
    content = str(record.get("statement") or "")
    if source_text:
        content += f"\n\n【根来源证据】\n{source_text}"
    return Document(page_content=content, metadata=metadata)


def retrieve_project_knowledge(
    query: str,
    project_id: str,
    user: Any,
    *,
    top_k: int = 2,
) -> list[tuple[Document, float]]:
    """读取当前项目 active 知识；任何权限或数据异常均 fail closed。"""
    if not str(project_id or "").strip() or not user or top_k <= 0:
        return []
    try:
        from src.api.services.research_service import research_service

        service_user = user
        if not isinstance(user, dict) and hasattr(user, "__dict__"):
            service_user = {
                key: getattr(user, key)
                for key in (
                    "user_id", "username", "role", "department", "department_name",
                    "department_path", "is_active",
                )
                if hasattr(user, key)
            }
        records = research_service.list_knowledge_records(
            str(project_id).strip(), service_user, status="active",
        )
    except Exception:
        # 项目知识是可选补充，不能因 Wiki 数据异常破坏 Raw Documents Only。
        return []

    ranked: list[tuple[Document, float]] = []
    for record in records:
        if record.get("status") != "active":
            continue
        if str(record.get("knowledge_origin") or "").strip() == "derived_only":
            continue
        if not record.get("source_ids") or not record.get("sources"):
            continue
        score = _relevance(query, str(record.get("statement") or ""))
        if score <= 0:
            continue
        ranked.append((_record_to_document(record, score), score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:max(0, int(top_k))]


def merge_project_knowledge(
    raw_results: list[tuple[Document, float]],
    query: str,
    project_id: str,
    user: Any,
    *,
    top_k: int = 2,
    limit: int = 5,
) -> list[tuple[Document, float]]:
    """在原始结果之后追加去重的项目知识，保证 raw source 不被覆盖。"""
    project_results = retrieve_project_knowledge(
        query, project_id, user, top_k=top_k,
    )
    if not project_results:
        return list(raw_results)[:limit]
    existing_ids = {
        str((doc.metadata or {}).get("knowledge_record_id"))
        for doc, _score in raw_results
        if (doc.metadata or {}).get("knowledge_record_id")
    }
    merged = list(raw_results)
    merged.extend(
        (doc, score) for doc, score in project_results
        if str((doc.metadata or {}).get("knowledge_record_id")) not in existing_ids
    )
    # 保留完整 raw top-k，再附加少量项目知识；这样评测时 raw recall 不会
    # 因为补充记录占位而下降，最终上下文上限由两者之和明确可见。
    return merged[:max(1, int(limit)) + max(0, int(top_k))]
