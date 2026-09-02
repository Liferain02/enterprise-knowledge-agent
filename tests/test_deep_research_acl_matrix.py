"""Deep Research V2 独立、确定性的 ACL 角色矩阵与全阶段传播测试。"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.agent.agents import research_team as team
from src.rag.retrieval.acl_filter import UserContext, check_doc_access


def _user(name: str, role: str, project: str = "") -> UserContext:
    return UserContext(
        user_id=name,
        username=name,
        role=role,
        department=project,
        department_name=project,
        department_path=f"/实验室/{project}" if project else "",
    )


USERS = {
    "anonymous": UserContext.anonymous(),
    "ordinary_user": _user("ordinary", "employee", "project-alpha"),
    "project_member": _user("member", "senior_student", "project-alpha"),
    "project_lead": _user("lead", "pi", "project-alpha"),
    "non_member": _user("outsider", "employee", "project-gamma"),
}


DOCUMENTS = {
    "public_no_project": {
        "visibility": "public", "confidentiality": "internal",
        "department_restrict": [], "role_restrict": [],
    },
    "project_same_project": {
        "visibility": "project", "confidentiality": "internal",
        "department_restrict": ["project-alpha"], "role_restrict": [],
    },
    "project_other_project": {
        "visibility": "project", "confidentiality": "internal",
        "department_restrict": ["project-beta"], "role_restrict": [],
    },
    "project_no_project": {
        "visibility": "project", "confidentiality": "internal",
        "department_restrict": [], "role_restrict": [],
    },
    "restricted_same_project": {
        "visibility": "restricted", "confidentiality": "confidential",
        "department_restrict": ["project-alpha"], "role_restrict": ["pi", "admin"],
    },
    "restricted_other_project": {
        "visibility": "restricted", "confidentiality": "confidential",
        "department_restrict": ["project-beta"], "role_restrict": ["pi", "admin"],
    },
    "restricted_no_project": {
        "visibility": "restricted", "confidentiality": "confidential",
        "department_restrict": [], "role_restrict": ["pi", "admin"],
    },
}


# 显式矩阵是产品权限协议，不从被测函数动态推导。
EXPECTED = {
    "anonymous":                 [True, False, False, False, False, False, False],
    "ordinary_user":             [True, False, False, False, False, False, False],
    "project_member":            [True, True,  False, True,  False, False, False],
    "project_lead":              [True, True,  False, True,  True,  False, True],
    "non_member":                [True, False, False, False, False, False, False],
}


@pytest.mark.parametrize("user_name", list(USERS))
def test_acl_role_visibility_project_matrix(user_name):
    actual = [
        check_doc_access(metadata, USERS[user_name])
        for metadata in DOCUMENTS.values()
    ]
    assert actual == EXPECTED[user_name]


@pytest.mark.asyncio
async def test_denied_document_never_propagates_through_deep_research(monkeypatch):
    allowed_marker = "ALLOWED_PUBLIC_EVIDENCE"
    denied_marker = "DENIED_RESTRICTED_SECRET_9274"
    allowed = Document(
        page_content=f"公开结论 {allowed_marker}",
        metadata={
            "source": "/kb/公开资料.md", "title": "公开资料",
            **DOCUMENTS["public_no_project"],
        },
    )
    denied = Document(
        page_content=f"受限结论 {denied_marker}",
        metadata={
            "source": "/kb/受限资料.md", "title": "受限资料",
            **DOCUMENTS["restricted_same_project"],
        },
    )
    grade = SimpleNamespace(decision=SimpleNamespace(value="high"))
    monkeypatch.setattr(
        "src.agent.agents.knowledge._retrieve_documents",
        AsyncMock(return_value=(
            [(denied, 0.99), (allowed, 0.90)], grade, ["综合检索"],
        )),
    )
    monkeypatch.setattr(
        team,
        "_plan_subquestions",
        AsyncMock(return_value=(["综合检索"], {"input_tokens": 0, "output_tokens": 0})),
    )
    monkeypatch.setattr(
        team,
        "_run_analyst",
        AsyncMock(return_value=(
            team.AnalysisReport(claims=[team.Claim(
                claim_id="C1", text=f"公开资料支持结论 {allowed_marker}",
                claim_type="fact", source_ids=["S1"],
            )]),
            {"input_tokens": 0, "output_tokens": 0},
            100,
        )),
    )
    monkeypatch.setattr(
        team,
        "_invoke_structured",
        AsyncMock(return_value=(
            team.ReviewReport(decision="PASS", acl_verified=True),
            {"input_tokens": 0, "output_tokens": 0},
        )),
    )

    captured_generation_state = {}

    async def fake_generation(state):
        captured_generation_state.update(state)
        return {
            "final_answer": f"公开资料支持结论 {allowed_marker}[文档1]",
            "generation_metrics": {"llm_calls": 1, "input_tokens": 1, "output_tokens": 1, "elapsed_ms": 1},
        }

    monkeypatch.setattr("src.agent.agents.knowledge.generation_agent_node", fake_generation)

    state = {
        "messages": [HumanMessage(content="综合公开资料形成研究建议")],
        "user_context": USERS["project_member"],
    }
    state.update(await team.research_agent_node(state))
    state.update(await team.analyst_agent_node(state))
    state.update(await team.reviewer_agent_node(state))
    state.update(await team.deep_research_generation_node(state))

    serialized_surfaces = {
        "EvidencePackage": state["evidence_package"],
        "AnalysisReport": state["analysis_report"],
        "ReviewReport": state["review_report"],
        "retrieval_context": captured_generation_state["retrieval_context"],
        "research_trace": state["research_trace"],
        "source_cards": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in state["retrieved_docs"]
        ],
        "final_answer": state["final_answer"],
    }
    wire = json.dumps(serialized_surfaces, ensure_ascii=False, default=str)
    assert denied_marker not in wire
    assert "受限资料" not in wire
    assert "/kb/受限资料.md" not in wire
    assert allowed_marker in wire
    assert state["research_trace"]["failure_attribution"] == "none"


@pytest.mark.asyncio
async def test_revision_rechecks_acl_before_merging_new_evidence(monkeypatch):
    allowed = Document(
        page_content="允许的补充证据",
        metadata={"source": "补充公开.md", "title": "补充公开", **DOCUMENTS["public_no_project"]},
    )
    denied = Document(
        page_content="DENIED_DURING_REVISION",
        metadata={"source": "补充受限.md", "title": "补充受限", **DOCUMENTS["restricted_same_project"]},
    )
    grade = SimpleNamespace(decision=SimpleNamespace(value="high"))
    monkeypatch.setattr(
        "src.agent.agents.knowledge._retrieve_documents",
        AsyncMock(return_value=(
            [(denied, 0.99), (allowed, 0.90)], grade, ["补充查询"],
        )),
    )
    monkeypatch.setattr(
        team,
        "_run_analyst",
        AsyncMock(return_value=(team.AnalysisReport(), {"input_tokens": 0, "output_tokens": 0}, 0)),
    )
    base = team.EvidencePackage(original_question="验证问题")
    result = await team.research_revision_node({
        "evidence_package": base.model_dump(),
        "analysis_report": team.AnalysisReport().model_dump(),
        "review_report": team.ReviewReport(
            decision="NEED_MORE_EVIDENCE", targeted_queries=["补充查询"],
        ).model_dump(),
        "user_context": USERS["project_member"],
    })
    wire = json.dumps(result, ensure_ascii=False, default=str)
    assert "补充公开" in wire
    assert "DENIED_DURING_REVISION" not in wire
    assert "补充受限" not in wire
