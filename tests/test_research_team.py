"""固定复杂科研团队的协议、路由、ACL 与循环上限测试。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.agent.agents import planner as planner_module
from src.agent.agents import research_team as team
from src.agent.graph import create_multi_agent_graph
from src.rag.retrieval.acl_filter import UserContext
from tests.eval.complex_research_dataset import COMPLEX_RESEARCH_DATASET


def _student() -> UserContext:
    return UserContext(
        user_id="u-1",
        username="student",
        role="student",
        department="p1",
        department_name="项目一",
        department_path="/实验室/项目一",
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("对比 RDMA 和 TCP 的差异", False),
        ("有哪些 RDMA 实验要求", False),
        ("综合比较多个项目的论文和实验记录，并基于证据给出下一步研究建议", True),
        ("分析新旧实验记录的冲突和时间演化，判断前提是否成立", True),
        ("跨研究方向比较两种方案，并结合组会纪要形成研究建议", True),
    ],
)
def test_complex_research_task_boundary(query, expected):
    assert team.is_complex_research_task(query) is expected


def test_complex_research_eval_dataset_has_required_size_categories_and_routes():
    assert 30 <= len(COMPLEX_RESEARCH_DATASET) <= 50
    assert {case.category for case in COMPLEX_RESEARCH_DATASET} == {
        "cross_scope", "multi_evidence", "temporal_conflict", "recommendation", "false_premise",
    }
    assert all(team.is_complex_research_task(case.query) for case in COMPLEX_RESEARCH_DATASET)


def test_production_team_gate_keeps_only_proven_false_premise_scenario():
    assert team.should_use_research_team(
        "综合多份论文和实验记录，验证 RDMA 在所有负载下都优于本地内存这一前提"
    ) is True
    assert team.should_use_research_team(
        "综合比较多个项目的论文和实验记录，并基于证据给出下一步研究建议"
    ) is False


@pytest.mark.asyncio
async def test_planner_remains_single_entry_and_routes_only_real_research_tasks(monkeypatch):
    monkeypatch.setattr(planner_module, "get_llm", lambda: (_ for _ in ()).throw(
        AssertionError("窄规则命中时不应调用 Planner LLM")
    ))
    team_state = await planner_module.planner_node({
        "messages": [HumanMessage(content=(
            "综合多份论文和实验记录，验证 RDMA 在所有负载下都优于本地内存这一前提"
        ))]
    })
    assert team_state["use_research_team"] is True
    assert planner_module.route_from_planner(team_state) == "research_agent"

    general_research_state = await planner_module.planner_node({
        "messages": [HumanMessage(content=(
            "综合比较多个项目的论文和实验记录，并基于证据给出下一步研究建议"
        ))]
    })
    assert general_research_state.get("use_research_team") is not True
    assert planner_module.route_from_planner(general_research_state) == "retrieval_agent"

    ordinary_state = await planner_module.planner_node({
        "messages": [HumanMessage(content="对比 RDMA 和 TCP 的差异")]
    })
    assert ordinary_state.get("use_research_team") is not True
    assert planner_module.route_from_planner(ordinary_state) == "retrieval_agent"


def test_graph_has_fixed_team_without_supervisor_or_review_loop():
    graph = create_multi_agent_graph().compile().get_graph()
    mermaid = graph.draw_mermaid()

    for node in (
        "research_agent", "analyst_agent", "reviewer_agent",
        "research_revision", "research_team_finalizer",
    ):
        assert node in mermaid
    assert "supervisor" not in mermaid
    assert "reviewer_agent -.-> research_revision" in mermaid
    assert "research_revision --> research_team_finalizer" in mermaid
    assert "research_revision --> reviewer_agent" not in mermaid


def test_review_report_only_accepts_three_decisions():
    with pytest.raises(ValidationError):
        team.ReviewReport(decision="DISCUSS")


def test_analysis_normalization_moves_transient_limitation_out_of_claims():
    package = team.EvidencePackage(original_question="问题")
    report = team.AnalysisReport(claims=[team.Claim(
        claim_id="", text="缺少定量实验数据", claim_type="limitation", source_ids=[],
    )])
    normalized = team._normalize_analysis(report, package)
    assert normalized.claims == []
    assert normalized.limitations == ["缺少定量实验数据"]


@pytest.mark.asyncio
async def test_researcher_rechecks_acl_and_drops_restricted_evidence(monkeypatch):
    public_doc = Document(
        page_content="公开实验结论",
        metadata={"source": "/private/公开.md", "title": "公开资料", "visibility": "public"},
    )
    restricted_doc = Document(
        page_content="保密实验结论",
        metadata={"source": "/private/保密.md", "title": "保密资料", "visibility": "restricted"},
    )
    grade = SimpleNamespace(decision=SimpleNamespace(value="high"))
    pipeline = SimpleNamespace(retrieve=AsyncMock(return_value=(
        [(public_doc, 0.9), (restricted_doc, 0.95)], grade, [],
    )))
    monkeypatch.setattr(
        "src.rag.evaluation.retrieval_grader.get_corrective_rag_pipeline",
        lambda: pipeline,
    )
    monkeypatch.setattr(
        team,
        "_plan_subquestions",
        AsyncMock(return_value=(["比较实验结论"], {"input_tokens": 10, "output_tokens": 5})),
    )

    result = await team.research_agent_node({
        "messages": [HumanMessage(content="综合比较多个项目实验并形成研究建议")],
        "user_context": _student(),
    })
    package = team.EvidencePackage.model_validate(result["evidence_package"])

    assert [item.title for item in package.evidences] == ["公开资料"]
    assert package.evidences[0].source == "公开.md"
    assert "/private" not in package.model_dump_json(ensure_ascii=False)
    assert package.acl_checked is True


@pytest.mark.asyncio
async def test_reviewer_overrides_pass_when_claim_has_invalid_citation(monkeypatch):
    package = team.EvidencePackage(
        original_question="问题",
        subquestions=["子问题"],
        evidences=[team.EvidenceItem(
            source_id="S1",
            subquestion="子问题",
            title="资料",
            source="资料.md",
            excerpt="证据",
        )],
    )
    analysis = team.AnalysisReport(claims=[team.Claim(
        claim_id="C1",
        text="一个没有有效引用的事实",
        claim_type="fact",
        source_ids=["S99"],
    )])
    monkeypatch.setattr(
        team,
        "_invoke_structured",
        AsyncMock(return_value=(team.ReviewReport(decision="PASS"), {
            "input_tokens": 10, "output_tokens": 2,
        })),
    )

    result = await team.reviewer_agent_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
    })
    report = team.ReviewReport.model_validate(result["review_report"])
    assert report.decision == "REVISE"
    assert report.items[0].issue_type == "invalid_source"


@pytest.mark.asyncio
async def test_reviewer_explicitly_flags_challenged_absolute_premise(monkeypatch):
    package = team.EvidencePackage(
        original_question="验证 RDMA 在所有负载下都优于本地内存这一前提",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="验证前提", title="规范", source="规范.md",
            excerpt="不同 NUMA 绑定策略会影响延迟和带宽",
        )],
    )
    analysis = team.AnalysisReport(
        claims=[team.Claim(
            claim_id="C1", text="不同绑定会影响性能", claim_type="fact", source_ids=["S1"],
        )],
        draft_answer="不同绑定会影响性能[S1]。",
    )
    monkeypatch.setattr(
        team,
        "_invoke_structured",
        AsyncMock(return_value=(team.ReviewReport(decision="PASS"), {
            "input_tokens": 1, "output_tokens": 1,
        })),
    )

    result = await team.reviewer_agent_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
    })
    report = team.ReviewReport.model_validate(result["review_report"])
    assert report.false_premise_detected is True
    assert report.decision == "REVISE"
    assert any(item.issue_type == "false_premise" for item in report.items)


def test_reviewer_can_trigger_at_most_one_revision():
    report = team.ReviewReport(decision="NEED_MORE_EVIDENCE")
    assert team.route_after_reviewer({
        "review_report": report.model_dump(),
        "research_revision_count": 0,
    }) == "research_revision"
    assert team.route_after_reviewer({
        "review_report": report.model_dump(),
        "research_revision_count": 1,
    }) == "research_team_finalizer"


@pytest.mark.asyncio
async def test_team_finalizer_distinguishes_fact_inference_and_recommendation():
    package = team.EvidencePackage(
        original_question="问题",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="问题", title="资料", source="资料.md", excerpt="证据",
        )],
    )
    analysis = team.AnalysisReport(
        claims=[
            team.Claim(claim_id="C1", text="资料明确记录了结果", claim_type="fact", source_ids=["S1"]),
            team.Claim(claim_id="C2", text="可推断瓶颈在网络", claim_type="inference", source_ids=["S1"]),
            team.Claim(claim_id="C3", text="建议补做消融实验", claim_type="recommendation", source_ids=["S1"]),
        ],
    )
    result = await team.research_team_finalizer_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
        "review_report": team.ReviewReport(decision="REVISE").model_dump(),
        "research_revision_count": 1,
    })

    assert "资料事实" in result["final_answer"]
    assert "模型推断" in result["final_answer"]
    assert "研究建议" in result["final_answer"]
    assert result["final_answer"].count("[S1]") == 3
    assert result["used_agent"] == "research_team"


@pytest.mark.asyncio
async def test_team_finalizer_uses_claim_citations_even_when_passed_draft_is_unquoted():
    package = team.EvidencePackage(
        original_question="分析两份资料是否冲突",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="问题", title="资料", source="资料.md", excerpt="证据",
        )],
    )
    analysis = team.AnalysisReport(
        claims=[team.Claim(
            claim_id="C1", text="资料只支持有限结论。", claim_type="fact", source_ids=["S1"],
        )],
        draft_answer="这是没有规范引用的自由草稿。",
    )
    result = await team.research_team_finalizer_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
        "review_report": team.ReviewReport(decision="PASS").model_dump(),
        "research_revision_count": 0,
    })
    assert "自由草稿" not in result["final_answer"]
    assert "有限结论[S1]。" in result["final_answer"]
    assert "冲突核验" in result["final_answer"]
