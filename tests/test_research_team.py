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
from tests.eval import run_research_team_eval as eval_module


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
        "cross_scope", "multi_evidence", "temporal_conflict", "recommendation",
        "false_premise", "supported_premise",
    }
    assert all(team.is_complex_research_task(case.query) for case in COMPLEX_RESEARCH_DATASET)
    assert sum(case.premise_expectation == "false" for case in COMPLEX_RESEARCH_DATASET) == 4
    assert sum(case.premise_expectation == "supported" for case in COMPLEX_RESEARCH_DATASET) == 5


@pytest.mark.parametrize("citation", ["[文档1]", "[文档 1]", "[文档   1]", "[S1]", "[S 1]"])
def test_citation_metrics_accept_real_output_whitespace_variants(citation):
    answer = f"RDMA 实验必须记录驱动版本{citation}。"
    docs = [Document(page_content="RDMA 实验必须记录驱动版本和固件版本")]
    assert eval_module._citation_coverage(answer) == 1.0
    assert eval_module._citation_support_single(answer, docs) == 1.0


def test_premise_metric_is_scoped_and_distinguishes_false_from_supported():
    ordinary = COMPLEX_RESEARCH_DATASET[0]
    false_case = next(case for case in COMPLEX_RESEARCH_DATASET if case.case_id == "C33")
    supported_case = next(case for case in COMPLEX_RESEARCH_DATASET if case.case_id == "C37")

    assert eval_module._premise_accuracy("普通答案", ordinary) is None
    assert eval_module._premise_accuracy("当前证据不支持该前提，该前提不成立。", false_case) == 1.0
    assert eval_module._premise_accuracy("不成立。资料并未证明这个判断。", false_case) == 1.0
    assert eval_module._premise_accuracy("当前证据支持该前提成立。", false_case) == 0.0
    assert eval_module._premise_accuracy("当前证据支持该前提成立。", supported_case) == 1.0
    assert eval_module._premise_accuracy("是的，根据制度该要求适用于所有共享节点。", supported_case) == 1.0
    assert eval_module._premise_accuracy("当前证据不足，无法确认该前提。", supported_case) == 0.0


def test_unproven_team_has_no_production_gate():
    assert not hasattr(team, "should_use_research_team")


@pytest.mark.parametrize("quantifier", ["所有", "任何", "每次", "必然", "始终"])
def test_absolute_premise_trigger_covers_supported_and_false_quantifiers(quantifier):
    assert team._contains_challenged_absolute_premise(
        f"综合两份制度，验证{quantifier}共享资源改动都必须登记这一前提是否成立"
    ) is True


@pytest.mark.asyncio
async def test_planner_remains_single_entry_and_routes_only_real_research_tasks(monkeypatch):
    monkeypatch.setattr(planner_module, "get_llm", lambda: (_ for _ in ()).throw(
        AssertionError("规则命中时不应调用 Planner LLM")
    ))
    premise_state = await planner_module.planner_node({
        "messages": [HumanMessage(content=(
            "综合多份论文和实验记录，验证 RDMA 在所有负载下都优于本地内存这一前提"
        ))]
    })
    assert premise_state.get("use_research_team") is not True
    assert planner_module.route_from_planner(premise_state) == "retrieval_agent"

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


def test_production_graph_excludes_unproven_team_supervisor_and_review_loop():
    graph = create_multi_agent_graph().compile().get_graph()
    mermaid = graph.draw_mermaid()

    for node in (
        "research_agent", "analyst_agent", "reviewer_agent",
        "research_revision", "research_team_finalizer",
    ):
        assert node not in mermaid
    assert "supervisor" not in mermaid
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
async def test_reviewer_requires_explicit_assessment_for_challenged_absolute_premise(monkeypatch):
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
    assert report.premise_assessment == "insufficient"
    assert report.false_premise_detected is False
    assert report.decision == "REVISE"
    assert any(item.issue_type == "false_premise" for item in report.items)


@pytest.mark.asyncio
async def test_reviewer_does_not_reject_supported_absolute_premise(monkeypatch):
    package = team.EvidencePackage(
        original_question="验证所有长时间任务都必须提前登记这一前提是否成立",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="验证前提", title="集群说明", source="集群说明.md",
            excerpt="所有长时间运行任务必须提前登记",
        )],
    )
    analysis = team.AnalysisReport(claims=[team.Claim(
        claim_id="C1", text="制度要求所有长任务提前登记", claim_type="fact", source_ids=["S1"],
    )])
    monkeypatch.setattr(
        team,
        "_invoke_structured",
        AsyncMock(return_value=(team.ReviewReport(
            decision="PASS", premise_assessment="supported",
        ), {"input_tokens": 1, "output_tokens": 1})),
    )

    result = await team.reviewer_agent_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
    })
    report = team.ReviewReport.model_validate(result["review_report"])
    assert report.decision == "PASS"
    assert report.premise_assessment == "supported"
    assert report.false_premise_detected is False

    final = await team.research_team_finalizer_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
        "review_report": report.model_dump(),
    })
    assert "当前证据支持该前提成立" in final["final_answer"]
    assert "可能不成立" not in final["final_answer"]


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
@pytest.mark.parametrize(
    "claim",
    [
        team.Claim(claim_id="C1", text="错误 S99 声明", claim_type="fact", source_ids=["S99"]),
        team.Claim(claim_id="C2", text="没有引用的资料事实", claim_type="fact", source_ids=[]),
    ],
)
async def test_post_revision_validation_blocks_invalid_claim_from_finalizer(claim):
    package = team.EvidencePackage(
        original_question="问题",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="问题", title="资料", source="资料.md", excerpt="合法证据",
        )],
    )
    result = await team.research_team_finalizer_node({
        "evidence_package": package.model_dump(),
        "analysis_report": team.AnalysisReport(claims=[claim]).model_dump(),
        "review_report": team.ReviewReport(decision="REVISE").model_dump(),
        "research_revision_count": 1,
    })
    assert claim.text not in result["final_answer"]
    assert "无合法证据绑定的声明已移除" in result["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim",
    [
        team.Claim(claim_id="C1", text="修订新造的 S99 事实", claim_type="fact", source_ids=["S99"]),
        team.Claim(claim_id="C2", text="修订生成的无引用事实", claim_type="fact", source_ids=[]),
    ],
)
async def test_revision_node_validates_new_claims_before_writing_state(monkeypatch, claim):
    package = team.EvidencePackage(
        original_question="核验问题",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="问题", title="资料", source="资料.md", excerpt="合法证据",
        )],
    )
    monkeypatch.setattr(
        team,
        "_run_analyst",
        AsyncMock(return_value=(
            team.AnalysisReport(claims=[claim]),
            {"input_tokens": 1, "output_tokens": 1},
            10,
        )),
    )
    revised = await team.research_revision_node({
        "evidence_package": package.model_dump(),
        "analysis_report": team.AnalysisReport().model_dump(),
        "review_report": team.ReviewReport(decision="REVISE").model_dump(),
    })
    analysis = team.AnalysisReport.model_validate(revised["analysis_report"])
    assert analysis.claims == []
    assert analysis.limitations == ["1 条无合法证据绑定的声明已移除"]
    assert revised["research_revision_count"] == 1


def test_final_claim_validation_drops_everything_when_acl_is_unchecked():
    package = team.EvidencePackage(
        original_question="问题",
        acl_checked=False,
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="问题", title="资料", source="资料.md", excerpt="证据",
        )],
    )
    analysis = team.AnalysisReport(claims=[team.Claim(
        claim_id="C1", text="看似有合法编号的事实", claim_type="fact", source_ids=["S1"],
    )])
    validated = team._validate_claims_for_finalization(package, analysis)
    assert validated.claims == []
    assert "未通过 ACL 校验" in validated.limitations[0]


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
