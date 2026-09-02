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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("验证当前原型是否已经完成这一前提", True),
        ("是否可以认为所有成员都必须登记", True),
        ("是否可以确认每次实验都必须记录环境", True),
        ("能否确认当前方案已经通过全部验证", True),
        ("判断所有成员是否必须登记共享资源", True),
        ("比较 A 和 B 的机制差异", False),
        ("总结三份资料并形成建议", False),
        ("判断两份资料是冲突还是互补", False),
    ],
)
def test_premise_task_boundary_is_about_user_intent_not_topic_keywords(query, expected):
    assert team._is_premise_task(query) is expected


@pytest.mark.asyncio
async def test_planner_remains_single_entry_and_routes_only_real_research_tasks(monkeypatch):
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


def test_production_graph_has_explicit_bounded_team_without_supervisor_or_review_loop():
    graph = create_multi_agent_graph().compile().get_graph()
    mermaid = graph.draw_mermaid()

    for node in (
        "research_agent", "analyst_agent", "reviewer_agent",
        "research_revision", "deep_research_generation",
    ):
        assert node in mermaid
    assert "research_team_finalizer" not in mermaid
    assert "supervisor" not in mermaid
    assert "research_revision --> reviewer_agent" not in mermaid


def test_deep_mode_is_explicit_and_normal_route_is_unchanged():
    base = {"is_complex": False, "_quick_agent": "knowledge_agent"}
    assert planner_module.route_from_planner(base) == "retrieval_agent"
    assert planner_module.route_from_planner({**base, "research_mode": "normal"}) == "retrieval_agent"
    assert planner_module.route_from_planner({**base, "research_mode": "deep"}) == "research_agent"


def test_review_report_only_accepts_three_decisions():
    with pytest.raises(ValidationError):
        team.ReviewReport(decision="DISCUSS")


def test_review_report_accepts_qwen_wrapped_premise_enum():
    report = team.ReviewReport.model_validate({
        "decision": "PASS",
        "premise_assessment": {"status": "supported", "reason": "有直接证据"},
    })
    assert report.premise_assessment == "supported"


def test_review_report_accepts_qwen_reviewer_decision_alias():
    report = team.ReviewReport.model_validate({
        "reviewer_decision": "REVISE",
        "premise_assessment": "not_applicable",
    })
    assert report.decision == "REVISE"


def test_analysis_normalization_moves_transient_limitation_out_of_claims():
    package = team.EvidencePackage(original_question="问题")
    report = team.AnalysisReport(claims=[team.Claim(
        claim_id="", text="缺少定量实验数据", claim_type="limitation", source_ids=[],
    )])
    normalized = team._normalize_analysis(report, package)
    assert normalized.claims == []
    assert normalized.limitations == ["缺少定量实验数据"]


def test_claim_accepts_qwen_wire_aliases_and_normalizes_missing_type():
    claim = team.Claim.model_validate({
        "claim_text": "建议下一步补做消融实验",
        "source_ids": ["S1"],
    })
    package = team.EvidencePackage(
        original_question="问题",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="问题", title="资料", source="资料.md", excerpt="证据",
        )],
    )
    report = team._normalize_analysis(team.AnalysisReport(claims=[claim]), package)
    assert report.claims[0].text == "建议下一步补做消融实验"
    assert report.claims[0].claim_type == "recommendation"


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
    monkeypatch.setattr(
        "src.agent.agents.knowledge._retrieve_documents",
        AsyncMock(return_value=(
            [(public_doc, 0.9), (restricted_doc, 0.95)], grade, [],
        )),
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
async def test_reviewer_skips_unactionable_revision_for_ordinary_task(monkeypatch):
    package = team.EvidencePackage(
        original_question="比较两种实验方案并总结差异",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="比较", title="实验记录",
            source="实验记录.md", excerpt="方案 A 和方案 B 使用不同的采样频率",
        )],
    )
    analysis = team.AnalysisReport(claims=[team.Claim(
        claim_id="C1", text="两种方案使用不同采样频率",
        claim_type="comparison", source_ids=["S1"],
    )])
    monkeypatch.setattr(
        team,
        "_invoke_structured",
        AsyncMock(return_value=(team.ReviewReport(
            decision="REVISE",
            overall_instruction="进一步润色并让表达更完整。",
        ), {"input_tokens": 10, "output_tokens": 2})),
    )

    result = await team.reviewer_agent_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
    })
    report = team.ReviewReport.model_validate(result["review_report"])
    reviewer_trace = result["research_trace"]["stages"]["reviewer"]

    assert report.decision == "PASS"
    assert report.overall_instruction == ""
    assert reviewer_trace["decision_before_actionability_gate"] == "REVISE"
    assert reviewer_trace["review_report_before_actionability_gate"]["decision"] == "REVISE"
    assert (
        reviewer_trace["review_report_before_actionability_gate"]["overall_instruction"]
        == "进一步润色并让表达更完整。"
    )
    assert reviewer_trace["revision_skipped_reason"]


@pytest.mark.asyncio
async def test_reviewer_keeps_revision_for_structured_issue(monkeypatch):
    package = team.EvidencePackage(
        original_question="总结实验结论",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="总结", title="实验记录",
            source="实验记录.md", excerpt="实验尚未完成，当前没有最终结论",
        )],
    )
    analysis = team.AnalysisReport(claims=[team.Claim(
        claim_id="C1", text="实验已经证明方案有效",
        claim_type="fact", source_ids=["S1"],
    )])
    issue = team.ReviewItem(
        claim="实验已经证明方案有效",
        source_ids=["S1"],
        supported=False,
        issue_type="unsupported",
        revision_instruction="删除未被证据支持的结论。",
    )
    monkeypatch.setattr(
        team,
        "_invoke_structured",
        AsyncMock(return_value=(team.ReviewReport(
            decision="REVISE", items=[issue],
        ), {"input_tokens": 10, "output_tokens": 2})),
    )

    result = await team.reviewer_agent_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
    })
    report = team.ReviewReport.model_validate(result["review_report"])

    assert report.decision == "REVISE"
    assert report.items == [issue]
    assert not result["research_trace"]["stages"]["reviewer"]["revision_skipped_reason"]


def test_review_report_accepts_nested_qwen_decision_object():
    report = team.ReviewReport.model_validate({
        "review_result": {"decision": "PASS", "revision_instruction": "无需修订"},
    })

    assert report.decision == "PASS"


@pytest.mark.asyncio
async def test_reviewer_failure_never_passes_actionability_gate(monkeypatch):
    package = team.EvidencePackage(
        original_question="总结实验记录",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="总结", title="记录", source="记录.md", excerpt="证据",
        )],
    )
    analysis = team.AnalysisReport(claims=[team.Claim(
        claim_id="C1", text="有证据的事实", source_ids=["S1"],
    )])
    monkeypatch.setattr(team, "_invoke_structured", AsyncMock(side_effect=ValueError("解析失败")))

    result = await team.reviewer_agent_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
    })
    report = team.ReviewReport.model_validate(result["review_report"])
    reviewer_trace = result["research_trace"]["stages"]["reviewer"]

    assert report.decision == "REVISE"
    assert reviewer_trace["reviewer_call_failed"] is True
    assert reviewer_trace["revision_skipped_reason"] == ""


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
            decision="PASS", premise_assessment="supported", premise_source_ids=["S1"],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("assessment", ["supported", "unsupported"])
async def test_reviewer_uses_sourced_analyst_premise_when_reviewer_is_inconclusive(
    monkeypatch, assessment,
):
    package = team.EvidencePackage(
        original_question="是否可以确认每次实验都必须记录环境",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="核验", title="实验规范", source="实验规范.md",
            excerpt="每次实验至少记录硬件、软件版本和随机种子",
        )],
    )
    analysis = team.AnalysisReport(
        claims=[team.Claim(
            claim_id="C1", text="规范要求每次实验记录环境", source_ids=["S1"],
        )],
        premise_assessment=assessment,
        premise_source_ids=["S1"],
    )
    monkeypatch.setattr(
        team,
        "_invoke_structured",
        AsyncMock(return_value=(team.ReviewReport(
            decision="PASS", premise_assessment="insufficient",
        ), {"input_tokens": 1, "output_tokens": 1})),
    )

    result = await team.reviewer_agent_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
    })
    report = team.ReviewReport.model_validate(result["review_report"])
    assert report.premise_assessment == assessment
    assert report.premise_source_ids == ["S1"]
    assert report.false_premise_detected is (assessment == "unsupported")


def test_premise_reconciliation_does_not_override_conflicting_evidence():
    package = team.EvidencePackage(
        original_question="验证所有任务都必须登记这一前提",
        conflicts=["两份制度对登记范围描述不一致"],
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="核验", title="制度", source="制度.md",
            excerpt="所有任务必须登记",
        )],
    )
    analysis = team.AnalysisReport(
        premise_assessment="supported", premise_source_ids=["S1"],
    )
    review = team.ReviewReport(decision="PASS", premise_assessment="insufficient")

    reconciled = team._reconcile_premise_assessment(package, analysis, review)
    assert reconciled.premise_assessment == "insufficient"
    assert reconciled.premise_source_ids == []


@pytest.mark.asyncio
async def test_reviewer_forces_not_applicable_for_ordinary_comparison(monkeypatch):
    package = team.EvidencePackage(
        original_question="比较 A 和 B，并给出下一步研究建议",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="比较", title="资料", source="资料.md", excerpt="A 与 B 不同",
        )],
    )
    analysis = team.AnalysisReport(claims=[team.Claim(
        claim_id="C1", text="A 与 B 采用不同机制", claim_type="comparison", source_ids=["S1"],
    )])
    monkeypatch.setattr(
        team,
        "_invoke_structured",
        AsyncMock(return_value=(team.ReviewReport(
            decision="PASS",
            premise_assessment="unsupported",
            premise_source_ids=["S1"],
            false_premise_detected=True,
            items=[team.ReviewItem(
                claim=package.original_question,
                supported=False,
                issue_type="false_premise",
            )],
        ), {"input_tokens": 1, "output_tokens": 1})),
    )

    reviewed = await team.reviewer_agent_node({
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
    })
    report = team.ReviewReport.model_validate(reviewed["review_report"])
    assert report.premise_assessment == "not_applicable"
    assert report.premise_source_ids == []
    assert report.false_premise_detected is False
    assert all(item.issue_type != "false_premise" for item in report.items)
    assert reviewed["research_trace"]["stages"]["reviewer"]["premise_task"] is False

    context = team._build_deep_research_context(package, analysis, report)
    assert "前提核验" not in context


def test_reviewer_can_trigger_at_most_one_revision():
    report = team.ReviewReport(decision="NEED_MORE_EVIDENCE")
    assert team.route_after_reviewer({
        "review_report": report.model_dump(),
        "research_revision_count": 0,
    }) == "research_revision"
    assert team.route_after_reviewer({
        "review_report": report.model_dump(),
        "research_revision_count": 1,
    }) == "deep_research_generation"


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


@pytest.mark.asyncio
async def test_deep_generation_reuses_existing_generation_agent_with_validated_claims(monkeypatch):
    package = team.EvidencePackage(
        original_question="综合资料形成研究简报",
        evidences=[team.EvidenceItem(
            source_id="S1", subquestion="问题", title="资料一", source="资料一.md",
            excerpt="资料明确记录了实验结果",
        )],
    )
    analysis = team.AnalysisReport(claims=[
        team.Claim(claim_id="C1", text="资料明确记录了实验结果", claim_type="fact", source_ids=["S1"]),
        team.Claim(claim_id="C2", text="应被 Reviewer 删除", claim_type="fact", source_ids=["S1"]),
    ])
    review = team.ReviewReport(
        decision="PASS",
        items=[team.ReviewItem(
            claim="应被 Reviewer 删除", source_ids=["S1"], supported=False,
            issue_type="unsupported",
        )],
    )
    generation = AsyncMock(return_value={
        "final_answer": "1. 研究问题\n问题\n3. 关键事实\n实验结果[文档1]。\n7. Sources\n资料一",
        "generation_metrics": {"llm_calls": 1, "input_tokens": 10, "output_tokens": 5, "elapsed_ms": 3},
    })
    monkeypatch.setattr("src.agent.agents.knowledge.generation_agent_node", generation)

    result = await team.deep_research_generation_node({
        "messages": [HumanMessage(content=package.original_question)],
        "evidence_package": package.model_dump(),
        "analysis_report": analysis.model_dump(),
        "review_report": review.model_dump(),
    })

    generation_state = generation.await_args.args[0]
    assert "资料明确记录了实验结果" in generation_state["retrieval_context"]
    assert "应被 Reviewer 删除" not in generation_state["retrieval_context"]
    assert "严格按以下七个标题输出" in generation_state["answer_format_instructions"]
    assert result["used_agent"] == "deep_research"
    assert result["research_team_metrics"]["llm_calls"] == 1
    assert result["research_trace"]["stages"]["generation"]["validated_claims"]
    generation_trace = result["research_trace"]["stages"]["generation"]
    assert generation_trace["reviewer_dropped_claims"][0]["issue_type"] == "unsupported"
    assert "omitted_validated_claim_ids" in generation_trace
    assert "final_validated_claim_coverage_proxy" in generation_trace
    assert result["research_trace"]["failure_attribution"] in {
        "retrieval", "analysis", "review", "generation", "knowledge_gap", "acl", "none",
    }
    assert "generation" in result["research_trace"]["stage_latency_ms"]
