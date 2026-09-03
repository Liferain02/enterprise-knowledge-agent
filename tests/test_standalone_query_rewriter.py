"""Standalone 多轮指代改写的确定性契约测试。"""
from langchain_core.messages import HumanMessage

from src.rag.retrieval.query_expander import StandaloneQueryRewriter
from tests.eval.multi_turn_coreference_dataset import MULTI_TURN_COREFERENCE_CASES


def test_frozen_multiturn_dataset_trigger_and_semantic_terms():
    for case in MULTI_TURN_COREFERENCE_CASES:
        result = StandaloneQueryRewriter.rewrite(
            case.followup_query,
            recent_messages=[
                HumanMessage(content=case.previous_user_query),
                HumanMessage(content=case.followup_query),
            ],
        )

        assert result.triggered is case.should_rewrite, case.case_id
        assert result.variants[0].strategy == "original", case.case_id
        assert result.variants[0].text == case.followup_query, case.case_id
        assert len(result.variants) <= 2, case.case_id
        if case.should_rewrite:
            assert all(term in result.standalone_query for term in case.required_terms), case.case_id


def test_standalone_does_not_trigger_without_context():
    result = StandaloneQueryRewriter.rewrite("它还需要记录什么？")

    assert result.triggered is False
    assert result.reason_code == "missing_context"
    assert [item.text for item in result.variants] == ["它还需要记录什么？"]


def test_standalone_can_fall_back_to_summary():
    result = StandaloneQueryRewriter.rewrite(
        "那尾延迟呢？",
        summary="此前讨论 RDMA 高性能网络实验的结果呈现要求。",
    )

    assert result.triggered is True
    assert "RDMA" in result.standalone_query
    assert result.reason_code == "contextual_followup"
