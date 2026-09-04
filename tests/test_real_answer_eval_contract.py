"""P4 真实回答评测脚本契约。"""
import asyncio

from tests.eval.脱敏真实回答评测 import _add_gate, _normal_cases, selected_cases


def test_representative_dataset_contains_complex_and_normal_cases():
    cases = selected_cases(6)
    assert len(cases) == 6
    assert {case.category for case in cases} == {"cross_scope", "temporal_conflict", "false_premise", "normal"}
    assert len(_normal_cases()) == 3


def test_real_answer_eval_does_not_use_all_complex_dataset():
    cases = selected_cases(6)
    assert len(cases) < 41
    assert {case.case_id for case in cases} == {"C01", "C17", "C33", "N01", "N02", "N03"}


def test_gate_rejects_complex_citation_regression():
    payload = {
        "results": [
            {"case": {"category": "cross_scope"}, "variants": {
                "A": {"keyword_correctness": 0.8, "citation_support_rate": 1.0},
                "C": {"keyword_correctness": 1.0, "citation_support_rate": 0.7},
            }},
        ],
    }
    result = _add_gate(payload)
    assert result["gate"]["passed"] is False
    assert "停止自动路由" in result["gate"]["decision"]
