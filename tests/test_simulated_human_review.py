"""人工盲评模拟契约测试。"""
import json

from tests.eval.人工盲评模拟与校准 import _score, evaluate


def test_scoring_rule_rewards_supported_false_premise():
    case = {
        "expected_keywords": ["前提", "证据"],
        "expected_key_points": [],
        "min_sources": 1,
        "premise_expectation": "false",
    }
    score = _score("前提不成立，现有证据不足以证明该结论。[文档1]", case)
    assert score["correctness"] == 2
    assert score["evidence_basis"] == 1


def test_simulated_review_is_explicitly_not_independent_human():
    payload = evaluate()
    assert payload["review_kind"] == "simulated_human"
    assert payload["independent_human"] is False
    for review in payload["reviews"]:
        assert all("hidden_variant" not in candidate for candidate in review["candidates"].values())

