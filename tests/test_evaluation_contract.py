"""统一评测入口与冻结盲评汇总协议测试。"""
import pytest

from tests.eval.deep_research_v2_blind_dataset import BLIND_RESEARCH_V2_DATASET
from tests.eval.summarize_blind_review import load_review, summarize


def _complete_scores():
    return {
        case.case_id: {
            "candidates": {
                "候选甲": {"correctness": 2, "completeness": 2, "evidence": 2},
                "候选乙": {"correctness": 1, "completeness": 1, "evidence": 1},
            },
            "winner": "候选甲",
            "reason": "候选甲覆盖关键点且引用更完整。",
        }
        for case in BLIND_RESEARCH_V2_DATASET
    }


def test_blind_review_only_unseals_after_all_cases_are_scored():
    scores = _complete_scores()
    mapping = {
        case.case_id: {"候选甲": "B", "候选乙": "C"}
        for case in BLIND_RESEARCH_V2_DATASET
    }

    payload = summarize(scores, mapping)

    assert payload["status"] == "independent_human_review_complete"
    assert payload["cases"] == len(BLIND_RESEARCH_V2_DATASET)
    assert payload["wins"] == {"B": len(BLIND_RESEARCH_V2_DATASET)}
    assert payload["average_total"] == {"B": 6.0, "C": 3.0}


def test_model_review_is_never_labeled_as_independent_human_review():
    scores = _complete_scores()
    mapping = {
        case.case_id: {"候选甲": "B", "候选乙": "C"}
        for case in BLIND_RESEARCH_V2_DATASET
    }

    payload = summarize(scores, mapping, review_kind="model")

    assert payload["status"] == "model_simulated_blind_review_complete"
    assert payload["review_kind"] == "model"


def test_blind_review_rejects_partial_or_blank_scores():
    scores = _complete_scores()
    scores.pop(next(iter(scores)))
    with pytest.raises(ValueError, match="不完整"):
        summarize(scores, {})

    scores = _complete_scores()
    first = next(iter(scores))
    scores[first]["candidates"]["候选甲"]["correctness"] = None
    mapping = {
        case.case_id: {"候选甲": "B", "候选乙": "C"}
        for case in BLIND_RESEARCH_V2_DATASET
    }
    with pytest.raises(ValueError, match="0～2"):
        summarize(scores, mapping)


def test_blank_scores_fail_before_sealed_mapping_is_read(tmp_path):
    import json

    scores = _complete_scores()
    first = next(iter(scores))
    scores[first]["candidates"]["候选甲"]["correctness"] = None
    scores_path = tmp_path / "空评分.json"
    scores_path.write_text(json.dumps(scores, ensure_ascii=False), encoding="utf-8")
    missing_mapping = tmp_path / "不应读取的密封映射.json"

    with pytest.raises(ValueError, match="0～2"):
        load_review(scores_path, missing_mapping)
