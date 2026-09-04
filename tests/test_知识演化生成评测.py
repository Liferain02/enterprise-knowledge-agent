"""知识演化生成评测的冻结契约。"""

from tests.eval.知识演化生成评测 import FROZEN_CASES, evaluate


def test_frozen_knowledge_evolution_dataset_has_eight_cases():
    assert len(FROZEN_CASES) == 8
    assert {case["case_id"] for case in FROZEN_CASES} == {
        "new_fact", "raw_already_covers", "supersede_only_latest", "revoked_never_hits",
        "acl_blocked", "derived_only_blocked", "provenance_required", "project_fact_reuse",
    }


def test_knowledge_evolution_safety_gate_is_zero():
    payload = evaluate()
    assisted = payload["variants"]["raw_plus_active_project_knowledge"]["aggregate"]
    assert assisted["revoked_or_superseded_wrong_hit"] == 0
    assert assisted["acl_violation"] == 0
    assert assisted["provenance_completeness"] == 1.0


def test_active_knowledge_recovers_new_fact_without_changing_raw_variant():
    payload = evaluate()
    raw = payload["variants"]["raw_documents_only"]["aggregate"]
    assisted = payload["variants"]["raw_plus_active_project_knowledge"]["aggregate"]
    assert raw["gold_fact_hit"] < assisted["gold_fact_hit"]
    assert assisted["answer_correctness"] == 1.0
