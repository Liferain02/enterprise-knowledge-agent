"""P4 选择性路由评测契约。"""
from tests.eval.选择性路由冻结评测 import evaluate


def test_selective_routing_reaches_frozen_gate_without_all_deep_cost():
    payload = evaluate()
    selective = payload["strategies"]["selective_deep"]
    all_deep = payload["strategies"]["all_deep"]
    assert selective["routing_recall"] >= 0.95
    assert selective["routing_f1"] >= 0.90
    assert selective["under_route_count"] == 0
    assert selective["estimated_logical_calls"] < all_deep["estimated_logical_calls"]
    assert payload["gate"]["selective_gate_passed"] is True


def test_all_normal_is_not_a_valid_complex_research_strategy():
    payload = evaluate()
    normal = payload["strategies"]["all_normal"]
    assert normal["routing_recall"] == 0.0
    assert normal["under_route_count"] == payload["complex_case_count"]
