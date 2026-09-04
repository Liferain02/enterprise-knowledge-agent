"""秋招版 Retrieval Benchmark 的冻结契约测试。"""

from collections import Counter

from tests.eval.秋招版检索评测数据集 import RETRIEVAL_BENCHMARK_CASES
from tests.eval.run_秋招版检索评测 import _metrics


def test_benchmark_has_frozen_size_and_category_distribution():
    assert len(RETRIEVAL_BENCHMARK_CASES) == 28
    assert Counter(case.category for case in RETRIEVAL_BENCHMARK_CASES) == {
        "simple_factual": 8,
        "technical_exact_token": 6,
        "paraphrase": 5,
        "distractor_heavy": 5,
        "multi_document": 4,
    }
    assert len({case.case_id for case in RETRIEVAL_BENCHMARK_CASES}) == 28


def test_benchmark_gold_is_document_level_and_paraphrase_cases_are_explicit():
    assert all(case.gold_level == "document" for case in RETRIEVAL_BENCHMARK_CASES)
    assert all(case.gold_doc_ids for case in RETRIEVAL_BENCHMARK_CASES)
    assert all(case.acceptable_sources == case.gold_doc_ids for case in RETRIEVAL_BENCHMARK_CASES)
    assert all(
        case.distractor_ids
        for case in RETRIEVAL_BENCHMARK_CASES
        if case.category == "distractor_heavy"
    )
    assert sum(bool(case.paraphrase_queries) for case in RETRIEVAL_BENCHMARK_CASES) == 5


def test_metrics_report_hit_mrr_coverage_and_distractor_top1():
    case = RETRIEVAL_BENCHMARK_CASES[0]
    metrics = _metrics(
        ["无关资料", "实验室组会制度与汇报要求"],
        case,
    )

    assert metrics["hit_at_1"] == 0
    assert metrics["hit_at_5"] == 1
    assert metrics["mrr_at_5"] == 0.5
    assert metrics["coverage_at_5"] == 1.0
    assert metrics["distractor_top1"] == 0
