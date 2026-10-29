"""Corpus and benchmark expansion contracts for the offline autumn demo."""
from pathlib import Path

from tests.eval.秋招版检索评测数据集 import RETRIEVAL_BENCHMARK_CASES

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_corpus_has_six_balanced_public_style_categories():
    root = ROOT / 'data/knowledge/synthetic_autumn_2026'
    files = sorted(root.glob('*.md'))
    assert len(files) == 240
    assert {path.name.split('_', 1)[0] for path in files} == {
        'rdma', 'numa', 'retrieval', 'systems', 'experiments', 'paper',
    } or {path.name.split('_', 1)[0] for path in files} == {
        'rdma', 'numa', 'retrieval', 'systems', 'experiments', 'paper_writing',
    }
    assert all('合成' in path.read_text(encoding='utf-8') for path in files)


def test_autumn_retrieval_benchmark_expanded_without_duplicate_ids():
    assert len(RETRIEVAL_BENCHMARK_CASES) == 88
    assert len({case.case_id for case in RETRIEVAL_BENCHMARK_CASES}) == 88
    assert sum(case.category.startswith('synthetic_') for case in RETRIEVAL_BENCHMARK_CASES) == 60
