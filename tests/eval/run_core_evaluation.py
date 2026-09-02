#!/usr/bin/env python
"""核心评测统一入口；不会覆盖冻结 Blind V2 产物。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(module: str, extra: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, *extra], cwd=PROJECT_ROOT, check=False
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _inventory() -> None:
    from tests.eval.complex_research_dataset import COMPLEX_RESEARCH_DATASET
    from tests.eval.deep_research_retrieval_dev_dataset import RETRIEVAL_DEV_DATASET
    from tests.eval.deep_research_v2_blind_dataset import BLIND_RESEARCH_V2_DATASET
    from tests.eval.eval_dataset import EVAL_DATASET

    blind_result = PROJECT_ROOT / "data" / "深度研究V2盲测结果.json"
    model_review = PROJECT_ROOT / "data" / "深度研究V2模型模拟盲评汇总.json"
    review_status = "missing"
    model_review_status = "missing"
    if blind_result.exists():
        review_status = json.loads(blind_result.read_text(encoding="utf-8")).get(
            "human_review_status", "unknown"
        )
    if model_review.exists():
        model_review_status = json.loads(model_review.read_text(encoding="utf-8")).get(
            "status", "unknown"
        )
    print(json.dumps({
        "basic_retrieval": len(EVAL_DATASET),
        "retrieval_development": len(RETRIEVAL_DEV_DATASET),
        "complex_research_development": len(COMPLEX_RESEARCH_DATASET),
        "blind_v2_frozen": len(BLIND_RESEARCH_V2_DATASET),
        "blind_v2_human_review_status": review_status,
        "blind_v2_model_review_status": model_review_status,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="实验室科研智能助手核心评测入口")
    parser.add_argument(
        "stage",
        choices=(
            "inventory", "ablation", "rag", "deep-retrieval", "deep-answer-dev",
            "premise-regression", "blind-template", "blind-review",
            "blind-model-review",
        ),
    )
    args, extra = parser.parse_known_args()
    if args.stage == "inventory":
        _inventory()
    elif args.stage == "ablation":
        _run("tests.eval.run_core_retrieval_ablation", extra)
    elif args.stage == "rag":
        _run("tests.eval.run_rag_eval", extra)
    elif args.stage == "deep-retrieval":
        _run("tests.eval.run_deep_research_retrieval_eval", extra)
    elif args.stage == "deep-answer-dev":
        _run("tests.eval.run_research_team_eval", extra)
    elif args.stage == "premise-regression":
        _run("tests.eval.run_premise_regression", extra)
    elif args.stage == "blind-template":
        _run("tests.eval.summarize_blind_review", ["--create-template", *extra])
    elif args.stage == "blind-review":
        _run("tests.eval.summarize_blind_review", extra)
    elif args.stage == "blind-model-review":
        _run("tests.eval.summarize_blind_review", ["--review-kind", "model", *extra])


if __name__ == "__main__":
    main()
