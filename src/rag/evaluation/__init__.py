"""
RAG 评估模块
提供检索和生成质量评估能力
"""
from .evaluator import (
    RAGEvaluator,
    EvalResult,
    EvalSummary,
    get_evaluator,
    evaluate_rag,
    evaluate_rag_batch
)

__all__ = [
    "RAGEvaluator",
    "EvalResult",
    "EvalSummary",
    "get_evaluator",
    "evaluate_rag",
    "evaluate_rag_batch"
]
