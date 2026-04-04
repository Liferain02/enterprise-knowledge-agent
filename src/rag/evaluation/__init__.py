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
    evaluate_rag_batch,
)
from .retrieval_grader import (
    RetrievalGrader,
    CorrectiveRAGPipeline,
    GradeLevel,
    DocumentGrade,
    GradeResult,
    get_retrieval_grader,
    get_corrective_rag_pipeline,
    grade_retrieval,
    corrective_retrieve,
    reset_crags,
)
from . import grade_cache
from . import conflict_detector

__all__ = [
    # RAG 事后评估（RAGEvaluator）
    "RAGEvaluator",
    "EvalResult",
    "EvalSummary",
    "get_evaluator",
    "evaluate_rag",
    "evaluate_rag_batch",
    # Corrective RAG 检索评估
    "RetrievalGrader",
    "CorrectiveRAGPipeline",
    "GradeLevel",
    "DocumentGrade",
    "GradeResult",
    "get_retrieval_grader",
    "get_corrective_rag_pipeline",
    "grade_retrieval",
    "corrective_retrieve",
    "reset_crags",
    # 评估缓存（Redis 持久化）
    "grade_cache",
]
