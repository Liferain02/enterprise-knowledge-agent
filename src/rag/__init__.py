"""
RAG 模块 - 检索增强生成
"""
from .storage.vectorstore import get_vectorstore_manager, VectorStoreManager
from .retrieval.retriever import get_retriever_manager
from .processing.document_loader import (
    get_document_loader_manager,
    TokenRecursiveTextSplitter,
    estimate_tokens,
    split_sentences,
)
from .processing.chunker import SemanticChunker, HybridChunker

# 评估模块
try:
    from .evaluation import RAGEvaluator, EvalResult, EvalSummary, get_evaluator
    _EVAL_AVAILABLE = True
except ImportError:
    _EVAL_AVAILABLE = False

# 多模态处理模块
try:
    from .processing.multimodal import MultimodalDocumentProcessor, get_multimodal_processor
    _MULTIMODAL_AVAILABLE = True
except ImportError:
    _MULTIMODAL_AVAILABLE = False

__all__ = [
    "get_vectorstore_manager",
    "VectorStoreManager",
    "get_retriever_manager",
    "get_document_loader_manager",
    "TokenRecursiveTextSplitter",
    "estimate_tokens",
    "split_sentences",
    "SemanticChunker",
    "HybridChunker",
]

# 条件性导出评估模块
if _EVAL_AVAILABLE:
    __all__.extend(["RAGEvaluator", "EvalResult", "EvalSummary", "get_evaluator"])

# 条件性导出多模态模块
if _MULTIMODAL_AVAILABLE:
    __all__.extend(["MultimodalDocumentProcessor", "get_multimodal_processor"])
