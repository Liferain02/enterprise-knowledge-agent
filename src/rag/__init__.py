"""RAG 模块公共入口。

保持入口轻量：具体组件按需加载，避免启动 API 或读取资料目录时提前导入
评估、多模态、Redis 和模型相关依赖。
"""
from importlib import import_module


_EXPORTS = {
    "get_vectorstore_manager": ("src.rag.storage.vectorstore", "get_vectorstore_manager"),
    "VectorStoreManager": ("src.rag.storage.vectorstore", "VectorStoreManager"),
    "get_retriever_manager": ("src.rag.retrieval.retriever", "get_retriever_manager"),
    "get_document_loader_manager": ("src.rag.processing.document_loader", "get_document_loader_manager"),
    "TokenRecursiveTextSplitter": ("src.rag.processing.document_loader", "TokenRecursiveTextSplitter"),
    "estimate_tokens": ("src.rag.processing.document_loader", "estimate_tokens"),
    "split_sentences": ("src.rag.processing.document_loader", "split_sentences"),
    "SemanticChunker": ("src.rag.processing.chunker", "SemanticChunker"),
    "HybridChunker": ("src.rag.processing.chunker", "HybridChunker"),
    "RAGEvaluator": ("src.rag.evaluation", "RAGEvaluator"),
    "EvalResult": ("src.rag.evaluation", "EvalResult"),
    "EvalSummary": ("src.rag.evaluation", "EvalSummary"),
    "get_evaluator": ("src.rag.evaluation", "get_evaluator"),
    "MultimodalDocumentProcessor": ("src.rag.processing.multimodal", "MultimodalDocumentProcessor"),
    "get_multimodal_processor": ("src.rag.processing.multimodal", "get_multimodal_processor"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
