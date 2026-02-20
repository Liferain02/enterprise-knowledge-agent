"""
检索器模块
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rag.vectorstore import get_vectorstore, VectorStoreManager
from config.settings import get_settings


class RetrieverManager:
    """检索器管理器"""
    
    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.top_k = top_k or self.settings.retrieval_top_k
        self.similarity_threshold = similarity_threshold or self.settings.similarity_threshold
        self._retriever = None
    
    @property
    def retriever(self):
        """获取检索器实例"""
        if self._retriever is None:
            vectorstore = get_vectorstore(self.collection_name)
            self._retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.top_k,
                    "score_threshold": self.similarity_threshold
                }
            )
        return self._retriever
    
    def get_compression_retriever(self, llm=None):
        """获取压缩检索器（使用LLM提取相关内容）"""
        # 注意：LangChain 1.0+ 中 LLMChainExtractor 已移除
        # 这里返回基础检索器
        return self.retriever
    
    def get_self_query_retriever(
        self,
        llm=None,
        metadata_field_info: Optional[List] = None
    ):
        """获取自查询检索器"""
        # 自查询检索器在 LangChain 1.0+ 中需要单独安装 langchain-experimental
        # 这里返回基础检索器作为替代
        return self.retriever
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """搜索文档"""
        k = k or self.top_k
        vectorstore = get_vectorstore(self.collection_name)
        
        return vectorstore.similarity_search(
            query,
            k=k,
            filter=filter
        )
    
    def search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[tuple[Document, float]]:
        """带分数的搜索"""
        k = k or self.top_k
        vectorstore = get_vectorstore(self.collection_name)
        
        return vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )
    
    def format_search_results(
        self,
        results: List[Document],
        include_metadata: bool = True
    ) -> str:
        """格式化搜索结果为文本"""
        if not results:
            return "未找到相关内容"
        
        formatted_parts = []
        
        for i, doc in enumerate(results, 1):
            part = f"【文档 {i}】\n"
            part += f"内容: {doc.page_content}\n"
            
            if include_metadata and doc.metadata:
                metadata_str = ", ".join(
                    f"{k}: {v}" for k, v in doc.metadata.items()
                )
                part += f"元数据: {metadata_str}\n"
            
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)


# 全局实例
_retriever_manager: Optional[RetrieverManager] = None


def get_retriever_manager() -> RetrieverManager:
    """获取检索器管理器实例"""
    global _retriever_manager
    if _retriever_manager is None:
        _retriever_manager = RetrieverManager()
    return _retriever_manager


def retrieve_documents(
    query: str,
    k: Optional[int] = None
) -> List[Document]:
    """检索文档的便捷函数"""
    manager = get_retriever_manager()
    return manager.search(query, k=k)


def format_retrieved_context(
    query: str,
    k: Optional[int] = None
) -> str:
    """检索并格式化上下文的便捷函数"""
    manager = get_retriever_manager()
    results = manager.search(query, k=k)
    return manager.format_search_results(results)

