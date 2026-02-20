"""
RAG Pipeline 模块
整合检索和生成流程
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain

from core.llm import get_llm
from core.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore
from rag.retriever import get_retriever_manager
from config.prompts import RAG_SYSTEM_PROMPT, RAG_QUESTION_PROMPT
from config.settings import get_settings


class RAGPipeline:
    """RAG 管道"""
    
    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        top_k: int = 5,
        use_compression: bool = False
    ):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.top_k = top_k or self.settings.retrieval_top_k
        self.use_compression = use_compression
        
        self.llm = get_llm()
        self.vectorstore = get_vectorstore(collection_name)
        self.retriever_manager = get_retriever_manager()
        
        # 构建检索链
        self._build_chain()
    
    def _build_chain(self):
        """构建 RAG 链"""
        # 提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_QUESTION_PROMPT)
        ])
        
        # 文档填充链
        self.document_chain = create_stuff_documents_chain(
            self.llm,
            prompt
        )
        
        # 检索链
        retriever = self.retriever_manager.retriever
        self.retrieval_chain = create_retrieval_chain(
            retriever,
            self.document_chain
        )
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """执行 RAG 流程"""
        result = self.retrieval_chain.invoke({
            "input": query
        })
        
        return {
            "answer": result["answer"],
            "context": result.get("context", []),
            "source_documents": result.get("source_documents", [])
        }
    
    def invoke_with_sources(self, query: str) -> Dict[str, Any]:
        """带来源信息的执行"""
        result = self.invoke(query)
        
        # 提取来源信息
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        result["sources"] = sources
        return result
    
    async def ainvoke(self, query: str) -> Dict[str, Any]:
        """异步执行 RAG 流程"""
        result = await self.retrieval_chain.ainvoke({
            "input": query
        })
        
        return {
            "answer": result["answer"],
            "context": result.get("context", []),
            "source_documents": result.get("source_documents", [])
        }
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """仅获取相关文档（不生成答案）"""
        retriever = self.retriever_manager.retriever
        return retriever.invoke(query)
    
    def format_context(self, query: str) -> str:
        """格式化检索到的上下文"""
        docs = self.get_relevant_documents(query)
        
        if not docs:
            return "未找到相关内容"
        
        formatted_parts = []
        
        for i, doc in enumerate(docs, 1):
            part = f"【参考 {i}】\n"
            part += f"{doc.page_content}\n"
            
            if doc.metadata:
                part += f"来源: {doc.metadata.get('source', '未知')}\n"
            
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)


class ConversationalRAGPipeline:
    """对话式 RAG 管道（支持历史记录）"""
    
    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        top_k: int = 5
    ):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.top_k = top_k
        
        self.llm = get_llm()
        self.rag_pipeline = RAGPipeline(collection_name, top_k)
    
    def invoke(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """执行带历史记录的 RAG"""
        # 如果有历史记录，将历史融入查询
        if history and len(history) > 0:
            history_text = self._format_history(history)
            # 可以选择将历史融入查询，或者在 prompt 中处理
            # 这里简单地将历史作为上下文
            enhanced_query = f"历史对话:\n{history_text}\n\n当前问题: {query}"
        else:
            enhanced_query = query
        
        result = self.rag_pipeline.invoke(enhanced_query)
        return result
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化历史记录"""
        parts = []
        for msg in history[-5:]:  # 只取最近5条
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


# 全局实例
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline(
    collection_name: str = "enterprise_knowledge",
    top_k: int = 5
) -> RAGPipeline:
    """获取 RAG 管道实例"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(collection_name, top_k)
    return _rag_pipeline


def run_rag_query(query: str) -> Dict[str, Any]:
    """运行 RAG 查询的便捷函数"""
    pipeline = get_rag_pipeline()
    return pipeline.invoke_with_sources(query)

