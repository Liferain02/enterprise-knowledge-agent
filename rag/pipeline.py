"""
RAG Pipeline 模块
整合检索和生成流程
使用 LangChain 1.x LCEL 方式
支持 Reranker 重排序
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.llm import get_llm
from core.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore
from rag.retriever import get_retriever_manager
from rag.reranker import get_reranker_manager
from config.settings import get_settings


# ==================== RAG 提示词 ====================

RAG_SYSTEM_PROMPT = """你是一个企业知识库助手。你的任务是根据给定的上下文信息回答用户的问题。

要求：
1. 只根据提供的上下文信息回答，不要编造信息
2. 如果上下文中没有相关信息，请明确告知用户
3. 在答案中标注信息来源
4. 保持回答简洁准确
"""

RAG_QUESTION_PROMPT = """请根据以下上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{input}

请给出答案："""

# 历史感知检索器系统提示词
HISTORY_AWARE_RETRIEVER_PROMPT = """给定聊天历史和最新用户问题，如果历史记录与问题相关，请将历史记录与当前问题结合形成一个独立的检索查询。

规则：
- 如果问题本身已经完整，不需要修改，直接返回原问题
- 如果问题有指代（如"它"、"这个"、"之前说的"），需要结合历史上下文
- 只返回优化后的查询语句，不需要其他内容

聊天历史：
{chat_history}

当前问题：{input}

优化后的查询："""


def format_docs(docs: List[Document]) -> str:
    """格式化文档为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGPipeline:
    """RAG 管道"""

    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        top_k: int = 5,
        use_compression: bool = False,
        use_reranker: bool = True
    ):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.top_k = top_k or self.settings.retrieval_top_k
        self.use_compression = use_compression
        self.use_reranker = use_reranker and self.settings.reranker_enabled

        self.llm = get_llm()
        self.vectorstore = get_vectorstore(collection_name)
        self.retriever_manager = get_retriever_manager()

        # 初始化 Reranker
        self.reranker_manager = None
        if self.use_reranker:
            try:
                self.reranker_manager = get_reranker_manager()
            except Exception as e:
                print(f"Warning: Reranker 初始化失败: {e}")
                self.use_reranker = False

        # 构建检索链
        self._build_chain()

    def _build_chain(self):
        """使用 LCEL 构建 RAG 链"""
        # 提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_QUESTION_PROMPT)
        ])

        # 基础检索器
        retriever = self.retriever_manager.retriever

        # 定义检索函数（支持 Reranker）
        def retrieve_and_rerank(input_dict: Dict) -> List[Document]:
            query = input_dict.get("input", "")
            docs = retriever.invoke(query)
            
            if self.use_reranker and self.reranker_manager and docs:
                try:
                    results = self.reranker_manager.rerank(query, docs, top_n=self.top_k)
                    return [doc for doc, score in results]
                except Exception as e:
                    print(f"Rerank 错误: {e}")
                    return docs[:self.top_k]
            return docs[:self.top_k]

        # 使用 LCEL 构建链
        self.retrieval_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(retrieve_and_rerank)
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, query: str) -> Dict[str, Any]:
        """执行 RAG 流程"""
        # 先获取文档（用于返回 source_documents）
        retriever = self.retriever_manager.retriever
        docs = retriever.invoke(query)
        
        # Reranker
        if self.use_reranker and self.reranker_manager and docs:
            try:
                results = self.reranker_manager.rerank(query, docs, top_n=self.top_k)
                docs = [doc for doc, score in results]
            except Exception as e:
                print(f"Rerank 错误: {e}")
                docs = docs[:self.top_k]
        else:
            docs = docs[:self.top_k]

        # 执行生成
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_QUESTION_PROMPT)
        ])
        
        context_str = format_docs(docs)
        
        answer = self.llm.invoke(
            prompt.format(context=context_str, input=query)
        )
        
        return {
            "answer": answer.content if hasattr(answer, 'content') else str(answer),
            "context": docs,
            "source_documents": docs
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
        return self.invoke(query)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """仅获取相关文档（不生成答案）"""
        retriever = self.retriever_manager.retriever
        docs = retriever.invoke(query)
        
        # Reranker
        if self.use_reranker and self.reranker_manager and docs:
            try:
                results = self.reranker_manager.rerank(query, docs, top_n=self.top_k)
                return [doc for doc, score in results]
            except Exception as e:
                print(f"Rerank 错误: {e}")
                return docs[:self.top_k]
        
        return docs[:self.top_k]
    
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
    """对话式 RAG 管道（支持历史记录）- 使用 LCEL"""
    
    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        top_k: int = 5
    ):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.top_k = top_k
        
        self.llm = get_llm()
        self.vectorstore = get_vectorstore(collection_name)
        self.retriever_manager = get_retriever_manager()
        
        # 构建带历史感知的检索链
        self._build_history_aware_chain()
    
    def _build_history_aware_chain(self):
        """构建支持历史记录的 RAG 链"""
        retriever = self.retriever_manager.retriever
        
        # 历史感知检索器 prompt
        history_prompt = ChatPromptTemplate.from_messages([
            ("system", HISTORY_AWARE_RETRIEVER_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}")
        ])
        
        # 使用 LLM 将历史问题改写为独立查询
        def rewrite_query(input_dict: Dict) -> str:
            query = input_dict.get("input", "")
            chat_history = input_dict.get("chat_history", [])
            
            if not chat_history:
                return query
            
            # 使用 LLM 改写
            try:
                result = history_prompt.format(
                    chat_history=chat_history,
                    input=query
                )
                response = self.llm.invoke(result)
                rewritten = response.content if hasattr(response, 'content') else str(response)
                return rewritten.strip()
            except Exception as e:
                print(f"Query rewrite error: {e}")
                return query
        
        # 问答 prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_QUESTION_PROMPT)
        ])
        
        # 定义检索和生成流程
        def retrieve_with_history(input_dict: Dict) -> List[Document]:
            query = rewrite_query(input_dict)
            return retriever.invoke(query)[:self.top_k]
        
        self.retrieval_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(retrieve_with_history)
            )
            | RunnablePassthrough.assign(
                chat_history=lambda x: x.get("chat_history", [])
            )
            | RunnableLambda(lambda x: {
                "input": x.get("input", ""),
                "context": format_docs(x.get("context", [])),
                "chat_history": x.get("chat_history", [])
            })
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        执行带历史记录的 RAG
        
        Args:
            query: 当前用户问题
            history: 历史消息列表 [{"role": "user"/"assistant", "content": "..."}]
        
        Returns:
            包含 answer, context, source_documents 的字典
        """
        # 将历史记录转换为 LangChain 消息格式
        chat_history = self._convert_history_to_messages(history)
        
        # 先检索文档
        retriever = self.retriever_manager.retriever
        
        # 如果有历史，先改写查询
        if chat_history:
            history_prompt = ChatPromptTemplate.from_messages([
                ("system", HISTORY_AWARE_RETRIEVER_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}")
            ])
            try:
                prompt_str = history_prompt.format(
                    chat_history=chat_history,
                    input=query
                )
                response = self.llm.invoke(prompt_str)
                rewritten_query = response.content if hasattr(response, 'content') else str(response)
                rewritten_query = rewritten_query.strip()
            except Exception as e:
                print(f"Query rewrite error: {e}")
                rewritten_query = query
        else:
            rewritten_query = query
        
        # 检索文档
        docs = retriever.invoke(rewritten_query)[:self.top_k]
        
        # 生成答案
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_QUESTION_PROMPT)
        ])
        
        context_str = format_docs(docs)
        answer = self.llm.invoke(
            qa_prompt.format(context=context_str, input=query)
        )
        
        return {
            "answer": answer.content if hasattr(answer, 'content') else str(answer),
            "context": docs,
            "source_documents": docs
        }
    
    def _convert_history_to_messages(self, history: Optional[List[Dict]]) -> List:
        """将历史记录转换为 LangChain 消息格式"""
        if not history:
            return []
        
        messages = []
        for msg in history[-10:]:  # 最多保留10条历史
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
        
        return messages
    
    async def ainvoke(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """异步执行带历史记录的 RAG"""
        return self.invoke(query, history)


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
