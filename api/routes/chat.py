"""
聊天 API 路由
"""
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import json
from agent.state import AgentRequest, AgentResponse
from agent.graph import run_agent
from agent.react import run_react_agent
from core.chat_history import get_chat_history_manager
from rag.retriever import get_retriever_manager
from config.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default", description="会话ID")
    use_rag: bool = Field(default=True, description="是否使用RAG")
    stream: bool = Field(default=False, description="是否流式输出")
    use_react: bool = Field(default=False, description="是否使用ReAct模式")


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(description="AI回复")
    sources: list = Field(default_factory=list, description="信息来源")
    session_id: str = Field(description="会话ID")
    context_used: bool = Field(description="是否使用了RAG")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口
    
    支持普通模式和ReAct模式
    """
    settings = get_settings()
    history_manager = get_chat_history_manager()
    
    try:
        # 获取对话历史
        history = history_manager.get_history_text(request.session_id)
        
        # 如果使用RAG，检索上下文
        context = ""
        sources = []
        
        if request.use_rag:
            try:
                retriever_manager = get_retriever_manager()
                docs = retriever_manager.search(request.message, k=settings.retrieval_top_k)
                
                if docs:
                    context = retriever_manager.format_search_results(docs)
                    sources = [
                        {"content": doc.page_content[:200], "metadata": doc.metadata}
                        for doc in docs
                    ]
            except Exception as e:
                print(f"RAG检索错误: {e}")
        
        # 根据模式选择执行方式
        if request.use_react:
            # 使用ReAct模式
            result = run_react_agent(
                question=request.message,
                context=context,
                history=history
            )
            answer = result.get("answer", "抱歉，无法生成答案。")
        else:
            # 使用LangGraph模式
            result = run_agent(
                input_text=request.message,
                session_id=request.session_id,
                use_rag=request.use_rag
            )
            answer = result.get("final_answer", "抱歉，无法生成答案。")
        
        # 注意：历史消息已在 generation_node 中保存，无需重复添加
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id,
            context_used=request.use_rag and len(sources) > 0,
            metadata={
                "use_rag": request.use_rag,
                "use_react": request.use_react
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口
    """
    # 简化版本，实际可以实现SSE流式输出
    return await chat(request)


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """获取会话历史"""
    history_manager = get_chat_history_manager()
    messages = history_manager.get_history(session_id)
    
    return {
        "session_id": session_id,
        "messages": [
            {"type": type(msg).__name__, "content": msg.content}
            for msg in messages
        ]
    }


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """清空会话历史"""
    history_manager = get_chat_history_manager()
    history_manager.clear_session(session_id)
    
    return {"message": "历史已清空", "session_id": session_id}


@router.get("/sessions")
async def get_sessions():
    """获取所有会话"""
    history_manager = get_chat_history_manager()
    sessions = history_manager.get_all_sessions()
    
    return {"sessions": sessions, "count": len(sessions)}

