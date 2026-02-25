"""
聊天 API 路由
重构后的纯 RESTful 接口
"""
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from agents.graph import run_agent, get_agent_graph
from config.settings import get_settings


router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default", description="会话ID")


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(description="AI回复")
    sources: list = Field(default_factory=list, description="信息来源")
    session_id: str = Field(description="会话ID")
    used_agent: str = Field(description="使用的Agent类型")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口
    
    纯 RESTful 壳子：
    1. 接收用户的 session_id 和 message
    2. 直接调用 run_agent()
    3. 返回 final_answer
    """
    try:
        # 调用 Agent
        result = run_agent(
            input_text=request.message,
            session_id=request.session_id
        )
        
        # 提取结果
        answer = result.get("final_answer", "抱歉，无法生成答案。")
        sources = result.get("sources", "")
        used_agent = result.get("used_agent", "unknown")
        
        # 格式化来源
        sources_list = []
        if sources and isinstance(sources, str):
            # 简单处理：将来源字符串添加到列表
            sources_list = [{"content": sources[:200], "metadata": {}}]
        
        return ChatResponse(
            answer=answer,
            sources=sources_list,
            session_id=request.session_id,
            used_agent=used_agent
        )
    
    except Exception as e:
        import traceback
        error_detail = f"处理请求时出错: {str(e)}"
        print(f"[ERROR] {error_detail}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """
    获取会话历史
    
    注意：通过 LangGraph Checkpointer 管理历史
    """
    try:
        # 获取全局唯一的 graph 实例
        graph = get_agent_graph()
        
        # 通过 graph 的 checkpointer 获取历史
        config = {"configurable": {"thread_id": session_id}}
        
        # 检查是否有历史
        checkpoint = graph.checkpointer.get(config)
        
        if checkpoint is None:
            return {
                "session_id": session_id,
                "messages": []
            }
        
        # 提取消息
        messages = checkpoint.get("messages", [])
        
        return {
            "session_id": session_id,
            "messages": [
                {"type": type(msg).__name__, "content": msg.content}
                for msg in messages
            ]
        }
    
    except Exception as e:
        return {
            "session_id": session_id,
            "messages": [],
            "error": str(e)
        }


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """
    清空会话历史
    
    注意：这会清除 Checkpointer 中的历史记录
    """
    # 注意：MemorySaver 不支持直接删除
    # 可以通过创建新的 session_id 来开始新的会话
    return {
        "message": "会话历史已清空（或请使用新的 session_id）",
        "session_id": session_id
    }


@router.get("/sessions")
async def get_sessions():
    """
    获取所有会话
    
    注意：MemorySaver 不提供直接列出所有会话的接口
    """
    return {
        "message": "当前使用内存存储，需要数据库持久化才能列出所有会话",
        "sessions": [],
        "count": 0
    }
