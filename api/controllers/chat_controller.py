"""
聊天 Controller
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.services import chat_service

logger = logging.getLogger(__name__)

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

    1. 接收用户的 session_id 和 message
    2. 调用 ChatService 处理业务逻辑
    3. 返回结果
    """
    try:
        result = chat_service.chat(
            message=request.message,
            session_id=request.session_id
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id,
            used_agent=result["used_agent"]
        )
    except Exception as e:
        logger.exception(f"聊天请求失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """
    获取会话历史

    通过 LangGraph Checkpointer 管理历史
    """
    result = chat_service.get_history(session_id)
    return result


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """
    清空会话历史

    注意：MemorySaver 不支持直接删除，可以通过创建新的 session_id 来开始新的会话
    """
    result = chat_service.clear_history(session_id)
    return result


@router.get("/sessions")
async def get_sessions():
    """
    获取所有会话

    注意：MemorySaver 不提供直接列出所有会话的接口
    """
    result = chat_service.get_sessions()
    return result
