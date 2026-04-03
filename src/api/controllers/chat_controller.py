"""
聊天 Router
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from ..schemas import (
    ChatRequest, ChatResponse,
    CreateSessionRequest, UpdateTitleRequest, ChatStreamRequest
)
from ..services import chat_service
from ..security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "enterprise-knowledge-assistant",
        "version": "1.0.0"
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    聊天接口（同步版本，返回完整答案）
    """
    try:
        username = current_user.get("username", "anonymous")

        images_data = None
        if request.images:
            images_data = [img.model_dump() if hasattr(img, "model_dump") else img for img in request.images]

        result = await chat_service.achat(
            message=request.message,
            session_id=request.session_id,
            username=username,
            images=images_data,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id,
            used_agent=result["used_agent"],
            image_understood=result.get("image_understood", False),
        )
    except Exception as e:
        logger.exception(f"聊天请求失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    流式聊天接口（SSE）。

    实时流式返回 LLM token、工具调用状态、思考进度。
    前端通过 EventSource 消费 /api/v1/chat/stream。

    SSE 事件格式：
        data: {"type": "thinking", "data": "正在检索知识库..."}
        data: {"type": "llm_token", "data": "根据"}
        data: {"type": "llm_token", "data": "公司"}
        ...
        data: {"type": "done", "data": "..."}
    """
    try:
        username = current_user.get("username", "anonymous")

        images_data = None
        if request.images:
            images_data = [
                img.model_dump() if hasattr(img, "model_dump") else img
                for img in request.images
            ]

        # achat_stream 是异步生成器
        generator = await chat_service.achat_stream(
            message=request.message,
            session_id=request.session_id,
            username=username,
            images=images_data,
        )

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as e:
        logger.exception(f"流式聊天请求失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@router.get("/history/{session_id}")
async def get_history(session_id: str, _: dict = Depends(get_current_user)):
    """获取会话历史"""
    result = chat_service.get_history(session_id)
    return result


@router.delete("/history/{session_id}")
async def clear_history(session_id: str, _: dict = Depends(get_current_user)):
    """清空会话历史"""
    result = chat_service.clear_history(session_id)
    return result


@router.get("/sessions")
async def get_sessions(_: dict = Depends(get_current_user)):
    """获取所有会话列表"""
    result = chat_service.get_sessions()
    return result


@router.post("/sessions")
async def create_session(request: CreateSessionRequest, _: dict = Depends(get_current_user)):
    """创建新会话"""
    result = chat_service.create_session(request.title)
    return result


@router.put("/sessions/{session_id}/title")
async def update_session_title(session_id: str, request: UpdateTitleRequest, _: dict = Depends(get_current_user)):
    """更新会话标题"""
    result = chat_service.update_session_title(session_id, request.title)
    return result


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, _: dict = Depends(get_current_user)):
    """删除会话"""
    result = chat_service.delete_session(session_id)
    return result
