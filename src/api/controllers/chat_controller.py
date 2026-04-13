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
from src.rag.retrieval.acl_filter import UserContext

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
    """聊天接口（异步，返回完整答案）"""
    try:
        username = current_user.get("username", "anonymous")

        # 从 JWT payload 构建 ACL UserContext
        user_context = UserContext.from_jwt_payload(current_user)

        images_data = None
        if request.images:
            images_data = [img.model_dump() if hasattr(img, "model_dump") else img for img in request.images]

        result = await chat_service.achat(
            message=request.message,
            session_id=request.session_id,
            username=username,
            images=images_data,
            user_context=user_context,
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
    """流式聊天接口（SSE）。"""
    try:
        username = current_user.get("username", "anonymous")

        # 从 JWT payload 构建 ACL UserContext
        user_context = UserContext.from_jwt_payload(current_user)

        images_data = None
        if request.images:
            images_data = [
                img.model_dump() if hasattr(img, "model_dump") else img
                for img in request.images
            ]

        generator = chat_service.achat_stream(
            message=request.message,
            session_id=request.session_id,
            username=username,
            images=images_data,
            user_context=user_context,
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
async def get_history(session_id: str, current_user: dict = Depends(get_current_user)):
    """获取会话历史（仅返回当前用户的）"""
    username = current_user.get("username", "anonymous")
    result = chat_service.get_history(username, session_id)
    return result


@router.delete("/history/{session_id}")
async def clear_history(session_id: str, current_user: dict = Depends(get_current_user)):
    """清空会话历史（仅限当前用户）"""
    username = current_user.get("username", "anonymous")
    result = chat_service.clear_history(username, session_id)
    return result


@router.get("/sessions")
async def get_sessions(current_user: dict = Depends(get_current_user)):
    """获取当前用户的所有会话列表"""
    username = current_user.get("username", "anonymous")
    result = chat_service.get_sessions(username)
    return result


@router.post("/sessions")
async def create_session(request: CreateSessionRequest, current_user: dict = Depends(get_current_user)):
    """创建新会话"""
    username = current_user.get("username", "anonymous")
    result = chat_service.create_session(username, request.title)
    return result


@router.put("/sessions/{session_id}/title")
async def update_session_title(session_id: str, request: UpdateTitleRequest, current_user: dict = Depends(get_current_user)):
    """更新会话标题"""
    username = current_user.get("username", "anonymous")
    result = chat_service.update_session_title(username, session_id, request.title)
    return result


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """删除会话"""
    username = current_user.get("username", "anonymous")
    result = chat_service.delete_session(username, session_id)
    return result
