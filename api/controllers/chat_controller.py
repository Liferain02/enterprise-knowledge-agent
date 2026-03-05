"""
聊天 Controller
"""
import logging
import json
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.services import chat_service
from api.security import get_current_user

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


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    title: str = Field(default=None, description="会话标题（可选）")


class UpdateTitleRequest(BaseModel):
    """更新标题请求"""
    title: str = Field(description="新标题")


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "enterprise-knowledge-assistant",
        "version": "1.0.0"
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _: dict = Depends(get_current_user)):
    """
    聊天接口

    1. 接收用户的 session_id 和 message
    2. 调用 ChatService 处理业务逻辑
    3. 返回结果

    注意：使用 async def + await 调用异步版本
    让 MCP 工具调用在主事件循环中执行，避免死锁
    """
    try:
        result = await chat_service.achat(
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


async def generate_chat_stream(message: str, session_id: str):
    """生成聊天流式响应 - 直接使用 LLM 流式输出"""
    try:
        from core.llm import get_streaming_llm
        from langchain_core.messages import HumanMessage
        from langchain_core.prompts import ChatPromptTemplate

        # 获取流式 LLM
        llm = get_streaming_llm()

        # 构建提示词
        system_prompt = """你是一个友好的AI助手。请用简洁的语言回答用户的问题。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{message}")
        ])

        # 创建 chain
        chain = prompt | llm

        # 流式调用 LLM
        full_content = ""
        async for chunk in chain.astream({"message": message}):
            content = chunk.content
            if content:
                full_content += content
                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

        # 如果没有内容，发送错误
        if not full_content:
            yield f"data: {json.dumps({'type': 'error', 'message': '生成回答失败'})}\n\n"
            return

        # 发送完成信号
        yield f"data: {json.dumps({'type': 'done', 'used_agent': 'general_agent', 'sources': []})}\n\n"

    except Exception as e:
        logger.exception(f"流式聊天出错: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, _: dict = Depends(get_current_user)):
    """
    流式聊天接口

    使用 Server-Sent Events (SSE) 实现流式输出
    前端需要使用 EventSource API 接收流式响应
    """
    return StreamingResponse(
        generate_chat_stream(request.message, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/history/{session_id}")
async def get_history(session_id: str, _: dict = Depends(get_current_user)):
    """
    获取会话历史

    通过数据库获取历史消息
    """
    result = chat_service.get_history(session_id)
    return result


@router.delete("/history/{session_id}")
async def clear_history(session_id: str, _: dict = Depends(get_current_user)):
    """
    清空会话历史

    会删除整个会话（包括所有消息）
    """
    result = chat_service.clear_history(session_id)
    return result


@router.get("/sessions")
async def get_sessions(_: dict = Depends(get_current_user)):
    """
    获取所有会话列表

    返回按更新时间倒序的所有会话
    """
    result = chat_service.get_sessions()
    return result


@router.post("/sessions")
async def create_session(request: CreateSessionRequest, _: dict = Depends(get_current_user)):
    """
    创建新会话

    返回新会话的 session_id，前端需要使用这个 ID
    """
    try:
        result = chat_service.create_session(request.title)
        return result
    except Exception as e:
        logger.exception(f"创建会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, _: dict = Depends(get_current_user)):
    """
    获取指定会话信息
    """
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, _: dict = Depends(get_current_user)):
    """
    删除指定会话
    """
    result = chat_service.delete_session(session_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail="会话不存在")
    return result


@router.put("/sessions/{session_id}/title")
async def update_session_title(session_id: str, request: UpdateTitleRequest, _: dict = Depends(get_current_user)):
    """
    更新会话标题
    """
    result = chat_service.update_session_title(session_id, request.title)
    return result
