"""
聊天服务 - 用户隔离版
session_id 全部通过 session_service 统一加上 username 前缀，实现用户间完全隔离。
"""
import logging
import re
import time
from typing import Dict, Any, AsyncGenerator, Optional, List
from src.agent.graph import run_agent, arun_agent, get_agent_graph, get_agent_graph_async
from config.settings import get_settings
from .session_service import session_service

logger = logging.getLogger(__name__)

# 问候语正则（用于跳过 Mem0 检索和标题生成）
_GREETING_PATTERNS = re.compile(
    r"^(你好|hi|hello|您好|早上好|下午好|晚上好|在吗|嗨)",
    re.IGNORECASE
)


class ChatService:
    """聊天服务类（用户隔离）"""

    def __init__(self):
        self.settings = get_settings()

    async def _prepare_message(
        self,
        message: str,
        images: Optional[List] = None,
    ) -> str:
        if not images:
            return message

        try:
            from src.models.vision import understand_images
            from src.api.schemas import ImageContent

            parsed_images = [
                ImageContent(**img) if isinstance(img, dict) else img
                for img in images
            ]
            vision_prompt = f"用户的问题是：「{message}」。请仔细看图，然后回答这个问题。"
            image_context = await understand_images(parsed_images, prompt=vision_prompt)
            if image_context and not image_context.startswith("[图片理解失败"):
                return (
                    f"【用户上传的图片内容如下，请结合图片回答用户问题】\n"
                    f"{image_context}\n\n"
                    f"【用户问题】\n{message}"
                )
            return message
        except Exception as e:
            logger.warning(f"[Vision] 图片理解出错: {e}")
            import traceback
            traceback.print_exc()
            return message

    def _generate_title(
        self,
        user_id: str,
        message: str,
        session_id: str,
    ) -> Optional[str]:
        """生成会话标题（若为首条消息）"""
        if _GREETING_PATTERNS.match(message.strip()):
            return "问候"

        session = session_service.get_session(user_id, session_id)
        if session and session.get("message_count", 0) == 0:
            return session_service.generate_title(message)
        return None

    def _save_chat_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        answer: str,
        used_agent: str,
    ) -> None:
        """保存用户和助手的聊天消息"""
        try:
            session_service.save_message(user_id, session_id, "user", message)
            metadata = {"agent": used_agent} if used_agent else None
            session_service.save_message(user_id, session_id, "assistant", answer, metadata)
        except Exception as e:
            logger.warning(f"保存消息失败: {e}")

    def _sse_event(self, event_type: str, data) -> str:
        """格式化为 SSE 事件"""
        import json
        content = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        return f"data: {content}\n\n"

    def _format_sources(self, sources: Any) -> List[Dict[str, Any]]:
        if sources and isinstance(sources, str):
            return [{"content": sources[:200], "metadata": {}}]
        return []

    # ==================== 对外接口 ====================

    async def achat(
        self,
        message: str,
        session_id: str,
        username: str = "anonymous",
        images: list = None,
    ) -> Dict[str, Any]:
        """
        处理聊天请求（异步版本，支持多模态图片输入）。
        user_id 传入 session_service，保证用户间 session 隔离。
        """
        total_start = time.time()
        user_id = username

        logger.info(
            f"收到聊天请求 - session: {session_id}, user: {user_id}, "
            f"message: {message[:50]}..., images={len(images) if images else 0}"
        )

        session_service.ensure_session_exists(user_id, session_id)

        # 图片理解
        processed_message = await self._prepare_message(message, images)

        result = await arun_agent(
            input_text=processed_message,
            session_id=session_id,
            user_id=user_id,
        )

        answer = result.get("final_answer", "抱歉，无法生成答案。")
        sources = result.get("sources", "")
        used_agent = result.get("used_agent", "unknown")

        # 生成标题（首条消息）
        title = self._generate_title(user_id, processed_message, session_id)
        if title:
            session_service.update_session_title(user_id, session_id, title)

        # 保存消息
        self._save_chat_message(user_id, session_id, message, answer, used_agent)

        elapsed = time.time() - total_start
        if elapsed > 10:
            logger.warning(
                f"[PERF] 慢查询警告 - 总耗时: {elapsed:.1f}秒, "
                f"agent: {used_agent}, query: {message[:30]}..."
            )
        else:
            logger.info(f"[PERF] 聊天完成 - 耗时: {elapsed:.1f}秒, agent: {used_agent}")

        return {
            "answer": answer,
            "sources": self._format_sources(sources),
            "used_agent": used_agent,
            "image_understood": bool(images),
        }

    async def achat_stream(
        self,
        message: str,
        session_id: str,
        username: str = "anonymous",
        images: list = None,
    ) -> AsyncGenerator[str, None]:
        """流式聊天 SSE Generator。user_id 传入所有 session 操作。"""
        from langchain_core.messages import HumanMessage

        user_id = username
        session_service.ensure_session_exists(user_id, session_id)

        # 图片理解
        if images:
            processed_message = await self._prepare_message(message, images)
            yield self._sse_event("thinking", "图片理解完成，开始检索知识库...")
        else:
            processed_message = message

        # 构建 Agent 配置
        graph = await get_agent_graph_async()
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }
        initial_state = {
            "messages": [HumanMessage(content=processed_message)],
            "session_id": session_id,
            "user_id": user_id,
        }

        collected_tokens = []

        try:
            async for event in graph.astream_events(
                initial_state,
                config,
                version="v2",
            ):
                event_type = event.get("event", "")

                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    token = getattr(chunk, "content", "") or ""
                    if token:
                        collected_tokens.append(token)
                        yield self._sse_event("llm_token", token)

                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    yield self._sse_event("tool_start", tool_name)
                    yield self._sse_event("thinking", f"正在调用工具: {tool_name}...")

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    input_data = event.get("data", {}).get("input", {})
                    summary = str(input_data)[:100] if input_data else ""
                    yield self._sse_event("tool_end", {"tool": tool_name, "summary": summary})
                    yield self._sse_event("thinking", f"工具 {tool_name} 执行完成")

                elif event_type == "on_chain_start":
                    name = event.get("name", "")
                    if name and name not in ("LangGraph",):
                        yield self._sse_event("agent_step", name)

                elif event_type == "on_chain_end":
                    name = event.get("name", "")
                    if name == "maybe_summarize":
                        yield self._sse_event("thinking", "对话摘要已完成")

            final_answer = "".join(collected_tokens)

            # 从 checkpointer 提取 sources
            config2 = {"configurable": {"thread_id": session_id}}
            checkpoint = graph.checkpointer.get(config2)
            sources_raw = ""
            used_agent = "unknown"
            if checkpoint:
                sources_raw = checkpoint.get("sources", "")
                used_agent = checkpoint.get("used_agent", "unknown")

            # 保存消息（带 user_id 隔离）
            self._save_chat_message(user_id, session_id, message, final_answer, used_agent)

            # 生成标题
            title = self._generate_title(user_id, processed_message, session_id)
            if title:
                session_service.update_session_title(user_id, session_id, title)

            yield self._sse_event("sources", sources_raw[:500] if sources_raw else "")
            yield self._sse_event("used_agent", used_agent)
            yield self._sse_event("done", final_answer[:100])

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield self._sse_event("error", str(e))

    # ==================== 会话管理（全部带 user_id） ====================

    def get_history(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """获取会话历史"""
        logger.info(f"获取历史记录 - user: {user_id}, session: {session_id}")
        try:
            messages = session_service.get_messages(user_id, session_id)
            if messages:
                return {"session_id": session_id, "messages": messages}

            # 降级：从 checkpointer 读取（仍然受 user_id 控制的 session_id 影响）
            graph = get_agent_graph()
            config = {"configurable": {"thread_id": session_id}}
            checkpoint = graph.checkpointer.get(config)
            if checkpoint is None:
                return {"session_id": session_id, "messages": []}

            messages = checkpoint.get("messages", [])
            return {
                "session_id": session_id,
                "messages": [
                    {"type": type(msg).__name__, "content": msg.content}
                    for msg in messages
                ],
            }
        except Exception as e:
            logger.exception(f"获取历史记录失败: {str(e)}")
            return {"session_id": session_id, "messages": [], "error": str(e)}

    def clear_history(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """清空会话历史"""
        logger.info(f"清空历史记录 - user: {user_id}, session: {session_id}")
        session_service.delete_session(user_id, session_id)
        return {"message": "会话历史已清空", "session_id": session_id}

    def get_sessions(self, user_id: str) -> Dict[str, Any]:
        """获取当前用户的所有会话列表"""
        try:
            sessions = session_service.list_sessions(user_id, limit=50)
            return {"sessions": sessions, "count": len(sessions)}
        except Exception as e:
            logger.exception(f"获取会话列表失败: {str(e)}")
            return {"sessions": [], "count": 0, "error": str(e)}

    def create_session(self, user_id: str, title: str = None) -> Dict[str, Any]:
        """创建新会话"""
        return session_service.create_session(user_id, title)

    def get_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        return session_service.get_session(user_id, session_id) or {}

    def delete_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """删除会话"""
        return session_service.delete_session(user_id, session_id)

    def update_session_title(self, user_id: str, session_id: str, title: str) -> Dict[str, Any]:
        """更新会话标题"""
        return session_service.update_session_title(user_id, session_id, title)


# 服务实例
chat_service = ChatService()
