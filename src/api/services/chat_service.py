"""
聊天服务
只包含业务逻辑，数据库操作委托给 session_service
"""
import logging
from typing import Dict, Any
from src.agent.graph import run_agent, arun_agent, get_agent_graph, get_agent_graph_async
from config.settings import get_settings
from .session_service import session_service

logger = logging.getLogger(__name__)


class ChatService:
    """聊天服务类"""

    def __init__(self):
        self.settings = get_settings()

    def chat(self, message: str, session_id: str) -> Dict[str, Any]:
        """
        处理聊天请求（同步版本）

        Args:
            message: 用户消息
            session_id: 会话ID

        Returns:
            包含 answer, sources, used_agent 的字典
        """
        logger.info(f"收到聊天请求 - session: {session_id}, message: {message[:50]}...")

        result = run_agent(
            input_text=message,
            session_id=session_id
        )

        answer = result.get("final_answer", "抱歉，无法生成答案。")
        sources = result.get("sources", "")
        used_agent = result.get("used_agent", "unknown")

        # 格式化来源
        sources_list = []
        if sources and isinstance(sources, str):
            sources_list = [{"content": sources[:200], "metadata": {}}]

        # 保存消息
        self._save_session_message(session_id, message, answer, used_agent)

        logger.info(f"聊天请求完成 - agent: {used_agent}, answer_length: {len(answer)}")

        return {
            "answer": answer,
            "sources": sources_list,
            "used_agent": used_agent
        }

    async def achat(
        self,
        message: str,
        session_id: str,
        username: str = "anonymous",
        images: list = None,
    ) -> Dict[str, Any]:
        """
        处理聊天请求（异步版本，支持多模态图片输入）

        使用 ainvoke 在主事件循环中运行
        避免跨事件循环导致的 MCP 死锁

        Args:
            message: 用户消息
            session_id: 会话ID
            username: 用户名（用于跨会话记忆）
            images: 图片列表（每项为 ImageContent 或 dict）

        Returns:
            包含 answer, sources, used_agent, image_understood 的字典
        """
        import time
        total_start = time.time()

        logger.info(f"收到聊天请求(异步) - session: {session_id}, message: {message[:50]}..., images={len(images) if images else 0}")

        # 确保会话存在
        session_service.ensure_session_exists(session_id)

        # ============================================================
        # 图片理解：先调用 Vision LLM 理解图片，再将理解结果加入消息
        # ============================================================
        image_understood = False
        image_context = ""
        if images:
            try:
                from src.models.vision import understand_images
                from src.api.schemas import ImageContent

                # 转换为统一格式
                parsed_images = []
                for img in images:
                    if isinstance(img, dict):
                        parsed_images.append(ImageContent(**img))
                    else:
                        parsed_images.append(img)

                # 构造图片理解提示
                vision_prompt = (
                    f"用户的问题是：「{message}」。"
                    f"请仔细看图，然后回答这个问题。"
                    f"如果图片中有图表、文字、数据，请完整提取。"
                    f"如果图片内容与问题无关，也请描述图片内容。"
                )

                image_context = await understand_images(parsed_images, prompt=vision_prompt)
                image_understood = True

                logger.info(f"[Vision] 图片理解完成，长度: {len(image_context)} 字符")

                # 将图片理解结果追加到消息中
                if image_context and not image_context.startswith("[图片理解失败"):
                    message = (
                        f"【用户上传的图片内容如下，请结合图片回答用户问题】\n"
                        f"{image_context}\n\n"
                        f"【用户问题】\n{message}"
                    )
            except Exception as e:
                logger.warning(f"[Vision] 图片理解出错: {e}")
                import traceback
                traceback.print_exc()
                image_context = f"[图片理解出错: {str(e)}]"
                image_understood = True  # 仍然标记为已理解（虽然失败了）

        result = await arun_agent(
            input_text=message,
            session_id=session_id,
            user_id=username  # 传递用户名作为 user_id，用于跨会话记忆
        )

        answer = result.get("final_answer", "抱歉，无法生成答案。")
        sources = result.get("sources", "")
        used_agent = result.get("used_agent", "unknown")

        # 如果是第一条消息，生成标题（问候语跳过，节省一次 LLM 调用）
        _GREETING_PATTERNS = r"^(你好|hi|hello|您好|早上好|下午好|晚上好|在吗|嗨)"
        import re
        if re.match(_GREETING_PATTERNS, message.strip(), re.IGNORECASE):
            title = "问候"
        else:
            session = session_service.get_session(session_id)
            if session and session.get("message_count", 0) == 0:
                title = session_service.generate_title(message)
            else:
                title = None
        if title:
            session_service.update_session_title(session_id, title)

        # 保存消息（使用原始消息，不含图片上下文）
        self._save_session_message(session_id, message, answer, used_agent)

        # 格式化来源
        sources_list = []
        if sources and isinstance(sources, str):
            sources_list = [{"content": sources[:200], "metadata": {}}]

        # 性能日志
        elapsed = time.time() - total_start
        if elapsed > 10:
            logger.warning(f"[PERF] 慢查询警告 - 总耗时: {elapsed:.1f}秒, agent: {used_agent}, query: {message[:30]}...")
        else:
            logger.info(f"[PERF] 聊天完成 - 耗时: {elapsed:.1f}秒, agent: {used_agent}")

        return {
            "answer": answer,
            "sources": sources_list,
            "used_agent": used_agent,
            "image_understood": image_understood,
        }

    async def achat_stream(
        self,
        message: str,
        session_id: str,
        username: str = "anonymous",
        images: list = None,
    ):
        """
        流式聊天（Generator）。供 StreamingResponse 使用。

        通过 graph.astream_events() 流式输出。
        SSE 事件类型：
        - session_id: 当前会话 ID
        - tool_start: 开始调用某个 tool
        - tool_end: tool 调用结束
        - agent_step: 当前在哪个 Agent 节点
        - llm_token: LLM 输出的 token（逐字流）
        - thinking: 系统思考状态
        - sources: 引用来源
        - done: 完成标识

        Usage:
            generator = await service.achat_stream(message, session_id, username)
            return StreamingResponse(generator, media_type="text/event-stream")
        """
        import json
        import time
        import asyncio
        from src.agent.graph import get_agent_graph_async
        from langchain_core.messages import HumanMessage

        session_service.ensure_session_exists(session_id)

        # 图片理解（与 achat 相同）
        if images:
            try:
                from src.models.vision import understand_images
                from src.api.schemas import ImageContent

                parsed_images = [
                    ImageContent(**img) if isinstance(img, dict) else img
                    for img in images
                ]
                vision_prompt = (
                    f"用户的问题是：「{message}」。"
                    f"请仔细看图，然后回答这个问题。"
                )
                image_context = await understand_images(parsed_images, prompt=vision_prompt)
                if image_context and not image_context.startswith("[图片理解失败"):
                    message = (
                        f"【用户上传的图片内容如下，请结合图片回答用户问题】\n"
                        f"{image_context}\n\n"
                        f"【用户问题】\n{message}"
                    )
                yield self._sse_event("thinking", "图片理解完成，开始检索知识库...")
            except Exception as e:
                yield self._sse_event("thinking", f"[Vision] 图片理解出错: {e}，继续处理...")

        # 构建 Agent 配置
        graph = await get_agent_graph_async()
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "user_id": username,
        }

        # 流式输出 Agent 事件
        collected_tokens = []

        try:
            # 使用 astream_events 获取每个 LLM token
            async for event in graph.astream_events(
                initial_state,
                config,
                version="v2",
            ):
                event_type = event.get("event", "")

                if event_type == "on_chat_model_stream":
                    # LLM token 输出
                    chunk = event.get("data", {}).get("chunk", {})
                    token = getattr(chunk, "content", "") or ""
                    if token:
                        collected_tokens.append(token)
                        yield self._sse_event("llm_token", token)

                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    yield self._sse_event("tool_start", tool_name)
                    yield self._sse_event(
                        "thinking",
                        f"正在调用工具: {tool_name}..."
                    )

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    input_data = event.get("data", {}).get("input", {})
                    # 提取简短描述
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

            # 最终结果
            final_answer = "".join(collected_tokens)

            # 提取 sources（从最终 state）
            config2 = {"configurable": {"thread_id": session_id}}
            checkpoint = graph.checkpointer.get(config2)
            sources_raw = ""
            used_agent = "unknown"
            if checkpoint:
                state = checkpoint
                sources_raw = state.get("sources", "")
                used_agent = state.get("used_agent", "unknown")

            # 保存消息
            self._save_session_message(session_id, message, final_answer, used_agent)

            # 生成标题（首条消息，跳过问候语节省 LLM 调用）
            import re
            _GREETING_PATTERNS = r"^(你好|hi|hello|您好|早上好|下午好|晚上好|在吗|嗨)"
            if re.match(_GREETING_PATTERNS, message.strip(), re.IGNORECASE):
                title = "问候"
            else:
                session = session_service.get_session(session_id)
                if session and session.get("message_count", 0) == 0:
                    title = session_service.generate_title(message)
                else:
                    title = None
            if title:
                session_service.update_session_title(session_id, title)

            yield self._sse_event("sources", sources_raw[:500] if sources_raw else "")
            yield self._sse_event("used_agent", used_agent)
            yield self._sse_event("done", final_answer[:100])

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield self._sse_event("error", str(e))

    def _sse_event(self, event_type: str, data) -> str:
        """格式化为 SSE 事件"""
        import json
        content = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        return f"data: {content}\n\n"

    def _save_session_message(self, session_id: str, user_message: str, ai_message: str, used_agent: str = None):
        """保存用户和AI的消息"""
        try:
            session_service.save_message(session_id, "user", user_message)
            # 保存助手消息时，带上 agent 信息
            metadata = {"agent": used_agent} if used_agent else None
            session_service.save_message(session_id, "assistant", ai_message, metadata)
        except Exception as e:
            logger.warning(f"保存消息失败: {e}")

    def get_history(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话历史

        Args:
            session_id: 会话ID

        Returns:
            包含消息历史的字典
        """
        logger.info(f"获取历史记录 - session: {session_id}")

        try:
            # 优先从 SQLite 获取
            messages = session_service.get_messages(session_id)

            if messages:
                return {
                    "session_id": session_id,
                    "messages": messages
                }

            # 备用：从 LangGraph checkpointer 获取
            graph = get_agent_graph()
            config = {"configurable": {"thread_id": session_id}}
            checkpoint = graph.checkpointer.get(config)

            if checkpoint is None:
                return {
                    "session_id": session_id,
                    "messages": []
                }

            messages = checkpoint.get("messages", [])

            return {
                "session_id": session_id,
                "messages": [
                    {"type": type(msg).__name__, "content": msg.content}
                    for msg in messages
                ]
            }
        except Exception as e:
            logger.exception(f"获取历史记录失败: {str(e)}")
            return {
                "session_id": session_id,
                "messages": [],
                "error": str(e)
            }

    def clear_history(self, session_id: str) -> Dict[str, Any]:
        """
        清空会话历史

        Args:
            session_id: 会话ID

        Returns:
            操作结果
        """
        logger.info(f"清空历史记录 - session: {session_id}")

        # 删除会话（会同时删除消息）
        session_service.delete_session(session_id)

        return {
            "message": "会话历史已清空",
            "session_id": session_id
        }

    def get_sessions(self) -> Dict[str, Any]:
        """
        获取所有会话列表

        Returns:
            会话列表
        """
        try:
            sessions = session_service.list_sessions(limit=50)
            return {
                "sessions": sessions,
                "count": len(sessions)
            }
        except Exception as e:
            logger.exception(f"获取会话列表失败: {str(e)}")
            return {
                "sessions": [],
                "count": 0,
                "error": str(e)
            }

    def create_session(self, title: str = None) -> Dict[str, Any]:
        """
        创建新会话

        Args:
            title: 会话标题（可选）

        Returns:
            新会话信息
        """
        return session_service.create_session(title)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息
        """
        return session_service.get_session(session_id)

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            操作结果
        """
        return session_service.delete_session(session_id)

    def update_session_title(self, session_id: str, title: str) -> Dict[str, Any]:
        """
        更新会话标题

        Args:
            session_id: 会话ID
            title: 新标题

        Returns:
            操作结果
        """
        return session_service.update_session_title(session_id, title)


# 服务实例
chat_service = ChatService()
