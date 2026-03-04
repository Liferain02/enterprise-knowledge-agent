"""
聊天服务
只包含业务逻辑，数据库操作委托给 session_service
"""
import logging
from typing import Dict, Any
from agents.graph import run_agent, arun_agent, get_agent_graph
from config.settings import get_settings
from api.services import session_service

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

    async def achat(self, message: str, session_id: str) -> Dict[str, Any]:
        """
        处理聊天请求（异步版本）

        使用 ainvoke 在主事件循环中运行
        避免跨事件循环导致的 MCP 死锁

        Args:
            message: 用户消息
            session_id: 会话ID

        Returns:
            包含 answer, sources, used_agent 的字典
        """
        logger.info(f"收到聊天请求(异步) - session: {session_id}, message: {message[:50]}...")

        # 确保会话存在
        session_service.ensure_session_exists(session_id)

        result = await arun_agent(
            input_text=message,
            session_id=session_id
        )

        answer = result.get("final_answer", "抱歉，无法生成答案。")
        sources = result.get("sources", "")
        used_agent = result.get("used_agent", "unknown")

        # 如果是第一条消息，生成标题
        session = session_service.get_session(session_id)
        if session and session.get("message_count", 0) == 0:
            title = session_service.generate_title(message)
            session_service.update_session_title(session_id, title)

        # 保存消息
        self._save_session_message(session_id, message, answer, used_agent)

        # 格式化来源
        sources_list = []
        if sources and isinstance(sources, str):
            sources_list = [{"content": sources[:200], "metadata": {}}]

        logger.info(f"聊天请求完成 - agent: {used_agent}, answer_length: {len(answer)}")

        return {
            "answer": answer,
            "sources": sources_list,
            "used_agent": used_agent
        }

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
