"""
聊天服务
"""
import logging
from typing import Dict, Any
from agents.graph import run_agent, arun_agent, get_agent_graph
from config.settings import get_settings

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

        result = await arun_agent(
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

        logger.info(f"聊天请求完成 - agent: {used_agent}, answer_length: {len(answer)}")

        return {
            "answer": answer,
            "sources": sources_list,
            "used_agent": used_agent
        }

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
        return {
            "message": "会话历史已清空（或请使用新的 session_id）",
            "session_id": session_id
        }

    def get_sessions(self) -> Dict[str, Any]:
        """
        获取所有会话

        Returns:
            会话列表（当前使用内存存储）
        """
        return {
            "message": "当前使用内存存储，需要数据库持久化才能列出所有会话",
            "sessions": [],
            "count": 0
        }


# 服务实例
chat_service = ChatService()
