"""
会话服务层
负责业务逻辑处理
"""
import uuid
from typing import Dict, Any, List
from ..repositories import session_dao, message_dao
from src.models.llm import get_llm
import logging

logger = logging.getLogger(__name__)


class SessionService:
    """会话服务类"""

    def create_session(self, title: str = None) -> Dict[str, Any]:
        """创建新会话"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        return session_dao.create_session(session_id, title)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        return session_dao.get_by_id(session_id)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出所有会话"""
        return session_dao.list_all(limit)

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """删除会话"""
        deleted = session_dao.delete(session_id)
        return {
            "success": deleted,
            "session_id": session_id
        }

    def update_session_title(self, session_id: str, title: str) -> Dict[str, Any]:
        """更新会话标题"""
        session_dao.update_title(session_id, title)
        return {
            "success": True,
            "session_id": session_id,
            "title": title
        }

    def ensure_session_exists(self, session_id: str):
        """确保会话存在（如果不存在则创建）"""
        session = session_dao.get_by_id(session_id)
        if not session:
            session_dao.create_session(session_id)

    def save_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """保存消息"""
        message_dao.save(session_id, role, content, metadata)

    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取会话消息"""
        return message_dao.get_by_session(session_id, limit)

    def generate_title(self, first_message: str) -> str:
        """根据第一条消息生成会话标题"""
        return _generate_title_with_llm(first_message)


def _generate_title_with_llm(first_message: str) -> str:
    """使用 LLM 生成会话标题"""
    llm = get_llm()

    prompt = f"""请为以下用户问题生成一个简洁的会话标题（不超过20个中文字符）。

用户问题：{first_message}

请直接输出标题，不要任何解释或格式。"""

    try:
        llm_with_max = llm.bind(max_tokens=50)
        response = llm_with_max.invoke(prompt)
        title = response.content.strip().replace('\n', '')

        if title and len(title) > 0:
            if len(title) > 20:
                title = title[:20]
            return title
    except Exception as e:
        logger.warning(f"LLM 生成标题失败: {e}")

    # 降级方案
    title = first_message[:20]
    if len(first_message) > 20:
        title += "..."
    return title


# 服务实例
session_service = SessionService()
