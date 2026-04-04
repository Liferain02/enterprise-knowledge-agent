"""
会话服务层 - 用户隔离版
session_id 统一加上 username_ 前缀，实现用户间完全隔离。

前端传入的 session_id 为原始 ID（如 "default", "abc123"），
在 service 层自动拼接为 "{username}_{原始id}" 后存入数据库。
安全性保证：
  - 前端只能操作自己 username 前缀下的 session
  - list_sessions 只返回当前用户的会话
  - 其他操作直接用完整 key，无法跨用户访问
"""


import uuid
from typing import Dict, Any, List, Optional
from ..repositories import session_dao, message_dao
from src.models.llm import get_llm
import logging

logger = logging.getLogger(__name__)


def _make_session_id(user_id: str, session_id: str) -> str:
    """
    生成用户隔离的 session_id（存入 DB 的完整 key）。

    安全处理：用户名中的 _ 会被转义为 __，
    因此 {"alice_test"}_{"session_abc"} → alice__test_session_abc
    """
    safe_user = user_id.replace("/", "_").replace("_", "__")
    return f"{safe_user}_{session_id}"


class SessionService:
    """会话服务类（用户隔离）"""

    def create_session(self, user_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        """创建新会话（自动加上 user 前缀）"""
        raw_id = f"session_{uuid.uuid4().hex[:12]}"
        full_id = _make_session_id(user_id, raw_id)
        result = session_dao.create_session(full_id, title)
        # 返回给前端的是原始 id（不含前缀）
        result["session_id"] = raw_id
        return result

    def get_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话（直接用 user_id 拼接 key，无法跨用户访问）"""
        full_id = _make_session_id(user_id, session_id)
        result = session_dao.get_by_id(full_id)
        if result:
            result["session_id"] = session_id
        return result

    def list_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """列出当前用户的所有会话（按前缀过滤）"""
        all_sessions = session_dao.list_all(limit * 10)
        prefix = f"{user_id.replace('/', '_').replace('_', '__')}_"
        user_sessions = [
            s for s in all_sessions
            if s["session_id"].startswith(prefix)
        ]
        # 去掉前缀，返还原始 id
        for s in user_sessions:
            s["session_id"] = s["session_id"][len(prefix):]
        return sorted(user_sessions, key=lambda x: x["updated_at"], reverse=True)[:limit]

    def delete_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """删除会话（直接用 user_id 拼接 key，无法跨用户删除）"""
        full_id = _make_session_id(user_id, session_id)
        deleted = session_dao.delete(full_id)
        return {"success": deleted, "session_id": session_id}

    def update_session_title(self, user_id: str, session_id: str, title: str) -> Dict[str, Any]:
        """更新会话标题（直接用 user_id 拼接 key）"""
        full_id = _make_session_id(user_id, session_id)
        session_dao.update_title(full_id, title)
        return {"success": True, "session_id": session_id, "title": title}

    def ensure_session_exists(self, user_id: str, session_id: str):
        """确保会话存在（不存在则自动创建，自动加前缀）"""
        full_id = _make_session_id(user_id, session_id)
        existing = session_dao.get_by_id(full_id)
        if not existing:
            session_dao.create_session(full_id)

    def save_message(self, user_id: str, session_id: str, role: str, content: str, metadata: Dict = None):
        """保存消息"""
        full_id = _make_session_id(user_id, session_id)
        message_dao.save(full_id, role, content, metadata)

    def get_messages(self, user_id: str, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取会话消息（直接用 user_id 拼接 key）"""
        full_id = _make_session_id(user_id, session_id)
        return message_dao.get_by_session(full_id, limit)

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

    title = first_message[:20]
    if len(first_message) > 20:
        title += "..."
    return title


# 服务实例
session_service = SessionService()
