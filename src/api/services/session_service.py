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
import sqlite3
import re
from typing import Dict, Any, List, Optional
from ..repositories import session_dao, message_dao
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

    def update_message_count(self, user_id: str, session_id: str, count: int):
        """直接设置消息数量（用于修复旧数据不一致）"""
        full_id = _make_session_id(user_id, session_id)
        session_dao.update_message_count(full_id, count)

    def migrate_orphaned_messages(self, user_id: str, session_id: str):
        """
        修复历史遗留的前缀不一致问题。

        问题场景：
        1. sessions 表用 full_id (如 Liferain_default)，messages 表用 raw_id (default)
        2. sessions 表无记录，但 messages 表有 raw_id 记录

        本方法将 messages 里的 raw_id 迁移到 full_id。
        """
        safe_user = user_id.replace("/", "_").replace("_", "__")
        full_id = f"{safe_user}_{session_id}"
        raw_id = session_id

        db_path = session_dao._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (raw_id,),
        )
        raw_count = cursor.fetchone()[0]
        conn.close()

        if raw_count == 0:
            return

        logger.info(
            f"迁移 messages: {raw_id} -> {full_id} ({raw_count} 条消息)"
        )

        # 如果 sessions 表没有 full_id，先创建
        existing = session_dao.get_by_id(full_id)
        if not existing:
            session_dao.create_session(full_id)

        # 迁移 messages 到 full_id（sessions 表已有正确记录，无需更新）
        session_dao.migrate_messages_only(raw_id, full_id)
        # 修正 message_count
        session_dao.update_message_count(full_id, raw_count)

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
        """
        获取会话消息（直接用 user_id 拼接 key）。
        如果 full_id 下无消息，fallback 到 raw_id（兼容旧数据）。
        """
        full_id = _make_session_id(user_id, session_id)
        raw_id = session_id

        messages = message_dao.get_by_session(full_id, limit)

        # Fallback: 如果 full_id 下没有，试试 raw_id（兼容旧数据）
        if not messages:
            messages = message_dao.get_by_session(raw_id, limit)

        return messages

    def generate_title(self, first_message: str) -> str:
        """根据第一条消息生成会话标题"""
        return _generate_title(first_message)


def _generate_title(first_message: str) -> str:
    """从首条问题生成确定性标题；标题不值得增加一次阻塞式 LLM 调用。"""
    title = re.sub(r"\s+", " ", first_message or "").strip()
    if not title:
        return "新会话"
    return title[:20] + ("..." if len(title) > 20 else "")


# 服务实例
session_service = SessionService()
