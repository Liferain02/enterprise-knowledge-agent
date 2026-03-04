"""
会话数据访问对象 (DAO)
负责数据库的 CRUD 操作
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SessionDAO:
    """会话数据访问对象"""

    def __init__(self):
        self._db_path = Path(__file__).parent.parent / "chroma_db" / "sessions.db"
        self._init_db()

    def _get_db_path(self) -> Path:
        """获取数据库路径"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return self._db_path

    def _init_db(self):
        """初始化数据库"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"会话数据库初始化完成: {db_path}")

    def create_session(self, session_id: str, title: str = None) -> Dict[str, Any]:
        """创建新会话"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        if not title:
            title = f"新会话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        try:
            cursor.execute("""
                INSERT INTO sessions (session_id, title, created_at, updated_at, message_count)
                VALUES (?, ?, ?, ?, 0)
            """, (session_id, title, now, now))
            conn.commit()

            return {
                "session_id": session_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "message_count": 0
            }
        except sqlite3.IntegrityError:
            cursor.execute(
                "SELECT session_id, title, created_at, updated_at, message_count FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            return {
                "session_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4]
            }
        finally:
            conn.close()

    def update_title(self, session_id: str, title: str):
        """更新会话标题"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute("""
            UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?
        """, (title, now, session_id))

        conn.commit()
        conn.close()

    def update_time(self, session_id: str):
        """更新会话最后活跃时间"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute("""
            UPDATE sessions SET updated_at = ? WHERE session_id = ?
        """, (now, session_id))

        conn.commit()
        conn.close()

    def get_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取会话"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, title, created_at, updated_at, message_count
            FROM sessions WHERE session_id = ?
        """, (session_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "session_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4]
            }
        return None

    def list_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出所有会话"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, title, created_at, updated_at, message_count
            FROM sessions ORDER BY updated_at DESC LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "session_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4]
            }
            for row in rows
        ]

    def delete(self, session_id: str) -> bool:
        """删除会话"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()

        return deleted


class MessageDAO:
    """消息数据访问对象"""

    def __init__(self):
        self._db_path = Path(__file__).parent.parent / "chroma_db" / "sessions.db"

    def _get_db_path(self) -> Path:
        return self._db_path

    def save(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """保存消息"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO messages (session_id, role, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, role, content, now, metadata_json))

        cursor.execute("""
            UPDATE sessions SET message_count = message_count + 1, updated_at = ?
            WHERE session_id = ?
        """, (now, session_id))

        conn.commit()
        conn.close()

    def get_by_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取会话的所有消息"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT role, content, created_at, metadata
            FROM messages WHERE session_id = ?
            ORDER BY created_at ASC LIMIT ?
        """, (session_id, limit))

        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in rows:
            metadata = json.loads(row[3]) if row[3] else {}
            messages.append({
                "role": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": metadata
            })

        return messages


# DAO 实例
session_dao = SessionDAO()
message_dao = MessageDAO()
