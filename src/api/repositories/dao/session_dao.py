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

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _ensure_feedback_schema(conn: sqlite3.Connection) -> None:
    """创建反馈表，并为旧数据库补齐知识缺口处理字段。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            used_agent TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            comment TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            resolution_note TEXT,
            resolved_by TEXT,
            resolved_at TEXT
        )
    """)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()
    }
    migrations = {
        "status": "ALTER TABLE feedback ADD COLUMN status TEXT NOT NULL DEFAULT 'open'",
        "resolution_note": "ALTER TABLE feedback ADD COLUMN resolution_note TEXT",
        "resolved_by": "ALTER TABLE feedback ADD COLUMN resolved_by TEXT",
        "resolved_at": "ALTER TABLE feedback ADD COLUMN resolved_at TEXT",
    }
    for column, statement in migrations.items():
        if column not in columns:
            conn.execute(statement)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_status_created "
        "ON feedback(status, created_at DESC)"
    )


class SessionDAO:
    
    """会话数据访问对象"""

    def __init__(self):
        settings = get_settings()
        self._db_path = settings.chroma_dir / "sessions.db"
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

        _ensure_feedback_schema(conn)

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

    def update_message_count(self, session_id: str, count: int):
        """直接设置消息数量（用于修复旧数据不一致问题）"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "UPDATE sessions SET message_count = ?, updated_at = ? WHERE session_id = ?",
            (count, now, session_id),
        )
        conn.commit()
        conn.close()

    def migrate_messages_only(self, old_id: str, new_id: str):
        """仅迁移 messages 表中的 session_id（不碰 sessions 表）"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE messages SET session_id = ? WHERE session_id = ?",
            (new_id, old_id),
        )
        conn.commit()
        conn.close()

    def rename_session_id(self, old_id: str, new_id: str):
        """重命名 session_id（用于修复前缀不一致问题）"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "UPDATE messages SET session_id = ? WHERE session_id = ?",
            (new_id, old_id),
        )
        cursor.execute(
            "UPDATE sessions SET session_id = ?, updated_at = ? WHERE session_id = ?",
            (new_id, now, old_id),
        )
        conn.commit()
        conn.close()

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
        settings = get_settings()
        self._db_path = settings.chroma_dir / "sessions.db"

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


class FeedbackDAO:
    """回答反馈数据访问对象"""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = (
            Path(db_path) if db_path else get_settings().chroma_dir / "sessions.db"
        )
        self._init_db()

    def _init_db(self) -> None:
        db_path = self._get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            _ensure_feedback_schema(conn)

    def _get_db_path(self) -> Path:
        return self._db_path

    def save(
        self,
        username: str,
        session_id: str,
        question: str,
        answer: str,
        used_agent: str,
        feedback_type: str,
        comment: Optional[str] = None,
    ) -> int:
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO feedback (
                username, session_id, question, answer, used_agent,
                feedback_type, comment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (username, session_id, question, answer, used_agent, feedback_type, comment, now))
        conn.commit()
        feedback_id = cursor.lastrowid
        conn.close()
        return feedback_id

    def get_stats(self, username: Optional[str] = None) -> Dict[str, int]:
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        if username:
            cursor.execute("""
                SELECT feedback_type, COUNT(*) FROM feedback
                WHERE username = ?
                GROUP BY feedback_type
            """, (username,))
        else:
            cursor.execute("""
                SELECT feedback_type, COUNT(*) FROM feedback
                GROUP BY feedback_type
            """)

        rows = cursor.fetchall()
        conn.close()

        stats = {
            "total": 0,
            "helpful": 0,
            "incorrect": 0,
            "missing_material": 0,
        }
        for feedback_type, count in rows:
            stats["total"] += count
            if feedback_type in stats:
                stats[feedback_type] = count
        return stats

    def list_issues(
        self,
        username: Optional[str] = None,
        limit: int = 20,
        status: str = "open",
    ) -> List[Dict[str, Any]]:
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        if status not in {"open", "resolved"}:
            conn.close()
            raise ValueError("反馈状态不合法")

        if username:
            cursor.execute("""
                SELECT id, username, session_id, feedback_type, question, comment,
                       status, resolution_note, resolved_by, resolved_at, created_at
                FROM feedback
                WHERE username = ?
                  AND feedback_type IN ('incorrect', 'missing_material')
                  AND status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (username, status, limit))
        else:
            cursor.execute("""
                SELECT id, username, session_id, feedback_type, question, comment,
                       status, resolution_note, resolved_by, resolved_at, created_at
                FROM feedback
                WHERE feedback_type IN ('incorrect', 'missing_material')
                  AND status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (status, limit))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "username": row[1],
                "session_id": row[2],
                "feedback_type": row[3],
                "question": row[4],
                "comment": row[5],
                "status": row[6],
                "resolution_note": row[7],
                "resolved_by": row[8],
                "resolved_at": row[9],
                "created_at": row[10],
            }
            for row in rows
        ]

    def update_issue_status(
        self,
        feedback_id: int,
        status: str,
        resolution_note: Optional[str],
        resolved_by: str,
    ) -> Dict[str, Any]:
        if status not in {"open", "resolved"}:
            raise ValueError("反馈状态不合法")
        note = (resolution_note or "").strip()
        if status == "resolved" and not note:
            raise ValueError("解决问题时必须填写解决说明")

        db_path = self._get_db_path()
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            issue = conn.execute(
                """SELECT id FROM feedback
                   WHERE id = ? AND feedback_type IN ('incorrect', 'missing_material')""",
                (feedback_id,),
            ).fetchone()
            if not issue:
                raise ValueError("反馈问题不存在")

            if status == "resolved":
                conn.execute(
                    """UPDATE feedback
                       SET status = 'resolved', resolution_note = ?,
                           resolved_by = ?, resolved_at = ?
                       WHERE id = ?""",
                    (note, resolved_by, datetime.now().isoformat(), feedback_id),
                )
            else:
                conn.execute(
                    """UPDATE feedback
                       SET status = 'open', resolution_note = NULL,
                           resolved_by = NULL, resolved_at = NULL
                       WHERE id = ?""",
                    (feedback_id,),
                )

            row = conn.execute(
                """SELECT id, username, session_id, feedback_type, question, comment,
                          status, resolution_note, resolved_by, resolved_at, created_at
                   FROM feedback WHERE id = ?""",
                (feedback_id,),
            ).fetchone()
        return dict(row)


# DAO 实例
session_dao = SessionDAO()
message_dao = MessageDAO()
feedback_dao = FeedbackDAO()
