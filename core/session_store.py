"""
会话存储模块
使用 SQLite 持久化会话信息
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 数据库路径（使用 chroma_db 目录，与 Chroma 共用）
DB_PATH = Path(__file__).parent.parent / "chroma_db" / "sessions.db"


def get_db_path() -> Path:
    """获取数据库路径"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


def init_db():
    """初始化数据库"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 创建会话表
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
    
    # 创建消息表
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


def create_session(session_id: str, title: str = None) -> Dict[str, Any]:
    """创建新会话"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    # 如果没有标题，生成默认标题
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
        # 会话已存在
        cursor.execute("SELECT session_id, title, created_at, updated_at, message_count FROM sessions WHERE session_id = ?", (session_id,))
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


def update_session_title(session_id: str, title: str):
    """更新会话标题"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    cursor.execute("""
        UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?
    """, (title, now, session_id))
    
    conn.commit()
    conn.close()


def update_session_time(session_id: str):
    """更新会话最后活跃时间"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    cursor.execute("""
        UPDATE sessions SET updated_at = ? WHERE session_id = ?
    """, (now, session_id))
    
    conn.commit()
    conn.close()


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """获取会话信息"""
    db_path = get_db_path()
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


def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    """列出所有会话（按更新时间倒序）"""
    db_path = get_db_path()
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


def delete_session(session_id: str) -> bool:
    """删除会话"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 删除消息
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    # 删除会话
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    
    return deleted


def save_message(session_id: str, role: str, content: str, metadata: Dict = None):
    """保存消息到数据库"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    metadata_json = json.dumps(metadata) if metadata else None
    
    cursor.execute("""
        INSERT INTO messages (session_id, role, content, created_at, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, role, content, now, metadata_json))
    
    # 更新会话的 message_count 和 updated_at
    cursor.execute("""
        UPDATE sessions SET message_count = message_count + 1, updated_at = ? 
        WHERE session_id = ?
    """, (now, session_id))
    
    conn.commit()
    conn.close()


def get_messages(session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """获取会话消息"""
    db_path = get_db_path()
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


def generate_session_title(first_message: str, llm=None) -> str:
    """根据第一条消息生成会话标题"""
    # 如果有 LLM，使用 LLM 生成更智能的标题
    if llm:
        try:
            prompt = f"""请为以下用户问题生成一个简洁的会话标题（不超过20个中文字符）。

用户问题：{first_message}

请直接输出标题，不要任何解释或格式。"""

            # 强制简短输出
            llm_with_max = llm.bind(max_tokens=50)
            response = llm_with_max.invoke(prompt)
            title = response.content.strip().replace('\n', '')

            # 确保标题不为空且不太长
            if title and len(title) > 0:
                # 限制长度（中文约20个字符）
                if len(title) > 20:
                    title = title[:20]
                return title
        except Exception as e:
            print(f"LLM 生成标题失败: {e}")

    # 降级方案：截取前20个字符
    title = first_message[:20]
    if len(first_message) > 20:
        title += "..."
    return title


# 初始化数据库
init_db()
