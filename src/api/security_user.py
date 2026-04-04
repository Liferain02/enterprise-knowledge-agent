"""
用户管理模块 - SQLite 数据库 + 密码哈希

支持：
- 用户注册（用户名 + 密码，加盐 SHA-256 哈希）
- 用户登录（验证密码）
- JWT Token 认证

用户数据存储在 data/users.db（SQLite），表结构：
  users(id, username, password_hash, created_at)
"""
import sqlite3
import hashlib
import secrets
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "users.db"

# SALT 长度
SALT_LENGTH = 32


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _db_cursor():
    conn = _get_connection()
    try:
        yield conn.cursor()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_user_db():
    """初始化用户数据库"""
    with _db_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        logger.info(f"用户数据库已初始化: {DB_PATH}")


def _hash_password(password: str, salt: str) -> str:
    """SHA-256 加盐哈希"""
    combined = salt + password + salt
    for _ in range(3):
        combined = hashlib.sha256(combined.encode()).hexdigest()
    return combined


def _generate_salt() -> str:
    return secrets.token_hex(SALT_LENGTH)


def register_user(username: str, password: str) -> tuple[bool, str]:
    """
    注册新用户

    Args:
        username: 用户名（3-32 字符，字母数字下划线）
        password: 密码（6-128 字符）

    Returns:
        (success, message)
    """
    username = username.strip()
    password = password.strip()

    # 验证
    if len(username) < 3 or len(username) > 32:
        return False, "用户名长度需在 3-32 个字符之间"
    if not username.replace("_", "").isalnum():
        return False, "用户名只能包含字母、数字和下划线"
    if len(password) < 6 or len(password) > 128:
        return False, "密码长度需在 6-128 个字符之间"

    # 检查是否已存在
    with _db_cursor() as cur:
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            return False, "用户名已存在"

        # 创建用户
        salt = _generate_salt()
        password_hash = _hash_password(password, salt)
        created_at = time.time()

        cur.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, created_at)
        )

    logger.info(f"用户注册成功: {username}")
    return True, "注册成功"


def verify_user(username: str, password: str) -> Optional[dict]:
    """
    验证用户名和密码

    Returns:
        用户信息 dict（包含 id, username, created_at）或 None
    """
    with _db_cursor() as cur:
        cur.execute(
            "SELECT id, username, password_hash, salt, created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()

    if not row:
        return None

    # 验证密码
    computed_hash = _hash_password(password, row["salt"])
    if not secrets.compare_digest(computed_hash, row["password_hash"]):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }


def get_user_by_username(username: str) -> Optional[dict]:
    """根据用户名查找用户"""
    with _db_cursor() as cur:
        cur.execute(
            "SELECT id, username, created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()

    if not row:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """修改密码"""
    new_password = new_password.strip()
    if len(new_password) < 6 or len(new_password) > 128:
        return False, "新密码长度需在 6-128 个字符之间"

    user = verify_user(username, old_password)
    if not user:
        return False, "原密码错误"

    with _db_cursor() as cur:
        salt = _generate_salt()
        password_hash = _hash_password(new_password, salt)
        cur.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
            (password_hash, salt, username)
        )

    logger.info(f"密码修改成功: {username}")
    return True, "密码修改成功"


# 初始化数据库（模块加载时自动执行）
try:
    init_user_db()
except Exception:
    logger.warning("用户数据库初始化失败（可能已有表）")
