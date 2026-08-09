"""
用户管理模块 - SQLite 数据库 + 密码哈希 + RBAC

支持：
- 用户注册（用户名 + 密码，Argon2 哈希）
- 用户登录（验证密码）
- JWT Token 认证
- RBAC 角色权限控制

用户数据存储在 data/users.db（SQLite），表结构：
  users(id, username, password_hash, salt, created_at)
  roles(id, name, description, created_at)
  user_roles(user_id, role_id, assigned_at)
  role_permissions(id, role_id, resource, action)
"""
import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager

from .passwords import hash_password, verify_password_and_upgrade

logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "users.db"
LEGACY_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "users.db"

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
        # RBAC 表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_roles (
                user_id INTEGER NOT NULL,
                role_id INTEGER NOT NULL,
                assigned_at REAL NOT NULL,
                PRIMARY KEY (user_id, role_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS role_permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role_id INTEGER NOT NULL,
                resource TEXT NOT NULL,
                action TEXT NOT NULL,
                UNIQUE(role_id, resource, action),
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
            )
        """)
        # 初始化默认角色
        _init_default_roles(cur)
        _merge_legacy_users(cur)
        logger.info(f"用户数据库已初始化: {DB_PATH}")


# ============================================================
# 预定义角色常量
# ============================================================
ADMIN_ROLE = "admin"
EDITOR_ROLE = "editor"
VIEWER_ROLE = "viewer"


def _init_default_roles(cur):
    """初始化默认角色和权限"""
    now = time.time()
    role_map = {}

    for name, description in [
        (ADMIN_ROLE, "管理员 - 全部权限"),
        (EDITOR_ROLE, "资料维护者 - 可读写实验室知识库"),
        (VIEWER_ROLE, "普通成员 - 仅可读"),
    ]:
        cur.execute("SELECT id FROM roles WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            role_map[name] = row["id"]
        else:
            cur.execute(
                "INSERT INTO roles (name, description, created_at) VALUES (?, ?, ?)",
                (name, description, now)
            )
            role_map[name] = cur.lastrowid

    # 设置权限
    permissions = {
        ADMIN_ROLE: [
            ("knowledge", "read"), ("knowledge", "write"), ("knowledge", "delete"), ("knowledge", "admin"),
            ("chat", "read"), ("chat", "write"), ("chat", "admin"),
            ("session", "read"), ("session", "write"), ("session", "delete"), ("session", "admin"),
            ("user", "read"), ("user", "write"), ("user", "delete"), ("user", "admin"),
            ("system", "admin"),
        ],
        EDITOR_ROLE: [
            ("knowledge", "read"), ("knowledge", "write"),
            ("chat", "read"), ("chat", "write"),
            ("session", "read"), ("session", "write"),
        ],
        VIEWER_ROLE: [
            ("knowledge", "read"),
            ("chat", "read"),
            ("session", "read"),
        ],
    }

    for role_name, perms in permissions.items():
        role_id = role_map.get(role_name)
        if role_id is None:
            continue
        for resource, action in perms:
            cur.execute(
                "INSERT OR IGNORE INTO role_permissions (role_id, resource, action) VALUES (?, ?, ?)",
                (role_id, resource, action)
            )


def _merge_legacy_users(cur):
    """增量合并旧路径账号，避免修正数据目录后丢失已有登录用户。"""
    if not LEGACY_DB_PATH.exists() or LEGACY_DB_PATH.resolve() == DB_PATH.resolve():
        return

    cur.execute("ATTACH DATABASE ? AS legacy_users", (str(LEGACY_DB_PATH),))
    cur.execute(
        """INSERT OR IGNORE INTO users (username, password_hash, salt, created_at)
           SELECT username, password_hash, salt, created_at
           FROM legacy_users.users"""
    )
    cur.execute(
        """INSERT OR IGNORE INTO user_roles (user_id, role_id, assigned_at)
           SELECT local_user.id, local_role.id, legacy_link.assigned_at
           FROM legacy_users.user_roles AS legacy_link
           JOIN legacy_users.users AS legacy_user ON legacy_user.id = legacy_link.user_id
           JOIN legacy_users.roles AS legacy_role ON legacy_role.id = legacy_link.role_id
           JOIN users AS local_user ON local_user.username = legacy_user.username
           JOIN roles AS local_role ON local_role.name = legacy_role.name"""
    )


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
        password_hash = hash_password(password)
        created_at = time.time()

        cur.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, "", created_at)
        )

    logger.info(f"用户注册成功: {username}")
    return True, "注册成功"


def verify_user(username: str, password: str) -> Optional[dict]:
    """
    验证用户名和密码

    Returns:
        用户信息 dict（包含 id, username, created_at, role, department 等）或 None
    """
    with _db_cursor() as cur:
        cur.execute(
            "SELECT id, username, password_hash, salt, created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()

    if not row:
        return None

    # 验证密码；旧 SHA-256 账号在首次成功登录后原位升级为 Argon2
    valid, upgraded_hash = verify_password_and_upgrade(
        password, row["password_hash"], row["salt"]
    )
    if not valid:
        return None
    if upgraded_hash:
        with _db_cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = ?, salt = '' WHERE id = ?",
                (upgraded_hash, row["id"]),
            )

    # 获取用户角色（取第一个角色）
    roles = get_user_roles(username)
    role = roles[0] if roles else VIEWER_ROLE

    # 获取用户部门信息（从 user_roles 关联获取，默认为空）
    department = ""
    department_name = ""
    department_path = ""

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
        "role": role,
        "department": department,
        "department_name": department_name,
        "department_path": department_path,
    }


def get_user_by_username(username: str) -> Optional[dict]:
    """根据用户名读取当前账号和授权上下文。"""
    with _db_cursor() as cur:
        cur.execute(
            "SELECT id, username, created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()

    if not row:
        return None

    roles = get_user_roles(username)

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
        "role": roles[0] if roles else VIEWER_ROLE,
        "department": "",
        "department_name": "",
        "department_path": "",
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
        password_hash = hash_password(new_password)
        cur.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
            (password_hash, "", username)
        )

    logger.info(f"密码修改成功: {username}")
    return True, "密码修改成功"


# ============================================================
# RBAC 角色权限管理
# ============================================================

def get_user_roles(username: str) -> List[str]:
    """获取用户的所有角色"""
    with _db_cursor() as cur:
        cur.execute("""
            SELECT r.name FROM roles r
            JOIN user_roles ur ON r.id = ur.role_id
            JOIN users u ON u.id = ur.user_id
            WHERE u.username = ?
            ORDER BY CASE r.name
                WHEN 'admin' THEN 0
                WHEN 'editor' THEN 1
                WHEN 'viewer' THEN 2
                ELSE 3
            END, r.name
        """, (username,))
        return [row["name"] for row in cur.fetchall()]


def assign_role(username: str, role_name: str) -> tuple[bool, str]:
    """为用户分配角色"""
    with _db_cursor() as cur:
        # 查找用户
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_row = cur.fetchone()
        if not user_row:
            return False, "用户不存在"

        # 查找角色
        cur.execute("SELECT id FROM roles WHERE name = ?", (role_name,))
        role_row = cur.fetchone()
        if not role_row:
            return False, f"角色 '{role_name}' 不存在"

        user_id = user_row["id"]
        role_id = role_row["id"]

        # 检查是否已分配
        cur.execute(
            "SELECT 1 FROM user_roles WHERE user_id = ? AND role_id = ?",
            (user_id, role_id)
        )
        if cur.fetchone():
            return True, f"用户已有角色 {role_name}"

        cur.execute(
            "INSERT INTO user_roles (user_id, role_id, assigned_at) VALUES (?, ?, ?)",
            (user_id, role_id, time.time())
        )

    logger.info(f"为用户 {username} 分配角色 {role_name}")
    return True, f"角色 {role_name} 分配成功"


def remove_role(username: str, role_name: str) -> tuple[bool, str]:
    """移除用户角色"""
    with _db_cursor() as cur:
        cur.execute("""
            SELECT u.id FROM users u
            JOIN user_roles ur ON u.id = ur.user_id
            JOIN roles r ON r.id = ur.role_id
            WHERE u.username = ? AND r.name = ?
        """, (username, role_name))
        if not cur.fetchone():
            return False, "用户没有此角色"

        cur.execute("""
            DELETE FROM user_roles
            WHERE user_id = (SELECT id FROM users WHERE username = ?)
            AND role_id = (SELECT id FROM roles WHERE name = ?)
        """, (username, role_name))

    logger.info(f"移除用户 {username} 的角色 {role_name}")
    return True, f"角色 {role_name} 已移除"


def check_permission(username: str, resource: str, action: str) -> bool:
    """
    检查用户是否有指定资源 + 动作的权限。

    admin 角色拥有所有权限，其他角色按权限表精确匹配。
    """
    with _db_cursor() as cur:
        # 检查 admin 角色
        cur.execute("""
            SELECT 1 FROM users u
            JOIN user_roles ur ON u.id = ur.user_id
            JOIN roles r ON r.id = ur.role_id
            WHERE u.username = ? AND r.name = 'admin'
        """, (username,))
        if cur.fetchone():
            return True

        # 检查具体权限
        cur.execute("""
            SELECT 1 FROM users u
            JOIN user_roles ur ON u.id = ur.user_id
            JOIN roles r ON r.id = ur.role_id
            JOIN role_permissions rp ON rp.role_id = r.id
            WHERE u.username = ?
              AND rp.resource = ?
              AND rp.action = ?
            LIMIT 1
        """, (username, resource, action))
        return cur.fetchone() is not None


def require_permission(resource: str, action: str):
    """
    FastAPI 依赖项工厂：检查当前用户是否有指定权限。

    使用方式：
        from fastapi import Depends

        def require(resource: str, action: str):
            return _require_permission_factory(resource, action)

        @router.post("/knowledge/delete")
        async def delete_doc(
            ...,
            user: dict = Depends(require("knowledge", "delete"))
        ):
            ...

    等价于：
        @router.post("/knowledge/delete")
        async def delete_doc(
            ...,
            user: dict = Depends(_require_permission("knowledge", "delete"))
        ):
            ...
    """
    from fastapi import Depends, HTTPException, status
    from .security import get_current_user

    def _check_permission(current_user: dict = Depends(get_current_user)) -> dict:
        username = current_user.get("username", "anonymous")
        if not check_permission(username, resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足：需要 {resource}:{action}"
            )
        return current_user
    return _check_permission


# 常用权限依赖快捷方式（避免每次调用 require_permission）
def require_knowledge_read():
    return require_permission("knowledge", "read")


def require_knowledge_write():
    return require_permission("knowledge", "write")


def require_knowledge_delete():
    return require_permission("knowledge", "delete")


def require_session_write():
    return require_permission("session", "write")


def require_session_delete():
    return require_permission("session", "delete")


def require_user_admin():
    return require_permission("user", "admin")


def require_system_admin():
    return require_permission("system", "admin")


# 初始化数据库（模块加载时自动执行）
try:
    init_user_db()
except Exception:
    logger.warning("用户数据库初始化失败（可能已有表）")
