"""
实验室科研智能助手数据库层 - SQLite
包含用户、角色、权限、项目组、文档元数据和审计日志等核心表。
"""
import sqlite3
import time
import logging
import re
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from enum import Enum
from dataclasses import dataclass

from .passwords import hash_password, verify_password_and_upgrade

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "lab_assistant.db"
LEGACY_DB_PATH = DATA_DIR / "enterprise.db"
if not DB_PATH.exists() and LEGACY_DB_PATH.exists():
    shutil.copy2(LEGACY_DB_PATH, DB_PATH)

class RoleEnum(str, Enum):
    ADMIN = "admin"
    PI = "pi"
    TEACHER = "teacher"
    LAB_ADMIN = "lab_admin"
    SENIOR_STUDENT = "senior_student"
    STUDENT = "student"
    ASSISTANT = "assistant"


class PermissionEnum(str, Enum):
    DOC_UPLOAD = "doc:upload"
    DOC_DELETE = "doc:delete"
    DOC_UPDATE = "doc:update"
    DOC_READ = "doc:read"
    KB_MANAGE = "kb:manage"
    KB_READ = "kb:read"
    USER_MANAGE = "user:manage"
    AUDIT_READ = "audit:read"
    SYS_CONFIG = "sys:config"


class DocStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class AuditAction(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    DOC_UPLOAD = "doc:upload"
    DOC_DELETE = "doc:delete"
    DOC_UPDATE = "doc:update"
    DOC_VIEW = "doc:view"
    DOC_SEARCH = "doc:search"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    PERMISSION_CHANGE = "permission:change"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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


def init_database():
    """初始化所有表结构"""
    with _db_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                parent_id INTEGER REFERENCES departments(id),
                path TEXT NOT NULL,
                description TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                is_system INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS role_permissions (
                role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
                permission TEXT NOT NULL,
                PRIMARY KEY (role_id, permission)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role_id INTEGER REFERENCES roles(id),
                department_id INTEGER REFERENCES departments(id),
                is_active INTEGER DEFAULT 1,
                is_superadmin INTEGER DEFAULT 0,
                last_login_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                tags TEXT,
                version TEXT NOT NULL DEFAULT '1.0',
                effective_date TEXT,
                expiry_date TEXT,
                confidentiality TEXT DEFAULT 'internal',
                department_restrict TEXT,
                role_restrict TEXT,
                status TEXT DEFAULT 'draft',
                file_path TEXT,
                file_hash TEXT,
                file_size INTEGER,
                uploaded_by TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                description TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                resource_name TEXT,
                ip_address TEXT,
                user_agent TEXT,
                detail TEXT,
                created_at REAL NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_category ON documents(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at)")
    _init_default_data()
    logger.info(f"Database initialized: {DB_PATH}")


def _init_default_data():
    now = time.time()
    default_roles = [
        ("admin", "系统管理员，拥有所有权限", 1, [
            "doc:upload", "doc:delete", "doc:update", "doc:read",
            "kb:manage", "kb:read", "user:manage", "audit:read", "sys:config",
        ]),
        ("pi", "导师/PI，可查看和管理受限资料", 1, [
            "doc:upload", "doc:delete", "doc:update", "doc:read", "kb:read",
        ]),
        ("teacher", "教师，可维护研究方向和项目资料", 1, [
            "doc:upload", "doc:delete", "doc:update", "doc:read", "kb:read",
        ]),
        ("lab_admin", "实验室管理员，可维护制度、FAQ 与流程资料", 1, [
            "doc:upload", "doc:delete", "doc:update", "doc:read", "kb:read",
        ]),
        ("senior_student", "高年级成员，可维护部分项目资料", 1, [
            "doc:upload", "doc:delete", "doc:update", "doc:read", "kb:read",
        ]),
        ("student", "研究生成员，仅可查看公共资料", 1, [
            "doc:read", "kb:read",
        ]),
        ("assistant", "助研/本科生，可查看公共资料", 1, [
            "doc:read", "kb:read",
        ]),
        ("manager", "兼容旧角色：项目负责人", 1, [
            "doc:upload", "doc:delete", "doc:update", "doc:read", "kb:read",
        ]),
        ("hr", "兼容旧角色：实验室管理员", 1, [
            "doc:upload", "doc:delete", "doc:update", "doc:read", "kb:read",
        ]),
        ("employee", "兼容旧角色：研究组成员", 1, [
            "doc:read", "kb:read",
        ]),
        ("it_support", "兼容旧角色：平台支持", 1, [
            "doc:upload", "doc:delete", "doc:update", "doc:read", "kb:read",
        ]),
    ]
    with _db_cursor() as cur:
        for name, desc, is_sys, perms in default_roles:
            cur.execute("SELECT id FROM roles WHERE name = ?", (name,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO roles (name, description, is_system, created_at) VALUES (?, ?, ?, ?)",
                    (name, desc, is_sys, now)
                )
                role_id = cur.lastrowid
                for perm in perms:
                    cur.execute(
                        "INSERT OR IGNORE INTO role_permissions (role_id, permission) VALUES (?, ?)",
                        (role_id, perm)
                    )
        cur.execute("SELECT id FROM departments WHERE name = '实验室'")
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO departments (name, parent_id, path, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("实验室", None, "/实验室", "实验室根节点", now, now)
            )


def write_audit_log(
    user_id: Optional[int],
    username: str,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    detail: Optional[str] = None,
):
    with _db_cursor() as cur:
        cur.execute(
            """INSERT INTO audit_logs
               (user_id, username, action, resource_type, resource_id,
                resource_name, ip_address, user_agent, detail, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, username, action, resource_type, resource_id,
             resource_name, ip_address, user_agent, detail, time.time())
        )


# ==================== 用户操作 ====================

def create_user(
    username: str,
    email: str,
    password: str,
    role_name: str = "student",
    department_id: Optional[int] = None,
) -> tuple[bool, str, Optional[int]]:
    username = username.strip()
    email = email.strip().lower()
    password = password.strip()
    if len(username) < 3 or len(username) > 32:
        return False, "用户名需在 3-32 字符之间", None
    if not re.match(r"^[a-zA-Z0-9_\u4e00-\u9fff]+$", username):
        return False, "用户名只能包含字母、数字、下划线和中文", None
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return False, "邮箱格式不正确", None
    if len(password) < 6 or len(password) > 128:
        return False, "密码需在 6-128 字符之间", None

    now = time.time()
    with _db_cursor() as cur:
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            return False, "用户名已存在", None
        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            return False, "邮箱已被注册", None
        cur.execute("SELECT id FROM roles WHERE name = ?", (role_name,))
        role_row = cur.fetchone()
        if not role_row:
            return False, f"角色 {role_name} 不存在", None
        password_hash = hash_password(password)
        cur.execute(
            """INSERT INTO users
               (username, email, password_hash, salt, role_id, department_id,
                is_active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (username, email, password_hash, "", role_row[0], department_id, now, now)
        )
        user_id = cur.lastrowid
        write_audit_log(user_id=user_id, username=username, action=AuditAction.USER_CREATE.value,
                        detail=f"创建用户: {username}")
    logger.info(f"User created: {username} (id={user_id})")
    return True, "用户创建成功", user_id


def verify_user(username: str, password: str) -> Optional[dict]:
    with _db_cursor() as cur:
        cur.execute(
            """SELECT u.id, u.username, u.email, u.password_hash, u.salt,
                      u.is_active, u.is_superadmin,
                      r.name as role_name, r.id as role_id,
                      d.name as department_name, d.id as department_id,
                      d.path as department_path,
                      u.created_at, u.last_login_at
               FROM users u
               LEFT JOIN roles r ON u.role_id = r.id
               LEFT JOIN departments d ON u.department_id = d.id
               WHERE u.username = ?""",
            (username,)
        )
        row = cur.fetchone()
    if not row:
        return None
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
    if not row["is_active"]:
        return None
    with _db_cursor() as cur:
        cur.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (time.time(), row["id"]))
        write_audit_log(user_id=row["id"], username=row["username"],
                         action=AuditAction.LOGIN.value, detail="用户登录")
    return {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"],
        "role": row["role_name"] or "student",
        "role_id": row["role_id"],
        "department": row["department_name"] or "",
        "department_id": row["department_id"],
        "department_path": row["department_path"] or "",
        "is_active": bool(row["is_active"]),
        "is_superadmin": bool(row["is_superadmin"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def get_user_permissions(role_id: int) -> List[str]:
    with _db_cursor() as cur:
        cur.execute(
            "SELECT permission FROM role_permissions WHERE role_id = ?", (role_id,)
        )
        return [r[0] for r in cur.fetchall()]


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _db_cursor() as cur:
        cur.execute(
            """SELECT u.id, u.username, u.email, u.is_active, u.is_superadmin,
                      r.name as role_name, r.id as role_id,
                      d.name as department_name, d.id as department_id,
                      d.path as department_path,
                      u.created_at, u.last_login_at
               FROM users u
               LEFT JOIN roles r ON u.role_id = r.id
               LEFT JOIN departments d ON u.department_id = d.id
               WHERE u.id = ?""",
            (user_id,)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"],
        "role": row["role_name"] or "student",
        "role_id": row["role_id"],
        "department": row["department_name"] or "",
        "department_id": row["department_id"],
        "department_path": row["department_path"] or "",
        "is_active": bool(row["is_active"]),
        "is_superadmin": bool(row["is_superadmin"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def get_user_permissions_by_username(username: str) -> List[str]:
    with _db_cursor() as cur:
        cur.execute(
            """SELECT rp.permission FROM users u
               JOIN role_permissions rp ON u.role_id = rp.role_id
               WHERE u.username = ? AND u.is_active = 1""",
            (username,)
        )
        return [r[0] for r in cur.fetchall()]


def list_users(
    page: int = 1,
    page_size: int = 20,
    role_filter: Optional[str] = None,
    keyword: Optional[str] = None,
) -> tuple[List[dict], int]:
    conditions = []
    params = []
    if role_filter:
        conditions.append("r.name = ?")
        params.append(role_filter)
    if keyword:
        conditions.append("(u.username LIKE ? OR u.email LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    with _db_cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM users u LEFT JOIN roles r ON u.role_id = r.id WHERE {where_clause}",
            params
        )
        total = cur.fetchone()[0]
        cur.execute(
            f"""SELECT u.id, u.username, u.email, u.is_active, u.is_superadmin,
                      r.name as role_name, d.name as department_name,
                      u.created_at, u.last_login_at
               FROM users u
               LEFT JOIN roles r ON u.role_id = r.id
               LEFT JOIN departments d ON u.department_id = d.id
               WHERE {where_clause}
               ORDER BY u.created_at DESC LIMIT ? OFFSET ?""",
            params + [page_size, (page - 1) * page_size]
        )
        rows = cur.fetchall()
    users = [{
        "id": r["id"],
        "username": r["username"],
        "email": r["email"],
        "role": r["role_name"] or "student",
        "department": r["department_name"] or "",
        "is_active": bool(r["is_active"]),
        "is_superadmin": bool(r["is_superadmin"]),
        "created_at": r["created_at"],
        "last_login_at": r["last_login_at"],
    } for r in rows]
    return users, total


def update_user(
    user_id: int,
    email: Optional[str] = None,
    role_name: Optional[str] = None,
    department_id: Optional[int] = None,
    is_active: Optional[bool] = None,
) -> tuple[bool, str]:
    now = time.time()
    fields = []
    params = []
    if email is not None:
        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
            return False, "邮箱格式不正确"
        fields.append("email = ?")
        params.append(email.strip().lower())
    if role_name is not None:
        with _db_cursor() as cur:
            cur.execute("SELECT id FROM roles WHERE name = ?", (role_name,))
            row = cur.fetchone()
        if not row:
            return False, f"角色 {role_name} 不存在"
        fields.append("role_id = ?")
        params.append(row[0])
    if department_id is not None:
        fields.append("department_id = ?")
        params.append(department_id)
    if is_active is not None:
        fields.append("is_active = ?")
        params.append(1 if is_active else 0)
    if not fields:
        return False, "没有需要更新的字段"
    fields.append("updated_at = ?")
    params.append(now)
    params.append(user_id)
    with _db_cursor() as cur:
        cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", params)
        write_audit_log(user_id=user_id, username="", action=AuditAction.USER_UPDATE.value,
                        detail=f"更新用户 id={user_id}")
    return True, "更新成功"


def delete_user(user_id: int) -> tuple[bool, str]:
    with _db_cursor() as cur:
        cur.execute("SELECT username, is_superadmin FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return False, "用户不存在"
        if row["is_superadmin"]:
            return False, "无法删除超级管理员"
        cur.execute("UPDATE users SET is_active = 0, updated_at = ? WHERE id = ?",
                    (time.time(), user_id))
        write_audit_log(user_id=user_id, username=row["username"],
                        action=AuditAction.USER_DELETE.value, detail=f"删除用户 id={user_id}")
    return True, "删除成功"


def change_password(user_id: int, old_password: str, new_password: str) -> tuple[bool, str]:
    new_password = new_password.strip()
    if len(new_password) < 6 or len(new_password) > 128:
        return False, "新密码需在 6-128 字符之间"
    with _db_cursor() as cur:
        cur.execute("SELECT password_hash, salt, username FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
    if not row:
        return False, "用户不存在"
    valid, _ = verify_password_and_upgrade(
        old_password, row["password_hash"], row["salt"]
    )
    if not valid:
        return False, "原密码错误"
    new_hash = hash_password(new_password)
    with _db_cursor() as cur:
        cur.execute("UPDATE users SET password_hash = ?, salt = ?, updated_at = ? WHERE id = ?",
                    (new_hash, "", time.time(), user_id))
    logger.info(f"Password changed: {row['username']}")
    return True, "密码修改成功"


# ==================== 部门操作 ====================

def create_department(
    name: str,
    parent_id: Optional[int] = None,
    description: str = "",
) -> tuple[bool, str, Optional[int]]:
    name = name.strip()
    if not name:
        return False, "部门名称不能为空", None
    now = time.time()
    path = f"/{name}"
    if parent_id:
        with _db_cursor() as cur:
            cur.execute("SELECT name, path FROM departments WHERE id = ?", (parent_id,))
            row = cur.fetchone()
        if not row:
            return False, "父部门不存在", None
        path = f"{row['path']}/{name}"
    with _db_cursor() as cur:
        cur.execute("SELECT id FROM departments WHERE name = ? AND parent_id IS ?",
                    (name, parent_id))
        if cur.fetchone():
            return False, "同级部门名称已存在", None
        cur.execute(
            """INSERT INTO departments (name, parent_id, path, description, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, parent_id, path, description, now, now)
        )
        dept_id = cur.lastrowid
    return True, "部门创建成功", dept_id


def list_departments() -> List[dict]:
    with _db_cursor() as cur:
        cur.execute(
            """SELECT id, name, parent_id, path, description, created_at
               FROM departments ORDER BY path"""
        )
        rows = cur.fetchall()
    return [{
        "id": r["id"],
        "name": r["name"],
        "parent_id": r["parent_id"],
        "path": r["path"],
        "description": r["description"],
        "created_at": r["created_at"],
    } for r in rows]


# ==================== 角色操作 ====================

def list_roles() -> List[dict]:
    with _db_cursor() as cur:
        cur.execute(
            """SELECT r.id, r.name, r.description, r.is_system, r.created_at,
                      GROUP_CONCAT(rp.permission) as permissions
               FROM roles r
               LEFT JOIN role_permissions rp ON r.id = rp.role_id
               GROUP BY r.id ORDER BY r.id"""
        )
        rows = cur.fetchall()
    return [{
        "id": r["id"],
        "name": r["name"],
        "description": r["description"],
        "is_system": bool(r["is_system"]),
        "permissions": r["permissions"].split(",") if r["permissions"] else [],
        "created_at": r["created_at"],
    } for r in rows]


# ==================== 文档元数据操作 ====================

def create_document_meta(
    doc_id: str,
    title: str,
    category: str,
    uploaded_by: str,
    version: str = "1.0",
    effective_date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    confidentiality: str = "internal",
    department_restrict: Optional[List[str]] = None,
    role_restrict: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    description: str = "",
    file_path: str = "",
    file_hash: str = "",
    file_size: int = 0,
) -> tuple[bool, str]:
    now = time.time()
    with _db_cursor() as cur:
        cur.execute(
            """INSERT INTO documents
               (id, title, category, tags, version, effective_date, expiry_date,
                confidentiality, department_restrict, role_restrict, status,
                file_path, file_hash, file_size, uploaded_by, description,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, title, category, json.dumps(tags or []), version,
             effective_date, expiry_date, confidentiality,
             json.dumps(department_restrict or []),
             json.dumps(role_restrict or []),
             DocStatus.DRAFT.value, file_path, file_hash, file_size,
             uploaded_by, description, now, now)
        )
        write_audit_log(user_id=None, username=uploaded_by, action=AuditAction.DOC_UPLOAD.value,
                        resource_type="document", resource_id=doc_id, resource_name=title,
                        detail=f"上传文档: {title} (v{version})")
    return True, "文档创建成功"


def update_document_meta(
    doc_id: str,
    title: Optional[str] = None,
    category: Optional[str] = None,
    version: Optional[str] = None,
    effective_date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    confidentiality: Optional[str] = None,
    department_restrict: Optional[List[str]] = None,
    role_restrict: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    chunk_count: Optional[int] = None,
) -> tuple[bool, str]:
    fields = []
    params = []
    def add_field(name, value):
        if value is not None:
            fields.append(f"{name} = ?")
            params.append(value)
    add_field("title", title)
    add_field("category", category)
    add_field("version", version)
    add_field("effective_date", effective_date)
    add_field("expiry_date", expiry_date)
    add_field("confidentiality", confidentiality)
    add_field("department_restrict", json.dumps(department_restrict or []))
    add_field("role_restrict", json.dumps(role_restrict or []))
    add_field("tags", json.dumps(tags or []))
    add_field("description", description)
    add_field("status", status)
    add_field("chunk_count", chunk_count)
    if not fields:
        return False, "没有需要更新的字段"
    fields.append("updated_at = ?")
    params.append(time.time())
    params.append(doc_id)
    with _db_cursor() as cur:
        cur.execute(f"UPDATE documents SET {', '.join(fields)} WHERE id = ?", params)
    return True, "更新成功"


def get_document(doc_id: str) -> Optional[dict]:
    with _db_cursor() as cur:
        cur.execute(
            """SELECT id, title, category, tags, version, effective_date, expiry_date,
                      confidentiality, department_restrict, role_restrict, status,
                      file_path, file_hash, file_size, uploaded_by, chunk_count,
                      description, created_at, updated_at
               FROM documents WHERE id = ?""",
            (doc_id,)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "title": row["title"],
        "category": row["category"],
        "tags": json.loads(row["tags"]) if row["tags"] else [],
        "version": row["version"],
        "effective_date": row["effective_date"],
        "expiry_date": row["expiry_date"],
        "confidentiality": row["confidentiality"],
        "department_restrict": json.loads(row["department_restrict"]) if row["department_restrict"] else [],
        "role_restrict": json.loads(row["role_restrict"]) if row["role_restrict"] else [],
        "status": row["status"],
        "file_path": row["file_path"],
        "file_hash": row["file_hash"],
        "file_size": row["file_size"],
        "uploaded_by": row["uploaded_by"],
        "chunk_count": row["chunk_count"],
        "description": row["description"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_documents(
    page: int = 1,
    page_size: int = 20,
    category: Optional[str] = None,
    status: Optional[str] = None,
    keyword: Optional[str] = None,
    uploaded_by: Optional[str] = None,
) -> tuple[List[dict], int]:
    conditions = []
    params = []
    if category:
        conditions.append("category = ?")
        params.append(category)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if keyword:
        conditions.append("(title LIKE ? OR description LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    if uploaded_by:
        conditions.append("uploaded_by = ?")
        params.append(uploaded_by)
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    with _db_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM documents WHERE {where_clause}", params)
        total = cur.fetchone()[0]
        cur.execute(
            f"""SELECT id, title, category, tags, version, effective_date,
                      confidentiality, status, file_size, uploaded_by,
                      chunk_count, created_at, updated_at
               FROM documents
               WHERE {where_clause}
               ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
            params + [page_size, (page - 1) * page_size]
        )
        rows = cur.fetchall()
    docs = [{
        "id": r["id"],
        "title": r["title"],
        "category": r["category"],
        "tags": json.loads(r["tags"]) if r["tags"] else [],
        "version": r["version"],
        "effective_date": r["effective_date"],
        "confidentiality": r["confidentiality"],
        "status": r["status"],
        "file_size": r["file_size"],
        "uploaded_by": r["uploaded_by"],
        "chunk_count": r["chunk_count"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    } for r in rows]
    return docs, total


def delete_document(doc_id: str, username: str) -> tuple[bool, str]:
    with _db_cursor() as cur:
        cur.execute("SELECT title FROM documents WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        if not row:
            return False, "文档不存在"
        cur.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
            (DocStatus.ARCHIVED.value, time.time(), doc_id)
        )
        write_audit_log(user_id=None, username=username, action=AuditAction.DOC_DELETE.value,
                        resource_type="document", resource_id=doc_id, resource_name=row["title"],
                        detail=f"删除文档: {row['title']}")
    return True, "删除成功"


def get_document_stats() -> dict:
    with _db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents WHERE status != ?", (DocStatus.ARCHIVED.value,))
        total_docs = cur.fetchone()[0]
        cur.execute(
            "SELECT category, COUNT(*) as cnt FROM documents WHERE status != ? GROUP BY category",
            (DocStatus.ARCHIVED.value,)
        )
        by_category = {r["category"]: r["cnt"] for r in cur.fetchall()}
        cur.execute("SELECT SUM(chunk_count) FROM documents WHERE status != ?",
                    (DocStatus.ARCHIVED.value,))
        total_chunks = cur.fetchone()[0][0] or 0
        cur.execute("SELECT SUM(file_size) FROM documents WHERE status != ?",
                    (DocStatus.ARCHIVED.value,))
        total_size = cur.fetchone()[0][0] or 0
    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_size_bytes": total_size,
        "by_category": by_category,
    }


# ==================== 审计日志查询 ====================

def list_audit_logs(
    page: int = 1,
    page_size: int = 50,
    action: Optional[str] = None,
    username: Optional[str] = None,
    resource_type: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> tuple[List[dict], int]:
    conditions = []
    params = []
    if action:
        conditions.append("action = ?")
        params.append(action)
    if username:
        conditions.append("username = ?")
        params.append(username)
    if resource_type:
        conditions.append("resource_type = ?")
        params.append(resource_type)
    if start_time:
        conditions.append("created_at >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("created_at <= ?")
        params.append(end_time)
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    with _db_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM audit_logs WHERE {where_clause}", params)
        total = cur.fetchone()[0]
        cur.execute(
            f"""SELECT id, user_id, username, action, resource_type, resource_id,
                      resource_name, ip_address, detail, created_at
               FROM audit_logs
               WHERE {where_clause}
               ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            params + [page_size, (page - 1) * page_size]
        )
        rows = cur.fetchall()
    logs = [{
        "id": r["id"],
        "user_id": r["user_id"],
        "username": r["username"],
        "action": r["action"],
        "resource_type": r["resource_type"],
        "resource_id": r["resource_id"],
        "resource_name": r["resource_name"],
        "ip_address": r["ip_address"],
        "detail": r["detail"],
        "created_at": r["created_at"],
    } for r in rows]
    return logs, total


def health_check() -> dict:
    try:
        with _db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users")
            cur.execute("SELECT COUNT(*) FROM documents")
            cur.execute("SELECT COUNT(*) FROM departments")
            cur.execute("SELECT COUNT(*) FROM audit_logs")
        return {"status": "healthy", "db": "ok"}
    except Exception as e:
        return {"status": "unhealthy", "db": str(e)}


try:
    init_database()
except Exception:
    logger.warning("Database init failed (tables may already exist)")
