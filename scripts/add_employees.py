"""批量创建员工脚本 - 直接操作SQLite数据库（写入 users.db）"""
import sqlite3
import hashlib
import secrets
import time
from pathlib import Path

# users.db 路径
DATA_DIR = Path("/share/home/lifr/workspace/code/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "users.db"
SALT_LENGTH = 32

# 定义要创建的员工列表： (用户名, 邮箱, 密码, 角色)
EMPLOYEES = [
    ("alice", "alice@company.com", "pass123", "admin"),
    ("bob", "bob@company.com", "pass123", "manager"),
    ("carol", "carol@company.com", "pass123", "hr"),
    ("david", "david@company.com", "pass123", "it_support"),
    ("eve", "eve@company.com", "pass123", "employee"),
    ("frank", "frank@company.com", "pass123", "employee"),
    ("grace", "grace@company.com", "pass123", "employee"),
    ("henry", "henry@company.com", "pass123", "employee"),
    ("iris", "iris@company.com", "pass123", "employee"),
    ("jack", "jack@company.com", "pass123", "employee"),
]

ROLE_NAMES_CN = {
    "admin": "管理员",
    "manager": "部门经理",
    "hr": "HR专员",
    "it_support": "IT支持",
    "employee": "普通员工",
}


def _hash_password(password: str, salt: str) -> str:
    combined = salt + password + salt
    for _ in range(3):
        combined = hashlib.sha256(combined.encode()).hexdigest()
    return combined


def _generate_salt() -> str:
    return secrets.token_hex(SALT_LENGTH)


def _get_connection():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = _get_connection()
    cur = conn.cursor()
    now = time.time()

    # 用户表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)

    # RBAC 角色表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at REAL NOT NULL
        )
    """)

    # 用户角色关联表
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

    # 角色权限表
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
    role_map = {}
    for name, description in [
        ("admin", "管理员 - 全部权限"),
        ("editor", "编辑 - 可读写知识库"),
        ("viewer", "访客 - 仅可读"),
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
        "admin": [
            ("knowledge", "read"), ("knowledge", "write"), ("knowledge", "delete"), ("knowledge", "admin"),
            ("chat", "read"), ("chat", "write"), ("chat", "admin"),
            ("session", "read"), ("session", "write"), ("session", "delete"), ("session", "admin"),
            ("user", "read"), ("user", "write"), ("user", "delete"), ("user", "admin"),
            ("system", "admin"),
        ],
        "editor": [
            ("knowledge", "read"), ("knowledge", "write"),
            ("chat", "read"), ("chat", "write"),
            ("session", "read"), ("session", "write"),
        ],
        "viewer": [
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

    conn.commit()
    conn.close()
    print(f"数据库初始化完成: {DB_PATH}")
    return role_map


def get_role_map():
    """获取角色ID映射"""
    conn = _get_connection()
    cur = conn.cursor()
    role_map = {}
    for name in ["admin", "editor", "viewer"]:
        cur.execute("SELECT id FROM roles WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            role_map[name] = row["id"]
    conn.close()
    return role_map


def create_user(username, email, password, role):
    """创建单个用户（security_user 体系）"""
    # 映射 security_user 角色
    role_map = {
        "admin": "admin",
        "manager": "editor",
        "hr": "editor",
        "it_support": "editor",
        "employee": "viewer",
    }
    internal_role = role_map.get(role, "viewer")

    conn = _get_connection()
    cur = conn.cursor()

    # 检查用户是否已存在
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cur.fetchone():
        conn.close()
        return False, "用户名已存在", None

    # 创建用户
    now = time.time()
    salt = _generate_salt()
    password_hash = _hash_password(password, salt)

    cur.execute(
        "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
        (username, password_hash, salt, now)
    )
    user_id = cur.lastrowid

    # 分配角色
    role_id = None
    role_map_all = get_role_map()
    role_id = role_map_all.get(internal_role)

    if role_id:
        cur.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id, assigned_at) VALUES (?, ?, ?)",
            (user_id, role_id, now)
        )

    conn.commit()
    conn.close()
    return True, "创建成功", user_id


def main():
    print("=" * 60)
    print("批量创建员工（写入 users.db）")
    print("=" * 60)

    role_map = init_db()
    success_list = []
    fail_list = []

    for username, email, password, role in EMPLOYEES:
        success, msg, user_id = create_user(username, email, password, role)
        if success:
            success_list.append((username, password, role))
            print(f"[OK] {username} ({ROLE_NAMES_CN.get(role, role)}) - ID: {user_id}")
        else:
            fail_list.append((username, role, msg))
            print(f"[SKIP] {username}: {msg}")

    print()
    print("=" * 60)
    print("创建结果汇总")
    print("=" * 60)
    print(f"成功: {len(success_list)} 人")
    print(f"跳过(已存在): {len(fail_list)} 人")

    if success_list:
        print()
        print("-" * 60)
        print("员工账号清单")
        print("-" * 60)
        print(f"{'用户名':<12} {'密码':<10} {'角色':<12} {'角色说明'}")
        print("-" * 60)
        for username, password, role in success_list:
            print(f"{username:<12} {password:<10} {role:<12} {ROLE_NAMES_CN.get(role, role)}")
        print("-" * 60)

    if fail_list:
        print()
        print("-" * 60)
        print("已存在的账号 (跳过)")
        print("-" * 60)
        for username, role, msg in fail_list:
            print(f"  {username} ({ROLE_NAMES_CN.get(role, role)})")


if __name__ == "__main__":
    main()
