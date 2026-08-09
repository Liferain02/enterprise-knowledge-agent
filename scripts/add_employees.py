"""批量创建实验室成员脚本 - 直接操作 SQLite 数据库（写入 users.db）"""
import getpass
import sqlite3
import sys
import time
from pathlib import Path

from pwdlib import PasswordHash

# users.db 路径
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "users.db"
PASSWORD_HASH = PasswordHash.recommended()

# 定义要创建的实验室成员列表： (用户名, 邮箱, 角色)
LAB_MEMBERS = [
    ("prof_chen", "prof_chen@lab.local", "admin"),
    ("wang", "wang@lab.local", "pi"),
    ("liu", "liu@lab.local", "teacher"),
    ("zhao", "zhao@lab.local", "lab_admin"),
    ("sun", "sun@lab.local", "senior_student"),
    ("lin", "lin@lab.local", "student"),
    ("guo", "guo@lab.local", "student"),
    ("he", "he@lab.local", "assistant"),
]

MIN_BOOTSTRAP_PASSWORD_LENGTH = 12
COMMON_PASSWORDS = {
    "12345678",
    "admin123",
    "changeme",
    "password",
    "password123",
}
ROLE_NAMES_CN = {
    "admin": "管理员",
    "pi": "导师/PI",
    "teacher": "教师",
    "lab_admin": "实验室管理员",
    "senior_student": "高年级成员",
    "student": "研究生",
    "assistant": "助研/本科生",
}


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


def user_exists(username: str) -> bool:
    """检查账号是否存在，确保幂等重跑不会再次询问密码。"""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def prompt_password(username: str, used_passwords: set[str]) -> str:
    """在交互终端隐藏读取并确认一个未与本批账号共享的密码。"""
    if not sys.stdin.isatty():
        raise RuntimeError("成员初始化必须在交互终端运行，密码不会从命令行读取")

    while True:
        password = getpass.getpass(f"为 {username} 设置密码（至少 12 字符）: ")
        if len(password) < MIN_BOOTSTRAP_PASSWORD_LENGTH:
            print(f"[RETRY] {username}: 密码至少需要 12 个字符")
            continue
        if password.lower() in COMMON_PASSWORDS:
            print(f"[RETRY] {username}: 请勿使用常见弱口令")
            continue
        if password in used_passwords:
            print(f"[RETRY] {username}: 每个账号必须使用不同密码")
            continue

        confirmation = getpass.getpass(f"再次输入 {username} 的密码: ")
        if password != confirmation:
            print(f"[RETRY] {username}: 两次输入不一致")
            continue
        return password


def create_user(username, email, password, role):
    """创建单个用户（security_user 体系）"""
    # 映射 security_user 角色
    role_map = {
        "admin": "admin",
        "pi": "editor",
        "teacher": "editor",
        "lab_admin": "editor",
        "senior_student": "editor",
        "student": "viewer",
        "assistant": "viewer",
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
    password_hash = PASSWORD_HASH.hash(password)

    cur.execute(
        "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
        (username, password_hash, "", now)
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


def main() -> int:
    print("=" * 60)
    print("批量创建实验室成员（写入 users.db）")
    print("=" * 60)

    init_db()
    success_list = []
    fail_list = []

    pending_members = []
    for username, email, role in LAB_MEMBERS:
        if user_exists(username):
            fail_list.append((username, role, "用户名已存在"))
        else:
            pending_members.append((username, email, role))

    passwords = {}
    used_passwords: set[str] = set()
    try:
        for username, _, _ in pending_members:
            password = prompt_password(username, used_passwords)
            passwords[username] = password
            used_passwords.add(password)
    except (EOFError, KeyboardInterrupt, RuntimeError) as exc:
        print(f"\n[ERROR] 初始化已取消，尚未创建任何新账号：{exc}")
        return 1

    for username, email, role in pending_members:
        password = passwords.pop(username)
        success, msg, user_id = create_user(username, email, password, role)
        if success:
            success_list.append((username, role))
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
        print("实验室成员账号清单")
        print("-" * 60)
        print(f"{'用户名':<12} {'角色':<12} {'角色说明'}")
        print("-" * 60)
        for username, role in success_list:
            print(f"{username:<12} {role:<12} {ROLE_NAMES_CN.get(role, role)}")
        print("-" * 60)

    if fail_list:
        print()
        print("-" * 60)
        print("已存在的账号 (跳过)")
        print("-" * 60)
        for username, role, msg in fail_list:
            print(f"  {username} ({ROLE_NAMES_CN.get(role, role)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
