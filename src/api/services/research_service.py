"""结构化科研工作流：项目空间、成员与实验记录。"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_DB_PATH = DATA_DIR / "research_workspace.db"
_PRIVILEGED_ROLES = {"admin", "pi"}
_PROJECT_EDITOR_ROLES = {
    "admin", "pi", "teacher", "lab_admin", "senior_student", "editor", "manager",
}
_VALID_VISIBILITIES = {"public", "project", "restricted"}
_VALID_PROJECT_STATUSES = {"planned", "active", "paused", "completed"}
_VALID_EXPERIMENT_STATUSES = {"planned", "running", "completed", "failed"}
_VALID_TASK_STATUSES = {"open", "in_progress", "done"}


class ResearchService:
    """持久化实验室科研工作流，并在服务端统一执行 ACL。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_projects (
                    id TEXT PRIMARY KEY,
                    slug TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    research_direction TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    visibility TEXT NOT NULL DEFAULT 'project',
                    lead TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_project_members (
                    project_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
                    username TEXT NOT NULL,
                    member_role TEXT NOT NULL DEFAULT 'member',
                    created_at REAL NOT NULL,
                    PRIMARY KEY (project_id, username)
                );
                CREATE TABLE IF NOT EXISTS research_experiments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    hypothesis TEXT NOT NULL DEFAULT '',
                    environment TEXT NOT NULL DEFAULT '',
                    code_commit TEXT NOT NULL DEFAULT '',
                    dataset_version TEXT NOT NULL DEFAULT '',
                    metrics TEXT NOT NULL DEFAULT '{}',
                    conclusion TEXT NOT NULL DEFAULT '',
                    next_steps TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'planned',
                    created_by TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_tasks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    assignee TEXT NOT NULL DEFAULT '',
                    due_date TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'open',
                    source TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_research_projects_status
                    ON research_projects(status);
                CREATE INDEX IF NOT EXISTS idx_research_project_members_username
                    ON research_project_members(username);
                CREATE INDEX IF NOT EXISTS idx_research_experiments_project
                    ON research_experiments(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_research_tasks_project
                    ON research_tasks(project_id, status, updated_at DESC);
                """
            )

    @staticmethod
    def _identity(user: dict) -> tuple[str, str]:
        return user.get("username", "anonymous"), user.get("role", "student")

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", value.strip()).strip("-").lower()
        return slug[:64] or f"project-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _validate_choice(value: str, choices: set[str], field: str) -> str:
        if value not in choices:
            raise ValueError(f"{field} 不合法")
        return value

    @staticmethod
    def _project_accessible(project: dict, user: dict, member_names: set[str]) -> bool:
        username, role = ResearchService._identity(user)
        if role in _PRIVILEGED_ROLES:
            return True
        if project["visibility"] == "public":
            return True
        if username == project["created_by"] or username == project["lead"]:
            return True
        if username in member_names:
            return True
        return False

    @staticmethod
    def _can_create_project(user: dict) -> bool:
        return user.get("role", "student") in _PROJECT_EDITOR_ROLES

    @staticmethod
    def _can_write_project(project: dict, user: dict, member_names: set[str]) -> bool:
        username, role = ResearchService._identity(user)
        return (
            role in _PRIVILEGED_ROLES
            or username == project["created_by"]
            or username == project["lead"]
            or username in member_names
        )

    def _members(self, conn: sqlite3.Connection, project_id: str) -> list[dict]:
        rows = conn.execute(
            """SELECT username, member_role, created_at
               FROM research_project_members WHERE project_id = ?
               ORDER BY member_role DESC, username""",
            (project_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _serialize_project(self, conn: sqlite3.Connection, row: sqlite3.Row | dict) -> dict:
        project = dict(row)
        members = self._members(conn, project["id"])
        experiment_count = conn.execute(
            "SELECT COUNT(*) FROM research_experiments WHERE project_id = ?",
            (project["id"],),
        ).fetchone()[0]
        task_count = conn.execute(
            "SELECT COUNT(*) FROM research_tasks WHERE project_id = ? AND status != 'done'",
            (project["id"],),
        ).fetchone()[0]
        project["members"] = members
        project["experiment_count"] = experiment_count
        project["open_task_count"] = task_count
        return project

    def create_project(self, payload: dict, user: dict) -> dict:
        if not self._can_create_project(user):
            raise PermissionError("当前角色无权创建项目空间")
        title = str(payload.get("title", "")).strip()
        if not title:
            raise ValueError("项目名称不能为空")
        visibility = self._validate_choice(
            str(payload.get("visibility", "project")), _VALID_VISIBILITIES, "可见范围"
        )
        status = self._validate_choice(
            str(payload.get("status", "active")), _VALID_PROJECT_STATUSES, "项目状态"
        )
        username, _ = self._identity(user)
        project_id = uuid.uuid4().hex
        now = time.time()
        lead = str(payload.get("lead") or username).strip()
        members = {str(item).strip() for item in payload.get("members", []) if str(item).strip()}
        members.update({username, lead})
        base_slug = self._slugify(str(payload.get("slug") or title))
        with self._connection() as conn:
            slug = base_slug
            index = 2
            while conn.execute("SELECT 1 FROM research_projects WHERE slug = ?", (slug,)).fetchone():
                slug = f"{base_slug}-{index}"
                index += 1
            conn.execute(
                """INSERT INTO research_projects
                   (id, slug, title, summary, research_direction, status, visibility,
                    lead, created_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id, slug, title, str(payload.get("summary", "")).strip(),
                    str(payload.get("research_direction", "")).strip(), status, visibility,
                    lead, username, now, now,
                ),
            )
            for member in members:
                conn.execute(
                    """INSERT INTO research_project_members
                       (project_id, username, member_role, created_at) VALUES (?, ?, ?, ?)""",
                    (project_id, member, "lead" if member == lead else "member", now),
                )
            row = conn.execute("SELECT * FROM research_projects WHERE id = ?", (project_id,)).fetchone()
            return self._serialize_project(conn, row)

    def list_projects(self, user: dict, query: str = "") -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM research_projects
                   WHERE (? = '' OR title LIKE ? OR summary LIKE ? OR research_direction LIKE ?)
                   ORDER BY updated_at DESC""",
                (query, f"%{query}%", f"%{query}%", f"%{query}%"),
            ).fetchall()
            projects = [self._serialize_project(conn, row) for row in rows]
        return [
            project for project in projects
            if self._project_accessible(project, user, {item["username"] for item in project["members"]})
        ]

    def get_project(self, project_id: str, user: dict) -> dict:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM research_projects WHERE id = ?", (project_id,)).fetchone()
            if not row:
                raise ValueError("项目不存在")
            project = self._serialize_project(conn, row)
        members = {item["username"] for item in project["members"]}
        if not self._project_accessible(project, user, members):
            raise PermissionError("无权访问该项目")
        return project

    def create_experiment(self, project_id: str, payload: dict, user: dict) -> dict:
        title = str(payload.get("title", "")).strip()
        if not title:
            raise ValueError("实验标题不能为空")
        status = self._validate_choice(
            str(payload.get("status", "planned")), _VALID_EXPERIMENT_STATUSES, "实验状态"
        )
        project = self.get_project(project_id, user)
        members = {item["username"] for item in project["members"]}
        if not self._can_write_project(project, user, members):
            raise PermissionError("无权为该项目添加实验记录")
        experiment_id = uuid.uuid4().hex
        username, _ = self._identity(user)
        now = time.time()
        metrics = payload.get("metrics") or {}
        if not isinstance(metrics, dict):
            raise ValueError("实验指标必须是键值对象")
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO research_experiments
                   (id, project_id, title, hypothesis, environment, code_commit,
                    dataset_version, metrics, conclusion, next_steps, status,
                    created_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id, project_id, title, str(payload.get("hypothesis", "")).strip(),
                    str(payload.get("environment", "")).strip(),
                    str(payload.get("code_commit", "")).strip(),
                    str(payload.get("dataset_version", "")).strip(),
                    json.dumps(metrics, ensure_ascii=False),
                    str(payload.get("conclusion", "")).strip(),
                    str(payload.get("next_steps", "")).strip(),
                    status, username, now, now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM research_experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            return self._serialize_experiment(row)

    @staticmethod
    def _serialize_experiment(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["metrics"] = json.loads(item["metrics"] or "{}")
        return item

    def list_experiments(self, project_id: str, user: dict) -> list[dict]:
        self.get_project(project_id, user)
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM research_experiments WHERE project_id = ?
                   ORDER BY updated_at DESC""",
                (project_id,),
            ).fetchall()
        return [self._serialize_experiment(row) for row in rows]

    def create_task(self, project_id: str, payload: dict, user: dict) -> dict:
        project = self.get_project(project_id, user)
        members = {item["username"] for item in project["members"]}
        if not self._can_write_project(project, user, members):
            raise PermissionError("无权为该项目添加待办")
        title = str(payload.get("title", "")).strip()
        if not title:
            raise ValueError("待办标题不能为空")
        status = self._validate_choice(
            str(payload.get("status", "open")), _VALID_TASK_STATUSES, "待办状态"
        )
        username, _ = self._identity(user)
        task_id = uuid.uuid4().hex
        now = time.time()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO research_tasks
                   (id, project_id, title, assignee, due_date, status, source,
                    created_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id, project_id, title, str(payload.get("assignee", "")).strip(),
                    str(payload.get("due_date", "")).strip(), status,
                    str(payload.get("source", "")).strip(), username, now, now,
                ),
            )
            row = conn.execute("SELECT * FROM research_tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row)

    def list_tasks(self, project_id: str, user: dict) -> list[dict]:
        self.get_project(project_id, user)
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM research_tasks WHERE project_id = ?
                   ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'in_progress' THEN 1 ELSE 2 END,
                            updated_at DESC""",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_task_status(self, task_id: str, status: str, user: dict) -> dict:
        self._validate_choice(status, _VALID_TASK_STATUSES, "待办状态")
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM research_tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise ValueError("待办不存在")
        project = self.get_project(row["project_id"], user)
        members = {item["username"] for item in project["members"]}
        if not self._can_write_project(project, user, members):
            raise PermissionError("无权更新该项目待办")
        with self._connection() as conn:
            conn.execute(
                "UPDATE research_tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), task_id),
            )
            updated = conn.execute("SELECT * FROM research_tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(updated)

    def extract_meeting_tasks(self, project_id: str, content: str, source: str, user: dict) -> list[dict]:
        """从常见组会纪要格式提取行动项并持久化。"""
        candidates = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            is_checkbox = bool(re.match(r"^[-*]\s*\[\s*[ xX]?\s*\]", line))
            is_action = bool(re.search(r"\bTODO\b|待办|行动项|后续动作", line, re.IGNORECASE))
            if not (is_checkbox or is_action):
                continue
            cleaned = re.sub(r"^[-*]\s*\[\s*[ xX]?\s*\]\s*", "", line)
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            parts = [part.strip() for part in re.split(r"[|｜]", cleaned) if part.strip()]
            title = parts[0]
            title = re.sub(r"^(TODO|待办|行动项|后续动作)\s*[:：]?\s*", "", title, flags=re.IGNORECASE)
            assignee = ""
            due_date = ""
            for part in parts[1:]:
                owner_match = re.search(r"(?:负责人|owner|assignee)\s*[:：]\s*(.+)", part, re.IGNORECASE)
                due_match = re.search(r"(?:截止(?:日期)?|due)\s*[:：]\s*(\d{4}-\d{2}-\d{2})", part, re.IGNORECASE)
                if owner_match:
                    assignee = owner_match.group(1).strip()
                if due_match:
                    due_date = due_match.group(1)
            if title:
                candidates.append(
                    self.create_task(
                        project_id,
                        {"title": title, "assignee": assignee, "due_date": due_date, "source": source},
                        user,
                    )
                )
        return candidates

    def get_overview(self, user: dict) -> dict:
        projects = self.list_projects(user)
        status_counts: dict[str, int] = {}
        experiments = 0
        open_tasks = 0
        members: set[str] = set()
        for project in projects:
            status_counts[project["status"]] = status_counts.get(project["status"], 0) + 1
            experiments += project["experiment_count"]
            open_tasks += project["open_task_count"]
            members.update(item["username"] for item in project["members"])
        return {
            "projects": len(projects),
            "experiments": experiments,
            "open_tasks": open_tasks,
            "members": len(members),
            "active_projects": status_counts.get("active", 0),
            "by_status": status_counts,
        }

    def seed_lab_samples(self, user: dict) -> dict:
        """幂等写入实验室方向样例，便于首次使用和演示。"""
        if not self._can_create_project(user):
            raise PermissionError("当前角色无权初始化科研样例")
        samples = [
            {
                "project": {
                    "title": "Distributed NUMA over RDMA",
                    "slug": "distributed-numa-rdma",
                    "summary": "面向跨节点内存访问，探索 NUMA 感知的数据布局、远端访问路径与 RDMA 传输优化。",
                    "research_direction": "分布式 NUMA / 高性能网络",
                    "visibility": "project",
                    "lead": user.get("username", "admin"),
                },
                "experiment": {
                    "title": "双节点远端内存访问基线",
                    "hypothesis": "在固定消息大小下，RDMA Read 的尾延迟将成为跨节点 NUMA 访问的首要瓶颈。",
                    "environment": "2 nodes; RDMA NIC; NUMA binding enabled; microbenchmark workload",
                    "code_commit": "baseline",
                    "dataset_version": "rdma-numa-microbench-v1",
                    "metrics": {"status": "待补充实测值"},
                    "next_steps": "补齐节点型号、NIC 型号、消息大小矩阵与 p50/p99 延迟。",
                    "status": "planned",
                },
                "task": {
                    "title": "补齐 RDMA Read 消息大小与并发度矩阵",
                    "assignee": user.get("username", "admin"),
                    "source": "Distributed NUMA 项目启动会",
                },
            },
            {
                "project": {
                    "title": "RDMA Network Benchmarking",
                    "slug": "rdma-network-benchmarking",
                    "summary": "沉淀高性能网络测试方法，统一记录带宽、时延、拥塞和 CPU 开销。",
                    "research_direction": "RDMA / 高性能网络",
                    "visibility": "public",
                    "lead": user.get("username", "admin"),
                },
                "experiment": {
                    "title": "RDMA 带宽与尾延迟测试矩阵",
                    "environment": "待填写 NIC、驱动、固件、链路速率与 CPU 亲和性。",
                    "dataset_version": "network-matrix-v1",
                    "metrics": {"status": "待补充实测值"},
                    "next_steps": "按消息大小和并发度补齐带宽、平均延迟与 p99 延迟。",
                    "status": "planned",
                },
                "task": {
                    "title": "登记 NIC、驱动、固件与链路速率",
                    "assignee": user.get("username", "admin"),
                    "source": "RDMA Benchmark 启动会",
                },
            },
        ]
        created_projects = 0
        created_experiments = 0
        created_tasks = 0
        for sample in samples:
            slug = sample["project"]["slug"]
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT * FROM research_projects WHERE slug = ?", (slug,)
                ).fetchone()
                project = self._serialize_project(conn, row) if row else None
            if project is None:
                project = self.create_project(sample["project"], user)
                created_projects += 1
            with self._connection() as conn:
                exists = conn.execute(
                    """SELECT 1 FROM research_experiments
                       WHERE project_id = ? AND title = ?""",
                    (project["id"], sample["experiment"]["title"]),
                ).fetchone()
            if not exists:
                self.create_experiment(project["id"], sample["experiment"], user)
                created_experiments += 1
            with self._connection() as conn:
                task_exists = conn.execute(
                    """SELECT 1 FROM research_tasks
                       WHERE project_id = ? AND title = ?""",
                    (project["id"], sample["task"]["title"]),
                ).fetchone()
            if not task_exists:
                self.create_task(project["id"], sample["task"], user)
                created_tasks += 1
        return {
            "message": "科研空间样例已初始化",
            "created_projects": created_projects,
            "created_experiments": created_experiments,
            "created_tasks": created_tasks,
        }


research_service = ResearchService()
