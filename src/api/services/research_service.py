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
                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT REFERENCES research_projects(id) ON DELETE SET NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'deep',
                    status TEXT NOT NULL DEFAULT 'completed',
                    final_answer TEXT NOT NULL DEFAULT '',
                    source_cards_json TEXT NOT NULL DEFAULT '[]',
                    evidence_package_json TEXT NOT NULL DEFAULT '{}',
                    analysis_report_json TEXT NOT NULL DEFAULT '{}',
                    review_report_json TEXT NOT NULL DEFAULT '{}',
                    trace_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS research_memory_confirmations (
                    run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
                    claim_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    confirmed_at REAL NOT NULL,
                    memory_result_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (run_id, claim_id, user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_research_projects_status
                    ON research_projects(status);
                CREATE INDEX IF NOT EXISTS idx_research_project_members_username
                    ON research_project_members(username);
                CREATE INDEX IF NOT EXISTS idx_research_experiments_project
                    ON research_experiments(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_research_tasks_project
                    ON research_tasks(project_id, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_research_runs_user_session
                    ON research_runs(user_id, session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_research_runs_project
                    ON research_runs(project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_research_memory_confirmations_user
                    ON research_memory_confirmations(user_id, confirmed_at DESC);
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

    @staticmethod
    def _json_object(value: Any, default: Any) -> Any:
        """把运行记录 JSON 字段安全还原为预期容器。"""
        if isinstance(value, type(default)):
            return value
        try:
            decoded = json.loads(value or json.dumps(default))
        except (TypeError, ValueError, json.JSONDecodeError):
            return default
        return decoded if isinstance(decoded, type(default)) else default

    def _run_accessible(self, run: dict, user: dict) -> bool:
        """无项目运行仅本人可见；项目运行沿用项目当前 ACL。"""
        username, _ = self._identity(user)
        if not run.get("project_id"):
            return run.get("user_id") == username
        try:
            self.get_project(str(run["project_id"]), user)
            return True
        except (PermissionError, ValueError):
            return False

    @staticmethod
    def _filter_run_payload_for_document_acl(run: dict, user: dict) -> dict:
        """权限变化后 fail closed，避免历史 trace 重新泄漏证据内容。"""
        from src.rag.retrieval.acl_filter import check_doc_access

        package = dict(run.get("evidence_package") or {})
        evidences = list(package.get("evidences") or [])
        allowed = []
        hidden = 0
        for evidence in evidences:
            if not isinstance(evidence, dict):
                hidden += 1
                continue
            metadata = evidence.get("metadata")
            if not isinstance(metadata, dict) or not check_doc_access(metadata, user):
                hidden += 1
                continue
            allowed.append(evidence)
        package["evidences"] = allowed
        run["evidence_package"] = package
        run["hidden_evidence_count"] = hidden

        if hidden:
            # 最终答案、声明和复核文本可能转述已撤权证据；无法逐 token
            # 可靠裁剪时宁可隐藏整块内容，也不能只删除来源卡片。
            run["final_answer"] = "该运行包含您当前无权访问的证据，回答内容已隐藏。"
            run["source_cards"] = []
            run["analysis_report"] = {}
            run["review_report"] = {}
        return run

    def save_research_run(self, payload: dict, user: dict) -> dict:
        """保存一次已经完成的 Deep Research；不参与回答关键路径。"""
        question = str(payload.get("question", "")).strip()
        session_id = str(payload.get("session_id", "")).strip()
        if not question:
            raise ValueError("研究问题不能为空")
        if not session_id:
            raise ValueError("会话 ID 不能为空")

        project_id = str(payload.get("project_id") or "").strip() or None
        if project_id:
            self.get_project(project_id, user)

        username, _ = self._identity(user)
        run_id = str(payload.get("id") or uuid.uuid4().hex)
        status = str(payload.get("status") or "completed")
        if status not in {"completed", "failed"}:
            raise ValueError("研究运行状态不合法")
        now = time.time()

        json_fields = {
            "source_cards_json": payload.get("source_cards") or [],
            "evidence_package_json": payload.get("evidence_package") or {},
            "analysis_report_json": payload.get("analysis_report") or {},
            "review_report_json": payload.get("review_report") or {},
            "trace_json": payload.get("research_trace") or {},
            "metrics_json": payload.get("metrics") or {},
        }
        encoded = {
            key: json.dumps(value, ensure_ascii=False, default=str)
            for key, value in json_fields.items()
        }
        with self._connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO research_runs
                   (id, project_id, session_id, user_id, question, mode, status,
                    final_answer, source_cards_json, evidence_package_json,
                    analysis_report_json, review_report_json, trace_json,
                    metrics_json, created_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, project_id, session_id, username, question, "deep", status,
                    str(payload.get("final_answer") or ""),
                    encoded["source_cards_json"], encoded["evidence_package_json"],
                    encoded["analysis_report_json"], encoded["review_report_json"],
                    encoded["trace_json"], encoded["metrics_json"], now, now,
                ),
            )
            row = conn.execute("SELECT * FROM research_runs WHERE id = ?", (run_id,)).fetchone()
        return self._serialize_research_run(row, include_payload=True, user=user)

    def _serialize_research_run(
        self,
        row: sqlite3.Row,
        *,
        include_payload: bool,
        user: dict,
    ) -> dict:
        item = dict(row)
        item["source_cards"] = self._json_object(item.pop("source_cards_json"), [])
        item["metrics"] = self._json_object(item.pop("metrics_json"), {})
        if include_payload:
            item["evidence_package"] = self._json_object(item.pop("evidence_package_json"), {})
            item["analysis_report"] = self._json_object(item.pop("analysis_report_json"), {})
            item["review_report"] = self._json_object(item.pop("review_report_json"), {})
            item["research_trace"] = self._json_object(item.pop("trace_json"), {})
            return self._filter_run_payload_for_document_acl(item, user)
        item.pop("source_cards", None)
        item["final_answer"] = str(item.get("final_answer") or "")[:240]
        for key in (
            "evidence_package_json", "analysis_report_json", "review_report_json", "trace_json",
        ):
            item.pop(key, None)
        return item

    def list_research_runs(
        self,
        user: dict,
        *,
        session_id: str = "",
        project_id: str = "",
        limit: int = 20,
    ) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        if project_id:
            self.get_project(project_id, user)
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM research_runs
                   WHERE (? = '' OR session_id = ?)
                     AND (? = '' OR project_id = ?)
                   ORDER BY created_at DESC LIMIT ?""",
                (session_id, session_id, project_id, project_id, limit),
            ).fetchall()
        return [
            self._serialize_research_run(row, include_payload=False, user=user)
            for row in rows
            if self._run_accessible(dict(row), user)
        ]

    def get_research_run(self, run_id: str, user: dict) -> dict:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM research_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            raise ValueError("研究运行不存在")
        if not self._run_accessible(dict(row), user):
            raise PermissionError("无权访问该研究运行")
        item = self._serialize_research_run(row, include_payload=True, user=user)
        username, _ = self._identity(user)
        with self._connection() as conn:
            confirmations = conn.execute(
                """SELECT claim_id FROM research_memory_confirmations
                   WHERE run_id = ? AND user_id = ? ORDER BY confirmed_at""",
                (run_id, username),
            ).fetchall()
        item["confirmed_claim_ids"] = [row["claim_id"] for row in confirmations]
        return item

    def find_reusable_research_episode(
        self,
        question: str,
        user: dict,
        *,
        project_id: str = "",
    ) -> Optional[dict]:
        """查找同题、同权限范围的最近一次成功运行，仅返回检索计划。

        历史答案和 Claim 不会进入新一轮上下文；新运行仍会按当前 ACL 重新检索，
        因而 Research Run 只充当可复用的“研究过程记忆”。
        """
        normalized = str(question or "").strip()
        if not normalized:
            return None
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM research_runs
                   WHERE question = ? AND status = 'completed'
                     AND ((? = '' AND project_id IS NULL) OR project_id = ?)
                   ORDER BY completed_at DESC LIMIT 20""",
                (normalized, project_id, project_id),
            ).fetchall()
        for row in rows:
            raw = dict(row)
            if not self._run_accessible(raw, user):
                continue
            detail = self._serialize_research_run(row, include_payload=True, user=user)
            if detail.get("hidden_evidence_count"):
                continue
            if (detail.get("review_report") or {}).get("decision") != "PASS":
                continue
            researcher = (
                (detail.get("research_trace") or {}).get("stages") or {}
            ).get("researcher") or {}
            subquestions = [
                str(item).strip()
                for item in researcher.get("subquestions") or []
                if str(item).strip()
            ]
            subquestions = list(dict.fromkeys(subquestions))[:4]
            if len(subquestions) < 2:
                continue
            return {
                "run_id": detail["id"],
                "created_at": detail["created_at"],
                "subquestions": subquestions,
                "reuse_policy": "仅复用检索计划；证据按当前 ACL 重新检索",
            }
        return None

    def prepare_confirmed_claim(self, run_id: str, claim_id: str, user: dict) -> dict:
        """验证某条 Research Run 事实是否满足长期记忆提升门槛。

        这里只准备候选，不写 Mem0。调用方必须收到用户显式确认后再执行写入。
        """
        detail = self.get_research_run(run_id, user)
        if detail.get("status") != "completed":
            raise ValueError("只有已完成的研究运行可以提升长期记忆")

        review = detail.get("review_report") or {}
        if review.get("decision") != "PASS" or review.get("acl_verified", True) is not True:
            raise ValueError("该结论尚未通过 Reviewer 与 ACL 复核")

        claims = (detail.get("analysis_report") or {}).get("claims") or []
        claim = next(
            (item for item in claims if str(item.get("claim_id") or "") == claim_id),
            None,
        )
        if not claim:
            raise ValueError("研究结论不存在")
        if claim.get("claim_type") != "fact":
            raise ValueError("只有事实类结论可以提升为长期记忆")

        source_ids = [str(item) for item in claim.get("source_ids") or [] if str(item)]
        evidences = {
            str(item.get("source_id")): item
            for item in (detail.get("evidence_package") or {}).get("evidences") or []
            if isinstance(item, dict) and item.get("source_id")
        }
        if not source_ids or any(source_id not in evidences for source_id in source_ids):
            raise ValueError("该事实缺少当前可访问的有效证据")

        claim_text = str(claim.get("text") or "").strip()
        rejected = [
            item for item in review.get("items") or []
            if isinstance(item, dict)
            and item.get("supported") is False
            and str(item.get("claim") or "").strip() == claim_text
        ]
        if rejected:
            raise ValueError("该事实被 Reviewer 标记为不受支持")

        return {
            "run_id": run_id,
            "claim_id": claim_id,
            "text": claim_text,
            "source_ids": source_ids,
            "source_titles": [
                str(evidences[source_id].get("title") or evidences[source_id].get("source") or source_id)
                for source_id in source_ids
            ],
            "already_confirmed": claim_id in (detail.get("confirmed_claim_ids") or []),
        }

    def record_memory_confirmation(
        self,
        run_id: str,
        claim_id: str,
        user: dict,
        memory_result: dict,
    ) -> None:
        """记录已经成功提交到 Mem0 的用户确认，提供幂等审计。"""
        username, _ = self._identity(user)
        with self._connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO research_memory_confirmations
                   (run_id, claim_id, user_id, confirmed_at, memory_result_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    run_id,
                    claim_id,
                    username,
                    time.time(),
                    json.dumps(memory_result or {}, ensure_ascii=False, default=str),
                ),
            )

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
