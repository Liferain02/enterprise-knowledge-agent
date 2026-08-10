"""反馈驱动的知识缺口闭环测试。"""
import sqlite3

import pytest

from src.api.repositories.dao.session_dao import FeedbackDAO
from src.api.services.feedback_service import FeedbackService


def _submit_issue(service: FeedbackService, username: str, question: str) -> int:
    result = service.submit_feedback(
        username=username,
        session_id=f"session-{username}",
        question=question,
        answer="知识库中暂时没有足够信息。",
        used_agent="knowledge_agent",
        feedback_type="missing_material",
        comment="希望补充相关说明",
    )
    return int(result["feedback_id"])


def test_feedback_dao_migrates_legacy_database(tmp_path):
    db_path = tmp_path / "legacy-sessions.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                used_agent TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            INSERT INTO feedback (
                username, session_id, question, answer, used_agent,
                feedback_type, comment, created_at
            ) VALUES ('alice', 's1', '缺什么资料？', '暂无资料',
                      'knowledge_agent', 'missing_material', '', '2026-08-09')
        """)

    dao = FeedbackDAO(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    assert {"status", "resolution_note", "resolved_by", "resolved_at"} <= columns
    assert dao.list_issues(username="alice")[0]["status"] == "open"


def test_issue_scope_and_reviewer_stats(tmp_path):
    service = FeedbackService(FeedbackDAO(tmp_path / "sessions.db"))
    _submit_issue(service, "alice", "如何申请集群账号？")
    _submit_issue(service, "bob", "RDMA 环境怎样配置？")

    student_result = service.list_feedback_issues(
        {"username": "alice", "role": "student"}
    )
    reviewer_result = service.list_feedback_issues(
        {"username": "mentor", "role": "pi"}
    )

    assert [item["username"] for item in student_result["issues"]] == ["alice"]
    assert {item["username"] for item in reviewer_result["issues"]} == {"alice", "bob"}
    assert service.get_feedback_stats({"username": "alice", "role": "student"})["total"] == 1
    assert service.get_feedback_stats({"username": "mentor", "role": "pi"})["total"] == 2


def test_reviewer_can_resolve_and_reopen_issue(tmp_path):
    service = FeedbackService(FeedbackDAO(tmp_path / "sessions.db"))
    feedback_id = _submit_issue(service, "alice", "服务器预约规则是什么？")
    student = {"username": "alice", "role": "student"}
    reviewer = {"username": "maintainer", "role": "lab_admin"}

    with pytest.raises(PermissionError):
        service.update_feedback_issue(feedback_id, "resolved", "已补充 FAQ", student)
    with pytest.raises(ValueError, match="解决说明"):
        service.update_feedback_issue(feedback_id, "resolved", "", reviewer)

    resolved = service.update_feedback_issue(
        feedback_id,
        "resolved",
        "已上传《设备预约与共享资源使用流程》并完成入库。",
        reviewer,
    )
    assert resolved["status"] == "resolved"
    assert resolved["resolved_by"] == "maintainer"
    assert service.list_feedback_issues(reviewer, status="open")["total"] == 0
    assert service.list_feedback_issues(reviewer, status="resolved")["total"] == 1

    reopened = service.update_feedback_issue(feedback_id, "open", None, reviewer)
    assert reopened["status"] == "open"
    assert reopened["resolution_note"] is None
    assert reopened["resolved_by"] is None
