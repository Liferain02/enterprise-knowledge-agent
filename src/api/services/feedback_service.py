"""
反馈服务
"""
from typing import Dict, Optional

from ..repositories import feedback_dao


_FEEDBACK_REVIEWER_ROLES = {
    "admin", "pi", "teacher", "lab_admin", "senior_student",
    "editor", "manager", "hr", "it_support",
}


class FeedbackService:
    """反馈服务类"""

    def __init__(self, dao=None):
        self.dao = dao or feedback_dao

    @staticmethod
    def can_review(user: Dict[str, str]) -> bool:
        return user.get("role", "student") in _FEEDBACK_REVIEWER_ROLES

    def submit_feedback(
        self,
        username: str,
        session_id: str,
        question: str,
        answer: str,
        used_agent: str,
        feedback_type: str,
        comment: Optional[str] = None,
    ) -> Dict[str, object]:
        feedback_id = self.dao.save(
            username=username,
            session_id=session_id,
            question=question,
            answer=answer,
            used_agent=used_agent,
            feedback_type=feedback_type,
            comment=comment,
        )
        return {
            "success": True,
            "message": "反馈已记录",
            "feedback_id": feedback_id,
        }

    def get_feedback_stats(self, user: Dict[str, str]) -> Dict[str, int]:
        username = None if self.can_review(user) else user.get("username", "anonymous")
        return self.dao.get_stats(username=username)

    def list_feedback_issues(
        self,
        user: Dict[str, str],
        limit: int = 20,
        status: str = "open",
    ) -> Dict[str, object]:
        username: Optional[str] = None
        if not self.can_review(user):
            username = user.get("username", "anonymous")
        issues = self.dao.list_issues(username=username, limit=limit, status=status)
        return {
            "issues": issues,
            "total": len(issues),
        }

    def update_feedback_issue(
        self,
        feedback_id: int,
        status: str,
        resolution_note: Optional[str],
        user: Dict[str, str],
    ) -> Dict[str, object]:
        if not self.can_review(user):
            raise PermissionError("当前角色无权处理知识缺口")
        return self.dao.update_issue_status(
            feedback_id=feedback_id,
            status=status,
            resolution_note=resolution_note,
            resolved_by=user.get("username", "anonymous"),
        )


feedback_service = FeedbackService()
