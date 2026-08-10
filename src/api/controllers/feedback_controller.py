"""
反馈 Router
"""
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas import (
    FeedbackIssueItem,
    FeedbackIssueListResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    UpdateFeedbackIssueRequest,
)
from ..security import get_current_user
from ..services.feedback_service import feedback_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


def _raise_service_error(error: Exception) -> None:
    if isinstance(error, PermissionError):
        raise HTTPException(status_code=403, detail=str(error))
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error))
    raise error


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        username = current_user.get("username", "anonymous")
        result = feedback_service.submit_feedback(
            username=username,
            session_id=request.session_id,
            question=request.question,
            answer=request.answer,
            used_agent=request.used_agent,
            feedback_type=request.feedback_type,
            comment=request.comment,
        )
        return FeedbackResponse(success=bool(result["success"]), message=str(result["message"]))
    except Exception as e:
        logger.exception("提交反馈失败: %s", e)
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(current_user: dict = Depends(get_current_user)):
    try:
        stats = feedback_service.get_feedback_stats(current_user)
        return FeedbackStatsResponse(**stats)
    except Exception as e:
        logger.exception("获取反馈统计失败: %s", e)
        raise HTTPException(status_code=500, detail=f"获取反馈统计失败: {str(e)}")


@router.get("/issues", response_model=FeedbackIssueListResponse)
async def get_feedback_issues(
    limit: int = Query(default=20, ge=1, le=100),
    status: Literal["open", "resolved"] = Query(default="open"),
    current_user: dict = Depends(get_current_user),
):
    try:
        result = feedback_service.list_feedback_issues(
            current_user,
            limit=limit,
            status=status,
        )
        return FeedbackIssueListResponse(**result)
    except Exception as e:
        logger.exception("获取反馈问题清单失败: %s", e)
        raise HTTPException(status_code=500, detail=f"获取反馈问题清单失败: {str(e)}")


@router.patch("/issues/{feedback_id}", response_model=FeedbackIssueItem)
async def update_feedback_issue(
    feedback_id: int,
    request: UpdateFeedbackIssueRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        result = feedback_service.update_feedback_issue(
            feedback_id=feedback_id,
            status=request.status,
            resolution_note=request.resolution_note,
            user=current_user,
        )
        return FeedbackIssueItem(**result)
    except (PermissionError, ValueError) as error:
        _raise_service_error(error)
    except Exception as error:
        logger.exception("更新反馈问题失败: %s", error)
        raise HTTPException(status_code=500, detail=f"更新反馈问题失败: {str(error)}")
