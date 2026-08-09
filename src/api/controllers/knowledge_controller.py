"""
知识管理 Router
"""
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, status

from ..schemas import (
    AddDocumentRequest,
    KnowledgeDocumentListResponse,
    KnowledgeOverviewResponse,
    IngestionJobItem,
    IngestionJobListResponse,
    IngestionSubmitResponse,
    SearchRequest,
    SearchResponse,
)
from ..services.knowledge_service import knowledge_service
from ..security import get_current_user
from src.rag.retrieval.acl_filter import UserContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

_EDITOR_ROLES = {
    "admin", "pi", "teacher", "lab_admin", "senior_student",
    # security_user.py 的兼容角色
    "editor",
    # 历史兼容角色
    "manager", "hr", "it_support",
}
_ADMIN_ROLES = {"admin", "pi", "lab_admin"}


def _require_role(current_user: dict, allowed_roles: set[str], action: str) -> None:
    role = current_user.get("role", "student")
    if role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"当前角色无权{action}",
        )


@router.post("/add")
async def add_document(request: AddDocumentRequest, current_user: dict = Depends(get_current_user)):
    """添加文档到知识库"""
    try:
        _require_role(current_user, _EDITOR_ROLES, "添加资料")
        result = knowledge_service.add_document(
            content=request.content,
            metadata=request.metadata
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"添加文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加文档失败: {str(e)}")


@router.post("/add/file", response_model=IngestionSubmitResponse)
async def add_document_from_file(
    file: UploadFile = File(...),
    category: str = Form(default="general"),
    title: str = Form(default=""),
    doc_type: str = Form(default="general"),
    author: str = Form(default=""),
    project_name: str = Form(default=""),
    research_direction: str = Form(default=""),
    visibility: str = Form(default="public"),
    created_at: str = Form(default=""),
    summary: str = Form(default=""),
    tags: str = Form(default=""),
    current_user: dict = Depends(get_current_user),
):
    """从文件添加文档"""
    try:
        _require_role(current_user, _EDITOR_ROLES, "上传资料")
        file_content = await file.read()
        metadata = {
            "title": title or None,
            "doc_type": doc_type or category,
            "author": author or None,
            "project_name": project_name or None,
            "research_direction": research_direction or None,
            "visibility": visibility or "public",
            "created_at": created_at or None,
            "summary": summary or None,
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
        }
        result = knowledge_service.enqueue_document_from_file(
            file_content=file_content,
            filename=file.filename,
            category=category,
            metadata=metadata,
            uploaded_by=current_user.get("username", "unknown"),
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"添加文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加文件失败: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest, current_user: dict = Depends(get_current_user)):
    """搜索知识库"""
    try:
        user_context = UserContext.from_jwt_payload(current_user)
        result = knowledge_service.search(
            query=request.query,
            top_k=request.top_k,
            filters={
                "doc_type": request.doc_type,
                "project_name": request.project_name,
                "visibility": request.visibility,
                "author": request.author,
                "research_direction": request.research_direction,
            },
            user_context=user_context,
        )
        return SearchResponse(results=result["results"], total=result["count"])
    except Exception as e:
        logger.exception(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.get("/stats")
async def get_knowledge_stats(_: dict = Depends(get_current_user)):
    """获取知识库统计信息"""
    try:
        result = knowledge_service.get_stats()
        return result
    except Exception as e:
        logger.exception(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/ingestion/jobs", response_model=IngestionJobListResponse)
async def list_ingestion_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    _: dict = Depends(get_current_user),
):
    """查询最近异步入库任务及状态统计。"""
    jobs = knowledge_service.list_ingestion_jobs(limit=limit)
    return IngestionJobListResponse(
        jobs=jobs,
        total=len(jobs),
        stats=knowledge_service.get_ingestion_stats(),
    )


@router.get("/ingestion/jobs/{job_id}", response_model=IngestionJobItem)
async def get_ingestion_job(job_id: str, _: dict = Depends(get_current_user)):
    """查询单个异步入库任务。"""
    job = knowledge_service.get_ingestion_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="入库任务不存在")
    return IngestionJobItem(**job)


@router.get("/documents", response_model=KnowledgeDocumentListResponse)
async def list_documents(
    doc_type: str | None = Query(default=None),
    project_name: str | None = Query(default=None),
    visibility: str | None = Query(default=None),
    query: str | None = Query(default=None),
    current_user: dict = Depends(get_current_user),
):
    """浏览资料目录，返回按来源聚合后的文档列表。"""
    try:
        documents = knowledge_service.list_documents(
            doc_type=doc_type,
            project_name=project_name,
            visibility=visibility,
            query=query,
            user_context=UserContext.from_jwt_payload(current_user),
        )
        return KnowledgeDocumentListResponse(documents=documents, total=len(documents))
    except Exception as e:
        logger.exception(f"获取资料目录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取资料目录失败: {str(e)}")


@router.get("/overview", response_model=KnowledgeOverviewResponse)
async def get_knowledge_overview(current_user: dict = Depends(get_current_user)):
    """获取资料中心概览。"""
    try:
        return KnowledgeOverviewResponse(
            **knowledge_service.get_overview(
                user_context=UserContext.from_jwt_payload(current_user),
            )
        )
    except Exception as e:
        logger.exception(f"获取资料概览失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取资料概览失败: {str(e)}")


@router.delete("/documents")
async def delete_document(
    source: str = Query(..., min_length=1),
    current_user: dict = Depends(get_current_user),
):
    """删除资料及其所有 chunks。"""
    try:
        _require_role(current_user, _EDITOR_ROLES, "删除资料")
        return knowledge_service.delete_document(source)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"删除资料失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除资料失败: {str(e)}")


@router.post("/seed-lab-samples")
async def seed_lab_samples(current_user: dict = Depends(get_current_user)):
    """导入一批实验室样例资料到知识库"""
    try:
        _require_role(current_user, _EDITOR_ROLES, "导入样例资料")
        return knowledge_service.seed_lab_sample_documents()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"导入实验室样例资料失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导入实验室样例资料失败: {str(e)}")


@router.delete("/clear")
async def clear_knowledge(current_user: dict = Depends(get_current_user)):
    """清空知识库"""
    try:
        _require_role(current_user, _ADMIN_ROLES, "清空知识库")
        result = knowledge_service.clear()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"清空知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空知识库失败: {str(e)}")
