"""
知识管理 Router
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends

from ..schemas import AddDocumentRequest, SearchRequest, SearchResponse
from ..services import knowledge_service
from ..security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.post("/add")
async def add_document(request: AddDocumentRequest, _: dict = Depends(get_current_user)):
    """添加文档到知识库"""
    try:
        result = knowledge_service.add_document(
            content=request.content,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        logger.exception(f"添加文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加文档失败: {str(e)}")


@router.post("/add/file")
async def add_document_from_file(
    file: UploadFile = File(...),
    category: str = Form(default="general"),
    _: dict = Depends(get_current_user),
):
    """从文件添加文档"""
    try:
        file_content = await file.read()
        result = knowledge_service.add_document_from_file(
            file_content=file_content,
            filename=file.filename,
            category=category
        )
        return result
    except Exception as e:
        logger.exception(f"添加文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加文件失败: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest, _: dict = Depends(get_current_user)):
    """搜索知识库"""
    try:
        result = knowledge_service.search(
            query=request.query,
            top_k=request.top_k
        )
        return result
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


@router.delete("/clear")
async def clear_knowledge(_: dict = Depends(get_current_user)):
    """清空知识库"""
    try:
        result = knowledge_service.clear()
        return result
    except Exception as e:
        logger.exception(f"清空知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空知识库失败: {str(e)}")
