"""
知识管理 Controller
"""
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from api.services import knowledge_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


class AddDocumentRequest(BaseModel):
    """添加文档请求"""
    content: str = Field(description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(description="搜索查询")
    top_k: int = Field(default=5, description="返回结果数量")


@router.post("/add")
async def add_document(request: AddDocumentRequest):
    """
    添加文档到知识库
    """
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
    category: str = Form(default="general")
):
    """
    从文件添加文档
    """
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


@router.post("/search")
async def search_knowledge(request: SearchRequest):
    """
    搜索知识库
    """
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
async def get_knowledge_stats():
    """
    获取知识库统计信息
    """
    try:
        result = knowledge_service.get_stats()
        return result
    except Exception as e:
        logger.exception(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.delete("/clear")
async def clear_knowledge():
    """
    清空知识库
    """
    try:
        result = knowledge_service.clear()
        return result
    except Exception as e:
        logger.exception(f"清空知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空知识库失败: {str(e)}")
