"""
知识管理 API 路由
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from rag.vectorstore import get_vectorstore_manager
from rag.retriever import get_retriever_manager
from rag.document_loader import get_document_loader_manager
from config.settings import get_settings
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
    """添加文档到知识库"""
    try:
        vectorstore_manager = get_vectorstore_manager()
        
        # 创建文档
        doc = Document(
            page_content=request.content,
            metadata=request.metadata
        )
        
        # 添加到向量存储
        ids = vectorstore_manager.add_documents([doc])
        
        return {
            "message": "文档添加成功",
            "ids": ids,
            "count": len(ids)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加文档失败: {str(e)}")
@router.post("/add/file")
async def add_document_from_file(
    file: UploadFile = File(...),
    category: str = Form(default="general")
):
    """从文件添加文档"""
    try:
        # 读取文件内容
        content = await file.read()
        
        # 保存到临时文件
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file.filename
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # 加载文档
        loader_manager = get_document_loader_manager()
        docs = loader_manager.load_file(tmp_path)
        
        # 添加元数据
        for doc in docs:
            doc.metadata["source"] = file.filename
            doc.metadata["category"] = category
        
        # 添加到向量存储
        vectorstore_manager = get_vectorstore_manager()
        ids = vectorstore_manager.add_documents(docs)
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        return {
            "message": "文件添加成功",
            "filename": file.filename,
            "count": len(ids)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加文件失败: {str(e)}")
@router.post("/search")
async def search_knowledge(request: SearchRequest):
    """搜索知识库"""
    try:
        retriever_manager = get_retriever_manager()
        
        results = retriever_manager.search(
            request.query,
            k=request.top_k
        )
        
        return {
            "query": request.query,
            "count": len(results),
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
@router.get("/stats")
async def get_knowledge_stats():
    """获取知识库统计信息"""
    try:
        vectorstore_manager = get_vectorstore_manager()
        info = vectorstore_manager.get_collection_info()
        
        return info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")
@router.delete("/clear")
async def clear_knowledge():
    """清空知识库"""
    try:
        vectorstore_manager = get_vectorstore_manager()
        vectorstore_manager.delete_collection()
        
        return {"message": "知识库已清空"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空知识库失败: {str(e)}")

