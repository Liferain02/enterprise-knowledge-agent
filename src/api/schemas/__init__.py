"""
Pydantic Schemas - 请求/响应模型
"""
from pydantic import BaseModel, Field
from typing import List, Any, Optional


# ==================== Request Models ====================
class ImageContent(BaseModel):
    """单张图片内容"""
    data: str = Field(description="图片数据，URL 或 base64 编码（data:image/xxx;base64,xxx 格式）")
    type: str = Field(default="base64", description="图片类型: base64 / url")
    filename: Optional[str] = Field(default=None, description="文件名（可选）")


class ChatRequest(BaseModel):
    """聊天请求（支持多模态）"""
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default", description="会话ID")
    images: Optional[List[ImageContent]] = Field(default=None, description="附带的图片（支持多张）")


class ChatStreamRequest(BaseModel):
    """流式聊天请求"""
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default", description="会话ID")
    images: Optional[List[ImageContent]] = Field(default=None, description="附带的图片（支持多张）")


class ChatStreamRequest(BaseModel):
    """流式聊天请求"""
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default", description="会话ID")
    images: Optional[List[ImageContent]] = Field(default=None, description="附带的图片（支持多张）")


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    title: Optional[str] = Field(default=None, description="会话标题（可选）")


class UpdateTitleRequest(BaseModel):
    """更新标题请求"""
    title: str = Field(description="新标题")


class AddDocumentRequest(BaseModel):
    """添加文档请求"""
    file_path: str = Field(description="文件路径")
    metadata: Optional[dict] = Field(default=None, description="元数据")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(description="搜索查询")
    top_k: int = Field(default=5, description="返回结果数量")


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(description="用户名")
    password: str = Field(description="密码")


# ==================== Response Models ====================
class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(description="AI回复")
    sources: List[Any] = Field(default_factory=list, description="信息来源")
    session_id: str = Field(description="会话ID")
    used_agent: str = Field(description="使用的Agent类型")
    image_understood: bool = Field(default=False, description="是否对图片进行了理解")


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[dict] = Field(default_factory=list, description="搜索结果")
    total: int = Field(description="结果总数")


class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str = Field(description="访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(description="服务状态")
    service: str = Field(description="服务名称")
    version: str = Field(description="版本号")


__all__ = [
    "ImageContent",
    "ChatRequest",
    "CreateSessionRequest",
    "UpdateTitleRequest",
    "AddDocumentRequest",
    "SearchRequest",
    "LoginRequest",
    "ChatResponse",
    "SearchResponse",
    "LoginResponse",
    "HealthResponse",
]
