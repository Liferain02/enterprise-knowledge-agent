"""
Pydantic Schemas - 请求/响应模型
"""
from pydantic import BaseModel, Field
from typing import List, Any, Optional, Literal


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
    research_mode: Literal["normal", "deep"] = Field(
        default="normal",
        description="研究模式：normal 保持原链路；deep 使用固定三角色深度研究链路",
    )
    project_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="可选科研项目 ID；仅用于明确关联研究运行，不从问题文本推断",
    )


class ChatStreamRequest(BaseModel):
    """流式聊天请求"""
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default", description="会话ID")
    images: Optional[List[ImageContent]] = Field(default=None, description="附带的图片（支持多张）")
    research_mode: Literal["normal", "deep"] = Field(
        default="normal",
        description="研究模式：normal 或 deep",
    )
    project_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="可选科研项目 ID；仅用于明确关联研究运行",
    )


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    title: Optional[str] = Field(default=None, description="会话标题（可选）")


class UpdateTitleRequest(BaseModel):
    """更新标题请求"""
    title: str = Field(description="新标题")


class AddDocumentRequest(BaseModel):
    """添加文档请求"""
    content: str = Field(description="文档内容")
    metadata: Optional[dict] = Field(default=None, description="元数据")


class DocumentUploadMetadata(BaseModel):
    """实验室文档上传元数据"""
    title: Optional[str] = Field(default=None, description="文档标题")
    doc_type: Literal[
        "lab_policy", "project_doc", "paper_note", "env_setup",
        "meeting_minutes", "faq", "experiment_log", "onboarding", "general"
    ] = Field(default="general", description="文档类型")
    author: Optional[str] = Field(default=None, description="作者")
    project_name: Optional[str] = Field(default=None, description="所属项目")
    research_direction: Optional[str] = Field(default=None, description="研究方向")
    visibility: Literal["public", "project", "restricted"] = Field(
        default="public",
        description="可见范围：公共 / 项目组内 / 负责人可见"
    )
    tags: List[str] = Field(default_factory=list, description="标签列表")
    created_at: Optional[str] = Field(default=None, description="文档时间")
    summary: Optional[str] = Field(default=None, description="文档简介")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(description="搜索查询")
    top_k: int = Field(default=5, description="返回结果数量")
    doc_type: Optional[str] = Field(default=None, description="按文档类型过滤")
    project_name: Optional[str] = Field(default=None, description="按项目过滤")
    visibility: Optional[str] = Field(default=None, description="按可见性过滤")
    author: Optional[str] = Field(default=None, description="按作者过滤")
    research_direction: Optional[str] = Field(default=None, description="按研究方向过滤")


class SourceItem(BaseModel):
    """回答来源卡片"""
    title: str = Field(description="来源标题")
    snippet: str = Field(description="证据片段")
    doc_type: str = Field(default="general", description="文档类型")
    author: Optional[str] = Field(default=None, description="作者")
    project_name: Optional[str] = Field(default=None, description="所属项目")
    research_direction: Optional[str] = Field(default=None, description="研究方向")
    created_at: Optional[str] = Field(default=None, description="文档时间")
    score: Optional[float] = Field(default=None, description="相关分")
    source: Optional[str] = Field(default=None, description="原始来源文件名")
    visibility: Optional[str] = Field(default=None, description="可见性")


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(description="用户名")
    password: str = Field(description="密码")


class RegisterRequest(BaseModel):
    """注册请求"""
    username: str = Field(description="用户名（3-32个字符，字母数字下划线）")
    password: str = Field(description="密码（6-128个字符）")


class RegisterResponse(BaseModel):
    """注册响应"""
    success: bool = Field(description="是否成功")
    message: str = Field(description="结果信息")


class FeedbackRequest(BaseModel):
    """回答反馈请求"""
    session_id: str = Field(description="会话ID")
    question: str = Field(description="用户问题")
    answer: str = Field(description="助手回答")
    used_agent: str = Field(description="使用的Agent")
    feedback_type: Literal["helpful", "incorrect", "missing_material"] = Field(
        description="反馈类型"
    )
    comment: Optional[str] = Field(default=None, description="补充说明")


class FeedbackResponse(BaseModel):
    """回答反馈响应"""
    success: bool = Field(description="是否提交成功")
    message: str = Field(description="结果信息")


class FeedbackStatsResponse(BaseModel):
    """反馈统计响应"""
    total: int = Field(description="反馈总数")
    helpful: int = Field(description="有帮助反馈数")
    incorrect: int = Field(description="不准确反馈数")
    missing_material: int = Field(description="缺少资料反馈数")


class FeedbackIssueItem(BaseModel):
    """反馈问题项"""
    id: int = Field(description="反馈ID")
    username: str = Field(description="提交人")
    session_id: str = Field(description="来源会话ID")
    feedback_type: str = Field(description="反馈类型")
    question: str = Field(description="问题")
    comment: Optional[str] = Field(default=None, description="备注")
    status: Literal["open", "resolved"] = Field(description="处理状态")
    resolution_note: Optional[str] = Field(default=None, description="解决说明")
    resolved_by: Optional[str] = Field(default=None, description="处理人")
    resolved_at: Optional[str] = Field(default=None, description="解决时间")
    created_at: str = Field(description="创建时间")


class FeedbackIssueListResponse(BaseModel):
    """反馈问题列表响应"""
    issues: List[FeedbackIssueItem] = Field(default_factory=list, description="问题列表")
    total: int = Field(description="问题总数")


class UpdateFeedbackIssueRequest(BaseModel):
    """更新知识缺口处理状态"""
    status: Literal["open", "resolved"] = Field(description="目标状态")
    resolution_note: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="解决说明；标记为已解决时必填",
    )


# ==================== Research Workspace Models ====================
class ProjectMemberItem(BaseModel):
    """项目成员"""
    username: str = Field(description="成员账号")
    member_role: str = Field(default="member", description="项目内角色")
    created_at: float = Field(description="加入时间戳")


class CreateProjectRequest(BaseModel):
    """创建科研项目空间"""
    title: str = Field(min_length=1, max_length=120, description="项目名称")
    slug: Optional[str] = Field(default=None, max_length=64, description="短标识")
    summary: str = Field(default="", max_length=1000, description="项目简介")
    research_direction: str = Field(default="", max_length=120, description="研究方向")
    status: Literal["planned", "active", "paused", "completed"] = Field(default="active")
    visibility: Literal["public", "project", "restricted"] = Field(default="project")
    lead: Optional[str] = Field(default=None, max_length=64, description="项目负责人账号")
    members: List[str] = Field(default_factory=list, description="项目成员账号")


class ProjectItem(BaseModel):
    """科研项目空间"""
    id: str
    slug: str
    title: str
    summary: str
    research_direction: str
    status: str
    visibility: str
    lead: str
    created_by: str
    created_at: float
    updated_at: float
    experiment_count: int = 0
    open_task_count: int = 0
    members: List[ProjectMemberItem] = Field(default_factory=list)


class ProjectListResponse(BaseModel):
    """科研项目列表"""
    projects: List[ProjectItem] = Field(default_factory=list)
    total: int


class CreateExperimentRequest(BaseModel):
    """创建结构化实验记录"""
    title: str = Field(min_length=1, max_length=160, description="实验标题")
    hypothesis: str = Field(default="", max_length=2000, description="实验假设")
    environment: str = Field(default="", max_length=4000, description="运行环境")
    code_commit: str = Field(default="", max_length=120, description="代码 commit")
    dataset_version: str = Field(default="", max_length=240, description="数据集版本")
    metrics: dict[str, Any] = Field(default_factory=dict, description="指标键值")
    conclusion: str = Field(default="", max_length=4000, description="实验结论")
    next_steps: str = Field(default="", max_length=4000, description="后续动作")
    status: Literal["planned", "running", "completed", "failed"] = Field(default="planned")


class ExperimentItem(BaseModel):
    """结构化实验记录"""
    id: str
    project_id: str
    title: str
    hypothesis: str
    environment: str
    code_commit: str
    dataset_version: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    conclusion: str
    next_steps: str
    status: str
    created_by: str
    created_at: float
    updated_at: float


class ExperimentListResponse(BaseModel):
    """实验记录列表"""
    experiments: List[ExperimentItem] = Field(default_factory=list)
    total: int


class CreateResearchTaskRequest(BaseModel):
    """创建项目待办"""
    title: str = Field(min_length=1, max_length=240)
    assignee: str = Field(default="", max_length=64)
    due_date: str = Field(default="", max_length=32)
    source: str = Field(default="", max_length=240)
    status: Literal["open", "in_progress", "done"] = Field(default="open")


class UpdateResearchTaskRequest(BaseModel):
    """更新项目待办状态"""
    status: Literal["open", "in_progress", "done"]


class ExtractMeetingTasksRequest(BaseModel):
    """从组会纪要中提取行动项"""
    content: str = Field(min_length=1, max_length=50000)
    source: str = Field(default="组会纪要", max_length=240)


class ResearchTaskItem(BaseModel):
    """项目待办"""
    id: str
    project_id: str
    title: str
    assignee: str
    due_date: str
    status: str
    source: str
    created_by: str
    created_at: float
    updated_at: float


class ResearchTaskListResponse(BaseModel):
    """项目待办列表"""
    tasks: List[ResearchTaskItem] = Field(default_factory=list)
    total: int


class ResearchOverviewResponse(BaseModel):
    """科研工作区概览"""
    projects: int
    experiments: int
    open_tasks: int
    members: int
    active_projects: int
    by_status: dict[str, int] = Field(default_factory=dict)


class ResearchRunSummaryItem(BaseModel):
    """研究运行列表项；不包含大体积阶段 payload。"""
    id: str
    project_id: Optional[str] = None
    session_id: str
    user_id: str
    question: str
    mode: str
    status: str
    final_answer: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: float
    completed_at: Optional[float] = None


class ResearchRunListResponse(BaseModel):
    runs: List[ResearchRunSummaryItem] = Field(default_factory=list)
    total: int


class ResearchRunDetailItem(ResearchRunSummaryItem):
    """结构化运行详情；服务层会按当前文档 ACL 重新过滤。"""
    source_cards: List[dict[str, Any]] = Field(default_factory=list)
    evidence_package: dict[str, Any] = Field(default_factory=dict)
    analysis_report: dict[str, Any] = Field(default_factory=dict)
    review_report: dict[str, Any] = Field(default_factory=dict)
    research_trace: dict[str, Any] = Field(default_factory=dict)
    hidden_evidence_count: int = 0
    confirmed_claim_ids: List[str] = Field(default_factory=list)
    published_claim_ids: List[str] = Field(default_factory=list)
    published_claim_statuses: dict[str, str] = Field(default_factory=dict)


class ConfirmResearchClaimResponse(BaseModel):
    """用户显式确认、且已通过证据与 Reviewer 门槛的长期记忆。"""
    stored: bool
    run_id: str
    claim_id: str
    text: str
    source_titles: List[str] = Field(default_factory=list)


class PublishKnowledgeRequest(BaseModel):
    """独立发布动作；知识正文与来源必须由服务端从 Research Run 读取。"""


class SupersedeKnowledgeRequest(BaseModel):
    """用另一次 Research Run 中已复核的 Claim 替代当前知识。"""
    run_id: str = Field(min_length=1, max_length=64)
    claim_id: str = Field(min_length=1, max_length=64)


class KnowledgeRecordItem(BaseModel):
    """可追溯的项目知识记录。"""
    id: str
    project_id: str
    knowledge_type: Literal["fact"]
    statement: str
    status: Literal["active", "superseded", "revoked"]
    version: int
    research_run_id: str
    claim_id: str
    source_ids: List[str] = Field(default_factory=list)
    created_by: str
    published_by: str
    created_at: float
    updated_at: float
    supersedes_id: Optional[str] = None
    research_question: str = ""
    sources: List[dict[str, str]] = Field(default_factory=list)


class KnowledgeRecordListResponse(BaseModel):
    records: List[KnowledgeRecordItem] = Field(default_factory=list)
    total: int


# ==================== Response Models ====================
class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(description="AI回复")
    sources: List[SourceItem] = Field(default_factory=list, description="结构化信息来源")
    session_id: str = Field(description="会话ID")
    used_agent: str = Field(description="使用的Agent类型")
    image_understood: bool = Field(default=False, description="是否对图片进行了理解")
    research_run_id: Optional[str] = Field(default=None, description="Deep Research 运行记录 ID")


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[dict] = Field(default_factory=list, description="搜索结果")
    total: int = Field(description="结果总数")


class KnowledgeDocumentItem(BaseModel):
    """资料中心中的单份文档"""
    source: str = Field(description="来源文件名")
    title: str = Field(description="资料标题")
    doc_type: str = Field(default="general", description="文档类型")
    doc_type_label: str = Field(default="通用资料", description="文档类型展示名")
    author: Optional[str] = Field(default=None, description="作者")
    project_name: Optional[str] = Field(default=None, description="所属项目")
    research_direction: Optional[str] = Field(default=None, description="研究方向")
    visibility: str = Field(default="public", description="可见范围")
    created_at: Optional[str] = Field(default=None, description="资料时间")
    summary: Optional[str] = Field(default=None, description="资料简介")
    chunk_count: int = Field(default=0, description="文档块数量")


class KnowledgeDocumentListResponse(BaseModel):
    """资料目录响应"""
    documents: List[KnowledgeDocumentItem] = Field(default_factory=list, description="资料列表")
    total: int = Field(description="资料总数")


class KnowledgeOverviewResponse(BaseModel):
    """资料中心概览"""
    documents: int = Field(description="资料总数")
    chunks: int = Field(description="文档块总数")
    projects: int = Field(description="项目数量")
    public_documents: int = Field(description="公共资料数量")
    restricted_documents: int = Field(description="受限资料数量")
    by_doc_type: dict[str, int] = Field(default_factory=dict, description="按类型统计")
    by_visibility: dict[str, int] = Field(default_factory=dict, description="按可见范围统计")


class IngestionJobItem(BaseModel):
    """异步入库任务"""
    job_id: str = Field(description="任务 ID")
    filename: str = Field(description="原始文件名")
    category: str = Field(description="资料分类")
    doc_type: str = Field(description="资料类型")
    status: Literal["pending", "running", "completed", "failed", "retrying"] = Field(description="任务状态")
    retry_count: int = Field(default=0, description="已重试次数")
    max_retries: int = Field(default=3, description="最大重试次数")
    error: Optional[str] = Field(default=None, description="失败原因")
    created_at: float = Field(description="创建时间戳")
    started_at: Optional[float] = Field(default=None, description="开始时间戳")
    completed_at: Optional[float] = Field(default=None, description="完成时间戳")
    result: dict = Field(default_factory=dict, description="处理结果")
    file_hash: Optional[str] = Field(default=None, description="SHA-256 文件哈希")


class IngestionJobListResponse(BaseModel):
    """异步入库任务列表"""
    jobs: List[IngestionJobItem] = Field(default_factory=list, description="任务列表")
    total: int = Field(description="任务数量")
    stats: dict[str, int] = Field(default_factory=dict, description="各状态数量")


class IngestionSubmitResponse(BaseModel):
    """文件提交入库响应"""
    message: str = Field(description="结果信息")
    job: IngestionJobItem = Field(description="入库任务")
    duplicate: bool = Field(default=False, description="是否命中相同文件去重")


class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str = Field(description="访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    user_info: Optional[dict] = Field(
        default=None,
        description="当前用户信息：username, role, department, department_name, role_display_name"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(description="服务状态")
    service: str = Field(description="服务名称")
    version: str = Field(description="版本号")


__all__ = [
    "ImageContent",
    "ChatRequest",
    "ChatStreamRequest",
    "CreateSessionRequest",
    "UpdateTitleRequest",
    "AddDocumentRequest",
    "DocumentUploadMetadata",
    "SearchRequest",
    "SourceItem",
    "LoginRequest",
    "RegisterRequest",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackStatsResponse",
    "FeedbackIssueItem",
    "FeedbackIssueListResponse",
    "ProjectMemberItem",
    "CreateProjectRequest",
    "ProjectItem",
    "ProjectListResponse",
    "CreateExperimentRequest",
    "ExperimentItem",
    "ExperimentListResponse",
    "CreateResearchTaskRequest",
    "UpdateResearchTaskRequest",
    "ExtractMeetingTasksRequest",
    "ResearchTaskItem",
    "ResearchTaskListResponse",
    "ResearchOverviewResponse",
    "PublishKnowledgeRequest",
    "SupersedeKnowledgeRequest",
    "KnowledgeRecordItem",
    "KnowledgeRecordListResponse",
    "ChatResponse",
    "SearchResponse",
    "KnowledgeDocumentItem",
    "KnowledgeDocumentListResponse",
    "KnowledgeOverviewResponse",
    "IngestionJobItem",
    "IngestionJobListResponse",
    "IngestionSubmitResponse",
    "LoginResponse",
    "RegisterResponse",
    "HealthResponse",
]
