"""
A2A 协议支持 - Agent Card 暴露

基于 AACP (Agent-to-Agent Communication Protocol) 草案规范。
提供 /well-known/agent.json 端点，支持框架互操作和能力发现。
"""
import asyncio
import hashlib
import time
from typing import Optional, Any, Literal, TypedDict
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from config.settings import get_settings


# ==================== Agent Card 数据模型 ====================

class AgentSkill(BaseModel):
    """Agent 技能定义"""
    id: str = Field(description="技能唯一标识")
    name: str = Field(description="技能名称")
    description: str = Field(description="技能描述")


class AgentCapability(BaseModel):
    """Agent 能力定义"""
    streaming: bool = Field(default=True, description="是否支持流式输出")
    pushNotifications: bool = Field(default=False, description="是否支持服务端推送")
    stateTransitionReports: bool = Field(
        default=False,
        description="是否报告状态转换（如任务完成）"
    )
    careAboutReceivedH: bool = Field(
        default=False,
        description="是否关心消息头部的 H（血腥/暴力/色情）标记"
    )


class AgentProvider(BaseModel):
    """服务提供方信息"""
    organization: str = Field(description="组织名称")
    version: str = Field(description="服务版本")


class Security(BaseModel):
    """安全配置"""
    schemes: list[str] = Field(
        default_factory=lambda: ["bearer"],
        description="支持的认证方案"
    )
    credentials: Optional[str] = Field(
        default=None,
        description="凭证要求描述"
    )


class AuthenticationVerify(BaseModel):
    """认证配置"""
    schemes: list[str] = Field(
        default_factory=lambda: ["bearer"],
        description="支持的认证方案"
    )


class AgentCard(BaseModel):
    """
    A2A Agent Card

    暴露本 Agent 的能力、端点、版本等信息，供其他 Agent 框架发现和交互。
    """
    name: str = Field(description="Agent 名称")
    description: str = Field(description="Agent 功能描述")
    url: str = Field(description="Agent 服务基础 URL")
    version: str = Field(description="Agent 版本")

    # 能力声明
    capabilities: AgentCapability = Field(
        default_factory=AgentCapability,
        description="Agent 能力"
    )

    # 支持的技能
    skills: list[AgentSkill] = Field(
        default_factory=list,
        description="Agent 支持的技能列表"
    )

    # 服务提供方
    provider: AgentProvider = Field(
        default_factory=lambda: AgentProvider(
            organization="Enterprise Knowledge Base Team",
            version="1.0.0"
        ),
        description="服务提供方"
    )

    # 认证信息
    authentication: AuthenticationVerify = Field(
        default_factory=AuthenticationVerify,
        description="认证配置"
    )

    # 端点信息
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="各端点路径映射"
    )

    # 元数据
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="额外元数据"
    )

    # 唯一标识（用于去重）
    agent_id: str = Field(description="Agent 唯一标识（UUID）")

    # 认证 scheme（与 authentication.schemes 一致）
    authentication_schemes_supported: list[str] = Field(
        default_factory=lambda: ["bearer"],
        description="支持的认证方案"
    )


# ==================== 全局 Agent Card 实例 ====================

def _build_agent_card() -> AgentCard:
    """构建本 Agent 的 Card"""
    settings = get_settings()
    base_url = f"http://{settings.api_host}:{settings.api_port}"

    # 计算 Agent ID（基于版本和名称的哈希）
    agent_id_input = f"lab-knowledge-assistant-v1.0.0"
    agent_id = hashlib.sha256(agent_id_input.encode()).hexdigest()[:16]

    # 定义支持的技能
    skills = [
        AgentSkill(
            id="knowledge_retrieval",
            name="实验室知识库检索",
            description=(
                "从实验室知识库检索制度、项目资料、论文笔记、组会纪要和流程说明。"
                "支持：入组导览、组会制度、环境配置、报销采购、RDMA/NUMA 相关资料等。"
            )
        ),
        AgentSkill(
            id="lab_onboarding_query",
            name="实验室导览与制度查询",
            description=(
                "查询新生入组流程、实验室考勤、组会规范、资源预约和公共流程资料。"
            )
        ),
        AgentSkill(
            id="operation_execution",
            name="操作执行",
            description=(
                "执行计算、时间查询、文件操作等工具调用任务。"
            )
        ),
        AgentSkill(
            id="general_conversation",
            name="通用对话",
            description=(
                "处理问候、寒暄、一般性闲聊。"
                "当用户问题不涉及具体知识库检索时，由通用 Agent 处理。"
            )
        ),
        AgentSkill(
            id="multi_step_planning",
            name="多步骤规划",
            description=(
                "处理复杂任务，自动拆解为多个子步骤并行或顺序执行。"
                "例如：对比两篇论文方案，或先查组会要求再整理汇报清单。"
            )
        ),
    ]

    # 端点映射
    endpoints = {
        "chat": "/api/v1/chat",
        "chat_stream": "/api/v1/chat/stream",
        "knowledge_search": "/api/v1/knowledge/search",
        "session_list": "/api/v1/sessions",
        "health": "/health",
        "metrics": "/metrics",
        "agent_card": "/.well-known/agent.json",
    }

    # 元数据
    metadata = {
        "supported_languages": ["zh-CN", "en-US"],
        "max_context_tokens": 128000,
        "embedding_model": getattr(settings, "embedding_model", "text-embedding-3-small"),
        "llm_model": getattr(settings, "llm_model", "qwen-plus"),
        "vector_store": "chroma",
        "features": [
            "corrective_rag",
            "mem0_memory",
            "acl_permission",
            "document_versioning",
            "conflict_detection",
        ],
    }

    return AgentCard(
        name="实验室智能助手",
        description=(
            "面向科研实验室的资料检索与协作助手。"
            "支持知识库检索、多步骤规划、Mem0 记忆、权限隔离与来源引用。"
        ),
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapability(
            streaming=True,
            pushNotifications=False,
            stateTransitionReports=False,
            careAboutReceivedH=False,
        ),
        skills=skills,
        provider=AgentProvider(
            organization="Research Lab Assistant Team",
            version="1.0.0"
        ),
        authentication=AuthenticationVerify(
            schemes=["bearer"]
        ),
        endpoints=endpoints,
        metadata=metadata,
        agent_id=agent_id,
        authentication_schemes_supported=["bearer"],
    )


# 全局 Agent Card（延迟构建）
_agent_card: Optional[AgentCard] = None


def get_agent_card() -> AgentCard:
    """获取全局 Agent Card 实例"""
    global _agent_card
    if _agent_card is None:
        _agent_card = _build_agent_card()
    return _agent_card


# ==================== A2A 任务消息协议 ====================

class MessagePartText(BaseModel):
    """文本消息片段"""
    type: str = "text"
    text: str


class MessagePart(BaseModel):
    """消息片段（支持多种类型）"""
    type: str = Field(description="片段类型：text/image/...")
    text: Optional[str] = Field(default=None, description="文本内容")
    mime_type: Optional[str] = Field(default=None, description="MIME 类型")
    data: Optional[str] = Field(default=None, description="数据内容（base64）")


class A2AMessage(BaseModel):
    """A2A 任务消息"""
    role: Literal["user", "agent", "assistant", "system"] = Field(
        description="消息角色"
    )
    parts: list[MessagePart] = Field(description="消息片段列表")
    message_id: Optional[str] = Field(default=None, description="消息 ID")


class TaskStatus(BaseModel):
    """任务状态"""
    state: Literal[
        "submitted", "working", "input-required",
        "completed", "failed", "canceled"
    ] = Field(description="任务状态")
    timestamp: int = Field(
        default_factory=lambda: int(time.time()),
        description="状态变更时间戳"
    )
    message: Optional[str] = Field(default=None, description="状态说明")


# ==================== A2A 任务卡片（供外部 Agent 查询） ====================

class AgentTask(BaseModel):
    """A2A 任务定义"""
    task_id: str = Field(description="任务 ID（UUID）")
    session_id: str = Field(description="会话 ID")
    status: TaskStatus = Field(description="任务状态")
    messages: list[A2AMessage] = Field(
        default_factory=list,
        description="消息历史"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="任务元数据"
    )


# ==================== API 路由 ====================

# 全局 Agent Card 路由
_a2a_router = APIRouter(prefix="", tags=["A2A Protocol"])


@_a2a_router.get("/.well-known/agent.json", response_model=AgentCard)
async def get_agent_card_handler(request: Request):
    """
    获取本 Agent 的 Agent Card

    遵循 AACP (Agent-to-Agent Communication Protocol) 草案规范。
    其他 Agent 框架可通过此端点发现本 Agent 的能力和端点。

    返回内容示例：
    {
        "name": "实验室智能助手",
        "description": "面向科研实验室的资料检索与协作系统...",
        "url": "http://localhost:8080",
        "version": "1.0.0",
        "capabilities": {
            "streaming": true,
            "pushNotifications": false,
            ...
        },
        "skills": [
            {"id": "knowledge_retrieval", "name": "实验室知识库检索", ...},
            ...
        ],
        "endpoints": {
            "chat": "/api/v1/chat",
            "agent_card": "/.well-known/agent.json",
            ...
        }
    }
    """
    card = get_agent_card()
    return card


@_a2a_router.get("/.well-known/agent.jsonl")
async def get_agent_card_jsonl(request: Request):
    """
    获取 Agent Card 的 JSONL 格式（便于脚本解析）
    """
    import json
    card = get_agent_card()
    card_dict = card.model_dump(mode="json")
    return f"data: {json.dumps(card_dict, ensure_ascii=False)}\n\n"


@_a2a_router.post("/api/v1/a2a/send", response_model=dict)
async def a2a_send_message(
    message: A2AMessage,
    session_id: Optional[str] = None,
    request: Request = None,
):
    """
    A2A 消息发送接口

    供外部 Agent 向本 Agent 发送消息。
    消息会路由到对应的 Agent 节点处理。

    注意：此接口为简化实现，完整 A2A 协议包含：
    - 任务生命周期管理（Task resource）
    - 流式推送（Server-Sent Events）
    - 任务取消
    - 技能选择
    """
    from src.api.services.chat_service import chat_service

    # 从消息中提取文本内容
    text_parts = []
    for part in message.parts:
        if part.type == "text" and part.text:
            text_parts.append(part.text)

    if not text_parts:
        raise HTTPException(status_code=400, detail="消息内容为空")

    user_message = "\n".join(text_parts)

    # 使用 session_id（若未提供则生成）
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())

    # 调用 Chat Service
    result = await chat_service.achat(
        message=user_message,
        session_id=session_id,
        username="a2a_agent",
    )

    # 构建响应
    answer = result.get("answer", "")

    return {
        "success": True,
        "session_id": session_id,
        "message_id": message.message_id or str(hashlib.md5(
            f"{user_message}{time.time()}".encode()
        ).hexdigest()[:16]),
        "role": "assistant",
        "parts": [{"type": "text", "text": answer}],
        "used_agent": result.get("used_agent", "unknown"),
    }


@_a2a_router.get("/api/v1/a2a/tasks/{task_id}")
async def a2a_get_task(task_id: str):
    """
    获取 A2A 任务状态

    简化实现：返回任务信息。
    完整 A2A 协议中，任务是一个独立的资源，有完整的状态机。
    """
    # 简化实现：直接从 session_service 查询
    from src.api.services.session_service import session_service

    # task_id 映射到 session_id
    session = session_service.get_session(task_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    messages = session_service.get_messages(task_id)
    return {
        "task_id": task_id,
        "status": "completed",
        "messages": messages,
        "metadata": session,
    }


@_a2a_router.get("/api/v1/a2a/skills")
async def a2a_list_skills():
    """
    列出本 Agent 支持的所有技能

    便于外部 Agent 进行能力发现和技能匹配。
    """
    card = get_agent_card()
    return {
        "skills": [
            {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "matches": [
                    # 返回与该技能相关的 Agent 路由
                    endpoint
                    for endpoint, path in card.endpoints.items()
                    if _skill_matches_endpoint(skill.id, endpoint)
                ]
            }
            for skill in card.skills
        ],
        "total": len(card.skills),
    }


def _skill_matches_endpoint(skill_id: str, endpoint: str) -> bool:
    """判断技能是否与某端点相关"""
    mapping = {
        "knowledge_retrieval": ["chat", "knowledge_search"],
        "lab_onboarding_query": ["chat", "knowledge_search"],
        "operation_execution": ["chat"],
        "general_conversation": ["chat"],
        "multi_step_planning": ["chat"],
    }
    return endpoint in mapping.get(skill_id, [])
