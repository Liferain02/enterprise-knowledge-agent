"""
Agent 状态定义模块
定义 LangGraph Agent 的状态结构
"""
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator


# LangGraph 0.2+ 使用 TypedDict 作为状态
# 这里定义状态结构
class AgentState(BaseModel):
    """Agent 状态类 - 用于 LangGraph (兼容版本)"""
    
    model_config = {"arbitrary_types_allowed": True}
    
    # 用户输入
    input: str = Field(default="", description="用户输入的问题")
    
    # 会话ID
    session_id: str = Field(default="default", description="会话ID")
    
    # 对话历史
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="对话消息历史"
    )
    
    # 规划结果
    plan: Optional[str] = Field(
        default=None,
        description="任务规划结果"
    )
    
    # 工具选择
    selected_tools: List[str] = Field(
        default_factory=list,
        description="选中的工具列表"
    )
    
    # 工具执行结果
    tool_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具执行结果"
    )
    
    # 上下文信息（用于RAG）
    context: str = Field(
        default="",
        description="检索到的上下文信息"
    )
    
    # 最终答案
    final_answer: Optional[str] = Field(
        default=None,
        description="最终答案"
    )
    
    # 迭代次数
    iteration: int = Field(
        default=0,
        description="当前迭代次数"
    )
    
    # 是否完成
    is_done: bool = Field(
        default=False,
        description="是否已完成"
    )
    
    # 错误信息
    error: Optional[str] = Field(
        default=None,
        description="错误信息"
    )
    
    # ReAct 追踪
    thought_history: List[str] = Field(
        default_factory=list,
        description="思考历史"
    )
    
    action_history: List[str] = Field(
        default_factory=list,
        description="行动历史"
    )
    
    observation_history: List[str] = Field(
        default_factory=list,
        description="观察历史"
    )


class ChatMessage(BaseModel):
    """聊天消息模型"""
    
    role: str = Field(description="消息角色: user, assistant, system")
    content: str = Field(description="消息内容")
    timestamp: Optional[str] = Field(default=None, description="时间戳")


class AgentRequest(BaseModel):
    """Agent 请求模型"""
    
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default", description="会话ID")
    use_rag: bool = Field(default=True, description="是否使用RAG")
    stream: bool = Field(default=False, description="是否流式输出")
    max_iterations: int = Field(default=10, description="最大迭代次数")


class AgentResponse(BaseModel):
    """Agent 响应模型"""
    
    answer: str = Field(description="Agent生成的答案")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="信息来源"
    )
    session_id: str = Field(description="会话ID")
    iterations: int = Field(description="使用的迭代次数")
    tools_used: List[str] = Field(
        default_factory=list,
        description="使用的工具"
    )
    context_used: bool = Field(description="是否使用了上下文")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="其他元数据"
    )


class ToolCall(BaseModel):
    """工具调用模型"""
    
    tool_name: str = Field(description="工具名称")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具参数"
    )
    result: Optional[str] = Field(
        default=None,
        description="执行结果"
    )
    error: Optional[str] = Field(
        default=None,
        description="错误信息"
    )

