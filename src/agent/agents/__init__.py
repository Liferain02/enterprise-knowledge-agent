"""Agent 节点兼容导出。

``supervisor_node`` 仅为旧外部导入保留，当前生产图不注册 Supervisor。
"""
from langgraph.graph import MessagesState

from .supervisor import supervisor_node
from .knowledge import knowledge_agent_node
from .operation import operation_agent_node
from .general import general_agent_node
from ._utils import get_last_user_message


class AgentState(MessagesState):
    """Agent 状态定义"""
    next_agent: str
    supervisor_reasoning: str
    supervisor_reason: str
    final_answer: str
    sources: str
    used_agent: str
    session_id: str


__all__ = [
    "supervisor_node",
    "knowledge_agent_node",
    "operation_agent_node",
    "general_agent_node",
    "AgentState",
    "get_last_user_message",
]
