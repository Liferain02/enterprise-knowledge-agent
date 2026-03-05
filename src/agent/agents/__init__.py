"""
Agent 节点模块
"""
from typing import List, Optional
from langchain_core.messages import HumanMessage
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


def get_last_user_message(messages: List) -> Optional[str]:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


__all__ = [
    "supervisor_node",
    "knowledge_agent_node",
    "operation_agent_node",
    "general_agent_node",
    "AgentState",
    "get_last_user_message",
]
