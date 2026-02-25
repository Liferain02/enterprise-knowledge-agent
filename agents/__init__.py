"""
Agents 模块初始化
Multi-Agent 架构
"""

# 导出节点
from agents.nodes.supervisor import supervisor_node, route_to_agent
from agents.nodes.knowledge import knowledge_agent_node
from agents.nodes.operation import operation_agent_node
from agents.nodes.general import general_agent_node

# 导出图
from agents.graph import get_agent_graph, run_agent, run_agent_stream

# 导出提示词
from agents.prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    KNOWLEDGE_AGENT_SYSTEM_PROMPT,
    OPERATION_AGENT_SYSTEM_PROMPT,
    GENERAL_AGENT_SYSTEM_PROMPT,
)

__all__ = [
    # 节点
    "supervisor_node",
    "route_to_agent",
    "knowledge_agent_node",
    "operation_agent_node",
    "general_agent_node",
    # 图
    "get_agent_graph",
    "run_agent",
    "run_agent_stream",
    # 提示词
    "SUPERVISOR_SYSTEM_PROMPT",
    "KNOWLEDGE_AGENT_SYSTEM_PROMPT",
    "OPERATION_AGENT_SYSTEM_PROMPT",
    "GENERAL_AGENT_SYSTEM_PROMPT",
]
