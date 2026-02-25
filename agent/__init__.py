"""
Agent 模块初始化
重构后的 Multi-Agent 架构
"""

# 导出工具
from agent.tools import (
    get_knowledge_tool,
    get_calculator_tool,
    get_datetime_tool,
    get_base_tools,
    get_all_agent_tools,
    get_tool_by_name,
)

# 导出 MCP 适配器
from agent.tools.mcp_adapter import (
    convert_mcp_tools,
    get_mcp_tools_as_langchain,
)

__all__ = [
    # 工具
    "get_knowledge_tool",
    "get_calculator_tool", 
    "get_datetime_tool",
    "get_base_tools",
    "get_all_agent_tools",
    "get_tool_by_name",
    # MCP 适配器
    "convert_mcp_tools",
    "get_mcp_tools_as_langchain",
]
