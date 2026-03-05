"""
File Operation Skill Tools - 文件操作工具
通过 MCP 获取文件系统工具
"""
from typing import List
from langchain_core.tools import BaseTool


def get_mcp_tools_by_names(tool_names: List[str]) -> List[BaseTool]:
    """
    根据名称从 MCP 获取特定工具

    Args:
        tool_names: 工具名称列表

    Returns:
        LangChain BaseTool 列表
    """
    from ...tools.mcp_adapter import convert_single_mcp_tool
    from src.models.mcp_client import mcp_manager
    
    tools = []
    try:
        all_mcp_tools = mcp_manager.get_tools()
        for mcp_tool in all_mcp_tools:
            if mcp_tool.name in tool_names:
                langchain_tool = convert_single_mcp_tool(mcp_tool)
                if langchain_tool:
                    tools.append(langchain_tool)
    except Exception as e:
        print(f"获取 MCP 工具失败: {e}")
    
    return tools


def list_directory(path: str) -> str:
    """
    列出目录内容
    
    Args:
        path: 目录路径
    
    Returns:
        目录内容
    """
    tools = get_mcp_tools_by_names(["list_directory"])
    if tools:
        return tools[0].invoke(path)
    return "MCP 文件系统工具未初始化"


def read_file(path: str) -> str:
    """
    读取文件内容
    
    Args:
        path: 文件路径
    
    Returns:
        文件内容
    """
    tools = get_mcp_tools_by_names(["read_file"])
    if tools:
        return tools[0].invoke(path)
    return "MCP 文件系统工具未初始化"


__all__ = ["get_mcp_tools_by_names", "list_directory", "read_file"]

