"""
MCP 工具适配器
将 MCP 工具转换为 LangChain BaseTool 格式
使用异步 coroutine 实现，优化性能
"""
from typing import List, Any, Optional, Type
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import BaseTool


def convert_mcp_tools(mcp_tools: List[Any]) -> List[BaseTool]:
    """
    将 MCP 工具转换为 LangChain BaseTool 格式
    
    Args:
        mcp_tools: MCP 工具列表
        
    Returns:
        LangChain BaseTool 列表
    """
    langchain_tools = []
    
    for mcp_tool in mcp_tools:
        try:
            langchain_tool = convert_single_mcp_tool(mcp_tool)
            if langchain_tool:
                langchain_tools.append(langchain_tool)
        except Exception as e:
            print(f"转换 MCP 工具 {getattr(mcp_tool, 'name', 'unknown')} 失败: {e}")
    
    return langchain_tools


def convert_single_mcp_tool(mcp_tool: Any) -> Optional[BaseTool]:
    """
    将单个 MCP 工具转换为 LangChain BaseTool
    
    Args:
        mcp_tool: MCP 工具对象
        
    Returns:
        LangChain BaseTool 或 None
    """
    from langchain_core.tools import StructuredTool
    
    # 获取工具名称和描述
    tool_name = getattr(mcp_tool, 'name', None)
    tool_description = getattr(mcp_tool, 'description', 'MCP 工具')
    
    # 增强工具描述，避免 LLM 选错工具
    if tool_name == 'list_directory':
        tool_description = "列出指定目录下的所有文件和子目录（类似 ls 命令）。参数：path（必填）：要列出的目录路径，例如 './data/knowledge'"
    elif tool_name == 'list_allowed_directories':
        tool_description = "查询系统允许访问的目录白名单列表（不是列出目录内容）。无参数。用于检查哪些目录可以被访问。"
    elif tool_name == 'read_file':
        tool_description = "读取单个文件的内容（文本文件）。参数：path（必填）：文件的完整路径"
    elif tool_name == 'read_text_file':
        tool_description = "读取文本文件内容（功能同 read_file）。参数：path（必填）：文件的完整路径"
    elif tool_name == 'write_file':
        tool_description = "写入内容到文件（会覆盖原文件）。参数：path（必填）：文件路径，content（必填）：要写入的内容"
    elif tool_name == 'create_directory':
        tool_description = "创建新目录。参数：path（必填）：要创建的目录路径"
    elif tool_name == 'search_files':
        tool_description = "在目录中搜索文件名包含关键词的文件。参数：path（必填）：搜索目录，pattern（必填）：搜索关键词"
    
    if not tool_name:
        return None
    
    # 预先保存工具名称供内部函数使用
    _tool_name = tool_name
    
    async def execute_tool_async(**kwargs) -> str:
        """执行 MCP 工具（异步实现）"""
        from core.mcp_client import mcp_manager
        
        try:
            tool_manager = mcp_manager.tool_manager
            if tool_manager is None:
                raise RuntimeError("MCP 工具管理器未初始化")
            
            # 查找工具所在的服务器
            server_name = None
            for s_name, client in tool_manager._mcp_servers.items():
                for t in client.get_tools():
                    if t.name == _tool_name:
                        server_name = s_name
                        break
                if server_name:
                    break
            
            if not server_name:
                raise RuntimeError(f"未找到工具 {_tool_name} 所在的服务器")
            
            result = await tool_manager.call_mcp_tool(server_name, _tool_name, kwargs)
            return format_mcp_result(result)
            
        except Exception as e:
            return f"MCP 工具执行错误: {str(e)}"
    
    # 获取输入模式用于描述参数
    input_schema = getattr(mcp_tool, 'inputSchema', {})
    if isinstance(input_schema, dict):
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])
    else:
        properties = {}
        required = []
    
    # 构建参数描述
    args_schema = create_pydantic_model(tool_name, properties, required)
    
    try:
        return StructuredTool.from_function(
            func=None,
            coroutine=execute_tool_async,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
        )
    except Exception as e:
        print(f"使用 StructuredTool 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_pydantic_model(
    tool_name: str,
    properties: dict,
    required: List[str]
) -> Type[BaseModel]:
    """
    根据 MCP 工具的输入模式创建 Pydantic 模型
    
    Args:
        tool_name: 工具名称
        properties: 属性定义
        required: 必填字段列表
        
    Returns:
        Pydantic 模型类
    """
    field_definitions = {}
    
    for prop_name, prop_info in properties.items():
        # 获取字段类型
        prop_type = str
        if prop_info.get('type') == 'integer':
            prop_type = int
        elif prop_info.get('type') == 'number':
            prop_type = float
        elif prop_info.get('type') == 'boolean':
            prop_type = bool
        
        # 获取描述
        description = prop_info.get('description', '')
        
        # 获取默认值
        default = ...
        if prop_name not in required:
            if 'default' in prop_info:
                default = prop_info['default']
            else:
                default = None
        
        # 创建字段
        field_definitions[prop_name] = (prop_type, Field(description=description, default=default))
    
    # 如果没有属性，添加一个任意类型字段
    if not field_definitions:
        field_definitions['input'] = (str, Field(default="", description="输入参数"))
    
    model_name = f"{tool_name.replace('-', '_').title()}Input"
    return create_model(model_name, **field_definitions)


def format_mcp_result(result: Any) -> str:
    """格式化 MCP 工具返回结果"""
    if result is None:
        return "工具执行完成，无返回结果"
    
    if isinstance(result, str):
        return result
    
    if isinstance(result, dict):
        if 'content' in result:
            contents = result['content']
            if isinstance(contents, list):
                return "\n".join([str(c) for c in contents])
            return str(contents)
        return str(result)
    
    if isinstance(result, list):
        return "\n".join([str(item) for item in result])
    
    return str(result)


# ==================== 便捷函数 ====================

def get_mcp_tools_as_langchain() -> List[BaseTool]:
    """
    获取所有 MCP 工具作为 LangChain BaseTool
    
    Returns:
        LangChain BaseTool 列表
    """
    from core.mcp_client import mcp_manager
    
    try:
        mcp_tools = mcp_manager.get_tools()
        return convert_mcp_tools(mcp_tools)
    except Exception as e:
        print(f"获取 MCP 工具失败: {e}")
        return []
