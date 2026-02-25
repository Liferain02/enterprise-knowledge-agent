"""
MCP 工具适配器
将 MCP 工具转换为 LangChain BaseTool 格式
"""
from typing import List, Any, Optional, Type
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun


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
    # 获取工具名称和描述
    tool_name = getattr(mcp_tool, 'name', None)
    tool_description = getattr(mcp_tool, 'description', 'MCP 工具')
    
    if not tool_name:
        return None
    
    # 获取输入模式
    input_schema = getattr(mcp_tool, 'inputSchema', {})
    if isinstance(input_schema, dict):
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])
    else:
        properties = {}
        required = []
    
    # 创建 Pydantic 模型
    args_schema = create_pydantic_model(tool_name, properties, required)
    
    # 创建工具类
    class MCPToolWrapper(BaseTool):
        """MCP 工具包装器"""
        
        name: str = tool_name
        description: str = tool_description
        args_schema: Type[BaseModel] = args_schema
        _mcp_tool: Any = mcp_tool
        
        def _run(
            self,
            **kwargs
        ) -> str:
            """执行 MCP 工具（同步版本）"""
            import asyncio
            from core.mcp_client import mcp_manager
            
            try:
                # 在新事件循环中运行异步调用
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self._async_call_tool(kwargs)
                    )
                    return format_mcp_result(result)
                finally:
                    loop.close()
            except Exception as e:
                return f"MCP 工具执行错误: {str(e)}"
        
        async def _async_call_tool(self, arguments: dict) -> Any:
            """异步调用 MCP 工具"""
            from core.mcp_client import mcp_manager
            
            # 查找工具所在的服务器
            tool_manager = mcp_manager.tool_manager
            if tool_manager is None:
                raise RuntimeError("MCP 工具管理器未初始化")
            
            # 查找工具所在的服务器
            server_name = None
            for s_name, client in tool_manager._mcp_servers.items():
                for t in client.get_tools():
                    if t.name == tool_name:
                        server_name = s_name
                        break
                if server_name:
                    break
            
            if not server_name:
                raise RuntimeError(f"未找到工具 {tool_name} 所在的服务器")
            
            return await tool_manager.call_mcp_tool(server_name, tool_name, arguments)
    
    return MCPToolWrapper()


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
        field_definitions['__input'] = (str, Field(default=""))
    
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

