"""
MCP 工具适配器
将 MCP 工具转换为 LangChain BaseTool 格式
使用异步 coroutine 实现，优化性能
"""
from typing import List, Any, Optional, Type, Dict, Union
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import BaseTool


# ==================== 类型化 Schema 注册表 ====================

# 预定义的 MCP 工具 Schema，增强参数校验可靠性
# 每个条目定义：工具名 → (Pydantic 模型类, 增强描述)
_MCP_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "list_directory": {
        "properties": {
            "path": {
                "type": "string",
                "description": "要列出的目录绝对路径或相对路径，例如 './data/knowledge' 或 '/home/user/docs'。必须是已存在的目录。",
                "default": "."
            }
        },
        "required": ["path"],
        "enhanced_description": (
            "列出指定目录下的所有文件和子目录（类似 ls -la 命令）。"
            "参数：path（必填）：要列出的目录路径，必须是已存在的目录。"
        ),
    },
    "read_file": {
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文件完整路径（绝对路径或相对于当前工作目录）。必须是文件，不能是目录。",
            },
            "max_lines": {
                "type": "integer",
                "description": "最多读取的行数（可选），用于大文件的部分读取。",
                "default": None,
            }
        },
        "required": ["path"],
        "enhanced_description": (
            "读取单个文件的内容（文本文件）。参数：path（必填）：文件的完整路径。"
            "可选 max_lines 参数限制读取行数。"
        ),
    },
    "read_text_file": {
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文本文件完整路径。",
            }
        },
        "required": ["path"],
        "enhanced_description": (
            "读取文本文件内容（功能同 read_file）。参数：path（必填）：文件的完整路径。"
        ),
    },
    "write_file": {
        "properties": {
            "path": {
                "type": "string",
                "description": "要写入的文件路径。如果文件已存在会被覆盖，请谨慎使用。",
            },
            "content": {
                "type": "string",
                "description": "要写入的文件内容。可以是多行文本。",
            },
            "append": {
                "type": "boolean",
                "description": "是否追加模式（True）或覆盖模式（False，默认）。",
                "default": False,
            }
        },
        "required": ["path", "content"],
        "enhanced_description": (
            "写入内容到文件（默认覆盖原文件）。"
            "参数：path（必填）：文件路径；content（必填）：要写入的内容；"
            "append（可选）：True=追加，False=覆盖（默认）。"
        ),
    },
    "create_directory": {
        "properties": {
            "path": {
                "type": "string",
                "description": "要创建的新目录路径。如果父目录不存在会自动创建中间目录。",
            }
        },
        "required": ["path"],
        "enhanced_description": (
            "创建新目录（类似 mkdir -p，会自动创建父目录）。"
            "参数：path（必填）：要创建的目录路径。"
        ),
    },
    "search_files": {
        "properties": {
            "path": {
                "type": "string",
                "description": "搜索的起始目录路径。",
            },
            "pattern": {
                "type": "string",
                "description": "搜索关键词（文件名中包含此关键词）。支持简单字符串匹配。",
            },
            "recursive": {
                "type": "boolean",
                "description": "是否递归搜索子目录（默认 True）。",
                "default": True,
            }
        },
        "required": ["path", "pattern"],
        "enhanced_description": (
            "在目录中搜索文件名包含关键词的文件。"
            "参数：path（必填）：搜索目录；pattern（必填）：文件名关键词；"
            "recursive（可选）：是否递归子目录（默认 True）。"
        ),
    },
    "list_allowed_directories": {
        "properties": {},
        "required": [],
        "enhanced_description": (
            "查询系统允许访问的目录白名单列表（不是列出目录内容）。无参数。"
            "用于检查哪些目录可以被访问，确保文件操作在允许范围内。"
        ),
    },
}


def convert_mcp_tools(mcp_tools: List[Any]) -> List[BaseTool]:
    """
    将 MCP 工具列表批量转换为 LangChain BaseTool 格式

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
    将单个 MCP 工具转换为 LangChain BaseTool（支持完整类型化 Schema）

    增强点：
    1. 使用预定义 Schema 增强参数描述，避免 LLM 选错工具
    2. 从 MCP inputSchema 提取类型信息，构建 Pydantic 模型
    3. 支持 enum/array/object 等复杂类型
    4. 工具描述包含完整的参数说明

    Args:
        mcp_tool: MCP 工具对象

    Returns:
        LangChain BaseTool 或 None
    """
    from langchain_core.tools import StructuredTool

    # 获取工具名称
    tool_name = getattr(mcp_tool, 'name', None)
    if not tool_name:
        return None

    # 使用预定义 Schema 或从 MCP inputSchema 动态构建
    predefined = _MCP_TOOL_SCHEMAS.get(tool_name)
    if predefined:
        properties = predefined["properties"]
        required = predefined["required"]
        tool_description = predefined["enhanced_description"]
    else:
        # 从 MCP inputSchema 提取
        input_schema = getattr(mcp_tool, 'inputSchema', {})
        if isinstance(input_schema, dict):
            properties = input_schema.get('properties', {})
            required = input_schema.get('required', [])
        else:
            properties = {}
            required = []
        tool_description = getattr(mcp_tool, 'description', f'MCP 工具: {tool_name}')

    # 构建类型化 Pydantic 模型
    args_schema = _create_typed_pydantic_model(tool_name, properties, required)

    # 异步执行函数
    _tool_name = tool_name

    async def execute_tool_async(**kwargs) -> str:
        """执行 MCP 工具（异步实现，带参数验证）"""
        from src.models.mcp_client import mcp_manager

        # 参数校验：使用 Pydantic 模型验证
        try:
            validated = args_schema(**kwargs)
            validated_kwargs = validated.model_dump()
        except Exception as e:
            # Pydantic 验证失败，返回友好错误
            return f"M参数校验失败: {str(e)}"

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

            result = await tool_manager.call_mcp_tool(server_name, _tool_name, validated_kwargs)
            return format_mcp_result(result)

        except Exception as e:
            return f"MCP 工具执行错误: {str(e)}"

    try:
        langchain_tool = StructuredTool.from_function(
            coroutine=execute_tool_async,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
        )
        return langchain_tool
    except Exception as e:
        print(f"创建工具失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _create_typed_pydantic_model(
    tool_name: str,
    properties: dict,
    required: List[str]
) -> Type[BaseModel]:
    """
    根据 MCP 工具的输入模式创建完整的类型化 Pydantic 模型。

    增强点：
    - 支持更完整的类型映射（integer/float/boolean/array/object/string）
    - 支持 enum 类型（从 JSON Schema enum 字段提取）
    - 支持数组类型（items 定义）
    - 支持默认值和必填字段区分
    - 字段描述作为 docstring 的一部分，便于 LLM 理解

    Args:
        tool_name: 工具名称（用于生成模型名）
        properties: JSON Schema properties
        required: 必填字段列表

    Returns:
        Pydantic BaseModel 类
    """
    field_definitions = {}

    for prop_name, prop_info in properties.items():
        # 字段描述（优先使用 description，否则用 title）
        description = prop_info.get('description') or prop_info.get('title', '')

        # 类型推断（支持嵌套类型）
        prop_type: Type = str
        json_type = prop_info.get('type', 'string')

        if json_type == 'integer':
            prop_type = int
        elif json_type == 'number':
            prop_type = float
        elif json_type == 'boolean':
            prop_type = bool
        elif json_type == 'array':
            # 尝试推断数组元素类型
            items = prop_info.get('items', {})
            item_type = items.get('type', 'string')
            if item_type == 'integer':
                prop_type = List[int]
            elif item_type == 'number':
                prop_type = List[float]
            elif item_type == 'boolean':
                prop_type = List[bool]
            else:
                prop_type = List[str]
        elif json_type == 'object':
            prop_type = dict

        # 处理 enum（创建 Literal Union 类型）
        if 'enum' in prop_info and prop_info['enum']:
            enum_values = prop_info['enum']
            if len(enum_values) == 1:
                # 单值枚举：直接使用该值
                prop_type = type(enum_values[0])
                default = enum_values[0]
                field_definitions[prop_name] = (
                    prop_type,
                    Field(default=default, description=description)
                )
                continue
            else:
                # 多值枚举：使用 Literal Union
                from typing import Literal
                union_types = [type(v) for v in enum_values]
                if len(set(union_types)) == 1:
                    # 同类型枚举
                    literal_type: Any = Literal[tuple(enum_values)]  # type: ignore
                    prop_type = literal_type
                else:
                    # 混合格式，使用 str
                    prop_type = str

        # 默认值处理
        default: Any = ...
        if prop_name not in required:
            if 'default' in prop_info:
                default = prop_info['default']
            else:
                default = None

        field_definitions[prop_name] = (prop_type, Field(default=default, description=description))

    # 如果没有属性，添加一个通用 input 字段
    if not field_definitions:
        field_definitions['input'] = (str, Field(default="", description="输入参数"))

    # 生成模型名（符合 Python 标识符规范）
    safe_name = tool_name.replace('-', '_').replace(' ', '_')
    model_name = f"{safe_name.title().replace('_', '')}Input"

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
        # 结构化返回（保持 dict 格式便于调试）
        import json
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception:
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
    from src.models.mcp_client import mcp_manager

    try:
        mcp_tools = mcp_manager.get_tools()
        return convert_mcp_tools(mcp_tools)
    except Exception as e:
        print(f"获取 MCP 工具失败: {e}")
        return []


def get_all_agent_tools() -> List[BaseTool]:
    """
    获取所有 Agent 可用的工具

    Returns:
        包含所有可用工具的列表
    """
    from src.models.mcp_client import mcp_manager

    tools = []

    # 添加 Skill 工具 (datetime, calculator 等)
    try:
        from ..skills.skill_loader import get_skill_loader
        loader = get_skill_loader()

        for skill_name in ['datetime', 'calculator', 'general']:
            try:
                skill = loader.load_skill(skill_name)
                tools.extend(skill.tools)
            except Exception as e:
                print(f"加载 {skill_name} skill 失败: {e}")

    except Exception as e:
        print(f"加载 Skill 工具失败: {e}")

    # 添加 MCP 工具（带完整类型化 Schema）
    try:
        mcp_tools = mcp_manager.get_tools()
        if mcp_tools:
            converted_tools = convert_mcp_tools(mcp_tools)
            tools.extend(converted_tools)
    except Exception as e:
        print(f"获取 MCP 工具失败: {e}")

    return tools
