"""
Agent 工具模块
重构后的标准 LangChain Tools
"""
from typing import Type, Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from datetime import datetime
import math


# ==================== 工具输入模型 ====================

class SearchKnowledgeInput(BaseModel):
    """搜索知识库输入"""
    query: str = Field(description="搜索查询字符串")
    top_k: int = Field(default=5, description="返回结果数量")


class CalculatorInput(BaseModel):
    """计算器输入"""
    expression: str = Field(description="数学表达式，例如: 2+2*3 或 sqrt(16)")


class DateTimeInput(BaseModel):
    """获取日期时间输入"""
    format: Optional[str] = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="日期时间格式，例如: %Y年%m月%d日"
    )


# ==================== RAG 知识搜索工具 ====================

class KnowledgeSearchTool(BaseTool):
    """
    知识库搜索工具
    封装 RAG 检索能力为标准 LangChain Tool
    """
    
    name: str = "knowledge_search"
    description: str = """搜索企业知识库，返回与查询相关的文档内容。
    适用场景：
    - 回答公司规章制度、技术文档、FAQ等问题
    - 询问某个政策、流程、规范的具体内容
    - 需要从企业文档中查找信息时
    
    输入：搜索查询字符串和返回数量。
    输出：相关文档内容列表。"""
    args_schema: Type[BaseModel] = SearchKnowledgeInput
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """执行知识库搜索"""
        from rag.retriever import get_retriever_manager
        
        try:
            retriever_manager = get_retriever_manager()
            results = retriever_manager.search(query, k=top_k)
            
            if not results:
                return "未在知识库中找到相关内容。建议尝试使用不同的关键词或简化查询。"
            
            # 格式化结果
            formatted_results = []
            for i, doc in enumerate(results, 1):
                source_info = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
                formatted_results.append(
                    f"【结果 {i}】来源: {source_info}\n{doc.page_content}\n"
                )
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"搜索知识库时出错: {str(e)}"


# ==================== 计算器工具 ====================

class CalculatorTool(BaseTool):
    """
    数学计算工具
    支持基本数学运算和常见数学函数
    """
    
    name: str = "calculator"
    description: str = """执行数学计算。
    适用场景：
    - 费用计算、统计数据
    - 数值运算、百分比计算
    - 任何需要数学计算的问题
    
    输入：数学表达式。
    输出：计算结果。"""
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """执行数学计算"""
        try:
            # 安全地计算数学表达式
            # 添加常用数学函数到安全命名空间
            safe_globals = {
                "__builtins__": {},
                "sqrt": math.sqrt,
                "pow": pow,
                "abs": abs,
                "round": round,
                "max": max,
                "min": min,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "pi": math.pi,
                "e": math.e,
            }
            
            result = eval(expression, {"__builtins__": {}}, safe_globals)
            return f"计算结果: {expression} = {result}"
        
        except Exception as e:
            return f"计算错误: {str(e)}"


# ==================== 日期时间工具 ====================

class DateTimeTool(BaseTool):
    """
    获取当前日期时间工具
    """
    
    name: str = "get_current_datetime"
    description: str = """获取当前日期和时间。
    适用场景：
    - 询问当前时间、日期
    - 需要获取系统时间作为参考
    
    输入：可选的日期时间格式。
    输出：格式化后的当前日期时间。"""
    args_schema: Type[BaseModel] = DateTimeInput
    
    def _run(
        self,
        format: str = "%Y-%m-%d %H:%M:%S",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """获取当前日期时间"""
        now = datetime.now()
        return now.strftime(format)


# ==================== 工具工厂函数 ====================

def get_knowledge_tool() -> BaseTool:
    """获取知识搜索工具"""
    return KnowledgeSearchTool()


def get_calculator_tool() -> BaseTool:
    """获取计算器工具"""
    return CalculatorTool()


def get_datetime_tool() -> BaseTool:
    """获取日期时间工具"""
    return DateTimeTool()


def get_base_tools() -> List[BaseTool]:
    """获取所有基础工具（不含 MCP 工具）"""
    return [
        get_knowledge_tool(),
        get_calculator_tool(),
        get_datetime_tool(),
    ]


# ==================== 统一工具获取 ====================

def get_all_agent_tools() -> List[BaseTool]:
    """
    获取所有 Agent 可用工具
    包含基础工具 + MCP 工具
    """
    from core.mcp_client import mcp_manager
    
    tools = get_base_tools()
    
    # 添加 MCP 工具
    try:
        mcp_tools = mcp_manager.get_tools()
        # MCP 工具需要转换为 LangChain BaseTool 格式
        from agent.tools.mcp_adapter import convert_mcp_tools
        mcp_langchain_tools = convert_mcp_tools(mcp_tools)
        tools.extend(mcp_langchain_tools)
    except Exception as e:
        print(f"加载 MCP 工具失败: {e}")
    
    return tools


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """根据名称获取工具"""
    for tool in get_all_agent_tools():
        if tool.name == name:
            return tool
    return None

