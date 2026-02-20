"""
工具基类模块
定义 Agent 工具的基础接口和通用工具
"""
from typing import Type, Callable, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from datetime import datetime
# ==================== 工具输入模型 ====================
class SearchKnowledgeInput(BaseModel):
    """搜索知识库输入"""
    query: str = Field(description="搜索查询字符串")
    top_k: int = Field(default=5, description="返回结果数量")


class CalculatorInput(BaseModel):
    """计算器输入"""
    expression: str = Field(description="数学表达式，例如: 2+2*3")


class DateTimeInput(BaseModel):
    """获取日期时间输入"""
    format: Optional[str] = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="日期时间格式"
    )


class FileSearchInput(BaseModel):
    """文件搜索输入"""
    directory: str = Field(description="要搜索的目录路径")
    pattern: str = Field(description="文件匹配模式，例如: *.txt")


# ==================== 基础工具类 ====================
class KnowledgeSearchTool(BaseTool):
    """知识库搜索工具"""
    
    name: str = "search_knowledge"
    description: str = "搜索企业知识库，返回与查询相关的文档内容。适用于回答公司规章制度、技术文档、FAQ等问题。"
    args_schema: Type[BaseModel] = SearchKnowledgeInput
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """执行搜索"""
        from rag.retriever import get_retriever_manager
        
        try:
            retriever_manager = get_retriever_manager()
            results = retriever_manager.search(query, k=top_k)
            
            if not results:
                return "未在知识库中找到相关内容"
            
            # 格式化结果
            formatted_results = []
            for i, doc in enumerate(results, 1):
                formatted_results.append(
                    f"【结果 {i}】\n{doc.page_content}\n"
                )
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"搜索知识库时出错: {str(e)}"


class CalculatorTool(BaseTool):
    """数学计算工具"""
    
    name: str = "calculate"
    description: str = "执行数学计算。适用于需要计算的问题，如费用计算、统计数据等。"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """执行计算"""
        try:
            # 安全地计算数学表达式
            # 注意：实际使用中应该使用更安全的计算方式
            allowed_globals = {"__builtins__": None}
            result = eval(expression, allowed_globals, {})
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"


class DateTimeTool(BaseTool):
    """获取当前日期时间工具"""
    
    name: str = "get_date"
    description: str = "获取当前日期和时间。适用于询问当前时间、日期相关的问题。"
    args_schema: Type[BaseModel] = DateTimeInput
    
    def _run(
        self,
        format: str = "%Y-%m-%d %H:%M:%S",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """获取当前日期时间"""
        now = datetime.now()
        return now.strftime(format)


class FileSearchTool(BaseTool):
    """文件搜索工具"""
    
    name: str = "search_files"
    description: str = "在指定目录中搜索文件。适用于查找特定类型的文件。"
    args_schema: Type[BaseModel] = FileSearchInput
    
    def _run(
        self,
        directory: str,
        pattern: str = "*",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """搜索文件"""
        from pathlib import Path
        
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return f"目录不存在: {directory}"
            
            files = list(dir_path.glob(pattern))
            
            if not files:
                return f"在 {directory} 中未找到匹配 {pattern} 的文件"
            
            result = [f"找到 {len(files)} 个文件:"]
            for f in files[:20]:  # 限制显示数量
                result.append(f"  - {f.name}")
            
            return "\n".join(result)
        
        except Exception as e:
            return f"搜索文件时出错: {str(e)}"


# ==================== 工具工厂函数 ====================
def create_knowledge_search_tool() -> BaseTool:
    """创建知识搜索工具"""
    return KnowledgeSearchTool()


def create_calculator_tool() -> BaseTool:
    """创建计算器工具"""
    return CalculatorTool()


def create_datetime_tool() -> BaseTool:
    """创建日期时间工具"""
    return DateTimeTool()


def create_file_search_tool() -> BaseTool:
    """创建文件搜索工具"""
    return FileSearchTool()


def get_all_tools() -> List[BaseTool]:
    """获取所有可用工具"""
    return [
        create_knowledge_search_tool(),
        create_calculator_tool(),
        create_datetime_tool(),
        create_file_search_tool()
    ]


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """根据名称获取工具"""
    tools = {tool.name: tool for tool in get_all_tools()}
    return tools.get(name)

