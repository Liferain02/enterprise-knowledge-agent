"""
Knowledge Skill Tools - 知识检索工具
"""
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class SearchInput(BaseModel):
    """搜索输入"""
    query: str = Field(description="搜索查询字符串")
    top_k: int = Field(default=5, description="返回结果数量")


def knowledge_search(query: str, top_k: int = 5) -> str:
    """
    搜索企业知识库
    
    Args:
        query: 搜索查询字符串
        top_k: 返回结果数量
    
    Returns:
        相关文档内容列表
    """
    from rag.retriever import get_retriever_manager
    
    try:
        retriever_manager = get_retriever_manager()
        results = retriever_manager.search(query, k=top_k)
        
        if not results:
            return "未在知识库中找到相关内容。建议尝试使用不同的关键词或简化查询。"
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
            formatted_results.append(f"【结果 {i}】来源: {source}\n{doc.page_content}\n")
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"搜索知识库时出错: {str(e)}"


def create_knowledge_search_tool() -> BaseTool:
    """创建知识搜索工具"""
    from langchain_core.tools import StructuredTool
    
    return StructuredTool.from_function(
        func=knowledge_search,
        name="knowledge_search",
        description="""搜索企业知识库，返回与查询相关的文档内容。

适用场景：
- 回答公司规章制度、技术文档、FAQ等问题
- 询问某个政策、流程、规范的具体内容
- 需要从企业文档中查找信息时

输入：搜索查询字符串和返回数量。
输出：相关文档内容列表。""",
        args_schema=SearchInput
    )


# 导出所有工具函数
__all__ = ["knowledge_search", "create_knowledge_search_tool"]
