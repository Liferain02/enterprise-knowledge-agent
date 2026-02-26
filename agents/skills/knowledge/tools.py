"""
Knowledge Skill Tools - 知识检索工具
"""
from typing import Type, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class SearchInput(BaseModel):
    """搜索输入"""
    query: str = Field(description="搜索查询字符串")
    top_k: int = Field(default=5, description="返回结果数量")


class KnowledgeSearchTool(BaseTool):
    """知识库搜索工具"""
    
    name: str = "knowledge_search"
    description: str = """搜索企业知识库，返回与查询相关的文档内容。
    
适用场景：
- 回答公司规章制度、技术文档、FAQ等问题
- 询问某个政策、流程、规范的具体内容
- 需要从企业文档中查找信息时

输入：搜索查询字符串和返回数量。
输出：相关文档内容列表。"""
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str, top_k: int = 5, **kwargs) -> str:
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


# 工具缓存
_tools = None


def get_tools() -> List[BaseTool]:
    """获取工具列表"""
    global _tools
    if _tools is None:
        _tools = [KnowledgeSearchTool()]
    return _tools

