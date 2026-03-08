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
        相关文档内容列表（包含分数和来源信息）
    """
    from src.rag.retrieval.retriever import get_retriever_manager
    from config.settings import get_settings

    settings = get_settings()
    min_score = getattr(settings, 'reranker_threshold', 0.1) or 0.1

    try:
        retriever_manager = get_retriever_manager()

        # 使用带 Reranker 的搜索
        if retriever_manager.use_reranker and retriever_manager.reranker_manager:
            results_with_score = retriever_manager.search_with_rerank(query, k=top_k)
        else:
            # 如果没有 Reranker，使用带分数的搜索
            results_with_score = retriever_manager.search_with_score(query, k=top_k)

        if not results_with_score:
            return "【检索结果】\n未在知识库中找到相关内容。建议尝试使用不同的关键词或简化查询。"

        # 过滤低相关性结果
        filtered_results = [(doc, score) for doc, score in results_with_score if score >= min_score]

        if not filtered_results:
            # 如果所有结果都被过滤，说明没有足够相关的内容
            return f"""【检索结果】
未找到足够相关的内容（相关性分数均低于 {min_score}）。
知识库中可能没有您需要的信息，请尝试其他问题或调整查询关键词。"""

        # 格式化结果，包含分数和来源
        formatted_results = []
        for i, (doc, score) in enumerate(filtered_results, 1):
            source = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
            # 将分数转换为可读性更高的形式（0-100）
            score_percent = round(score * 100, 1)
            formatted_results.append(
                f"【结果 {i}】相关性: {score_percent}%\n"
                f"来源: {source}\n"
                f"内容: {doc.page_content}\n"
            )

        result_text = "【检索结果】\n" + "\n---\n".join(formatted_results)

        # 添加使用提示
        result_text += "\n\n【使用说明】\n以上结果按相关性分数排序，分数越高表示与问题越相关。"

        return result_text

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


def get_knowledge_tools() -> list:
    """获取知识搜索工具列表"""
    return [create_knowledge_search_tool()]


# 导出所有工具函数
__all__ = ["knowledge_search", "create_knowledge_search_tool", "get_knowledge_tools"]
