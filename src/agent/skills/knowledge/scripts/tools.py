"""
Knowledge Skill Tools - 知识检索工具
支持 Corrective RAG（检索结果评估与自我纠错）
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
    搜索企业知识库（同步封装，异步逻辑在内部处理）。

    支持 Corrective RAG：
    - 检索后评估文档与查询的相关性
    - 低相关时自动重写查询并重新检索（最多重试 2 次）
    - 最终返回真正相关的高质量结果

    注意：此函数在同步和异步上下文中均可安全调用。
    在已有事件循环时会复用现有循环，不会创建新的。

    Args:
        query: 搜索查询字符串
        top_k: 返回结果数量

    Returns:
        相关文档内容列表（包含分数、来源和评估信息）
    """
    import asyncio
    from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline
    from config.settings import get_settings

    settings = get_settings()
    min_score = getattr(settings, 'reranker_threshold', 0.1) or 0.1

    try:
        # 选择检索策略：Corrective RAG 或 传统 Rerank
        if getattr(settings, 'crag_enabled', True):
            # Corrective RAG：检索 → 评估 → 决策（使用/重写/放弃）
            result = _run_crag_search(query, top_k)
        else:
            # 传统 Rerank 流程
            result = _knowledge_search_rerank(query, top_k, min_score)

        return result

    except Exception as e:
        # CRAG 降级：如果出现异常，回退到传统检索
        import traceback
        traceback.print_exc()
        return f"搜索知识库时出错: {str(e)}\n\n[降级模式] 尝试使用传统检索...\n" + \
            _knowledge_search_rerank(query, top_k, min_score)


def _run_crag_search(query: str, top_k: int) -> str:
    """
    安全运行 CRAG 异步搜索。

    解决 asyncio.run() 在已有事件循环中崩溃的问题：
    - 如果没有运行中的循环 → 创建新循环（asyncio.run）
    - 如果已有循环 → 复用现有循环（asyncio.run_until_complete）
    """
    import asyncio

    try:
        # 尝试直接获取当前事件循环
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行中的循环 → 可以安全使用 asyncio.run()
        return asyncio.run(_knowledge_search_cragn(query, top_k))

    # 已有运行中的循环 → 创建 Task 在现有循环中执行
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(
            asyncio.run,
            _knowledge_search_cragn(query, top_k)
        )
        return future.result()


async def _knowledge_search_cragn(query: str, top_k: int) -> str:
    """
    Corrective RAG 检索流程

    流程：
    1. pipeline.retrieve() 内部完成：检索 → 评估 → 决策
    2. 根据决策结果格式化输出
    """
    from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline

    pipeline = get_corrective_rag_pipeline()
    results_with_scores, grade_result, rewrite_history = await pipeline.retrieve(
        query, top_k=top_k
    )

    # 无结果
    if not results_with_scores:
        rewrite_note = ""
        if len(rewrite_history) > 1:
            rewrite_note = (
                f"\n\n【检索历程】\n"
                f"已尝试 {len(rewrite_history)} 种查询表述，均未找到相关内容。"
                f"\n查询变化：{' -> '.join(rewrite_history)}"
            )

        base_msg = (
            "【检索结果】\n"
            "未在知识库中找到相关内容。"
            f"\n\n{grade_result.decision_reason}"
            f"{rewrite_note}"
            "\n\n建议您：1) 尝试使用不同的关键词；2) 简化问题表述；"
            "3) 联系管理员补充相关文档。"
        )

        # 如果评估结果认为有部分相关性但不够高，也一并告知
        if grade_result.avg_score > 0.3:
            return base_msg

        return base_msg

    # 格式化结果
    formatted_results = []
    for i, (doc, score) in enumerate(results_with_scores, 1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "未知来源")

        # 获取该文档的评估信息
        grade_info = None
        for g in grade_result.grades:
            if g.doc.page_content == doc.page_content:
                grade_info = g
                break

        score_percent = round(score * 100, 1)
        grade_tag = ""
        if grade_info:
            if grade_info.grade.value == "high":
                grade_tag = " [高相关]"
            else:
                grade_tag = " [低相关]"

        formatted_results.append(
            f"【结果 {i}】相关性: {score_percent}%{grade_tag}\n"
            f"来源: {source}\n"
            f"内容: {doc.page_content}\n"
        )

    result_text = "【检索结果】\n" + "\n---\n".join(formatted_results)

    # 添加 CRAG 评估摘要（帮助后续生成时理解结果质量）
    crag_summary = (
        f"\n\n【CRAG 评估摘要】\n"
        f"决策: {grade_result.decision.value.upper()} | "
        f"高相关: {grade_result.high_count}/{grade_result.total_docs} | "
        f"平均相关分: {grade_result.avg_score:.2f}\n"
        f"决策理由: {grade_result.decision_reason}"
    )

    if len(rewrite_history) > 1:
        crag_summary += f"\n查询已重写: {' -> '.join(rewrite_history)}"

    result_text += crag_summary
    result_text += "\n\n【使用说明】\n以上结果按相关性分数排序，分数越高表示与问题越相关。"

    return result_text


def _knowledge_search_rerank(query: str, top_k: int, min_score: float) -> str:
    """
    传统 Rerank 检索流程（Corrective RAG 禁用时的降级方案）
    """
    from src.rag.retrieval.retriever import get_retriever_manager

    retriever_manager = get_retriever_manager()

    if retriever_manager.use_reranker and retriever_manager.reranker_manager:
        results_with_score = retriever_manager.search_with_rerank(query, k=top_k)
    else:
        results_with_score = retriever_manager.search_with_score(query, k=top_k)

    if not results_with_score:
        return "【检索结果】\n未在知识库中找到相关内容。建议尝试使用不同的关键词或简化查询。"

    # 过滤低相关性结果
    filtered_results = [(doc, score) for doc, score in results_with_score if score >= min_score]

    if not filtered_results:
        return f"""【检索结果】
未找到足够相关的内容（相关性分数均低于 {min_score}）。
知识库中可能没有您需要的信息，请尝试其他问题或调整查询关键词。"""

    formatted_results = []
    for i, (doc, score) in enumerate(filtered_results, 1):
        source = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
        score_percent = round(score * 100, 1)
        formatted_results.append(
            f"【结果 {i}】相关性: {score_percent}%\n"
            f"来源: {source}\n"
            f"内容: {doc.page_content}\n"
        )

    result_text = "【检索结果】\n" + "\n---\n".join(formatted_results)
    result_text += "\n\n【使用说明】\n以上结果按相关性分数排序，分数越高表示与问题越相关。"

    return result_text


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
