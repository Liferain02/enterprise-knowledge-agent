"""
Knowledge Skill Tools - 知识检索工具
支持 Corrective RAG（检索结果评估与自我纠错）
支持 Query Expansion（复杂查询主动分解）
"""
import re
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


# ==================== 复杂查询识别（供外部调用）====================

# 需要主动触发 Query Expansion 的查询模式
# ⚠️ 此列表与 Planner._quick_complexity_check() 的 _HIGH_PRIORITY_PATTERNS 保持同步
_QUERY_EXPANSION_PATTERNS = [
    # 对比类
    re.compile(r"对比|比较.{0,8}[和与跟]|和.{0,8}区别|差异|不同点|哪个.好", re.IGNORECASE),
    re.compile(r"\bvs\b|VS|versus| versus ", re.IGNORECASE),
    # 列举类
    re.compile(r"有哪些|有些什么|都有哪些|都有什么", re.IGNORECASE),
    # 多实体类
    re.compile(r".{2,6}和.{2,6}.{0,8}职责|.{2,6}与.{2,6}.{0,8}区别", re.IGNORECASE),
    re.compile(r".{2,6}和.{2,6}的|.{2,6}与.{2,6}的", re.IGNORECASE),
    # 多问号类
    re.compile(r"[？\?].{0,20}[和与跟][^和与跟]", re.IGNORECASE),
]


def needs_query_expansion(query: str) -> bool:
    """
    判断查询是否需要主动触发 Query Expansion。
    在 CRAG 评估之前就识别复杂查询类型，避免等 CRAG 失败后再走 expansion。

    注意：此函数应与 Planner._quick_complexity_check() 保持同步，
    两者使用相同的 pattern 定义。优先通过 state 传递 Planner 的判断结果，
    此函数仅作为外部调用时的 fallback。
    """
    for pattern in _QUERY_EXPANSION_PATTERNS:
        if pattern.search(query):
            return True
    # 多个问号也触发
    if query.count("？") >= 2 or query.count("?") >= 2:
        return True
    return False


# ==================== Schema & Tools ====================

class SearchInput(BaseModel):
    """搜索输入"""
    query: str = Field(description="搜索查询字符串")
    top_k: int = Field(default=5, description="返回结果数量")
    needs_expansion: bool = Field(
        default=None,
        description="是否需要 Query Expansion（由 Planner 根据复杂度判断传入，为 None 时自动判断）"
    )


def knowledge_search(
    query: str,
    top_k: int = 5,
    needs_expansion: bool = None,
) -> str:
    """
    搜索实验室知识库（同步封装，异步逻辑在内部处理）。

    检索策略选择逻辑（统一由 Planner 复杂度判断）：
    1. CRAG_ENABLED=true 时，按 needs_expansion 决定是否先做 Query Expansion；
    2. CRAG_ENABLED=false（默认）时，使用标准 Hybrid + Rerank 流程；复杂查询
       仍由主检索节点按配置选择 Query Expansion；
    3. needs_expansion=None（外部调用未传入）时，回退到 needs_query_expansion()。

    Args:
        query: 搜索查询字符串
        top_k: 返回结果数量
        needs_expansion: Planner 传入的复杂度判断，None 时自动判断

    Returns:
        相关文档内容列表（包含分数、来源和评估信息）
    """
    import asyncio
    from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline
    from config.settings import get_settings

    settings = get_settings()
    min_score = getattr(settings, 'reranker_threshold', 0.1) or 0.1

    # 优先信任上游 Planner 的判断，None 时才自己做 fallback
    force_expansion = (
        needs_expansion
        if needs_expansion is not None
        else needs_query_expansion(query)
    )

    try:
        # 缺失属性也必须按默认关闭处理，避免旧配置意外启用高延迟 CRAG。
        if getattr(settings, 'crag_enabled', False):
            # CRAG 主路径（内嵌 Query Expansion 前置 + 评估 + rewrite）
            # needs_expansion=True 时，pipeline.retrieve() 会在评估前先分解查询
            result = _run_crag_search(query, top_k, needs_expansion=force_expansion)
        else:
            # 传统 Rerank 流程
            result = _knowledge_search_rerank(query, top_k, min_score)

        return result

    except Exception as e:
        # 降级：如果出现异常，回退到传统检索
        import traceback
        traceback.print_exc()
        return f"搜索知识库时出错: {str(e)}\n\n[降级模式] 尝试使用传统检索...\n" + \
            _knowledge_search_rerank(query, top_k, min_score)


def _run_crag_search(query: str, top_k: int, needs_expansion: bool = None) -> str:
    """
    安全运行 CRAG 异步搜索。

    解决 asyncio.run() 在已有事件循环中崩溃的问题：
    - 如果没有运行中的循环 → 创建新循环（asyncio.run）
    - 如果已有循环 → 复用现有循环（asyncio.run_until_complete）
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_knowledge_search_cragn(query, top_k, needs_expansion))

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(
            asyncio.run,
            _knowledge_search_cragn(query, top_k, needs_expansion)
        )
        return future.result()


async def _run_query_expansion_async(query: str, top_k: int) -> str:
    """
    Query Expansion 检索流程：分解 → 并行检索 → RRF 合并 → 精排 → 返回

    不经过 CRAG rewrite，避免 24s+ 的延迟。
    """
    from src.rag.retrieval.query_expander import decompose_and_retrieve
    from src.rag.retrieval.retriever import get_retriever_manager

    # 分解查询
    results_with_meta, exp_result = await decompose_and_retrieve(
        query=query,
        top_k=top_k,
        strategy=None,  # 使用默认配置
    )

    if not results_with_meta:
        return (
            "【检索结果】\n"
            "Query Expansion 未找到相关内容。\n"
            f"已分解为 {len(exp_result.all_queries)} 个子查询：{exp_result.all_queries}"
        )

    # 精排（如果有 reranker）
    retriever_manager = get_retriever_manager()
    docs = [doc for doc, _ in results_with_meta]

    if retriever_manager.use_reranker and retriever_manager.reranker_manager:
        reranked = retriever_manager.reranker_manager.rerank(query, docs)
        results_with_score = [
            (reranked[i], results_with_meta[i][1])
            for i in range(len(reranked))
        ]
    else:
        results_with_score = results_with_meta

    # 过滤低分
    threshold = getattr(
        __import__('config.settings', fromlist=['get_settings']).get_settings(),
        'reranker_threshold', 0.1
    )
    filtered = [(d, s) for d, s in results_with_score if s >= threshold]

    # 格式化
    formatted = []
    for i, (doc, score) in enumerate(filtered[:top_k], 1):
        source = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
        formatted.append(
            f"【结果 {i}】相关性: {round(score * 100, 1)}%\n"
            f"来源: {source}\n"
            f"内容: {doc.page_content}\n"
        )

    if not formatted:
        return "【检索结果】\nQuery Expansion 未找到足够相关的内容。"

    result_text = "【检索结果 (Query Expansion)】\n" + "\n---\n".join(formatted)
    result_text += (
        f"\n\n【Query Expansion 摘要】\n"
        f"分解为 {len(exp_result.all_queries)} 个子查询：{exp_result.all_queries}\n"
        f"融合策略: RRF (k=60)"
    )
    result_text += "\n\n【使用说明】\n以上结果按相关性分数排序。"
    return result_text


def _run_query_expansion_search(query: str, top_k: int) -> str:
    """同步封装 Query Expansion 搜索"""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_query_expansion_async(query, top_k))

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(
            asyncio.run,
            _run_query_expansion_async(query, top_k)
        )
        return future.result()


async def _knowledge_search_cragn(
    query: str,
    top_k: int,
    needs_expansion: bool = None,
) -> str:
    """
    Corrective RAG 检索流程

    流程：
    1. pipeline.retrieve() 内部完成：检索 → 评估 → 决策
    2. 根据决策结果格式化输出
    """
    from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline

    pipeline = get_corrective_rag_pipeline()
    results_with_scores, grade_result, rewrite_history = await pipeline.retrieve(
        query, top_k=top_k, needs_expansion=needs_expansion
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
            elif grade_info.grade.value == "medium":
                grade_tag = " [中等]"
            else:
                grade_tag = " [低]"

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
        f"HIGH: {grade_result.high_count} | MEDIUM: {grade_result.medium_count} | LOW: {grade_result.low_count} | "
        f"avg={grade_result.avg_score:.2f}\n"
        f"平均相关度: {grade_result.avg_score:.0%}"
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
        description="""搜索实验室知识库，返回与查询相关的文档内容。

适用场景：
- 回答实验室制度、项目资料、论文笔记、技术文档、FAQ等问题
- 询问某个流程、规范、组会要求或环境配置的具体内容
- 需要从实验室文档中查找信息时

输入：搜索查询字符串和返回数量。
输出：相关文档内容列表。""",
        args_schema=SearchInput
    )


def get_knowledge_tools() -> list:
    """获取知识搜索工具列表"""
    return [create_knowledge_search_tool()]


# 导出所有工具函数
__all__ = ["knowledge_search", "create_knowledge_search_tool", "get_knowledge_tools"]
