"""
Knowledge Retrieval Pipeline - 知识检索管线

重构后（参考 OpenClaw / Agentic RAG 最佳实践）：
- retrieval_agent_node：检索阶段（无 ReAct 循环，直接执行 Hybrid → Rerank → ACL → CRAG 评估）
- generation_agent_node：生成阶段（基于评估后的文档，独立生成答案）
- conflict_detection_node：可选的冲突检测（检测多文档中的矛盾信息）

核心改进：
1. 消除 ReAct 循环开销：检索阶段直接调用 pipeline，不经过 LLM 决策循环
2. 评估是独立的：CRAG Grading Agent 作为独立阶段，Supervisor 可感知评估结果
3. 查询改写是可选的预处理：在检索前判断，评估失败后才触发重写
4. 冲突检测可插拔：作为可选节点，不影响主流程性能

流程：
  Supervisor → retrieval_agent_node → (conflict_detection_node) → generation_agent_node → save_to_mem0 → END
"""
import asyncio
import time
import re
from typing import Dict, Any, List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from ._utils import get_last_user_message, inject_summary_to_messages, inject_context_to_messages
from src.observability import traced


# ============================================================
# retrieval_agent_node - 检索阶段（核心改进）
# ============================================================

@traced("agent.retrieval.node")
async def retrieval_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    检索阶段节点 - 替代原 knowledge_agent_node 中的 ReAct 循环

    职责：
    1. 直接调用 CorrectiveRAGPipeline.retrieve()
       流程：Hybrid 检索 → Rerank 精排 → CRAG LLM 评估 → 查询改写/分解（条件触发）
    2. 将评估结果写入 state，供 Supervisor 感知和 generation_agent_node 使用
    3. 生成检索上下文字符串（供生成阶段用）

    不再：
    - 使用 SkillLoader 创建 ReAct Agent
    - 让 LLM 自主决定调用哪个工具（tool_call 循环）
    """
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    summary = state.get("summary", "") or ""
    mem0_memories = state.get("mem0_memories", "") or ""
    session_id = state.get("session_id", "default")
    user_id = state.get("user_id", "default_user")

    if not last_user_message:
        return {
            "final_answer": "抱歉，我无法理解您的问题。",
        }

    # Supervisor 的 expansion 判断（优先使用；若为空则内部判断）
    needs_expansion = state.get("needs_expansion", False)

    try:
        t0 = time.time()

        # ── 核心：直接调用 Corrective RAG Pipeline ──────────────────────
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline

        pipeline = get_corrective_rag_pipeline()

        # 获取 top_k：从配置或 state 读取，默认 5
        top_k = state.get("retrieval_top_k", 5)

        print(f"[Retrieval] 开始检索: '{last_user_message[:50]}...' top_k={top_k}")

        results, grade_result, rewrite_history = await pipeline.retrieve(
            query=last_user_message,
            top_k=top_k,
            needs_expansion=needs_expansion,
        )

        retrieval_time = time.time() - t0

        # ── 解析评估结果 ──────────────────────────────────────────────
        decision = grade_result.decision.value if grade_result else "no_results"
        avg_score = grade_result.avg_score if grade_result else 0.0
        decision_reason = grade_result.decision_reason if grade_result else "无评估结果"

        print(
            f"[Retrieval] 完成: decision={decision}, "
            f"high={grade_result.high_count if grade_result else 0}/"
            f"{grade_result.total_docs if grade_result else 0}, "
            f"avg={avg_score:.3f}, rewrite_history={rewrite_history}, "
            f"耗时={retrieval_time:.2f}s"
        )

        # ── 构建检索上下文（供生成阶段用）─────────────────────────────
        docs = [doc for doc, _ in results]
        retrieval_context = _build_retrieval_context(last_user_message, results, grade_result)

        # ── 检测 NO_RESULTS：直接返回"知识库无相关信息"───────────────
        if decision == "no_results" or not docs:
            print(f"[Retrieval] 知识库无相关信息，返回空结果")
            return {
                "retrieval_context": "【知识库检索结果】\n未在知识库中找到与您问题相关的信息。",
                "retrieved_docs": [],
                "retrieval_decision": "no_results",
                "retrieval_decision_reason": decision_reason,
                "retrieval_avg_score": 0.0,
                "retrieval_rewrite_history": rewrite_history,
                "conflict_warnings": [],
            }

        # ── 可选：检测多文档冲突 ──────────────────────────────────────
        from src.rag.evaluation.conflict_detector import detect_document_conflicts
        conflicts = detect_document_conflicts(docs, last_user_message)
        if conflicts:
            print(f"[Retrieval] 检测到 {len(conflicts)} 个文档冲突")

        return {
            "retrieval_context": retrieval_context,
            "retrieved_docs": docs,
            "retrieval_decision": decision,
            "retrieval_decision_reason": decision_reason,
            "retrieval_avg_score": avg_score,
            "retrieval_rewrite_history": rewrite_history,
            "conflict_warnings": conflicts,
        }

    except Exception as e:
        print(f"[Retrieval] 检索出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            "retrieval_context": f"【知识库检索出错】{str(e)}",
            "retrieved_docs": [],
            "retrieval_decision": "error",
            "retrieval_decision_reason": str(e),
            "retrieval_avg_score": 0.0,
            "retrieval_rewrite_history": [],
            "conflict_warnings": [],
        }


def _build_retrieval_context(
    query: str,
    results: List[Tuple[Document, float]],
    grade_result: Any,
) -> str:
    """将检索结果格式化为上下文字符串（供生成阶段用）"""
    if not results:
        return "【知识库检索结果】\n未找到相关文档。"

    lines = ["【知识库检索结果】"]
    lines.append(f"（共检索到 {len(results)} 篇相关文档）\n")

    # 添加评估摘要（若有）
    if grade_result:
        decision = grade_result.decision.value
        lines.append(f"【相关性评估】{decision.upper()} - {grade_result.decision_reason}")
        lines.append("")

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "未知来源") if doc.metadata else "未知来源"
        title = doc.metadata.get("title", source) if doc.metadata else source

        # 截断长文档
        content = doc.page_content
        if len(content) > 800:
            content = content[:800] + "..."

        lines.append(f"--- 文档 {i} ---")
        lines.append(f"来源：{title}")
        lines.append(f"相关度：{score:.4f}")
        lines.append(f"内容：{content}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# generation_agent_node - 生成阶段
# ============================================================

@traced("agent.generation.node")
async def generation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成阶段节点 - 基于评估后的文档，独立生成答案

    职责：
    1. 读取 retrieval_context 和 retrieval_decision
    2. 基于高质量文档生成答案，强制引用文档
    3. 若 retrieval_decision == "no_results"，返回"知识库无相关信息"

    不再：
    - 调用 SkillLoader 的 knowledge agent
    - 让 LLM 自主决定是否需要检索（评估已由 retrieval_agent_node 完成）
    """
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    summary = state.get("summary", "") or ""
    mem0_memories = state.get("mem0_memories", "") or ""
    session_id = state.get("session_id", "default")

    retrieval_context = state.get("retrieval_context", "") or ""
    retrieval_decision = state.get("retrieval_decision", "")
    conflict_warnings = state.get("conflict_warnings", []) or []

    if not last_user_message:
        return {"final_answer": "抱歉，我无法理解您的问题。"}

    # ── NO_RESULTS 快速路径 ──────────────────────────────────────────
    if retrieval_decision == "no_results":
        return {
            "final_answer": (
                "抱歉，知识库中没有找到关于您所提问题的相关文档。"
                "建议您：尝试使用其他关键词进行搜索；调整问题表述方式；"
                "联系管理员补充相关文档。"
            ),
            "sources": "knowledge_base",
            "used_agent": "knowledge_agent",
        }

    try:
        t0 = time.time()

        # ── 构建生成 Prompt ────────────────────────────────────────────
        system_prompt = _build_generation_prompt(
            query=last_user_message,
            retrieval_context=retrieval_context,
            conflict_warnings=conflict_warnings,
            summary=summary,
            mem0_memories=mem0_memories,
        )

        # 获取 Mem0 记忆注入
        messages_with_context = inject_context_to_messages(messages, summary, mem0_memories)

        # ── 调用 LLM 生成 ──────────────────────────────────────────────
        from src.models.llm import get_llm

        llm = get_llm(temperature=0.3)
        response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages_with_context)

        final_answer = response.content.strip()
        gen_time = time.time() - t0
        print(f"[Generation] 生成答案长度: {len(final_answer)} 字符, 耗时: {gen_time:.2f}s")

        # ── 追踪引用质量 ──────────────────────────────────────────────
        if retrieval_decision in ("high", "medium"):
            print(
                f"[Generation] 检索质量: {retrieval_decision.upper()}, "
                f"avg_score={state.get('retrieval_avg_score', 0):.3f}"
            )

        return {
            "final_answer": final_answer,
            "sources": "knowledge_base",
            "used_agent": "knowledge_agent",
        }

    except Exception as e:
        print(f"[Generation] 生成出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_answer": f"生成答案时出错: {str(e)}",
            "sources": "",
            "used_agent": "knowledge_agent",
        }


def _build_generation_prompt(
    query: str,
    retrieval_context: str,
    conflict_warnings: List[str],
    summary: str,
    mem0_memories: str,
) -> str:
    """构建生成阶段的系统提示"""
    lines = [
        "你是一个企业知识库问答助手。",
        "",
    ]

    # 摘要上下文
    if summary:
        lines.append(f"【对话摘要】\n{summary}\n")

    # Mem0 记忆
    if mem0_memories:
        lines.append(f"【用户历史偏好】\n{mem0_memories}\n")

    # 检索结果
    lines.append(f"{retrieval_context}\n")

    # 冲突警告
    if conflict_warnings:
        lines.append("【⚠️ 文档冲突警告】")
        for warn in conflict_warnings:
            lines.append(f"  - {warn}")
        lines.append("请在回答中明确指出这些冲突，并客观呈现不同文档中的不一致信息。\n")

    # 引用要求
    lines.extend([
        "【回答要求】",
        "1. 优先基于以上检索文档回答，若文档中没有相关信息，直接告知用户",
        "2. 必须引用来源，使用 [文档N] 格式标注（如：本政策规定[文档1]）",
        "3. 若存在多个来源，引用优先级：来源标题更精确的 > 相关度更高的",
        "4. 回答使用中文，简洁专业，不超过 500 字",
        "5. 若文档内容不足以完整回答，诚实说明局限性",
        "",
        f"用户问题：{query}",
    ])

    return "\n".join(lines)


# ============================================================
# 兼容性保留：旧的 knowledge_agent_node（向后兼容）
# 仍然使用 SkillLoader ReAct 方式，后续可考虑废弃
# ============================================================

@traced("agent.knowledge.node")
async def knowledge_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    知识 Agent 节点（保留旧版 ReAct 方式，向后兼容）

    新流程请使用 retrieval_agent_node + generation_agent_node。
    """
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    summary = state.get("summary", "") or ""
    mem0_memories = state.get("mem0_memories", "") or ""
    session_id = state.get("session_id", "default")

    if not last_user_message:
        return {"final_answer": "抱歉，我无法理解您的问题。"}

    try:
        from ..skills import get_skill_loader

        loader = get_skill_loader()
        agent = loader.create_agent("knowledge")

        config = {"configurable": {"thread_id": f"{session_id}_knowledge"}}
        messages_with_context = inject_context_to_messages(messages, summary, mem0_memories)

        agent_inject = state.get("agent_inject_prompt", "") or ""
        if agent_inject:
            inject_msg = SystemMessage(content=("【系统指令】" + agent_inject.strip()))
            messages_with_context = [inject_msg] + messages_with_context

        result = await agent.ainvoke({"messages": messages_with_context}, config)
        agent_messages = result.get("messages", [])
        final_answer = agent_messages[-1].content

        print(f"[Knowledge Agent (ReAct)] 生成答案长度: {len(final_answer)} 字符")

        return {
            "final_answer": final_answer,
            "sources": "knowledge_base",
            "used_agent": "knowledge_agent",
            "messages": agent_messages,
        }

    except Exception as e:
        print(f"[Knowledge Agent] 执行出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_answer": f"搜索知识库时出错: {str(e)}",
            "sources": "",
            "used_agent": "knowledge_agent",
        }
