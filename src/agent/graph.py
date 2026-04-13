"""
LangGraph Multi-Agent 工作流图
架构：maybe_summarize → retrieve_mem0_memories → Planner → Supervisor → Worker Agents → END
"""
import asyncio
import time as _time
import logging
import traceback
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.types import Send
from langchain_core.messages import HumanMessage, RemoveMessage



logger = logging.getLogger(__name__)

from .agents.supervisor import supervisor_node, route_to_agent
from .agents.knowledge import knowledge_agent_node, retrieval_agent_node, generation_agent_node
from src.rag.retrieval.acl_filter import UserContext
from .agents.operation import operation_agent_node
from .agents.general import general_agent_node
from .agents.planner import (
    planner_node, route_from_planner,
    execute_plan_node, route_execute_plan,
    execute_plan_continue_node,
    knowledge_step_node, operation_step_node, general_step_node,
)
from .checkpointer import get_sync_checkpointer, get_async_checkpointer


# ==================== 状态定义 ====================

class AgentState(MessagesState):
    """
    Agent 状态定义
    继承 MessagesState 以支持自动消息管理
    """
    # 路由决策
    next_agent: str
    supervisor_reasoning: str
    supervisor_reason: str

    # 执行结果
    final_answer: str
    sources: str
    used_agent: str

    # 会话ID（用于在各节点中维护历史）
    session_id: str

    # 语义总结记忆：存储旧对话的压缩摘要
    summary: str

    # ==================== Mem0 记忆字段 ====================
    mem0_memories: str  # 检索到的 Mem0 记忆（格式化后）
    user_id: str        # 用户ID（用于 Mem0 检索）
    user_context: UserContext = None  # ACL 权限上下文（透传至检索链路）

    # ==================== Planner 状态 ====================
    is_complex: bool       # 是否复杂任务
    plan_steps: list       # 计划步骤列表
    plan_reasoning: str    # 计划决策理由
    current_step: int      # 当前执行的步骤索引
    completed_steps: list  # 已完成的步骤
    plan_results: list     # 各步骤的执行结果

    # ==================== Supervisor 决策 ====================
    needs_expansion: bool          # 是否需要 Query Expansion
    agent_inject_prompt: str      # 透传给子 agent 的 prompt 注入内容

    # ==================== Retrieval Pipeline 状态 ====================
    # retrieval_agent_node 执行后写入，供 generation_agent_node 读取
    retrieval_context: str          # 格式化后的检索上下文（供生成用）
    retrieved_docs: list           # 原始检索文档列表（用于冲突检测）
    retrieval_decision: str         # HIGH / MEDIUM / LOW / NO_RESULTS（用于 Supervisor 感知）
    retrieval_decision_reason: str # 评估理由
    retrieval_avg_score: float     # 平均相关分
    retrieval_rewrite_history: list # 查询改写/分解历史
    conflict_warnings: list        # 文档冲突警告列表
    version_source: str             # 版本溯源信息（格式化的 Markdown，供前端展示）


# ==================== 语义总结记忆节点 ====================


async def maybe_summarize_node(state: AgentState) -> Dict[str, Any]:
    """
    语义总结记忆节点（对话压缩）

    当 messages 超过阈值时，用 LLM 将旧消息压缩为摘要，
    并从 state 中移除旧消息，只保留最近 N 条原始消息。
    摘要会滚动累积：新摘要 = 旧摘要 + 本批旧消息的总结。

    本节点在每轮对话开始时运行（入口节点），轻量无副作用。
    当消息数量未超过阈值时，直接透传不做任何操作。
    """
    from config.settings import get_settings
    from src.models.llm import get_llm

    settings = get_settings()
    threshold = settings.summary_threshold
    keep_recent = settings.summary_keep_recent

    messages = state["messages"]
    existing_summary = state.get("summary", "") or ""

    # 未超过阈值，直接透传
    if len(messages) <= threshold:
        return {}

    t0 = _time.time()

    old_messages = messages[:-keep_recent]
    # 若旧消息过少则不值得总结
    if len(old_messages) < 2:
        return {}

    # 格式化旧消息为文本供 LLM 总结
    lines = []
    for msg in old_messages:
        role = "用户"
        msg_type = getattr(msg, "type", None) or type(msg).__name__
        if msg_type in ("ai", "AIMessage"):
            role = "助手"
        elif msg_type in ("tool", "ToolMessage"):
            role = "工具"
        content = getattr(msg, "content", "")
        if content:
            lines.append(f"{role}：{content[:300]}")  # 单条最多取 300 字

    if not lines:
        return {}

    conversation_text = "\n".join(lines)

    existing_part = (
        f"\n\n【已有摘要（请在此基础上追加）】\n{existing_summary}\n"
        if existing_summary else ""
    )

    prompt = (
        f"请将以下对话内容总结成简洁的摘要，保留关键信息、用户意图和重要上下文。"
        f"摘要控制在 300 字以内，使用中文。"
        f"{existing_part}"
        f"\n\n【需要总结的对话片段】\n{conversation_text}"
        f"\n\n摘要："
    )

    llm = get_llm()
    response = await llm.ainvoke(prompt)
    new_summary = response.content.strip()

    # 用 RemoveMessage 删除旧消息（LangGraph add_messages reducer 支持）
    remove_ops = [RemoveMessage(id=m.id) for m in old_messages if getattr(m, "id", None)]

    logger.debug("触发摘要：原 %d 条 → 删除 %d 条，保留 %d 条，耗时 %.2fs",
                 len(messages), len(remove_ops), keep_recent, _time.time() - t0)

    return {
        "summary": new_summary,
        "messages": remove_ops,
    }


# ==================== Mem0 记忆检索节点 ====================


async def retrieve_mem0_memories_node(state: AgentState) -> Dict[str, Any]:
    """
    Mem0 记忆检索节点

    从 Mem0 语义记忆存储中检索与当前对话相关的记忆，
    包括用户偏好、历史交互上下文等信息。
    检索结果格式化后注入到 Agent 上下文中。
    """
    from config.settings import get_settings

    settings = get_settings()

    # 检查是否启用 Mem0
    if not getattr(settings, "mem0_enabled", False):
        logger.debug("Mem0 未启用，跳过检索")
        return {}

    # 问候语跳过（无上下文可关联，且节省一次 LLM 检索开销）
    _GREETING_PATTERNS = (
        r"^(你好|hi|hello|您好|早上好|下午好|晚上好|在吗|嗨|你好！|你好。|hi！|hi。|hello！|hello。)"
    )
    import re
    messages = state.get("messages", [])
    last_msg = ""
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None) or type(msg).__name__
        if msg_type in ("human", "HumanMessage"):
            last_msg = getattr(msg, "content", "")
            break
    if last_msg and re.match(_GREETING_PATTERNS, last_msg.strip(), re.IGNORECASE):
        logger.debug("Mem0 问候语，跳过检索")
        return {}

    try:
        from .memory import get_mem0_manager

        if not messages:
            logger.debug("Mem0 无消息，跳过检索")
            return {}

        logger.debug("Mem0 开始检索，消息数量: %d", len(messages))
        t0 = _time.time()

        # 提取当前用户问题
        current_query = ""
        for msg in reversed(messages):
            msg_type = getattr(msg, "type", None) or type(msg).__name__
            if msg_type in ("human", "HumanMessage"):
                current_query = getattr(msg, "content", "")
                break

        if not current_query:
            return {}

        # 获取用户ID和会话ID
        session_id = state.get("session_id", "default")
        user_id = state.get("user_id", "default_user")

        # 检索 Mem0 记忆 - 同时检索当前会话和跨会话记忆
        mem0_manager = get_mem0_manager()

        # 1. 优先检索当前会话记忆
        t1 = _time.time()
        current_session_memories = await mem0_manager.search(
            query=current_query,
            user_id=user_id,
            session_id=session_id,
            limit=2
        )
        logger.debug("Mem0 检索当前会话: %d 条，耗时 %.2fs", len(current_session_memories), _time.time()-t1)

        # 2. 检索跨会话记忆（不限制会话）
        t2 = _time.time()
        cross_session_memories = await mem0_manager.search(
            query=current_query,
            user_id=user_id,
            session_id=None,  # 跨会话检索
            limit=3
        )
        logger.debug("Mem0 检索跨会话: %d 条，耗时 %.2fs", len(cross_session_memories), _time.time()-t2)

        # 合并记忆，去重
        all_memories = {m["id"]: m for m in current_session_memories}
        for m in cross_session_memories:
            if m["id"] not in all_memories:
                all_memories[m["id"]] = m
        memories = list(all_memories.values())[:5]

        if not memories:
            logger.debug("Mem0 无相关记忆，耗时 %.2fs", _time.time()-t0)
            return {}

        # 格式化记忆
        formatted_memories = mem0_manager.format_memories_for_context(
            memories,
            max_chars=getattr(settings, "mem0_max_context_chars", 500)
        )

        logger.debug("Mem0 检索到 %d 条相关记忆，耗时 %.2fs", len(memories), _time.time()-t0)

        return {"mem0_memories": formatted_memories}

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Mem0 记忆检索失败: {e}")
        return {}


# ==================== 图创建函数 ====================

def create_multi_agent_graph() -> StateGraph:
    """
    创建 Multi-Agent 工作流图

    工作流程：
    1. 接收用户消息
    2. Planner 判断任务复杂度
       - 简单任务 → Supervisor 路由 → Worker Agent → END
       - 复杂任务 → Execute Plan（逐步执行各子步骤）→ END
    """
    workflow = StateGraph(AgentState)

    # 语义摘要节点（入口）
    workflow.add_node("maybe_summarize", maybe_summarize_node)

    # Mem0 记忆检索节点
    workflow.add_node("retrieve_mem0_memories", retrieve_mem0_memories_node)

    # Mem0 记忆保存节点
    workflow.add_node("save_to_mem0", save_to_mem0_node)

    # Worker Agent 节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("knowledge_agent", knowledge_agent_node)  # 保留（向后兼容）
    workflow.add_node("retrieval_agent", retrieval_agent_node)   # 新增：检索阶段
    workflow.add_node("generation_agent", generation_agent_node) # 新增：生成阶段
    workflow.add_node("operation_agent", operation_agent_node)
    workflow.add_node("general_agent", general_agent_node)

    # Planner 节点（任务规划）
    workflow.add_node("planner", planner_node)

    # Execute Plan 节点（复杂任务逐步执行）
    workflow.add_node("execute_plan", execute_plan_node)

    # Send Worker 节点（通过 Send 原语调用的专用 Worker）
    # 这些节点接收 Send 分派的独立步骤，执行后返回 StepResult
    workflow.add_node("knowledge_step_node", knowledge_step_node)
    workflow.add_node("operation_step_node", operation_step_node)
    workflow.add_node("general_step_node", general_step_node)

    # Execute Plan → 继续分发下一批次 或 汇总
    workflow.add_conditional_edges(
        "execute_plan",
        route_execute_plan,
        {
            "continue": "execute_plan",  # 继续分发下一批次
            "execute_plan": "execute_plan",
            "save_to_mem0": "save_to_mem0",
        }
    )

    # 入口：maybe_summarize → retrieve_mem0_memories → planner
    workflow.set_entry_point("maybe_summarize")
    workflow.add_edge("maybe_summarize", "retrieve_mem0_memories")
    workflow.add_edge("retrieve_mem0_memories", "planner")

    # Planner → Supervisor / Worker Agent（简单）或 Execute Plan（复杂）
    # route_from_planner 可能返回：supervisor / general_agent / operation_agent / knowledge_agent / execute_plan
    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "supervisor": "supervisor",
            "execute_plan": "execute_plan",
            "general_agent": "general_agent",
            "operation_agent": "operation_agent",
            "knowledge_agent": "supervisor",  # knowledge → supervisor 再路由到 retrieval_agent
        }
    )

    # Supervisor → Worker Agent
    # route_to_agent: knowledge_agent → retrieval_agent（两阶段）
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "retrieval_agent": "retrieval_agent",  # 检索阶段
            "operation_agent": "operation_agent",
            "general_agent": "general_agent",
        }
    )

    # retrieval_agent → generation_agent（检索完成后生成）
    workflow.add_edge("retrieval_agent", "generation_agent")

    # Worker Agent → save_mem0 → END
    workflow.add_edge("generation_agent", "save_to_mem0")   # 新流程
    workflow.add_edge("knowledge_agent", "save_to_mem0")    # 保留旧版
    workflow.add_edge("operation_agent", "save_to_mem0")
    workflow.add_edge("general_agent", "save_to_mem0")
    workflow.add_edge("save_to_mem0", END)

    return workflow


# ==================== Mem0 记忆保存节点 ====================


async def save_to_mem0_node(state: AgentState) -> Dict[str, Any]:
    """
    保存对话到 Mem0 记忆节点

    在 Agent 执行完成后，将本次对话内容保存到 Mem0，
    供后续会话检索使用。
    """
    from config.settings import get_settings

    settings = get_settings()

    # 检查是否启用 Mem0
    if not getattr(settings, "mem0_enabled", False):
        logger.debug("Mem0 未启用，跳过保存")
        return {}

    # 问候语跳过（无上下文可关联，且节省一次 LLM 保存开销）
    _GREETING_PATTERNS = (
        r"^(你好|hi|hello|您好|早上好|下午好|晚上好|在吗|嗨|你好！|你好。|hi！|hi。|hello！|hello。)"
    )
    import re
    messages = state.get("messages", [])
    last_msg = ""
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None) or type(msg).__name__
        if msg_type in ("human", "HumanMessage"):
            last_msg = getattr(msg, "content", "")
            break
    if last_msg and re.match(_GREETING_PATTERNS, last_msg.strip(), re.IGNORECASE):
        logger.debug("Mem0 问候语，跳过保存")
        return {}

    try:
        from .memory import get_mem0_manager

        if not messages or len(messages) < 2:
            logger.debug("Mem0 消息不足，跳过保存")
            return {}

        logger.debug("Mem0 开始保存，消息数量: %d", len(messages))
        t0 = _time.time()

        # 获取用户ID和会话ID
        session_id = state.get("session_id", "default")
        user_id = state.get("user_id", "default_user")

        # 转换消息格式
        msg_list = []
        for msg in messages[-4:]:  # 只保存最近4条消息
            msg_type = getattr(msg, "type", None) or type(msg).__name__
            role = "user" if msg_type in ("human", "HumanMessage") else "assistant"
            content = getattr(msg, "content", "")
            if content:
                msg_list.append({"role": role, "content": content})

        if not msg_list:
            return {}

        # 保存到 Mem0 - 同时保存当前会话和跨会话记忆
        mem0_manager = get_mem0_manager()

        # 1. 保存当前会话记忆
        t1 = _time.time()
        result = await mem0_manager.add_conversation(
            messages=msg_list,
            user_id=user_id,
            session_id=session_id
        )
        logger.debug("Mem0 保存当前会话，耗时 %.2fs", _time.time()-t1)

        # 2. 保存跨会话记忆（不带 session_id 限制，用于跨会话检索）
        cross_session_msg_list = [
            {"role": m["role"], "content": m["content"]}
            for m in msg_list
        ]
        t2 = _time.time()
        await mem0_manager.add_conversation(
            messages=cross_session_msg_list,
            user_id=user_id,
            session_id=None  # 跨会话记忆
        )
        logger.debug("Mem0 保存跨会话，耗时 %.2fs", _time.time()-t2)

        logger.debug("Mem0 保存完成，耗时 %.2fs", _time.time()-t0)

        return {}

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Mem0 保存记忆失败: {e}")
        return {}


# ==================== 图编译与缓存 ====================

# 同步图实例（MemorySaver，用于测试/run_agent）
_sync_graph = None

# 异步图实例（AsyncSqliteSaver，用于 FastAPI/arun_agent）
_async_graph = None


def get_agent_graph():
    """
    获取同步图实例（单例，MemorySaver）

    适用于测试和脚本调用，不存在事件循环冲突问题。
    注意：MemorySaver 不持久化，进程重启后历史丢失。
    生产环境请使用 get_agent_graph_async()。
    """
    global _sync_graph
    if _sync_graph is None:
        checkpointer = get_sync_checkpointer()
        workflow = create_multi_agent_graph()
        _sync_graph = workflow.compile(checkpointer=checkpointer)
    return _sync_graph


async def get_agent_graph_async():
    """
    获取异步图实例（单例，AsyncSqliteSaver）

    在当前 event loop（FastAPI 的循环）中初始化，
    确保 aiosqlite 连接与 graph.ainvoke() 在同一个 loop 中运行，
    彻底避免跨循环阻塞问题。首次调用后缓存复用。
    """
    global _async_graph
    if _async_graph is None:
        checkpointer = await get_async_checkpointer()
        workflow = create_multi_agent_graph()
        _async_graph = workflow.compile(checkpointer=checkpointer)
    return _async_graph


# ==================== 工具函数 ====================

def _build_run_config(session_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    run_config = config or {}
    if "configurable" not in run_config:
        run_config["configurable"] = {}
    run_config["configurable"]["thread_id"] = session_id
    return run_config


def _extract_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "final_answer": result.get("final_answer", "抱歉，无法生成答案。"),
        "sources": result.get("sources", ""),
        "used_agent": result.get("used_agent", "unknown"),
        "version_source": result.get("version_source", ""),
        "messages": result.get("messages", [])
    }


# ==================== 执行入口函数 ====================

def run_agent(
    input_text: str,
    session_id: str = "default",
    user_id: str = "default_user",
    user_context: UserContext = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    运行 Agent（同步封装，使用 MemorySaver）

    适用于测试和脚本。在新建的 event loop 中运行，
    不存在跨循环冲突。

    Args:
        input_text: 用户输入文本
        session_id: 会话ID
        user_id: 用户ID（用于跨会话记忆）
        user_context: ACL 权限上下文（用于检索权限过滤）
        config: 额外配置
    """
    async def _run():
        graph = get_agent_graph()
        run_config = _build_run_config(session_id, config)
        initial_state = {
            "messages": [HumanMessage(content=input_text)],
            "session_id": session_id,
            "user_id": user_id,
            "user_context": user_context,
        }
        return await graph.ainvoke(initial_state, run_config)

    try:
        result = asyncio.run(_run())
        return _extract_result(result)
    except Exception as e:
        logger.exception("Agent 执行出错: %s", e)
        return {
            "final_answer": f"处理请求时出错: {str(e)}",
            "sources": "",
            "used_agent": "error",
            "messages": [],
        }


async def arun_agent(
    input_text: str,
    session_id: str = "default",
    user_id: str = "default_user",
    user_context: UserContext = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    运行 Agent（异步，使用 AsyncSqliteSaver）

    适用于 FastAPI 生产环境，支持跨请求多轮对话持久化。

    Args:
        input_text: 用户输入文本
        session_id: 会话ID
        user_id: 用户ID（用于跨会话记忆）
        user_context: ACL 权限上下文（用于检索权限过滤）
        config: 额外配置
    """
    graph = await get_agent_graph_async()
    run_config = _build_run_config(session_id, config)
    initial_state = {
        "messages": [HumanMessage(content=input_text)],
        "session_id": session_id,
        "user_id": user_id,
        "user_context": user_context,
    }

    t0 = _time.time()
    try:
        result = await graph.ainvoke(initial_state, run_config)
        logger.info("arun_agent 总耗时: %.2fs", _time.time()-t0)
        return _extract_result(result)
    except Exception as e:
        logger.exception("Agent 执行出错: %s", e)
        return {
            "final_answer": f"处理请求时出错: {str(e)}",
            "sources": "",
            "used_agent": "error",
            "messages": [],
        }
