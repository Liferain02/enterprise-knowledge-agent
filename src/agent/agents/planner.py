"""
Planner 节点
负责分析任务复杂度并拆解步骤
支持并行执行独立步骤以提高效率
"""
import re
import json
import traceback
from typing import Dict, Any, List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Send
from src.models.llm import get_llm

import logging

logger = logging.getLogger(__name__)


# 使用 TypedDict 而非 Pydantic BaseModel：
# TypedDict 是 LangGraph 的原生类型，with_structured_output 支持它，
# 且不会触发 Pydantic 在序列化 OpenAI 响应时的 "Expected none but got ..." 警告。
class PlanStep(TypedDict):
    """单个步骤的定义"""
    step_id: int
    description: str
    agent: str        # knowledge_agent / operation_agent / general_agent
    depends_on: List[int]


class TaskPlan(TypedDict):
    """任务规划结构化输出"""
    is_complex: bool
    reasoning: str
    steps: List[PlanStep]
    final_agent: str


class SummaryOutput(TypedDict):
    """汇总步骤结果的结构化输出"""
    final_answer: str


# ==================== 快速复杂度预判 ====================

# 命中任意一条 → 判定为复杂任务（走 LLM planner 拆步骤）
#
# ⚠️ 注意：此列表与 knowledge_search.needs_query_expansion() 保持同步。
# 新增复杂 pattern 时，请同步更新两处，避免 Planner 判 simple 但
# knowledge_search 仍触发 Query Expansion 的不一致行为。
_COMPLEX_PATTERNS = [
    r"对比|比较|区别|差异|异同|不同点",          # 对比类
    r"\bvs\b|VS|versus",                          # 英文对比
    r"分别.*查|分别.*看|各自|各个",               # 多信息并行
    r"先.{0,10}再|先.{0,10}然后|然后再|之后再",  # 顺序执行
    r"总结.{0,15}和|汇总|综合.{0,15}和|梳理",    # 汇总类
    r"第一.{0,20}第二|①.{0,20}②",               # 列举多项
    r"多个.{0,10}政策|多个.{0,10}文档",          # 多文档
    # ── 以下来自 knowledge_search._QUERY_EXPANSION_PATTERNS ──
    r"有哪些|有些什么|都有哪些|都有什么",         # 列举类（Plannner ≤40字快速路径会跳过！）
    r".{2,6}和.{2,6}.{0,8}职责|.{2,6}与.{2,6}.{0,8}区别",  # 多实体职责/区别
]

# 命中任意一条 → 判定为简单任务（直接跳过 LLM planner）
_SIMPLE_PATTERNS = [
    r"^(你好|hi|hello|您好|早上好|下午好|晚上好|在吗|嗨).{0,10}$",  # 问候
    r"^(谢谢|感谢|多谢|thanks|thank you).{0,15}$",                    # 致谢
    r"^(再见|拜拜|bye|晚安|好的|okay|ok|收到).{0,10}$",              # 结束语
    r"^现在(几点|时间|日期)|^今天(几号|星期|日期)|^当前时间",         # 时间查询
]

_COMPILED_COMPLEX = [re.compile(p, re.IGNORECASE) for p in _COMPLEX_PATTERNS]
_COMPILED_SIMPLE  = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]


def _quick_complexity_check(message: str) -> str:
    """
    快速复杂度预判（纯规则，无 LLM 调用，耗时 < 1ms）

    Returns:
        "simple"    确定是简单任务，跳过 LLM planner，节省一次 LLM 调用
        "complex"   确定是复杂任务，走 LLM planner 拆解步骤
        "uncertain" 无法确定，走 LLM planner 精确判断

    策略：宁可漏报 complex（降级为 uncertain → LLM 判断），
    不可误报 simple（跳过 LLM 导致复杂任务未被拆解）。
    """
    msg = message.strip()

    # ── 列举类 / 多实体类 pattern 优先检查 ──────────────────────────
    # 这些 pattern 命中即 complex，不受长度短路影响。
    # 原因："列举类"（有哪些/有些什么）在短消息中也可能是复杂任务，
    # 必须优先于 `len(msg) <= 40` 短路检查，否则 Planner 会漏判。
    _HIGH_PRIORITY_PATTERNS = [
        # 列举类
        re.compile(r"有哪些|有些什么|都有哪些|都有什么", re.IGNORECASE),
        # 对比类关键词（即使很短也是复杂任务）
        re.compile(r"对比|比较|区别|差异|异同", re.IGNORECASE),
        # 顺序执行关键词（即使很短也是复杂任务）
        re.compile(r"先.{0,10}再|先.{0,10}然后|然后再|之后再", re.IGNORECASE),
        # 多实体职责/区别
        re.compile(r".{2,6}和.{2,6}.{0,8}职责|.{2,6}与.{2,6}.{0,8}区别"),
    ]
    for pattern in _HIGH_PRIORITY_PATTERNS:
        if pattern.search(msg):
            return "complex"

    # ── 极短消息 → 必然简单（问候/单词回复）───
    # 但对于上述列举类 query，由于已在上面特殊处理，不会落入此处
    if len(msg) <= 8:
        return "simple"

    # ── 其他复杂信号 → complex ──
    for pattern in _COMPILED_COMPLEX:
        if pattern.search(msg):
            return "complex"

    # ── 简单信号 → simple ──
    for pattern in _COMPILED_SIMPLE:
        if pattern.search(msg):
            return "simple"

    # ── 多个问号 → complex ──
    if msg.count("？") >= 2 or msg.count("?") >= 2:
        return "complex"

    # ── 消息较短（≤ 40字）且无复杂信号 → 大概率 simple ──
    if len(msg) <= 40:
        return "simple"

    # ── 其余交给 LLM 判断 ──
    return "uncertain"


# ==================== Planner 节点 ====================


async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner 节点 - 分析任务复杂度并拆解步骤

    工作逻辑：
    1. 快速规则预判（无 LLM，< 1ms）
       - 确定简单 → 直接返回，跳过 LLM（节省 5-10 秒）
       - 确定复杂 / 不确定 → 走 LLM 精确判断
    2. LLM 判断（仅对 complex / uncertain 触发）
       - 简单任务 → 返回空步骤，后续由 Supervisor 处理
       - 复杂任务 → 拆解成步骤序列，由 execute_plan 执行

    复杂任务示例：
    - "对比 A 政策和 B 政策的差异" → 需要两步：查A政策，查B政策
    - "算一下我下个月能休几天假" → 需要多步：查假期政策，查日历，计算
    - "查询多个文档后总结" → 需要多步：检索多个文档，汇总
    """
    # 获取用户消息
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)

    if not last_user_message:
        return {
            "is_complex": False,
            "plan_steps": [],
            "plan_reasoning": "无用户消息输入"
        }

    # ── 快速路径：规则预判 ──────────────────────────────────────────
    quick_result = _quick_complexity_check(last_user_message)

    if quick_result == "simple":
        logger.debug("快速判断: 简单任务（跳过 LLM）")
        # 简单任务也可快速确定 Agent → 设置 _quick_agent，跳过 Supervisor
        agent = _quick_route(last_user_message)
        return {
            "is_complex": False,
            "plan_steps": [],
            "plan_reasoning": f"快速路由: {agent}",
            "current_step": 0,
            "_quick_agent": agent,
        }

    if quick_result == "complex":
        logger.debug("快速判断: 复杂任务（进入 LLM 拆步骤）")
    else:
        # ── 中等复杂度（< 40字 + 无复杂信号）：快速路由不调 LLM ──────────
        if len(last_user_message) <= 40:
            logger.debug("快速判断: 中等任务（快速路由，不调 LLM）")
            agent = _quick_route(last_user_message)
            return {
                "is_complex": False,
                "plan_steps": [],
                "plan_reasoning": f"快速路由: {agent}",
                "current_step": 0,
                "_quick_agent": agent,
            }
        logger.debug("快速判断: 不确定（进入 LLM 精确判断）")

    # ── LLM 路径：精确判断（仅 complex / uncertain 到达此处）─────────
    llm = get_llm()
    
    # 获取 Mem0 记忆上下文
    mem0_memories = state.get("mem0_memories", "")
    memory_context = f"\n\n## 相关记忆上下文\n以下是你之前与用户交流时记录的相关信息：\n{mem0_memories}\n\n请结合以上记忆上下文来理解用户问题。" if mem0_memories else ""
    
    # 构建分析提示词
    prompt = f"""请分析以下用户问题，判断是否需要拆解成多个步骤来处理。

用户问题：{last_user_message}{memory_context}

## 复杂度判断标准

**简单任务**（不需要拆解）：
- 单一问题，可以一步回答
- 如："现在几点"、"公司年假多少天"、"你好"

**复杂任务**（需要拆解成多步骤）：
- 需要对比多个内容（如"对比A和B"）
- 需要先查询多个信息再计算或总结
- 包含多个子问题
- 如：
  - "对比年假和病假政策的差异" → 需要分别查年假政策、病假政策，然后对比
  - "我下个月能休几天假" → 需要查假期政策 + 查当前月份 + 计算
  - "总结Q1和Q2的业绩表现" → 需要分别查Q1、Q2数据，然后汇总

## Agent 类型说明

- **knowledge_agent**: 知识检索，从向量数据库查询信息
- **operation_agent**: 执行计算、时间查询、工具调用
- **general_agent**: 通用对话、汇总答案

## 输出格式

请严格按照以下 JSON 格式输出：

{{
  "is_complex": true/false,
  "reasoning": "判断理由",
  "steps": [
    {{
      "step_id": 1,
      "description": "步骤描述",
      "agent": "knowledge_agent/operation_agent/general_agent",
      "depends_on": []
    }}
    // ... 更多步骤
  ],
  "final_agent": "general_agent"
}}

**重要**：
- 如果 is_complex 为 false，steps 为空数组
- 如果 is_complex 为 true，至少有2个步骤
- depends_on 填写前置步骤的 step_id，如 [1] 表示依赖第1步
- final_agent 通常是 general_agent，用于汇总最终答案
"""
    
    try:
        llm_structured = llm.with_structured_output(TaskPlan)
        plan: TaskPlan = await llm_structured.ainvoke(prompt)

        # TypedDict 返回普通 dict，直接用 [] 访问
        is_complex = plan["is_complex"]
        reasoning = plan["reasoning"]
        steps = plan.get("steps", [])  # 每个 step 已是 dict

        logger.info("任务复杂度: is_complex=%s steps=%d", is_complex, len(steps))
        logger.debug("判断理由: %s", reasoning[:100])

        if is_complex:
            for s in steps:
                logger.debug("步骤 %s: %s -> %s", s["step_id"], s["description"], s["agent"])

        return {
            "is_complex": is_complex,
            "plan_steps": steps,
            "plan_reasoning": reasoning,
            "current_step": 0,
        }
        
    except Exception as e:
        logger.exception("规划失败: %s", e)
        
        # 降级：认为是非复杂任务
        return {
            "is_complex": False,
            "plan_steps": [],
            "plan_reasoning": f"规划失败，降级为简单任务: {str(e)}",
            "current_step": 0
        }


def _quick_route(question: str) -> str:
    """
    快速路由（纯规则，无 LLM 调用）

    根据关键词快速判断应该路由到哪个 Agent。
    """
    q = question.lower().strip()

    # 问候、寒暄
    greetings = ["你好", "hello", "hi", "早上好", "下午好", "晚上好", "最近怎样", "在吗", "嗨", "您好"]
    if any(g in q for g in greetings):
        return "general_agent"

    # 明确的时间查询（不涉及知识库）
    time_keywords = ["现在几点", "几点钟", "当前时间", "今天星期几", "几点了", "今天几号"]
    if any(k in q for k in time_keywords):
        return "operation_agent"

    # 明确的历史查询
    history_keywords = ["上一", "之前的问题", "之前的对话", "前一次", "刚才"]
    if any(k in q for k in history_keywords):
        return "operation_agent"

    # 明确的计算（只涉及纯数字计算）
    calc_only = ["1+1", "2*3", "计算器", "算术"]
    if any(k in q for k in calc_only):
        return "operation_agent"

    # 默认路由到知识库（处理公司制度、政策、具体信息查询）
    # 包括：年假、病假、报销、请假、流程、制度等问题
    return "knowledge_agent"


def _get_last_user_message(messages: list) -> str:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ==================== 路由函数 ====================

# Planner 返回的 state 中包含 is_complex 标记，
# Supervisor / Agent 节点可以通过 state["is_complex"] 访问此信息，
# 用于下游决策（如 knowledge_search 是否需要 Query Expansion）。
def route_from_planner(state: Dict[str, Any]) -> str:
    """
    从 Planner 路由到下一个节点

    - 复杂任务 -> execute_plan
    - 简单任务 + 有 _quick_agent（快速路由）-> 直接跳转到对应 Worker Agent，跳过 Supervisor
    - 简单任务 + 无 _quick_agent -> 交给 Supervisor 决策
    """
    is_complex = state.get("is_complex", False)

    if is_complex:
        return "execute_plan"

    # Planner 已经通过快速路由确定了 Agent → 直接跳转，跳过 Supervisor（省去一次 LLM 调用）
    quick_agent = state.get("_quick_agent")
    if quick_agent:
        logger.debug("快速路由: %s，跳过 Supervisor", quick_agent)
        return quick_agent

    # Planner 无法快速确定 → 交给 Supervisor 决策
    return "supervisor"


# ==================== 计划执行节点 ====================

# ==================== Send-based Fan-out/Fan-in 执行节点 ====================

# 是否使用 LangGraph Send 模式（替代 asyncio.gather）
# Send 模式：LangGraph 自动完成 fan-out（并行分发）和 fan-in（结果收集）
# 优势：失败隔离（单步骤失败不影响其他步骤）、断点续跑、天然状态可见性
USE_LANGGRAPH_SEND = True

# 后备：是否启用 asyncio.gather 并行模式（当 Send 模式不可用时）
PARALLEL_EXECUTION_ENABLED = False



async def execute_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行计划步骤的节点

    支持两种并行执行模式：
    - Send 模式（默认）：使用 LangGraph Send 原语，实现真正的 fan-out/fan-in
      优势：失败隔离、断点续跑、天然状态管理
    - Gather 模式（后备）：使用 asyncio.gather()，保留原有逻辑
    """
    global PARALLEL_EXECUTION_ENABLED, USE_LANGGRAPH_SEND

    plan_steps = state.get("plan_steps", [])
    current_step = state.get("current_step", 0)

    if not plan_steps:
        return {"current_step": -1}

    # 检查是否启用 Send 模式
    if USE_LANGGRAPH_SEND and current_step == 0:
        return await _execute_plan_with_send(state)

    # 后备：使用 asyncio.gather 模式
    if PARALLEL_EXECUTION_ENABLED:
        return await _execute_plan_parallel(state)

    return await _execute_plan_sequential(state)


async def _execute_plan_with_send(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用 LangGraph Send 实现真正的 fan-out/fan-in 并行执行。

    工作原理：
    1. analyze_dependencies 将步骤分批（拓扑排序）
    2. 同一批次内的步骤通过 Send 同时分发（fan-out）
    3. LangGraph 自动等待所有分支完成（fan-in）
    4. 收集所有批次结果后进入下一步

    关键优势：
    - 失败隔离：一个步骤崩溃不会影响同批次其他步骤
    - 断点续跑：图结构天然支持部分节点恢复
    - 状态可见：每个分支独立更新状态

    注意：本函数只返回 Send 列表，实际执行由 LangGraph 调度。
    返回 Send 列表时，LangGraph 会在当前节点完成后立即分发所有 Send。
    """
    from .parallel_executor import analyze_step_dependencies

    plan_steps = state.get("plan_steps", [])
    messages = state.get("messages", [])
    session_id = state.get("session_id", "default")
    summary = state.get("summary", "") or ""
    mem0_memories = state.get("mem0_memories", "") or ""

    if not plan_steps:
        return {"current_step": -1}

    # ── 预注入 Mem0 + 摘要上下文 ────────────────────────────────────
    from ._utils import inject_worker_context
    messages_with_context = inject_worker_context(messages, summary, mem0_memories)

    # ── 拓扑排序分批 ────────────────────────────────────────────────
    batches = analyze_step_dependencies(plan_steps)
    logger.info("Execute Plan (Send): 分 %d 批处理 %d 个步骤", len(batches), len(plan_steps))

    if len(batches) == 1:
        # 单批次：直接用 Send 分发所有步骤
        return _send_batch(plan_steps, batches[0], messages_with_context, session_id, summary)

    # 多批次：使用 Send 循环分发
    # 每个 Send 会使 LangGraph 在当前节点完成后立即分发到目标节点
    # 注意：返回多个 Send 时，LangGraph 会将它们合并为一个 Send 列表
    # 然后同时分发——这意味着第一批次会立即执行，其他批次会在第一批次完成后执行
    #
    # 实现方式：第一个批次返回 Send，后续批次通过 state 记录，
    # route_execute_plan 判断 current_batch_idx 决定是否继续分发
    first_batch = batches[0]
    remaining_batches = batches[1:]

    # 记录剩余批次信息到 state
    remaining_batches_data = [
        {"indices": list(batch), "steps": [plan_steps[i] for i in batch]}
        for batch in remaining_batches
    ]

    # Fan-out：第一批次的所有步骤通过 Send 并行分发
    first_send = _send_batch(plan_steps, first_batch, messages_with_context, session_id, summary)

    # 如果只有一批，直接返回 Send
    if not remaining_batches_data:
        return first_send

    # 多批次：返回 Send 并更新 state 记录后续批次
    result = dict(first_send) if isinstance(first_send, dict) else {}
    if not isinstance(first_send, list):
        # 第一个 Send 可能返回 dict（single Send）
        result = first_send if isinstance(first_send, list) else [first_send]

    return {
        **result,  # 第一批次的 Send 列表
        "current_batch_index": 1,  # 下一批次索引
        "remaining_batches": remaining_batches_data,  # 剩余批次信息
        "pending_batch_count": len(remaining_batches_data),
    }


def _send_batch(
    plan_steps: List[Dict[str, Any]],
    batch: set,
    messages_with_context: list,
    session_id: str,
    summary: str,
) -> List[Send]:
    """
    将一个批次的所有步骤通过 Send 分发到对应的 Worker 节点。

    Args:
        plan_steps: 所有步骤列表
        batch: 当前批次的步骤索引集合
        messages_with_context: 预注入上下文的消息列表
        session_id: 会话 ID
        summary: 对话摘要

    Returns:
        Send 列表，每个 Send 对应一个步骤
    """
    sends = []
    for step_idx in batch:
        step = plan_steps[step_idx]
        agent = step.get("agent", "general_agent")

        # 映射 agent → 节点名
        node_map = {
            "knowledge_agent": "knowledge_step_node",
            "operation_agent": "operation_step_node",
            "general_agent": "general_step_node",
        }
        node_name = node_map.get(agent, "general_step_node")

        # 通过 Send 将步骤分派到对应的 Worker 节点
        # 注意：LangGraph Send 会将 state 的副本传递给目标节点
        # 这里我们覆盖 messages 为预注入后的版本，避免每个 Worker 重复序列化
        step_state = {
            # 步骤数据（供 Worker 节点使用）
            "step": step,
            "step_id": step["step_id"],
            "step_agent": agent,
            # 预注入的上下文（避免重复序列化）
            "messages": messages_with_context,
            "session_id": session_id,
            "summary": summary,
        }

        sends.append(Send(node_name, step_state))

    return sends



async def execute_plan_continue_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    继续执行下一批次的节点。

    当 execute_plan 通过 Send 完成了第一批次的 fan-out 后，
    本节点负责分发下一批次。
    """
    current_batch_index = state.get("current_batch_index", 0)
    remaining_batches = state.get("remaining_batches", [])

    if current_batch_index >= len(remaining_batches):
        # 所有批次完成，进入汇总
        return await _finalize_plan(state)

    # 获取当前批次
    batch_data = remaining_batches[current_batch_index]
    batch_steps = batch_data["steps"]
    messages = state.get("messages", [])
    session_id = state.get("session_id", "default")
    summary = state.get("summary", "") or ""

    from ._utils import inject_worker_context
    messages_with_context = inject_worker_context(messages, summary, "")

    # Send 当前批次
    sends = []
    for step in batch_steps:
        agent = step.get("agent", "general_agent")
        node_map = {
            "knowledge_agent": "knowledge_step_node",
            "operation_agent": "operation_step_node",
            "general_agent": "general_step_node",
        }
        node_name = node_map.get(agent, "general_step_node")
        sends.append(Send(node_name, {
            "step": step,
            "step_id": step["step_id"],
            "step_agent": agent,
            "messages": messages_with_context,
            "session_id": session_id,
            "summary": summary,
        }))

    # 更新状态，进入下一批次
    next_batch_index = current_batch_index + 1
    remaining = [
        bd for i, bd in enumerate(remaining_batches)
        if i >= next_batch_index
    ]

    return {
        "send_batch": sends,
        "current_batch_index": next_batch_index,
        "remaining_batches": remaining,
    }


async def _finalize_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    汇总所有步骤结果，生成最终答案。
    """
    plan_steps = state.get("plan_steps", [])
    messages = state.get("messages", [])

    # 收集所有步骤结果（从 state 中的 plan_results）
    plan_results = state.get("plan_results", [])
    if not plan_results:
        plan_results = []

    def _step_to_dict(r) -> dict:
        if hasattr(r, "step_id"):
            return {
                "step_id": r.step_id,
                "description": r.description,
                "agent": r.agent,
                "result": r.result,
                "sources": r.sources,
                "success": r.success,
                "error": r.error,
            }
        return r

    # 汇总
    final_answer = await _summarize_results(
        messages,
        [_step_to_dict(r) for r in plan_results]
    )

    completed = [
        (r.step_id if hasattr(r, 'step_id') else r.get("step_id"))
        for r in plan_results
        if (hasattr(r, 'success') and r.success) or r.get("success")
    ]

    return {
        "current_step": -1,
        "completed_steps": completed,
        "plan_results": plan_results,
        "final_answer": final_answer,
        "used_agent": "planner_send",
    }


# ==================== Send Worker 节点（接收 Send 分派的步骤）====================

async def knowledge_step_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Knowledge Worker 节点（通过 Send 调用）"""
    step = state.get("step", {})
    messages = state.get("messages", [])
    session_id = state.get("session_id", "default")

    if not step or not messages:
        return _make_step_result(step, "", success=False, error="无步骤数据或消息")

    try:
        from .knowledge import knowledge_agent_node

        result = await knowledge_agent_node({
            "messages": messages,
            "session_id": session_id,
        })

        from ._schemas import serialize_step_result, _extract_numeric_data
        return serialize_step_result({
            "step_id": step.get("step_id", 0),
            "description": step.get("description", ""),
            "agent": "knowledge_agent",
            "result": result.get("final_answer", ""),
            "sources": result.get("sources", ""),
            "success": True,
            "structured_data": _extract_numeric_data(result.get("final_answer", "")),
        })

    except Exception as e:
        return _make_step_result(step, "", success=False, error=str(e))


async def operation_step_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Operation Worker 节点（通过 Send 调用）"""
    step = state.get("step", {})
    messages = state.get("messages", [])
    session_id = state.get("session_id", "default")

    if not step or not messages:
        return _make_step_result(step, "", success=False, error="无步骤数据或消息")

    try:
        from .operation import operation_agent_node

        result = await operation_agent_node({
            "messages": messages,
            "session_id": session_id,
        })

        from ._schemas import serialize_step_result, _extract_numeric_data
        return serialize_step_result({
            "step_id": step.get("step_id", 0),
            "description": step.get("description", ""),
            "agent": "operation_agent",
            "result": result.get("final_answer", ""),
            "success": True,
            "structured_data": _extract_numeric_data(result.get("final_answer", "")),
        })

    except Exception as e:
        return _make_step_result(step, "", success=False, error=str(e))


async def general_step_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """General Worker 节点（通过 Send 调用）"""
    step = state.get("step", {})
    messages = state.get("messages", [])
    session_id = state.get("session_id", "default")

    if not step or not messages:
        return _make_step_result(step, "", success=False, error="无步骤数据或消息")

    try:
        from .general import general_agent_node

        result = await general_agent_node({
            "messages": messages,
            "session_id": session_id,
        })

        from ._schemas import serialize_step_result
        return serialize_step_result({
            "step_id": step.get("step_id", 0),
            "description": step.get("description", ""),
            "agent": "general_agent",
            "result": result.get("final_answer", ""),
            "success": True,
        })

    except Exception as e:
        return _make_step_result(step, "", success=False, error=str(e))


def _make_step_result(step: dict, result: str, success: bool, error: str = None) -> dict:
    """创建步骤结果 dict（用于 Send 节点返回值）"""
    return {
        "step_id": step.get("step_id", 0),
        "description": step.get("description", ""),
        "agent": step.get("agent", "unknown"),
        "result": result,
        "success": success,
        "error": error,
    }
    
async def _execute_plan_parallel(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    并行执行计划步骤
    分析依赖关系，将独立步骤并行执行
    """
    from .parallel_executor import get_parallel_executor, analyze_step_dependencies

    plan_steps = state.get("plan_steps", [])
    messages = state.get("messages", [])
    session_id = state.get("session_id", "default")
    summary = state.get("summary", "") or ""
    mem0_memories = state.get("mem0_memories", "") or ""

    if not plan_steps:
        return {"current_step": -1}

    # ── 预注入 Mem0 + 摘要上下文（只注入一次，供所有 Worker 复用）──────
    # 这样每个并行步骤不必各自重复序列化 Mem0 + 摘要，节省 token 开销
    from ._utils import inject_worker_context
    messages_with_context = inject_worker_context(messages, summary, mem0_memories)

    # 分析依赖关系并显示执行计划
    batches = analyze_step_dependencies(plan_steps)
    logger.info("Execute Plan: 分 %d 批处理 %d 个步骤", len(batches), len(plan_steps))

    # 使用并行执行器（传入预注入的上下文消息）
    executor = get_parallel_executor()
    plan_results = await executor.execute_parallel(
        plan_steps,
        messages_with_context,  # ← 传入预注入后的消息
        session_id,
        summary  # summary 仍传入，用于降级备选
    )

    # 转换结果格式（StepResult → dict，保持向后兼容）
    # 注意：StepResult 有 .step_id / .success 属性，直接用 .get() 会报错
    def _step_to_dict(r) -> dict:
        if hasattr(r, "step_id"):  # StepResult 对象
            return {
                "step_id": r.step_id,
                "description": r.description,
                "agent": r.agent,
                "result": r.result,
                "sources": r.sources,
                "success": r.success,
                "error": r.error,
            }
        return r  # 已经是 dict

    completed = [r.step_id for r in plan_results if hasattr(r, 'step_id') and r.success]

    # 汇总结果
    final_answer = await _summarize_results(
        messages,
        [_step_to_dict(r) for r in plan_results]
    )

    # 追加最终答案到 messages，供 save_to_mem0_node 写入记忆
    updated_messages = messages + [AIMessage(content=final_answer)]

    return {
        "current_step": -1,
        "completed_steps": completed,
        "plan_results": plan_results,
        "final_answer": final_answer,
        "messages": updated_messages,
        "used_agent": "planner_parallel"
    }


async def _execute_plan_sequential(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    顺序执行计划步骤（原有逻辑）
    """
    from .knowledge import knowledge_agent_node
    from .operation import operation_agent_node
    from .general import general_agent_node
    from ._utils import inject_worker_context

    plan_steps = state.get("plan_steps", [])
    current_step = state.get("current_step", 0)
    messages = state.get("messages", [])
    summary = state.get("summary", "") or ""
    mem0_memories = state.get("mem0_memories", "") or ""

    if not plan_steps or current_step >= len(plan_steps):
        return {"current_step": -1}

    # ── 预注入 Mem0 + 摘要上下文（只注入一次，供所有 Worker 复用）
    messages_with_context = inject_worker_context(messages, summary, mem0_memories)

    # 获取当前步骤
    step = plan_steps[current_step]
    logger.debug("执行步骤 %s: %s", step["step_id"], step["description"])
    logger.debug("分配给 Agent: %s", step["agent"])

    # 根据步骤指定的 agent 执行
    agent_name = step.get("agent", "general_agent")
    step_result = ""
    sources = ""

    try:
        if agent_name == "knowledge_agent":
            sub_question = step['description']
            sub_messages = messages_with_context + [HumanMessage(content=sub_question)]

            result = await knowledge_agent_node({
                "messages": sub_messages,
                "session_id": state.get("session_id", "default")
            })
            step_result = result.get("final_answer", "")
            sources = result.get("sources", "")

        elif agent_name == "operation_agent":
            sub_question = step['description']
            sub_messages = messages_with_context + [HumanMessage(content=sub_question)]

            result = await operation_agent_node({
                "messages": sub_messages,
                "session_id": state.get("session_id", "default")
            })
            step_result = result.get("final_answer", "")

        elif agent_name == "general_agent":
            sub_question = step['description']
            sub_messages = messages_with_context + [HumanMessage(content=sub_question)]

            result = await general_agent_node({
                "messages": sub_messages,
                "session_id": state.get("session_id", "default")
            })
            step_result = result.get("final_answer", "")

        logger.debug("步骤 %s 结果: %s...", step["step_id"], step_result[:50])

    except Exception as e:
        logger.warning("Execute Plan 步骤执行失败: %s", e)
        step_result = f"步骤执行出错: {str(e)}"

    # 收集步骤结果（使用 StepResult 结构化格式）
    from ._schemas import serialize_step_result
    plan_results = state.get("plan_results", [])
    plan_results.append(serialize_step_result({
        "step_id": step['step_id'],
        "description": step['description'],
        "agent": agent_name,
        "result": step_result,
        "sources": sources,
    }))

    # 更新已完成步骤
    completed = state.get("completed_steps", [])
    completed.append(step['step_id'])

    # 检查是否还有更多步骤
    next_step = current_step + 1
    if next_step >= len(plan_steps):
        # 所有步骤完成，汇总结果
        logger.info("Execute Plan: 所有步骤完成，开始汇总")

        final_answer = await _summarize_results(
            state.get("messages", []),
            plan_results
        )

        # 追加最终答案到 messages，供 save_to_mem0_node 写入记忆
        updated_messages = state.get("messages", []) + [AIMessage(content=final_answer)]

        return {
            "current_step": -1,
            "completed_steps": completed,
            "plan_results": plan_results,
            "final_answer": final_answer,
            "messages": updated_messages,
            "used_agent": "planner"
        }
    else:
        logger.debug("Execute Plan: 步骤 %s 完成", step["step_id"])
        return {
            "current_step": next_step,
            "completed_steps": completed,
            "plan_results": plan_results
        }


async def _summarize_results(messages: list, plan_results: list) -> str:
    """
    汇总所有步骤的结果，生成最终答案
    """
    llm = get_llm()
    user_question = _get_last_user_message(messages)
    
    # 构建汇总提示
    results_text = ""
    for r in plan_results:
        results_text += f"\n步骤 {r['step_id']}: {r['description']}\n结果: {r['result']}\n"
    
    prompt = f"""用户问题：{user_question}

各步骤执行结果：
{results_text}

请根据以上各步骤的结果，生成一个完整、连贯的最终答案。

要求：
1. 直接回答用户的问题
2. 整合各步骤的结果
3. 保持答案连贯流畅
4. 不要提及"步骤1"、"步骤2"等内部过程

最终答案："""

    try:
        llm_structured = llm.with_structured_output(SummaryOutput)
        summary: SummaryOutput = await llm_structured.ainvoke(prompt)
        # TypedDict 返回普通 dict
        return summary["final_answer"]
    except Exception as e:
        # 如果汇总失败，简单拼接结果
        logger.warning("Execute Plan 汇总失败: %s", e)
        return "\n\n".join([r['result'] for r in plan_results])


def route_execute_plan(state: Dict[str, Any]) -> str:
    """
    从计划执行节点路由

    支持两种模式：
    - Send 模式（有 current_batch_index）：
      有剩余批次 → 继续分发下一批次（返回 "execute_plan" 触发循环）
      无剩余批次 → 进入汇总
    - Gather 模式（无 current_batch_index）：
      current_step == -1 → 所有步骤完成，保存到 Mem0
      否则 → 继续执行下一个步骤
    """
    # Send 模式：检查是否还有剩余批次
    current_batch_index = state.get("current_batch_index", 0)
    remaining_batches = state.get("remaining_batches", [])

    if current_batch_index > 0 or remaining_batches:
        # Send 模式：检查是否还有剩余批次需要分发
        if remaining_batches:
            return "execute_plan"  # 继续分发下一批次（触发 Send 循环）
        else:
            return "save_to_mem0"  # 所有批次完成

    # Gather/顺序模式
    current_step = state.get("current_step", 0)

    if current_step == -1:
        return "save_to_mem0"
    else:
        return "execute_plan"
