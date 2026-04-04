"""
Supervisor 节点
负责路由决策
"""
import json
from typing import Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from src.models.llm import get_llm
from src.observability import traced


class RouteDecision(TypedDict):
    """Supervisor 路由决策结构化输出"""
    reasoning: str
    next_agent: str   # knowledge_agent / operation_agent / general_agent
    reason: str
    needs_expansion: bool  # 新增：知识检索时是否需要 Query Expansion


@traced("agent.supervisor.node")
async def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor 节点 - 负责路由决策

    分析用户问题，决定将任务路由到哪个 Worker Agent。
    同时判断知识检索时是否需要 Query Expansion（由 Planner 的复杂度结论决定）。

    注意：Planner 判为复杂的任务 → is_complex=True → needs_expansion=True
    （由 knowledge_search 自行判断，仅在 Planner 未调用时兜底）
    """
    llm = get_llm()

    # 获取用户最新消息
    messages = state.get("messages", [])
    if not messages:
        return {
            "next_agent": "general_agent",
            "supervisor_reasoning": "无消息输入",
            "supervisor_reason": "",
            "needs_expansion": False,
        }

    # 获取最后一条用户消息
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    if not last_user_message:
        return {
            "next_agent": "general_agent",
            "supervisor_reasoning": "无法获取用户消息",
            "supervisor_reason": "",
            "needs_expansion": False,
        }

    # ── 获取 Planner 的快速路由结果（已做判断则跳过 LLM）───
    # Planner 已判断了简单任务，直接复用其路由结果
    quick_agent = state.get("_quick_agent")
    if quick_agent:
        print(f"[Supervisor] 复用 Planner 快速路由: {quick_agent}（跳过 LLM）")
        needs_expansion = (quick_agent == "knowledge_agent") and _needs_expansion_for_agent(
            last_user_message
        )
        return {
            "next_agent": quick_agent,
            "supervisor_reasoning": "复用 Planner 快速路由",
            "supervisor_reason": "",
            "needs_expansion": needs_expansion,
        }

    # Planner 没有快速结果，继续正常的 LLM 路由

    # 构建提示词
    prompt = f"""请分析以下用户问题，决定应该路由到哪个 Agent 处理。

用户问题：{last_user_message}

## Agent 职责说明
- **knowledge_agent**: 回答企业知识库相关问题（规章制度、技术文档、FAQ等），从向量数据库检索答案
- **operation_agent**: 执行操作类任务，包括：时间日期查询、数学计算、调用外部工具（如文件系统 MCP 工具）
- **general_agent**: 通用对话、闲聊、意图不明确的问题

## 路由规则
- 询问"现在几点"、"今天日期"、"当前时间"等 → operation_agent（它有时间工具）
- 需要数学计算（"计算"、"多少"）→ operation_agent
- 询问公司制度、政策、文档内容 → knowledge_agent
- 问候、闲聊、无法归类 → general_agent

## Query Expansion 判断（仅 knowledge_agent 需要关注）
如果路由到 knowledge_agent，还需要判断是否需要 Query Expansion（复杂查询主动分解）：
- 对比类（"A和B的区别"、"A vs B"）→ needs_expansion=True
- 列举类（"有哪些X"、"都有些什么"）→ needs_expansion=True
- 多实体类（"A和B的职责"、"张三和李四的"）→ needs_expansion=True
- 多个问号 → needs_expansion=True
- 上述之外的简单单问题 → needs_expansion=False

请严格按照以下 JSON 格式输出：
{{
  "reasoning": "你的分析理由",
  "next_agent": "knowledge_agent/operation_agent/general_agent",
  "reason": "选择该 Agent 的原因",
  "needs_expansion": true/false
}}
"""

    # 使用 with_structured_output 强制结构化输出
    is_complex_from_planner = state.get("is_complex", False)
    try:
        llm_structured = llm.with_structured_output(RouteDecision)
        # 使用 ainvoke
        decision: RouteDecision = await llm_structured.ainvoke(prompt)

        # TypedDict 返回普通 dict，用 [] 访问
        next_agent = decision["next_agent"]
        reasoning = decision["reasoning"]
        reason = decision["reason"]
        # Planner 已判复杂时强制 expansion（Planner 的判断更权威）
        needs_expansion = decision.get("needs_expansion", False) or is_complex_from_planner

    except Exception as e:
        # 降级处理：根据关键词判断
        print(f"结构化输出失败，使用关键词匹配: {e}")
        next_agent, reasoning, reason, needs_expansion = fallback_routing(last_user_message)
        # Planner 已判复杂时强制 expansion
        needs_expansion = needs_expansion or is_complex_from_planner

    print(f"[Supervisor] 路由决策: {next_agent} - {reasoning}")
    print(f"[Supervisor] Query Expansion: {needs_expansion} (Planner 判复杂={is_complex_from_planner})")

    return {
        "next_agent": next_agent,
        "supervisor_reasoning": reasoning,
        "supervisor_reason": reason,
        "needs_expansion": needs_expansion,
        # 透传给下游 agent 的上下文（通过 prompt 注入）
        "agent_inject_prompt": (
            f"\n\n[系统提示] 当前任务复杂度判断：needs_expansion={needs_expansion}\n"
            f"当调用 knowledge_search 工具时，请将 needs_expansion 参数设置为上述值。"
        ) if next_agent == "knowledge_agent" and needs_expansion else "",
    }


def _needs_expansion_for_agent(question: str) -> bool:
    """
    判断 knowledge_agent 是否需要 Query Expansion（纯规则，无 LLM）

    用于快速路径：Supervisor 复用 Planner 快速路由结果时，
    需要判断是否需要 expansion。
    """
    from ..skills.knowledge.scripts.tools import needs_query_expansion
    return needs_query_expansion(question)


def fallback_routing(question: str) -> tuple:
    """降级路由策略 - 基于关键词匹配"""
    question_lower = question.lower()

    # ── Query Expansion 判断 ──
    # 列举类关键词（不需要 Planner，已在 knowledge_search 中同步）
    from ..skills.knowledge.scripts.tools import needs_query_expansion
    exp_needs = needs_query_expansion(question)

    # ── Agent 路由 ──
    # 问候、寒暄
    greetings = ["你好", "hello", "hi", "早上好", "下午好", "晚上好", "最近怎样", "在吗"]
    if any(g in question_lower for g in greetings):
        return "general_agent", "检测到问候语", "问候无需 expansion", exp_needs

    # 需要计算的问题
    calc_keywords = ["计算", "多少", "加", "减", "乘", "除", "等于", "费用", "价格", "统计"]
    if any(k in question_lower for k in calc_keywords):
        return "operation_agent", "检测到计算相关关键词", "计算类无需 expansion", False

    # 需要获取时间的问题
    time_keywords = ["时间", "日期", "现在几点", "几点", "几点了", "今天几号", "当前时间", "星期几", "年", "月", "日"]
    if any(k in question_lower for k in time_keywords):
        return "operation_agent", "检测到时间相关关键词", "时间类无需 expansion", False

    # 回溯历史/上下文相关问题
    history_keywords = ["上一", "之前", "之前的问题", "之前的对话", "前一次", "刚才", "历史"]
    if any(k in question_lower for k in history_keywords):
        return "operation_agent", "检测到历史查询关键词", "历史查询无需 expansion", False

    # 默认路由到知识库
    return "knowledge_agent", "默认路由到知识库", "默认知识库路由", exp_needs


from langchain_core.runnables import RunnableLambda


def route_to_agent(state: Dict[str, Any]) -> str:
    """
    路由执行节点 - 根据 Supervisor 的决策跳转到对应的 Agent

    注意：返回值必须是节点名称（与 graph.py 中的 add_node 名称一致）
    即：retrieval_agent, operation_agent, general_agent

    重构说明：
    - knowledge_agent 节点不再直接执行完整检索+生成
    - 改为 routing 到 retrieval_agent → generation_agent 两阶段
    - 旧版 knowledge_agent_node 保留为向后兼容（ReAct 方式）
    """
    next_agent = state.get("next_agent", "general_agent")

    # knowledge_agent → 路由到 retrieval_agent（两阶段：检索 + 生成）
    if next_agent == "knowledge_agent":
        return "retrieval_agent"

    # operation_agent / general_agent 保持不变
    valid_agents = ["retrieval_agent", "operation_agent", "general_agent"]

    if next_agent in valid_agents:
        return next_agent

    # 降级到 general_agent
    print(f"[route_to_agent] 无效的 agent: {next_agent}，降级到 general_agent")
    return "general_agent"

