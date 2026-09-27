"""主图唯一的请求路由器：规则优先，Jev 可选。

它只回答两个产品问题：请求应该进入哪个已存在分支，以及知识查询是否需要
Query Expansion。不会生成无法执行的计划，也不会触发隐藏的多 Agent fan-out。
Deep Research 仅由调用方显式选择。
"""
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from src.rag.retrieval.query_expander import RuleBasedDecomposer


def _quick_route(question: str) -> str:
    """Compatibility wrapper for the default offline policy."""
    from src.agent.routing.policy import rules_route
    return rules_route(question).handler


def _get_last_user_message(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """产生唯一业务路由，并为知识查询标记是否需要查询扩展。"""
    question = _get_last_user_message(state.get("messages", []))
    if not question:
        return {
            "is_complex": False,
            "needs_expansion": False,
            "plan_steps": [],
            "plan_reasoning": "无用户消息，降级到知识检索",
            "_quick_agent": "knowledge_agent",
            "route_decision": {},
            "harness_report": {},
        }

    from config.settings import get_settings
    from src.agent.routing.jev import decide_route
    from src.agent.routing.policy import RouteDecision
    if state.get("research_mode", "normal") == "deep":
        # Explicit mode never triggers an external routing call.
        decision = RouteDecision(rule_id="explicit_deep", matched=True)
    else:
        recent = [str(m.content)[:1000] for m in state.get("messages", []) if isinstance(m, HumanMessage)][-3:-1]
        decision = await decide_route(question, get_settings(), previous=state.get("route_decision"), recent=recent)
    agent = decision.handler
    needs_expansion = (
        agent == "knowledge_agent"
        and RuleBasedDecomposer.needs_expansion(question)
    )
    return {
        "is_complex": needs_expansion,
        "needs_expansion": needs_expansion,
        "plan_steps": [],
        "plan_reasoning": (
            "知识查询需要规则分解" if needs_expansion else f"路由({decision.provider}): {agent}"
        ),
        "_quick_agent": agent,
        "route_decision": decision.to_dict(),
        "harness_report": {},
    }


def route_from_planner(state: Dict[str, Any]) -> str:
    """把确定性路由结果映射为主图节点。"""
    if state.get("research_mode", "normal") == "deep":
        return "research_agent"

    agent = state.get("_quick_agent", "knowledge_agent")
    if agent == "knowledge_agent":
        return "retrieval_agent"
    if agent in ("operation_agent", "general_agent"):
        return agent
    return "retrieval_agent"


__all__ = ["planner_node", "route_from_planner"]
