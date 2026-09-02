"""主图唯一的确定性请求路由器。

它只回答两个产品问题：请求应该进入哪个已存在分支，以及知识查询是否需要
Query Expansion。不会生成无法执行的计划，也不会触发隐藏的多 Agent fan-out。
Deep Research 仅由调用方显式选择。
"""
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from src.rag.retrieval.query_expander import RuleBasedDecomposer


def _quick_route(question: str) -> str:
    """通过明确意图将请求路由到一个已有产品分支。"""
    query = question.lower().strip()

    greetings = (
        "你好", "hello", "hi", "早上好", "下午好", "晚上好",
        "最近怎样", "在吗", "嗨", "您好",
    )
    if any(word in query for word in greetings):
        return "general_agent"

    time_keywords = (
        "现在几点", "几点钟", "当前时间", "今天星期几", "几点了", "今天几号",
    )
    if any(word in query for word in time_keywords):
        return "operation_agent"

    history_keywords = ("上一", "之前的问题", "之前的对话", "前一次", "刚才")
    if any(word in query for word in history_keywords):
        return "operation_agent"

    arithmetic = re.search(
        r"\d+(?:\.\d+)?\s*(?:[+\-*/×÷%^]|乘以|除以|加上|减去)\s*\d+(?:\.\d+)?",
        query,
    )
    if arithmetic or any(word in query for word in ("计算器", "算术")):
        return "operation_agent"

    # 产品默认能力是内部知识问答，未知意图安全降级到检索而非自由生成。
    return "knowledge_agent"


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
        }

    agent = _quick_route(question)
    needs_expansion = (
        agent == "knowledge_agent"
        and RuleBasedDecomposer.needs_expansion(question)
    )
    return {
        "is_complex": needs_expansion,
        "needs_expansion": needs_expansion,
        "plan_steps": [],
        "plan_reasoning": (
            "知识查询需要规则分解" if needs_expansion else f"确定性路由: {agent}"
        ),
        "_quick_agent": agent,
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
