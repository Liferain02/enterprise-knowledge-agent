"""
Supervisor 节点
负责路由决策
"""
import json
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from core.llm import get_llm


class RouteDecision(BaseModel):
    """Supervisor 路由决策结构化输出"""
    reasoning: str = Field(description="分析问题的理由")
    next_agent: str = Field(description="下一个处理的 Agent: knowledge_agent/operation_agent/general_agent")
    reason: str = Field(description="选择该 Agent 的原因")


async def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor 节点 - 负责路由决策
    
    分析用户问题，决定将任务路由到哪个 Worker Agent
    使用 with_structured_output 强制输出结构化 JSON
    
    改为 async def，与其他节点保持一致
    """
    llm = get_llm()
    
    # 获取用户最新消息
    messages = state.get("messages", [])
    if not messages:
        return {
            "next_agent": "general_agent",
            "supervisor_reasoning": "无消息输入"
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
            "supervisor_reasoning": "无法获取用户消息"
        }
    
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

请严格按照以下 JSON 格式输出：
{{
  "reasoning": "你的分析理由",
  "next_agent": "knowledge_agent/operation_agent/general_agent",
  "reason": "选择该 Agent 的原因"
}}
"""
    
    # 使用 with_structured_output 强制结构化输出
    try:
        llm_structured = llm.with_structured_output(RouteDecision)
        # 使用 ainvoke
        decision: RouteDecision = await llm_structured.ainvoke(prompt)
        
        next_agent = decision.next_agent
        reasoning = decision.reasoning
        reason = decision.reason
        
    except Exception as e:
        # 降级处理：根据关键词判断
        print(f"结构化输出失败，使用关键词匹配: {e}")
        next_agent, reasoning = fallback_routing(last_user_message)
        reason = "自动降级路由"
    
    print(f"[Supervisor] 路由决策: {next_agent} - {reasoning}")
    
    return {
        "next_agent": next_agent,
        "supervisor_reasoning": reasoning,
        "supervisor_reason": reason
    }


def fallback_routing(question: str) -> tuple:
    """降级路由策略 - 基于关键词匹配"""
    question_lower = question.lower()
    
    # 问候、寒暄
    greetings = ["你好", "hello", "hi", "早上好", "下午好", "晚上好", "最近怎样", "在吗"]
    if any(g in question_lower for g in greetings):
        return "general_agent", "检测到问候语"
    
    # 需要计算的问题
    calc_keywords = ["计算", "多少", "加", "减", "乘", "除", "等于", "费用", "价格", "统计"]
    if any(k in question_lower for k in calc_keywords):
        return "operation_agent", "检测到计算相关关键词"
    
    # 需要获取时间的问题
    time_keywords = ["时间", "日期", "现在几点", "几点", "几点了", "今天几号", "当前时间", "星期几", "年", "月", "日"]
    if any(k in question_lower for k in time_keywords):
        return "operation_agent", "检测到时间相关关键词"
    
    # 回溯历史/上下文相关问题
    history_keywords = ["上一", "之前", "之前的问题", "之前的对话", "前一次", "刚才", "历史"]
    if any(k in question_lower for k in history_keywords):
        return "operation_agent", "检测到历史查询关键词"
    
    # 默认路由到知识库
    return "knowledge_agent", "默认路由到知识库"


from langchain_core.runnables import RunnableLambda


def route_to_agent(state: Dict[str, Any]) -> str:
    """
    路由执行节点 - 根据 Supervisor 的决策跳转到对应的 Agent

    注意：返回值必须是节点名称（与 graph.py 中的 add_node 名称一致）
    即：knowledge_agent, operation_agent, general_agent
    """
    next_agent = state.get("next_agent", "general_agent")

    # 直接返回 Agent 名称（与节点名称一致）
    # 可选值: knowledge_agent, operation_agent, general_agent
    valid_agents = ["knowledge_agent", "operation_agent", "general_agent"]

    if next_agent in valid_agents:
        return next_agent

    # 降级到 general_agent
    print(f"[route_to_agent] 无效的 agent: {next_agent}，降级到 general_agent")
    return "general_agent"


async def aroute_to_agent(state: Dict[str, Any]) -> str:
    """
    异步版本的路由函数 - 用于 astream
    """
    return route_to_agent(state)
