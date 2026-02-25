"""
Multi-Agent 节点模块
定义 Supervisor、Knowledge Agent、Operation Agent、General Agent 的核心逻辑
"""
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from core.llm import get_llm
from agent.tools import get_knowledge_tool, get_all_agent_tools, get_tool_by_name
from agent.prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    KNOWLEDGE_AGENT_SYSTEM_PROMPT,
    OPERATION_AGENT_SYSTEM_PROMPT,
    GENERAL_AGENT_SYSTEM_PROMPT,
)


# ==================== Supervisor 结构化输出模型 ====================

class RouteDecision(BaseModel):
    """Supervisor 路由决策结构化输出"""
    reasoning: str = Field(description="分析问题的理由")
    next_agent: str = Field(description="下一个处理的 Agent: knowledge_agent/operation_agent/general_agent")
    reason: str = Field(description="选择该 Agent 的原因")


# ==================== Supervisor 节点 ====================

def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor 节点 - 负责路由决策
    
    分析用户问题，决定将任务路由到哪个 Worker Agent
    使用 with_structured_output 强制输出结构化 JSON
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
        decision: RouteDecision = llm_structured.invoke(prompt)
        
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
    time_keywords = ["时间", "日期", "现在几点", "今天几号", "当前时间"]
    if any(k in question_lower for k in time_keywords):
        return "operation_agent", "检测到时间相关关键词"
    
    # 默认路由到知识库
    return "knowledge_agent", "默认路由到知识库"


# ==================== Knowledge Agent 节点 ====================

def knowledge_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Knowledge Agent 节点 - 负责知识检索和回答
    
    使用 RAG 从知识库中检索信息并生成答案
    """
    llm = get_llm()
    knowledge_tool = get_knowledge_tool()
    
    # 获取用户最新消息
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)
    
    if not last_user_message:
        return {
            "final_answer": "抱歉，我无法理解您的问题。"
        }
    
    # 检索知识库
    try:
        search_result = knowledge_tool.invoke({
            "query": last_user_message,
            "top_k": 5
        })
    except Exception as e:
        search_result = f"知识库检索出错: {str(e)}"
    
    # 构建生成提示词
    if "未在知识库中找到相关内容" in search_result or "出错" in search_result:
        prompt = f"""用户问题：{last_user_message}

知识库检索结果：{search_result}

请基于以上结果回答用户问题。如果知识库中没有相关信息，请明确告知用户，并建议用户换一种方式提问或联系管理员。"""
    else:
        prompt = f"""用户问题：{last_user_message}

从知识库检索到的相关内容：
{search_result}

请根据以上检索结果回答用户问题。
要求：
1. 直接回答用户问题
2. 在答案中标注信息来源
3. 如果没有找到相关信息，明确说明"""

    # 调用 LLM 生成答案
    response = llm.invoke(prompt)
    
    print(f"[Knowledge Agent] 生成答案长度: {len(response.content)} 字符")
    
    return {
        "final_answer": response.content,
        "sources": search_result,
        "used_agent": "knowledge_agent"
    }


# ==================== Operation Agent 节点 ====================

def operation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Operation Agent 节点 - 负责执行操作任务
    
    使用原生 Tool Calling：llm.bind_tools() + 手动执行工具
    """
    llm = get_llm()
    
    # 获取所有可用工具
    tools = get_all_agent_tools()
    
    # 获取用户最新消息
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)
    
    if not last_user_message:
        return {
            "final_answer": "抱歉，我无法理解您的问题。"
        }
    
    # 构建消息列表（包含历史）
    chat_messages = [
        SystemMessage(content=OPERATION_AGENT_SYSTEM_PROMPT),
    ]
    
    # 添加对话历史（最近 10 条）
    for msg in messages[-10:]:
        chat_messages.append(msg)
    
    # 绑定工具到 LLM（使用 bind_tools）
    llm_with_tools = llm.bind_tools(tools)
    
    # 第一次调用：让 LLM 决定是否调用工具
    response = llm_with_tools.invoke(chat_messages)
    
    # 检查是否有 tool_calls
    final_answer = ""
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # 有工具调用，执行工具并获取结果
        tool_results = []
        
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('args', {})
            
            # 获取工具
            tool = get_tool_by_name(tool_name)
            
            if tool:
                try:
                    # 执行工具
                    result = tool.invoke(tool_args)
                    tool_results.append(f"【{tool_name}】执行结果: {result}")
                    print(f"[Operation Agent] 工具 {tool_name} 执行成功")
                except Exception as e:
                    tool_results.append(f"【{tool_name}】执行出错: {str(e)}")
                    print(f"[Operation Agent] 工具 {tool_name} 执行失败: {e}")
            else:
                tool_results.append(f"【{tool_name}】工具不存在")
        
        # 将工具结果拼接到消息中，再次调用 LLM 生成最终答案
        tool_result_message = AIMessage(content=json.dumps(response.tool_calls))
        tool_results_content = "\n".join(tool_results)
        
        chat_messages.append(tool_result_message)
        chat_messages.append(HumanMessage(content=f"工具执行结果:\n{tool_results_content}\n\n请基于以上工具执行结果，给出最终答案。"))
        
        # 第二次调用：基于工具结果生成最终答案
        final_response = llm.invoke(chat_messages)
        final_answer = final_response.content
        
    else:
        # 没有工具调用，直接使用 LLM 的回复
        final_answer = response.content
    
    print(f"[Operation Agent] 生成答案长度: {len(final_answer)} 字符")
    
    return {
        "final_answer": final_answer,
        "used_agent": "operation_agent"
    }


# ==================== General Agent 节点 ====================

def general_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    General Agent 节点 - 负责通用回答
    
    处理问候、寒暄、一般性闲聊
    """
    llm = get_llm()
    
    # 获取用户最新消息
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)
    
    if not last_user_message:
        return {
            "final_answer": "你好！有什么可以帮助你的吗？"
        }
    
    # 构建提示词
    prompt = f"""{GENERAL_AGENT_SYSTEM_PROMPT}

用户说：{last_user_message}

请给出友好、简洁的回答。
"""
    
    # 调用 LLM 生成答案
    response = llm.invoke(prompt)
    
    print(f"[General Agent] 生成答案长度: {len(response.content)} 字符")
    
    return {
        "final_answer": response.content,
        "used_agent": "general_agent"
    }


# ==================== 辅助函数 ====================

def _get_last_user_message(messages: List) -> Optional[str]:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _get_chat_history(messages: List) -> List:
    """获取对话历史（不含最新用户消息）"""
    history = []
    found_user_msg = False
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if not found_user_msg:
                found_user_msg = True
                continue
        history.append(msg)
    
    return history


# ==================== 路由执行节点 ====================

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
