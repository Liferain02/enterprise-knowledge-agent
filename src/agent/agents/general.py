"""
General Agent 节点
负责通用回答（问候、寒暄等）
使用 SkillLoader 创建 ReAct Agent，支持工具调用
"""
import asyncio
from typing import Dict, Any
from langchain_core.messages import AIMessage
from src.agent.skills.skill_loader import get_skill_loader
from ._utils import get_last_user_message, inject_context_to_messages
from src.observability import traced


# Agent 缓存
_agent_cache: Dict[str, Any] = {}

# 操作超时设置（秒）
GENERAL_TIMEOUT = 120


def _get_general_agent():
    """获取 General Agent（带缓存）"""
    cache_key = "general"
    if cache_key not in _agent_cache:
        loader = get_skill_loader()
        _agent_cache[cache_key] = loader.create_agent("general")
    return _agent_cache[cache_key]


@traced("agent.general.node")
async def general_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    General Agent 节点 - 负责通用回答（问候、寒暄等）

    使用 SkillLoader 创建 ReAct Agent，支持工具调用：
    - general_search：搜索通用知识
    - search_conversation_history：搜索对话历史

    不维护自己的历史，由主图统一管理。
    """
    # 获取 General ReAct Agent
    agent = _get_general_agent()

    # 获取完整的消息历史
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    summary = state.get("summary", "") or ""
    mem0_memories = state.get("mem0_memories", "") or ""
    session_id = state.get("session_id", "default")

    if not last_user_message:
        return {
            "final_answer": "你好！有什么可以帮助你的吗？",
            "used_agent": "general_agent",
            "messages": [AIMessage(content="你好！有什么可以帮助你的吗？")]
        }

    try:
        # 传递消息历史，若存在摘要和 Mem0 记忆则在头部注入 SystemMessage
        config = {"configurable": {"thread_id": f"general_{session_id}"}}
        messages_with_context = inject_context_to_messages(messages, summary, mem0_memories)

        # 使用 await 直接调用（ReAct Agent 内部处理工具调用循环）
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": messages_with_context}, config),
            timeout=GENERAL_TIMEOUT
        )

        # 获取 Agent 返回的所有消息（包含工具调用和最终回复）
        agent_messages = result.get("messages", [])
        final_answer = agent_messages[-1].content

        print(f"[General Agent] 生成答案长度: {len(final_answer)} 字符")

        return {
            "final_answer": final_answer,
            "used_agent": "general_agent",
            "messages": agent_messages
        }

    except asyncio.TimeoutError:
        return {
            "final_answer": f"⏱️ General Agent 执行超时（{GENERAL_TIMEOUT}秒）\n\n请重新尝试。",
            "used_agent": "general_agent",
            "messages": [AIMessage(content=f"抱歉，回答超时了，请重新尝试。")]
        }
    except Exception as e:
        print(f"[General Agent] 执行出错: {e}")
        import traceback
        traceback.print_exc()

        return {
            "final_answer": f"处理请求时出错: {str(e)}",
            "used_agent": "general_agent",
            "messages": [AIMessage(content=f"抱歉，处理您的请求时遇到了问题。")]
        }
