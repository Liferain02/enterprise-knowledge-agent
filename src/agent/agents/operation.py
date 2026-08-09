"""
Operation Agent 节点
负责执行操作任务（计算、时间、MCP工具等）
使用 langgraph-prebuilt 的 create_react_agent
"""
import os
from typing import Dict, Any
import asyncio
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from src.models.llm import get_llm
from ..tools import get_all_agent_tools
from ..prompts import OPERATION_AGENT_SYSTEM_PROMPT
from ._utils import get_last_user_message, inject_summary_to_messages, inject_user_identity_to_messages
from config.settings import get_settings

import logging

logger = logging.getLogger(__name__)


# Agent 缓存
_agent_cache = {}

# 操作超时设置（秒）- 3分钟超时
OPERATION_TIMEOUT = 180


def _get_operation_agent(tools):
    """获取 Operation Agent（带缓存）"""
    cache_key = f"op_{len(tools)}"
    if cache_key not in _agent_cache:
        llm = get_llm()

        _agent_cache[cache_key] = create_react_agent(
            model=llm,
            tools=tools,
            prompt=OPERATION_AGENT_SYSTEM_PROMPT,
        )
    return _agent_cache[cache_key]



async def operation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Operation Agent 节点 - 负责执行操作任务

    使用 langgraph-prebuilt 的 create_react_agent
    自动处理工具调用循环

    改为 async def，直接在当前事件循环中运行
    避免创建新事件循环导致 MCP 死锁

    注意：不使用 Agent 自己的 checkpointer，
    历史由主图统一管理，通过 messages 传递
    """
    # 获取所有可用工具
    tools = get_all_agent_tools()

    # 获取完整的消息历史（包含之前的对话）
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    summary = state.get("summary", "") or ""

    # 获取 Mem0 记忆
    mem0_memories = state.get("mem0_memories", "") or ""

    # 获取 session_id
    session_id = state.get("session_id", "default")

    if not last_user_message:
        return {
            "final_answer": "抱歉，我无法理解您的问题。"
        }

    try:
        # 获取预建的 Tool Calling Agent（注意：不传 checkpointer，避免重复管理状态）
        agent = _get_operation_agent(tools)

        # 传递消息历史，若存在摘要和 Mem0 记忆则在头部注入 SystemMessage
        # 会话历史只由主图维护，子 Agent 不再创建第二套持久化状态。
        config = {}
        messages_with_context = inject_user_identity_to_messages(
            messages,
            user_context=state.get("user_context"),
            summary=summary,
            mem0_memories=mem0_memories,
        )

        # 直接使用 await ainvoke
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": messages_with_context}, config),
            timeout=OPERATION_TIMEOUT
        )

        # 子 Agent 内部消息只用于本次工具循环，不写回主会话状态。
        agent_messages = result.get("messages", [])

        # 获取最终回复
        final_answer = agent_messages[-1].content

        logger.debug("操作完成: len=%d", len(final_answer))

        # 主图只接收最终用户可见回答，避免工具调用污染摘要和长期记忆。
        return {
            "final_answer": final_answer,
            "used_agent": "operation_agent",
            "messages": [AIMessage(content=final_answer)],
        }
        
    except asyncio.TimeoutError:
        return {
            "final_answer": f"⏱️ 操作超时（{OPERATION_TIMEOUT}秒）\n\n可能原因：\n1. 首次调用 MCP 工具需要加载配置（约 10-30 秒）\n2. 模型推理耗时较长\n3. 文件系统操作较慢\n\n建议：请重新尝试，通常第二次调用会更快。",
            "used_agent": "operation_agent"
        }
    except Exception as e:
        logger.exception("操作执行出错: %s", e)
        
        return {
            "final_answer": f"执行操作时出错: {str(e)}",
            "used_agent": "operation_agent"
        }
