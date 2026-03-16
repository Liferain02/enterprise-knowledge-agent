"""
General Agent 节点
负责通用回答（问候、寒暄等）
"""
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from src.models.llm import get_llm
from ..prompts import GENERAL_AGENT_SYSTEM_PROMPT
from ._utils import get_last_user_message, build_summary_context


async def general_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    General Agent 节点 - 负责通用回答

    处理问候、寒暄、一般性闲聊

    注意：不维护自己的历史，由主图统一管理
    """
    llm = get_llm()

    # 获取完整的消息历史
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    summary = state.get("summary", "") or ""

    # 获取 Mem0 记忆
    mem0_memories = state.get("mem0_memories", "") or ""
    
    # 调试输出
    print(f"[General Agent] mem0_memories 内容: {repr(mem0_memories[:200] if mem0_memories else '')}")

    # 摘要上下文（旧对话压缩后的摘要）
    summary_context = build_summary_context(summary)

    # 构建近期对话上下文（最近 6 条原始消息）
    history_context = ""
    if len(messages) > 1:
        history_msgs = []
        for msg in messages[:-1]:  # 排除当前消息
            if hasattr(msg, 'content'):
                role = "用户"
                if hasattr(msg, 'type'):
                    role = "用户" if msg.type == 'human' else "助手"
                history_msgs.append(f"{role}: {msg.content}")

        if history_msgs:
            history_context = f"\n\n## 近期对话记录\n" + "\n".join(history_msgs[-6:])

    if not last_user_message:
        # 没有用户消息，返回默认问候
        return {
            "final_answer": "你好！有什么可以帮助你的吗？",
            "used_agent": "general_agent",
            "messages": [AIMessage(content="你好！有什么可以帮助你的吗？")]
        }

    # 构建提示词（Mem0记忆 -> 摘要 -> 近期历史 -> 当前消息）
    prompt = f"""{GENERAL_AGENT_SYSTEM_PROMPT}
{mem0_memories}
{summary_context}{history_context}

## 当前用户消息
用户说：{last_user_message}

请给出友好、简洁的回答。
"""
    
    # 调用 LLM 生成答案
    response = await llm.ainvoke(prompt)
    
    print(f"[General Agent] 生成答案长度: {len(response.content)} 字符")
    
    # 创建 AIMessage 返回给主图
    ai_message = AIMessage(content=response.content)
    
    return {
        "final_answer": response.content,
        "used_agent": "general_agent",
        "messages": [ai_message]  # 添加到主图历史
    }
