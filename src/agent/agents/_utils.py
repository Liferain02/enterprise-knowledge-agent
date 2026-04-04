"""
Agents 共享工具函数
"""
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage


def get_last_user_message(messages: List) -> Optional[str]:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def build_summary_context(summary: str) -> str:
    """
    将摘要格式化为可嵌入 Prompt 的字符串块。
    当 summary 为空时返回空字符串，调用方无需判断。
    """
    if not summary:
        return ""
    return f"\n\n## 历史对话摘要\n以下是本次对话早期内容的摘要，供你了解上下文：\n{summary}\n"


def inject_summary_to_messages(messages: List, summary: str) -> List:
    """
    若存在摘要，在消息列表头部插入一条 SystemMessage 作为上下文。
    用于 knowledge_agent / operation_agent 向子 Agent 传递 messages 时附带摘要。
    """
    if not summary:
        return messages
    summary_msg = SystemMessage(
        content=f"【历史对话摘要】以下是本次对话早期内容的摘要，请结合它理解用户的上下文：\n{summary}"
    )
    return [summary_msg] + list(messages)


def inject_mem0_to_messages(messages: List, mem0_memories: str) -> List:
    """
    若存在 Mem0 记忆，在消息列表头部插入一条 SystemMessage 作为上下文。
    用于注入用户偏好、历史交互等信息。
    """
    if not mem0_memories:
        return messages
    mem0_msg = SystemMessage(content=mem0_memories)
    return [mem0_msg] + list(messages)


def inject_context_to_messages(
    messages: List,
    summary: str = "",
    mem0_memories: str = ""
) -> List:
    """
    综合注入摘要和 Mem0 记忆到消息列表。
    优先注入 Mem0 记忆（用户画像），然后是摘要（对话历史）。
    """
    injected = list(messages)

    # 先注入 Mem0 记忆（用户偏好等）
    if mem0_memories:
        mem0_msg = SystemMessage(content=mem0_memories)
        injected = [mem0_msg] + injected

    # 再注入对话摘要
    if summary:
        summary_msg = SystemMessage(
            content=f"【历史对话摘要】以下是本次对话早期内容的摘要，请结合它理解用户的上下文：\n{summary}"
        )
        injected = [summary_msg] + injected

    return injected


def inject_worker_context(
    messages: List,
    summary: str = "",
    mem0_memories: str = ""
) -> List:
    """
    为 Worker Agent 预注入上下文。

    与 inject_context_to_messages 的区别：
    - 本方法将 Mem0 记忆和摘要合并为一条 SystemMessage（减少 token 开销）
    - 适用于 parallel_executor 中多个 Worker 共享同一份预注入上下文

    格式：「【背景上下文】Mem0记忆（若存在）\n---\n对话摘要（若存在）」
    """
    parts = []

    if mem0_memories:
        parts.append(f"【用户背景与历史记忆】\n{mem0_memories.strip()}")

    if summary:
        parts.append(f"【本次对话早期摘要】\n{summary.strip()}")

    if not parts:
        return list(messages)

    context_content = "\n\n---\n\n".join(parts)
    context_msg = SystemMessage(
        content=f"【背景上下文】以下是你需要了解的背景信息，请结合它回答用户问题：\n\n{context_content}"
    )
    return [context_msg] + list(messages)
