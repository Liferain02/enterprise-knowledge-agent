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
