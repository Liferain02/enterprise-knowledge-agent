"""
Agents 共享工具函数
"""
from typing import List, Optional
from langchain_core.messages import HumanMessage


def get_last_user_message(messages: List) -> Optional[str]:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None
