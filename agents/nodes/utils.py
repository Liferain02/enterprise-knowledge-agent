"""
Agents 节点共享工具函数
提取各节点通用的辅助函数
"""
from typing import List, Optional
from langchain_core.messages import HumanMessage


def get_last_user_message(messages: List) -> Optional[str]:
    """
    获取最后一条用户消息
    
    Args:
        messages: 消息列表
        
    Returns:
        最后一条用户消息内容，如果没有则返回 None
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None

