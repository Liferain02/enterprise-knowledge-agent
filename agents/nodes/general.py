"""
General Agent 节点
负责通用回答（问候、寒暄等）
"""
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage
from core.llm import get_llm
from agents.prompts import GENERAL_AGENT_SYSTEM_PROMPT


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


def _get_last_user_message(messages: List) -> Optional[str]:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None

