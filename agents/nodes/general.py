"""
General Agent 节点
负责通用回答（问候、寒暄等）
"""
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from core.llm import get_llm
from agents.prompts import GENERAL_AGENT_SYSTEM_PROMPT
from agents.nodes.utils import get_last_user_message


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
    
    # 构建系统提示词，包含历史上下文
    history_context = ""
    if len(messages) > 1:
        # 提取历史对话用于上下文
        history_msgs = []
        for msg in messages[:-1]:  # 排除当前消息
            if hasattr(msg, 'content'):
                role = "用户" if isinstance(msg, type(messages[0])) else "助手"
                if hasattr(msg, 'type'):
                    role = "用户" if msg.type == 'human' else "助手"
                history_msgs.append(f"{role}: {msg.content}")
        
        if history_msgs:
            history_context = f"\n\n## 对话历史\n" + "\n".join(history_msgs[-6:])  # 最近6条
    
    if not last_user_message:
        # 没有用户消息，返回默认问候
        return {
            "final_answer": "你好！有什么可以帮助你的吗？",
            "used_agent": "general_agent",
            "messages": [AIMessage(content="你好！有什么可以帮助你的吗？")]
        }
    
    # 构建提示词
    prompt = f"""{GENERAL_AGENT_SYSTEM_PROMPT}
{history_context}

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
