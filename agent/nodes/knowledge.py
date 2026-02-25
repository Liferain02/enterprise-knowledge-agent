"""
Knowledge Agent 节点
负责知识检索和回答
"""
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage
from core.llm import get_llm
from tools import get_knowledge_tool


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


def _get_last_user_message(messages: List) -> Optional[str]:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None

