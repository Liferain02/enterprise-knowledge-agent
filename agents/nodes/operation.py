"""
Operation Agent 节点
负责执行操作任务（计算、时间、MCP工具等）
"""
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from core.llm import get_llm
from tools import get_all_agent_tools, get_tool_by_name
from agents.prompts import OPERATION_AGENT_SYSTEM_PROMPT


def operation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Operation Agent 节点 - 负责执行操作任务
    
    使用原生 Tool Calling：llm.bind_tools() + 手动执行工具
    """
    llm = get_llm()
    
    # 获取所有可用工具
    tools = get_all_agent_tools()
    
    # 获取用户最新消息
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)
    
    if not last_user_message:
        return {
            "final_answer": "抱歉，我无法理解您的问题。"
        }
    
    # 构建消息列表（包含历史）
    chat_messages = [
        SystemMessage(content=OPERATION_AGENT_SYSTEM_PROMPT),
    ]
    
    # 添加对话历史（最近 10 条）
    for msg in messages[-10:]:
        chat_messages.append(msg)
    
    # 绑定工具到 LLM（使用 bind_tools）
    llm_with_tools = llm.bind_tools(tools)
    
    # 第一次调用：让 LLM 决定是否调用工具
    response = llm_with_tools.invoke(chat_messages)
    
    # 检查是否有 tool_calls
    final_answer = ""
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # 有工具调用，执行工具并获取结果
        tool_results = []
        
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('args', {})
            
            # 获取工具
            tool = get_tool_by_name(tool_name)
            
            if tool:
                try:
                    # 执行工具
                    result = tool.invoke(tool_args)
                    tool_results.append(f"【{tool_name}】执行结果: {result}")
                    print(f"[Operation Agent] 工具 {tool_name} 执行成功")
                except Exception as e:
                    tool_results.append(f"【{tool_name}】执行出错: {str(e)}")
                    print(f"[Operation Agent] 工具 {tool_name} 执行失败: {e}")
            else:
                tool_results.append(f"【{tool_name}】工具不存在")
        
        # 将工具结果拼接到消息中，再次调用 LLM 生成最终答案
        tool_result_message = AIMessage(content=json.dumps(response.tool_calls))
        tool_results_content = "\n".join(tool_results)
        
        chat_messages.append(tool_result_message)
        chat_messages.append(HumanMessage(content=f"工具执行结果:\n{tool_results_content}\n\n请基于以上工具执行结果，给出最终答案。"))
        
        # 第二次调用：基于工具结果生成最终答案
        final_response = llm.invoke(chat_messages)
        final_answer = final_response.content
        
    else:
        # 没有工具调用，直接使用 LLM 的回复
        final_answer = response.content
    
    print(f"[Operation Agent] 生成答案长度: {len(final_answer)} 字符")
    
    return {
        "final_answer": final_answer,
        "used_agent": "operation_agent"
    }


def _get_last_user_message(messages: List) -> Optional[str]:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None

