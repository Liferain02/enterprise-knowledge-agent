"""
Operation Agent 节点
负责执行操作任务（计算、时间、MCP工具等）
使用 langgraph-prebuilt 的 create_tool_calling_agent
"""
from typing import Dict, Any
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from core.llm import get_llm
from tools import get_all_agent_tools
from agents.prompts import OPERATION_AGENT_SYSTEM_PROMPT
from agents.nodes.utils import get_last_user_message

# Agent 缓存
_agent_cache = {}


def _get_operation_agent(tools):
    """获取 Operation Agent（带缓存）"""
    cache_key = f"op_{len(tools)}"
    if cache_key not in _agent_cache:
        llm = get_llm()
        checkpointer = MemorySaver()
        _agent_cache[cache_key] = create_react_agent(
            model=llm,
            tools=tools,
            prompt=OPERATION_AGENT_SYSTEM_PROMPT,
            checkpointer=checkpointer
        )
    return _agent_cache[cache_key]


def operation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Operation Agent 节点 - 负责执行操作任务
    
    使用 langgraph-prebuilt 的 create_tool_calling_agent
    自动处理工具调用循环
    """
    # 获取所有可用工具
    tools = get_all_agent_tools()
    
    # 获取用户最新消息
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    
    if not last_user_message:
        return {
            "final_answer": "抱歉，我无法理解您的问题。"
        }
    
    try:
        # 获取预建的 Tool Calling Agent
        agent = _get_operation_agent(tools)
        
        # 构建配置
        config = {"configurable": {"thread_id": "operation_agent"}}
        
        # 执行 Agent
        result = agent.invoke(
            {"messages": [last_user_message]},
            config
        )
        
        # 获取最终回复
        final_answer = result["messages"][-1].content
        
        print(f"[Operation Agent] 生成答案长度: {len(final_answer)} 字符")
        
        return {
            "final_answer": final_answer,
            "used_agent": "operation_agent"
        }
        
    except Exception as e:
        print(f"[Operation Agent] 执行出错: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "final_answer": f"执行操作时出错: {str(e)}",
            "used_agent": "operation_agent"
        }
