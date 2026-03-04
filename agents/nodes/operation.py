"""
Operation Agent 节点
负责执行操作任务（计算、时间、MCP工具等）
使用 langgraph-prebuilt 的 create_react_agent
"""
from typing import Dict, Any
import asyncio
import uuid
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from core.llm import get_llm
from tools import get_all_agent_tools
from agents.prompts import OPERATION_AGENT_SYSTEM_PROMPT
from agents.nodes.utils import get_last_user_message


# Agent 缓存
_agent_cache = {}

# 操作超时设置（秒）
OPERATION_TIMEOUT = 600


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


async def operation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Operation Agent 节点 - 负责执行操作任务
    
    使用 langgraph-prebuilt 的 create_react_agent
    自动处理工具调用循环
    
    改为 async def，直接在当前事件循环中运行
    避免创建新事件循环导致 MCP 死锁
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
        
        # 使用唯一的 thread_id 避免消息历史冲突
        thread_id = f"op_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 直接使用 await ainvoke，去掉线程池包裹
        # 这样可以在 FastAPI 的主事件循环中运行
        # 避免跨事件循环导致的 MCP 死锁
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [last_user_message]}, config),
            timeout=OPERATION_TIMEOUT
        )
        
        # 获取最终回复
        final_answer = result["messages"][-1].content
        
        print(f"[Operation Agent] 生成答案长度: {len(final_answer)} 字符")
        
        return {
            "final_answer": final_answer,
            "used_agent": "operation_agent"
        }
        
    except asyncio.TimeoutError:
        return {
            "final_answer": f"⏱️ 操作超时（{OPERATION_TIMEOUT}秒）\n\n可能原因：\n1. 首次调用 MCP 工具需要加载配置（约 10-30 秒）\n2. 模型推理耗时较长\n3. 文件系统操作较慢\n\n建议：请重新尝试，通常第二次调用会更快。",
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
