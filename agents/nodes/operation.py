"""
Operation Agent 节点
负责执行操作任务（计算、时间、MCP工具等）
使用 langgraph-prebuilt 的 create_tool_calling_agent
"""
from typing import Dict, Any
import asyncio
import concurrent.futures
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
# 10 分钟：用于处理较慢的工具调用/多轮推理
OPERATION_TIMEOUT = 600

# 线程池执行器
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


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


def _run_agent_sync(agent, messages, config):
    """在线程中运行 agent（同步包装）"""
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            asyncio.wait_for(
                agent.ainvoke({"messages": messages}, config),
                timeout=OPERATION_TIMEOUT
            )
        )
    finally:
        loop.close()


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
        
        # 使用唯一的 thread_id 避免消息历史冲突
        thread_id = f"op_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 使用线程池执行异步 agent（在新事件循环中）
        future = _executor.submit(_run_agent_sync, agent, [last_user_message], config)
        # 给线程池的等待时间稍微加一点缓冲
        result = future.result(timeout=OPERATION_TIMEOUT + 10)
        
        # 获取最终回复
        final_answer = result["messages"][-1].content
        
        print(f"[Operation Agent] 生成答案长度: {len(final_answer)} 字符")
        
        return {
            "final_answer": final_answer,
            "used_agent": "operation_agent"
        }
        
    except concurrent.futures.TimeoutError:
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
