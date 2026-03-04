"""
LangGraph Multi-Agent 工作流图
重构后的完整 Agent 架构
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage

from agents.nodes.supervisor import supervisor_node, route_to_agent
from agents.nodes.knowledge import knowledge_agent_node
from agents.nodes.operation import operation_agent_node
from agents.nodes.general import general_agent_node


# ==================== Checkpointer 实例 ====================

def get_checkpointer() -> MemorySaver:
    """
    获取 Memory Saver
    
    注意：LangGraph 的状态持久化使用内存
    会话元数据由 session_store.py 管理
    """
    return MemorySaver()


# ==================== 状态定义 ====================

class AgentState(MessagesState):
    """
    Agent 状态定义
    继承 MessagesState 以支持自动消息管理
    """
    # 路由决策
    next_agent: str
    supervisor_reasoning: str
    supervisor_reason: str
    
    # 执行结果
    final_answer: str
    sources: str
    used_agent: str
    
    # 会话ID（用于在各节点中维护历史）
    session_id: str


# ==================== 图创建函数 ====================

def create_multi_agent_graph() -> StateGraph:
    """
    创建 Multi-Agent 工作流图
    
    工作流程：
    1. 接收用户消息
    2. Supervisor 分析意图并路由
    3. 根据路由选择对应的 Worker Agent
    4. Worker Agent 生成答案
    5. 返回最终答案
    """
    
    # 创建图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("knowledge_agent", knowledge_agent_node)
    workflow.add_node("operation_agent", operation_agent_node)
    workflow.add_node("general_agent", general_agent_node)
    
    # 设置入口点
    workflow.set_entry_point("supervisor")
    
    # 添加条件边 - 根据 Supervisor 决策路由
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "knowledge_agent": "knowledge_agent",
            "operation_agent": "operation_agent",
            "general_agent": "general_agent"
        }
    )
    
    # 所有 Agent 节点都指向 END
    workflow.add_edge("knowledge_agent", END)
    workflow.add_edge("operation_agent", END)
    workflow.add_edge("general_agent", END)
    
    return workflow


# ==================== 编译图 ====================

def compile_graph(checkpointer: MemorySaver = None) -> StateGraph:
    """
    编译并返回可执行的图
    
    Args:
        checkpointer: 状态持久化检查点，默认使用 SQLite 持久化
        
    Returns:
        编译后的 LangGraph
    """
    if checkpointer is None:
        checkpointer = get_checkpointer()
    
    workflow = create_multi_agent_graph()
    compiled = workflow.compile(checkpointer=checkpointer)
    
    return compiled


# ==================== 全局图实例 ====================

_agent_graph = None


def get_agent_graph() -> StateGraph:
    """
    获取 Agent 图实例（单例模式）
    
    Returns:
        编译后的 LangGraph
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = compile_graph()
    return _agent_graph


# ==================== 执行入口函数 ====================

def run_agent(
    input_text: str,
    session_id: str = "default",
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    运行 Agent 的入口函数（同步版本）
    
    Args:
        input_text: 用户输入
        session_id: 会话ID（用于状态持久化）
        config: 额外配置
        
    Returns:
        包含 final_answer 的字典
    """
    # 获取图
    graph = get_agent_graph()
    
    # 构建配置（包含 thread_id 用于状态持久化）
    run_config = config or {}
    if "configurable" not in run_config:
        run_config["configurable"] = {}
    run_config["configurable"]["thread_id"] = session_id
    
    # 构建初始状态（包含 session_id 供各节点使用）
    initial_state = {
        "messages": [HumanMessage(content=input_text)],
        "session_id": session_id  # 传递 session_id 到状态中
    }
    
    # 执行图
    try:
        result = graph.invoke(initial_state, run_config)
        
        # 提取最终答案
        final_answer = result.get("final_answer", "抱歉，无法生成答案。")
        sources = result.get("sources", "")
        used_agent = result.get("used_agent", "unknown")
        
        return {
            "final_answer": final_answer,
            "sources": sources,
            "used_agent": used_agent,
            "messages": result.get("messages", [])
        }
        
    except Exception as e:
        print(f"Agent 执行出错: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "final_answer": f"处理请求时出错: {str(e)}",
            "sources": "",
            "used_agent": "error",
            "messages": []
        }


async def arun_agent(
    input_text: str,
    session_id: str = "default",
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    运行 Agent 的入口函数（异步版本）
    
    使用 ainvoke 在主事件循环中运行
    避免跨事件循环导致的 MCP 死锁
    
    Args:
        input_text: 用户输入
        session_id: 会话ID（用于状态持久化）
        config: 额外配置
        
    Returns:
        包含 final_answer 的字典
    """
    # 获取图
    graph = get_agent_graph()
    
    # 构建配置（包含 thread_id 用于状态持久化）
    run_config = config or {}
    if "configurable" not in run_config:
        run_config["configurable"] = {}
    run_config["configurable"]["thread_id"] = session_id
    
    # 构建初始状态（包含 session_id 供各节点使用）
    initial_state = {
        "messages": [HumanMessage(content=input_text)],
        "session_id": session_id  # 传递 session_id 到状态中
    }
    
    # 执行图（异步版本）
    try:
        result = await graph.ainvoke(initial_state, run_config)
        
        # 提取最终答案
        final_answer = result.get("final_answer", "抱歉，无法生成答案。")
        sources = result.get("sources", "")
        used_agent = result.get("used_agent", "unknown")
        
        return {
            "final_answer": final_answer,
            "sources": sources,
            "used_agent": used_agent,
            "messages": result.get("messages", [])
        }
        
    except Exception as e:
        print(f"Agent 执行出错: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "final_answer": f"处理请求时出错: {str(e)}",
            "sources": "",
            "used_agent": "error",
            "messages": []
        }


# ==================== 流式执行 ====================

def run_agent_stream(
    input_text: str,
    session_id: str = "default",
    config: Dict[str, Any] = None
):
    """
    流式运行 Agent
    
    Args:
        input_text: 用户输入
        session_id: 会话ID
        config: 额外配置
        
    Yields:
        流式输出
    """
    # 获取图
    graph = get_agent_graph()
    
    # 构建配置
    run_config = config or {}
    if "configurable" not in run_config:
        run_config["configurable"] = {}
    run_config["configurable"]["thread_id"] = session_id
    
    # 构建初始状态
    initial_state = {
        "messages": [HumanMessage(content=input_text)]
    }
    
    # 流式执行
    for chunk in graph.stream(initial_state, run_config):
        yield chunk
