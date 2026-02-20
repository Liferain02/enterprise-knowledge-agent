"""
LangGraph 图定义模块
定义 Agent 的工作流图
"""
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.nodes import (
    planning_node,
    memory_node,
    tool_selection_node,
    tool_execution_node,
    generation_node,
    check_completion_node,
    error_handling_node
)


# LangGraph 0.2+ 使用 TypedDict 定义状态
class GraphState(TypedDict):
    """图状态定义"""
    input: str
    session_id: str
    use_rag: bool
    iteration: int
    messages: list
    tool_results: dict
    plan: str | None
    selected_tools: list
    context: str
    final_answer: str | None
    is_done: bool
    error: str | None


def create_agent_graph():
    """
    创建 Agent 工作流图
    
    工作流程：
    1. 规划 (Planning) - 分析问题并制定计划
    2. 记忆 (Memory) - 检索相关上下文
    3. 工具选择 (Tool Selection) - 决定使用哪些工具
    4. 工具执行 (Tool Execution) - 执行选定的工具
    5. 生成 (Generation) - 生成最终答案
    """
    
    # 创建图 - 使用 GraphState
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("planning", planning_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("tool_selection", tool_selection_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("check_completion", check_completion_node)
    workflow.add_node("error_handling", error_handling_node)
    
    # 设置入口点
    workflow.set_entry_point("memory")
    
    # 添加边
    workflow.add_edge("memory", "generation")
    workflow.add_edge("generation", "check_completion")
    
    # 条件边 - 检查完成状态
    workflow.add_conditional_edges(
        "check_completion",
        lambda x: "end" if x.get("is_done", False) else "planning",
        {
            "end": END,
            "planning": "planning"
        }
    )
    
    workflow.add_edge("planning", "tool_selection")
    workflow.add_edge("tool_selection", "tool_execution")
    workflow.add_edge("tool_execution", "generation")
    
    # 编译图
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    return compiled_graph
def create_simple_agent_graph() -> StateGraph:
    """
    创建简化的 Agent 工作流图
    
    简化流程：
    1. 记忆 (Memory) - 检索相关上下文
    2. 生成 (Generation) - 生成最终答案
    """
    
    # 创建图 - 使用 GraphState
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("memory", memory_node)
    workflow.add_node("generation", generation_node)
    
    # 设置入口点
    workflow.set_entry_point("memory")
    
    # 添加边
    workflow.add_edge("memory", "generation")
    workflow.add_edge("generation", END)
    
    # 编译图
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    return compiled_graph
# 全局图实例
_agent_graph = None
_simple_agent_graph = None
def get_agent_graph() -> StateGraph:
    """获取 Agent 图实例"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph
def get_simple_agent_graph() -> StateGraph:
    """获取简化版 Agent 图实例"""
    global _simple_agent_graph
    if _simple_agent_graph is None:
        _simple_agent_graph = create_simple_agent_graph()
    return _simple_agent_graph
def run_agent(
    input_text: str,
    session_id: str = "default",
    use_rag: bool = True,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    运行 Agent
    
    Args:
        input_text: 用户输入
        session_id: 会话ID
        use_rag: 是否使用RAG
        config: 配置
    
    Returns:
        Agent 执行结果
    """
    graph = get_agent_graph() if use_rag else get_simple_agent_graph()
    
    initial_state = {
        "input": input_text,
        "session_id": session_id,
        "use_rag": use_rag,
        "iteration": 0,
        "messages": [],
        "tool_results": {}
    }
    
    # LangGraph 需要 config 中包含 thread_id
    config = config or {}
    if "configurable" not in config:
        config["configurable"] = {}
    config["configurable"]["thread_id"] = session_id
    
    result = graph.invoke(initial_state, config)
    
    return result

