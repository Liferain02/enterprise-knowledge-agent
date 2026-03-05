"""
Agent 模块 - Agent 工厂和图构建
"""
from functools import lru_cache
from typing import Optional

from langgraph.graph import StateGraph, END

from .agents.supervisor import supervisor_node
from .agents.knowledge import knowledge_agent_node
from .agents.operation import operation_agent_node
from .agents.general import general_agent_node
from .agents import AgentState


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
    # 使用 RunnableLambda 包装以支持异步
    from langchain_core.runnables import RunnableLambda
    from .graph import route_to_agent

    workflow.add_conditional_edges(
        "supervisor",
        RunnableLambda(route_to_agent),
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

    return workflow.compile()


@lru_cache(maxsize=1)
def get_agent_graph():
    """获取编译后的 Agent 图（带缓存）"""
    return create_multi_agent_graph()
