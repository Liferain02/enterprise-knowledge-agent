"""
Agent 模块
"""
from .graph import (
    run_agent,
    arun_agent,
    get_agent_graph,
    get_agent_graph_async,
    AgentState,
)

# Legacy / Deprecated 兼容导出；生产图不调用并行计划执行器。
try:
    from .agents.parallel_executor import (
        ParallelExecutor,
        get_parallel_executor,
        execute_steps_parallel,
        analyze_step_dependencies
    )
    _PARALLEL_AVAILABLE = True
except ImportError:
    _PARALLEL_AVAILABLE = False

__all__ = [
    "run_agent",
    "arun_agent",
    "get_agent_graph",
    "get_agent_graph_async",
    "AgentState",
]

if _PARALLEL_AVAILABLE:
    __all__.extend([
        "ParallelExecutor",
        "get_parallel_executor",
        "execute_steps_parallel",
        "analyze_step_dependencies"
    ])
