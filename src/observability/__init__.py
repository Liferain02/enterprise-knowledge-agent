"""
可观测性子模块
- tracer: 分布式追踪（trace_id / span）
- metrics: Prometheus metrics 收集器
- cost_tracker: LLM 调用成本估算
"""
from .tracer import (
    start_span, end_span, get_trace_context, clear_trace_context,
    SpanStatus, Span, traced,
)
from .metrics import (
    get_metrics_collector, get_metrics, get_content_type,
    MetricsCollector,
)
from .cost_tracker import get_cost_tracker, CostTracker, CostRecord

__all__ = [
    # tracer
    "start_span", "end_span", "get_trace_context", "clear_trace_context",
    "SpanStatus", "Span", "traced",
    # metrics
    "get_metrics_collector", "get_metrics", "get_content_type", "MetricsCollector",
    # cost
    "get_cost_tracker", "CostTracker", "CostRecord",
]
