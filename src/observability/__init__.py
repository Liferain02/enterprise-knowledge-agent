"""
可观测性子模块
- tracer: 分布式追踪（trace_id / span）
- metrics: Prometheus metrics 收集器
- cost_tracker: LLM 调用成本估算
- structured_logging: 结构化 JSON 日志
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
from .structured_logging import (
    configure_logging, structured_logger, set_log_context, clear_log_context,
    get_log_context, LogContextFilter, log_chat_request, log_retrieval_event,
    log_llm_error,
)

__all__ = [
    # tracer
    "start_span", "end_span", "get_trace_context", "clear_trace_context",
    "SpanStatus", "Span", "traced",
    # metrics
    "get_metrics_collector", "get_metrics", "get_content_type", "MetricsCollector",
    # cost
    "get_cost_tracker", "CostTracker", "CostRecord",
    # structured logging
    "configure_logging", "structured_logger", "set_log_context", "clear_log_context",
    "get_log_context", "LogContextFilter",
    "log_chat_request", "log_retrieval_event", "log_llm_error",
]
