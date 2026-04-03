"""
分布式追踪（Trace Context）
为每个请求生成 trace_id，追踪 CRAG、QueryExpansion、生成等所有子链路的延迟和错误。
"""
import uuid
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class Span:
    """单个追踪 span"""
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    start_ms: float = field(default_factory=lambda: time.time() * 1000)
    end_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    error: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    def end(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None):
        self.end_ms = time.time() * 1000
        self.duration_ms = round(self.end_ms - self.start_ms, 2)
        self.status = status
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error": self.error,
            **self.attrs,
        }


# Thread-safe context variables using ContextVar
_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_spans_var: ContextVar[List[Span]] = ContextVar("spans", default=[])
_span_stack_var: ContextVar[List[Span]] = ContextVar("span_stack", default=[])


# ==================== 公开 API ====================

def start_span(
    name: str,
    attrs: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Span:
    """
    开始一个新的追踪 span。

    嵌套语义：
    - 首次调用时生成 trace_id
    - 后续调用继承当前 trace_id，parent_id 指向前一个活跃 span

    用法示例：
        span = start_span("crag.grade_retrieval", {"doc_count": 10})
        try:
            result = await grader.grade_retrieval(...)
            end_span(span, SpanStatus.OK)
            return result
        except Exception as e:
            end_span(span, SpanStatus.ERROR, str(e))
            raise
    """
    # 获取或生成 trace_id
    current_trace_id = _trace_id_var.get()
    if current_trace_id is None:
        current_trace_id = uuid.uuid4().hex[:12]
        _trace_id_var.set(current_trace_id)

    # 获取当前活跃 span 作为父 span（栈顶）
    stack = _span_stack_var.get()
    parent = stack[-1] if stack else None

    span = Span(
        name=name,
        trace_id=current_trace_id,
        parent_id=parent.span_id if parent else None,
        attrs=attrs or {},
    )

    # 注册到 spans 列表
    spans = list(_spans_var.get())
    spans.append(span)
    _spans_var.set(spans)

    # 推入栈顶（设为当前活跃 span）
    stack = list(stack)
    stack.append(span)
    _span_stack_var.set(stack)

    logger.debug(
        f"[Trace] start span: name={name}, span_id={span.span_id}, "
        f"parent_id={span.parent_id}, trace_id={span.trace_id}"
    )
    return span


def end_span(span: Span, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None):
    """结束一个 span"""
    span.end(status, error)

    # 从栈顶弹出
    stack = list(_span_stack_var.get())
    if stack and stack[-1].span_id == span.span_id:
        stack.pop()
        _span_stack_var.set(stack)

    logger.debug(
        f"[Trace] end span: name={span.name}, "
        f"duration_ms={span.duration_ms}, status={span.status.value}"
    )


def get_trace_context() -> Dict[str, Any]:
    """
    获取当前请求的完整 trace 上下文。
    包含所有 span 的追踪数据，用于上报和调试。
    """
    spans = _spans_var.get()
    trace_id = _trace_id_var.get()

    total_duration = 0.0
    if spans:
        first = spans[0]
        last = spans[-1]
        if first.start_ms and last.end_ms:
            total_duration = round(last.end_ms - first.start_ms, 2)

    return {
        "trace_id": trace_id,
        "total_duration_ms": total_duration,
        "span_count": len(spans),
        "spans": [s.to_dict() for s in spans],
    }


def clear_trace_context():
    """清除当前请求的 trace 上下文（在请求结束时调用）"""
    _trace_id_var.set(None)
    _spans_var.set([])
    _span_stack_var.set([])


# ==================== 便捷装饰器 ====================

import functools
import asyncio


def traced(
    name: Optional[str] = None,
    attrs_func: Optional[callable] = None,
):
    """
    异步函数的追踪装饰器。

    用法：
        @traced("crag.grade_retrieval")
        async def grade_retrieval(self, query, docs):
            ...

        @traced(attrs_func=lambda args, kwargs: {"doc_count": len(kwargs["documents"])})
        async def some_method(self, documents, ...):
            ...
    """
    def decorator(func):
        _name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attrs = {}
            if attrs_func:
                try:
                    attrs = attrs_func(args, kwargs)
                except Exception:
                    pass

            span = start_span(_name, attrs)
            try:
                result = await func(*args, **kwargs)
                end_span(span, SpanStatus.OK)
                return result
            except asyncio.TimeoutError:
                end_span(span, SpanStatus.TIMEOUT, "function timed out")
                raise
            except Exception as e:
                end_span(span, SpanStatus.ERROR, str(e))
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            attrs = {}
            if attrs_func:
                try:
                    attrs = attrs_func(args, kwargs)
                except Exception:
                    pass

            span = start_span(_name, attrs)
            try:
                result = func(*args, **kwargs)
                end_span(span, SpanStatus.OK)
                return result
            except Exception as e:
                end_span(span, SpanStatus.ERROR, str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return wrapper
        return sync_wrapper

    return decorator
