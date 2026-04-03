"""
单元测试 - 可观测性（tracer + metrics + cost_tracker）
"""
import asyncio
import pytest
import time
from src.observability.tracer import (
    start_span, end_span, get_trace_context,
    clear_trace_context, SpanStatus, traced,
)
from src.observability.metrics import (
    get_metrics_collector, get_metrics,
)
from src.observability.cost_tracker import (
    CostTracker, MODEL_PRICING,
)


class TestTracer:
    """分布式追踪单元测试"""

    def test_span_lifecycle(self):
        """span 生命周期：创建 → 结束"""
        clear_trace_context()
        span = start_span("test.operation", {"key": "value"})

        assert span.name == "test.operation"
        assert span.status == SpanStatus.OK
        assert span.duration_ms is None  # 未结束

        end_span(span, SpanStatus.OK)
        assert span.status == SpanStatus.OK
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_span_error(self):
        """span 错误状态"""
        clear_trace_context()
        span = start_span("test.error")
        end_span(span, SpanStatus.ERROR, "something went wrong")

        assert span.status == SpanStatus.ERROR
        assert span.error == "something went wrong"

    def test_nested_spans(self):
        """嵌套 span"""
        clear_trace_context()
        parent = start_span("parent")
        child = start_span("child")

        assert child.parent_id == parent.span_id
        assert child.trace_id == parent.trace_id

        end_span(child, SpanStatus.OK)
        end_span(parent, SpanStatus.OK)

    def test_trace_context(self):
        """trace 上下文汇总"""
        clear_trace_context()
        span1 = start_span("op1")
        end_span(span1, SpanStatus.OK)
        span2 = start_span("op2")
        end_span(span2, SpanStatus.OK)

        ctx = get_trace_context()
        assert ctx["trace_id"] is not None
        assert ctx["span_count"] == 2
        assert len(ctx["spans"]) == 2

    def test_traced_decorator_sync(self):
        """同步函数追踪装饰器"""
        clear_trace_context()

        @traced("my.operation")
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

        ctx = get_trace_context()
        assert any(s["name"] == "my.operation" for s in ctx["spans"])

    @pytest.mark.asyncio
    async def test_traced_decorator_async(self):
        """异步函数追踪装饰器"""
        clear_trace_context()

        @traced("async.operation")
        async def my_async():
            await asyncio.sleep(0.01)
            return 42

        result = await my_async()
        assert result == 42

        # 在同一事件循环内获取 trace context
        captured_ctx = None
        @traced("capture")
        async def capture():
            nonlocal captured_ctx
            captured_ctx = get_trace_context()

        await capture()
        assert any(s["name"] == "async.operation" for s in captured_ctx["spans"])

    def test_timeout_status(self):
        """超时状态"""
        clear_trace_context()
        span = start_span("slow.op")
        end_span(span, SpanStatus.TIMEOUT, "operation timed out")

        assert span.status == SpanStatus.TIMEOUT
        assert "timed out" in span.error

    def test_rate_limited_status(self):
        """限流状态"""
        clear_trace_context()
        span = start_span("api.call")
        end_span(span, SpanStatus.RATE_LIMITED, "429 Too Many Requests")

        assert span.status == SpanStatus.RATE_LIMITED


class TestMetrics:
    """Prometheus Metrics 单元测试"""

    def test_record_crag_decision(self):
        """记录 CRAG 决策"""
        mc = get_metrics_collector()
        # 不应抛出异常
        mc.record_crag_decision("high")
        mc.record_crag_decision("low")
        mc.record_crag_decision("no_results")

    def test_record_llm_error(self):
        """记录 LLM 错误"""
        mc = get_metrics_collector()
        mc.record_llm_error("rate_limited", "qwen3.5-flash")
        mc.record_llm_error("server_error", "gpt-4o")
        mc.record_llm_error("timeout", "qwen3.5-flash")
        mc.record_llm_error("parse_error", "qwen3.5-flash")

    def test_record_conflict(self):
        """记录冲突检测"""
        mc = get_metrics_collector()
        mc.record_conflict("high", "reject")
        mc.record_conflict("medium", "conflict_summary")
        mc.record_conflict("low", "none")

    def test_get_metrics(self):
        """生成 metrics 文本"""
        content = get_metrics()
        assert isinstance(content, bytes)
        assert b"ekb_crag_decisions_total" in content or b"ekb_chat" in content


class TestCostTracker:
    """成本追踪单元测试"""

    def test_estimate_tokens_chinese(self):
        """中文 token 估算"""
        tracker = CostTracker(auto_record=False)
        text = "公司年假政策是员工最关心的话题之一"
        tokens = tracker.estimate_tokens(text, "qwen3.5-flash")
        assert tokens > 0
        assert tokens < len(text)  # token 数应少于字符数（中文）

    def test_estimate_tokens_english(self):
        """英文 token 估算"""
        tracker = CostTracker(auto_record=False)
        text = "The annual leave policy is important for all employees"
        tokens = tracker.estimate_tokens(text, "qwen3.5-flash")
        assert tokens > 0
        assert tokens < len(text)

    def test_record_cost(self):
        """记录成本"""
        tracker = CostTracker(auto_record=False)
        record = tracker.record(
            model="qwen3.5-flash",
            input_text="公司年假政策",
            output_text="公司年假为15天",
            latency_ms=150.0,
            endpoint="chat/completions",
        )
        assert record.model == "qwen3.5-flash"
        assert record.input_tokens > 0
        assert record.output_tokens > 0
        assert record.total_tokens > 0
        assert record.estimated_cost_usd > 0
        assert record.estimated_cost_usd < 0.001  # 小额估算

    def test_model_pricing(self):
        """模型定价表完整性"""
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing
            assert "output" in pricing
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0

    def test_session_total_cost(self):
        """会话累计成本（不依赖 metrics）"""
        tracker = CostTracker(auto_record=False)
        r1 = tracker.record(
            model="qwen3.5-flash",
            input_text="公司年假政策具体规定和申请流程是什么？",
            output_text="根据公司制度，年假天数为15天，具体申请流程如下...",
            latency_ms=100.0,
        )
        r2 = tracker.record(
            model="qwen3.5-flash",
            input_text="员工病假工资如何计算？",
            output_text="员工病假期间，第一个月按正常工资的100%发放。",
            latency_ms=100.0,
        )
        # 单条记录的成本应 > 0
        assert r1.estimated_cost_usd > 0, f"cost 应 > 0，实际 {r1.estimated_cost_usd}"
        assert r2.estimated_cost_usd > 0
        # 记录数
        assert len(tracker._records) == 2
        assert r1.total_tokens > 0
        assert r2.total_tokens > 0
        # raw 累计成本应 > 0（session_total_cost_usd 会四舍五入到 6 位，tiny values 可能为 0）
        assert tracker._session_total_cost > 0, f"raw cost 应 > 0，实际 {tracker._session_total_cost}"
        assert tracker._session_total_tokens > r1.total_tokens
