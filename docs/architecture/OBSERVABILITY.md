# 可观测性：成本、延迟、稳定性设计

> 定位：企业内部制度问答与流程检索系统 → 可评估、可观测

## 1. 目标指标体系

### 1.1 延迟指标（P50 / P95 / P99）

| 路径 | P50 目标 | P95 目标 | P99 目标 | 告警阈值 |
|------|---------|---------|---------|---------|
| FastAPI `/chat` 端到端 | < 5s | < 15s | < 30s | > 30s |
| CRAG 评估（单次）| < 3s | < 8s | < 15s | > 15s |
| Query Expansion | < 2s | < 5s | < 10s | > 10s |
| Rerank 调用 | < 500ms | < 2s | < 5s | > 5s |
| 向量检索 | < 200ms | < 500ms | < 1s | > 1s |
| Mem0 检索 | < 100ms | < 300ms | < 500ms | > 500ms |

### 1.2 成本指标（每次对话）

| 组件 | token 消耗估算 | 备注 |
|------|--------------|------|
| 检索路径选择（Planner fast-path）| 0 | regex < 1ms |
| 检索路径选择（Planner LLM）| ~200 input / ~50 output | 仅复杂任务 |
| CRAG grading（5 篇并发）| ~350 input × 5 / ~20 output × 5 | 最大并发 |
| 查询改写（rewrite）| ~500 input / ~30 output | LOW 时触发 |
| Query Expansion（LLM）| ~300 input / ~100 output | 复杂查询 |
| 最终生成 | ~2000 input / ~500 output | answer generation |
| **Fast path（简单查询）** | **~500 input / ~200 output** | **禁用 CRAG** |
| **Slow path（复杂查询）** | **~4000+ input / ~600 output** | **全链路** |

**Fast path 触发条件**：Planner `_quick_complexity_check()` 返回 "simple" 且 `needs_expansion=False`

### 1.3 质量指标

| 指标 | 目标值 | 测量方式 |
|------|-------|---------|
| 拒答率 | 5-15% | NO_RESULTS + LOW(rewrite失败) / 总请求 |
| 高置信率 | > 60% | HIGH 决策 / 总请求 |
| 检索召回率@5 | > 75% | eval_dataset 人工标注 |
| 版本冲突检测率 | > 90% | 对抗测试集 |
| 幻觉率 | < 5% | 对抗测试集：检验答案是否引用不存在的内容 |

---

## 2. 埋点实现

### 2.1 追踪上下文（Trace Context）

```python
# src/observability/tracer.py — 新文件
import uuid
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"

@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    start_ms: float = field(default_factory=lambda: time.time() * 1000)
    end_ms: Optional[float] = None
    duration_ms: Optional[float] = field(default=None)
    status: SpanStatus = SpanStatus.OK
    error: Optional[str] = None
    attrs: dict = field(default_factory=dict)

    def end(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None):
        self.end_ms = time.time() * 1000
        self.duration_ms = self.end_ms - self.start_ms
        self.status = status
        self.error = error

# Thread-local trace context
_current_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_current_spans: ContextVar[List[Span]] = ContextVar("spans", default_factory=list)

def start_span(name: str, attrs: dict = None) -> Span:
    trace_id = _current_trace_id.get() or uuid.uuid4().hex
    _current_trace_id.set(trace_id)

    parent = _current_spans.get()[-1] if _current_spans.get() else None
    span = Span(
        name=name,
        parent_id=parent.span_id if parent else None,
        attrs=attrs or {},
    )
    _current_spans.get().append(span)
    return span

def end_span(span: Span, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None):
    span.end(status, error)

def get_trace_context() -> dict:
    """获取当前 trace 的所有 span，格式化为上报用"""
    return {
        "trace_id": _current_trace_id.get(),
        "spans": [
            {
                "name": s.name,
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "duration_ms": s.duration_ms,
                "status": s.status.value,
                "error": s.error,
                **s.attrs,
            }
            for s in _current_spans.get()
        ]
    }
```

### 2.2 CRAG 链路埋点

```python
# retrieval_grader.py — 改动：在 grade_retrieval() 中埋点

async def grade_retrieval(self, query: str, documents: List[Document]) -> GradeResult:
    from src.observability.tracer import start_span, end_span, SpanStatus

    span = start_span("crag.grade_retrieval", {
        "query_length": len(query),
        "doc_count": len(documents),
        "crag_max_retries": self.max_retries,
    })

    try:
        start = time.time()
        result = await self._grade_retrieval_impl(query, documents)
        span.end(status=SpanStatus.OK, attrs={
            "decision": result.decision.value,
            "high_count": result.high_count,
            "medium_count": result.medium_count,
            "low_count": result.low_count,
            "avg_score": result.avg_score,
            "latency_ms": (time.time() - start) * 1000,
        })
        return result
    except Exception as e:
        span.end(status=SpanStatus.ERROR, error=str(e))
        raise
```

### 2.3 端到端 Chat 埋点

```python
# chat_service.py — 改动：在 achat() 中埋点

async def achat(self, message, session_id, username, user_context=None):
    from src.observability.tracer import start_span, end_span, get_trace_context, SpanStatus

    total_span = start_span("chat.total", {
        "session_id": session_id,
        "username": username,
        "message_length": len(message),
    })

    try:
        # ... 现有逻辑 ...

        result = await arun_agent(...)
        total_span.end(status=SpanStatus.OK, attrs={
            "answer_length": len(result["answer"]),
            "used_agent": result["used_agent"],
            "sources_length": len(result["sources"]) if result["sources"] else 0,
            "trace": get_trace_context(),  # 上报完整链路
        })
        return result
    except Exception as e:
        total_span.end(status=SpanStatus.ERROR, error=str(e))
        raise
```

---

## 3. 成本估算装饰器

```python
# src/observability/cost_tracker.py — 新文件
import tiktoken
from dataclasses import dataclass
from typing import Optional

@dataclass
class CostRecord:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    endpoint: str
    error: Optional[str] = None

    @property
    def estimated_cost_usd(self) -> float:
        """基于各模型定价估算（单位：美元）"""
        pricing = {
            "qwen3.5-flash": (0.0001, 0.0001),   # $ / 1M tokens (input, output)
            "qwen-max": (0.02, 0.06),
            "gpt-4o-mini": (0.00015, 0.0006),
            "gte-rerank-v2": (0.0001, 0.0001),   # 按调用次计
        }
        inp, out = pricing.get(self.model, (0.001, 0.001))
        return self.input_tokens / 1_000_000 * inp + self.output_tokens / 1_000_000 * out


class CostTracker:
    """
    跟踪每次 LLM 调用的 token 消耗和成本。
    使用 tiktoken 估算 token 数（不支持 qwen，需用粗略比率 1 token ≈ 1.5 chars）。
    """

    def estimate_tokens(self, text: str, model: str) -> int:
        if model.startswith("gpt"):
            try:
                enc = tiktoken.encoding_for_model("gpt-4o-mini")
                return len(enc.encode(text))
            except Exception:
                return int(len(text) / 1.5)
        else:
            # qwen / 通用估算
            return int(len(text) / 1.5)

    def record(self, model: str, input_text: str, output_text: str,
               latency_ms: float, endpoint: str, error: Optional[str] = None) -> CostRecord:
        record = CostRecord(
            model=model,
            input_tokens=self.estimate_tokens(input_text, model),
            output_tokens=self.estimate_tokens(output_text, model),
            latency_ms=latency_ms,
            endpoint=endpoint,
            error=error,
        )
        # 上报到 metrics collector（见下节）
        metrics_collector.record_cost(record)
        return record
```

---

## 4. Metrics 上报到 Prometheus

```python
# src/observability/metrics.py — 新文件
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

registry = CollectorRegistry()

# 计数器
chat_requests_total = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["status", "agent"],
    registry=registry,
)

crag_decisions_total = Counter(
    "crag_decisions_total",
    "CRAG decision counts",
    ["decision"],  # high / medium / low / no_results
    registry=registry,
)

llm_errors_total = Counter(
    "llm_errors_total",
    "LLM errors by type",
    ["type", "model"],  # rate_limited / server_error / timeout / parse_error
    registry=registry,
)

# 直方图
chat_latency_seconds = Histogram(
    "chat_latency_seconds",
    "End-to-end chat latency",
    buckets=[0.5, 1, 2, 5, 10, 15, 30, 60],
    registry=registry,
)

crag_grading_latency_seconds = Histogram(
    "crag_grading_latency_seconds",
    "CRAG grading latency per batch",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20],
    registry=registry,
)

llm_token_usage = Histogram(
    "llm_tokens_total",
    "LLM token usage",
    ["model", "direction"],  # direction: input / output
    buckets=[10, 100, 500, 1000, 5000, 20000],
    registry=registry,
)

# 仪表
retrieval_avg_score = Gauge(
    "crag_retrieval_avg_score",
    "Average retrieval relevance score",
    registry=registry,
)

vectorstore_doc_count = Gauge(
    "vectorstore_doc_count",
    "Number of chunks in vector store",
    registry=registry,
)


class MetricsCollector:
    """中央 metrics 收集器"""

    def record_chat(self, latency_s: float, agent: str, status: str):
        chat_requests_total.labels(status=status, agent=agent).inc()
        chat_latency_seconds.observe(latency_s)

    def record_crag_decision(self, decision: str):
        crag_decisions_total.labels(decision=decision).inc()

    def record_llm_error(self, error_type: str, model: str):
        llm_errors_total.labels(type=error_type, model=model).inc()

    def record_cost(self, record: CostRecord):
        if not record.error:
            llm_token_usage.labels(model=record.model, direction="input").observe(record.input_tokens)
            llm_token_usage.labels(model=record.model, direction="output").observe(record.output_tokens)

metrics_collector = MetricsCollector()
```

### `/metrics` 端点

```python
# main.py 新增
from src.observability.metrics import registry
from prometheus_client import generate_latest

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(registry),
        media_type="text/plain",
    )
```

---

## 5. 实施计划

| 阶段 | 内容 | 改动文件 |
|------|------|---------|
| Phase 1 | `Span` + `start_span/end_span` 埋点框架 | `src/observability/tracer.py` (新) |
| Phase 2 | CRAG + chat 链路埋点集成 | `retrieval_grader.py`, `chat_service.py` |
| Phase 3 | `CostTracker` + tiktoken 估算 | `src/observability/cost_tracker.py` (新) |
| Phase 4 | Prometheus metrics 注册 | `src/observability/metrics.py` (新) |
| Phase 5 | `/metrics` 端点 + Grafana dashboard JSON | `main.py` + `docs/grafana/` |
| Phase 6 | P50/P95/P99 告警规则 | `docs/observability/alerts.yml` |
