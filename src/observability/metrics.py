"""
Prometheus Metrics 收集器
定义所有业务指标的 Counter / Histogram / Gauge，注册到 Prometheus registry。
"""
import logging
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# 全局 registry（单例）
_metrics_registry = CollectorRegistry()

# ──────────────────────────────────────────────────────────────────
# 计数器
# ──────────────────────────────────────────────────────────────────

chat_requests_total = Counter(
    "ekb_chat_requests_total",
    "Total chat requests",
    ["status", "agent"],
    registry=_metrics_registry,
)

crag_decisions_total = Counter(
    "ekb_crag_decisions_total",
    "CRAG evaluation decision counts",
    ["decision"],  # high / medium / low / no_results
    registry=_metrics_registry,
)

llm_errors_total = Counter(
    "ekb_llm_errors_total",
    "LLM errors by error type and model",
    ["error_type", "model"],  # rate_limited / server_error / timeout / parse_error
    registry=_metrics_registry,
)

query_expansion_total = Counter(
    "ekb_query_expansion_total",
    "Query expansion trigger counts",
    ["strategy", "status"],  # rule / llm / hyde, success / failed
    registry=_metrics_registry,
)

conflict_detection_total = Counter(
    "ekb_conflict_detection_total",
    "Document conflict detection results",
    ["severity", "action"],  # high / medium / low, reject / conflict_summary / none
    registry=_metrics_registry,
)

retrieval_retries_total = Counter(
    "ekb_retrieval_retries_total",
    "CRAG rewrite/retrieval retry counts",
    ["attempt"],  # 1 / 2 / 3
    registry=_metrics_registry,
)

# ──────────────────────────────────────────────────────────────────
# 安全指标（Rate Limiting / RBAC / Audit）
# ──────────────────────────────────────────────────────────────────

rate_limit_hits_total = Counter(
    "ekb_rate_limit_hits_total",
    "Rate limit hits by endpoint and identifier type",
    ["endpoint", "id_type"],
    registry=_metrics_registry,
)

auth_attempts_total = Counter(
    "ekb_auth_attempts_total",
    "Authentication attempts",
    ["result"],
    registry=_metrics_registry,
)

rbac_checks_total = Counter(
    "ekb_rbac_checks_total",
    "RBAC permission checks",
    ["resource", "action", "result"],
    registry=_metrics_registry,
)

audit_events_total = Counter(
    "ekb_audit_events_total",
    "Audit log events",
    ["event_type", "result"],
    registry=_metrics_registry,
)

# ──────────────────────────────────────────────────────────────────
# 直方图（延迟分布）
# ──────────────────────────────────────────────────────────────────

chat_latency_seconds = Histogram(
    "ekb_chat_latency_seconds",
    "End-to-end chat request latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0],
    registry=_metrics_registry,
)

crag_grade_latency_seconds = Histogram(
    "ekb_crag_grade_latency_seconds",
    "CRAG grading latency per batch in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0],
    registry=_metrics_registry,
)

query_rewrite_latency_seconds = Histogram(
    "ekb_query_rewrite_latency_seconds",
    "Query rewrite latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    registry=_metrics_registry,
)

query_expansion_latency_seconds = Histogram(
    "ekb_query_expansion_latency_seconds",
    "Query expansion latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    registry=_metrics_registry,
)

reranker_latency_seconds = Histogram(
    "ekb_reranker_latency_seconds",
    "Reranker call latency in seconds",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    registry=_metrics_registry,
)

vector_search_latency_seconds = Histogram(
    "ekb_vector_search_latency_seconds",
    "Vector search latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    registry=_metrics_registry,
)

llm_token_usage = Histogram(
    "ekb_llm_tokens",
    "LLM token usage (estimated)",
    ["model", "direction"],
    buckets=[10, 50, 100, 500, 1000, 5000, 20000],
    registry=_metrics_registry,
)

# ──────────────────────────────────────────────────────────────────
# 仪表（Gauge）
# ──────────────────────────────────────────────────────────────────

retrieval_avg_score = Gauge(
    "ekb_crag_retrieval_avg_score",
    "Latest CRAG retrieval average relevance score",
    registry=_metrics_registry,
)

vectorstore_chunk_count = Gauge(
    "ekb_vectorstore_chunk_count",
    "Number of chunks in vector store",
    registry=_metrics_registry,
)

active_sessions = Gauge(
    "ekb_active_sessions",
    "Number of currently active chat sessions",
    registry=_metrics_registry,
)

grading_batch_size = Gauge(
    "ekb_crag_grading_batch_size",
    "Latest CRAG grading batch size (number of docs graded)",
    registry=_metrics_registry,
)


# ──────────────────────────────────────────────────────────────────
# Metrics Collector（业务层调用入口）
# ──────────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    中央 metrics 收集器。
    业务代码通过此类记录指标，避免直接引用 prometheus 对象。
    """

    def record_chat(self, latency_s: float, agent: str, status: str):
        chat_requests_total.labels(status=status, agent=agent).inc()
        chat_latency_seconds.observe(latency_s)

    def record_crag_decision(self, decision: str):
        crag_decisions_total.labels(decision=decision).inc()

    def record_crag_grading(self, latency_s: float, batch_size: int):
        crag_grade_latency_seconds.observe(latency_s)
        grading_batch_size.set(batch_size)

    def record_llm_error(self, error_type: str, model: str):
        llm_errors_total.labels(error_type=error_type, model=model).inc()

    def record_llm_tokens(self, model: str, direction: str, tokens: int):
        llm_token_usage.labels(model=model, direction=direction).observe(tokens)

    def record_query_rewrite(self, latency_s: float, success: bool):
        query_rewrite_latency_seconds.observe(latency_s)

    def record_query_expansion(self, latency_s: float, strategy: str, success: bool):
        status = "success" if success else "failed"
        query_expansion_total.labels(strategy=strategy, status=status).inc()
        query_expansion_latency_seconds.observe(latency_s)

    def record_reranker(self, latency_s: float):
        reranker_latency_seconds.observe(latency_s)

    def record_vector_search(self, latency_s: float):
        vector_search_latency_seconds.observe(latency_s)

    def record_conflict(self, severity: str, action: str):
        conflict_detection_total.labels(severity=severity, action=action).inc()

    def record_retrieval_avg_score(self, score: float):
        retrieval_avg_score.set(score)

    def update_chunk_count(self, count: int):
        vectorstore_chunk_count.set(count)

    def record_rate_limit_hit(self, endpoint: str, id_type: str):
        rate_limit_hits_total.labels(endpoint=endpoint, id_type=id_type).inc()

    def record_auth_attempt(self, result: str):
        auth_attempts_total.labels(result=result).inc()

    def record_rbac_check(self, resource: str, action: str, result: str):
        rbac_checks_total.labels(resource=resource, action=action, result=result).inc()

    def record_audit_event(self, event_type: str, result: str):
        audit_events_total.labels(event_type=event_type, result=result).inc()


_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_metrics() -> bytes:
    return generate_latest(_metrics_registry)


def get_content_type() -> str:
    return CONTENT_TYPE_LATEST
