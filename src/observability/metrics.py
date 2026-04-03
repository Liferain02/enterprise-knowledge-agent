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
    ["model", "direction"],  # input / output
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

    用法示例：
        mc = MetricsCollector()
        mc.record_chat(latency_s=3.2, agent="knowledge", status="success")
        mc.record_crag_decision("high")
        mc.record_llm_error("rate_limited", "qwen3.5-flash")
    """

    def record_chat(self, latency_s: float, agent: str, status: str):
        """记录一次 chat 请求"""
        chat_requests_total.labels(status=status, agent=agent).inc()
        chat_latency_seconds.observe(latency_s)

    def record_crag_decision(self, decision: str):
        """记录 CRAG 评估决策"""
        crag_decisions_total.labels(decision=decision).inc()

    def record_crag_grading(self, latency_s: float, batch_size: int):
        """记录一次 CRAG grading batch 的延迟"""
        crag_grade_latency_seconds.observe(latency_s)
        grading_batch_size.set(batch_size)

    def record_llm_error(self, error_type: str, model: str):
        """记录 LLM 调用错误"""
        llm_errors_total.labels(error_type=error_type, model=model).inc()

    def record_llm_tokens(self, model: str, direction: str, tokens: int):
        """记录 LLM token 消耗（估算）"""
        llm_token_usage.labels(model=model, direction=direction).observe(tokens)

    def record_query_rewrite(self, latency_s: float, success: bool):
        """记录查询改写"""
        query_rewrite_latency_seconds.observe(latency_s)

    def record_query_expansion(self, latency_s: float, strategy: str, success: bool):
        """记录查询扩展"""
        status = "success" if success else "failed"
        query_expansion_total.labels(strategy=strategy, status=status).inc()
        query_expansion_latency_seconds.observe(latency_s)

    def record_reranker(self, latency_s: float):
        """记录 Reranker 调用"""
        reranker_latency_seconds.observe(latency_s)

    def record_vector_search(self, latency_s: float):
        """记录向量检索延迟"""
        vector_search_latency_seconds.observe(latency_s)

    def record_conflict(self, severity: str, action: str):
        """记录冲突检测结果"""
        conflict_detection_total.labels(severity=severity, action=action).inc()

    def record_retrieval_avg_score(self, score: float):
        """记录最近一次检索的平均相关分"""
        retrieval_avg_score.set(score)

    def update_chunk_count(self, count: int):
        """更新向量库 chunk 总数"""
        vectorstore_chunk_count.set(count)


# 全局实例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# ==================== 便捷导出 ====================

def get_metrics() -> bytes:
    """生成 Prometheus metrics 文本格式（供 /metrics 端点返回）"""
    return generate_latest(_metrics_registry)


def get_content_type() -> str:
    """返回 Prometheus metrics 的 Content-Type"""
    return CONTENT_TYPE_LATEST

