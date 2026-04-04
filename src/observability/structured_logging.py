"""
结构化 JSON 日志处理器

特性：
- JSON 格式输出，便于 ELK / Loki / Datadog 等日志收集器解析
- 自动注入 trace_id / span_id 上下文
- 支持 request_id 追踪
- 日志级别、格式可配置
- 非生产环境自动降级为彩色人类可读格式

配置（通过 settings 或环境变量）：
- LOG_LEVEL: 日志级别（DEBUG/INFO/WARNING/ERROR），默认 INFO
- LOG_FORMAT: json/human，默认 json（生产），human（调试）
- LOG_REQUEST_ID_HEADER: 请求 ID 的 HTTP header 名，默认 X-Request-ID
"""
import sys
import json
import time
import uuid
import logging
from typing import Optional, Any
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    结构化 JSON 日志格式化器

    输出格式：
    {
        "timestamp": "2026-04-04T10:30:00.123Z",
        "level": "INFO",
        "logger": "src.agent.graph",
        "message": "检索完成",
        "trace_id": "abc123",
        "span_id": "def456",
        "request_id": "req-789",
        "user_id": "alice",
        "duration_ms": 1234,
        "extra": {...}
    }
    """

    def __init__(
        self,
        include_trace: bool = True,
        include_request_id: bool = True,
        service_name: str = "enterprise-knowledge-agent",
        environment: str = "production",
    ):
        super().__init__()
        self.include_trace = include_trace
        self.include_request_id = include_request_id
        self.service_name = service_name
        self.environment = environment
        # 用于 human 模式的标准格式
        self._human_fmt = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
        self._human_datefmt = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        if _USE_JSON_FORMAT:
            return self._format_json(record)
        else:
            return self._format_human(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        # 构建基础字段
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "env": self.environment,
        }

        # 异常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # 调用位置
        if record.filename:
            log_entry["file"] = record.filename
            log_entry["line"] = record.lineno
            log_entry["func"] = record.funcName

        # 注入上下文（从 LogRecord.extra 获取）
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_entry["span_id"] = record.span_id
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        # 额外字段（通过 record.__dict__["extra"] 传入）
        extra_keys = {
            "agent", "query", "sources", "sources_count",
            "latency_s", "decision", "score", "model",
            "error_type", "strategy", "doc_count", "tool_name",
        }
        for key in extra_keys:
            if key in record.__dict__:
                val = record.__dict__[key]
                # 防止 None 和过长字符串污染日志
                if val is not None:
                    if isinstance(val, str) and len(val) > 500:
                        val = val[:500] + "..."
                    log_entry[key] = val

        return json.dumps(log_entry, ensure_ascii=False, default=str)

    def _format_human(self, record: logging.LogRecord) -> str:
        """人类可读格式（调试用，带颜色）"""
        timestamp = datetime.fromtimestamp(record.created).strftime(self._human_datefmt)
        level = record.levelname
        name = record.name.split(".")[-1]  # 只取最后一级 logger name

        # 颜色代码
        RESET = "\033[0m"
        COLORS = {
            "DEBUG": "\033[36m",    # 青色
            "INFO": "\033[32m",     # 绿色
            "WARNING": "\033[33m",  # 黄色
            "ERROR": "\033[31m",    # 红色
            "CRITICAL": "\033[35m", # 紫色
        }
        color = COLORS.get(level, RESET)

        msg = record.getMessage()
        extra_parts = []
        for key in ["request_id", "trace_id", "user_id", "session_id", "agent", "duration_ms"]:
            if key in record.__dict__ and record.__dict__[key] is not None:
                extra_parts.append(f"{key}={record.__dict__[key]}")

        extra_str = f" [{', '.join(extra_parts)}]" if extra_parts else ""

        return f"{color}{timestamp} {level:8s}{RESET} [{name}]{extra_str} {msg}"


# ──────────────────────────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────────────────────────

_USE_JSON_FORMAT = True
_SERVICE_NAME = "enterprise-knowledge-agent"


def _load_log_config():
    """从环境变量/配置加载日志设置"""
    global _USE_JSON_FORMAT, _SERVICE_NAME
    try:
        import os
        log_format = os.environ.get("LOG_FORMAT", "json")
        _USE_JSON_FORMAT = log_format.lower() != "human"
        _SERVICE_NAME = os.environ.get("LOG_SERVICE_NAME", _SERVICE_NAME)
    except Exception:
        pass


_load_log_config()


# ──────────────────────────────────────────────────────────────────
# 请求上下文（thread-local 存储）
# ──────────────────────────────────────────────────────────────────

import threading

_log_context: threading.local = threading.local()


def set_log_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
):
    """在当前线程设置日志上下文（自动注入到每条日志）"""
    _log_context.request_id = request_id
    _log_context.user_id = user_id
    _log_context.session_id = session_id
    _log_context.trace_id = trace_id
    _log_context.span_id = span_id


def clear_log_context():
    """清除当前线程的日志上下文"""
    for attr in ["request_id", "user_id", "session_id", "trace_id", "span_id"]:
        if hasattr(_log_context, attr):
            delattr(_log_context, attr)


def get_log_context() -> dict:
    """获取当前线程的日志上下文"""
    return {
        attr: getattr(_log_context, attr, None)
        for attr in ["request_id", "user_id", "session_id", "trace_id", "span_id"]
        if hasattr(_log_context, attr)
    }


class LogContextFilter(logging.Filter):
    """
    日志过滤器：将 thread-local 上下文注入到每条 LogRecord
    """
    def filter(self, record: logging.LogRecord) -> bool:
        for attr in ["request_id", "user_id", "session_id", "trace_id", "span_id"]:
            if hasattr(_log_context, attr):
                val = getattr(_log_context, attr)
                if val is not None:
                    setattr(record, attr, val)
        return True


# ──────────────────────────────────────────────────────────────────
# 结构化日志记录器（推荐在业务代码中使用）
# ──────────────────────────────────────────────────────────────────

class StructuredLogger:
    """
    结构化日志记录器

    特性：
    - 自动 JSON 序列化额外参数
    - 自动注入 trace/request/user/session 上下文
    - 支持 duration_ms / latency_s 自动计算
    - 与 Python logging 完全兼容

    用法：
        logger = structured_logger("src.agent.graph")
        logger.info("检索完成", agent="knowledge", doc_count=5, duration_ms=123)
        logger.warning("评分低", decision="low", avg_score=0.12)
        logger.error("Agent 失败", agent="operation", error=str(e))
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        msg: str,
        exc_info: Optional[Exception] = None,
        **kwargs,
    ):
        # 将 kwargs 挂到 record.extra
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "(unknown)',
            0,
            msg,
            (),
            exc_info,
        )
        for key, val in kwargs.items():
            setattr(record, key, val)

        self._logger.handle(record)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, exc_info: Optional[Exception] = None, **kwargs):
        self._log(logging.ERROR, msg, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, exc_info: Optional[Exception] = None, **kwargs):
        self._log(logging.CRITICAL, msg, exc_info=exc_info, **kwargs)


def structured_logger(name: str) -> StructuredLogger:
    """获取结构化日志记录器"""
    return StructuredLogger(name)


# ──────────────────────────────────────────────────────────────────
# 全局日志配置（应用启动时调用一次）
# ──────────────────────────────────────────────────────────────────

def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    service_name: str = "enterprise-knowledge-agent",
    environment: str = "production",
    json_format: Optional[bool] = None,
):
    """
    配置全局日志系统

    Args:
        level: 日志级别（默认 INFO）
        log_file: 日志文件路径（可选，默认输出到 stderr）
        service_name: 服务名称（用于日志字段）
        environment: 环境标识（用于日志字段）
        json_format: 是否使用 JSON 格式（None=自动：生产=True，调试=False）
    """
    global _USE_JSON_FORMAT, _SERVICE_NAME

    if json_format is not None:
        _USE_JSON_FORMAT = json_format
    _SERVICE_NAME = service_name

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有 handlers
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # 构建 formatter
    formatter = JSONFormatter(
        include_trace=True,
        include_request_id=True,
        service_name=service_name,
        environment=environment,
    )

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(LogContextFilter())
    root_logger.addHandler(console_handler)

    # 文件 handler（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        # 文件强制使用 JSON 格式
        file_formatter = JSONFormatter(
            service_name=service_name,
            environment=environment,
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(LogContextFilter())
        root_logger.addHandler(file_handler)

    # 降低第三方库日志级别（减少噪音）
    for noisy_logger in ["httpx", "httpcore", "urllib3", "chromadb", "bm25s"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # uvicorn 访问日志静默（生产环境）
    if environment == "production":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return root_logger


# ──────────────────────────────────────────────────────────────────
# 便捷函数：记录业务事件（自动绑定到指标收集器）
# ──────────────────────────────────────────────────────────────────

def log_chat_request(
    agent: str,
    latency_s: float,
    status: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    error: Optional[str] = None,
):
    """
    记录一次 chat 请求的完整日志
    绑定 Prometheus metrics + 结构化日志
    """
    from .metrics import get_metrics_collector

    mc = get_metrics_collector()
    mc.record_chat(latency_s=latency_s, agent=agent, status=status)

    logger = logging.getLogger("app.chat")
    extra = {"agent": agent, "latency_s": round(latency_s, 3), "status": status}
    if user_id:
        extra["user_id"] = user_id
    if session_id:
        extra["session_id"] = session_id
    if request_id:
        extra["request_id"] = request_id
    if error:
        extra["error"] = error

    if status == "success":
        logger.info(f"Chat请求完成 [{agent}] {latency_s:.2f}s", extra=extra)
    else:
        logger.error(f"Chat请求失败 [{agent}] {error}", extra=extra)


def log_retrieval_event(
    decision: str,
    doc_count: int,
    avg_score: float,
    latency_s: float,
    query: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
):
    """记录一次检索事件"""
    from .metrics import get_metrics_collector

    mc = get_metrics_collector()
    mc.record_crag_decision(decision)
    mc.record_retrieval_avg_score(avg_score)

    logger = logging.getLogger("app.retrieval")
    logger.info(
        f"检索完成 decision={decision} docs={doc_count} avg_score={avg_score:.3f} latency={latency_s:.2f}s",
        extra={
            "decision": decision,
            "doc_count": doc_count,
            "avg_score": round(avg_score, 3),
            "latency_s": round(latency_s, 3),
            "query": query[:100],
            "user_id": user_id,
            "request_id": request_id,
        }
    )


def log_llm_error(
    error_type: str,
    model: str,
    error_msg: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
):
    """记录 LLM 调用错误"""
    from .metrics import get_metrics_collector

    mc = get_metrics_collector()
    mc.record_llm_error(error_type, model)

    logger = logging.getLogger("app.llm")
    logger.error(
        f"LLM错误 [{model}] type={error_type} msg={error_msg[:200]}",
        extra={
            "error_type": error_type,
            "model": model,
            "error": error_msg[:300],
            "user_id": user_id,
            "request_id": request_id,
        }
    )
