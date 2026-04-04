"""
OpenTelemetry 集成 - Tier 1 开源复用

支持导出 traces 到任意 OTLP 兼容的接收端（Jaeger、Tempo、Honeycomb、阿里云 ARMS 等）。
通过环境变量 OTEL_ENABLED=true 启用。

依赖：
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
    pip install opentelemetry-instrumentation-fastapi
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_otel_initialized = False
_tracer_provider = None


async def init_otel() -> bool:
    """
    初始化 OpenTelemetry（环境变量 OTEL_ENABLED=true 时启用）。

    Returns:
        True 表示初始化成功
    """
    global _otel_initialized, _tracer_provider

    if not os.environ.get("OTEL_ENABLED", "").lower() in ("true", "1", "yes"):
        logger.info("[OTEL] 未启用（设置 OTEL_ENABLED=true 开启）")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    except ImportError:
        logger.warning("[OTEL] opentelemetry 未安装，跳过 OTEL 初始化")
        return False

    try:
        service_name = os.environ.get("OTEL_SERVICE_NAME", "enterprise-knowledge-agent")
        resource = Resource(attributes={
            SERVICE_NAME: service_name,
        })
        _tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(_tracer_provider)

        exporter_type = os.environ.get("OTEL_EXPORTER", "console")
        if exporter_type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                endpoint = os.environ.get(
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "http://localhost:4317"
                )
                exporter = OTLPSpanExporter(endpoint=endpoint)
                _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(f"[OTEL] 已连接到 OTLP 端点: {endpoint}")
            except ImportError:
                logger.warning("[OTEL] opentelemetry-exporter-otlp 未安装，使用 console exporter")
                _add_console_exporter(_tracer_provider)
        else:
            _add_console_exporter(_tracer_provider)

        _otel_initialized = True
        logger.info("[OTEL] OpenTelemetry 初始化完成")
        return True

    except Exception as e:
        logger.warning(f"[OTEL] 初始化失败: {e}")
        return False


def _add_console_exporter(provider):
    """添加 console span exporter（用于调试）"""
    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("[OTEL] 使用 console span exporter（调试模式）")
    except Exception as e:
        logger.warning(f"[OTEL] console exporter 失败: {e}")


def instrument_fastapi_app(app):
    """自动 instrument FastAPI 应用（注入 trace context）"""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[OTEL] FastAPI instrumentation 已启用")
    except ImportError:
        logger.warning("[OTEL] opentelemetry-instrumentation-fastapi 未安装")


def shutdown_otel():
    """优雅关闭 OTEL"""
    global _tracer_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("[OTEL] 已关闭")
        except Exception as e:
            logger.warning(f"[OTEL] 关闭失败: {e}")


# ==================== traced 装饰器（供 tracer.py 复用）====================
# traced 提供 @traced("span_name") 装饰器，内部使用自定义 Span + 可选 OTEL 上报
# 不依赖 OTEL 初始化（OTEL 未启用时仅使用本地 Span）
def traced(span_name: str, attrs_func=None):
    """
    函数装饰器：为异步函数添加分布式追踪。

    内部使用 src.observability.tracer 中的自定义 Span，
    并在 OTEL 已初始化时自动上报到 OTEL Collector。

    Args:
        span_name: span 名称
        attrs_func: 可选函数，接收 (args, kwargs)，返回 dict，
                    用于在 span 上附加动态属性

    用法：
        @traced("my.operation")
        async def my_func():
            ...

        @traced("my.op", attrs_func=lambda args, kw: {"user": args[0]})
        async def my_func2(user):
            ...
    """
    import functools
    import asyncio
    from src.observability.tracer import start_span, end_span, SpanStatus

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                attrs = {}
                if attrs_func:
                    try:
                        attrs = attrs_func(args, kwargs)
                    except Exception:
                        pass
                with start_span(span_name, attrs=attrs) as span:
                    try:
                        result = await func(*args, **kwargs)
                        end_span(span, SpanStatus.OK)
                        return result
                    except Exception as e:
                        end_span(span, SpanStatus.ERROR, error=str(e))
                        raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                attrs = {}
                if attrs_func:
                    try:
                        attrs = attrs_func(args, kwargs)
                    except Exception:
                        pass
                with start_span(span_name, attrs=attrs) as span:
                    try:
                        result = func(*args, **kwargs)
                        end_span(span, SpanStatus.OK)
                        return result
                    except Exception as e:
                        end_span(span, SpanStatus.ERROR, error=str(e))
                        raise
            return sync_wrapper
    return decorator

