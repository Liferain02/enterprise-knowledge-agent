"""
熔断器（Circuit Breaker）+ 重试机制

基于状态机的熔断器，防止外部依赖（LLM/Redis/ChromaDB）故障时产生雪崩效应。
当失败次数超过阈值时进入 OPEN 状态，直接返回降级响应而非调用外部服务。
状态转换：CLOSED → OPEN → HALF_OPEN → CLOSED

同时提供统一的重试装饰器，基于 tenacity 实现指数退避。
"""
import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Optional, Callable, Any, TypeVar, Union, Type
from functools import wraps
from dataclasses import dataclass, field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from tenacity import AsyncRetrying, StopAfterAttempt, wait_random_exponential

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 熔断器状态
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"       # 正常，允许请求通过
    OPEN = "open"           # 熔断，拒绝所有请求
    HALF_OPEN = "half_open"  # 半开，允许一个测试请求


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    name: str = "default"
    failure_threshold: int = 5        # 失败次数超过此值则熔断
    success_threshold: int = 3        # 半开状态下成功次数超过此值则恢复
    timeout_seconds: float = 30.0     # OPEN 状态持续时间后进入 HALF_OPEN
    half_open_max_calls: int = 1      # 半开状态下允许的并发测试请求数
    excluded_exceptions: tuple = ()    # 不计入失败的异常类型


@dataclass
class CircuitBreakerMetrics:
    """熔断器指标"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float = 0.0
    last_failure_reason: str = ""


class CircuitBreaker:
    """
    线程安全的熔断器实现。

    状态转换：
    CLOSED ─(失败>=阈值)─→ OPEN ─(超时)─→ HALF_OPEN
      ↑                           │
      │                   (成功<阈值)
      └──────(成功>=阈值)──────────┘
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.RLock()
        self._metrics = CircuitBreakerMetrics()
        self._fallback_handler: Optional[Callable] = None

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._get_state_unlocked()

    def _get_state_unlocked(self) -> CircuitState:
        """获取状态（需持有锁）"""
        if self._state == CircuitState.OPEN:
            # 检查是否超时
            if time.time() - self._last_failure_time >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def _transition_to(self, new_state: CircuitState):
        """状态转换"""
        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0

        logger.info(
            f"[CircuitBreaker/{self.config.name}] "
            f"状态变更: {old_state.value} → {new_state.value}"
        )

    def is_request_allowed(self) -> tuple[bool, str]:
        """
        检查是否允许请求。

        Returns:
            (allowed, reason)
        """
        with self._lock:
            state = self._get_state_unlocked()

            if state == CircuitState.CLOSED:
                return True, "closed"

            if state == CircuitState.OPEN:
                return False, "open"

            # HALF_OPEN：允许有限数量的测试请求
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True, "half_open"

            return False, "half_open_busy"

    def record_success(self):
        """记录成功"""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            else:
                # 成功后重置失败计数
                self._failure_count = 0

    def record_failure(self, exc: Optional[Exception] = None):
        """记录失败"""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = time.time()
            self._metrics.last_failure_reason = type(exc).__name__ if exc else "Unknown"

            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态下失败，立即回到 OPEN
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def record_rejected(self):
        """记录被拒绝的请求"""
        with self._lock:
            self._metrics.rejected_calls += 1

    def get_metrics(self) -> dict:
        """获取熔断器指标"""
        with self._lock:
            return {
                "name": self.config.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": self._metrics.total_calls,
                "successful_calls": self._metrics.successful_calls,
                "failed_calls": self._metrics.failed_calls,
                "rejected_calls": self._metrics.rejected_calls,
                "state_changes": self._metrics.state_changes,
                "last_failure_reason": self._metrics.last_failure_reason,
                "last_failure_time": self._metrics.last_failure_time,
            }

    def set_fallback(self, handler: Callable):
        """设置降级处理器"""
        self._fallback_handler = handler

    def get_fallback_result(self, *args, **kwargs) -> Any:
        """执行降级处理"""
        if self._fallback_handler:
            try:
                return self._fallback_handler(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[CircuitBreaker/{self.config.name}] 降级处理异常: {e}")
        return None

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        异步调用（带熔断保护）。

        用法：
            result = await circuit_breaker.call_async(my_async_func, arg1, arg2)
        """
        allowed, reason = self.is_request_allowed()
        if not allowed:
            self.record_rejected()
            logger.warning(
                f"[CircuitBreaker/{self.config.name}] 请求被拒绝 (state={reason})"
            )
            return self.get_fallback_result(*args, **kwargs)

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            self.record_success()
            return result
        except self.config.excluded_exceptions:
            self.record_success()
            raise
        except Exception as e:
            self.record_failure(e)
            raise

    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        同步调用（带熔断保护）。
        """
        allowed, reason = self.is_request_allowed()
        if not allowed:
            self.record_rejected()
            logger.warning(
                f"[CircuitBreaker/{self.config.name}] 请求被拒绝 (state={reason})"
            )
            return self.get_fallback_result(*args, **kwargs)

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except self.config.excluded_exceptions:
            self.record_success()
            raise
        except Exception as e:
            self.record_failure(e)
            raise


# ============================================================
# 全局熔断器实例
# ============================================================

_circuit_breakers: dict[str, CircuitBreaker] = {}
_cb_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """获取或创建熔断器"""
    with _cb_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(config or CircuitBreakerConfig(name=name))
        return _circuit_breakers[name]


# 预定义熔断器名称
CB_LLM = "llm"
CB_VECTORSTORE = "vectorstore"
CB_REDIS = "redis"
CB_RERANKER = "reranker"


# ============================================================
# 预配置熔断器
# ============================================================

def get_llm_circuit_breaker() -> CircuitBreaker:
    """LLM 熔断器：失败阈值较低（API 易限流），超时较长"""
    cb = get_circuit_breaker(CB_LLM, CircuitBreakerConfig(
        name=CB_LLM,
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=60.0,
    ))
    cb.set_fallback(lambda *args, **kwargs: {
        "answer": "服务暂时繁忙，请稍后重试。",
        "sources": "",
        "used_agent": "circuit_breaker_fallback",
    })
    return cb


def get_vectorstore_circuit_breaker() -> CircuitBreaker:
    """向量库熔断器：降级为纯 BM25 搜索"""
    cb = get_circuit_breaker(CB_VECTORSTORE, CircuitBreakerConfig(
        name=CB_VECTORSTORE,
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=30.0,
    ))
    return cb


def get_redis_circuit_breaker() -> CircuitBreaker:
    """Redis 熔断器：降级为内存缓存"""
    cb = get_circuit_breaker(CB_REDIS, CircuitBreakerConfig(
        name=CB_REDIS,
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=15.0,
    ))
    return cb


# ============================================================
# 重试装饰器
# ============================================================

def retry_on_failure(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    指数退避重试装饰器（同步）。

    用法：
        @retry_on_failure(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"[Retry] {func.__name__} 达到最大重试次数 {max_attempts}")
                        raise
                    wait_time = min(max_wait, min_wait * (multiplier ** (attempt - 1)))
                    logger.warning(
                        f"[Retry] {func.__name__} 重试 {attempt}/{max_attempts}，"
                        f"等待 {wait_time:.1f}s，异常: {e}"
                    )
                    time.sleep(wait_time)
        return wrapper
    return decorator


def async_retry_on_failure(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    指数退避重试装饰器（异步）。

    用法：
        @async_retry_on_failure(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        async def my_async_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"[AsyncRetry] {func.__name__} 达到最大重试次数 {max_attempts}")
                        raise
                    wait_time = min(max_wait, min_wait * (multiplier ** (attempt - 1)))
                    logger.warning(
                        f"[AsyncRetry] {func.__name__} 重试 {attempt}/{max_attempts}，"
                        f"等待 {wait_time:.1f}s，异常: {e}"
                    )
                    await asyncio.sleep(wait_time)
        return wrapper
    return decorator


# ============================================================
# 指标上报（与 Prometheus 集成）
# ============================================================

def record_circuit_breaker_metrics():
    """将熔断器指标上报到 Prometheus"""
    try:
        from src.observability.metrics import get_metrics_collector
        mc = get_metrics_collector()
        for name, cb in _circuit_breakers.items():
            metrics = cb.get_metrics()
            # 记录状态变化事件
            if metrics["state"] == "open":
                mc.record_llm_error(f"circuit_open_{name}", name)
    except Exception:
        pass
