"""
API 限流模块

基于 Redis 的分布式限流（滑动窗口算法），支持：
- 匿名用户：按 IP 限流
- 登录用户：按用户名限流
- 全局限流：防止整体 DDoS

使用 Redis INCR + EXPIRE 实现原子计数，支持多实例部署。
当 Redis 不可用时，自动降级到内存限流（单实例）。
"""
import asyncio
import hashlib
import logging
import threading
import time
from typing import Optional, Tuple
from dataclasses import dataclass

from config.settings import get_settings

logger = logging.getLogger(__name__)

# 限流键前缀
_RATE_LIMIT_PREFIX = "ekb:ratelimit:"

# 默认限制（每分钟）
DEFAULT_RATE_LIMIT = 60
DEFAULT_RATE_LIMIT_ANONYMOUS = 30


@dataclass
class RateLimitResult:
    """限流检查结果"""
    allowed: bool  # 是否允许请求
    limit: int     # 限制数量
    remaining: int  # 剩余次数
    reset_at: float # 重置时间戳
    retry_after: float  # 距离下次可请求的秒数（仅在被限流时）


class RateLimiter:
    """
    滑动窗口限流器（基于 Redis，Redis 不可用时降级到内存）
    """

    def __init__(
        self,
        default_limit: int = DEFAULT_RATE_LIMIT,
        window_seconds: int = 60,
    ):
        self._settings = None
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self._redis_client = None
        self._redis_available = False
        self._memory_cache: dict = {}
        self._memory_locks: dict = {}
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def settings(self):
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    async def _init_redis(self):
        """初始化 Redis 连接"""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                import redis.asyncio as aioredis
                settings = self.settings
                host = getattr(settings, "redis_host", "disabled")
                if not host or host == "disabled":
                    self._redis_available = False
                else:
                    self._redis_client = aioredis.Redis(
                        host=host,
                        port=getattr(settings, "redis_port", 6379),
                        password=getattr(settings, "redis_password", None) or None,
                        db=getattr(settings, "redis_db", 0),
                        decode_responses=True,
                        socket_connect_timeout=2,
                        socket_timeout=3,
                    )
                    await self._redis_client.ping()
                    self._redis_available = True
                    logger.info("[RateLimiter] Redis 连接成功")
            except Exception as e:
                logger.warning(f"[RateLimiter] Redis 不可用，降级到内存限流: {e}")
                self._redis_available = False
                self._redis_client = None
            self._initialized = True

    def _make_key(self, identifier: str, endpoint: str = "default") -> str:
        """生成限流 key"""
        return f"{_RATE_LIMIT_PREFIX}{endpoint}:{identifier}"

    async def check(
        self,
        identifier: str,
        endpoint: str = "default",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> RateLimitResult:
        """
        检查是否允许请求

        Args:
            identifier: 用户标识（用户名或 IP）
            endpoint: 接口端点（不同端点可设置不同限制）
            limit: 限制次数（None 使用默认值）
            window_seconds: 时间窗口秒数

        Returns:
            RateLimitResult
        """
        await self._init_redis()

        limit = limit or self.default_limit
        window_seconds = window_seconds or self.window_seconds
        now = time.time()
        reset_at = now + window_seconds

        key = self._make_key(identifier, endpoint)

        if self._redis_available and self._redis_client:
            return await self._check_redis(key, limit, window_seconds, now, reset_at)
        else:
            return self._check_memory(key, limit, window_seconds, now, reset_at)

    async def _check_redis(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        now: float,
        reset_at: float,
    ) -> RateLimitResult:
        """Redis 滑动窗口限流"""
        try:
            pipe = self._redis_client.pipeline()
            pipe.incr(key)
            pipe.ttl(key)
            results = await pipe.execute()
            count = results[0]
            ttl = results[1]

            # 首次请求，设置过期时间
            if ttl == -1:
                await self._redis_client.expire(key, window_seconds)
                ttl = window_seconds

            remaining = max(0, limit - count)
            retry_after = 0.0
            if count > limit:
                retry_after = ttl if ttl > 0 else window_seconds
                logger.info(f"[RateLimiter] 限流触发: key={key}, count={count}, limit={limit}")

            return RateLimitResult(
                allowed=count <= limit,
                limit=limit,
                remaining=remaining,
                reset_at=now + (ttl if ttl > 0 else window_seconds),
                retry_after=retry_after,
            )
        except Exception as e:
            logger.warning(f"[RateLimiter] Redis 操作失败，降级到内存: {e}")
            self._redis_available = False
            return self._check_memory(key, limit, window_seconds, now, reset_at)

    def _check_memory(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        now: float,
        reset_at: float,
    ) -> RateLimitResult:
        """内存滑动窗口限流（单实例降级）"""
        with self._lock:
            if key not in self._memory_cache:
                self._memory_cache[key] = {
                    "count": 0,
                    "window_start": now,
                }

            entry = self._memory_cache[key]
            # 窗口过期，重置
            if now - entry["window_start"] >= window_seconds:
                entry["count"] = 0
                entry["window_start"] = now

            entry["count"] += 1
            remaining = max(0, limit - entry["count"])
            retry_after = 0.0
            if entry["count"] > limit:
                retry_after = window_seconds - (now - entry["window_start"])
                logger.info(f"[RateLimiter] 限流触发(内存): key={key}, count={entry['count']}, limit={limit}")

            return RateLimitResult(
                allowed=entry["count"] <= limit,
                limit=limit,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=max(0, retry_after),
            )

    async def reset(self, identifier: str, endpoint: str = "default"):
        """重置限流计数（管理员操作）"""
        await self._init_redis()
        key = self._make_key(identifier, endpoint)

        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.delete(key)
            except Exception:
                pass
        else:
            with self._lock:
                self._memory_cache.pop(key, None)

    def close(self):
        """关闭"""
        if self._redis_client:
            try:
                import redis.asyncio as aioredis
                asyncio.create_task(self._redis_client.aclose())
            except Exception:
                pass


# 全局限流器实例
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """获取限流器实例"""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        # 从配置读取限流参数
        default_limit = getattr(settings, "rate_limit_per_minute", DEFAULT_RATE_LIMIT)
        _rate_limiter = RateLimiter(default_limit=default_limit, window_seconds=60)
    return _rate_limiter
