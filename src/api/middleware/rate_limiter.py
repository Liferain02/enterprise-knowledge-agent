"""
Rate Limiter 中间件 - 基于内存滑动窗口算法
支持：按 IP 限速、按用户 ID 限速、按端点限速。

算法：滑动窗口（Sliding Window），精度高，内存占用 O(window_size)。

依赖：pip install slowapi
用法：
    from src.api.middleware.rate_limiter import rate_limiter, limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/items", dependencies=[Depends(rate_limiter.limit("5/minute"))])
    async def read_items():
        ...
"""
import time
import threading
from collections import defaultdict
from typing import Optional, Callable, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class SlidingWindowRateLimiter:
    """
    滑动窗口限流器

    支持多维度 key（如 IP、用户ID、端点）的并发限流。
    使用链表记录每个请求的时间戳，自动清理过期记录。
    """

    def __init__(self, default_limit: int = 60, window_seconds: int = 60):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self._store: Dict[str, list] = defaultdict(list)
        self._lock = threading.RLock()

    def _cleanup(self, key: str):
        """清理超过窗口的记录"""
        cutoff = time.time() - self.window_seconds
        self._store[key] = [t for t in self._store[key] if t > cutoff]

    def is_allowed(self, key: str, limit: Optional[int] = None) -> tuple[bool, int, int]:
        """
        检查是否允许请求

        Returns:
            (is_allowed, remaining, reset_seconds)
            - is_allowed: True 表示允许，False 表示被限流
            - remaining: 剩余请求数
            - reset_seconds: 距离窗口重置的秒数
        """
        limit = limit or self.default_limit
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            # 清理过期记录
            self._store[key] = [t for t in self._store[key] if t > cutoff]

            count = len(self._store[key])
            if count >= limit:
                # 计算距离最旧记录过期的秒数
                oldest = self._store[key][0] if self._store[key] else now
                reset_seconds = int(oldest + self.window_seconds - now)
                return False, 0, max(1, reset_seconds)

            # 记录当前请求
            self._store[key].append(now)
            remaining = limit - count - 1

            # 计算 reset 时间
            oldest = self._store[key][0] if self._store[key] else now
            reset_seconds = int(oldest + self.window_seconds - now)

            return True, remaining, max(1, reset_seconds)

    def get_usage(self, key: str) -> int:
        """获取当前窗口内的请求数"""
        cutoff = time.time() - self.window_seconds
        with self._lock:
            return len([t for t in self._store[key] if t > cutoff])


# 全局限流器实例
_limiter: Optional[SlidingWindowRateLimiter] = None


def get_limiter() -> SlidingWindowRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = SlidingWindowRateLimiter(
            default_limit=60,
            window_seconds=60,
        )
    return _limiter


def reset_limiter():
    """重置限流器（用于测试）"""
    global _limiter
    _limiter = None


# ==================== FastAPI 集成 ====================

async def _rate_limit_exceeded_handler(request: Request, exc) -> JSONResponse:
    """Rate limit Exceeded 异常处理器"""
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "请求过于频繁，请稍后再试",
                "detail": str(exc),
            }
        },
        headers={
            "Retry-After": str(getattr(exc, "retry_after", 60)),
        }
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    全局限流中间件

    对所有请求进行限速，保护后端服务不被滥用。
    默认规则：
    - 全局限流：60 次/分钟（所有请求）
    - 写操作：10 次/分钟（POST/PUT/DELETE）
    - 认证接口：5 次/分钟（/auth/*）
    """

    def __init__(
        self,
        app: ASGIApp,
        global_limit: int = 60,
        write_limit: int = 10,
        auth_limit: int = 5,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.global_limiter = SlidingWindowRateLimiter(global_limit, window_seconds)
        self.write_limiter = SlidingWindowRateLimiter(write_limit, window_seconds)
        self.auth_limiter = SlidingWindowRateLimiter(auth_limit, window_seconds)

    def _get_client_ip(self, request: Request) -> str:
        """获取客户端 IP（支持代理）"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        if request.client:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        path = request.url.path
        method = request.method

        # 判断限流器
        if path.startswith("/api/v1/auth"):
            limiter = self.auth_limiter
            limit_key = f"auth:{client_ip}"
        elif method in ("POST", "PUT", "PATCH", "DELETE"):
            limiter = self.write_limiter
            limit_key = f"write:{client_ip}"
        else:
            limiter = self.global_limiter
            limit_key = f"global:{client_ip}"

        allowed, remaining, reset_seconds = limiter.is_allowed(limit_key)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "请求过于频繁，请稍后再试",
                    }
                },
                headers={
                    "Retry-After": str(reset_seconds),
                    "X-RateLimit-Limit": str(limiter.default_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_seconds),
                }
            )

        response = await call_next(request)

        # 在响应头中注入限流信息
        response.headers["X-RateLimit-Limit"] = str(limiter.default_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_seconds)

        return response
