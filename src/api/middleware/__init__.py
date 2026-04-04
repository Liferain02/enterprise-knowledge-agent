"""
统一异常处理中间件
替代分散在 Controller 层的 try/except
"""
import logging
import sys
import traceback
from typing import Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


# ==================== 异常分类 ====================

class AppException(Exception):
    """应用层基础异常"""
    def __init__(self, message: str, code: str = "APP_ERROR", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class ChatException(AppException):
    """聊天服务异常"""
    def __init__(self, message: str, code: str = "CHAT_ERROR", status_code: int = 500):
        super().__init__(message, code, status_code)


class KnowledgeException(AppException):
    """知识库服务异常"""
    def __init__(self, message: str, code: str = "KNOWLEDGE_ERROR", status_code: int = 500):
        super().__init__(message, code, status_code)


class ValidationException(AppException):
    """参数校验异常"""
    def __init__(self, message: str, code: str = "VALIDATION_ERROR", status_code: int = 422):
        super().__init__(message, code, status_code)


class AuthenticationException(AppException):
    """认证异常"""
    def __init__(self, message: str = "认证失败", code: str = "AUTH_ERROR", status_code: int = 401):
        super().__init__(message, code, status_code)


class ResourceNotFoundException(AppException):
    """资源不存在异常"""
    def __init__(self, resource: str = "资源", resource_id: str = ""):
        message = f"{resource} {resource_id} 不存在" if resource_id else f"{resource} 不存在"
        super().__init__(message, "NOT_FOUND", 404)


class RateLimitException(AppException):
    """限流异常"""
    def __init__(
        self,
        message: str = "请求过于频繁，请稍后重试",
        retry_after: float = 60.0,
    ):
        super().__init__(message, "RATE_LIMITED", 429)
        self.retry_after = retry_after


class PermissionDeniedException(AppException):
    """权限不足异常"""
    def __init__(self, resource: str = "", action: str = ""):
        detail = f"权限不足：需要 {resource}:{action}" if resource and action else "权限不足"
        super().__init__(detail, "PERMISSION_DENIED", 403)


# ==================== HTTP 异常映射 ====================

# 常见第三方库异常到 HTTP 状态的映射
_EXCEPTION_STATUS_MAP = {
    "ValueError": 400,
    "TypeError": 400,
    "KeyError": 400,
    "FileNotFoundError": 404,
    "PermissionError": 403,
    "TimeoutError": 504,
    "ConnectionError": 503,
    "JSONDecodeError": 400,
}


def _get_status_code_from_exception(exc: Exception) -> int:
    """根据异常类型推断 HTTP 状态码"""
    exc_type_name = type(exc).__name__

    # 优先检查已知异常映射
    if exc_type_name in _EXCEPTION_STATUS_MAP:
        return _EXCEPTION_STATUS_MAP[exc_type_name]

    # 检查异常消息关键词
    msg_lower = str(exc).lower()
    if "not found" in msg_lower or "不存在" in msg_lower:
        return 404
    if "auth" in msg_lower or "认证" in msg_lower or "token" in msg_lower:
        return 401
    if "permission" in msg_lower or "权限" in msg_lower or "forbidden" in msg_lower:
        return 403
    if "timeout" in msg_lower or "超时" in msg_lower:
        return 504
    if "connection" in msg_lower or "连接" in msg_lower:
        return 503

    # 默认 500
    return 500


# ==================== 统一错误响应格式 ====================

def make_error_response(
    message: str,
    code: str = "INTERNAL_ERROR",
    status_code: int = 500,
    detail: str = None,
    request_id: str = None,
) -> dict:
    """构建统一格式的错误响应"""
    response = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "status": status_code,
        }
    }
    if detail is not None:
        response["error"]["detail"] = detail
    if request_id is not None:
        response["error"]["request_id"] = request_id
    return response


# ==================== 统一异常处理中间件 ====================

class UnifiedExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """
    统一异常处理中间件

    功能：
    1. 捕获所有未处理的异常，统一返回 JSON 格式错误
    2. 根据异常类型自动推断 HTTP 状态码
    3. 记录完整堆栈日志
    4. 支持 request_id 追踪
    5. 对客户端隐藏内部错误细节（生产模式）

    不捕获的异常：
    - HTTPException（由 FastAPI 统一处理）
    """

    def __init__(self, app: ASGIApp, debug: bool = False):
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next):
        # 生成请求追踪 ID
        import uuid
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            return response

        except HTTPException:
            # HTTPException 由 FastAPI 的 Exception handlers 处理，重新抛出
            raise

        except RateLimitException as exc:
            # 限流异常
            logger.info(f"[{request_id}] RateLimitException: {exc.message}")
            return JSONResponse(
                status_code=exc.status_code,
                content=make_error_response(
                    message=exc.message,
                    code=exc.code,
                    status_code=exc.status_code,
                    request_id=request_id,
                ),
                headers={"Retry-After": str(int(exc.retry_after))} if exc.retry_after else {},
            )

        except AppException as exc:
            # 应用层已知异常
            logger.warning(
                f"[{request_id}] AppException: {exc.code} - {exc.message}"
            )
            return JSONResponse(
                status_code=exc.status_code,
                content=make_error_response(
                    message=exc.message,
                    code=exc.code,
                    status_code=exc.status_code,
                    request_id=request_id,
                )
            )

        except Exception as exc:
            # 所有未处理的异常
            exc_type = type(exc).__name__
            exc_msg = str(exc)

            # 记录完整堆栈
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            logger.error(
                f"[{request_id}] Unhandled Exception: {exc_type}: {exc_msg}\n{tb_str}"
            )

            # 根据异常类型推断状态码
            status_code = _get_status_code_from_exception(exc)

            # 生产模式：隐藏内部细节
            if self.debug:
                error_detail = exc_msg
                error_code = "DEBUG_ERROR"
            else:
                error_detail = tb_str if self.debug else None
                error_code = exc_type.upper()

            return JSONResponse(
                status_code=status_code,
                content=make_error_response(
                    message="服务器内部错误",
                    code=error_code,
                    status_code=status_code,
                    detail=error_detail,
                    request_id=request_id,
                )
            )


# ==================== FastAPI 异常处理器注册 ====================

def register_exception_handlers(app: FastAPI, debug: bool = False) -> None:
    """
    注册 FastAPI 全局异常处理器

    Args:
        app: FastAPI 应用实例
        debug: 是否调试模式（调试模式下返回详细错误）
    """
    # AppException 处理器
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        import uuid
        request_id = str(uuid.uuid4())[:8]
        logger.warning(f"[{request_id}] AppException: {exc.code} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=make_error_response(
                message=exc.message,
                code=exc.code,
                status_code=exc.status_code,
                request_id=request_id,
            )
        )

    # 通用 Exception 处理器（兜底）
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        import uuid
        request_id = str(uuid.uuid4())[:8]
        exc_type = type(exc).__name__
        exc_msg = str(exc)

        tb_str = "".join(traceback.format_exception(*sys.exc_info()))
        logger.error(
            f"[{request_id}] Unhandled Exception: {exc_type}: {exc_msg}\n{tb_str}"
        )

        status_code = _get_status_code_from_exception(exc)

        if debug:
            error_message = f"{exc_type}: {exc_msg}"
            error_detail = tb_str
        else:
            error_message = "服务器内部错误"
            error_detail = None

        return JSONResponse(
            status_code=status_code,
            content=make_error_response(
                message=error_message,
                code=exc_type.upper(),
                status_code=status_code,
                detail=error_detail,
                request_id=request_id,
            )
        )


# ==================== Rate Limiting 中间件 ====================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    API 限流中间件

    策略：
    1. 已登录用户按 username 限流
    2. 匿名用户按 IP 限流
    3. 聊天接口和入库接口单独限流
    """

    # 不需要限流的路径
    EXEMPT_PATHS = {
        "/health", "/health/live", "/health/ready",
        "/metrics", "/docs", "/openapi.json",
        "/redoc", "/favicon.ico",
    }

    def __init__(self, app: ASGIApp, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self._limiter = None

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # 豁免路径
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        from src.api.rate_limiter import get_rate_limiter
        from src.api.audit import get_audit_logger
        from config.settings import get_settings

        limiter = get_rate_limiter()
        settings = get_settings()

        # 提取用户标识
        username = "anonymous"
        try:
            from src.api.security import get_current_user
            from fastapi.security import OAuth2PasswordBearer
            from jose import jwt, JWTError

            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                try:
                    payload = jwt.decode(
                        token,
                        settings.jwt_secret_key,
                        algorithms=["HS256"]
                    )
                    username = payload.get("sub", "anonymous")
                except JWTError:
                    pass
        except Exception:
            pass

        # 提取客户端 IP
        ip = self._get_client_ip(request)

        # 确定限流端点和限制值
        if request.url.path.startswith("/api/v1/chat"):
            limit = getattr(settings, "rate_limit_chat_per_minute", 20)
        elif request.url.path.startswith("/api/v1/knowledge/ingest"):
            limit = getattr(settings, "rate_limit_ingest_per_minute", 10)
        elif username != "anonymous":
            limit = getattr(settings, "rate_limit_per_minute", 60)
        else:
            limit = getattr(settings, "rate_limit_anonymous_per_minute", 30)

        # 执行限流检查
        identifier = username if username != "anonymous" else ip
        endpoint = self._get_endpoint_key(request.url.path)
        result = await limiter.check(
            identifier=identifier,
            endpoint=endpoint,
            limit=limit,
        )

        if not result.allowed:
            audit = get_audit_logger()
            audit.log_rate_limited(username, request.url.path, request)
            headers = {
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(result.reset_at)),
                "Retry-After": str(int(result.retry_after)),
            }
            return JSONResponse(
                status_code=429,
                content=make_error_response(
                    message=f"请求过于频繁，请 {int(result.retry_after)} 秒后重试",
                    code="RATE_LIMITED",
                    status_code=429,
                    detail={
                        "limit": result.limit,
                        "retry_after": int(result.retry_after),
                        "reset_at": int(result.reset_at),
                    },
                    request_id=getattr(request.state, "request_id", ""),
                ),
                headers=headers,
            )

        # 添加限流头到响应
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))
        return response

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        if hasattr(request, "client") and request.client:
            return request.client.host
        return "unknown"

    def _get_endpoint_key(self, path: str) -> str:
        """将路径映射为限流端点 key"""
        if path.startswith("/api/v1/auth"):
            return "auth"
        if path.startswith("/api/v1/chat"):
            return "chat"
        if path.startswith("/api/v1/knowledge"):
            return "knowledge"
        if path.startswith("/api/v1/session"):
            return "session"
        return "default"


def register_rate_limit_middleware(app: FastAPI, enabled: bool = True) -> None:
    """注册限流中间件"""
    app.add_middleware(RateLimitMiddleware, enabled=enabled)
