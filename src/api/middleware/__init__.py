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
