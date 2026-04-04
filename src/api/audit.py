"""
审计日志模块

记录所有敏感操作（登录/登出/上传/删除/配置变更）为结构化 JSON 日志，
支持文件输出和 OTEL 上报。

审计日志格式：
{
  "timestamp": "2026-04-04T12:00:00Z",
  "event": "document.upload",
  "user": "alice",
  "ip": "192.168.1.1",
  "resource": "knowledge/contracts",
  "action": "upload",
  "result": "success",
  "trace_id": "abc123",
  "detail": {...}
}

敏感操作事件类型：
- auth.login / auth.logout / auth.register / auth.failed
- document.upload / document.delete / document.update
- session.create / session.delete / session.clear
- user.create / user.delete / user.role.assign / user.role.remove
- config.change / system.backup
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

from config.settings import get_settings

logger = logging.getLogger(__name__)

# 审计事件类型
class AuditEvent(str, Enum):
    # 认证
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    REGISTER = "auth.register"
    AUTH_FAILED = "auth.failed"
    # 文档
    DOC_UPLOAD = "document.upload"
    DOC_DELETE = "document.delete"
    DOC_UPDATE = "document.update"
    DOC_VIEW = "document.view"
    # 会话
    SESSION_CREATE = "session.create"
    SESSION_DELETE = "session.delete"
    SESSION_CLEAR = "session.clear"
    # 用户管理
    USER_CREATE = "user.create"
    USER_DELETE = "user.delete"
    USER_ROLE_ASSIGN = "user.role.assign"
    USER_ROLE_REMOVE = "user.role.remove"
    # 系统
    CONFIG_CHANGE = "config.change"
    SYSTEM_BACKUP = "system.backup"
    SYSTEM_RESTORE = "system.restore"
    # API
    API_RATE_LIMITED = "api.rate_limited"
    API_ERROR = "api.error"


class AuditLogger:
    """
    审计日志记录器（线程安全）
    支持文件输出 + OTEL 上报
    """

    def __init__(self):
        self._settings = None
        self._file_handler: Optional[logging.FileHandler] = None
        self._audit_logger: Optional[logging.Logger] = None
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def settings(self):
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def _ensure_initialized(self):
        """延迟初始化（仅在第一次记录时）"""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                self._setup_logger()
                self._initialized = True
            except Exception as e:
                logger.warning(f"审计日志初始化失败: {e}")

    def _setup_logger(self):
        """配置审计日志记录器"""
        self._audit_logger = logging.getLogger("audit")
        self._audit_logger.setLevel(logging.INFO)
        self._audit_logger.propagate = False

        # 文件处理器
        log_dir = self.settings.project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "audit.jsonl"

        self._file_handler = logging.FileHandler(
            str(log_file),
            encoding="utf-8",
            mode="a"
        )
        self._file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        self._file_handler.setFormatter(formatter)
        self._audit_logger.addHandler(self._file_handler)

    def log(
        self,
        event: str,
        username: str,
        result: str = "success",
        resource: str = "",
        action: str = "",
        ip: str = "",
        request_id: str = "",
        detail: Optional[Dict[str, Any]] = None,
        trace_id: str = "",
        **extra,
    ):
        """
        记录审计日志

        Args:
            event: 事件类型（AuditEvent 值）
            username: 用户名
            result: 结果 success/failed/error
            resource: 资源路径
            action: 操作名称
            ip: 客户端 IP
            request_id: 请求追踪 ID
            trace_id: 分布式追踪 ID
            detail: 额外详情
            **extra: 其他字段
        """
        self._ensure_initialized()

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "user": username,
            "result": result,
            "resource": resource,
            "action": action,
            "ip": ip,
            "request_id": request_id,
            "trace_id": trace_id,
            "detail": detail or {},
            **extra,
        }

        # 过滤 None 值
        log_entry = {k: v for k, v in log_entry.items() if v is not None and v != ""}

        # 写文件
        if self._audit_logger:
            self._audit_logger.info(json.dumps(log_entry, ensure_ascii=False))

    def _get_client_ip(self, request) -> str:
        """从请求中提取客户端 IP"""
        if request is None:
            return ""
        # 支持代理头
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        if hasattr(request, "client") and request.client:
            return request.client.host
        return ""

    def log_request(
        self,
        event: str,
        request,
        username: str = "anonymous",
        result: str = "success",
        resource: str = "",
        action: str = "",
        detail: Optional[Dict[str, Any]] = None,
        **extra,
    ):
        """从 FastAPI 请求对象记录审计日志"""
        ip = self._get_client_ip(request)
        request_id = getattr(request.state, "request_id", "") if request else ""
        self.log(
            event=event,
            username=username,
            result=result,
            resource=resource,
            action=action,
            ip=ip,
            request_id=request_id,
            detail=detail,
            **extra,
        )

    # ── 快捷方法 ────────────────────────────────────────────────

    def log_login(self, username: str, request=None, result: str = "success", **kwargs):
        self.log_request(
            event=AuditEvent.LOGIN,
            request=request,
            username=username,
            result=result,
            resource="/api/v1/auth/login",
            action="login",
        )

    def log_logout(self, username: str, request=None, **kwargs):
        self.log_request(
            event=AuditEvent.LOGOUT,
            request=request,
            username=username,
            resource="/api/v1/auth/logout",
            action="logout",
        )

    def log_register(self, username: str, request=None, result: str = "success", **kwargs):
        self.log_request(
            event=AuditEvent.REGISTER,
            request=request,
            username=username,
            result=result,
            resource="/api/v1/auth/register",
            action="register",
        )

    def log_auth_failed(self, username: str, request=None, reason: str = "", **kwargs):
        self.log_request(
            event=AuditEvent.AUTH_FAILED,
            request=request,
            username=username,
            result="failed",
            resource="/api/v1/auth/login",
            action="login",
            detail={"reason": reason},
        )

    def log_doc_upload(
        self, username: str, filename: str, request=None, result: str = "success", **kwargs
    ):
        self.log_request(
            event=AuditEvent.DOC_UPLOAD,
            request=request,
            username=username,
            result=result,
            resource=f"knowledge/{filename}",
            action="upload",
        )

    def log_doc_delete(
        self, username: str, doc_id: str, request=None, result: str = "success", **kwargs
    ):
        self.log_request(
            event=AuditEvent.DOC_DELETE,
            request=request,
            username=username,
            result=result,
            resource=f"knowledge/{doc_id}",
            action="delete",
        )

    def log_session_delete(
        self, username: str, session_id: str, request=None, result: str = "success", **kwargs
    ):
        self.log_request(
            event=AuditEvent.SESSION_DELETE,
            request=request,
            username=username,
            result=result,
            resource=f"session/{session_id}",
            action="delete",
        )

    def log_rate_limited(
        self, username: str, endpoint: str, request=None, **kwargs
    ):
        self.log_request(
            event=AuditEvent.API_RATE_LIMITED,
            request=request,
            username=username,
            result="blocked",
            resource=endpoint,
            action="rate_limit",
        )

    def log_role_assign(
        self, admin: str, target_user: str, role: str, request=None, **kwargs
    ):
        self.log_request(
            event=AuditEvent.USER_ROLE_ASSIGN,
            request=request,
            username=admin,
            result="success",
            resource=f"user/{target_user}",
            action="role_assign",
            detail={"role": role, "target_user": target_user},
        )

    def log_role_remove(
        self, admin: str, target_user: str, role: str, request=None, **kwargs
    ):
        self.log_request(
            event=AuditEvent.USER_ROLE_REMOVE,
            request=request,
            username=admin,
            result="success",
            resource=f"user/{target_user}",
            action="role_remove",
            detail={"role": role, "target_user": target_user},
        )

    def close(self):
        """关闭审计日志（应用关闭时调用）"""
        if self._file_handler:
            self._file_handler.close()
            if self._audit_logger:
                self._audit_logger.removeHandler(self._file_handler)
            self._file_handler = None


# 全局实例
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """获取审计日志实例"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def close_audit_logger():
    """关闭审计日志（应用关闭时）"""
    global _audit_logger
    if _audit_logger:
        _audit_logger.close()
        _audit_logger = None
