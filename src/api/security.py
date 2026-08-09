"""
认证与鉴权

 - /api/v1/auth/register 注册新用户
 - /api/v1/auth/login 登录获取 JWT
 - 其他接口通过 Authorization: Bearer <token> 访问

支持两种用户：
 1. 数据库用户（data/users.db，SQLite，通过 register 注册）
 2. 单管理员（config/.env 中的 admin_username/password，仅作为回退）
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

from config.settings import get_settings, Settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(
    *,
    subject: str,
    secret_key: str,
    expires_minutes: int,
    role: str = None,
    department: str = None,
    department_name: str = None,
    department_path: str = None,
    auth_source: str = "database",
) -> str:
    """
    创建 JWT Token，支持携带用户角色和部门信息。

    Args:
        subject: 用户名（会写入 JWT "sub" 字段）
        secret_key: 加密密钥
        expires_minutes: 过期时间（分钟）
        role: 角色（employee / hr / admin / it_support / manager）
        department: 部门 ID
        department_name: 部门名称
        department_path: 树形部门路径
        auth_source: 登录来源（database / config_admin）
    """
    now = _utcnow()
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    if role is not None:
        payload["role"] = role
    if department is not None:
        payload["department"] = department
    if department_name is not None:
        payload["department_name"] = department_name
    if department_path is not None:
        payload["department_path"] = department_path
    if auth_source is not None:
        payload["auth_source"] = auth_source
    return jwt.encode(payload, secret_key, algorithm="HS256")


def verify_password(plain_password: str, expected_password: str) -> bool:
    # 常量时间比较，避免时序侧信道
    return secrets.compare_digest(plain_password, expected_password)


def authenticate_user(
    *, username: str, password: str, settings: Settings
) -> Optional[Dict[str, Any]]:
    if not settings.auth_enabled:
        return {"username": username}

    if username != settings.admin_username:
        return None
    if not verify_password(password, settings.admin_password):
        return None
    return {"username": settings.admin_username}


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    从 JWT 解码当前用户信息。

    返回完整的用户上下文信息（username / role / department），
    用于 ACL 权限过滤和前端角色展示。数据库用户的授权信息在每次请求时刷新，
    JWT 中的 role/department 仅作为历史兼容快照，不作为授权依据。

    JWT payload 格式：
        {
            "sub": "username",
            "role": "employee",
            "department": "dept_id",
            "department_name": "技术部",
            "department_path": "/技术部/后端组",
            "auth_source": "database",
            "iat": ...,
            "exp": ...
        }
    """
    settings = get_settings()

    if not settings.auth_enabled:
        return {
            "username": "anonymous",
            "role": "employee",
            "department": "",
            "department_name": "",
            "department_path": "",
        }

    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        sub = payload.get("sub")
        if not sub:
            raise JWTError("Missing subject")

        # JWT 只证明身份与登录来源；当前授权始终从可信数据源刷新。
        from .security_user import get_user_by_username

        db_user = get_user_by_username(sub)
        auth_source = payload.get("auth_source")

        if auth_source == "config_admin":
            # 数据库同名账号优先，避免配置管理员令牌与数据库账号混淆。
            if sub != settings.admin_username or db_user is not None:
                raise JWTError("Invalid config admin subject")
            return {
                "username": settings.admin_username,
                "role": "admin",
                "department": "admin",
                "department_name": "管理员",
                "department_path": "/管理员",
            }

        # 无 auth_source 的旧令牌仅兼容仍存在的数据库用户；旧配置管理员需重新登录。
        if auth_source not in (None, "database") or db_user is None:
            raise JWTError("User no longer exists")

        return {
            "username": sub,
            "role": db_user.get("role", "viewer"),
            "department": db_user.get("department", ""),
            "department_name": db_user.get("department_name", ""),
            "department_path": db_user.get("department_path", ""),
        }

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
