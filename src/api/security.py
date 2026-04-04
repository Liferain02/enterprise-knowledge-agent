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


def create_access_token(*, subject: str, secret_key: str, expires_minutes: int) -> str:
    now = _utcnow()
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
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
    settings = get_settings()

    if not settings.auth_enabled:
        return {"username": "anonymous"}

    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        sub = payload.get("sub")
        if not sub:
            raise JWTError("Missing subject")
        # 验证用户存在于数据库或 admin 配置中
        from .security_user import get_user_by_username
        db_user = get_user_by_username(sub)
        if db_user is not None:
            return {"username": sub}
        # 回退：只允许 admin 用户
        if sub == settings.admin_username:
            return {"username": sub}
        raise JWTError("Invalid subject")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

