"""
认证 Router - 支持用户注册 + JWT 登录
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from ..schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from config.settings import get_settings
from ..security import create_access_token
from ..security_user import register_user, verify_user, get_user_by_username


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest):
    """
    用户注册接口

    - 支持用户名 + 密码注册
    - 用户数据存储在 data/users.db（SQLite）
    - 密码使用 SHA-256 加盐哈希
    """
    success, message = register_user(username=req.username, password=req.password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return RegisterResponse(success=True, message=message)


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """
    登录接口 - 支持两种模式：
    1. 数据库用户（data/users.db）
    2. 单管理员（config/auth 中配置的用户名密码，仅当无数据库用户时回退）
    """
    settings = get_settings()

    # 优先验证数据库用户
    user = verify_user(username=req.username, password=req.password)

    # 回退：单管理员模式（auth_enabled=True 且使用 .env 中的 admin 配置）
    if not user and settings.auth_enabled:
        # 只有当没有数据库用户时，才回退到 admin 认证
        db_user = get_user_by_username(req.username)
        if db_user is None:
            # 没有数据库用户，回退到 admin
            if req.username == settings.admin_username and req.password == settings.admin_password:
                user = {"username": settings.admin_username}
            else:
                user = None

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    token = create_access_token(
        subject=user["username"],
        secret_key=settings.jwt_secret_key,
        expires_minutes=settings.jwt_expire_minutes,
    )
    return LoginResponse(access_token=token)
