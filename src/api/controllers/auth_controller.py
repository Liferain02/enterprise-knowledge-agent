"""
认证 Router - 支持用户注册 + JWT 登录
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends

from ..schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from config.settings import get_settings
from ..security import create_access_token
from ..security import get_current_user
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

    返回：
        - access_token: JWT 令牌
        - user_info: 当前用户的完整信息（username, role, department 等）
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
                user = {
                    "username": settings.admin_username,
                    "role": "admin",
                    "department": "admin",
                    "department_name": "管理员",
                    "department_path": "/管理员",
                }
            else:
                user = None

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    # 构建用户信息（用于前端展示）
    user_info = {
        "username": user["username"],
        "role": user.get("role", "employee"),
        "department": user.get("department", ""),
        "department_name": user.get("department_name", ""),
        "department_path": user.get("department_path", ""),
    }

    token = create_access_token(
        subject=user["username"],
        secret_key=settings.jwt_secret_key,
        expires_minutes=settings.jwt_expire_minutes,
        role=user_info["role"],
        department=user_info["department"],
        department_name=user_info["department_name"],
        department_path=user_info["department_path"],
    )
    return LoginResponse(access_token=token, user_info=user_info)


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    获取当前登录用户的详细信息。

    返回与登录时相同的 user_info 结构，供前端随时查询当前用户角色。
    """
    return {
        "username": current_user.get("username", "anonymous"),
        "role": current_user.get("role", "employee"),
        "department": current_user.get("department", ""),
        "department_name": current_user.get("department_name", ""),
        "department_path": current_user.get("department_path", ""),
        "role_display_name": _ROLE_DISPLAY_NAMES.get(
            current_user.get("role", "employee"), current_user.get("role", "employee")
        ),
        "permission_hint": _ROLE_PERMISSION_HINTS.get(
            current_user.get("role", "employee"), ""
        ),
    }


# 角色展示名映射（供前端显示）
_ROLE_DISPLAY_NAMES = {
    "admin": "管理员",
    "manager": "部门经理",
    "hr": "HR专员",
    "it_support": "IT支持",
    "employee": "普通员工",
}


# 角色权限提示（供前端展示）
_ROLE_PERMISSION_HINTS = {
    "admin": "您拥有系统全部权限，可以管理所有文档和用户。",
    "manager": "您可以访问机密级文档，管理本部门知识内容。",
    "hr": "您可以访问机密级 HR 文档，管理人事相关制度。",
    "it_support": "您可以访问机密级 IT 文档，处理技术支持。",
    "employee": "您可以访问内部公开文档，检索企业知识库。",
}
