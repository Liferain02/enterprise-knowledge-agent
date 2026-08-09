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
    - 新密码使用 Argon2 哈希，旧哈希在成功登录后自动升级
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
    auth_source = "database"

    # 优先验证数据库用户
    user = verify_user(username=req.username, password=req.password)

    # 回退：单管理员模式（auth_enabled=True 且使用 .env 中的 admin 配置）
    if not user and settings.auth_enabled:
        # 只有当没有数据库用户时，才回退到 admin 认证
        db_user = get_user_by_username(req.username)
        if db_user is None:
            # 没有数据库用户，回退到 admin
            if req.username == settings.admin_username and req.password == settings.admin_password:
                auth_source = "config_admin"
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
        "role": user.get("role", "student"),
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
        auth_source=auth_source,
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
        "role": current_user.get("role", "student"),
        "department": current_user.get("department", ""),
        "department_name": current_user.get("department_name", ""),
        "department_path": current_user.get("department_path", ""),
        "role_display_name": _ROLE_DISPLAY_NAMES.get(
            current_user.get("role", "student"), current_user.get("role", "student")
        ),
        "permission_hint": _ROLE_PERMISSION_HINTS.get(
            current_user.get("role", "student"), ""
        ),
    }


# 角色展示名映射（供前端显示）
_ROLE_DISPLAY_NAMES = {
    "admin": "管理员",
    "pi": "导师/PI",
    "teacher": "教师",
    "lab_admin": "实验室管理员",
    "senior_student": "高年级成员",
    "student": "研究生",
    "assistant": "助研/本科生",
    "editor": "资料维护者",
    "viewer": "普通成员",
    "manager": "项目负责人",
    "hr": "实验室管理员",
    "it_support": "平台支持",
    "employee": "研究组成员",
}


# 角色权限提示（供前端展示）
_ROLE_PERMISSION_HINTS = {
    "admin": "您可管理全部实验室资料与用户。",
    "pi": "您可查看公共、项目组内和负责人可见资料。",
    "teacher": "您可查看公共与项目组内资料。",
    "lab_admin": "您可维护公共流程、FAQ 与资料入口。",
    "senior_student": "您可查看公共与项目组内资料，并维护部分项目资料。",
    "student": "您可查看实验室公共资料与新人导览内容。",
    "assistant": "您可查看公共资料与基础流程说明。",
    "editor": "您可维护实验室公共与项目资料。",
    "viewer": "您可查看实验室公共资料。",
    "manager": "您可查看公共与项目组内资料。",
    "hr": "您可维护公共流程资料。",
    "it_support": "您可维护环境配置和平台说明资料。",
    "employee": "您可查看实验室公共资料。",
}
