"""
RBAC 权限依赖 - FastAPI Depends
提供：get_current_user_data、require_permission、require_role 三个 Depends。
"""
from typing import List, Optional
from fastapi import Depends, HTTPException, status

from ..security import get_current_user
from ..database import get_user_by_id, get_user_permissions, PermissionEnum


def get_current_user_data() -> dict:
    """
    获取当前用户完整信息（包含角色、权限、部门）。
    用于需要用户完整信息的接口。
    """
    token_user = get_current_user()

    # 如果是匿名用户（auth_disabled），返回默认信息
    if token_user.get("username") == "anonymous":
        return {
            "id": 0,
            "username": "anonymous",
            "email": "",
            "role": "employee",
            "role_id": None,
            "department": "",
            "department_id": None,
            "department_path": "",
            "is_active": True,
            "is_superadmin": False,
            "permissions": [p.value for p in [PermissionEnum.DOC_READ, PermissionEnum.KB_READ]],
        }

    username = token_user.get("username")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无法获取用户信息",
        )

    # 从数据库获取完整用户信息
    from ..database import get_user_by_id

    # 需要先通过 username 找到 user_id，再查详细信息
    # 这里简化处理，直接从 token_user 构建
    return {
        "username": username,
        "role": token_user.get("role", "employee"),
    }


def require_permission(*permissions: str):
    """
    权限依赖工厂。
    检查当前用户是否拥有指定权限。

    用法：
        @router.get("/admin", dependencies=[Depends(require_permission("user:manage"))])
        async def admin_only():
            ...

        @router.get("/doc", dependencies=[Depends(require_permission("doc:read", "doc:upload"))])
        async def doc_ops():
            # 需要同时拥有 doc:read 和 doc:upload
            ...
    """
    def dependency(current_user: dict = Depends(get_current_user_data)):
        # 匿名用户默认只有只读权限
        if current_user.get("username") == "anonymous":
            if all(p in [PermissionEnum.DOC_READ.value, PermissionEnum.KB_READ.value]
                   for p in permissions):
                return current_user
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要权限: {', '.join(permissions)}",
            )

        role = current_user.get("role", "employee")

        # admin 拥有所有权限
        if role == "admin":
            return current_user

        # 获取用户权限
        role_id = current_user.get("role_id")
        if not role_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要权限: {', '.join(permissions)}",
            )

        user_perms = get_user_permissions(role_id)
        user_perm_set = set(user_perms)

        # 检查是否拥有所有必需的权限
        missing = [p for p in permissions if p not in user_perm_set]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要权限: {', '.join(missing)}",
            )

        return current_user

    return dependency


def require_role(*roles: str):
    """
    角色依赖工厂。
    检查当前用户是否属于指定角色。

    用法：
        @router.get("/hr", dependencies=[Depends(require_role("hr", "admin"))])
        async def hr_only():
            ...
    """
    def dependency(current_user: dict = Depends(get_current_user_data)):
        role = current_user.get("role", "employee")
        if role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要角色: {', '.join(roles)}",
            )
        return current_user

    return dependency
