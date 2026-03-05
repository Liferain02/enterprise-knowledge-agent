"""
认证 Router
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from ..schemas import LoginRequest, LoginResponse
from config.settings import get_settings
from ..security import authenticate_user, create_access_token


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    settings = get_settings()

    user = authenticate_user(username=req.username, password=req.password, settings=settings)
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
