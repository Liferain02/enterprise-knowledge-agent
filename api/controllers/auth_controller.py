"""
认证 Controller
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from config.settings import get_settings
from api.security import authenticate_user, create_access_token


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(description="用户名")
    password: str = Field(description="密码")


class LoginResponse(BaseModel):
    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="token 类型")


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

