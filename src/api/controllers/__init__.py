"""
API Routers - 路由模块
"""
from .chat_controller import router as chat_router
from .knowledge_controller import router as knowledge_router
from .auth_controller import router as auth_router

__all__ = [
    "chat_router",
    "knowledge_router",
    "auth_router",
]
