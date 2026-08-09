"""
API Routers - 路由模块
"""
from .chat_controller import router as chat_router
from .knowledge_controller import router as knowledge_router
from .research_controller import router as research_router
from .auth_controller import router as auth_router
from .feedback_controller import router as feedback_router
from .vision_controller import router as vision_router
from ..routes.a2a_routes import _a2a_router

__all__ = [
    "chat_router",
    "knowledge_router",
    "research_router",
    "auth_router",
    "feedback_router",
    "vision_router",
    "_a2a_router",
]
