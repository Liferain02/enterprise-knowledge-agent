"""API 公共入口，避免导入局部模块时加载完整应用依赖树。"""
from importlib import import_module


_EXPORTS = {
    "chat_router": ("src.api.controllers", "chat_router"),
    "knowledge_router": ("src.api.controllers", "knowledge_router"),
    "research_router": ("src.api.controllers", "research_router"),
    "auth_router": ("src.api.controllers", "auth_router"),
    "feedback_router": ("src.api.controllers", "feedback_router"),
    "chat_service": ("src.api.services", "chat_service"),
    "knowledge_service": ("src.api.services", "knowledge_service"),
    "research_service": ("src.api.services", "research_service"),
    "session_service": ("src.api.services", "session_service"),
    "feedback_service": ("src.api.services", "feedback_service"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
