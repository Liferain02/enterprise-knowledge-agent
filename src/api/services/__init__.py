"""服务层公共入口，组件按需加载。"""
from importlib import import_module


_EXPORTS = {
    "chat_service": ("src.api.services.chat_service", "chat_service"),
    "ChatService": ("src.api.services.chat_service", "ChatService"),
    "feedback_service": ("src.api.services.feedback_service", "feedback_service"),
    "FeedbackService": ("src.api.services.feedback_service", "FeedbackService"),
    "knowledge_service": ("src.api.services.knowledge_service", "knowledge_service"),
    "KnowledgeService": ("src.api.services.knowledge_service", "KnowledgeService"),
    "research_service": ("src.api.services.research_service", "research_service"),
    "ResearchService": ("src.api.services.research_service", "ResearchService"),
    "session_service": ("src.api.services.session_service", "session_service"),
    "SessionService": ("src.api.services.session_service", "SessionService"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
