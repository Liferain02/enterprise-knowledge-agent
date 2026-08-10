"""控制器必须绑定服务实例，不能因子模块导入顺序拿到模块对象。"""

from src.api.controllers import (
    chat_controller,
    feedback_controller,
    knowledge_controller,
    research_controller,
)
from src.api.services.chat_service import ChatService
from src.api.services.feedback_service import FeedbackService
from src.api.services.knowledge_service import KnowledgeService
from src.api.services.research_service import ResearchService


def test_controllers_bind_service_instances():
    assert isinstance(chat_controller.chat_service, ChatService)
    assert isinstance(feedback_controller.feedback_service, FeedbackService)
    assert isinstance(knowledge_controller.knowledge_service, KnowledgeService)
    assert isinstance(research_controller.research_service, ResearchService)
