"""
Knowledge Skill - 知识检索技能
"""
from agents.skills import get_skill_loader

# 便捷访问
def get_knowledge_agent():
    """获取 Knowledge Agent"""
    loader = get_skill_loader()
    return loader.create_agent("knowledge")

__all__ = ["get_knowledge_agent"]
