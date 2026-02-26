"""
Calculator Skill - 计算技能
"""
from agents.skills import get_skill_loader

# 便捷访问
def get_calculator_agent():
    """获取 Calculator Agent"""
    loader = get_skill_loader()
    return loader.create_agent("calculator")

__all__ = ["get_calculator_agent"]
