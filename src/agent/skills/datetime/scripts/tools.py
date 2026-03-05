"""
DateTime Skill Tools - 日期时间工具
"""
from datetime import datetime
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class DateTimeInput(BaseModel):
    """日期时间输入"""
    format: str = Field(
        default="%Y年%m月%d日 %H:%M:%S",
        description="可选的时间格式，如 '%Y年%m月%d日 %H:%M:%S' 或 '%Y-%m-%d %H:%M'"
    )


def get_current_datetime(format: str = "%Y年%m月%d日 %H:%M:%S") -> str:
    """
    获取当前日期和时间

    Args:
        format: 时间格式，默认为中文格式

    Returns:
        格式化后的当前日期时间字符串
    """
    now = datetime.now()
    return now.strftime(format)


def create_datetime_tool() -> BaseTool:
    """创建日期时间工具"""
    from langchain_core.tools import StructuredTool
    
    return StructuredTool.from_function(
        func=get_current_datetime,
        name="get_current_datetime",
        description="""获取当前日期和时间。

适用场景：
- 用户询问当前时间
- 用户询问今天是几号
- 需要获取系统时间时

输入：可选的时间格式字符串。
输出：格式化后的日期时间字符串。""",
        args_schema=DateTimeInput
    )


def get_datetime_tools() -> list:
    """获取日期时间工具列表"""
    return [create_datetime_tool()]


# 导出所有工具函数
__all__ = ["get_current_datetime", "create_datetime_tool", "get_datetime_tools"]
