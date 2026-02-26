"""
Calculator Skill Tools - 计算工具
"""
import math
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class CalculatorInput(BaseModel):
    """计算器输入"""
    expression: str = Field(description="数学表达式，例如: 2+2*3 或 sqrt(16)")


def calculator(expression: str) -> str:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式
    
    Returns:
        计算结果
    """
    try:
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "pow": pow,
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
        }
        
        result = eval(expression, {"__builtins__": {}}, safe_globals)
        return f"计算结果: {expression} = {result}"
    
    except Exception as e:
        return f"计算错误: {str(e)}"


def create_calculator_tool() -> BaseTool:
    """创建计算器工具"""
    from langchain_core.tools import StructuredTool
    
    return StructuredTool.from_function(
        func=calculator,
        name="calculator",
        description="""执行数学计算。

适用场景：
- 费用计算、统计数据
- 数值运算、百分比计算
- 任何需要数学计算的问题

输入：数学表达式。
输出：计算结果。""",
        args_schema=CalculatorInput
    )


__all__ = ["calculator", "create_calculator_tool"]

