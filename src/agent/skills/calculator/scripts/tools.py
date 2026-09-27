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
        import ast
        import operator
        if not expression or len(expression) > 512:
            raise ValueError("表达式长度必须在 1–512 字符之间")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError("表达式过于复杂")
        functions = {name: getattr(math, name) for name in ("sqrt", "sin", "cos", "tan", "log", "log10")}
        functions.update(abs=abs, round=round, min=min, max=max)
        binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Mod: operator.mod}
        def checked(value):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e100:
                raise ValueError("数值超出有限计算范围")
            return value
        def power(left, right):
            if abs(right) > 100:
                raise ValueError("指数超出范围")
            return checked(pow(left, right))
        functions['pow'] = power
        def evaluate(node):
            if isinstance(node, ast.Constant):
                return checked(node.value)
            if isinstance(node, ast.Name) and node.id in ("pi", "e"):
                return getattr(math, node.id)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Pow):
                    return power(left, right)
                if type(node.op) in binary:
                    return checked(binary[type(node.op)](left, right))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                if node.keywords or not 1 <= len(node.args) <= 10:
                    raise ValueError("函数参数数量不合法")
                return checked(functions[node.func.id](*(evaluate(arg) for arg in node.args)))
            raise ValueError("仅支持数值、算术运算和白名单数学函数")
        result = evaluate(tree.body)
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


def get_calculator_tools():
    """Compatibility export used by the skill package."""
    return [create_calculator_tool()]


__all__ = ["calculator", "create_calculator_tool", "get_calculator_tools"]
