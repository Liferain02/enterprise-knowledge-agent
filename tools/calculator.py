"""
计算器工具模块
"""
import ast
import operator
from typing import Any, Callable, Dict, Union
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CalculateInput(BaseModel):
    """计算输入模型"""
    expression: str = Field(description="数学表达式，例如: 2+2*3")


# 支持的操作
ops: Dict[Union[type, str], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


class SafeEval:
    """安全的数学表达式求值"""
    
    ALLOWED_NAMES = {
        'abs': abs,
        'min': min,
        'max': max,
        'pow': pow,
        'round': round,
        'len': len,
        'sum': sum,
        'sorted': sorted,
        'range': range,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
    }
    
    def __init__(self):
        self.variables = {}
    
    def visit(self, node):
        """访问 AST 节点"""
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        """通用访问"""
        raise ValueError(f"不支持的节点类型: {type(node).__name__}")
    
    def visit_Num(self, node):
        """处理数字"""
        return node.n
    
    def visit_Constant(self, node):
        """处理常量"""
        return node.value
    
    def visit_Name(self, node):
        """处理变量名"""
        if node.id in self.ALLOWED_NAMES:
            return self.ALLOWED_NAMES[node.id]
        if node.id in self.variables:
            return self.variables[node.id]
        raise ValueError(f"未定义的名称: {node.id}")
    
    def visit_BinOp(self, node):
        """处理二元运算"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        
        if op_type not in ops:
            raise ValueError(f"不支持的操作: {op_type.__name__}")
        
        return ops[op_type](left, right)
    
    def visit_UnaryOp(self, node):
        """处理一元运算"""
        operand = self.visit(node.operand)
        
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.Not):
            return not operand
        
        raise ValueError(f"不支持的一元操作: {type(node.op).__name__}")
    
    def visit_Expr(self, node):
        """处理表达式语句"""
        return self.visit(node.value)
    
    def eval(self, expr: str) -> Any:
        """求值"""
        # 清理表达式
        expr = expr.strip()
        # 解析为 AST
        tree = ast.parse(expr, mode='eval')
        # 求值
        return self.visit(tree.body)


class CalculatorTool(BaseTool):
    """安全计算器工具"""
    
    name: str = "calculate"
    description: str = "执行安全的数学计算。支持加减乘除、幂运算、函数调用等。"
    args_schema: type[BaseModel] = CalculateInput
    
    def _run(self, expression: str, **kwargs) -> str:
        """执行计算"""
        try:
            # 使用安全的求值器
            evaluator = SafeEval()
            result = evaluator.eval(expression)
            return f"计算结果: {expression} = {result}"
        except ValueError as e:
            return f"计算错误: {str(e)}"
        except Exception as e:
            return f"未知错误: {str(e)}"


def get_calculator_tool() -> BaseTool:
    """获取计算器工具"""
    return CalculatorTool()

