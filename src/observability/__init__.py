"""
可观测性模块（简化版）
仅保留 @traced 装饰器，无额外依赖。
如需完整追踪/指标/OTEL 导出，请安装对应依赖并重新引入。
"""
from functools import wraps
from typing import Callable, Any


def traced(
    name: str,
    attrs_func: Callable = None,
) -> Callable:
    """
    空操作追踪装饰器。
    保留函数签名兼容，不记录任何数据。
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await func(*args, **kwargs)
        return wrapper
    return decorator
