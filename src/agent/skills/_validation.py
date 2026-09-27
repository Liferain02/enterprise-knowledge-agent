"""Bounded, deterministic inputs and JSON results for local research tools."""
import functools
import json
import math

MAX_TEXT = 40000


def bounded_text(text: str, limit: int = MAX_TEXT) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请提供非空文本")
    if len(text) > limit:
        raise ValueError(f"文本不能超过 {limit} 字符，请缩小输入范围")
    return text


def finite(value) -> float:
    if isinstance(value, bool):
        raise ValueError("布尔值不是实验数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("数值必须有限，不支持 NaN/Infinity")
    return number


def json_result(func):
    """Retain the tool signature and return explicit errors instead of guesses."""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return json.dumps({"ok": True, **func(*args, **kwargs)}, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError, OverflowError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
    return wrapped
