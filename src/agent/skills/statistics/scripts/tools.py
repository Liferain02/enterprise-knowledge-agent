"""实验指标的确定性统计工具。"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MetricsInput(BaseModel):
    """指标汇总输入。"""

    values: str = Field(
        description="数值列表，可写成 JSON（如 [0.81, 0.84]）或逗号分隔文本（如 0.81, 0.84）"
    )
    sample: bool = Field(
        default=False,
        description="是否按样本标准差计算；默认使用总体标准差",
    )


class CompareMetricsInput(BaseModel):
    """指标对比输入。"""

    baseline: float = Field(description="基线结果")
    candidate: float = Field(description="候选方案结果")
    higher_is_better: bool = Field(
        default=True,
        description="指标是否越大越好，例如 Recall 为 true，延迟为 false",
    )


def _parse_values(raw: str) -> list[float]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("values 不能为空")

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.replace(";", ",").split(",")]

    if not isinstance(parsed, list):
        parsed = [parsed]
    if not parsed or len(parsed) > 1000:
        raise ValueError("数值数量必须在 1 到 1000 之间")

    values: list[float] = []
    for item in parsed:
        try:
            value = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无法解析数值: {item}") from exc
        if not math.isfinite(value):
            raise ValueError("不支持 NaN 或无穷大")
        values.append(value)
    return values


def analyze_metrics(values: str, sample: bool = False) -> str:
    """汇总一组实验指标。"""
    try:
        numbers = _parse_values(values)
        if sample and len(numbers) < 2:
            raise ValueError("样本标准差至少需要 2 个数值")
        stdev = statistics.stdev(numbers) if sample else statistics.pstdev(numbers)
        result = {
            "数量": len(numbers),
            "均值": round(statistics.fmean(numbers), 6),
            "中位数": round(statistics.median(numbers), 6),
            "最小值": round(min(numbers), 6),
            "最大值": round(max(numbers), 6),
            "标准差": round(stdev, 6),
            "标准差口径": "样本" if sample else "总体",
        }
        return json.dumps(result, ensure_ascii=False)
    except ValueError as exc:
        return f"统计输入错误: {exc}"


def compare_metrics(
    baseline: float,
    candidate: float,
    higher_is_better: bool = True,
) -> str:
    """比较基线和候选指标，返回绝对变化、相对变化和是否改善。"""
    try:
        baseline = float(baseline)
        candidate = float(candidate)
        if not math.isfinite(baseline) or not math.isfinite(candidate):
            raise ValueError("不支持 NaN 或无穷大")
        change = candidate - baseline
        relative = None if baseline == 0 else change / abs(baseline)
        improved = change > 0 if higher_is_better else change < 0
        result = {
            "基线": round(baseline, 6),
            "候选": round(candidate, 6),
            "绝对变化": round(change, 6),
            "相对变化": None if relative is None else round(relative, 6),
            "指标方向": "越大越好" if higher_is_better else "越小越好",
            "是否改善": improved,
        }
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        return f"对比输入错误: {exc}"


def create_statistics_tools() -> list[BaseTool]:
    """创建科研统计技能的 LangChain 工具。"""
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            func=analyze_metrics,
            name="analyze_metrics",
            description="汇总用户提供的实验指标，返回数量、均值、中位数、极值和标准差。",
            args_schema=MetricsInput,
        ),
        StructuredTool.from_function(
            func=compare_metrics,
            name="compare_metrics",
            description="比较基线和候选指标，返回绝对变化、相对变化和是否改善；延迟等指标应设置 higher_is_better=false。",
            args_schema=CompareMetricsInput,
        ),
    ]


__all__ = ["analyze_metrics", "compare_metrics", "create_statistics_tools"]
