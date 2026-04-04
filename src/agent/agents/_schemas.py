"""
Agent 通信 Schema 模块
定义 Agent 间结构化数据传递的 Pydantic 模型
"""
from typing import List, Optional, Any, Literal
from pydantic import BaseModel, Field


class StepResult(BaseModel):
    """
    单个计划步骤的执行结果（结构化传递）。

    替代原来的自由文本字符串，实现机器可读的步骤间数据传递。
    例如：第1步检索的政策数字可以直接作为结构化数据传给第2步。
    """
    step_id: int = Field(description="步骤 ID，与 plan_steps 中的 step_id 对应")
    description: str = Field(description="步骤描述")
    agent: Literal["knowledge_agent", "operation_agent", "general_agent"] = Field(
        description="执行该步骤的 Agent 类型"
    )

    # 核心结果字段（替代纯字符串 result）
    result: str = Field(default="", description="步骤执行的文本结果")
    sources: Optional[str] = Field(default=None, description="知识来源（仅 knowledge_agent）")

    # 结构化数据（可选，供下游步骤使用）
    structured_data: Optional[dict] = Field(
        default=None,
        description=(
            "机器可读的结构化数据，供下游步骤使用。例："
            "{\"type\": \"numeric\", \"value\": 15, \"unit\": \"天\", \"label\": \"年假天数\"}"
        )
    )

    # 执行状态
    success: bool = Field(default=True, description="是否成功执行")
    error: Optional[str] = Field(default=None, description="错误信息（若失败）")
    confidence: Optional[float] = Field(
        default=None,
        description="执行置信度 0.0-1.0"
    )


class PlanExecutionResult(BaseModel):
    """
    完整计划执行结果。

    包含所有步骤的结果汇总，以及最终生成的答案。
    """
    plan_id: str = Field(
        default="",
        description="计划 ID（UUID），用于追踪和去重"
    )
    steps: List[StepResult] = Field(
        default_factory=list,
        description="所有步骤的执行结果"
    )
    final_answer: str = Field(
        default="",
        description="由 LLM 汇总各步骤结果生成的最终答案"
    )
    used_agent: Literal[
        "planner_parallel", "planner_sequential", "supervisor"
    ] = Field(
        default="planner_parallel",
        description="实际执行计划的 Agent 类型"
    )
    execution_mode: Literal["parallel", "sequential"] = Field(
        default="parallel",
        description="执行模式"
    )
    total_steps: int = Field(default=0, description="总步骤数")
    successful_steps: int = Field(default=0, description="成功步骤数")
    failed_steps: int = Field(default=0, description="失败步骤数")

    def get_step(self, step_id: int) -> Optional[StepResult]:
        """根据 step_id 获取步骤结果"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_structured_data(self, step_id: int) -> Optional[dict]:
        """便捷方法：获取指定步骤的结构化数据"""
        step = self.get_step(step_id)
        return step.structured_data if step else None

    def get_numeric_result(self, step_id: int) -> Optional[float]:
        """
        便捷方法：从结构化数据中提取数值结果。
        用于第1步检索 → 第2步计算的管道。
        """
        data = self.get_structured_data(step_id)
        if data and data.get("type") == "numeric":
            return data.get("value")
        return None


def serialize_step_result(result: dict) -> StepResult:
    """
    将 dict 格式的步骤结果转换为 StepResult。

    兼容旧格式（纯字符串 result）和新格式（包含 structured_data）。
    """
    return StepResult(
        step_id=result.get("step_id", 0),
        description=result.get("description", ""),
        agent=result.get("agent", "general_agent"),
        result=result.get("result", ""),
        sources=result.get("sources"),
        structured_data=result.get("structured_data"),
        success=result.get("success", True),
        error=result.get("error"),
        confidence=result.get("confidence"),
    )


def serialize_plan_results(results: List[dict]) -> List[StepResult]:
    """将 dict 列表转换为 StepResult 列表"""
    return [serialize_step_result(r) for r in results]


# ==================== 辅助函数 ====================

def _extract_numeric_data(text: str) -> Optional[dict]:
    """
    从文本结果中提取数值数据（用于下游步骤的结构化输入）。

    例如：
    - "根据公司规定，年假天数为 15 天" → {"type": "numeric", "value": 15, "unit": "天", "label": "年假天数"}
    - "计算结果为 1250.5 元" → {"type": "numeric", "value": 1250.5, "unit": "元", "label": "计算结果"}

    这使得第1步检索的结果可以机器可读地传给第2步计算。
    """
    import re

    patterns = [
        # "年假天数为 15 天" / "年假 = 15 天" / "共计 15 天"
        (re.compile(r"(?:为|=|共计|共|共可休)\s*(\d+(?:\.\d+)?)\s*(?:天|日)"), "天"),
        # "工资 1250.5 元" / "共计 1250.5 元"
        (re.compile(r"(?:工资|共计|共需|实发)\s*(\d+(?:\.\d+)?)\s*元"), "元"),
        # "共计 3000 元"（单独出现）
        (re.compile(r"共计\s*(\d+(?:\.\d+)?)\s*元"), "元"),
        # "绩效 0.8 倍" / "1.5 倍"
        (re.compile(r"(\d+(?:\.\d+)?)\s*倍"), "倍数"),
        # "提前 (\d+) 天通知"
        (re.compile(r"提前\s*(\d+)\s*(?:天|日)"), "天"),
    ]

    for pattern, unit in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = float(match.group(1))
                return {
                    "type": "numeric",
                    "value": value,
                    "unit": unit,
                    "raw_match": match.group(0),
                }
            except (ValueError, IndexError):
                continue

    return None
