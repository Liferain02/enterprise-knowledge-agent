"""
并行执行模块
支持复杂任务中独立步骤的并行执行
"""
import asyncio
import re
from typing import List, Dict, Any, Tuple, Set
from langchain_core.messages import HumanMessage
from ._schemas import StepResult, serialize_step_result


class ParallelExecutor:
    """
    并行执行器
    分析任务依赖关系，将独立步骤并行执行
    """

    def __init__(self):
        pass

    def analyze_dependencies(
        self,
        steps: List[Dict[str, Any]]
    ) -> List[Set[int]]:
        """
        分析步骤依赖关系，返回可并行执行的批次

        Args:
            steps: 步骤列表，每个步骤包含 step_id 和 depends_on

        Returns:
            批次列表，每批包含可以并行执行的步骤索引

        示例:
            steps = [
                {"step_id": 1, "depends_on": []},
                {"step_id": 2, "depends_on": [1]},  # 依赖步骤1
                {"step_id": 3, "depends_on": [1]},  # 依赖步骤1，可以与步骤2并行
                {"step_id": 4, "depends_on": [2, 3]}  # 依赖步骤2和3
            ]
            返回: [{1}, {2, 3}, {4}]
        """
        if not steps:
            return []

        # 构建依赖图
        step_map = {step["step_id"]: i for i, step in enumerate(steps)}
        in_degree = {i: len(step.get("depends_on", [])) for i, step in enumerate(steps)}
        dependents = {i: [] for i in range(len(steps))}

        # 记录每个步骤依赖的其他步骤索引
        for i, step in enumerate(steps):
            for dep_id in step.get("depends_on", []):
                if dep_id in step_map:
                    dep_idx = step_map[dep_id]
                    dependents[dep_idx].append(i)

        # 拓扑排序，收集可以并行的批次
        batches = []
        processed = set()

        while len(processed) < len(steps):
            # 找出所有入度为0（没有未处理依赖）的步骤
            current_batch = []
            for i, degree in in_degree.items():
                if i not in processed and degree == 0:
                    current_batch.append(i)

            if not current_batch:
                # 有循环依赖，选择剩余步骤中任意一个
                remaining = set(range(len(steps))) - processed
                current_batch = list(remaining)[:1] if remaining else []

            if current_batch:
                batches.append(set(current_batch))
                processed.update(current_batch)

                # 更新依赖计数
                for idx in current_batch:
                    for dependent in dependents[idx]:
                        in_degree[dependent] -= 1

        return batches

    async def execute_step(
        self,
        step: Dict[str, Any],
        messages: List,
        session_id: str,
        summary: str = "",
        prior_results: List[StepResult] = None,
    ) -> StepResult:
        """
        执行单个步骤

        Args:
            step: 步骤定义
            messages: 消息列表
            session_id: 会话ID
            summary: 对话摘要

        Returns:
            步骤执行结果
        """
        from .knowledge import knowledge_agent_node
        from .operation import operation_agent_node
        from .general import general_agent_node

        agent_name = step.get("agent", "general_agent")
        sub_question = step["description"]

        # 复用预注入的上下文消息（execute_plan_node 已注入 Mem0 + 摘要）
        # 不再重复调用 inject_summary_to_messages，减少 token 开销
        sub_messages = messages + [HumanMessage(content=sub_question)]

        # 注入上游结构化结果（若有依赖步骤）
        # 这使得第1步检索的数值可以直接传给第2步计算
        if prior_results:
            enriched_description = _inject_structured_context(sub_question, prior_results)
            sub_messages = messages + [HumanMessage(content=enriched_description)]

        try:
            if agent_name == "knowledge_agent":
                result = await knowledge_agent_node({
                    "messages": sub_messages,
                    "session_id": session_id
                })
                return serialize_step_result({
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "agent": agent_name,
                    "result": result.get("final_answer", ""),
                    "sources": result.get("sources", ""),
                    "success": True,
                    # 尝试从答案中提取数值（用于下游步骤）
                    "structured_data": _extract_numeric_data(result.get("final_answer", "")),
                })

            elif agent_name == "operation_agent":
                result = await operation_agent_node({
                    "messages": sub_messages,
                    "session_id": session_id
                })
                return serialize_step_result({
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "agent": agent_name,
                    "result": result.get("final_answer", ""),
                    "success": True,
                    "structured_data": _extract_numeric_data(result.get("final_answer", "")),
                })

            elif agent_name == "general_agent":
                result = await general_agent_node({
                    "messages": sub_messages,
                    "session_id": session_id
                })
                return serialize_step_result({
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "agent": agent_name,
                    "result": result.get("final_answer", ""),
                    "success": True,
                })

            else:
                # 未知 Agent 类型，默认使用 general
                result = await general_agent_node({
                    "messages": sub_messages,
                    "session_id": session_id
                })
                return serialize_step_result({
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "agent": agent_name,
                    "result": result.get("final_answer", ""),
                    "success": True,
                })

        except Exception as e:
            return StepResult(
                step_id=step["step_id"],
                description=step["description"],
                agent=agent_name,
                result=f"步骤执行出错: {str(e)}",
                success=False,
                error=str(e),
            )

    async def execute_parallel(
        self,
        steps: List[Dict[str, Any]],
        messages: List,
        session_id: str,
        summary: str = ""
    ) -> List[StepResult]:
        """
        并行执行多个独立步骤

        Args:
            steps: 步骤列表
            messages: 消息列表
            session_id: 会话ID
            summary: 对话摘要

        Returns:
            所有步骤的执行结果
        """
        # 分析依赖关系
        batches = self.analyze_dependencies(steps)

        print(f"[Parallel Execute] 计划分 {len(batches)} 批执行")

        all_results = []

        # 按批次执行
        for batch_idx, batch in enumerate(batches):
            batch_steps = [steps[i] for i in batch]
            print(f"[Parallel Execute] 批次 {batch_idx + 1}: 执行 {len(batch_steps)} 个步骤")

            # 并行执行当前批次
            tasks = [
                self.execute_step(step, messages, session_id, summary, prior_results=completed_results)
                for step in batch_steps
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    error_result = StepResult(
                        step_id=-1,
                        description="并行执行出错",
                        agent="unknown",
                        result=f"并行执行出错: {str(result)}",
                        success=False,
                        error=str(result),
                    )
                    all_results.append(error_result)
                else:
                    all_results.append(result)

            # 打印批次结果
            for result in batch_results:
                if not isinstance(result, Exception):
                    print(f"[Parallel Execute] 步骤 {result.get('step_id')} 完成: {result.get('result', '')[:50]}...")

        return all_results

    async def execute_sequential(
        self,
        steps: List[Dict[str, Any]],
        messages: List,
        session_id: str,
        summary: str = ""
    ) -> List[StepResult]:
        """
        顺序执行步骤（保留原有逻辑）

        Args:
            steps: 步骤列表
            messages: 消息列表
            session_id: 会话ID
            summary: 对话摘要

        Returns:
            所有步骤的执行结果
        """
        results = []

        for step in steps:
            print(f"[Sequential Execute] 执行步骤 {step['step_id']}: {step['description']}")
            result = await self.execute_step(step, messages, session_id, summary)
            results.append(result)
            print(f"[Sequential Execute] 步骤 {step['step_id']} 完成")

        return results


# 全局实例
_parallel_executor: ParallelExecutor = None


def get_parallel_executor() -> ParallelExecutor:
    """获取并行执行器实例"""
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = ParallelExecutor()
    return _parallel_executor


def analyze_step_dependencies(steps: List[Dict[str, Any]]) -> List[Set[int]]:
    """分析步骤依赖关系的便捷函数"""
    executor = get_parallel_executor()
    return executor.analyze_dependencies(steps)


# ==================== 辅助函数 ====================

def _extract_numeric_data(text: str) -> dict:
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


def _inject_structured_context(
    description: str,
    prior_results: List[StepResult],
) -> str:
    """
    为步骤注入上游结构化结果上下文。

    当步骤指定了 depends_on 时，将上游步骤的结构化数据
    以提示词形式注入到当前步骤的描述中。

    例如：
    - 当前步骤: "计算年假工资"
    - 上游结果: StepResult(step_id=1, structured_data={"type": "numeric", "value": 15, "unit": "天"})
    - 输出: "计算年假工资。上年年假天数为 15 天，请据此计算工资。"
    """
    if not prior_results:
        return description

    context_parts = []
    for result in prior_results:
        if result.structured_data and result.structured_data.get("type") == "numeric":
            data = result.structured_data
            context_parts.append(
                f"【步骤 {result.step_id} 结果】{data.get('label', result.description)}"
                f" = {data.get('value')} {data.get('unit', '')}"
            )

    if not context_parts:
        return description

    context_intro = f"{description}\n\n【上游步骤结果参考】\n"
    return context_intro + "\n".join(context_parts)
