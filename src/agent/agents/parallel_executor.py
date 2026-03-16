"""
并行执行模块
支持复杂任务中独立步骤的并行执行
"""
import asyncio
from typing import List, Dict, Any, Tuple, Set
from langchain_core.messages import HumanMessage


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
        summary: str = ""
    ) -> Dict[str, Any]:
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

        # 注入摘要到消息
        from ._utils import inject_summary_to_messages
        sub_messages = messages + [HumanMessage(content=sub_question)]
        if summary:
            sub_messages = inject_summary_to_messages(sub_messages, summary)

        try:
            if agent_name == "knowledge_agent":
                result = await knowledge_agent_node({
                    "messages": sub_messages,
                    "session_id": session_id
                })
                return {
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "agent": agent_name,
                    "result": result.get("final_answer", ""),
                    "sources": result.get("sources", ""),
                    "success": True
                }

            elif agent_name == "operation_agent":
                result = await operation_agent_node({
                    "messages": sub_messages,
                    "session_id": session_id
                })
                return {
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "agent": agent_name,
                    "result": result.get("final_answer", ""),
                    "success": True
                }

            elif agent_name == "general_agent":
                result = await general_agent_node({
                    "messages": sub_messages,
                    "session_id": session_id
                })
                return {
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "agent": agent_name,
                    "result": result.get("final_answer", ""),
                    "success": True
                }

            else:
                # 未知 Agent 类型，默认使用 general
                result = await general_agent_node({
                    "messages": sub_messages,
                    "session_id": session_id
                })
                return {
                    "step_id": step["step_id"],
                    "description": step["description"],
                    "agent": agent_name,
                    "result": result.get("final_answer", ""),
                    "success": True
                }

        except Exception as e:
            return {
                "step_id": step["step_id"],
                "description": step["description"],
                "agent": agent_name,
                "result": f"步骤执行出错: {str(e)}",
                "success": False,
                "error": str(e)
            }

    async def execute_parallel(
        self,
        steps: List[Dict[str, Any]],
        messages: List,
        session_id: str,
        summary: str = ""
    ) -> List[Dict[str, Any]]:
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
                self.execute_step(step, messages, session_id, summary)
                for step in batch_steps
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    all_results.append({
                        "result": f"并行执行出错: {str(result)}",
                        "success": False,
                        "error": str(result)
                    })
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
    ) -> List[Dict[str, Any]]:
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


async def execute_steps_parallel(
    steps: List[Dict[str, Any]],
    messages: List,
    session_id: str,
    summary: str = ""
) -> List[Dict[str, Any]]:
    """并行执行步骤的便捷函数"""
    executor = get_parallel_executor()
    return await executor.execute_parallel(steps, messages, session_id, summary)


async def execute_steps_sequential(
    steps: List[Dict[str, Any]],
    messages: List,
    session_id: str,
    summary: str = ""
) -> List[Dict[str, Any]]:
    """顺序执行步骤的便捷函数"""
    executor = get_parallel_executor()
    return await executor.execute_sequential(steps, messages, session_id, summary)


def analyze_step_dependencies(steps: List[Dict[str, Any]]) -> List[Set[int]]:
    """分析步骤依赖关系的便捷函数"""
    executor = get_parallel_executor()
    return executor.analyze_dependencies(steps)
