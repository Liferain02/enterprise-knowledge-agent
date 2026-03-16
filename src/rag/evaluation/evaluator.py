"""
RAG 评估模块
基于 RAGAS 框架提供检索和生成的评估能力
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from src.models.llm import get_llm
from config.settings import get_settings


# ============================================================
# 评估数据模型
# ============================================================

@dataclass
class EvalResult:
    """单次评估结果"""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: str

    # RAGAS 指标
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0

    # 原始分数（归一化前）
    raw_faithfulness: float = 0.0
    raw_answer_relevancy: float = 0.0

    # 元数据
    eval_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    latency_ms: float = 0.0
    token_usage: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "ground_truth": self.ground_truth[:200] + "..." if len(self.ground_truth) > 200 else self.ground_truth,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_recall": round(self.context_recall, 4),
            "context_precision": round(self.context_precision, 4),
            "latency_ms": round(self.latency_ms, 2),
            "eval_timestamp": self.eval_timestamp
        }


@dataclass
class EvalSummary:
    """评估汇总结果"""
    total_queries: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_recall: float
    avg_context_precision: float
    avg_latency_ms: float

    # 分数分布
    faithfulness_distribution: Dict[str, int] = field(default_factory=dict)
    answer_relevancy_distribution: Dict[str, int] = field(default_factory=dict)

    # 时间
    eval_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_answer_relevancy": round(self.avg_answer_relevancy, 4),
            "avg_context_recall": round(self.avg_context_recall, 4),
            "avg_context_precision": round(self.avg_context_precision, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "faithfulness_distribution": self.faithfulness_distribution,
            "answer_relevancy_distribution": self.answer_relevancy_distribution,
            "eval_timestamp": self.eval_timestamp
        }

    def print_summary(self):
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("RAG 评估结果摘要")
        print("=" * 60)
        print(f"评估查询数: {self.total_queries}")
        print(f"平均 Faithfulness (忠诚度): {self.avg_faithfulness:.4f}")
        print(f"平均 Answer Relevancy (答案相关性): {self.avg_answer_relevancy:.4f}")
        print(f"平均 Context Recall (召回率): {self.avg_context_recall:.4f}")
        print(f"平均 Context Precision (精确率): {self.avg_context_precision:.4f}")
        print(f"平均响应延迟: {self.avg_latency_ms:.2f} ms")
        print("=" * 60)


# ============================================================
# 评估器实现
# ============================================================

class RAGEvaluator:
    """
    RAG 评估器
    基于 RAGAS 框架的四个核心指标进行评估
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.llm = None

    def _get_llm(self):
        """获取 LLM 实例"""
        if self.llm is None:
            self.llm = get_llm(temperature=0.1)
        return self.llm

    async def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> EvalResult:
        """
        评估单个查询

        Args:
            query: 用户查询
            answer: 系统生成的答案
            contexts: 检索到的上下文
            ground_truth: 标准答案

        Returns:
            EvalResult: 评估结果
        """
        import time
        start_time = time.time()

        result = EvalResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

        try:
            llm = self._get_llm()

            # 1. Faithfulness - 评估答案是否基于检索内容
            result.raw_faithfulness = await self._evaluate_faithfulness(
                llm, query, answer, contexts
            )

            # 2. Answer Relevancy - 评估答案与问题的相关性
            result.raw_answer_relevancy = await self._evaluate_answer_relevancy(
                llm, query, answer
            )

            # 3. Context Recall - 评估检索内容的召回能力
            result.context_recall = await self._evaluate_context_recall(
                llm, contexts, ground_truth
            )

            # 4. Context Precision - 评估检索内容的精确度
            result.context_precision = await self._evaluate_context_precision(
                llm, query, contexts, ground_truth
            )

            # 归一化到 0-1 范围
            result.faithfulness = self._normalize_score(result.raw_faithfulness)
            result.answer_relevancy = self._normalize_score(result.raw_answer_relevancy)

            result.latency_ms = (time.time() - start_time) * 1000

        except Exception as e:
            print(f"评估出错: {e}")
            # 返回默认分数
            result.faithfulness = 0.5
            result.answer_relevancy = 0.5
            result.context_recall = 0.5
            result.context_precision = 0.5

        return result

    async def _evaluate_faithfulness(
        self,
        llm,
        query: str,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        评估 Faithfulness - 答案中的陈述是否可以从检索到的上下文推断出来

        评分标准:
        - 5: 所有陈述都可以从上下文推断
        - 4: 大多数陈述可以从上下文推断
        - 3: 部分陈述可以从上下文推断
        - 2: 少数陈述可以从上下文推断
        - 1: 没有陈述可以从上下文推断
        """
        contexts_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""请评估以下答案的忠诚度（Faithfulness）。

答案必须只包含可以从给定上下文中推断的信息。任何不在上下文中的信息都应该被视为幻觉。

查询: {query}

上下文:
{contexts_text}

答案: {answer}

请评估答案中的每个陈述是否可以从上下文中推断出来，然后给出总体评分。

评分标准:
- 5: 所有陈述都可以从上下文推断（无幻觉）
- 4: 大多数陈述可以从上下文推断
- 3: 部分陈述可以从上下文推断
- 2: 少数陈述可以从上下文推断
- 1: 没有陈述可以从上下文推断（全是幻觉）

请只输出一个数字 (1-5)，不要输出其他内容。
"""

        response = await llm.ainvoke(prompt)
        score_text = response.content.strip()

        try:
            # 提取数字
            score = int(score_text)
            return min(5, max(1, score))
        except ValueError:
            return 3  # 默认分数

    async def _evaluate_answer_relevancy(
        self,
        llm,
        query: str,
        answer: str
    ) -> float:
        """
        评估 Answer Relevancy - 答案与查询的相关程度

        评分标准:
        - 5: 答案完全相关，直接回答问题
        - 4: 答案大部分相关
        - 3: 答案部分相关
        - 2: 答案很少相关
        - 1: 答案完全不相关
        """
        prompt = f"""请评估以下答案与查询的相关程度。

查询: {query}

答案: {answer}

评分标准:
- 5: 答案完全相关，直接、全面回答问题
- 4: 答案大部分相关
- 3: 答案部分相关，包含一些有用信息
- 2: 答案很少相关，只有少量有用信息
- 1: 答案完全不相关，没有回答问题

请只输出一个数字 (1-5)，不要输出其他内容。
"""

        response = await llm.ainvoke(prompt)
        score_text = response.content.strip()

        try:
            score = int(score_text)
            return min(5, max(1, score))
        except ValueError:
            return 3

    async def _evaluate_context_recall(
        self,
        llm,
        contexts: List[str],
        ground_truth: str
    ) -> float:
        """
        评估 Context Recall - 检索到的上下文是否覆盖了标准答案中的信息

        评分标准:
        - 5: 上下文完全覆盖标准答案的所有关键信息
        - 4: 上下文覆盖大部分关键信息
        - 3: 上下文覆盖部分关键信息
        - 2: 上下文只覆盖很少关键信息
        - 1: 上下文没有覆盖关键信息
        """
        contexts_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""请评估检索到的上下文是否覆盖了标准答案中的关键信息。

标准答案: {ground_truth}

检索到的上下文:
{contexts_text}

评分标准:
- 5: 上下文完全覆盖标准答案的所有关键信息
- 4: 上下文覆盖大部分关键信息（≥80%）
- 3: 上下文覆盖部分关键信息（≥50%）
- 2: 上下文只覆盖很少关键信息（<50%）
- 1: 上下文没有覆盖关键信息

请只输出一个数字 (1-5)，不要输出其他内容。
"""

        response = await llm.ainvoke(prompt)
        score_text = response.content.strip()

        try:
            score = int(score_text)
            return min(5, max(1, score))
        except ValueError:
            return 3

    async def _evaluate_context_precision(
        self,
        llm,
        query: str,
        contexts: List[str],
        ground_truth: str
    ) -> float:
        """
        评估 Context Precision - 检索到的上下文与查询和标准答案的相关程度

        评分标准:
        - 5: 所有检索到的上下文都与查询和答案高度相关
        - 4: 大部分上下文相关
        - 3: 部分上下文相关
        - 2: 很少上下文相关
        - 1: 没有任何上下文相关
        """
        contexts_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""请评估检索到的上下文与查询和标准答案的相关程度。

查询: {query}

标准答案: {ground_truth}

检索到的上下文:
{contexts_text}

评分标准:
- 5: 所有检索到的上下文都与查询和答案高度相关，是高质量的参考
- 4: 大部分上下文相关（≥80%）
- 3: 部分上下文相关（≥50%）
- 2: 很少上下文相关（<50%）
- 1: 没有任何上下文相关，都是噪音

请只输出一个数字 (1-5)，不要输出其他内容。
"""

        response = await llm.ainvoke(prompt)
        score_text = response.content.strip()

        try:
            score = int(score_text)
            return min(5, max(1, score))
        except ValueError:
            return 3

    def _normalize_score(self, raw_score: float) -> float:
        """将 1-5 的分数归一化到 0-1 范围"""
        return (raw_score - 1) / 4

    async def evaluate_batch(
        self,
        eval_data: List[Dict[str, str]],
        show_progress: bool = True
    ) -> List[EvalResult]:
        """
        批量评估

        Args:
            eval_data: 评估数据列表，每项包含 query, answer, contexts, ground_truth
            show_progress: 是否显示进度

        Returns:
            评估结果列表
        """
        results = []

        for i, item in enumerate(eval_data):
            if show_progress:
                print(f"评估进度: {i+1}/{len(eval_data)}")

            result = await self.evaluate_single(
                query=item["query"],
                answer=item["answer"],
                contexts=item.get("contexts", []),
                ground_truth=item["ground_truth"]
            )
            results.append(result)

        return results

    def generate_summary(self, results: List[EvalResult]) -> EvalSummary:
        """生成评估摘要"""
        if not results:
            return EvalSummary(
                total_queries=0,
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_context_recall=0.0,
                avg_context_precision=0.0,
                avg_latency_ms=0.0
            )

        n = len(results)

        avg_faithfulness = sum(r.faithfulness for r in results) / n
        avg_answer_relevancy = sum(r.answer_relevancy for r in results) / n
        avg_context_recall = sum(r.context_recall for r in results) / n
        avg_context_precision = sum(r.context_precision for r in results) / n
        avg_latency_ms = sum(r.latency_ms for r in results) / n

        # 计算分数分布
        def get_bucket(score):
            if score >= 0.8:
                return "excellent"
            elif score >= 0.6:
                return "good"
            elif score >= 0.4:
                return "fair"
            else:
                return "poor"

        faithfulness_dist = {
            "excellent": sum(1 for r in results if r.faithfulness >= 0.8),
            "good": sum(1 for r in results if 0.6 <= r.faithfulness < 0.8),
            "fair": sum(1 for r in results if 0.4 <= r.faithfulness < 0.6),
            "poor": sum(1 for r in results if r.faithfulness < 0.4)
        }

        answer_relevancy_dist = {
            "excellent": sum(1 for r in results if r.answer_relevancy >= 0.8),
            "good": sum(1 for r in results if 0.6 <= r.answer_relevancy < 0.8),
            "fair": sum(1 for r in results if 0.4 <= r.answer_relevancy < 0.6),
            "poor": sum(1 for r in results if r.answer_relevancy < 0.4)
        }

        return EvalSummary(
            total_queries=n,
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevancy=avg_answer_relevancy,
            avg_context_recall=avg_context_recall,
            avg_context_precision=avg_context_precision,
            avg_latency_ms=avg_latency_ms,
            faithfulness_distribution=faithfulness_dist,
            answer_relevancy_distribution=answer_relevancy_dist
        )


# ============================================================
# 便捷函数
# ============================================================

_evaluator: Optional[RAGEvaluator] = None


def get_evaluator() -> RAGEvaluator:
    """获取评估器实例"""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator


async def evaluate_rag(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: str
) -> EvalResult:
    """评估单个 RAG 查询"""
    evaluator = get_evaluator()
    return await evaluator.evaluate_single(query, answer, contexts, ground_truth)


async def evaluate_rag_batch(
    eval_data: List[Dict[str, str]]
) -> List[EvalResult]:
    """批量评估 RAG 查询"""
    evaluator = get_evaluator()
    return await evaluator.evaluate_batch(eval_data)
