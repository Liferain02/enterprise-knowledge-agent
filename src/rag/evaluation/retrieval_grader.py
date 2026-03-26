"""
Corrective RAG - 检索结果评估与自我纠错

参考 2026 年 Agentic RAG 最佳实践：
- 在检索后、生成前评估文档与查询的相关性
- 高置信 → 继续生成
- 低置信 → 触发查询重写 + 重新检索
- 无结果 → 返回"知识库无相关信息"

这样做的好处：
- 对比类问题（"对比 A 和 B"）中，如果只检索到 A，评估器可触发补充检索 B
- 防止低质量检索结果污染生成
- 与事后评估（RAGEvaluator）形成互补

核心类：
- RetrievalGrader: 评估单篇文档与查询的相关性（LLM-as-judge）
- CorrectiveRAGPipeline: 完整管线：检索 → 评估 → 决策（使用/重写/放弃）
- GradeResult: 评估结果数据结构
"""
import asyncio
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.models.llm import get_llm
from config.settings import get_settings


logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

class GradeLevel(str, Enum):
    """检索结果评级"""
    HIGH = "high"      # 文档与查询高度相关，继续生成
    LOW = "low"        # 文档相关性低，需要重写查询并重新检索
    NO_RESULTS = "no_results"  # 知识库无相关信息


@dataclass
class DocumentGrade:
    """单篇文档的评估结果"""
    doc: Document
    relevance_score: float   # 0.0 ~ 1.0，LLM 评估的原始分数（归一化后）
    raw_score: float         # 1 ~ 5，LLM 输出的原始分数
    reasoning: str           # LLM 给出的评估理由
    grade: GradeLevel        # 评级结果


@dataclass
class GradeResult:
    """一组文档的整体评估结果"""
    query: str
    grades: List[DocumentGrade]

    # 汇总统计
    total_docs: int = field(init=False)
    high_count: int = field(init=False)
    low_count: int = field(init=False)
    avg_score: float = field(init=False)

    # 决策
    decision: GradeLevel = field(init=False)
    decision_reason: str = field(init=False)

    # 性能
    latency_ms: float = 0.0

    def __post_init__(self):
        self.total_docs = len(self.grades)
        self.high_count = sum(1 for g in self.grades if g.grade == GradeLevel.HIGH)
        self.low_count = sum(1 for g in self.grades if g.grade == GradeLevel.LOW)

        scores = [g.relevance_score for g in self.grades]
        self.avg_score = sum(scores) / len(scores) if scores else 0.0

        self._make_decision()

    def _make_decision(self):
        """根据评估结果做出决策"""
        if self.total_docs == 0:
            self.decision = GradeLevel.NO_RESULTS
            self.decision_reason = "检索结果为空"
            return

        high_ratio = self.high_count / self.total_docs
        low_ratio = self.low_count / self.total_docs

        # 决策逻辑：
        # - 没有任何高相关文档，且大多数是低相关 → NO_RESULTS
        # - 高相关文档 ≥ 1 个，且平均分 ≥ 阈值 → HIGH
        # - 其他情况 → LOW（需要 query rewrite）
        settings = get_settings()
        grade_threshold = getattr(settings, 'crag_grade_threshold', 0.5)
        min_high_ratio = getattr(settings, 'crag_min_high_ratio', 0.3)

        if self.high_count == 0 and low_ratio >= 0.7:
            self.decision = GradeLevel.NO_RESULTS
            self.decision_reason = (
                f"所有 {self.total_docs} 篇文档相关性均较低（<{grade_threshold}），"
                f"知识库中可能不存在相关信息"
            )
        elif high_ratio >= min_high_ratio and self.avg_score >= grade_threshold:
            self.decision = GradeLevel.HIGH
            self.decision_reason = (
                f"高相关文档 {self.high_count}/{self.total_docs} 篇，"
                f"平均相关分 {self.avg_score:.2f} ≥ {grade_threshold}，可直接使用"
            )
        else:
            self.decision = GradeLevel.LOW
            self.decision_reason = (
                f"高相关文档 {self.high_count}/{self.total_docs} 篇，"
                f"平均相关分 {self.avg_score:.2f} < {grade_threshold}，"
                f"建议重写查询后重新检索"
            )

    def filter_high_grade(self) -> List[Document]:
        """只返回高相关文档"""
        return [g.doc for g in self.grades if g.grade == GradeLevel.HIGH]

    def filter_above_threshold(self, threshold: float) -> List[Document]:
        """返回相关分 >= threshold 的文档"""
        return [g.doc for g in self.grades if g.relevance_score >= threshold]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "decision": self.decision.value,
            "decision_reason": self.decision_reason,
            "total_docs": self.total_docs,
            "high_count": self.high_count,
            "low_count": self.low_count,
            "avg_score": round(self.avg_score, 3),
            "latency_ms": round(self.latency_ms, 2),
            "grades": [
                {
                    "relevance_score": round(g.relevance_score, 3),
                    "raw_score": g.raw_score,
                    "grade": g.grade.value,
                    "reasoning": g.reasoning[:100],
                    "source": g.doc.metadata.get("source", "unknown") if g.doc.metadata else "unknown",
                }
                for g in self.grades
            ]
        }


# ============================================================
# 检索评估器 - Corrective RAG 核心
# ============================================================

class RetrievalGrader:
    """
    Corrective RAG - 检索结果评估器

    职责：
    1. 评估每篇检索到的文档与用户查询的相关性（LLM-as-judge）
    2. 根据整体评估结果决定下一步行动：
       - HIGH → 返回高相关文档，继续生成
       - LOW  → 返回低分原因 + 建议的查询改写
       - NO_RESULTS → 返回空结果，告知用户知识库无相关信息

    评估策略：
    - 逐篇评估（而非批量），确保每篇都有独立评分）
    - 支持并行评估（asyncio.gather），减少延迟
    - 低于 grade_threshold 的文档直接标记为 LOW
    """

    def __init__(
        self,
        grade_threshold: float = 0.5,
        llm_temperature: float = 0.1,
        max_concurrent: int = 5,
    ):
        """
        初始化评估器

        Args:
            grade_threshold: 相关性阈值，>= 此值视为 HIGH（0.0 ~ 1.0）
            llm_temperature: LLM 评估时的温度
            max_concurrent: 最大并发评估数
        """
        settings = get_settings()
        self.grade_threshold = grade_threshold or getattr(
            settings, 'crag_grade_threshold', 0.5
        )
        self.llm_temperature = llm_temperature
        self.max_concurrent = max_concurrent
        self._llm = None

    @property
    def llm(self):
        """延迟加载 LLM"""
        if self._llm is None:
            self._llm = get_llm(temperature=self.llm_temperature)
        return self._llm

    # ========================================================
    # 核心评估 API
    # ========================================================

    async def grade_single(self, query: str, doc: Document) -> DocumentGrade:
        """
        评估单篇文档与查询的相关性

        Args:
            query: 用户查询
            doc: 待评估的文档

        Returns:
            DocumentGrade: 包含评分、理由和评级
        """
        start = time.time()

        prompt = self._build_grading_prompt(query, doc.page_content)

        try:
            response = await self.llm.ainvoke(prompt)
            raw_text = response.content.strip()

            score, reasoning = self._parse_grade_response(raw_text)
            normalized = self._normalize_score(score)

            grade_level = (
                GradeLevel.HIGH if normalized >= self.grade_threshold
                else GradeLevel.LOW
            )

            latency = (time.time() - start) * 1000

            return DocumentGrade(
                doc=doc,
                relevance_score=normalized,
                raw_score=float(score),
                reasoning=reasoning,
                grade=grade_level,
            )

        except Exception as e:
            logger.warning(f"评估文档失败: {e}")
            # 评估失败时保守处理，标记为 LOW
            return DocumentGrade(
                doc=doc,
                relevance_score=0.0,
                raw_score=1.0,
                reasoning=f"评估失败: {str(e)}",
                grade=GradeLevel.LOW,
            )

    async def grade_batch(
        self,
        query: str,
        documents: List[Document],
        show_progress: bool = False,
    ) -> List[DocumentGrade]:
        """
        并行评估多篇文档

        Args:
            query: 用户查询
            documents: 待评估的文档列表
            show_progress: 是否打印进度

        Returns:
            按原始顺序排列的评估结果列表
        """
        if not documents:
            return []

        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def grade_with_limit(idx: int, doc: Document) -> tuple[int, DocumentGrade]:
            async with semaphore:
                if show_progress and idx % 5 == 0:
                    print(f"  评估进度: {idx + 1}/{len(documents)}")
                grade = await self.grade_single(query, doc)
                return idx, grade

        tasks = [
            grade_with_limit(i, doc)
            for i, doc in enumerate(documents)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 按原始顺序排列
        graded = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"评估任务异常: {result}")
                # 用占位符保持顺序
                graded.append((len(graded), DocumentGrade(
                    doc=documents[len(graded)] if len(graded) < len(documents) else documents[0],
                    relevance_score=0.0,
                    raw_score=1.0,
                    reasoning=f"评估异常: {str(result)}",
                    grade=GradeLevel.LOW,
                )))
            else:
                graded.append(result)

        graded.sort(key=lambda x: x[0])
        return [g for _, g in graded]

    async def grade_retrieval(
        self,
        query: str,
        documents: List[Document],
    ) -> GradeResult:
        """
        完整评估流程：评估 + 决策

        这是 Corrective RAG 的主要入口方法。

        Args:
            query: 用户查询
            documents: 检索到的文档列表

        Returns:
            GradeResult: 包含每篇评估结果和整体决策
        """
        start = time.time()

        if not documents:
            return GradeResult(
                query=query,
                grades=[],
                decision=GradeLevel.NO_RESULTS,
                decision_reason="检索结果为空",
                latency_ms=0.0,
            )

        # 并行评估所有文档
        grades = await self.grade_batch(query, documents)

        # 构建结果
        result = GradeResult(query=query, grades=grades)
        result.latency_ms = (time.time() - start) * 1000

        logger.info(
            f"[CRAG] query='{query[:30]}...' -> decision={result.decision.value}, "
            f"high={result.high_count}/{result.total_docs}, "
            f"avg={result.avg_score:.3f}, latency={result.latency_ms:.0f}ms"
        )

        return result

    # ========================================================
    # 查询改写
    # ========================================================

    async def rewrite_query(
        self,
        original_query: str,
        grade_result: GradeResult,
    ) -> str:
        """
        根据评估结果重写查询

        当文档相关性整体偏低时（如对比类问题只检索到一半），
        使用 LLM 分析原因并生成更好的查询。

        Args:
            original_query: 原始查询
            grade_result: 评估结果

        Returns:
            改写后的查询
        """
        if not grade_result.grades:
            return original_query

        # 构建低分文档的摘要
        low_docs = [g for g in grade_result.grades if g.grade == GradeLevel.LOW]
        high_docs = [g for g in grade_result.grades if g.grade == GradeLevel.HIGH]

        low_summary = "\n".join(
            f"  - [低相关 {g.relevance_score:.2f}] {g.doc.page_content[:200]}..."
            for g in low_docs[:3]
        ) if low_docs else "  （无低相关文档）"

        high_summary = "\n".join(
            f"  - [高相关 {g.relevance_score:.2f}] {g.doc.page_content[:200]}..."
            for g in high_docs[:3]
        ) if high_docs else "  （无高相关文档）"

        prompt = f"""你是一个查询优化专家。用户的原始查询在知识库中检索效果不佳，
请分析原因并生成改进的查询。

## 原始查询
{original_query}

## 检索结果分析

【高相关文档】（{len(high_docs)} 篇）：
{high_summary}

【低相关文档】（{len(low_docs)} 篇）：
{low_summary}

## 评估总结
{grade_result.decision_reason}

## 任务
请分析为什么检索效果不佳，可能的原因包括：
1. 查询过于口语化或模糊
2. 缺少关键实体名称
3. 对比类查询需要分别指定对比对象
4. 关键词不匹配文档中的表述

请生成 1-2 个改进的查询，要求：
1. 更精确地表达信息需求
2. 包含可能的关键词
3. 如果原查询是对比类，请拆分为多个具体查询
4. 直接输出改进后的查询，不要解释

改进后的查询："""

        try:
            response = await self.llm.ainvoke(prompt)
            rewritten = response.content.strip()

            # 提取多行查询（如果有）
            lines = [l.strip() for l in rewritten.split("\n") if l.strip()]
            if len(lines) > 1:
                # 返回第一个查询，主流程用第一个
                # 后续可以用第二个做补充检索
                logger.info(f"[CRAG] 查询改写：'{original_query}' -> '{lines[0]}'")
                return lines[0]

            logger.info(f"[CRAG] 查询改写：'{original_query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.warning(f"查询改写失败: {e}")
            return original_query

    # ========================================================
    # Prompt 构建与解析
    # ========================================================

    def _build_grading_prompt(self, query: str, doc_content: str) -> str:
        """构建评估 prompt"""
        # 截断文档内容避免超出 token 限制
        max_chars = 2000
        if len(doc_content) > max_chars:
            doc_content = doc_content[:max_chars] + "..."

        return f"""你是一个检索质量评估专家。请评估以下文档片段与用户查询的相关程度。

## 用户查询
{query}

## 文档内容
{doc_content}

## 评估标准（请给出 1-5 分）：
- 5分：文档**直接、明确**回答了用户问题，包含用户查询的核心概念
- 4分：文档**高度相关**，包含查询的核心概念，但表述不够直接
- 3分：文档**部分相关**，包含一些有用信息但不完整
- 2分：文档**勉强相关**，只有零星信息点
- 1分：文档**完全不相关**，与查询毫无关系

## 输出格式
请严格按以下格式输出（只输出这两行，不要其他内容）：
SCORE: <数字1-5>
REASONING: <一句话说明评分理由>"""

    def _parse_grade_response(self, text: str) -> tuple[int, str]:
        """解析 LLM 的评估响应"""
        lines = text.strip().split("\n")

        score = 3  # 默认分数
        reasoning = "解析失败，使用默认分数"

        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_str = line.replace("SCORE:", "").strip()
                    # 尝试提取数字
                    import re
                    match = re.search(r'\d+', score_str)
                    if match:
                        score = int(match.group())
                        score = min(5, max(1, score))
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return score, reasoning

    def _normalize_score(self, raw_score: float) -> float:
        """将 1-5 的分数归一化到 0.0-1.0"""
        return (raw_score - 1) / 4


# ============================================================
# Corrective RAG 完整管线
# ============================================================

class CorrectiveRAGPipeline:
    """
    Corrective RAG 管线

    完整流程：
    1. 检索（使用 RetrieverManager）
    2. 评估（使用 RetrievalGrader）
    3. 决策：
       - HIGH → 返回高相关文档
       - LOW  → 阶段1: 重写查询 → 重新检索
                阶段2（仍LOW）: QueryExpander 分解 + 多查询并行检索 → RRF 合并
       - NO_RESULTS → 最多重试 2 次，仍无结果则返回空

    QueryExpander 集成点：
    - 当简单重写仍无法提升质量时，使用 QueryExpander.decompose_and_retrieve()
      对复杂查询（对比类、多问题类）进行多路并行检索 + RRF 排序合并

    与现有 RetrieverManager 的关系：
    - 封装 RetrieverManager.search_with_rerank()
    - 在其结果上叠加 CRAG 评估和自我纠错
    """

    def __init__(
        self,
        max_retries: int = 2,
        grade_threshold: float = 0.5,
        use_reranker: bool = True,
        candidate_multiplier: int = 3,
    ):
        """
        Args:
            max_retries: 查询重写后的最大重试次数
            grade_threshold: 相关性阈值
            use_reranker: 是否在检索时使用 Reranker
            candidate_multiplier: 检索候选文档数 = top_k × multiplier
        """
        settings = get_settings()
        self.max_retries = max_retries
        self.candidate_multiplier = candidate_multiplier

        # 评估器
        self.grader = RetrievalGrader(
            grade_threshold=grade_threshold,
            llm_temperature=0.1,
        )

        # 查询扩展器（延迟初始化）
        self._query_expander = None

        # 检索器（延迟初始化）
        self._retriever_manager = None

    @property
    def query_expander(self):
        """延迟加载 QueryExpander"""
        if self._query_expander is None:
            from src.rag.retrieval.query_expander import get_query_expander
            self._query_expander = get_query_expander()
        return self._query_expander

    @property
    def retriever_manager(self):
        """延迟加载 RetrieverManager"""
        if self._retriever_manager is None:
            from src.rag.retrieval.retriever import get_retriever_manager
            self._retriever_manager = get_retriever_manager()
        return self._retriever_manager

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[List[tuple[Document, float]], GradeResult, List[str]]:
        """
        Corrective RAG 检索

        Args:
            query: 用户查询
            top_k: 期望返回的高相关文档数量

        Returns:
            (results, grade_result, rewrite_history)
            - results: (文档, 相关分数) 列表
            - grade_result: 评估结果
            - rewrite_history: 查询改写/分解历史，用于调试
        """
        rewrite_history = [query]
        current_query = query

        for attempt in range(self.max_retries + 1):
            # Step 1: 检索更多候选（用于评估和筛选）
            candidate_k = top_k * self.candidate_multiplier

            candidates = self.retriever_manager.search_with_score(
                current_query, k=candidate_k
            )

            if not candidates:
                logger.info(f"[CRAG] 第 {attempt + 1} 次检索为空: '{current_query}'")
                if attempt < self.max_retries:
                    current_query = await self.grader.rewrite_query(
                        query,
                        GradeResult(query=current_query, grades=[])
                    )
                    rewrite_history.append(current_query)
                    continue
                else:
                    return [], GradeResult(
                        query=query,
                        grades=[],
                        decision=GradeLevel.NO_RESULTS,
                        decision_reason="检索结果为空",
                    ), rewrite_history

            # Step 2: 评估检索结果
            docs = [doc for doc, _ in candidates]
            grade_result = await self.grader.grade_retrieval(current_query, docs)

            # Step 3: 根据决策行动
            if grade_result.decision == GradeLevel.HIGH:
                # 高相关：直接返回
                high_docs = grade_result.filter_high_grade()
                # 用原始 rerank 分数排序
                high_results = self._reorder_with_original_scores(
                    high_docs, candidates, top_k
                )
                return high_results, grade_result, rewrite_history

            elif grade_result.decision == GradeLevel.LOW:
                # 低相关：重写查询 + 重试
                if attempt < self.max_retries:
                    logger.info(
                        f"[CRAG] 第 {attempt + 1} 次检索质量不足，尝试重写查询"
                    )
                    current_query = await self.grader.rewrite_query(query, grade_result)
                    rewrite_history.append(current_query)
                    continue
                else:
                    # ========================================================
                    # 最终手段：使用 QueryExpander 分解 + 多查询检索
                    # ========================================================
                    logger.info(
                        f"[CRAG] 重写仍不足，使用 QueryExpander 分解查询"
                    )
                    try:
                        decomp_results, exp_result = await self._decompose_and_search(
                            query, top_k
                        )
                        if decomp_results:
                            logger.info(
                                f"[CRAG] QueryExpander 找到 {len(decomp_results)} 篇文档"
                            )
                            # 评估分解检索的结果
                            decomp_docs = [doc for doc, _ in decomp_results]
                            decomp_grades = await self.grader.grade_retrieval(
                                query, decomp_docs
                            )
                            # 返回 RRF 合并结果
                            rewrite_history.extend(exp_result.all_queries)
                            return (
                                decomp_results,
                                decomp_grades,
                                rewrite_history,
                            )
                    except Exception as e:
                        logger.warning(f"[CRAG] QueryExpander 失败: {e}")

                    # 降级：返回所有评估过的文档中最好的
                    logger.warning(
                        f"[CRAG] 达到最大重试次数，返回次优结果"
                    )
                    all_graded = grade_result.grades
                    all_graded.sort(key=lambda g: g.relevance_score, reverse=True)
                    best_docs = [g.doc for g in all_graded[:top_k]]
                    best_results = self._reorder_with_original_scores(
                        best_docs, candidates, top_k
                    )
                    return best_results, grade_result, rewrite_history

            else:  # NO_RESULTS
                # 检索为空或无任何相关文档
                if attempt < self.max_retries:
                    current_query = await self.grader.rewrite_query(query, grade_result)
                    rewrite_history.append(current_query)
                    continue
                else:
                    return [], grade_result, rewrite_history

        # 理论上不会到达这里
        return [], GradeResult(
            query=query, grades=[],
            decision=GradeLevel.NO_RESULTS,
            decision_reason="超出重试次数",
        ), rewrite_history

    def _reorder_with_original_scores(
        self,
        docs: List[Document],
        original_candidates: List[tuple[Document, float]],
        top_k: int,
    ) -> List[tuple[Document, float]]:
        """根据原始检索分数重新排序"""
        # 构建 doc_id -> score 的映射
        doc_to_score = {}
        for doc, score in original_candidates:
            doc_key = id(doc)
            if doc_key not in doc_to_score:
                doc_to_score[doc_key] = score

        results = []
        for doc in docs:
            doc_key = id(doc)
            score = doc_to_score.get(doc_key, 0.0)
            results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def _decompose_and_search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[List[tuple[Document, float]], Any]:
        """
        使用 QueryExpander 分解查询并进行多路并行检索

        流程：
        1. QueryExpander.expand_async() 分解查询
        2. multi_query_retrieve() 并行检索所有子查询
        3. RRF 合并结果

        Returns:
            (结果列表[(doc, score)], ExpansionResult)
            注意：结果格式与 pipeline.retrieve() 一致（2-tuple，不含 source_query）
        """
        from src.rag.retrieval.query_expander import decompose_and_retrieve

        # decompose_and_retrieve 返回 List[(doc, score, source_query)]
        # 我们只取前两个元素以匹配 pipeline 格式
        full_results, exp_result = await decompose_and_retrieve(
            query=query,
            top_k=top_k,
            strategy=self.query_expander.strategy,
        )

        # 转换为 (doc, score) 格式
        stripped_results: List[tuple[Document, float]] = [
            (doc, score) for doc, score, _ in full_results
        ]

        return stripped_results, exp_result


# ============================================================
# 全局实例
# ============================================================

_grader: Optional[RetrievalGrader] = None
_pipeline: Optional[CorrectiveRAGPipeline] = None


def get_retrieval_grader(
    grade_threshold: float = None,
) -> RetrievalGrader:
    """获取 RetrievalGrader 单例"""
    global _grader
    if _grader is None:
        settings = get_settings()
        _grader = RetrievalGrader(
            grade_threshold=grade_threshold or getattr(
                settings, 'crag_grade_threshold', 0.5
            )
        )
    return _grader


def get_corrective_rag_pipeline(
    max_retries: int = None,
    grade_threshold: float = None,
) -> CorrectiveRAGPipeline:
    """获取 CorrectiveRAGPipeline 单例"""
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        _pipeline = CorrectiveRAGPipeline(
            max_retries=max_retries or getattr(settings, 'crag_max_retries', 2),
            grade_threshold=grade_threshold or getattr(
                settings, 'crag_grade_threshold', 0.5
            ),
        )
    return _pipeline


def reset_crags():
    """重置全局实例（用于测试）"""
    global _grader, _pipeline
    _grader = None
    _pipeline = None


# ============================================================
# 便捷函数
# ============================================================

async def grade_retrieval(
    query: str,
    documents: List[Document],
) -> GradeResult:
    """评估检索结果的相关性（便捷函数）"""
    grader = get_retrieval_grader()
    return await grader.grade_retrieval(query, documents)


async def corrective_retrieve(
    query: str,
    top_k: int = 5,
) -> tuple[List[tuple[Document, float]], GradeResult, list]:
    """
    Corrective RAG 检索（便捷函数）

    等价于：
        pipeline = get_corrective_rag_pipeline()
        return await pipeline.retrieve(query, top_k)
    """
    pipeline = get_corrective_rag_pipeline()
    return await pipeline.retrieve(query, top_k)
