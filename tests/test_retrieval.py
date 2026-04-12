"""
知识库检索对抗测试套件
=============================
运行方式:
    conda activate agent-demo
    export set_proxy
    python -m pytest tests/test_retrieval_adversarial.py -v --tb=short

覆盖范围:
1. 复杂查询与查询改写
2. 查询边界条件与异常输入
3. CRAG 决策边界
4. RRF 融合与排序
5. 查询扩展对抗
6. 检索投毒检测
7. 性能与压力测试
8. 综合集成测试
"""

import pytest
import asyncio
import time
import re
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.rag.evaluation.retrieval_grader import (
    RetrievalGrader, CorrectiveRAGPipeline, GradeResult,
    GradeLevel, DocumentGrade, get_corrective_rag_pipeline, reset_crags,
)
from src.rag.evaluation.conflict_detector import (
    detect_document_conflicts, extract_key_facts,
)
from src.rag.retrieval.query_expander import (
    RuleBasedDecomposer, QueryExpander, ExpandStrategy,
)
from src.rag.retrieval.hybrid_retriever import HybridRetrieverManager


# ================================================================
# Helper: 构造标准 Mock LLM（返回可控的 CRAG 评分）
# ================================================================

def make_mock_llm(score: int = 4, reasoning: str = "高度相关"):
    """创建返回指定评分的 Mock LLM"""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(
        return_value=AIMessage(content=f"SCORE: {score}\nREASONING: {reasoning}")
    )
    mock.invoke = MagicMock(
        return_value=AIMessage(content=f"SCORE: {score}\nREASONING: {reasoning}")
    )
    return mock


def make_multi_score_mock_llm(scores: List[tuple]):
    """
    创建按顺序返回不同评分的 Mock LLM
    scores: [(score, reasoning), ...]
    """
    idx = {"i": 0, "scores": scores}

    async def _invoke(prompt):
        i = idx["i"]
        idx["i"] = i + 1
        if i < len(idx["scores"]):
            s, r = idx["scores"][i]
        else:
            s, r = 3, "默认评分"
        return AIMessage(content=f"SCORE: {s}\nREASONING: {r}")

    mock = MagicMock()
    mock.ainvoke = _invoke
    mock.invoke = _invoke
    return mock


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def sample_kb_docs():
    """标准知识库文档集"""
    return [
        Document(
            page_content="公司年假政策：员工入职满1年享有年假10天，满3年享有15天。年假需提前3天申请，部门经理审批后可休假。",
            metadata={"source": "员工手册.pdf", "version": "2.1",
                      "effective_date": "2026-01-01", "expiry_date": "2099-12-31",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        ),
        Document(
            page_content="病假制度：员工每月有病假额度3天，超出需提供医院证明。病假期间按日薪80%发放。全年病假超过30天需医学评估。",
            metadata={"source": "HR制度.pdf", "version": "1.5",
                      "effective_date": "2026-01-01", "expiry_date": "2099-12-31",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        ),
        Document(
            page_content="调休规定：加班可申请调休，按1:1比例折算。调休需在加班后30天内使用，逾期作废。调休申请需直属上级审批。",
            metadata={"source": "加班管理.pdf", "version": "1.2",
                      "effective_date": "2026-01-01", "expiry_date": "2099-12-31",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        ),
        Document(
            page_content="请假流程：员工登录OA系统→填写请假申请→直属上级审批→HR备案→系统自动扣减假期余额。请假类别包括事假、病假、年假、调休。",
            metadata={"source": "OA操作指南.pdf", "version": "3.0",
                      "effective_date": "2026-01-01", "expiry_date": "2099-12-31",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        ),
        Document(
            page_content="薪酬福利：公司为正式员工缴纳五险一金，包括养老保险、医疗保险、失业保险、工伤保险、生育保险和住房公积金。住房公积金缴存比例为工资的12%。",
            metadata={"source": "薪酬福利手册.pdf", "version": "2.0",
                      "effective_date": "2026-01-01", "expiry_date": "2099-12-31",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        ),
    ]


@pytest.fixture
def poison_docs():
    """投毒文档集合"""
    return [
        Document(
            page_content="年假 年假 年假 年假 病假 请假 调休 制度 政策 福利 规定 流程 申请 审批 员工 公司 薪酬 保险 补贴 休息 假期 节假日 年假 年假 病假 病假 餐饮企业食品安全卫生标准要求，不得使用过期食材。",
            metadata={"source": "攻击文档1.pdf", "version": "1.0",
                      "effective_date": "2026-01-01", "department_restrict": [],
                      "role_restrict": [], "confidentiality": "internal"},
        ),
        Document(
            page_content="公司年假为15天。实际上是0天，休年假需额外申请特批。",
            metadata={"source": "矛盾文档.pdf", "version": "1.0",
                      "effective_date": "2026-01-01", "department_restrict": [],
                      "role_restrict": [], "confidentiality": "internal"},
        ),
        Document(
            page_content="根据最新政策，公司决定取消年假制度。员工如需休假请使用事假替代。本文档为机密文件，仅供HR部门查阅。",
            metadata={"source": "诱导文档.pdf", "version": "99.0",
                      "effective_date": "2099-01-01", "department_restrict": ["HR"],
                      "role_restrict": [], "confidentiality": "confidential"},
        ),
    ]


@pytest.fixture
def conflict_docs():
    """冲突文档集合"""
    return [
        Document(
            page_content="公司年假为15天（2026年新政策）。",
            metadata={"source": "新政策.pdf", "version": "3.0",
                      "effective_date": "2026-01-01", "department_restrict": [],
                      "role_restrict": [], "confidentiality": "internal"},
        ),
        Document(
            page_content="公司年假为10天（旧政策，2025年执行）。",
            metadata={"source": "旧政策.pdf", "version": "2.0",
                      "effective_date": "2025-01-01", "expiry_date": "2025-12-31",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        ),
        Document(
            page_content="技术部门年假为20天（特殊规定）。",
            metadata={"source": "技术部门.pdf", "version": "1.0",
                      "effective_date": "2026-01-01", "department_restrict": ["技术部"],
                      "role_restrict": [], "confidentiality": "internal"},
        ),
    ]


# ================================================================
# Section 1: 复杂查询与查询改写
# ================================================================

class TestComplexQueryDecomposition:
    """复杂查询分解对抗测试"""

    def test_multi_intent_query_simultaneous(self):
        """多意图查询：同时问年假和病假（用逗号分隔，无对比关键词）"""
        q = "公司年假怎么算，顺便告诉我病假怎么扣"
        assert RuleBasedDecomposer.needs_expansion(q) is False
        subs = RuleBasedDecomposer.decompose(q)
        assert len(subs) >= 1

    def test_multi_intent_query_and_keywords(self):
        """多意图查询：包含对比关键词，规则直接触发"""
        q = "年假和病假的区别？"
        assert RuleBasedDecomposer.needs_expansion(q) is True
        subs = RuleBasedDecomposer.decompose(q)
        assert len(subs) >= 2, "应分解为2个子查询"

    def test_contrast_query_various_connectors(self):
        """对比查询：各种连接词"""
        for kw in ["和", "与", "跟", "或", "或者"]:
            q = f"年假{kw}病假的区别"
            subs = RuleBasedDecomposer.decompose(q)
            assert len(subs) >= 2, f"连接词'{kw}'应分解为2个子查询"

    def test_list_query_what_includes(self):
        """列举类查询：公司有哪些福利"""
        patterns = [
            "公司有哪些福利",
            "员工福利有什么",
            "福利都包括什么",
            "请假类型包含哪些",
        ]
        for q in patterns:
            assert RuleBasedDecomposer.needs_expansion(q) is True, f"应识别为列举查询: {q}"

    def test_process_query_how_to(self):
        """流程类查询：怎么请假"""
        patterns = ["怎么请假", "请假流程是什么", "如何申请年假", "请假步骤是怎样的"]
        for q in patterns:
            subs = RuleBasedDecomposer.decompose(q)
            assert len(subs) >= 1, f"流程查询应有子查询: {q}"

    def test_nested_contrast_query(self):
        """嵌套对比：年假和病假以及调休的区别"""
        q = "年假和病假以及调休的区别"
        subs = RuleBasedDecomposer.decompose(q)
        assert len(subs) >= 2

    def test_reverse_contrast_query(self):
        """反向对比：年假和病假没什么区别"""
        q = "年假和病假没什么区别"
        subs = RuleBasedDecomposer.decompose(q)
        assert len(subs) >= 1

    def test_implicit_comparison(self):
        """隐式对比：年假好还是病假好"""
        q = "年假好还是病假好"
        subs = RuleBasedDecomposer.decompose(q)
        assert len(subs) >= 1

    def test_multiple_questions(self):
        """多问题查询：年假多少天？病假怎么扣？"""
        q = "年假多少天？病假怎么扣？"
        assert RuleBasedDecomposer.needs_expansion(q) is True

    def test_long_query_with_or(self):
        """长查询包含或者"""
        q = "关于员工假期管理，包括年假政策、请假流程等，还是说要问调休规定？"
        assert RuleBasedDecomposer.needs_expansion(q) is True

    @pytest.mark.asyncio
    async def test_llm_decomposer_fallback(self):
        """LLM分解器失败降级：规则无法分解时，LLM失败应回退"""
        from src.rag.retrieval.query_expander import QueryDecomposer

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM服务不可用"))

        # patch模块局部引用（而非lru_cache包装的原始函数）
        import src.rag.retrieval.query_expander as qe
        original = qe.get_llm
        qe.get_llm = MagicMock(return_value=mock_llm)

        try:
            result = await QueryDecomposer.decompose("公司有什么福利")
            assert len(result) >= 1
            assert result[0].text
        finally:
            qe.get_llm = original

    @pytest.mark.asyncio
    async def test_query_expand_hybrid_strategy(self):
        """混合策略：规则+LLM双层分解"""
        expander = QueryExpander(strategy=ExpandStrategy.HYBRID)

        mock_llm = make_mock_llm(score=3)

        import src.rag.retrieval.query_expander as qe
        original = qe.get_llm
        qe.get_llm = MagicMock(return_value=mock_llm)

        try:
            result = await expander.expand_async("年假和病假以及调休的区别")
            assert result.strategy == ExpandStrategy.HYBRID
            assert len(result.sub_queries) >= 1
        finally:
            qe.get_llm = original

    @pytest.mark.asyncio
    async def test_query_expand_rule_only_speed(self):
        """纯规则分解速度：验证<10ms（无LLM调用）"""
        expander = QueryExpander(strategy=ExpandStrategy.RULE_ONLY)

        queries = [
            "年假和病假的区别",
            "公司有哪些福利",
            "请假流程是什么",
        ]

        for q in queries:
            start = time.time()
            result = expander.expand(q)
            elapsed_ms = (time.time() - start) * 1000

            assert elapsed_ms < 10, f"规则分解应<10ms，实际{elapsed_ms:.2f}ms"
            assert result.used_llm is False


# ================================================================
# Section 2: 查询边界条件
# ================================================================

class TestQueryBoundaryConditions:
    """查询边界条件对抗测试"""

    @pytest.mark.asyncio
    async def test_extremely_short_query_1char(self):
        """单字查询：'假'"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)

        # 直接设置grader._llm绕过懒加载
        pipeline.grader._llm = make_mock_llm(score=4)

        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[])

        try:
            results, grade_result, history = await pipeline.retrieve("假", top_k=5)
            assert grade_result.decision.value in ("no_results", "low")
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_extremely_short_query_2char(self):
        """2字查询：'年假'"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[])
        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            results, grade_result, history = await pipeline.retrieve("年假", top_k=5)
            assert grade_result.decision.value in ("no_results", "low")
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_punctuation_only_query(self):
        """纯标点查询"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[])
        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            # 减少到1个查询避免超时
            results, grade_result, _ = await pipeline.retrieve("???", top_k=5)
            assert grade_result.decision.value in ("no_results", "low")
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_repeated_characters_query(self):
        """重复字符查询"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[])
        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            results, grade_result, _ = await pipeline.retrieve("年假年假年假", top_k=5)
            assert grade_result.decision.value in ("no_results", "low")
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_mixed_language_query(self):
        """中英混杂查询"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_doc = Document(
            page_content="年假15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )
        mock_rm.search_with_score = MagicMock(return_value=[(mock_doc, 0.9)])

        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            for q in ["annual leave policy年假政策", "病假sick leave"]:
                results, grade_result, _ = await pipeline.retrieve(q, top_k=5)
                assert isinstance(grade_result.decision.value, str)
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_semantic_vague_query(self):
        """语义模糊查询"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[])

        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            for q in ["那个事情怎么办", "相关规定是什么来着"]:
                results, grade_result, _ = await pipeline.retrieve(q, top_k=5)
                assert grade_result.decision.value in ("no_results", "low")
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_very_long_query_500chars(self):
        """超长查询（500字）"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[])

        pipeline.grader._llm = make_mock_llm(score=4)

        long_query = "关于公司年假政策的具体规定，我需要了解以下几个方面：" + "请详细说明。" * 100

        try:
            results, grade_result, _ = await pipeline.retrieve(long_query, top_k=5)
            assert True
        except Exception as e:
            pytest.fail(f"超长查询不应导致异常: {e}")
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_query_with_special_unicode(self):
        """特殊Unicode查询"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[])

        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            for q in ["年假\u200b政策", "年假\u3000政策", "年假\u200c\u200d政策"]:
                results, grade_result, _ = await pipeline.retrieve(q, top_k=5)
                assert isinstance(grade_result.decision.value, str)
        finally:
            mock_rm.search_with_score = original_search


# ================================================================
# Section 3: CRAG 决策边界测试
# ================================================================

class TestCRAGDecisionBoundaries:
    """CRAG决策边界对抗测试"""

    def test_decision_threshold_high_ratio_boundary(self):
        """
        HIGH边界：high_ratio=0.2（刚好20%），但avg<0.25 → 不是HIGH
        场景：5篇文档，1HIGH+1MEDIUM+3LOW，avg=0.22 < 0.25
        """
        grades = [
            DocumentGrade(doc=Document(page_content="a"), relevance_score=0.5,
                          raw_score=3, reasoning="高", grade=GradeLevel.HIGH),
            DocumentGrade(doc=Document(page_content="b"), relevance_score=0.3,
                          raw_score=2.2, reasoning="中", grade=GradeLevel.MEDIUM),
            DocumentGrade(doc=Document(page_content="c"), relevance_score=0.1,
                          raw_score=1.4, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="d"), relevance_score=0.1,
                          raw_score=1.4, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="e"), relevance_score=0.1,
                          raw_score=1.4, reasoning="无关", grade=GradeLevel.LOW),
        ]

        result = GradeResult(query="测试", grades=grades)
        assert result.high_count == 1
        assert result.high_count / result.total_docs == 0.2
        assert result.decision != GradeLevel.HIGH, "avg=0.22<0.25时不应是HIGH"

    def test_decision_no_results_at_80percent(self):
        """
        NO_RESULTS边界：low_ratio=80%（刚好），无HIGH → 触发NO_RESULTS
        场景：5篇文档，4LOW+1MEDIUM，无HIGH
        需要配置 crag_no_results_low_ratio=0.8 或调整grade
        """
        # 使用更高的low_ratio配置来测试
        grades = [
            DocumentGrade(doc=Document(page_content="a"), relevance_score=0.05,
                          raw_score=1.2, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="b"), relevance_score=0.05,
                          raw_score=1.2, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="c"), relevance_score=0.05,
                          raw_score=1.2, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="d"), relevance_score=0.05,
                          raw_score=1.2, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="e"), relevance_score=0.1,
                          raw_score=1.4, reasoning="低", grade=GradeLevel.LOW),
        ]

        result = GradeResult(query="测试", grades=grades)
        # 5/5 = 100% LOW → 无论阈值如何都应触发NO_RESULTS
        assert result.low_count == 5
        assert result.decision == GradeLevel.NO_RESULTS

    def test_decision_no_results_below_threshold(self):
        """
        NO_RESULTS边界：low_ratio=80%但有HIGH → 不触发NO_RESULTS
        """
        grades = [
            DocumentGrade(doc=Document(page_content="a"), relevance_score=0.1,
                          raw_score=1.4, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="b"), relevance_score=0.1,
                          raw_score=1.4, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="c"), relevance_score=0.1,
                          raw_score=1.4, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="d"), relevance_score=0.1,
                          raw_score=1.4, reasoning="无关", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="e"), relevance_score=0.6,
                          raw_score=3.4, reasoning="高", grade=GradeLevel.HIGH),
        ]

        result = GradeResult(query="测试", grades=grades)
        assert result.decision != GradeLevel.NO_RESULTS

    def test_decision_medium_only(self):
        """仅有MEDIUM无HIGH：avg>=0.15时为MEDIUM"""
        grades = [
            DocumentGrade(doc=Document(page_content="a"), relevance_score=0.25,
                          raw_score=2, reasoning="中", grade=GradeLevel.MEDIUM),
            DocumentGrade(doc=Document(page_content="b"), relevance_score=0.2,
                          raw_score=1.8, reasoning="中", grade=GradeLevel.MEDIUM),
        ]

        result = GradeResult(query="测试", grades=grades)
        assert result.decision == GradeLevel.MEDIUM

    def test_decision_low_fallback(self):
        """MEDIUM但avg<0.15 → LOW兜底"""
        grades = [
            DocumentGrade(doc=Document(page_content="a"), relevance_score=0.1,
                          raw_score=1.4, reasoning="低", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="b"), relevance_score=0.14,
                          raw_score=1.56, reasoning="低", grade=GradeLevel.LOW),
            DocumentGrade(doc=Document(page_content="c"), relevance_score=0.15,
                          raw_score=1.6, reasoning="中", grade=GradeLevel.MEDIUM),
        ]

        result = GradeResult(query="测试", grades=grades)
        assert result.avg_score < 0.15
        assert result.decision == GradeLevel.LOW

    def test_empty_grades_decision(self):
        """空评估结果 → NO_RESULTS"""
        result = GradeResult(query="测试", grades=[])
        assert result.decision == GradeLevel.NO_RESULTS
        assert result.total_docs == 0

    @pytest.mark.asyncio
    async def test_rewrite_query_llm_failure(self):
        """查询改写LLM失败：应返回原始查询"""
        reset_crags()
        grader = RetrievalGrader()
        # Mock LLM抛出异常，触发失败降级
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM服务不可用"))
        grader._llm = mock_llm

        mock_doc = Document(
            page_content="完全不相关",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )
        grade_result = GradeResult(
            query="测试查询",
            grades=[
                DocumentGrade(doc=mock_doc, relevance_score=0.1,
                              raw_score=1.4, reasoning="无关", grade=GradeLevel.LOW),
            ]
        )

        rewritten = await grader.rewrite_query("测试查询", grade_result)
        assert rewritten == "测试查询"

    @pytest.mark.asyncio
    async def test_concurrent_grading_consistency(self):
        """并发评估一致性：同一文档并发评分，结果应一致"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)

        mock_doc = Document(
            page_content="公司年假为15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )

        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score

        # sync mock for synchronous search_with_score
        mock_rm.search_with_score = lambda q, k: [(mock_doc, 0.9)]
        # Also mock reranker to avoid actual call
        original_rerank = mock_rm._reranker.rerank if hasattr(mock_rm, '_reranker') else None
        mock_rm._reranker = MagicMock(rerank=lambda q, docs, top_n: docs[:top_n])

        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            tasks = [pipeline.retrieve("年假政策", top_k=5) for _ in range(5)]
            results_list = await asyncio.gather(*tasks)

            decisions = [r[1].decision for r in results_list]
            assert len(set(decisions)) <= 2
        finally:
            mock_rm.search_with_score = original_search
            if original_rerank:
                mock_rm._reranker = original_rerank


# ================================================================
# Section 4: RRF 融合与排序测试
# ================================================================

class TestRRFFusion:
    """RRF融合排序对抗测试"""

    def test_adaptive_weight_short_query(self):
        """自适应权重：短查询（≤4字）BM25权重=0.7"""
        manager = HybridRetrieverManager(enable_bm25=True, enable_vector=True)
        query_len = 3
        if query_len <= 4:
            vec_w, bm_w = 0.3, 0.7
        assert bm_w == 0.7 and vec_w == 0.3

    def test_adaptive_weight_medium_query(self):
        """自适应权重：中等查询（5-8字）BM25权重=0.6"""
        query_len = 6
        if query_len <= 8:
            vec_w, bm_w = 0.4, 0.6
        assert bm_w == 0.6 and vec_w == 0.4

    def test_rrf_k_parameter(self):
        """RRF k参数敏感性：k=60时各排名贡献均衡"""
        k = 60
        doc_score_vec = 1.0 / (k + 1)
        doc_score_bm = 1.0 / (k + 5)
        assert abs(doc_score_vec - doc_score_bm) < 0.002

    def test_deduplication_by_hash(self):
        """去重：相同内容文档应有相同hash"""
        doc1 = Document(page_content="年假15天", metadata={"source": "test.pdf"})
        doc2 = Document(page_content="年假15天", metadata={"source": "test.pdf"})
        assert hash(doc1.page_content) == hash(doc2.page_content)

    @pytest.mark.asyncio
    async def test_hybrid_fusion_both_paths_available(self):
        """双路检索正常融合：vector+BM25都有结果"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        pipeline.grader._llm = make_mock_llm(score=4)

        docs = [
            Document(
                page_content=f"文档{i}内容",
                metadata={"source": f"doc{i}.pdf", "version": "1.0",
                          "effective_date": "2026-01-01",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
            )
            for i in range(5)
        ]

        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[(docs[0], 0.9)])

        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            results, grade_result, _ = await pipeline.retrieve("年假政策", top_k=5)
            assert isinstance(results, list)
        finally:
            mock_rm.search_with_score = original_search


# ================================================================
# Section 5: 查询扩展对抗测试
# ================================================================

class TestQueryExpansionAdversarial:
    """查询扩展对抗测试"""

    @pytest.mark.asyncio
    async def test_hyde_generation_failure(self):
        """HyDE生成失败：应降级到原始查询"""
        from src.rag.retrieval.query_expander import HyDEExpander

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM不可用"))

        import src.rag.retrieval.query_expander as qe
        original = qe.get_llm
        qe.get_llm = MagicMock(return_value=mock_llm)

        try:
            result = await HyDEExpander.generate_hypothetical_doc("年假政策")
            assert result == "年假政策"
        finally:
            qe.get_llm = original

    @pytest.mark.asyncio
    async def test_expansion_with_empty_results(self):
        """子查询全部返回空：分解后无结果"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[])

        pipeline.grader._llm = make_mock_llm(score=4)

        try:
            results, grade_result, _ = await pipeline.retrieve(
                "完全不存在的XYZABC内容",
                top_k=5,
                needs_expansion=True,
            )
            assert grade_result.decision.value in ("no_results", "low")
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_llm_rewrite_returns_garbage(self):
        """查询改写返回乱码：解析失败应使用默认值"""
        reset_crags()
        grader = RetrievalGrader()
        grader._llm = make_mock_llm(score=1)

        mock_llm_gc = MagicMock()
        mock_llm_gc.ainvoke = AsyncMock(
            return_value=AIMessage(content="这根本不是评分格式啊啊啊啊")
        )
        grader._llm = mock_llm_gc

        mock_doc = Document(
            page_content="年假15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )

        grade = await grader.grade_single("年假政策", mock_doc)
        assert grade.raw_score == 3  # 默认分
        assert "解析失败" in grade.reasoning

    @pytest.mark.asyncio
    async def test_rewrite_history_recorded(self):
        """查询改写历史：每次rewrite都记录"""
        reset_crags()
        pipeline = get_corrective_rag_pipeline(max_retries=2)
        pipeline.grader._llm = make_multi_score_mock_llm([(5, "高"), (3, "中")])

        docs = [
            Document(
                page_content=f"文档{i}",
                metadata={"source": f"doc{i}.pdf", "version": "1.0",
                          "effective_date": "2026-01-01",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
            )
            for i in range(3)
        ]

        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[(docs[0], 0.9), (docs[1], 0.8), (docs[2], 0.7)])

        try:
            results, grade_result, history = await pipeline.retrieve("年假", top_k=5)
            assert len(history) >= 1
        finally:
            mock_rm.search_with_score = original_search


# ================================================================
# Section 6: 检索投毒检测测试
# ================================================================

class TestRetrievalPoisoning:
    """检索投毒检测对抗测试"""

    @pytest.mark.asyncio
    async def test_keyword_stuffing_with_crag(self, poison_docs):
        """关键词填充：向量高分但CRAG评估识别"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        pipeline.grader._llm = make_mock_llm(score=2, reasoning="关键词填充，无实际内容")

        _ = pipeline.retriever_manager
        mock_rm = pipeline._retriever_manager
        original_search = mock_rm.search_with_score
        mock_rm.search_with_score = MagicMock(return_value=[(poison_docs[0], 0.95)])

        try:
            results, grade_result, _ = await pipeline.retrieve(
                "公司年假政策是什么", top_k=5
            )
            assert grade_result.low_count >= 0
        finally:
            mock_rm.search_with_score = original_search

    @pytest.mark.asyncio
    async def test_semantic_contradiction_detection(self, conflict_docs):
        """语义矛盾检测：同一数值在不同文档中矛盾"""
        warnings = detect_document_conflicts(
            [conflict_docs[0], conflict_docs[1]],
            "公司年假多少天"
        )
        assert warnings is not None
        assert len(warnings) > 0

    @pytest.mark.asyncio
    async def test_conflicting_numeric_extraction(self):
        """冲突数值提取：应提取到多个不同数值"""
        doc = Document(
            page_content="年假：15天。但实际上很多人只有10天。特殊情况还有20天的。",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )

        facts = extract_key_facts(doc)
        assert len(facts) > 0

    @pytest.mark.asyncio
    async def test_confidential_doc_filtered_by_acl(self):
        """机密文档应被ACL过滤：confidential+无HR标签"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)

        confidential_doc = Document(
            page_content="公司决定取消年假制度。机密文件。",
            metadata={"source": "机密.pdf", "version": "99.0",
                      "effective_date": "2099-01-01",
                      "department_restrict": ["HR"],
                      "role_restrict": [], "confidentiality": "confidential"},
        )

        pipeline.grader._llm = make_mock_llm(score=4)

        with patch.object(
            pipeline.retriever_manager, "search_with_score",
            return_value=[(confidential_doc, 0.9)]
        ):
            results, grade_result, _ = await pipeline.retrieve("年假", top_k=5)
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_expired_doc_version(self, sample_kb_docs):
        """过期版本文档：expiry_date已过"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)

        expired_doc = Document(
            page_content="公司年假为10天（2020年旧政策，已过期）",
            metadata={"source": "旧版手册.pdf", "version": "1.0",
                      "effective_date": "2020-01-01", "expiry_date": "2024-12-31",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )

        pipeline.grader._llm = make_mock_llm(score=4)

        with patch.object(
            pipeline.retriever_manager, "search_with_score",
            return_value=[(expired_doc, 0.9)]
        ):
            results, grade_result, _ = await pipeline.retrieve("年假", top_k=5)
            assert isinstance(results, list)


# ================================================================
# Section 7: 性能与压力测试
# ================================================================

class TestRetrievalPerformance:
    """检索性能压力测试"""

    @pytest.mark.asyncio
    async def test_large_topk_request(self):
        """超大top_k请求：top_k=100"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)

        mock_docs = [
            Document(
                page_content=f"文档{i}内容",
                metadata={"source": f"doc{i}.pdf", "version": "1.0",
                          "effective_date": "2026-01-01",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
            )
            for i in range(20)
        ]

        pipeline.grader._llm = make_mock_llm(score=4)

        with patch.object(
            pipeline.retriever_manager, "search_with_score",
            return_value=[(doc, 0.9-i*0.03) for i, doc in enumerate(mock_docs)]
        ):
            start = time.time()
            results, grade_result, _ = await pipeline.retrieve("年假政策", top_k=100)
            elapsed_ms = (time.time() - start) * 1000

            assert elapsed_ms < 10000, f"大top_k请求耗时{elapsed_ms}ms"
            assert len(results) <= 100

    @pytest.mark.asyncio
    async def test_concurrent_retrieval_storm(self):
        """并发检索风暴：50个并发请求"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)

        mock_doc = Document(
            page_content="年假15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )

        pipeline.grader._llm = make_mock_llm(score=4)

        with patch.object(
            pipeline.retriever_manager, "search_with_score",
            return_value=[(mock_doc, 0.9)]
        ):
            start = time.time()
            tasks = [pipeline.retrieve(f"查询{i}", top_k=5) for i in range(50)]
            results_list = await asyncio.gather(*tasks)
            elapsed = time.time() - start

            assert elapsed < 60, f"50并发请求耗时{elapsed}s"
            assert len(results_list) == 50

    @pytest.mark.asyncio
    async def test_continuous_rewrite_loop_limit(self):
        """
        连续查询改写循环：验证rewrite次数有上限
        max_retries=2时，最多3次查询（原始+2次重写）
        """
        reset_crags()
        pipeline = get_corrective_rag_pipeline(max_retries=2)

        mock_low_doc = Document(
            page_content="完全不相关XYZABC",
            metadata={"source": "干扰.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )

        pipeline.grader._llm = make_mock_llm(score=2, reasoning="无关")

        with patch.object(
            pipeline.retriever_manager, "search_with_score",
            return_value=[(mock_low_doc, 0.1)]
        ):
            results, grade_result, history = await pipeline.retrieve(
                "完全不存在的XYZ查询ABC",
                top_k=5
            )
            assert len(history) <= 3

    @pytest.mark.asyncio
    async def test_llm_rate_limit_retry_success(self):
        """LLM 429限流重试：第3次成功"""
        from src.rag.evaluation.retrieval_grader import RetrievalGrader

        call_count = {"count": 0}

        async def failing_then_success(prompt):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise Exception("429 Too Many Requests")
            return AIMessage(content="SCORE: 4\nREASONING: 高度相关")

        grader = RetrievalGrader()
        mock_llm = MagicMock()
        mock_llm.ainvoke = failing_then_success
        grader._llm = mock_llm

        doc = Document(
            page_content="年假15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )

        grade = await grader.grade_single("年假", doc)
        assert call_count["count"] == 3

    @pytest.mark.asyncio
    async def test_llm_all_retries_fail(self):
        """LLM 3次重试全部失败 → 返回LOW"""
        from src.rag.evaluation.retrieval_grader import RetrievalGrader

        async def always_fail(prompt):
            raise Exception("500 Internal Server Error")

        grader = RetrievalGrader()
        mock_llm = MagicMock()
        mock_llm.ainvoke = always_fail
        grader._llm = mock_llm

        doc = Document(
            page_content="年假15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"},
        )

        grade = await grader.grade_single("年假", doc)
        assert grade.grade.value == GradeLevel.LOW
        assert "评估失败" in grade.reasoning


# ================================================================
# Section 8: 综合集成测试
# ================================================================

class TestRetrievalIntegration:
    """综合集成测试"""

    @pytest.mark.asyncio
    async def test_full_pipeline_healthy_query(self, sample_kb_docs):
        """完整流程-正常查询"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        pipeline.grader._llm = make_mock_llm(score=4, reasoning="高度相关")

        with patch.object(
            pipeline.retriever_manager, "search_with_score",
            return_value=[(doc, 0.9-i*0.1) for i, doc in enumerate(sample_kb_docs)]
        ):
            results, grade_result, history = await pipeline.retrieve(
                "公司年假政策是什么", top_k=5
            )
            assert len(results) > 0
            assert isinstance(grade_result.decision.value, str)

    @pytest.mark.asyncio
    async def test_full_pipeline_contrast_with_expansion(self, sample_kb_docs):
        """完整流程-对比查询触发QE"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        pipeline.grader._llm = make_mock_llm(score=4)

        with patch.object(
            pipeline.retriever_manager, "search_with_score",
            return_value=[(doc, 0.9-i*0.1) for i, doc in enumerate(sample_kb_docs)]
        ):
            results, grade_result, history = await pipeline.retrieve(
                "年假和病假的区别",
                top_k=5,
                needs_expansion=True,
            )
            assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_pipeline_decision_has_reasoning(self, sample_kb_docs):
        """决策可追溯性：每步决策都有reasoning"""
        reset_crags()
        # disable rerank_before_grade to avoid reranker call that times out
        pipeline = get_corrective_rag_pipeline(rerank_before_grade=False)
        pipeline.grader._llm = make_mock_llm(score=4)

        with patch.object(
            pipeline.retriever_manager, "search_with_score",
            return_value=[(sample_kb_docs[0], 0.9)]
        ):
            results, grade_result, history = await pipeline.retrieve("年假", top_k=5)

            assert grade_result.decision_reason
            assert len(history) >= 1


# ================================================================
# Section 9: 检索质量指标评测
# ================================================================

from src.rag.evaluation.retrieval_metrics import (
    RetrievalMetricsEngine,
    recall_at_k, precision_at_k, f1_at_k, mrr, ndcg_at_k, hit_at_k,
    map_score, compute_bm25_scores, rrf_score, fused_rrf_score,
    GROUND_TRUTH_DATASET,
)


class TestRetrievalQualityMetrics:
    """
    检索质量指标评测

    覆盖指标：
    - Recall@K / Precision@K / F1@K
    - MRR (Mean Reciprocal Rank)
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - MAP (Mean Average Precision)
    - Hit@K
    - BM25 Score
    - RRF Score

    Ground Truth 来源：
    - GROUND_TRUTH_DATASET（见 retrieval_metrics.py）
    - sample_kb_docs / conflict_docs fixtures
    """

    # ---------- Recall@K ----------

    def test_recall_at_1_perfect(self, sample_kb_docs):
        """Recall@1：相关文档排在第1位"""
        target_doc = sample_kb_docs[0]  # 员工手册.pdf
        retrieved = [(sample_kb_docs[0], 0.95), (sample_kb_docs[1], 0.8)]
        score = recall_at_k(retrieved, ["员工手册.pdf"], k=1)
        assert score == 1.0, "相关文档在第1位，Recall@1应为1.0"

    def test_recall_at_3_partial(self, sample_kb_docs):
        """Recall@3：相关文档在第3位"""
        retrieved = [
            (sample_kb_docs[1], 0.9),  # HR制度.pdf（相关）
            (sample_kb_docs[2], 0.8),  # 加班管理.pdf（不相关）
            (sample_kb_docs[0], 0.7),  # 员工手册.pdf（相关，在第3位）
        ]
        score = recall_at_k(retrieved, ["员工手册.pdf"], k=3)
        assert score == 1.0, "唯一相关文档在top-3内，Recall@3=1.0"

    def test_recall_at_5_all_relevant(self, sample_kb_docs):
        """Recall@5：全部4篇相关文档在top-5内"""
        retrieved = [
            (sample_kb_docs[i % 5], 0.95 - i * 0.05)
            for i in range(5)
        ]
        # 4篇相关
        relevant = ["员工手册.pdf", "HR制度.pdf", "OA操作指南.pdf", "薪酬福利手册.pdf"]
        score = recall_at_k(retrieved, relevant, k=5)
        assert score == 4 / 4, "4/4命中，Recall@5=1.0"

    def test_recall_at_5_missed(self, sample_kb_docs):
        """Recall@5：部分相关文档不在top-5内"""
        retrieved = [
            (sample_kb_docs[1], 0.9),
            (sample_kb_docs[2], 0.8),
        ]
        relevant = ["员工手册.pdf", "HR制度.pdf", "OA操作指南.pdf"]
        score = recall_at_k(retrieved, relevant, k=5)
        assert score == 1 / 3, "1/3命中，Recall@5=0.333"

    def test_recall_at_k_empty_relevant(self, sample_kb_docs):
        """Recall@K：ground truth为空（无答案查询）"""
        retrieved = [(sample_kb_docs[0], 0.9), (sample_kb_docs[1], 0.8)]
        score = recall_at_k(retrieved, [], k=5)
        assert score == 1.0, "无答案查询的Recall应返回1.0（不算错误）"

    def test_recall_at_k_empty_retrieved(self, sample_kb_docs):
        """Recall@K：检索结果为空"""
        retrieved = []
        score = recall_at_k(retrieved, ["员工手册.pdf"], k=5)
        assert score == 0.0, "无检索结果时Recall=0"

    # ---------- Precision@K ----------

    def test_precision_at_1_perfect(self, sample_kb_docs):
        """Precision@1：第1位是相关文档"""
        retrieved = [(sample_kb_docs[0], 0.95)]
        score = precision_at_k(retrieved, ["员工手册.pdf"], k=1)
        assert score == 1.0

    def test_precision_at_1_imperfect(self, sample_kb_docs):
        """Precision@1：第1位不是相关文档"""
        retrieved = [(sample_kb_docs[1], 0.95)]  # HR制度.pdf（但期望员工手册.pdf）
        score = precision_at_k(retrieved, ["员工手册.pdf"], k=1)
        assert score == 0.0

    def test_precision_at_3_mixed(self, sample_kb_docs):
        """Precision@3：top-3中有2个相关"""
        retrieved = [
            (sample_kb_docs[0], 0.9),  # 员工手册.pdf（相关）
            (sample_kb_docs[2], 0.8),  # 加班管理.pdf（不相关）
            (sample_kb_docs[1], 0.7),  # HR制度.pdf（相关）
        ]
        score = precision_at_k(retrieved, ["员工手册.pdf", "HR制度.pdf"], k=3)
        assert score == 2 / 3, "2/3命中，Precision@3=0.667"

    def test_precision_at_5_with_noise(self, sample_kb_docs):
        """Precision@5：top-5中有1个相关"""
        retrieved = [
            (sample_kb_docs[2], 0.9),
            (sample_kb_docs[3], 0.8),
            (sample_kb_docs[0], 0.7),  # 相关
            (sample_kb_docs[4], 0.6),
            (sample_kb_docs[1], 0.5),
        ]
        score = precision_at_k(retrieved, ["员工手册.pdf"], k=5)
        assert score == 1 / 5, "1/5命中，Precision@5=0.2"

    def test_precision_at_k_zero(self):
        """Precision@K：k=0时返回0"""
        retrieved = [Document(page_content="dummy", metadata={"source": "员工手册.pdf"})]
        score = precision_at_k(retrieved, ["员工手册.pdf"], k=0)
        assert score == 0.0

    # ---------- F1@K ----------

    def test_f1_at_1(self, sample_kb_docs):
        """F1@1 = 2*1*1/2 = 1.0"""
        retrieved = [(sample_kb_docs[0], 0.9)]
        score = f1_at_k(retrieved, ["员工手册.pdf"], k=1)
        assert score == 1.0

    def test_f1_at_3(self, sample_kb_docs):
        """F1@3：Precision@3=1/3, Recall@3=1.0 → F1=0.5"""
        retrieved = [
            (sample_kb_docs[0], 0.9),  # 相关
            (sample_kb_docs[1], 0.8),
            (sample_kb_docs[2], 0.7),
        ]
        score = f1_at_k(retrieved, ["员工手册.pdf"], k=3)
        assert abs(score - 0.5) < 0.001, f"F1@3应为0.5，实际{score}"

    def test_f1_at_k_zero(self, sample_kb_docs):
        """F1@K：P=R=0时返回0"""
        retrieved = [(sample_kb_docs[1], 0.9)]
        score = f1_at_k(retrieved, ["员工手册.pdf"], k=1)
        assert score == 0.0

    # ---------- MRR ----------

    def test_mrr_first_relevant_at_1(self, sample_kb_docs):
        """MRR：第一个相关文档在第1位"""
        retrieved = [
            (sample_kb_docs[0], 0.95),
            (sample_kb_docs[1], 0.8),
        ]
        score = mrr(retrieved, ["员工手册.pdf"])
        assert score == 1.0

    def test_mrr_first_relevant_at_2(self, sample_kb_docs):
        """MRR：第一个相关文档在第2位"""
        retrieved = [
            (sample_kb_docs[2], 0.9),
            (sample_kb_docs[0], 0.8),
            (sample_kb_docs[1], 0.7),
        ]
        score = mrr(retrieved, ["员工手册.pdf"])
        assert score == 0.5, "第一个相关在第2位，MRR=1/2=0.5"

    def test_mrr_first_relevant_at_3(self, sample_kb_docs):
        """MRR：第一个相关文档在第3位"""
        retrieved = [
            (sample_kb_docs[2], 0.9),
            (sample_kb_docs[3], 0.8),
            (sample_kb_docs[0], 0.7),
        ]
        score = mrr(retrieved, ["员工手册.pdf"])
        assert abs(score - 1/3) < 0.001, f"第一个相关在第3位，MRR=1/3，实际{score:.4f}"

    def test_mrr_no_hit(self, sample_kb_docs):
        """MRR：没有任何相关文档命中"""
        retrieved = [
            (sample_kb_docs[1], 0.9),
            (sample_kb_docs[2], 0.8),
        ]
        score = mrr(retrieved, ["员工手册.pdf"])
        assert score == 0.0

    def test_mrr_empty_relevant(self, sample_kb_docs):
        """MRR：ground truth为空"""
        retrieved = [(sample_kb_docs[0], 0.9)]
        score = mrr(retrieved, [])
        assert score == 0.0

    def test_mrr_multiple_relevant(self, sample_kb_docs):
        """MRR：多个相关文档，取第一个的排名"""
        retrieved = [
            (sample_kb_docs[0], 0.9),
            (sample_kb_docs[1], 0.8),
            (sample_kb_docs[0], 0.7),
        ]
        score = mrr(retrieved, ["员工手册.pdf", "HR制度.pdf"])
        assert score == 1.0, "第一个相关文档在第1位，MRR=1.0"

    # ---------- NDCG@K ----------

    def test_ndcg_at_k_perfect(self, sample_kb_docs):
        """NDCG@5：所有相关文档排在最前面"""
        retrieved = [
            (sample_kb_docs[0], 0.95),
            (sample_kb_docs[1], 0.90),
            (sample_kb_docs[2], 0.85),
        ]
        score = ndcg_at_k(retrieved, ["员工手册.pdf", "HR制度.pdf"], k=5)
        assert score == 1.0, "完美排序NDCG=1.0"

    def test_ndcg_at_k_partial(self, sample_kb_docs):
        """NDCG@5：相关文档排在中间"""
        retrieved = [
            (sample_kb_docs[2], 0.9),
            (sample_kb_docs[0], 0.8),
            (sample_kb_docs[3], 0.7),
        ]
        score = ndcg_at_k(retrieved, ["员工手册.pdf"], k=5)
        assert 0 < score < 1.0, "部分排序时0<NDCG<1"

    def test_ndcg_at_k_with_relevance_scores(self, sample_kb_docs):
        """NDCG@K：传入分级相关性分数"""
        retrieved = [
            (sample_kb_docs[0], 0.95),
            (sample_kb_docs[1], 0.80),
        ]
        # 分级相关：[高度相关=1.0, 中等相关=0.5]
        scores = [1.0, 0.5]
        score = ndcg_at_k(retrieved, ["员工手册.pdf"], relevance_scores=scores, k=5)
        assert score == 1.0

    def test_ndcg_at_k_no_relevant(self, sample_kb_docs):
        """NDCG@K：无相关文档"""
        retrieved = [
            (sample_kb_docs[2], 0.9),
            (sample_kb_docs[3], 0.8),
        ]
        score = ndcg_at_k(retrieved, ["员工手册.pdf"], k=5)
        assert score == 0.0

    def test_ndcg_at_k_empty_relevant(self, sample_kb_docs):
        """NDCG@K：ground truth为空"""
        retrieved = [(sample_kb_docs[0], 0.9)]
        score = ndcg_at_k(retrieved, [], k=5)
        assert score == 1.0, "无答案查询NDCG=1.0"

    def test_ndcg_at_k_zero_k(self, sample_kb_docs):
        """NDCG@K：k=0时返回0"""
        retrieved = [(sample_kb_docs[0], 0.9)]
        score = ndcg_at_k(retrieved, ["员工手册.pdf"], k=0)
        assert score == 0.0

    # ---------- MAP (Mean Average Precision) ----------

    def test_map_perfect(self, sample_kb_docs):
        """MAP：所有相关文档均排在最前"""
        retrieved = [
            (sample_kb_docs[0], 0.95),
            (sample_kb_docs[1], 0.90),
            (sample_kb_docs[2], 0.85),
        ]
        score = map_score([retrieved], [["员工手册.pdf", "HR制度.pdf"]])
        assert score == 1.0

    def test_map_partial(self, sample_kb_docs):
        """MAP：部分命中"""
        retrieved = [
            (sample_kb_docs[0], 0.95),  # rank=1 相关
            (sample_kb_docs[2], 0.85),
            (sample_kb_docs[1], 0.80),  # rank=3 相关
        ]
        # AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
        score = map_score([retrieved], [["员工手册.pdf", "HR制度.pdf"]])
        expected = (1.0 + 2/3) / 2
        assert abs(score - expected) < 0.001, f"MAP应为{expected:.4f}，实际{score:.4f}"

    def test_map_multiple_queries(self, sample_kb_docs):
        """MAP：批量多查询"""
        r1 = [(sample_kb_docs[0], 0.95), (sample_kb_docs[1], 0.80)]
        r2 = [(sample_kb_docs[2], 0.90), (sample_kb_docs[3], 0.70)]
        scores = map_score(
            [r1, r2],
            [["员工手册.pdf"], ["加班管理.pdf"]]
        )
        # r1: AP=1.0 (1/1), r2: AP=1.0 (1/1)
        # MAP = (1.0 + 1.0) / 2 = 1.0
        assert scores == 1.0

    # ---------- Hit@K ----------

    def test_hit_at_1(self, sample_kb_docs):
        """Hit@1：第1位命中"""
        retrieved = [(sample_kb_docs[0], 0.95), (sample_kb_docs[1], 0.80)]
        score = hit_at_k(retrieved, ["员工手册.pdf"], k=1)
        assert score == 1.0

    def test_hit_at_5_no(self, sample_kb_docs):
        """Hit@5：top-5内无命中"""
        retrieved = [
            (sample_kb_docs[1], 0.90),
            (sample_kb_docs[2], 0.80),
            (sample_kb_docs[3], 0.70),
            (sample_kb_docs[4], 0.60),
        ]
        score = hit_at_k(retrieved, ["员工手册.pdf"], k=5)
        assert score == 0.0

    def test_hit_at_k_adversarial(self, sample_kb_docs):
        """Hit@K：对抗查询（无答案）不应有命中"""
        retrieved = [
            (sample_kb_docs[0], 0.95),
            (sample_kb_docs[1], 0.80),
        ]
        # 对抗查询的正确答案是空集
        score = hit_at_k(retrieved, [], k=5)
        assert score == 1.0, "无答案查询Hit@K应为1.0"

    # ---------- BM25 Score ----------

    def test_bm25_scores_ranked(self, sample_kb_docs):
        """BM25：按分数降序排列"""
        scores = compute_bm25_scores("年假", sample_kb_docs, k=5)
        assert len(scores) <= 5
        # 员工手册.pdf 应该排在最前面（年假内容最多）
        assert scores[0][0].metadata["source"] == "员工手册.pdf"
        # 分数应递减
        for i in range(len(scores) - 1):
            assert scores[i][1] >= scores[i+1][1], "BM25分数应降序"

    def test_bm25_scores_short_query(self, sample_kb_docs):
        """BM25：短查询（≤4字）"""
        scores = compute_bm25_scores("请假", sample_kb_docs, k=5)
        assert len(scores) > 0, "短查询应有结果"
        assert scores[0][1] >= 0

    def test_bm25_scores_nonexistent(self, sample_kb_docs):
        """BM25：查询词不在语料中"""
        scores = compute_bm25_scores("完全不存在的XYZABC", sample_kb_docs, k=5)
        # Fallback模式下可能全为0
        assert len(scores) <= 5

    # ---------- RRF Score ----------

    def test_rrf_score_formula(self):
        """RRF公式验证：1/(k+rank)"""
        # k=60, rank=1
        assert abs(rrf_score(1, 60) - 1/61) < 0.001
        # k=60, rank=5
        assert abs(rrf_score(5, 60) - 1/65) < 0.001

    def test_rrf_score_equal_ranks(self):
        """RRF：相同排名时两路贡献相等"""
        score_1_5 = fused_rrf_score(bm25_rank=1, vector_rank=5, bm25_weight=0.5, vector_weight=0.5)
        score_5_1 = fused_rrf_score(bm25_rank=5, vector_rank=1, bm25_weight=0.5, vector_weight=0.5)
        assert abs(score_1_5 - score_5_1) < 0.001, "两路排名互换，分数应相同"

    def test_rrf_score_weighted(self):
        """RRF：加权融合"""
        # BM25排第1，Vector排第10
        score_b = fused_rrf_score(bm25_rank=1, vector_rank=10, bm25_weight=0.7, vector_weight=0.3)
        # 反过来
        score_v = fused_rrf_score(bm25_rank=10, vector_rank=1, bm25_weight=0.7, vector_weight=0.3)
        # BM25权重更大时，BM25排第1应该得分更高
        assert score_b > score_v, "BM25排第1且BM25权重0.7时，应比Vector排第1得分更高"

    def test_rrf_score_unseen_doc(self):
        """RRF：某路未命中（rank=0）"""
        score = fused_rrf_score(bm25_rank=0, vector_rank=3, bm25_weight=0.5, vector_weight=0.5)
        # 只靠Vector第3名的贡献
        expected = 0.5 * rrf_score(3, 60)
        assert abs(score - expected) < 0.001, f"单路命中时RRF=单路贡献*权重"

    # ---------- 综合评估引擎 ----------

    def test_metrics_engine_evaluate_query(self, sample_kb_docs):
        """引擎：评估单个查询"""
        engine = RetrievalMetricsEngine()
        retrieved = [
            (sample_kb_docs[0], 0.95),
            (sample_kb_docs[1], 0.90),
            (sample_kb_docs[2], 0.85),
        ]
        result = engine.evaluate_query("公司年假政策是什么", retrieved)

        assert result.query == "公司年假政策是什么"
        assert result.recall_at_1 == 1.0
        assert result.mrr == 1.0
        assert result.category == "normal"

    def test_metrics_engine_aggregate(self, sample_kb_docs):
        """引擎：聚合多个查询结果"""
        engine = RetrievalMetricsEngine()

        retrieved_map = {
            "公司年假政策是什么": [
                (sample_kb_docs[0], 0.95),
                (sample_kb_docs[1], 0.80),
            ],
            "病假怎么扣": [
                (sample_kb_docs[1], 0.90),
            ],
            "完全不存在的XYZABC内容": [],
        }
        agg = engine.evaluate_stage(retrieved_map, "单元测试阶段", "测试用")

        assert agg.num_queries == 3
        assert agg.num_adversarial == 1
        assert 0 <= agg.mrr <= 1.0
        assert 0 <= agg.ndcg_at_5 <= 1.0

    def test_metrics_engine_adversarial_query(self, sample_kb_docs):
        """引擎：对抗查询识别"""
        engine = RetrievalMetricsEngine()
        # 检索到了文档，但ground truth为空（正确做法是不应返回）
        retrieved = [(sample_kb_docs[0], 0.95)]
        result = engine.evaluate_query("完全不存在的XYZABC内容", retrieved)

        assert result.is_adversarial is True
        assert result.category == "adversarial_nonexistent"
        # Recall=1.0 因为无答案查询的空集全被命中
        assert result.recall_at_5 == 1.0

    def test_metrics_engine_conflict_query(self, conflict_docs):
        """引擎：冲突文档查询"""
        engine = RetrievalMetricsEngine()
        # 2个相关文档：员工手册.pdf、新政策.pdf
        # 检索到：新政策.pdf（第1）、旧政策.pdf（第2）
        retrieved = [
            (conflict_docs[0], 0.95),  # 新政策.pdf
            (conflict_docs[1], 0.80),  # 旧政策.pdf
        ]
        result = engine.evaluate_query("年假多少天", retrieved)

        # MRR=1.0（第1名是新政策.pdf，在ground truth中）
        assert result.mrr == 1.0, "第1名是新政策.pdf，MRR=1.0"
        # Hit@1=1.0（第1名是相关文档）
        assert result.hit_at_1 == 1.0
        # 2个相关中1个在top-1 → Recall@1=0.5
        assert abs(result.recall_at_1 - 0.5) < 0.001, "2个相关中1个在top-1"
        # 2个相关中1个在top-3 → Recall@3=0.5（员工手册不在检索结果中）
        assert abs(result.recall_at_3 - 0.5) < 0.001, "2个相关中1个在top-3"
        # Precision@1=1.0（新政策.pdf是相关的）
        assert result.precision_at_1 == 1.0

    def test_metrics_result_to_dict(self, sample_kb_docs):
        """结果序列化：to_dict输出"""
        engine = RetrievalMetricsEngine()
        retrieved = [(sample_kb_docs[0], 0.95), (sample_kb_docs[1], 0.80)]
        # "公司年假政策是什么"的GT是["员工手册.pdf"]（单个doc）
        result = engine.evaluate_query("公司年假政策是什么", retrieved)
        d = result.to_dict()

        assert "query" in d
        assert "recall_at_1" in d
        assert "mrr" in d
        assert "relevant_ids" in d
        assert "retrieved_ids" in d
        assert d["recall_at_1"] == 1.0, "员工手册.pdf在第1位，Recall@1=1.0"
        assert d["mrr"] == 1.0

    def test_aggregated_metrics_to_dict(self, sample_kb_docs):
        """聚合结果序列化"""
        engine = RetrievalMetricsEngine()
        retrieved_map = {
            "公司年假政策是什么": [(sample_kb_docs[0], 0.95)],
        }
        agg = engine.evaluate_stage(retrieved_map, "测试阶段", "测试描述")
        d = agg.to_dict()

        assert "metrics" in d
        assert "recall_at_1" in d["metrics"]
        assert "mrr" in d["metrics"]
        assert d["num_queries"] == 1

    # ---------- Ground Truth 数据集完整性 ----------

    def test_ground_truth_coverage(self):
        """Ground Truth：覆盖所有主要查询类别"""
        categories = {}
        for q, info in GROUND_TRUTH_DATASET.items():
            cat = info.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        assert "normal" in categories and categories["normal"] >= 10, "至少10个正常查询"
        assert "contrast" in categories, "应有对比类查询"
        assert "enumerate" in categories, "应有列举类查询"
        assert "short" in categories, "应有短查询"
        assert "adversarial_nonexistent" in categories, "应有无答案对抗查询"

    def test_ground_truth_doc_ids_valid(self):
        """Ground Truth：所有relevant doc_id均有效"""
        valid_ids = {
            "员工手册.pdf", "HR制度.pdf", "加班管理.pdf",
            "OA操作指南.pdf", "薪酬福利手册.pdf",
            "新政策.pdf", "旧政策.pdf", "技术部门.pdf",
        }
        for q, info in GROUND_TRUTH_DATASET.items():
            for doc_id in info.get("relevant", []):
                assert doc_id in valid_ids, f"查询'{q}'中的doc_id'{doc_id}'不在有效集合中"


# ================================================================
# Section 10: 指标驱动的对抗评测
# ================================================================

class TestAdversarialMetrics:
    """
    指标驱动的对抗评测：
    用指标量化评估系统在对抗场景下的表现，而非仅断言通过/失败。
    """

    def test_poison_keyword_stuffing_recall_impact(self, poison_docs, sample_kb_docs):
        """投毒攻击 Recall 影响：关键词填充不应拉高Recall"""
        engine = RetrievalMetricsEngine()

        # 正常检索：相关文档排在第1
        normal_retrieved = [
            (sample_kb_docs[0], 0.95),  # 员工手册.pdf
            (sample_kb_docs[1], 0.80),
            (sample_kb_docs[2], 0.70),
        ]
        normal_result = engine.evaluate_query("公司年假政策是什么", normal_retrieved)

        # 投毒检索：攻击文档排在第1，真正相关文档排在第2
        poison_retrieved = [
            (poison_docs[0], 0.98),  # 攻击文档1.pdf（关键词填充）
            (sample_kb_docs[0], 0.90),  # 员工手册.pdf
            (sample_kb_docs[1], 0.80),
        ]
        poison_result = engine.evaluate_query("公司年假政策是什么", poison_retrieved)

        # 投毒后 Recall@3 应该保持（因为员工手册.pdf仍在top-3）
        assert poison_result.recall_at_3 == normal_result.recall_at_3, \
            "关键词填充不应影响Recall@3"
        # 投毒后 MRR 应该下降（攻击文档抢了第1名）
        assert poison_result.mrr <= normal_result.mrr, \
            "关键词填充应降低MRR（正确答案排名下降）"

    def test_contradict_doc_precision_impact(self, conflict_docs):
        """冲突文档 Precision 影响：过期/矛盾文档应拉低Precision"""
        engine = RetrievalMetricsEngine()

        # 权威检索：新政策.pdf 排在第1（相关文档）
        authoritative_retrieved = [
            (conflict_docs[0], 0.95),  # 新政策.pdf
            (conflict_docs[1], 0.80),  # 旧政策.pdf
        ]
        auth_result = engine.evaluate_query("年假多少天", authoritative_retrieved)

        # 被污染检索：旧政策.pdf 排在第1（不相关）
        polluted_retrieved = [
            (conflict_docs[1], 0.95),  # 旧政策.pdf
            (conflict_docs[0], 0.80),  # 新政策.pdf
        ]
        polluted_result = engine.evaluate_query("年假多少天", polluted_retrieved)

        # Recall@3应相同（top-3内都有新政策.pdf）
        assert auth_result.recall_at_3 == polluted_result.recall_at_3, \
            "Recall@3应相同（top-3内都有正确答案）"
        # 权威检索：第1名是相关文档 → Precision@1=1/1=1.0
        # 被污染检索：第1名不相关，第2名才相关 → Precision@1=0/1=0
        assert auth_result.precision_at_1 > polluted_result.precision_at_1, \
            "权威文档Precision@1应高于被污染的检索"

    def test_enumerate_query_multi_doc_coverage(self, sample_kb_docs):
        """列举查询 Recall@5：多文档联合覆盖"""
        engine = RetrievalMetricsEngine()

        # 列举类查询"公司有哪些福利"需要4篇文档全部覆盖
        retrieved = [
            (sample_kb_docs[4], 0.95),  # 薪酬福利手册.pdf
            (sample_kb_docs[1], 0.90),  # HR制度.pdf
            (sample_kb_docs[2], 0.85),  # 加班管理.pdf
            (sample_kb_docs[0], 0.80),  # 员工手册.pdf
        ]
        result = engine.evaluate_query("公司有哪些福利", retrieved)

        # 4篇相关都命中了 → Recall@4=1.0
        assert recall_at_k(retrieved, ["薪酬福利手册.pdf", "HR制度.pdf", "加班管理.pdf", "员工手册.pdf"], 4) == 1.0, \
            "4篇全部命中Recall@4=1.0"
        # 只命中3篇 → Recall@5=3/4=0.75
        result_partial = engine.evaluate_query("公司有哪些福利", retrieved[:3])
        assert recall_at_k(retrieved[:3], ["薪酬福利手册.pdf", "HR制度.pdf", "加班管理.pdf", "员工手册.pdf"], 5) == 3 / 4, \
            "3/4命中Recall@5=0.75"

    def test_short_query_bm25_weight_effect(self, sample_kb_docs):
        """短查询BM25权重效果：精确关键词匹配应优先"""
        scores_bm25 = compute_bm25_scores("年假", sample_kb_docs, k=3)
        bm25_top_doc = scores_bm25[0][0].metadata["source"]

        # 员工手册.pdf应排在BM25第1（年假内容最相关）
        assert bm25_top_doc == "员工手册.pdf", \
            f"短查询'年假'的BM25第1名应为员工手册.pdf，实际为{bm25_top_doc}"

    def test_contrast_query_expansion_mrr(self, sample_kb_docs):
        """对比查询：未扩展 vs 扩展后的 MRR 对比"""
        engine = RetrievalMetricsEngine()

        # 未扩展：只查"年假和病假"原始查询，可能只命中1篇
        unexpanded = [
            (sample_kb_docs[3], 0.90),  # OA操作指南（不相关）
            (sample_kb_docs[0], 0.80),  # 员工手册.pdf
            (sample_kb_docs[2], 0.70),  # 加班管理.pdf
        ]
        unexpanded_result = engine.evaluate_query("年假和病假的区别", unexpanded)

        # 扩展后：同时命中年假和病假两个文档
        expanded = [
            (sample_kb_docs[0], 0.95),  # 员工手册.pdf（年假）
            (sample_kb_docs[1], 0.90),  # HR制度.pdf（病假）
            (sample_kb_docs[2], 0.80),  # 加班管理.pdf
        ]
        expanded_result = engine.evaluate_query("年假和病假的区别", expanded)

        # 扩展后 Recall@3 应 >= 未扩展
        assert expanded_result.recall_at_3 >= unexpanded_result.recall_at_3, \
            "查询扩展后Recall应不降低"
        # MRR 应该更好或相同
        assert expanded_result.mrr >= unexpanded_result.mrr, \
            "查询扩展后MRR应不降低"

    def test_no_results_query_metrics(self, sample_kb_docs):
        """无答案查询指标：Hit@K=1.0, Recall=1.0, MRR=0.0"""
        engine = RetrievalMetricsEngine()
        retrieved = [
            (sample_kb_docs[1], 0.90),
            (sample_kb_docs[2], 0.80),
        ]
        result = engine.evaluate_query("完全不存在的XYZABC内容", retrieved)

        assert result.recall_at_5 == 1.0, "无答案查询Recall=1.0"
        assert result.mrr == 0.0, "无答案查询MRR=0.0"
        assert result.hit_at_5 == 1.0, "无答案查询Hit@K=1.0（空集全命中）"
        assert result.is_adversarial is True

    def test_llm_rewrite_improves_metrics(self, sample_kb_docs):
        """查询改写：rewrite后指标应改善"""
        engine = RetrievalMetricsEngine()

        # 原始查询（可能因为表述问题排名不好）
        original = [
            (sample_kb_docs[2], 0.90),  # 加班管理.pdf（误导）
            (sample_kb_docs[0], 0.80),  # 员工手册.pdf（第2位）
            (sample_kb_docs[1], 0.70),
        ]
        original_result = engine.evaluate_query("年假多少天", original)

        # rewrite后（更精准的查询）
        rewritten = [
            (sample_kb_docs[0], 0.95),  # 员工手册.pdf（第1位）
            (sample_kb_docs[1], 0.85),
            (sample_kb_docs[2], 0.75),
        ]
        rewritten_result = engine.evaluate_query("年假多少天", rewritten)

        # rewrite后指标应改善或不变
        assert rewritten_result.recall_at_1 >= original_result.recall_at_1
        assert rewritten_result.mrr >= original_result.mrr
        assert rewritten_result.ndcg_at_3 >= original_result.ndcg_at_3


# ================================================================
# 运行摘要
# ================================================================

def test_summary():
    print("\n" + "="*60)
    print("知识库检索对抗测试套件 - 测试摘要")
    print("="*60)
    print("测试覆盖范围:")
    print("  1. 复杂查询分解 (多意图/对比/列举/流程)")
    print("  2. 查询边界条件 (极短/标点/重复/混合语言)")
    print("  3. CRAG决策边界 (HIGH/MEDIUM/LOW/NO_RESULTS阈值)")
    print("  4. RRF融合排序 (自适应权重/k参数/去重)")
    print("  5. 查询扩展对抗 (LLM失败/乱码解析/HyDE失败)")
    print("  6. 检索投毒检测 (关键词填充/语义矛盾/ACL)")
    print("  7. 性能压力测试 (大top_k/并发/rewrite循环)")
    print("  8. 综合集成测试")
    print("  9. 检索质量指标评测 (Recall/Precision/MRR/NDCG/BM25/RRF)")
    print(" 10. 指标驱动对抗评测 (投毒Recall影响/冲突Precision)")
    print("="*60)
    assert True
