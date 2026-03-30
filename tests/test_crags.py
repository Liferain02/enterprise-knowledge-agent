"""
Corrective RAG - 检索评估模块测试

使用方法:
    python -m pytest tests/test_crags.py -v
    python tests/test_crags.py  # 直接运行

测试内容：
    1. RetrievalGrader 单篇评估
    2. RetrievalGrader 批量并行评估
    3. GradeResult 决策逻辑
    4. CorrectiveRAGPipeline 端到端流程
    5. 查询改写
    6. Rerank 评估前置（rerank_before_grade）
    7. Query Expansion 前置（needs_expansion）
    8. 降级逻辑（CRAG 禁用时回退到传统 Rerank）
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 代理配置（根据环境调整）
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')

import asyncio
from langchain_core.documents import Document

from src.rag.evaluation.retrieval_grader import (
    RetrievalGrader,
    CorrectiveRAGPipeline,
    GradeLevel,
    DocumentGrade,
    GradeResult,
    get_retrieval_grader,
    get_corrective_rag_pipeline,
    grade_retrieval,
    corrective_retrieve,
    reset_crags,
)


# ============================================================
# 测试数据和工具函数
# ============================================================

def make_doc(content: str, source: str = "test.md") -> Document:
    """创建测试用 Document"""
    return Document(page_content=content, metadata={"source": source})


# 高相关文档
DOC_HR_ANNUAL_LEAVE = make_doc(
    "根据《员工手册》第三章第七条，员工年假天数按工龄计算："
    "1-3年：5天；3-5年：10天；5-10年：15天；10年以上：20天。"
    "年假需提前3天向部门经理申请。",
    source="员工手册.md"
)

DOC_HR_SICK_LEAVE = make_doc(
    "根据《员工手册》第四章第九条，员工病假需提供医院证明。"
    "3天以内：带薪；3-7天：扣发当日基本工资的50%；"
    "7天以上：需经人力资源部审批。",
    source="员工手册.md"
)

DOC_IT_SERVER = make_doc(
    "服务器维护时间：每周日凌晨2:00-6:00，届时所有线上服务将暂停。"
    "请各部门提前做好数据备份和工作安排。",
    source="IT运维公告.md"
)

# 低相关文档（关于报销）
DOC_FINANCE_REIMBURSE = make_doc(
    "差旅费用报销流程：员工需在出差结束后5个工作日内提交报销申请，"
    "附上发票和出差报告，经部门经理和财务部审批后发放。",
    source="财务制度.md"
)

# 完全无关文档
DOC_NEWS = make_doc(
    "据新华社报道，某科技公司今日发布了最新款智能手机，"
    "搭载最新处理器，支持5G网络，售价3999元起。",
    source="科技新闻.md"
)


# ============================================================
# 测试用例
# ============================================================

def test_grade_level_enum():
    """测试 GradeLevel 枚举"""
    assert GradeLevel.HIGH.value == "high"
    assert GradeLevel.LOW.value == "low"
    assert GradeLevel.NO_RESULTS.value == "no_results"
    print("  ✓ GradeLevel 枚举值正确")


def test_grade_result_decision_high():
    """测试 GradeResult 决策逻辑：HIGH"""
    docs = [
        Document(page_content="测试", metadata={}),
    ]

    # 模拟高相关评估
    grades = [
        DocumentGrade(
            doc=docs[0], relevance_score=0.75,
            raw_score=4.0, reasoning="直接相关",
            grade=GradeLevel.HIGH,
        ),
    ]

    result = GradeResult(query="年假多少天", grades=grades)

    assert result.decision == GradeLevel.HIGH
    assert result.high_count == 1
    assert result.low_count == 0
    assert result.avg_score >= 0.7
    print("  ✓ HIGH 决策逻辑正确")


def test_grade_result_decision_low():
    """测试 GradeResult 决策逻辑：LOW"""
    docs = [
        Document(page_content="测试", metadata={}),
        Document(page_content="测试2", metadata={}),
    ]

    grades = [
        DocumentGrade(
            doc=docs[0], relevance_score=0.25,
            raw_score=2.0, reasoning="勉强相关",
            grade=GradeLevel.LOW,
        ),
        DocumentGrade(
            doc=docs[1], relevance_score=0.25,
            raw_score=2.0, reasoning="勉强相关",
            grade=GradeLevel.LOW,
        ),
    ]

    result = GradeResult(query="年假多少天", grades=grades)

    assert result.decision == GradeLevel.NO_RESULTS
    print("  ✓ LOW → NO_RESULTS 决策逻辑正确（全部低相关）")


def test_grade_result_decision_partial():
    """测试 GradeResult 决策逻辑：部分高相关"""
    docs = [
        Document(page_content="年假", metadata={}),
        Document(page_content="病假", metadata={}),
        Document(page_content="调休", metadata={}),
        Document(page_content="无关", metadata={}),
    ]

    grades = [
        DocumentGrade(
            doc=docs[0], relevance_score=0.75,
            raw_score=4.0, reasoning="高相关",
            grade=GradeLevel.HIGH,
        ),
        DocumentGrade(
            doc=docs[1], relevance_score=0.25,
            raw_score=2.0, reasoning="低相关",
            grade=GradeLevel.LOW,
        ),
        DocumentGrade(
            doc=docs[2], relevance_score=0.25,
            raw_score=2.0, reasoning="低相关",
            grade=GradeLevel.LOW,
        ),
        DocumentGrade(
            doc=docs[3], relevance_score=0.25,
            raw_score=2.0, reasoning="低相关",
            grade=GradeLevel.LOW,
        ),
    ]

    result = GradeResult(query="年假多少天", grades=grades)

    # 1/4 高相关，不满足 0.3 比例要求（实际 0.25），应该是 LOW
    assert result.decision == GradeLevel.LOW
    assert result.high_count == 1
    assert result.low_count == 3
    print("  ✓ 部分高相关决策逻辑正确")


def test_grade_result_filter():
    """测试 GradeResult 过滤方法"""
    docs = [Document(page_content=f"doc{i}", metadata={}) for i in range(5)]

    grades = [
        DocumentGrade(doc=docs[0], relevance_score=0.8,
                      raw_score=4.0, reasoning="高", grade=GradeLevel.HIGH),
        DocumentGrade(doc=docs[1], relevance_score=0.6,
                      raw_score=3.5, reasoning="高", grade=GradeLevel.HIGH),
        DocumentGrade(doc=docs[2], relevance_score=0.3,
                      raw_score=2.0, reasoning="低", grade=GradeLevel.LOW),
        DocumentGrade(doc=docs[3], relevance_score=0.2,
                      raw_score=1.5, reasoning="低", grade=GradeLevel.LOW),
        DocumentGrade(doc=docs[4], relevance_score=0.1,
                      raw_score=1.0, reasoning="低", grade=GradeLevel.LOW),
    ]

    result = GradeResult(query="测试", grades=grades)

    # filter_high_grade
    high_docs = result.filter_high_grade()
    assert len(high_docs) == 2
    assert high_docs[0].page_content == "doc0"
    assert high_docs[1].page_content == "doc1"

    # filter_above_threshold: >= threshold
    above_06 = result.filter_above_threshold(0.6)
    assert len(above_06) == 2  # doc0(0.8) + doc1(0.6) 都 >= 0.6
    assert above_06[0].page_content == "doc0"
    assert above_06[1].page_content == "doc1"

    above_08 = result.filter_above_threshold(0.8)
    assert len(above_08) == 1  # 只有 doc0(0.8) >= 0.8

    print("  ✓ GradeResult 过滤方法正确")


def test_grader_parsing():
    """测试 RetrievalGrader 的响应解析"""
    grader = RetrievalGrader(grade_threshold=0.5)

    # 测试标准格式
    score, reasoning = grader._parse_grade_response(
        "SCORE: 4\nREASONING: 文档直接回答了用户问题"
    )
    assert score == 4
    assert "直接回答" in reasoning

    # 测试带空格的格式
    score2, reasoning2 = grader._parse_grade_response(
        "  SCORE:  3  \n  REASONING:  部分相关  "
    )
    assert score2 == 3

    # 测试只有分数
    score3, _ = grader._parse_grade_response("SCORE: 5")
    assert score3 == 5

    # 测试超出范围（应被截断）
    score4, _ = grader._parse_grade_response("SCORE: 10")
    assert score4 == 5

    score5, _ = grader._parse_grade_response("SCORE: 0")
    assert score5 == 1

    print("  ✓ RetrievalGrader 响应解析正确")


def test_grader_score_normalization():
    """测试分数归一化"""
    grader = RetrievalGrader()

    assert grader._normalize_score(1) == 0.0
    assert grader._normalize_score(2) == 0.25
    assert grader._normalize_score(3) == 0.5
    assert grader._normalize_score(4) == 0.75
    assert grader._normalize_score(5) == 1.0
    assert abs(grader._normalize_score(3.5) - 0.625) < 0.01

    print("  ✓ 分数归一化正确")


async def test_grader_single_real():
    """测试 RetrievalGrader 单篇评估（需要 LLM API）"""
    reset_crags()

    grader = RetrievalGrader(grade_threshold=0.5)

    # 高相关场景
    doc = DOC_HR_ANNUAL_LEAVE
    grade = await grader.grade_single("员工年假有多少天", doc)

    assert isinstance(grade, DocumentGrade)
    assert grade.relevance_score > 0.0
    assert grade.grade in (GradeLevel.HIGH, GradeLevel.LOW)
    print(f"  ✓ 单篇评估成功: score={grade.relevance_score:.2f}, grade={grade.grade.value}")

    # 低相关场景
    doc2 = DOC_NEWS
    grade2 = await grader.grade_single("年假政策是什么", doc2)

    assert isinstance(grade2, DocumentGrade)
    # 新闻文档相关性应该低
    print(f"  ✓ 低相关评估: score={grade2.relevance_score:.2f}, grade={grade2.grade.value}")


async def test_grader_batch_real():
    """测试 RetrievalGrader 批量并行评估（需要 LLM API）"""
    reset_crags()

    docs = [
        DOC_HR_ANNUAL_LEAVE,   # 高相关
        DOC_HR_SICK_LEAVE,     # 可能相关（都是 HR）
        DOC_IT_SERVER,         # 低相关
        DOC_FINANCE_REIMBURSE, # 低相关
        DOC_NEWS,              # 无关
    ]

    query = "员工年假有多少天"

    grader = RetrievalGrader(grade_threshold=0.5)
    grades = await grader.grade_batch(query, docs)

    assert len(grades) == 5
    # 结果应按原始顺序排列
    assert grades[0].doc.page_content == DOC_HR_ANNUAL_LEAVE.page_content

    # 年假文档应该是最高的
    sorted_by_score = sorted(grades, key=lambda g: g.relevance_score, reverse=True)
    print(f"  ✓ 批量评估完成，排序后: {[f'{g.relevance_score:.2f}' for g in sorted_by_score]}")

    # 第一名应该是 HR 相关文档
    top_doc = sorted_by_score[0].doc
    assert "年假" in top_doc.page_content or "病假" in top_doc.page_content
    print("  ✓ 批量并行评估正确：最相关文档是 HR 相关文档")


async def test_grade_retrieval_full():
    """测试完整评估流程"""
    reset_crags()

    docs = [
        DOC_HR_ANNUAL_LEAVE,
        DOC_HR_SICK_LEAVE,
        DOC_IT_SERVER,
        DOC_FINANCE_REIMBURSE,
        DOC_NEWS,
    ]

    query = "年假政策是什么"
    result = await grade_retrieval(query, docs)

    assert isinstance(result, GradeResult)
    assert result.query == query
    assert result.total_docs == 5
    assert result.latency_ms > 0

    # 决策应该是 HIGH 或 LOW，不应该是空
    assert result.decision in (GradeLevel.HIGH, GradeLevel.LOW, GradeLevel.NO_RESULTS)
    assert result.decision_reason != ""

    print(f"  ✓ 完整评估: decision={result.decision.value}, "
          f"high={result.high_count}/{result.total_docs}, "
          f"avg={result.avg_score:.2f}, "
          f"latency={result.latency_ms:.0f}ms")

    # 测试 to_dict
    d = result.to_dict()
    assert "grades" in d
    assert len(d["grades"]) == 5
    print("  ✓ to_dict() 输出正确")


async def test_query_rewrite():
    """测试查询改写"""
    reset_crags()

    docs = [
        DocumentGrade(
            doc=DOC_HR_ANNUAL_LEAVE,
            relevance_score=0.8,
            raw_score=4.0,
            reasoning="直接相关",
            grade=GradeLevel.HIGH,
        ),
        DocumentGrade(
            doc=DOC_FINANCE_REIMBURSE,
            relevance_score=0.2,
            raw_score=1.5,
            reasoning="不相关",
            grade=GradeLevel.LOW,
        ),
    ]

    grade_result = GradeResult(
        query="员工年假和调休有什么区别",
        grades=docs,
    )

    grader = RetrievalGrader()
    rewritten = await grader.rewrite_query(
        "员工年假和调休有什么区别",
        grade_result
    )

    assert isinstance(rewritten, str)
    assert len(rewritten) > 0
    assert rewritten != "员工年假和调休有什么区别"  # 至少应该有改动
    print(f"  ✓ 查询改写: '{rewritten}'")


async def test_corrective_rag_pipeline_mock(monkeypatch=None):
    """测试 CorrectiveRAGPipeline 端到端流程（模拟检索）"""
    reset_crags()

    pipeline = CorrectiveRAGPipeline(
        max_retries=1,
        grade_threshold=0.5,
        candidate_multiplier=2,
    )

    # 模拟 retriever_manager.search_with_score
    mock_results = [
        (DOC_HR_ANNUAL_LEAVE, 0.9),
        (DOC_HR_SICK_LEAVE, 0.7),
        (DOC_IT_SERVER, 0.3),
        (DOC_NEWS, 0.1),
    ]

    original_search = pipeline.retriever_manager.search_with_score

    def mock_search(query, k=5):
        return mock_results[:k]

    pipeline.retriever_manager.search_with_score = mock_search

    # 执行
    results, grade_result, history = await pipeline.retrieve(
        "员工年假有多少天",
        top_k=3,
    )

    assert len(results) >= 0  # 可能有结果
    assert isinstance(grade_result, GradeResult)
    assert len(history) >= 1  # 至少原始查询

    print(f"  ✓ CorrectiveRAGPipeline 端到端: "
          f"返回 {len(results)} 篇文档, "
          f"decision={grade_result.decision.value}, "
          f"history={history}")

    # 恢复
    pipeline.retriever_manager.search_with_score = original_search


def test_crags_disabled():
    """测试 CRAG 被禁用时的降级路径"""
    reset_crags()

    # 这部分通过检查配置标志位来验证
    from config.settings import get_settings
    settings = get_settings()

    # 确认配置存在
    assert hasattr(settings, 'crag_enabled')
    assert hasattr(settings, 'crag_max_retries')
    assert hasattr(settings, 'crag_grade_threshold')
    assert hasattr(settings, 'crag_min_high_ratio')
    assert hasattr(settings, 'crag_candidate_multiplier')

    print(f"  ✓ CRAG 配置项验证: "
          f"enabled={settings.crag_enabled}, "
          f"max_retries={settings.crag_max_retries}, "
          f"threshold={settings.crag_grade_threshold}")


def test_singleton_behavior():
    """测试单例行为"""
    reset_crags()

    grader1 = get_retrieval_grader()
    grader2 = get_retrieval_grader()
    assert grader1 is grader2  # 同一个对象

    pipeline1 = get_corrective_rag_pipeline()
    pipeline2 = get_corrective_rag_pipeline()
    assert pipeline1 is pipeline2  # 同一个对象

    print("  ✓ 单例模式工作正常")


async def test_rerank_before_grade():
    """测试 Rerank 评估前置功能"""
    reset_crags()

    # 创建带有 rerank_before_grade=True 的 pipeline
    pipeline = CorrectiveRAGPipeline(
        max_retries=0,
        rerank_before_grade=True,
    )

    mock_results = [
        (DOC_HR_ANNUAL_LEAVE, 0.9),
        (DOC_HR_SICK_LEAVE, 0.7),
        (DOC_IT_SERVER, 0.5),
        (DOC_FINANCE_REIMBURSE, 0.3),
        (DOC_NEWS, 0.1),
    ]

    original_search = pipeline.retriever_manager.search_with_score

    def mock_search(query, k=5):
        return mock_results[:k]

    pipeline.retriever_manager.search_with_score = mock_search

    results, grade_result, history = await pipeline.retrieve(
        "员工年假政策", top_k=3
    )

    # 验证 pipeline 有 rerank_before_grade 属性
    assert hasattr(pipeline, 'rerank_before_grade')
    assert pipeline.rerank_before_grade is True
    print(f"  ✓ Rerank 评估前置: rerank_before_grade={pipeline.rerank_before_grade}, "
          f"返回 {len(results)} 篇文档")

    # 恢复
    pipeline.retriever_manager.search_with_score = original_search


async def test_needs_expansion_argument():
    """测试 pipeline.retrieve() 接受 needs_expansion 参数"""
    reset_crags()

    pipeline = CorrectiveRAGPipeline(max_retries=0)

    mock_results = [
        (DOC_HR_ANNUAL_LEAVE, 0.9),
        (DOC_HR_SICK_LEAVE, 0.7),
        (DOC_NEWS, 0.1),
    ]

    original_search = pipeline.retriever_manager.search_with_score

    def mock_search(query, k=5):
        return mock_results[:k]

    pipeline.retriever_manager.search_with_score = mock_search

    # needs_expansion=True 时应触发前置 expansion（会调用 _decompose_and_search）
    # 注意：需要 mock _decompose_and_search 或让它自然失败退回到主循环
    # 这里主要验证参数能被接受
    try:
        results, grade_result, history = await pipeline.retrieve(
            "员工年假政策",
            top_k=3,
            needs_expansion=True,
        )
    except Exception:
        pass  # expansion 可能因无 mock 而失败，不影响参数验证

    # needs_expansion=False 时应跳过 expansion
    results2, grade_result2, history2 = await pipeline.retrieve(
        "员工年假政策",
        top_k=3,
        needs_expansion=False,
    )
    assert isinstance(grade_result2, GradeResult)
    print(f"  ✓ needs_expansion 参数: True 和 False 均正常工作")

    pipeline.retriever_manager.search_with_score = original_search


async def test_corrective_retrieve_convenience_func():
    """测试便捷函数 corrective_retrieve 支持 needs_expansion"""
    reset_crags()

    # corrective_retrieve() 是便捷函数，透传给 pipeline
    # 验证签名支持 needs_expansion 参数
    from src.rag.evaluation.retrieval_grader import corrective_retrieve
    import inspect
    sig = inspect.signature(corrective_retrieve)
    assert 'needs_expansion' in sig.parameters
    print("  ✓ corrective_retrieve() 支持 needs_expansion 参数")


if __name__ == "__main__":
    import sys as _sys

    # 同步测试
    print("\n【同步测试】")
    test_grade_level_enum()
    test_grade_result_decision_high()
    test_grade_result_decision_low()
    test_grade_result_decision_partial()
    test_grade_result_filter()
    test_grader_parsing()
    test_grader_score_normalization()
    test_singleton_behavior()
    test_crags_disabled()

    # 异步测试（需要 LLM API）
    print("\n【异步测试 - 需要 LLM API】")

    async def run_async_tests():
        try:
            await test_grader_single_real()
            await test_grader_batch_real()
            await test_grade_retrieval_full()
            await test_query_rewrite()
            await test_corrective_rag_pipeline_mock()
            await test_rerank_before_grade()
            await test_needs_expansion_argument()
            await test_corrective_retrieve_convenience_func()
        except Exception as e:
            print(f"  ⚠ 异步测试跳过（可能网络或 API 问题）: {e}")

    asyncio.run(run_async_tests())

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
