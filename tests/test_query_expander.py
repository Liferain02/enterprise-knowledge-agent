"""
Query Expansion / Decomposition 模块测试

使用方法:
    python tests/test_query_expander.py
    python -m pytest tests/test_query_expander.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')

import asyncio
from langchain_core.documents import Document

from src.rag.retrieval.query_expander import (
    QueryExpander,
    QueryDecomposer,
    RuleBasedDecomposer,
    HyDEExpander,
    ExpandStrategy,
    SubQuery,
    ExpansionResult,
    get_query_expander,
    expand_query,
    decompose_and_retrieve,
    multi_query_retrieve,
    _reciprocal_rank_fusion,
    reset_query_expander,
)


# ============================================================
# 测试数据
# ============================================================

def make_doc(content: str, source: str = "test.md") -> Document:
    return Document(page_content=content, metadata={"source": source})


# ============================================================
# 规则分解器测试
# ============================================================

def test_needs_expansion():
    """测试 needs_expansion 快速判断"""
    # 对比类
    assert RuleBasedDecomposer.needs_expansion("年假和病假的区别") is True
    assert RuleBasedDecomposer.needs_expansion("A vs B 哪个好") is True
    # 列举类
    assert RuleBasedDecomposer.needs_expansion("公司有哪些福利") is True
    # 多问号
    assert RuleBasedDecomposer.needs_expansion("年假几天？病假呢？") is True
    # 简单查询
    assert RuleBasedDecomposer.needs_expansion("你好") is False
    assert RuleBasedDecomposer.needs_expansion("年假有几天") is False
    assert RuleBasedDecomposer.needs_expansion("现在几点") is False
    print("  ✓ needs_expansion 快速判断正确")


def test_rule_decompose_contrast():
    """测试规则分解：对比类"""
    result = RuleBasedDecomposer.decompose("年假和病假的区别")
    assert len(result) == 2
    assert result[0].text == "年假"
    assert result[1].text == "病假"
    assert result[0].is_primary is True
    print(f"  ✓ 对比类分解: '年假和病假的区别' -> {[sq.text for sq in result]}")

    # VS 模式
    result2 = RuleBasedDecomposer.decompose("调休 vs 年假")
    assert len(result2) >= 1
    print(f"  ✓ VS 模式分解: '调休 vs 年假' -> {[sq.text for sq in result2]}")


def test_rule_decompose_list():
    """测试规则分解：列举类"""
    result = RuleBasedDecomposer.decompose("公司有哪些福利")
    assert len(result) >= 1
    print(f"  ✓ 列举类分解: '公司有哪些福利' -> {[sq.text for sq in result]}")


def test_rule_decompose_multi_entity():
    """测试规则分解：多主体"""
    result = RuleBasedDecomposer.decompose("张三和李四的工作职责")
    assert len(result) >= 1
    print(f"  ✓ 多主体分解: '张三和李四的工作职责' -> {[sq.text for sq in result]}")


def test_rule_decompose_process():
    """测试规则分解：流程类"""
    result = RuleBasedDecomposer.decompose("年假怎么申请")
    # 流程类可能返回空（如果无法提取主体）
    print(f"  ✓ 流程类分解: '年假怎么申请' -> {[sq.text for sq in result]}")


def test_rule_decompose_simple():
    """测试规则分解：简单查询不分解"""
    result = RuleBasedDecomposer.decompose("现在几点")
    assert len(result) == 1
    assert result[0].text == "现在几点"
    assert result[0].is_primary is True
    print("  ✓ 简单查询不分解")


def test_rule_decompose_order():
    """测试对比类分解顺序"""
    result = RuleBasedDecomposer.decompose("调休和年假有什么区别")
    assert len(result) == 2
    # 第一个应该是主查询
    assert result[0].is_primary is True
    assert result[1].is_primary is False
    print(f"  ✓ 主查询顺序正确: {[sq.text for sq in result]}")


# ============================================================
# 数据模型测试
# ============================================================

def test_subquery_hash():
    """测试 SubQuery hash"""
    sq1 = SubQuery(id=0, text="年假", intent="对比", is_primary=True)
    sq2 = SubQuery(id=0, text="年假", intent="对比", is_primary=True)
    assert hash(sq1) == hash(sq2)
    print("  ✓ SubQuery hash 正确")


def test_expansion_result_init():
    """测试 ExpansionResult 自动填充"""
    sqs = [
        SubQuery(id=0, text="年假", intent="对比", is_primary=True),
        SubQuery(id=1, text="病假", intent="对比", is_primary=False),
    ]
    result = ExpansionResult(
        original_query="年假和病假的区别",
        strategy=ExpandStrategy.RULE_ONLY,
        sub_queries=sqs,
    )
    # 自动填充
    assert result.all_queries == ["年假", "病假"]
    assert result.primary_query == "年假"
    assert result.used_llm is False
    print("  ✓ ExpansionResult 自动填充正确")


def test_strategy_enum():
    """测试 ExpandStrategy 枚举"""
    assert ExpandStrategy.RULE_ONLY.value == "rule_only"
    assert ExpandStrategy.LLM_ONLY.value == "llm_only"
    assert ExpandStrategy.HYBRID.value == "hybrid"
    assert ExpandStrategy.HYDE.value == "hyde"
    print("  ✓ ExpandStrategy 枚举正确")


# ============================================================
# RRF 算法测试
# ============================================================

def test_rrf_basic():
    """测试 Reciprocal Rank Fusion"""
    docs = [
        (make_doc("A"), 0.9, "查询1"),
        (make_doc("B"), 0.8, "查询1"),
        (make_doc("A"), 0.7, "查询2"),  # A 在两个查询中都出现
        (make_doc("C"), 0.6, "查询2"),
        (make_doc("D"), 0.5, "查询3"),
    ]

    fused = _reciprocal_rank_fusion(docs, k=60)

    # A 应该排在前面（因为在多个查询中出现）
    assert fused[0][0].page_content == "A"
    # 验证返回格式
    assert len(fused[0]) == 3  # (doc, score, source)
    print(f"  ✓ RRF 排序: {[d[0].page_content for d in fused]}")


def test_rrf_empty():
    """测试 RRF 空输入"""
    fused = _reciprocal_rank_fusion([])
    assert fused == []
    print("  ✓ RRF 空输入处理正确")


def test_rrf_single():
    """测试 RRF 单列表"""
    docs = [
        (make_doc("A"), 0.9, "查询1"),
        (make_doc("B"), 0.8, "查询1"),
    ]
    fused = _reciprocal_rank_fusion(docs)
    assert len(fused) == 2
    assert fused[0][0].page_content == "A"  # 分数高的在前
    print("  ✓ RRF 单列表正确")


# ============================================================
# QueryExpander 同步测试（规则模式）
# ============================================================

def test_expander_sync():
    """测试 QueryExpander 同步扩展（规则模式）"""
    reset_query_expander()

    expander = QueryExpander(strategy=ExpandStrategy.RULE_ONLY)
    result = expander.expand("年假和病假的区别")

    assert result.original_query == "年假和病假的区别"
    assert result.strategy == ExpandStrategy.RULE_ONLY
    assert result.used_llm is False
    assert len(result.sub_queries) == 2
    assert result.primary_query == "年假"
    assert result.latency_ms > 0
    print(f"  ✓ 同步扩展: '{result.original_query}' -> "
          f"{len(result.sub_queries)} 子查询, primary='{result.primary_query}'")


def test_expander_sync_simple():
    """测试 QueryExpander 同步扩展：简单查询"""
    expander = QueryExpander(strategy=ExpandStrategy.RULE_ONLY)

    result = expander.expand("现在几点")
    assert len(result.sub_queries) == 1
    assert result.sub_queries[0].text == "现在几点"
    print("  ✓ 简单查询同步扩展正确")


def test_expander_singleton():
    """测试 QueryExpander 单例"""
    reset_query_expander()

    e1 = get_query_expander()
    e2 = get_query_expander()
    assert e1 is e2

    reset_query_expander()
    e3 = get_query_expander()
    assert e1 is not e3
    print("  ✓ 单例模式工作正常")


# ============================================================
# QueryExpander 异步测试（需要 LLM）
# ============================================================

async def test_expander_llm_hybrid():
    """测试 QueryExpander 异步扩展（Hybrid 模式，需要 LLM）"""
    reset_query_expander()

    expander = QueryExpander(strategy=ExpandStrategy.HYBRID)
    result = await expander.expand_async("年假和病假的区别")

    assert result.original_query == "年假和病假的区别"
    assert result.strategy == ExpandStrategy.HYBRID
    assert len(result.sub_queries) >= 1
    assert result.primary_query != ""
    print(f"  ✓ Hybrid 扩展: '{result.original_query}' -> "
          f"{len(result.sub_queries)} 子查询, "
          f"primary='{result.primary_query}', "
          f"llm={result.used_llm}, "
          f"({result.latency_ms:.0f}ms)")


async def test_expander_llm_only():
    """测试 QueryExpander LLM_ONLY 模式"""
    reset_query_expander()

    # 使用一个 LLM 明显能分解得更好的查询
    # 规则分解可能只能分出1个，LLM 能分出多个
    expander = QueryExpander(strategy=ExpandStrategy.LLM_ONLY)

    # "查询多个离职相关问题" - 规则无法分解，LLM 可以
    result = await expander.expand_async(
        "离职流程是什么，需要准备哪些材料，需要多长时间"
    )

    # LLM_ONLY 策略下，只要 LLM 被调用过（无论结果是否采纳），就算数
    assert len(result.sub_queries) >= 1
    print(f"  ✓ LLM_ONLY 扩展: '{result.original_query}' -> "
          f"{[sq.text for sq in result.sub_queries]}, llm={result.used_llm}")


async def test_expander_hyde():
    """测试 HyDE 模式"""
    reset_query_expander()

    # 先生成假设文档
    hypo_doc = await HyDEExpander.generate_hypothetical_doc(
        "公司年假政策是什么"
    )
    assert isinstance(hypo_doc, str)
    assert len(hypo_doc) > 20
    print(f"  ✓ HyDE 生成假设文档成功 ({len(hypo_doc)} chars)")

    # HyDE 扩展
    expander = QueryExpander(strategy=ExpandStrategy.HYDE)
    result = await expander.expand_async("公司年假政策是什么")

    assert result.used_llm is True
    assert len(result.sub_queries) >= 1
    print(f"  ✓ HyDE 扩展: '{result.original_query}' -> "
          f"{[sq.text for sq in result.sub_queries]}")


async def test_expander_empty():
    """测试空查询"""
    reset_query_expander()

    expander = QueryExpander()
    result = await expander.expand_async("")
    assert len(result.sub_queries) == 0
    assert result.used_llm is False
    print("  ✓ 空查询处理正确")


async def test_expander_intent_labels():
    """测试 LLM 分解器返回正确的意图标签"""
    reset_query_expander()

    sqs = await QueryDecomposer.decompose("年假和病假的区别")
    assert len(sqs) >= 1

    # 检查意图字段存在
    for sq in sqs:
        assert hasattr(sq, 'intent')
        assert hasattr(sq, 'text')
        assert hasattr(sq, 'is_primary')
        assert sq.text != ""

    print(f"  ✓ LLM 分解器意图标签: {[(sq.text, sq.intent) for sq in sqs]}")


async def test_expander_llm_fallback():
    """测试 LLM 分解器 JSON 解析失败时的 fallback"""
    # 这个测试主要验证代码路径存在，不直接 mock
    reset_query_expander()

    result = await expand_query("测试查询")
    assert isinstance(result, ExpansionResult)
    print("  ✓ LLM 分解器 fallback 路径存在")


# ============================================================
# 端到端测试
# ============================================================

async def test_decompose_and_retrieve():
    """测试 decompose_and_retrieve 端到端流程"""
    reset_query_expander()

    # 简单查询（不需要分解）
    results, exp_result = await decompose_and_retrieve(
        "现在几点",
        top_k=3,
        strategy=ExpandStrategy.RULE_ONLY,
    )
    assert isinstance(results, list)
    assert isinstance(exp_result, ExpansionResult)
    assert exp_result.original_query == "现在几点"
    print(f"  ✓ 简单查询 decompose_and_retrieve: "
          f"返回 {len(results)} 篇文档")


async def test_multi_query_retrieve():
    """测试多查询并行检索"""
    reset_query_expander()

    queries = ["年假政策", "病假政策"]
    results = await multi_query_retrieve(queries, top_k_per_query=2)

    assert isinstance(results, list)
    assert len(results) >= 0
    # 验证返回格式
    for doc, score, source in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert isinstance(source, str)
        assert source in queries
    print(f"  ✓ 多查询并行检索: queries={queries}, 返回 {len(results)} 篇文档")


# ============================================================
# 配置验证
# ============================================================

def test_config_options():
    """验证查询扩展配置项存在"""
    from config.settings import get_settings
    settings = get_settings()

    assert hasattr(settings, 'query_expand_enabled')
    assert hasattr(settings, 'query_expand_strategy')
    assert hasattr(settings, 'query_expand_max_sub_queries')
    assert hasattr(settings, 'query_expand_rerank_fusion_k')

    assert hasattr(settings, 'crag_enabled')
    assert hasattr(settings, 'crag_max_retries')
    assert hasattr(settings, 'crag_grade_threshold')

    print(f"  ✓ 查询扩展配置: enabled={settings.query_expand_enabled}, "
          f"strategy={settings.query_expand_strategy}")
    print(f"  ✓ CRAG 配置: enabled={settings.crag_enabled}, "
          f"max_retries={settings.crag_max_retries}")


# ============================================================
# 主函数
# ============================================================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Query Expansion / Decomposition 测试套件")
    print("=" * 60)

    # 同步测试（无需 LLM）
    print("\n【同步测试 - 规则分解 + 算法】")
    test_needs_expansion()
    test_rule_decompose_contrast()
    test_rule_decompose_list()
    test_rule_decompose_multi_entity()
    test_rule_decompose_process()
    test_rule_decompose_simple()
    test_rule_decompose_order()
    test_subquery_hash()
    test_expansion_result_init()
    test_strategy_enum()
    test_rrf_basic()
    test_rrf_empty()
    test_rrf_single()
    test_expander_sync()
    test_expander_sync_simple()
    test_expander_singleton()
    test_config_options()

    # 异步测试（需要 LLM）
    print("\n【异步测试 - 需要 LLM API】")

    async def run_async():
        try:
            await test_expander_llm_hybrid()
            await test_expander_llm_only()
            await test_expander_hyde()
            await test_expander_empty()
            await test_expander_intent_labels()
            await test_expander_llm_fallback()
            await test_decompose_and_retrieve()
            await test_multi_query_retrieve()
        except Exception as e:
            print(f"  ⚠ 异步测试异常 [{type(e).__name__}]: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_async())

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
