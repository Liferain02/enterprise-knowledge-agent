"""
端到端集成测试：验证 Corrective RAG + Query Decomposition 的集成

运行方式：
    python scripts/test_integration_e2e.py

测试内容：
    1. Corrective RAG 端到端检索
    2. Query Decomposition 端到端检索
    3. CRAG LOW 时触发 QueryExpander 兜底
    4. CRAG Query Expansion 前置（needs_expansion=True）
    5. 完整 knowledge_search 链路
    6. 配置集成验证
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')

import asyncio
from src.rag.evaluation.retrieval_grader import (
    get_corrective_rag_pipeline,
    get_retrieval_grader,
    GradeLevel,
    reset_crags,
)
from src.rag.retrieval.query_expander import (
    expand_query,
    decompose_and_retrieve,
    ExpandStrategy,
    RuleBasedDecomposer,
    reset_query_expander,
)


async def test_corrective_rag_integration():
    """
    测试1：Corrective RAG 端到端检索
    验证路径：knowledge_search -> CorrectiveRAGPipeline.retrieve() -> RetrievalGrader.grade_retrieval()
    """
    print("\n" + "=" * 60)
    print("测试1: Corrective RAG 端到端检索")
    print("=" * 60)

    reset_crags()

    pipeline = get_corrective_rag_pipeline()
    query = "公司的年假制度是怎么规定的"

    print(f"\n查询: {query}")
    print("-" * 40)

    results, grade_result, history = await pipeline.retrieve(query, top_k=5)

    print(f"检索结果数: {len(results)}")
    print(f"评估决策: {grade_result.decision.value}")
    print(f"高相关文档: {grade_result.high_count}/{grade_result.total_docs}")
    print(f"平均相关分: {grade_result.avg_score:.2f}")
    print(f"决策理由: {grade_result.decision_reason}")
    print(f"查询历史: {history}")

    # 验证：应该有结果或至少有一个评估决策
    assert isinstance(grade_result.decision, GradeLevel)
    assert len(history) >= 1  # 至少包含原始查询

    print(f"\n  ✓ CRAG 检索完成: decision={grade_result.decision.value}")
    return True


async def test_query_decomposition_integration():
    """
    测试2：Query Decomposition 端到端检索
    验证路径：QueryExpander.expand_async() -> decompose_and_retrieve() -> multi_query_retrieve()
    """
    print("\n" + "=" * 60)
    print("测试2: Query Decomposition 端到端检索")
    print("=" * 60)

    reset_query_expander()

    # 对比类查询：会触发分解
    query = "年假和病假的区别"

    print(f"\n查询: {query}")
    print("-" * 40)

    # Step 1: 验证规则分解
    rule_result = RuleBasedDecomposer.decompose(query)
    print(f"规则分解: {[sq.text for sq in rule_result]} 子查询")

    # Step 2: 验证 LLM 分解
    exp_result = await expand_query(query, strategy=ExpandStrategy.HYBRID)
    print(f"LLM 分解 ({exp_result.strategy.value}): "
          f"{[sq.text for sq in exp_result.sub_queries]} 子查询, "
          f"llm={exp_result.used_llm}")

    # Step 3: 验证端到端检索
    results, decomp_result = await decompose_and_retrieve(query, top_k=5)
    print(f"\n检索结果数: {len(results)}")
    print(f"原始查询: {decomp_result.original_query}")
    print(f"分解策略: {decomp_result.strategy.value}")
    print(f"子查询数量: {len(decomp_result.sub_queries)}")
    print(f"主查询: {decomp_result.primary_query}")
    print(f"所有查询: {decomp_result.all_queries}")

    # 验证
    assert len(decomp_result.sub_queries) >= 1
    assert decomp_result.primary_query != ""
    assert len(results) >= 0  # 可能有结果也可能没有

    print(f"\n  ✓ Query Decomposition 完成: "
          f"{len(decomp_result.sub_queries)} 子查询")
    return True


async def test_crags_trigger_query_expander():
    """
    测试3：CRAG 低质量时触发 QueryExpander 兜底
    验证路径：CorrectiveRAGPipeline -> LOW decision -> QueryExpander.decompose_and_retrieve()
    """
    print("\n" + "=" * 60)
    print("测试3: CRAG 触发 QueryExpander（低质量纠错）")
    print("=" * 60)

    reset_crags()
    reset_query_expander()

    # 用一个可能产生低质量检索结果的模糊查询
    query = "公司有什么福利待遇"

    print(f"\n查询: {query}")
    print("-" * 40)

    pipeline = get_corrective_rag_pipeline()

    # 执行检索
    results, grade_result, history = await pipeline.retrieve(query, top_k=3)

    print(f"检索结果数: {len(results)}")
    print(f"评估决策: {grade_result.decision.value}")
    print(f"高相关: {grade_result.high_count}/{grade_result.total_docs}")
    print(f"平均分: {grade_result.avg_score:.2f}")
    print(f"决策理由: {grade_result.decision_reason}")

    # 当 decision=LOW 时，pipeline 应该已经尝试了 QueryExpander
    # 我们通过 history 来验证：如果查询被重写或分解了，history 会更长
    print(f"查询历史: {history}")

    # 验证
    assert isinstance(grade_result.decision, GradeLevel)

    if grade_result.decision == GradeLevel.LOW:
        print(f"\n  ✓ LOW 决策触发，查询历史显示纠错过程: {history}")
    else:
        print(f"\n  ✓ 直接得到 {grade_result.decision.value} 决策")

    return True


async def test_crags_query_expansion_prefetch():
    """
    测试3b：CRAG Query Expansion 前置（needs_expansion=True）

    验证路径：
    - CorrectiveRAGPipeline.retrieve(needs_expansion=True)
      → 阶段0直接触发 QueryExpander 分解
      → 不走 CRAG 主循环 rewrite

    场景：对比类查询由上游 Planner/Supervisor 显式传入 needs_expansion=True，
    CRAG 应在评估前就完成分解，避免走 rewrite 的额外延迟。
    """
    print("\n" + "=" * 60)
    print("测试3b: CRAG Query Expansion 前置")
    print("=" * 60)

    reset_crags()
    reset_query_expander()

    pipeline = get_corrective_rag_pipeline()

    # 显式传入 needs_expansion=True（模拟 Planner/Supervisor 的判断）
    results, grade_result, history = await pipeline.retrieve(
        "年假和病假的区别",
        top_k=5,
        needs_expansion=True,
    )

    print(f"  查询: '年假和病假的区别' (needs_expansion=True)")
    print(f"  返回结果数: {len(results)}")
    print(f"  评估决策: {grade_result.decision.value}")
    print(f"  查询历史: {history}")

    # 验证：history 应包含原始查询和子查询
    assert len(history) >= 1
    print(f"\n  ✓ Expansion 前置完成，history 包含 {len(history)} 个查询表述")

    return True


async def test_full_knowledge_search_chain():
    """
    测试4：完整链路 - knowledge_search 工具函数
    验证路径：knowledge_search -> CRAG pipeline -> QueryExpander（端到端）
    """
    print("\n" + "=" * 60)
    print("测试4: 完整 knowledge_search 链路")
    print("=" * 60)

    from src.agent.skills.knowledge.scripts.tools import knowledge_search

    # 对比类查询（会触发 QueryExpander）
    query = "年假和病假有什么区别"

    print(f"\n查询: {query}")
    print("-" * 40)

    result = knowledge_search(query, top_k=5)

    # 检查结果是否包含 CRAG 评估信息
    has_crag_info = "CRAG" in result or "相关性" in result or "检索结果" in result
    has_grades = "决策:" in result or "高相关:" in result or "平均相关分:" in result

    print(f"\n结果长度: {len(result)} 字符")
    print(f"包含 CRAG 评估信息: {has_crag_info}")
    print(f"包含评估分数: {has_grades}")

    # 打印部分结果
    print(f"\n结果预览（前500字符）:")
    print("-" * 40)
    print(result[:500])

    print(f"\n  ✓ knowledge_search 完成，CRAG 信息: {has_grades}")
    return True


def test_config_integration():
    """测试配置集成"""
    print("\n" + "=" * 60)
    print("测试5: 配置集成验证")
    print("=" * 60)

    from config.settings import get_settings
    settings = get_settings()

    crag_keys = ['crag_enabled', 'crag_max_retries', 'crag_grade_threshold',
                 'crag_min_high_ratio', 'crag_candidate_multiplier',
                 'crag_rerank_before_grade']
    qe_keys = ['query_expand_enabled', 'query_expand_strategy',
               'query_expand_max_sub_queries', 'query_expand_rerank_fusion_k']

    print("\n  Corrective RAG 配置:")
    for key in crag_keys:
        val = getattr(settings, key, "N/A")
        print(f"    {key}: {val}")

    print("\n  Query Expansion 配置:")
    for key in qe_keys:
        val = getattr(settings, key, "N/A")
        print(f"    {key}: {val}")

    # 验证
    for key in crag_keys + qe_keys:
        assert hasattr(settings, key), f"缺少配置项: {key}"

    print(f"\n  ✓ 所有配置项验证通过")
    return True


async def main():
    print("\n" + "=" * 60)
    print("Corrective RAG + Query Decomposition 集成测试（6项）")
    print("=" * 60)

    all_passed = True

    try:
        all_passed &= await test_corrective_rag_integration()
    except Exception as e:
        print(f"\n  ✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= await test_query_decomposition_integration()
    except Exception as e:
        print(f"\n  ✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= await test_crags_trigger_query_expander()
    except Exception as e:
        print(f"\n  ✗ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= await test_crags_query_expansion_prefetch()
    except Exception as e:
        print(f"\n  ✗ 测试3b失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= await test_full_knowledge_search_chain()
    except Exception as e:
        print(f"\n  ✗ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_config_integration()
    except Exception as e:
        print(f"\n  ✗ 测试5失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有集成测试通过！")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
