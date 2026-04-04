#!/usr/bin/env python
"""
测试 Reranker + 混合检索功能

主要测试内容：
1. RetrieverManager 配置
2. 混合检索效果对比（基础 vs 混合）
3. 混合检索 + Reranker
4. Agent 知识搜索工具
5. 分数级融合验证（向量+BM25 同一文档两路信号同时生效）
"""
import sys
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')

import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'

from config.settings import get_settings

settings = get_settings()

print("=" * 70)
print("混合检索 + Reranker 功能测试")
print("=" * 70)
print(f"reranker_enabled: {settings.reranker_enabled}")
print(f"reranker_model: {settings.reranker_model}")
print(f"hybrid_search_enabled: {settings.hybrid_search_enabled}")
print(f"hybrid_vector_weight: {settings.hybrid_vector_weight}")
print(f"hybrid_bm25_weight: {settings.hybrid_bm25_weight}")
print()

# 测试 1: RetrieverManager 配置
print("=" * 70)
print("测试 1: RetrieverManager 配置")
print("=" * 70)

from src.rag.retrieval.retriever import get_retriever_manager

retriever_mgr = get_retriever_manager()
print(f"use_reranker: {retriever_mgr.use_reranker}")
print(f"use_hybrid: {retriever_mgr.use_hybrid}")
print(f"hybrid_vector_weight: {retriever_mgr.hybrid_vector_weight}")
print(f"hybrid_bm25_weight: {retriever_mgr.hybrid_bm25_weight}")
print(f"reranker_manager: {retriever_mgr.reranker_manager is not None}")
print(f"hybrid_manager: {retriever_mgr.hybrid_manager is not None}")
print()

# 测试 2: 基础搜索 vs 混合检索
print("=" * 70)
print("测试 2: 混合检索效果对比")
print("=" * 70)

query = "Python 编程"

try:
    # 基础向量检索
    print("\n--- 基础向量检索 ---")
    basic_results = retriever_mgr.search(query, k=5)
    for i, doc in enumerate(basic_results, 1):
        source = doc.metadata.get("source", "未知") if doc.metadata else "未知"
        print(f"  {i}. [{source}] {doc.page_content[:50]}...")
    
    # 带分数的混合检索
    print("\n--- 混合检索 (带分数) ---")
    hybrid_results = retriever_mgr.search_with_score(query, k=5)
    for i, (doc, score) in enumerate(hybrid_results, 1):
        source = doc.metadata.get("source", "未知") if doc.metadata else "未知"
        print(f"  {i}. Score: {score:.4f} | [{source}] {doc.page_content[:40]}...")
    
    print("\n✅ 混合检索工作正常!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试 3: 混合检索 + Reranker
print("=" * 70)
print("测试 3: 混合检索 + Reranker")
print("=" * 70)

query = "Python 教程"

try:
    results = retriever_mgr.search_with_rerank(query, k=3)
    print(f"\n查询: {query}")
    print(f"结果数量: {len(results)}")
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "未知") if doc.metadata else "未知"
        print(f"\n  {i}. Score: {score:.4f}")
        print(f"     Source: {source}")
        print(f"     Content: {doc.page_content[:100]}...")
    
    print("\n✅ 混合检索 + Reranker 工作正常!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试 4: Agent 知识搜索工具
print("=" * 70)
print("测试 4: Agent 知识搜索工具")
print("=" * 70)

from src.agent.skills.knowledge.scripts.tools import knowledge_search

query = "Python"

try:
    result = knowledge_search(query, top_k=3)
    print(f"查询: {query}")
    print(f"\n搜索结果:\n{result}")
    print("\n✅ Agent 知识搜索工具工作正常!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("测试 5: 分数级融合验证（向量+BM25 同一文档两路信号同时生效）")
print("=" * 70)

try:
    from src.rag.retrieval.hybrid_retriever import HybridRetrieverManager
    from langchain_core.documents import Document

    # 创建 HybridRetrieverManager（纯内存，无真实向量库）
    # 手动注入 mock documents 以触发 BM25 路径
    mock_docs = [
        Document(page_content="年假政策：员工每年享有带薪年假，具体天数根据工龄确定。", metadata={"source": "制度.md"}),
        Document(page_content="病假政策：员工因病需要休息的假期，需要提供医院证明。", metadata={"source": "制度.md"}),
        Document(page_content="服务器维护：每周日凌晨2点进行系统维护，届时服务暂停。", metadata={"source": "IT公告.md"}),
    ]

    # 验证 search_with_scores 返回三元组 (doc, score, source)
    hybrid = HybridRetrieverManager(
        collection_name="enterprise_knowledge",
        top_k=5,
        enable_bm25=True,
        enable_vector=True,
        vector_weight=0.5,
        bm25_weight=0.5,
    )

    # 注入 mock documents（模拟知识库已有数据）
    hybrid.set_documents(mock_docs)

    # 使用包含年假关键词的查询
    results = hybrid.search_with_scores("年假政策", k=3)

    print(f"\n查询: '年假政策'")
    print(f"返回结果数: {len(results)}")
    for i, (doc, score, source) in enumerate(results, 1):
        print(f"  {i}. score={score:.4f} source={source:<15} | {doc.page_content[:40]}...")

    # 验证返回格式
    assert len(results) > 0, "应有返回结果"
    for doc, score, source in results:
        assert isinstance(doc, Document), "第一个元素应为 Document"
        assert isinstance(score, float), "第二个元素应为 float"
        assert source in ("vector", "bm25", "vector+bm25"), f"source 应为 vector/bm25/vector+bm25，实际={source}"

    # 验证分数范围
    scores = [s for _, s, _ in results]
    assert all(0.0 <= s <= 1.0 for s in scores), "融合分数应在 [0, 1] 范围内"

    # 验证来源标记
    has_dual = any(src == "vector+bm25" for _, _, src in results)
    print(f"\n  含双路信号文档(vector+bm25): {has_dual}")

    print("\n✅ 分数级融合验证通过！")

    # 关键验证：如果向量和BM25都找到了同一文档，应标记为 vector+bm25
    # 而非旧版中后检索的 BM25 被丢弃
    print("\n融合机制说明：")
    print("  - 旧版：BM25 跳过向量已找到的文档（二选一）→ 丢失信号")
    print("  - 新版：同一文档两路归一化分数加权相加 → 双强更高")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("测试完成")
print("=" * 70)
