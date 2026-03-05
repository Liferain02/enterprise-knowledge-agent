#!/usr/bin/env python
"""
测试 Reranker + 混合检索 功能
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

from rag.retriever import get_retriever_manager

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

from agents.skills.knowledge.scripts.tools import knowledge_search

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
print("测试完成")
print("=" * 70)
