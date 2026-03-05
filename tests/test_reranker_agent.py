#!/usr/bin/env python
"""
测试 Reranker 功能是否在 Agent 系统中生效
"""
import sys
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')

import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'

from config.settings import get_settings

settings = get_settings()

print("=" * 60)
print("Reranker 功能测试")
print("=" * 60)
print(f"reranker_enabled: {settings.reranker_enabled}")
print(f"reranker_model: {settings.reranker_model}")
print(f"reranker_provider: {settings.reranker_provider}")
print()

# 测试 1: 直接测试 RetrieverManager 的 search_with_rerank
print("=" * 60)
print("测试 1: RetrieverManager search_with_rerank")
print("=" * 60)

from rag.retriever import get_retriever_manager

retriever_mgr = get_retriever_manager()
print(f"use_reranker: {retriever_mgr.use_reranker}")
print(f"reranker_manager: {retriever_mgr.reranker_manager is not None}")

# 构造一个查询，确保能返回多个结果
query = "Python 编程"

try:
    # 测试不带 reranker 的搜索
    results_no_rerank = retriever_mgr.search(query, k=5)
    print(f"\n原始检索结果数量: {len(results_no_rerank)}")
    
    # 测试带 reranker 的搜索
    if retriever_mgr.use_reranker and retriever_mgr.reranker_manager:
        results_with_rerank = retriever_mgr.search_with_rerank(query, k=3)
        print(f"Rerank 后结果数量: {len(results_with_rerank)}")
        
        print("\nRerank 结果详情:")
        for i, (doc, score) in enumerate(results_with_rerank, 1):
            source = doc.metadata.get("source", "未知") if doc.metadata else "未知"
            print(f"  {i}. Score: {score:.4f}")
            print(f"     Source: {source}")
            print(f"     Content: {doc.page_content[:80]}...")
        print("\n✅ RetrieverManager Reranker 工作正常!")
    else:
        print("\n❌ Reranker 未启用")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试 2: 模拟 Agent 调用知识搜索
print("=" * 60)
print("测试 2: Agent 知识搜索工具")
print("=" * 60)

from agents.skills.knowledge.scripts.tools import knowledge_search

query = "什么是 Python"

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

# 测试 3: 对比测试 - 验证 Reranker 确实改变了排序
print("=" * 60)
print("测试 3: Reranker 排序效果对比")
print("=" * 60)

# 构造一个明确的查询，让不相关文档排在前面
query = "Python 教程"

try:
    # 获取原始结果
    raw_results = retriever_mgr.search(query, k=5)
    print(f"\n原始检索 Top 5:")
    for i, doc in enumerate(raw_results, 1):
        source = doc.metadata.get("source", "未知") if doc.metadata else "未知"
        print(f"  {i}. [{source}] {doc.page_content[:50]}...")
    
    # 获取 Rerank 后的结果
    reranked_results = retriever_mgr.search_with_rerank(query, k=5)
    print(f"\nRerank 后 Top 5:")
    for i, (doc, score) in enumerate(reranked_results, 1):
        source = doc.metadata.get("source", "未知") if doc.metadata else "未知"
        print(f"  {i}. [{score:.4f}] {source}")
        print(f"     {doc.page_content[:50]}...")
    
    # 检查排序是否发生变化
    raw_sources = [doc.metadata.get("source", "") for doc in raw_results[:3]]
    reranked_sources = [doc.metadata.get("source", "") for doc, score in reranked_results[:3]]
    
    if raw_sources != reranked_sources:
        print("\n✅ Reranker 确实改变了排序!")
    else:
        print("\n⚠️ 排序未变化，可能文档相关性差异明显")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("测试完成")
print("=" * 60)
