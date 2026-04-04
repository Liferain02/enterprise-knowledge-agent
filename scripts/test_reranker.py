#!/usr/bin/env python
"""测试 Reranker API 调用"""
import sys
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')

import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'

from config.settings import get_settings

settings = get_settings()

print("=" * 60)
print("Reranker 配置检查")
print("=" * 60)
print(f"reranker_enabled: {settings.reranker_enabled}")
print(f"reranker_model: {settings.reranker_model}")
print(f"reranker_provider: {settings.reranker_provider}")
print(f"reranker_top_n: {settings.reranker_top_n}")
print(f"reranker_threshold: {settings.reranker_threshold}")
print()

# 直接测试 dashscope API
print("=" * 60)
print("直接测试 dashscope TextReRank API")
print("=" * 60)

import dashscope
from dashscope import TextReRank
from dotenv import load_dotenv

load_dotenv('config/.env')
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

docs = [
    "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
    "量子计算是计算科学的一个前沿领域",
    "预训练语言模型的发展给文本排序模型带来了新的进展",
    "Python是一种高级编程语言",
    "机器学习是人工智能的分支"
]

query = "什么是文本排序模型"

try:
    resp = TextReRank.call(
        model='gte-rerank-v2',
        query=query,
        documents=docs,
        top_n=3,
        return_documents=True
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print("✅ Reranker API 调用成功!")
        for r in resp.output['results']:
            idx = r['index']
            score = r['relevance_score']
            print(f"  {score:.4f} - {docs[idx][:40]}...")
    else:
        print(f"❌ Error: {resp.message}")
except Exception as e:
    print(f"❌ Exception: {e}")
print()

# 测试 RAG Pipeline 的 reranker
print("=" * 60)
print("RAG Pipeline Reranker 测试")
print("=" * 60)

from rag.reranker import get_reranker_manager
from langchain_core.documents import Document

test_docs = [
    Document(page_content="这是关于Python编程的文档。Python是一种高级编程语言。", metadata={"source": "test1"}),
    Document(page_content="关于Java编程的内容。Java是一种面向对象的编程语言。", metadata={"source": "test2"}),
    Document(page_content="机器学习是人工智能的一个分支。深度学习是机器学习的子领域。", metadata={"source": "test3"}),
    Document(page_content="Python也常用于数据科学和机器学习领域。", metadata={"source": "test4"}),
    Document(page_content="Web开发可以使用Python的Django或Flask框架。", metadata={"source": "test5"}),
]

query = "Python编程语言"

try:
    reranker_mgr = get_reranker_manager(
        provider='qwen',
        reranker_model='gte-rerank-v2',
        top_n=3,
        score_threshold=0.3
    )
    results = reranker_mgr.rerank(query, test_docs, top_n=3)
    
    print(f"查询: {query}")
    print(f"Rerank 结果 (Top 3):")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. Score: {score:.4f} - {doc.page_content[:40]}...")
    print("✅ Reranker 工作正常!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
