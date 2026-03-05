#!/usr/bin/env python
"""
测试增强后的分块策略
验证：
1. 元数据增强（文档标题、章节信息）
2. 语义分块功能
"""
import sys
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')

import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'

from rag.document_loader import get_document_loader_manager
from config.settings import get_settings

settings = get_settings()

print("=" * 60)
print("测试 1: 元数据增强")
print("=" * 60)

loader_manager = get_document_loader_manager()

# 测试 Markdown 文件
test_md = "/share/home/lifr/workspace/code/enterprise-knowledge-agent/data/knowledge/公司简介.md"

print(f"\n加载文件: {test_md}")
docs = loader_manager.load_file(test_md)

print(f"\n原始文档元数据:")
for i, doc in enumerate(docs[:2]):
    print(f"\n--- 文档 {i+1} ---")
    for key, value in doc.metadata.items():
        if key != 'all_sections':  # 避免打印过长内容
            print(f"  {key}: {value}")
    if 'all_sections' in doc.metadata:
        print(f"  all_sections (共 {len(doc.metadata['all_sections'])} 个章节):")
        for sec in doc.metadata['all_sections'][:5]:
            print(f"    - {'#' * sec['level']} {sec['title']}")

print("\n" + "=" * 60)
print("测试 2: 不同分块策略")
print("=" * 60)

# 测试不同分块策略
strategies = ["recursive", "semantic", "hybrid"]

for strategy in strategies:
    print(f"\n--- 分块策略: {strategy} ---")
    try:
        chunks = loader_manager.split_documents(
            docs,
            chunk_size=500,
            splitter_type=strategy
        )
        print(f"分块数量: {len(chunks)}")
        
        # 打印第一个块的元数据
        if chunks:
            print(f"\n第一个分块的元数据:")
            for key, value in list(chunks[0].metadata.items())[:10]:
                print(f"  {key}: {value}")
            
            # 显示内容预览
            content_preview = chunks[0].page_content[:150].replace('\n', ' ')
            print(f"\n内容预览: {content_preview}...")
    except Exception as e:
        print(f"  ❌ 错误: {e}")

print("\n" + "=" * 60)
print("测试 3: PDF 元数据")
print("=" * 60)

test_pdf = "/share/home/lifr/workspace/code/enterprise-knowledge-agent/data/knowledge/思路.pdf"

print(f"\n加载文件: {test_pdf}")
pdf_docs = loader_manager.load_file(test_pdf)

print(f"\nPDF 文档元数据:")
if pdf_docs:
    for key, value in list(pdf_docs[0].metadata.items())[:10]:
        print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
