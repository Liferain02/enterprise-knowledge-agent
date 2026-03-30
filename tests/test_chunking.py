#!/usr/bin/env python
"""
测试增强后的分块策略
验证：
1. 元数据增强（文档标题、章节信息）
2. Token-based 分块功能
3. 优化分隔符（标题边界、分号、省略号）
4. Title + Content 拼接
5. 语义 Overlap
6. 句子分割（支持省略号、分号、列表序号）
"""
import sys
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')

import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'

from src.rag.processing.document_loader import (
    get_document_loader_manager,
    estimate_tokens,
    split_sentences,
    OPTIMIZED_SEPARATORS,
)
from config.settings import get_settings

settings = get_settings()

print("=" * 60)
print("测试 1: Token 估算")
print("=" * 60)

test_texts = [
    "这是一个中文句子。" * 10,        # 纯中文
    "This is an English sentence. " * 20,  # 纯英文
    "公司年假政策：工作满1年可休5天。报销流程需要发票和出差报告。",  # 混合
]

for text in test_texts:
    tokens = estimate_tokens(text)
    chars = len(text)
    ratio = chars / max(tokens, 1)
    print(f"  字符数={chars}, Token数≈{tokens}, 比例≈{ratio:.1f} 字符/token")
    print(f"  预览: {text[:50]}...")
    print()


print("=" * 60)
print("测试 2: 句子分割（优化正则）")
print("=" * 60)

test_sentences_texts = [
    # 省略号
    "用户手册……请仔细阅读以下内容。",
    # 分号
    "第一步：提交申请；第二步：等待审批；第三步：领取结果。",
    # 列表序号
    "1. 第一项内容。2. 第二项内容。3. 第三项内容。",
    # 圆圈序号
    "① 第一条。② 第二条。③ 第三条。",
    # 混合
    "公司福利包括：年假……调休；补贴；培训。① 技术培训。② 管理培训。",
]

for text in test_sentences_texts:
    sentences = split_sentences(text)
    print(f"  原文: {text}")
    print(f"  分割后: {sentences}")
    print()


print("=" * 60)
print("测试 3: 分隔符列表（优化版）")
print("=" * 60)

print(f"  共 {len(OPTIMIZED_SEPARATORS)} 个分隔符:")
for sep in OPTIMIZED_SEPARATORS:
    repr_sep = repr(sep)
    print(f"    {repr_sep}")
print()


print("=" * 60)
print("测试 4: 不同分块策略")
print("=" * 60)

loader_manager = get_document_loader_manager()
test_md = "/share/home/lifr/workspace/code/enterprise-knowledge-agent/data/knowledge/公司简介.md"

print(f"\n加载文件: {test_md}")
docs = loader_manager.load_file(test_md)

strategies = ["recursive", "semantic", "hybrid", "markdown"]

for strategy in strategies:
    print(f"\n--- 分块策略: {strategy} ---")
    try:
        chunks = loader_manager.split_documents(
            docs,
            splitter_type=strategy,
        )
        print(f"分块数量: {len(chunks)}")

        if chunks:
            # 统计 token
            total_chars = sum(len(c.page_content) for c in chunks)
            total_tokens = sum(
                estimate_tokens(c.page_content) for c in chunks
            )
            avg_chars = total_chars / len(chunks)
            avg_tokens = total_tokens / len(chunks)
            print(f"  平均: {avg_chars:.0f} 字符, {avg_tokens:.0f} tokens/块")
            print(f"  Token范围: "
                  f"{min(estimate_tokens(c.page_content) for c in chunks):.0f}"
                  f" ~ "
                  f"{max(estimate_tokens(c.page_content) for c in chunks):.0f}")

            # 第一个块
            print(f"\n第一个分块:")
            meta = chunks[0].metadata
            print(f"  元数据: chunking_method={meta.get('chunking_method')}, "
                  f"chunk_token_estimate={meta.get('chunk_token_estimate')}, "
                  f"parent_section={meta.get('parent_section', 'N/A')}")
            content = chunks[0].page_content
            preview = content[:200].replace('\n', ' ')
            print(f"  内容预览: {preview}...")

            # 检查 Title + Content 拼接
            if chunks[0].page_content.startswith('#'):
                print(f"  ✓ 包含 Markdown 标题")
            elif any(kw in chunks[0].page_content for kw in ['第一章', '公司', '简介']):
                print(f"  ✓ 可能包含父级标题")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("测试 5: Token-Based 分块器")
print("=" * 60)

from src.rag.processing.document_loader import TokenRecursiveTextSplitter

splitter = TokenRecursiveTextSplitter(
    chunk_token_size=200,
    chunk_token_overlap=50,
    concat_title=True,
    semantic_overlap=True,
)

long_text = """
# 员工手册

## 第一章 年假制度

员工年假天数按工龄计算：1-3年：5天；3-5年：10天；5-10年：15天；10年以上：20天。年假需提前3天向部门经理申请。

## 第二章 病假制度

员工病假需提供医院证明。3天以内：带薪；3-7天：扣发当日基本工资的50%；7天以上：需经人力资源部审批。

## 第三章 报销流程

差旅费用报销流程：员工需在出差结束后5个工作日内提交报销申请，附上发票和出差报告，经部门经理和财务部审批后发放。

### 报销标准

1. 交通费：实报实销，需提供发票。
2. 住宿费：每人每天不超过300元。
3. 餐费：每人每天不超过100元。

## 第四章 绩效考核

KPI绩效考核每季度进行一次……员工需在规定时间内完成自评和上级评价。考核结果将直接影响年终奖金和晋升机会。
"""

chunks = splitter.split_text(long_text)
print(f"原始文本: {len(long_text)} 字符, ≈{estimate_tokens(long_text)} tokens")
print(f"分块数量: {len(chunks)}")
print()

for i, chunk in enumerate(chunks):
    tokens = estimate_tokens(chunk)
    has_title = chunk.startswith('#') or any(
        kw in chunk[:50] for kw in ['第一章', '第二章', '员工手册', '报销', '考核']
    )
    title_flag = "✓ 有标题" if has_title else ""
    print(f"  Chunk {i+1}: {len(chunk)} 字符, ≈{tokens} tokens {title_flag}")
    preview = chunk[:80].replace('\n', ' ')
    print(f"    开头: {preview}...")
    print()


print("=" * 60)
print("测试 6: 语义分块（SemanticChunker）")
print("=" * 60)

from src.rag.processing.chunker import SemanticChunker, split_sentences as chunker_split

test_chunks = """
根据《员工手册》第三章第七条，员工年假天数按工龄计算：
1-3年：5天。
3-5年：10天。
5-10年：15天。
10年以上：20天。
年假需提前3天向部门经理申请。

员工病假需提供医院证明。3天以内：带薪；3-7天：扣发当日基本工资的50%；7天以上：需经人力资源部审批。

差旅费用报销流程：员工需在出差结束后5个工作日内提交报销申请，附上发票和出差报告，经部门经理和财务部审批后发放。
"""

sentences = chunker_split(test_chunks)
print(f"句子分割测试: 共 {len(sentences)} 个句子")
for j, s in enumerate(sentences[:8]):
    print(f"  [{j}] {s[:50]}")
print()

try:
    semantic_chunker = SemanticChunker(
        threshold=0.3,
        max_tokens=150,
        min_tokens=30,
        concat_title=True,
        semantic_overlap=True,
    )
    chunks = semantic_chunker.split_text(test_chunks)
    print(f"语义分块: {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        tokens = estimate_tokens(chunk)
        print(f"  Chunk {i+1}: ≈{tokens} tokens")
        print(f"    {chunk[:80].replace(chr(10), ' ')}...")
except Exception as e:
    print(f"  ❌ 语义分块失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
