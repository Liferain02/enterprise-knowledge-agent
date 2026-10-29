#!/usr/bin/env python3
"""Generate deterministic public-style lab notes for offline retrieval demos.

The files are synthetic and explicitly labeled; they contain no secrets or real
personal data. Re-running replaces only this script's own namespace.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/knowledge/synthetic_autumn_2026'
CATEGORIES = ('rdma', 'numa', 'retrieval', 'systems', 'experiments', 'paper_writing')
TOPICS = {
    'rdma': ('RDMA queue pair', 'completion polling', 'memory registration', 'RoCE congestion', 'NUMA NIC affinity', 'ib_write_bw benchmarking'),
    'numa': ('remote memory', 'first-touch placement', 'page migration', 'socket locality', 'cache coherence', 'tail latency'),
    'retrieval': ('BM25 fusion', 'vector recall', 'reranking', 'ACL filtering', 'query rewriting', 'chunk overlap'),
    'systems': ('Linux perf', 'CPU pinning', 'huge pages', 'container limits', 'thread pools', 'checkpoint recovery'),
    'experiments': ('ablation design', 'warm cache latency', 'throughput measurement', 'failure injection', 'confidence intervals', 'reproducibility'),
    'paper_writing': ('method section', 'experiment table', 'rebuttal response', 'citation audit', 'limitation statement', 'abstract structure'),
}

def body(category, topic, idx):
    return f'''# 合成科研笔记 {category}-{idx:03d}：{topic}

> 资料性质：公开风格的合成练习资料，仅用于本地知识库扩容和检索回归，不代表实验室真实制度、实验结果或作者观点。

## 研究问题
本文围绕“{topic}”整理一个可复现的实验观察框架。查询时应区分概念解释、配置建议、实验事实和待验证假设，不能把本笔记的示例数字当作真实测量。

## 方法与条件
记录硬件、软件版本、数据集、线程数、消息大小、NUMA 节点、网络配置和随机种子。每次修改只改变一个主要变量，并保留原始日志、配置快照、commit、运行时间和失败原因。对 {topic} 的分析应先确认样本口径，再比较均值、中位数、P95、吞吐与错误率。

## 示例观察
在合成场景中，实验者可以比较 baseline 与 candidate，并报告相对变化率；如果只观察到单次运行，不应写成稳定提升或统计显著。缓存命中、权限过滤、版本状态和检索来源都应单独记录。涉及文档证据时，回答需要指出来源名称和版本。

## 复现清单
- 记录环境：Python、驱动、内核、容器镜像和模型版本。
- 记录输入：查询、候选文档、过滤条件和用户角色。
- 记录输出：原始结果、聚合指标、延迟分位数和错误日志。
- 记录限制：样本量、未控制变量、权限边界和潜在偏差。

## 讨论
本笔记只提供实验设计语言和合成例子。若用户问“项目实际采用了什么”，必须优先检索项目资料和当前有效版本；若资料不足，应明确说无法从本笔记推出真实结论。
'''

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('*.md'):
        old.unlink()
    count = 0
    for category in CATEGORIES:
        for idx in range(1, 41):
            topic = TOPICS[category][(idx - 1) % len(TOPICS[category])]
            (OUT / f'{category}_{idx:03d}_{topic.replace(" ", "_")}.md').write_text(
                body(category, topic, idx), encoding='utf-8')
            count += 1
    print(f'generated={count} directory={OUT}')

if __name__ == '__main__':
    main()
