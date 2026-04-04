#!/usr/bin/env python
"""
RAG 检索系统评估模块

使用方法:
    python scripts/test_retrieval_eval.py

该模块提供:
    1. 检索召回率/精度计算
    2. MRR (Mean Reciprocal Rank) 计算
    3. NDCG (Normalized Discounted Cumulative Gain) 计算
    4. RAGAs 框架集成评估
"""
import sys
import os
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')

import os
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

# 导入项目模块
from src.rag.retrieval.retriever import get_retriever_manager
from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager
from src.rag.retrieval.reranker import get_reranker_manager
from langchain_core.documents import Document

# 导入评估数据集
from scripts.eval_dataset import EVAL_DATASET, EvalQuery


@dataclass
class RetrievalResult:
    """单次检索结果"""
    query: str
    retrieved_docs: List[Document]
    retrieved_doc_ids: List[str]
    relevant_doc_ids: List[str]
    
    @property
    def num_retrieved(self) -> int:
        return len(self.retrieved_doc_ids)
    
    @property
    def num_relevant(self) -> int:
        return len(self.relevant_doc_ids)
    
    @property
    def retrieved_relevant_ids(self) -> set:
        return set(self.retrieved_doc_ids) & set(self.relevant_doc_ids)
    
    @property
    def num_retrieved_relevant(self) -> int:
        return len(self.retrieved_relevant_ids)


@dataclass
class RetrievalMetrics:
    """检索评估指标"""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "Recall@1": self.recall_at_1,
            "Recall@3": self.recall_at_3,
            "Recall@5": self.recall_at_5,
            "Precision@1": self.precision_at_1,
            "Precision@3": self.precision_at_3,
            "Precision@5": self.precision_at_5,
            "MRR": self.mrr,
            "NDCG@3": self.ndcg_at_3,
            "NDCG@5": self.ndcg_at_5,
        }


def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """计算 Recall@K
    
    Args:
        retrieved_ids: 检索到的文档ID列表
        relevant_ids: 相关文档ID列表
        k: 考虑的top K结果
    
    Returns:
        Recall@K 值
    """
    if len(relevant_ids) == 0:
        return 0.0
    
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    
    return len(retrieved_k & relevant) / len(relevant)


def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """计算 Precision@K
    
    Args:
        retrieved_ids: 检索到的文档ID列表
        relevant_ids: 相关文档ID列表
        k: 考虑的top K结果
    
    Returns:
        Precision@K 值
    """
    if k == 0:
        return 0.0
    
    retrieved_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    
    return len([d for d in retrieved_k if d in relevant]) / k


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """计算 MRR (Mean Reciprocal Rank)
    
    Args:
        retrieved_ids: 检索到的文档ID列表
        relevant_ids: 相关文档ID列表
    
    Returns:
        MRR 值
    """
    relevant = set(relevant_ids)
    
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant:
            return 1.0 / i
    
    return 0.0


def calculate_dcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """计算 DCG (Discounted Cumulative Gain)
    
    使用二进制相关性（相关=1，不相关=0）
    """
    relevant = set(relevant_ids)
    dcg = 0.0
    
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant:
            dcg += 1.0 / (i ** 2 - i + 1) ** 0.5  # 使用 NDCG 的标准折扣公式
    
    return dcg


def calculate_ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """计算 NDCG (Normalized Discounted Cumulative Gain)
    
    Args:
        retrieved_ids: 检索到的文档ID列表
        relevant_ids: 相关文档ID列表
        k: 考虑的top K结果
    
    Returns:
        NDCG@K 值
    """
    if len(relevant_ids) == 0:
        return 0.0
    
    dcg = calculate_dcg(retrieved_ids, relevant_ids, k)
    
    # 理想DCG：按相关性排序（所有相关文档都在前面）
    ideal_ids = list(relevant_ids) + [id for id in retrieved_ids if id not in relevant_ids]
    idcg = calculate_dcg(ideal_ids, relevant_ids, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def extract_doc_id_from_metadata(doc: Document, index: int) -> str:
    """从元数据中提取文档ID
    
    从完整路径中提取文件名作为文档ID
    例如: /path/to/员工手册.md -> 员工手册
    """
    # 优先使用 source 或 source_file
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    
    if source:
        # 从完整路径中提取文件名（去掉扩展名）
        filename = os.path.splitext(os.path.basename(source))[0]
        return filename
    
    # 如果没有source，使用索引
    return f"doc_{index}"


def evaluate_retrieval(
    retrieved_docs: List[Document],
    eval_query: EvalQuery,
    k_values: List[int] = [1, 3, 5]
) -> RetrievalMetrics:
    """评估单次检索结果
    
    Args:
        retrieved_docs: 检索到的文档列表
        eval_query: 评估查询对象
        k_values: 要计算的K值列表
    
    Returns:
        检索评估指标
    """
    # 提取文档ID - 使用新函数从元数据中提取
    retrieved_ids = [extract_doc_id_from_metadata(doc, i) 
                     for i, doc in enumerate(retrieved_docs)]
    
    # 计算各项指标
    metrics = RetrievalMetrics()
    
    for k in k_values:
        # Recall
        if k == 1:
            metrics.recall_at_1 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 3:
            metrics.recall_at_3 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 5:
            metrics.recall_at_5 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
        
        # Precision
        if k == 1:
            metrics.precision_at_1 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 3:
            metrics.precision_at_3 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 5:
            metrics.precision_at_5 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
        
        # NDCG
        if k == 3:
            metrics.ndcg_at_3 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 5:
            metrics.ndcg_at_5 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, k)
    
    # MRR
    metrics.mrr = calculate_mrr(retrieved_ids, eval_query.relevant_doc_ids)
    
    return metrics


def average_metrics(all_metrics: List[RetrievalMetrics]) -> RetrievalMetrics:
    """计算平均指标"""
    if not all_metrics:
        return RetrievalMetrics()
    
    avg = RetrievalMetrics()
    n = len(all_metrics)
    
    avg.recall_at_1 = sum(m.recall_at_1 for m in all_metrics) / n
    avg.recall_at_3 = sum(m.recall_at_3 for m in all_metrics) / n
    avg.recall_at_5 = sum(m.recall_at_5 for m in all_metrics) / n
    avg.precision_at_1 = sum(m.precision_at_1 for m in all_metrics) / n
    avg.precision_at_3 = sum(m.precision_at_3 for m in all_metrics) / n
    avg.precision_at_5 = sum(m.precision_at_5 for m in all_metrics) / n
    avg.mrr = sum(m.mrr for m in all_metrics) / n
    avg.ndcg_at_3 = sum(m.ndcg_at_3 for m in all_metrics) / n
    avg.ndcg_at_5 = sum(m.ndcg_at_5 for m in all_metrics) / n
    
    return avg


def run_basic_retrieval_eval():
    """运行基础向量检索评估"""
    print("\n" + "=" * 70)
    print("基础向量检索评估")
    print("=" * 70)
    
    retriever_mgr = get_retriever_manager()
    all_metrics: List[RetrievalMetrics] = []
    
    for eval_query in EVAL_DATASET:
        print(f"\n查询: {eval_query.query}")
        
        # 执行检索
        retrieved_docs = retriever_mgr.search(eval_query.query, k=10)
        
        # 评估
        metrics = evaluate_retrieval(retrieved_docs, eval_query)
        all_metrics.append(metrics)
        
        print(f"  检索到 {len(retrieved_docs)} 文档")
        print(f"  相关文档: {eval_query.relevant_doc_ids}")
        print(f"  Recall@3: {metrics.recall_at_3:.4f}, Precision@3: {metrics.precision_at_3:.4f}")
        print(f"  MRR: {metrics.mrr:.4f}, NDCG@3: {metrics.ndcg_at_3:.4f}")
    
    # 计算平均指标
    avg_metrics = average_metrics(all_metrics)
    
    print("\n" + "-" * 70)
    print("基础向量检索 - 平均指标")
    print("-" * 70)
    for key, value in avg_metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    return avg_metrics


def run_hybrid_retrieval_eval():
    """运行混合检索评估"""
    print("\n" + "=" * 70)
    print("混合检索评估 (BM25 + 向量)")
    print("=" * 70)
    
    from config.settings import get_settings
    settings = get_settings()
    
    # 创建混合检索管理器
    hybrid_mgr = get_hybrid_retriever_manager(
        collection_name="enterprise_knowledge",
        top_k=10,
        vector_weight=settings.hybrid_vector_weight,
        bm25_weight=settings.hybrid_bm25_weight
    )
    
    all_metrics: List[RetrievalMetrics] = []
    
    for eval_query in EVAL_DATASET:
        print(f"\n查询: {eval_query.query}")
        
        # 执行混合检索
        retrieved_docs = hybrid_mgr.search(eval_query.query, k=10)
        
        # 评估
        metrics = evaluate_retrieval(retrieved_docs, eval_query)
        all_metrics.append(metrics)
        
        print(f"  检索到 {len(retrieved_docs)} 文档")
        print(f"  相关文档: {eval_query.relevant_doc_ids}")
        print(f"  Recall@3: {metrics.recall_at_3:.4f}, Precision@3: {metrics.precision_at_3:.4f}")
        print(f"  MRR: {metrics.mrr:.4f}, NDCG@3: {metrics.ndcg_at_3:.4f}")
    
    # 计算平均指标
    avg_metrics = average_metrics(all_metrics)
    
    print("\n" + "-" * 70)
    print("混合检索 - 平均指标")
    print("-" * 70)
    for key, value in avg_metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    return avg_metrics


def run_reranker_eval():
    """运行带Reranker的检索评估"""
    print("\n" + "=" * 70)
    print("带Reranker的检索评估")
    print("=" * 70)
    
    retriever_mgr = get_retriever_manager()
    reranker_mgr = get_reranker_manager()
    
    all_metrics: List[RetrievalMetrics] = []
    
    for eval_query in EVAL_DATASET:
        print(f"\n查询: {eval_query.query}")
        
        # 1. 先检索更多候选文档
        candidates = retriever_mgr.search(eval_query.query, k=15)
        
        if not candidates:
            print("  无检索结果")
            continue
        
        # 2. 使用Reranker重排序
        reranked_results = reranker_mgr.rerank(eval_query.query, candidates, top_n=5)
        retrieved_docs = [doc for doc, score in reranked_results]
        
        # 评估
        metrics = evaluate_retrieval(retrieved_docs, eval_query)
        all_metrics.append(metrics)
        
        print(f"  检索到 {len(retrieved_docs)} 文档")
        print(f"  相关文档: {eval_query.relevant_doc_ids}")
        print(f"  Recall@3: {metrics.recall_at_3:.4f}, Precision@3: {metrics.precision_at_3:.4f}")
        print(f"  MRR: {metrics.mrr:.4f}, NDCG@3: {metrics.ndcg_at_3:.4f}")
        
        # 显示Top 3结果及分数
        print("  Top 3 结果:")
        for i, (doc, score) in enumerate(reranked_results[:3], 1):
            doc_id = extract_doc_id_from_metadata(doc, i-1)
            print(f"    {i}. [score={score:.4f}] {doc_id}: {doc.page_content[:50]}...")
    
    # 计算平均指标
    avg_metrics = average_metrics(all_metrics)
    
    print("\n" + "-" * 70)
    print("Reranker检索 - 平均指标")
    print("-" * 70)
    for key, value in avg_metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    return avg_metrics


def run_ragas_eval():
    """运行RAGAs框架评估（端到端RAG质量）"""
    print("\n" + "=" * 70)
    print("RAGAs 框架评估 (端到端)")
    print("=" * 70)
    
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from datasets import Dataset

        # 准备评估数据
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        retriever_mgr = get_retriever_manager()
        
        # 生成答案和收集上下文
        from src.models.llm import get_llm
        llm = get_llm()
        
        for eval_query in EVAL_DATASET:
            # 检索上下文
            retrieved_docs = retriever_mgr.search(eval_query.query, k=5)
            contexts = [doc.page_content for doc in retrieved_docs]
            
            # 构建提示
            prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文:
{chr(10).join(contexts)}

问题: {eval_query.query}

回答:"""
            
            # 调用LLM生成答案
            try:
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"  LLM调用失败: {e}")
                answer = "无法生成答案"
            
            eval_data["question"].append(eval_query.query)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts)
            eval_data["ground_truth"].append(eval_query.ground_truth)
            
            print(f"\n查询: {eval_query.query}")
            print(f"答案: {answer[:100]}...")
        
        # 创建数据集
        dataset = Dataset.from_dict(eval_data)
        
        # 执行评估
        print("\n正在执行RAGAs评估...")
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        
        print("\n" + "-" * 70)
        print("RAGAs 评估结果")
        print("-" * 70)
        print(results)
        
        return results
        
    except ImportError as e:
        print(f"  RAGAs相关包未安装: {e}")
        print("  请运行: pip install ragas datasets")
        return None
    except Exception as e:
        print(f"  RAGAs评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_evaluations():
    """运行所有评估"""
    print("\n" + "=" * 70)
    print("RAG 检索系统全面评估")
    print("=" * 70)
    
    results = {}
    
    # 1. 基础向量检索评估
    results["basic_vector"] = run_basic_retrieval_eval()
    
    # 2. 混合检索评估
    results["hybrid"] = run_hybrid_retrieval_eval()
    
    # 3. Reranker检索评估
    results["reranker"] = run_reranker_eval()
    
    # 4. RAGAs端到端评估
    results["ragas"] = run_ragas_eval()
    
    # 汇总对比
    print("\n" + "=" * 70)
    print("评估结果汇总对比")
    print("=" * 70)
    
    comparison_data = []
    for method, metrics in results.items():
        if isinstance(metrics, RetrievalMetrics):
            row = {"方法": method}
            row.update(metrics.to_dict())
            comparison_data.append(row)
    
    # 打印对比表格
    if comparison_data:
        # 获取所有指标键
        metric_keys = list(comparison_data[0].keys())[1:]
        
        print(f"\n{'方法':<20}", end="")
        for key in metric_keys:
            print(f"{key:>12}", end="")
        print()
        print("-" * (20 + 12 * len(metric_keys)))
        
        for row in comparison_data:
            print(f"{row['方法']:<20}", end="")
            for key in metric_keys:
                print(f"{row[key]:>12.4f}", end="")
            print()
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG检索系统评估")
    parser.add_argument(
        "--type", 
        choices=["basic", "hybrid", "reranker", "ragas", "all"],
        default="all",
        help="评估类型"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出结果到JSON文件"
    )
    
    args = parser.parse_args()
    
    # 根据类型运行评估
    if args.type == "basic":
        results = {"basic_vector": run_basic_retrieval_eval()}
    elif args.type == "hybrid":
        results = {"hybrid": run_hybrid_retrieval_eval()}
    elif args.type == "reranker":
        results = {"reranker": run_reranker_eval()}
    elif args.type == "ragas":
        results = {"ragas": run_ragas_eval()}
    else:
        results = run_all_evaluations()
    
    # 输出到文件
    if args.output:
        # 转换结果为可JSON序列化的格式
        output_data = {}
        for key, value in results.items():
            if isinstance(value, RetrievalMetrics):
                output_data[key] = value.to_dict()
            else:
                output_data[key] = str(value)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()
