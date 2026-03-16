#!/usr/bin/env python
"""
RAG 评估测试脚本
运行评估测试并输出结果
"""
import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.eval_dataset import EVAL_DATASET
from src.rag.evaluation import get_evaluator, evaluate_rag_batch
from src.agent.graph import arun_agent


async def main():
    """运行评估测试"""
    print("=" * 60)
    print("RAG 评估测试")
    print("=" * 60)

    # 选择评估的查询数量（可以全部或部分）
    eval_queries = EVAL_DATASET[:10]  # 测试前10个
    print(f"准备评估 {len(eval_queries)} 个查询...")

    # 收集评估数据
    eval_data = []
    for eq in eval_queries:
        print(f"\n处理查询: {eq.query}")

        # 运行 Agent 获取答案
        try:
            result = await arun_agent(
                input_text=eq.query,
                session_id=f"eval_{eq.query[:10]}"
            )

            # 提取上下文（从 sources 中获取）
            # 这里简化处理，实际应该从 Agent 结果中获取检索到的文档
            contexts = []
            if "sources" in result:
                # 如果有 sources 信息，尝试从中提取上下文
                # 这里暂时使用空列表，实际需要从 Agent state 中获取
                pass

            eval_data.append({
                "query": eq.query,
                "answer": result.get("final_answer", ""),
                "contexts": contexts,  # 需要从 Agent 获取
                "ground_truth": eq.ground_truth
            })

            print(f"  答案: {result.get('final_answer', '')[:100]}...")

        except Exception as e:
            print(f"  错误: {e}")

    # 运行评估
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)

    evaluator = get_evaluator()
    results = await evaluator.evaluate_batch(eval_data, show_progress=True)

    # 生成摘要
    summary = evaluator.generate_summary(results)

    # 打印结果
    summary.print_summary()

    # 打印详细结果
    print("\n详细结果:")
    for i, result in enumerate(results):
        print(f"\n--- 查询 {i+1} ---")
        print(f"问题: {result.query}")
        print(f"Faithfulness: {result.faithfulness:.4f}")
        print(f"Answer Relevancy: {result.answer_relevancy:.4f}")
        print(f"Context Recall: {result.context_recall:.4f}")
        print(f"Context Precision: {result.context_precision:.4f}")

    # 保存结果到文件
    output_file = "eval_results.json"
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in results]
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
