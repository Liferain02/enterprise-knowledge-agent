#!/usr/bin/env python
"""端到端 QA 准确率测试（跳过前面卡住的步骤）"""
import sys, os

for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.graph import run_agent
from scripts.eval_dataset import EVAL_DATASET

def test_qa_accuracy(test_cases):
    print("=" * 60)
    print("端到端 QA 准确率测试")
    print("=" * 60)

    correct = 0
    partial = 0
    incorrect = 0
    errors = 0

    for eq in test_cases:
        try:
            result = run_agent(eq.query, session_id=f"qa-test-{hash(eq.query)}")
            answer = result.get("final_answer", "")
            gt_keywords = eq.ground_truth.split("：")[0].split("、")[:3]
            gt_main = eq.ground_truth.split("：")[0][:10]
            matched_keywords = sum(1 for kw in gt_keywords if kw in answer)
            matched_main = gt_main in answer

            if matched_main or matched_keywords >= 2:
                verdict = "correct"
                correct += 1
            elif matched_keywords >= 1:
                verdict = "partial"
                partial += 1
            else:
                verdict = "incorrect"
                incorrect += 1

            print(f"  [{verdict.upper():<8}] {eq.query[:35]:<37} gt={gt_main}")
        except Exception as e:
            errors += 1
            print(f"  [ERROR] {eq.query[:40]} - {e}")

    total = correct + partial + incorrect
    acc = correct / total if total else 0
    partial_rate = partial / total if total else 0
    print(f"\n  完全正确: {correct}/{total} = {acc:.1%}")
    print(f"  部分正确: {partial}/{total} = {partial_rate:.1%}")
    print(f"  错误: {incorrect}/{total} = {incorrect/total:.1%}")
    print(f"  异常: {errors}")
    print(f"\n  >>> QA 准确率 = {acc:.1%}")
    return acc

if __name__ == "__main__":
    test_qa_accuracy(EVAL_DATASET[:15])
