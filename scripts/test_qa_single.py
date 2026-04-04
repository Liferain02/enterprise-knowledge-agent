#!/usr/bin/env python
"""单条 QA 准确率测试"""
import sys, os

for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.graph import run_agent
from scripts.eval_dataset import EVAL_DATASET

eq = EVAL_DATASET[0]
print(f"查询: {eq.query}")
print(f"期望答案: {eq.ground_truth}")

result = run_agent(eq.query, session_id="qa-single-test")
answer = result.get("final_answer", "")
print(f"\n模型回答:\n{answer}")

gt_keywords = eq.ground_truth.split("：")[0].split("、")[:3]
gt_main = eq.ground_truth.split("：")[0][:10]
matched = sum(1 for kw in gt_keywords if kw in answer)
print(f"\n关键词命中: {matched}/3")
print(f"主短语命中: {gt_main in answer}")
