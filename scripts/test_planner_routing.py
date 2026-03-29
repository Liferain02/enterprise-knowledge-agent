"""
Planner 复杂度判断准确率测试
直接调用 planner_node，对测试数据集逐条评估，输出分类报告

运行方式：
    cd /home/xypp/code/enterprise-knowledge-agent
    conda activate agent-demo
    python scripts/test_planner_routing.py

测试数据集说明：
    - 第1类：简单任务（直接 Supervisor，跳过 LLM）
    - 第2类：复杂任务（Execute Plan，LLM 拆步骤）
    - 第3类：多子问题（同属知识库，Supervisor 可处理）
    - 第4类：边界案例
    - 第5类：高优先级 pattern（列举类/多实体，短查询但应判 complex）
      新增原因：这些短查询命中 _HIGH_PRIORITY_PATTERNS（优先于 len ≤ 40 短路），
      确保 "公司假期都有哪些？" 等列举类查询被正确判定为 complex。
"""
import asyncio
import json
import sys
import os
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 移除代理设置（直连）
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        os.environ.pop(k, None)

from langchain_core.messages import HumanMessage
from src.agent.agents.planner import planner_node


# =============================================================================
# 测试数据集
# =============================================================================
# 每个样例: (query, expected_is_complex)
# expected_is_complex: True = 复杂任务（需要拆解步骤）, False = 简单任务（直接路由到 Supervisor）

PLANNER_TEST_CASES: list[tuple[str, bool, str]] = [
    # =====================================================================
    # 1. 简单任务 — 应该跳过 LLM 规划，直接由 Supervisor 处理
    # =====================================================================
    ("公司的年假政策是什么？", False, "单一知识查询"),
    ("你好", False, "问候语"),
    ("现在几点？", False, "时间查询"),
    ("帮我计算一下 123 加 456", False, "简单计算"),
    ("报销流程是什么？", False, "单一知识查询"),
    ("KPI考核标准有哪些？", False, "单一知识查询"),
    ("新员工培训内容是什么？", False, "单一知识查询"),
    ("客户投诉怎么处理？", False, "单一知识查询"),
    ("请介绍一下弹性工作制", False, "单一知识查询"),
    ("试用期为多长时间？", False, "单一知识查询"),
    ("年假有多少天？", False, "单一知识查询"),
    ("门禁卡丢了怎么办？", False, "单一知识查询"),
    ("你好！", False, "问候语"),
    ("在吗？", False, "问候语"),
    ("今天天气怎么样？", False, "闲聊"),
    ("这个bug怎么修？", False, "模糊请求"),
    ("随便聊聊", False, "闲聊"),
    ("明天的会议几点开始？", False, "简单时间查询"),
    ("帮我算平均值", False, "简单计算"),
    ("2024年有多少天？", False, "简单时间查询"),

    # =====================================================================
    # 2. 复杂任务 — 需要拆解成多个步骤
    # =====================================================================
    ("对比年假和病假政策的差异", True, "对比类：需分别查询两个政策再对比"),
    ("年假和病假有什么区别？", True, "对比类：需分别查询两个政策再对比"),
    ("对比A政策和B政策的差异", True, "对比类：多政策对比"),
    ("对比Q1和Q2的业绩表现", True, "对比类：多季度数据对比"),
    ("总结Q1和Q2的业绩表现", True, "总结类：多数据汇总"),
    ("对比内部培训和外部培训的优劣", True, "对比类：培训方式对比"),
    ("报销标准是什么？加上已报销的金额，总额是多少？", True, "复合类：查询+计算"),
    ("年假还剩多少？顺便告诉我政策规定", True, "复合类：计算+知识查询"),
    ("先查年假政策，再算一下我能休几天", True, "顺序类：知识→计算"),
    ("帮我查一下公司有哪些福利，再总结一下重点", True, "顺序类：查询→总结"),
    ("查询张三和李四的绩效考核结果，然后对比差异", True, "对比类：多人员数据对比"),
    ("先查年假政策，再查病假政策，最后对比两者的区别", True, "顺序+对比：多次查询+对比"),
    ("预算执行率怎么算？同时告诉我公司的预算制度", True, "复合类：计算+知识"),
    ("介绍一下弹性工作制，然后帮我算一下本周工作时长", True, "复合类：知识介绍+计算"),
    ("报销流程是什么？同时帮我算一下本月已报销多少", True, "复合类：流程+计算"),

    # =====================================================================
    # 3. 多子问题（但同类）— 视复杂度策略决定
    # =====================================================================
    ("KPI标准是什么？同时我的绩效数据在哪里查", False, "同属知识库，Supervisor可处理"),
    ("公司的福利政策和报销流程分别是什么？", False, "同属知识库，Supervisor可处理"),
    ("员工手册和绩效考核制度各有什么规定？", False, "同属知识库，Supervisor可处理"),

    # =====================================================================
    # 4. 边界案例
    # =====================================================================
    ("对比", True, "极短对比词"),
    ("总结", True, "极短总结词"),
    ("你好，请问病假怎么处理", False, "问候+单一问题"),
    ("在吗？我想了解一下医保报销", False, "问候+单一问题"),
    ("帮我查一下张三的年假余额，然后算一下还剩多少", True, "顺序：查询→计算"),
    ("张三和李四谁的KPI更高？", True, "对比类：两人数据对比"),
    ("今年和去年的销售额对比怎么样？", True, "对比类：跨年数据对比"),

    # =====================================================================
    # 5. 高优先级 pattern（列举类/多实体）- 优先于 len ≤ 40 短路
    #    这些短查询命中 _HIGH_PRIORITY_PATTERNS，应判定为 complex
    #    之前因 len ≤ 40 被错误短路为 simple
    # =====================================================================
    ("公司假期都有哪些？", True, "列举类：短查询但含'有哪些'，应为 complex"),
    ("有什么福利？", True, "列举类：短查询但含'有什么'，应为 complex"),
    ("年假病假都有哪些？", True, "列举类：短查询但含'有哪些'，应为 complex"),
    ("张三和李四的工作职责", True, "多实体类：短查询但含'和'字，应为 complex"),
    ("年假有什么规定？", True, "列举类：短查询含'有什么'，应为 complex"),
    ("福利有哪些？", True, "列举类：短查询含'有哪些'，应为 complex"),
    ("报销制度都有哪些内容？", True, "列举类：短查询含'有哪些'，应为 complex"),
]


async def run_single_test(query: str, expected_complex: bool, reason: str) -> dict:
    """运行单条测试"""
    state = {
        "messages": [HumanMessage(content=query)],
        "mem0_memories": "",
    }
    try:
        result = await planner_node(state)
        predicted_complex = result.get("is_complex", False)
        predicted_steps = result.get("plan_steps", [])
        reasoning = result.get("plan_reasoning", "")
        correct = predicted_complex == expected_complex

        # 分析错误类型
        if not correct:
            if expected_complex and not predicted_complex:
                error_type = "complex→simple（漏判复杂任务）"
            else:
                error_type = "simple→complex（误判简单任务）"
        else:
            error_type = "correct"

        return {
            "query": query,
            "expected_complex": expected_complex,
            "predicted_complex": predicted_complex,
            "correct": correct,
            "error_type": error_type,
            "reason": reason,
            "reasoning": reasoning,
            "steps_count": len(predicted_steps),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "query": query,
            "expected_complex": expected_complex,
            "predicted_complex": f"ERROR: {e}",
            "correct": False,
            "error_type": "exception",
            "reason": reason,
            "reasoning": str(e),
            "steps_count": 0,
        }


async def run_all_tests():
    print(f"\n{'='*60}")
    print(f"Planner 复杂度判断准确率测试")
    print(f"测试样例数: {len(PLANNER_TEST_CASES)}")
    print(f"{'='*60}\n")

    results = []
    for i, (query, expected, reason) in enumerate(PLANNER_TEST_CASES):
        r = await run_single_test(query, expected, reason)
        results.append(r)
        status = "✓" if r["correct"] else "✗"
        complex_mark = "复杂" if r["predicted_complex"] else "简单"
        expected_mark = "复杂" if r["expected_complex"] else "简单"
        print(f"  [{i+1:3d}/{len(PLANNER_TEST_CASES)}] {status}  "
              f"期望={expected_mark:<4} 预测={complex_mark:<4}  "
              f"步骤={r['steps_count']:2d}  | {r['query'][:35]}")

    return results


def print_report(results: list):
    print(f"\n{'='*60}")
    print("详细报告")
    print(f"{'='*60}")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100 if total else 0

    print(f"\n整体准确率: {correct}/{total} = {accuracy:.1f}%\n")

    # 按期望类型统计
    print(f"{'类别':<20} {'期望数':>6} {'正确数':>6} {'准确率':>8}")
    print("-" * 50)

    for label, name in [("simple", "简单任务"), ("complex", "复杂任务")]:
        expected_val = False if label == "simple" else True
        expected_count = sum(1 for r in results if r["expected_complex"] == expected_val)
        true_positive = sum(1 for r in results
                            if r["expected_complex"] == expected_val
                            and r["predicted_complex"] == expected_val)
        acc = true_positive / expected_count * 100 if expected_count else 0
        print(f"{name:<20} {expected_count:>6} {true_positive:>6} {acc:>7.1f}%")

    # 混淆矩阵
    print(f"\n混淆矩阵 (行=期望, 列=预测):")
    print(f"{'':>18}{'simple':>10}{'complex':>10}")
    print("-" * 40)

    for expected_val, expected_name in [(False, "simple"), (True, "complex")]:
        print(f"{expected_name:>18}", end="")
        for predicted_val in [False, True]:
            count = sum(
                1 for r in results
                if r["expected_complex"] == expected_val and r["predicted_complex"] == predicted_val
            )
            print(f"{count:>10}", end="")
        print()

    # 错误样例
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n错误样例分析 ({len(errors)} 条):")
        print("-" * 60)

        # 按错误类型分组
        by_type = {}
        for r in errors:
            t = r["error_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r)

        for error_type, errs in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{error_type}] ({len(errs)} 条):")
            for r in errs:
                print(f"    查询: {r['query']}")
                print(f"    期望: {'复杂' if r['expected_complex'] else '简单'}  "
                      f"预测: {r['predicted_complex']}  原因: {r['reason']}")
                if r.get("reasoning"):
                    print(f"    推理: {r['reasoning'][:100]}")
                print()

    # 快速路径分析（rule-based vs LLM）
    simple_count = sum(1 for r in results if not r["expected_complex"])
    complex_count = sum(1 for r in results if r["expected_complex"])
    simple_correct = sum(1 for r in results
                         if not r["expected_complex"] and r["correct"])
    complex_correct = sum(1 for r in results
                          if r["expected_complex"] and r["correct"])

    print(f"\n快速路径 (rule-based) 分析:")
    print(f"  简单任务: {simple_correct}/{simple_count} = "
          f"{simple_correct/simple_count*100:.1f}% 准确率")
    print(f"  复杂任务: {complex_correct}/{complex_count} = "
          f"{complex_correct/complex_count*100:.1f}% 准确率")

    # 保存结果
    output_path = os.path.join(
        os.path.dirname(__file__),
        "planner_test_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
            },
            "results": [
                {
                    "query": r["query"],
                    "expected_complex": r["expected_complex"],
                    "predicted_complex": r["predicted_complex"],
                    "correct": r["correct"],
                    "error_type": r["error_type"],
                    "reason": r["reason"],
                    "reasoning": r["reasoning"],
                    "steps_count": r["steps_count"],
                }
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {output_path}")


async def main():
    results = await run_all_tests()
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
