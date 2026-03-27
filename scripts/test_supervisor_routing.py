"""
Supervisor 路由准确率测试
直接调用 supervisor_node，对测试数据集逐条评估，输出分类报告
"""
import asyncio
import json
import sys
import re
import os
from collections import defaultdict
from typing import Literal

# 确保 src 在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage
from src.agent.agents.supervisor import supervisor_node


# =============================================================================
# 测试数据集
# =============================================================================
# 每个样例: (query, expected_agent)
# expected: knowledge_agent / operation_agent / general_agent

TEST_CASES: list[tuple[str, str]] = [
    # =====================================================================
    # 1. knowledge_agent — 企业知识、制度、文档内容检索
    # =====================================================================
    ("公司的年假政策是什么？", "knowledge_agent"),
    ("加班调休是怎么规定的？", "knowledge_agent"),
    ("请介绍一下弹性工作制", "knowledge_agent"),
    ("离职流程需要几天？", "knowledge_agent"),
    ("报销流程是什么？", "knowledge_agent"),
    ("绩效考核的标准有哪些？", "knowledge_agent"),
    ("试用期转正有什么要求？", "knowledge_agent"),
    ("请假制度是怎么样的？", "knowledge_agent"),
    ("公司有什么福利待遇？", "knowledge_agent"),
    ("差旅费报销标准是多少？", "knowledge_agent"),
    ("新员工培训有哪些内容？", "knowledge_agent"),
    ("项目管理流程是什么？", "knowledge_agent"),
    ("技术架构文档在哪里？", "knowledge_agent"),
    ("API接口文档的地址是什么？", "knowledge_agent"),
    ("数据库设计规范有哪些？", "knowledge_agent"),
    ("代码审查标准是什么？", "knowledge_agent"),
    ("部署流程需要注意什么？", "knowledge_agent"),
    ("安全规范有哪些要求？", "knowledge_agent"),
    ("数据隐私政策是什么？", "knowledge_agent"),
    ("客户投诉处理流程是什么？", "knowledge_agent"),
    ("客户服务标准有哪些？", "knowledge_agent"),
    ("FAQ里面有关于密码重置的吗？", "knowledge_agent"),
    ("文档里怎么描述权限管理的？", "knowledge_agent"),
    ("知识库里有接口调试说明吗？", "knowledge_agent"),
    ("公司内部有哪些报销类型？", "knowledge_agent"),
    ("员工手册最新版本是什么时候更新的？", "knowledge_agent"),
    ("代码规范中关于命名有什么要求？", "knowledge_agent"),
    ("KPI考核周期是按季度还是按月？", "knowledge_agent"),
    ("会议纪要模板在哪里能找到？", "knowledge_agent"),
    ("内部培训申请表怎么填？", "knowledge_agent"),
    ("团建活动有什么规定？", "knowledge_agent"),
    ("门禁卡丢了怎么办？", "knowledge_agent"),
    ("公司班车路线有哪些？", "knowledge_agent"),

    # =====================================================================
    # 2. operation_agent — 时间日期、数学计算、工具调用
    # =====================================================================
    ("现在几点？", "operation_agent"),
    ("今天几月几号？", "operation_agent"),
    ("现在是几点几分？", "operation_agent"),
    ("今天星期几？", "operation_agent"),
    ("2025年春节是哪天？", "operation_agent"),
    ("帮我计算一下 123 加 456 等于多少？", "operation_agent"),
    ("100 乘以 25 再加 50 等于多少？", "operation_agent"),
    ("这个月的考勤天数是多少？", "operation_agent"),
    ("我的年假还剩多少天？", "operation_agent"),
    ("本月报销总额是多少？", "operation_agent"),
    ("帮我算一下这个月的工资税后多少", "operation_agent"),
    ("两个日期之间相差多少天？", "operation_agent"),
    ("从北京到上海的距离是多少公里？", "operation_agent"),
    ("当前汇率下 100 美元等于多少人民币？", "operation_agent"),
    ("帮我查一下北京现在的天气", "operation_agent"),
    ("明天的会议是几点开始？", "operation_agent"),
    ("帮我算一下季度目标完成了多少", "operation_agent"),
    ("这个季度的销售额增长了多少？", "operation_agent"),
    ("帮我计算ROI是多少", "operation_agent"),
    ("预算执行率怎么算？", "operation_agent"),
    ("今年已经过去多少天了？", "operation_agent"),
    ("距离下个季度还有几天？", "operation_agent"),
    ("12月的考勤汇总是多少？", "operation_agent"),
    ("1TB等于多少GB？", "operation_agent"),
    ("帮我算一下复利", "operation_agent"),
    ("现在的Unix时间戳是多少？", "operation_agent"),
    ("2024年有多少天？", "operation_agent"),
    ("帮我计算平均值", "operation_agent"),
    ("两个数的最大公约数怎么算？", "operation_agent"),
    ("帮我做个单位换算", "operation_agent"),

    # =====================================================================
    # 3. general_agent — 闲聊、问候、意图模糊
    # =====================================================================
    ("你好", "general_agent"),
    ("早上好", "general_agent"),
    ("在吗？", "general_agent"),
    ("嗨，最近怎么样？", "general_agent"),
    ("你好呀！", "general_agent"),
    ("有人吗？", "general_agent"),
    ("你好，我想问一下", "general_agent"),
    ("我想咨询一些事情", "general_agent"),
    ("你叫什么名字？", "general_agent"),
    ("你能做什么？", "general_agent"),
    ("介绍一下你自己", "general_agent"),
    ("帮我看看这个怎么弄", "general_agent"),
    ("有没有什么建议？", "general_agent"),
    ("随便聊聊", "general_agent"),
    ("今天心情不好", "general_agent"),
    ("周末有什么推荐吗？", "general_agent"),
    ("怎么看这个情况？", "general_agent"),
    ("你觉得这个方案怎么样？", "general_agent"),
    ("有什么好看的电影推荐吗？", "general_agent"),
    ("天气怎么样？", "general_agent"),
    ("这个项目压力大不大？", "general_agent"),
    ("为什么系统这么慢？", "general_agent"),
    ("这个bug怎么修？", "general_agent"),
    ("为什么代码跑不通？", "general_agent"),
    ("这个问题很奇怪", "general_agent"),
    ("随便问问", "general_agent"),
    ("我不太确定该问什么", "general_agent"),
    ("你能帮我个忙吗？", "general_agent"),
    ("这个功能什么时候上线？", "general_agent"),
    ("为什么版本又延期了？", "general_agent"),
    ("这个问题你遇到过吗？", "general_agent"),
    ("能聊聊AI发展趋势吗？", "general_agent"),
    ("有什么技术书籍推荐吗？", "general_agent"),
    ("最近有什么好用的工具吗？", "general_agent"),

    # =====================================================================
    # 4. 边界/歧义/混合场景（标注真实期望路由）
    # =====================================================================
    # 混合: 既有知识询问，又有计算
    ("公司的年假政策是什么？帮我算一下我能休几天", "knowledge_agent"),  # 知识为主，计算是后续
    ("报销标准是什么？加上已报销的金额，总额是多少？", "operation_agent"),  # 计算为主
    ("年假还剩多少？顺便告诉我政策规定", "operation_agent"),  # 计算为主

    # 歧义: 表面闲聊，实际有明确意图
    ("你好，请问病假怎么处理", "knowledge_agent"),  # 实际是问政策
    ("在吗？我想了解一下医保报销", "knowledge_agent"),
    ("嗨，年假有什么规定吗", "knowledge_agent"),

    # 边界: 短问句，关键词覆盖
    ("年假", "knowledge_agent"),  # 极短，关键词命中
    ("报销", "knowledge_agent"),
    ("几点开会", "operation_agent"),  # 关键词"几点"
    ("计算", "operation_agent"),
    ("你好", "general_agent"),  # 纯问候

    # 复杂多意图
    ("请介绍一下弹性工作制，然后帮我算一下本周工作时长", "operation_agent"),  # 两个问题，操作类
    ("KPI标准是什么？同时我的绩效数据在哪里查", "knowledge_agent"),  # 知识为主
    ("公司的福利政策和报销流程分别是什么？", "knowledge_agent"),  # 多子问题，同类

    # 英文干扰
    ("What is the annual leave policy?", "knowledge_agent"),
    ("What's the time now?", "operation_agent"),
    ("Hello, how are you?", "general_agent"),
    ("How many days off do I have left?", "operation_agent"),
]


# =============================================================================
# 测试执行
# =============================================================================

async def run_single_test(query: str, expected: str) -> dict:
    """运行单条测试"""
    state = {"messages": [HumanMessage(content=query)]}
    try:
        result = await supervisor_node(state)
        predicted = result.get("next_agent", "unknown")
        reasoning = result.get("supervisor_reasoning", "")
        reason = result.get("supervisor_reason", "")
        correct = predicted == expected
        return {
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "reasoning": reasoning,
            "reason": reason,
        }
    except Exception as e:
        return {
            "query": query,
            "expected": expected,
            "predicted": f"ERROR: {e}",
            "correct": False,
            "reasoning": "",
            "reason": str(e),
        }


async def run_all_tests():
    """运行全部测试"""
    print(f"\n{'='*60}")
    print(f"Supervisor 路由准确率测试")
    print(f"测试样例数: {len(TEST_CASES)}")
    print(f"{'='*60}\n")

    results = []
    for i, (query, expected) in enumerate(TEST_CASES):
        r = await run_single_test(query, expected)
        results.append(r)
        status = "✓" if r["correct"] else "✗"
        print(f"  [{i+1:3d}/{len(TEST_CASES)}] {status}  expected={expected:<18} predicted={r['predicted']:<18} | {r['query'][:40]}")

    return results


def print_report(results: list):
    """打印分类报告"""
    print(f"\n{'='*60}")
    print("详细报告")
    print(f"{'='*60}")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100 if total else 0

    print(f"\n整体准确率: {correct}/{total} = {accuracy:.1f}%\n")

    # 按类别统计
    labels = ["knowledge_agent", "operation_agent", "general_agent"]
    label_names = {"knowledge_agent": "知识检索", "operation_agent": "操作执行", "general_agent": "通用问答"}

    print(f"{'类别':<12} {'期望数':>6} {'正确数':>6} {'准确率':>8} {'召回率':>8} {'F1':>8}")
    print("-" * 60)

    for label in labels:
        expected_mask = [r["expected"] == label for r in results]
        predicted_mask = [r["predicted"] == label for r in results]
        true_positive = sum(1 for i in range(len(results)) if expected_mask[i] and predicted_mask[i])
        expected_count = sum(1 for v in expected_mask if v)
        predicted_count = sum(1 for v in predicted_mask if v)

        precision = true_positive / predicted_count * 100 if predicted_count else 0
        recall = true_positive / expected_count * 100 if expected_count else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        name = label_names.get(label, label)
        print(f"{name:<12} {expected_count:>6} {true_positive:>6} {precision:>7.1f}% {recall:>7.1f}% {f1:>7.1f}%")

    # 混淆矩阵
    print(f"\n混淆矩阵 (行=期望, 列=预测):")
    print(f"{'':>18}", end="")
    for label in labels:
        short = label.replace("_agent", "")
        print(f"{short:>10}", end="")
    print()

    for expected_label in labels:
        short = expected_label.replace("_agent", "")
        print(f"{short:>18}", end="")
        for pred_label in labels:
            count = sum(
                1 for r in results
                if r["expected"] == expected_label and r["predicted"] == pred_label
            )
            print(f"{count:>10}", end="")
        print()

    # 错误样例
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n错误样例分析 ({len(errors)} 条):")
        print("-" * 60)
        for r in errors:
            print(f"\n  查询: {r['query']}")
            print(f"  期望: {r['expected']}  预测: {r['predicted']}")
            if r["reasoning"]:
                print(f"  推理: {r['reasoning']}")
            if r["reason"]:
                print(f"  原因: {r['reason']}")

    # 按类别汇总错误
    if errors:
        error_by_expected = defaultdict(list)
        for r in errors:
            error_by_expected[r["expected"]].append(r)

        print(f"\n错误分布:")
        for label, errs in sorted(error_by_expected.items(), key=lambda x: -len(x[1])):
            print(f"  {label}: {len(errs)}/{sum(1 for r in results if r['expected']==label)} 条错误")

    # 保存完整结果
    output_path = os.path.join(os.path.dirname(__file__), "supervisor_test_results.json")
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
                    "expected": r["expected"],
                    "predicted": r["predicted"],
                    "correct": r["correct"],
                    "reasoning": r["reasoning"],
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
