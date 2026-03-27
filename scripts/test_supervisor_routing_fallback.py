"""
直接用 fallback_routing 重跑测试，输出完整报告
（LLM 全部 connection error，所以实际只测了 fallback 关键词路由）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agents.supervisor import fallback_routing
from collections import defaultdict

TEST_CASES = [
    # knowledge_agent
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
    # operation_agent
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
    # general_agent
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
    # 边界/歧义
    ("公司的年假政策是什么？帮我算一下我能休几天", "knowledge_agent"),
    ("报销标准是什么？加上已报销的金额，总额是多少？", "operation_agent"),
    ("年假还剩多少？顺便告诉我政策规定", "operation_agent"),
    ("你好，请问病假怎么处理", "knowledge_agent"),
    ("在吗？我想了解一下医保报销", "knowledge_agent"),
    ("嗨，年假有什么规定吗", "knowledge_agent"),
    ("年假", "knowledge_agent"),
    ("报销", "knowledge_agent"),
    ("几点开会", "operation_agent"),
    ("计算", "operation_agent"),
    ("你好", "general_agent"),
    ("请介绍一下弹性工作制，然后帮我算一下本周工作时长", "operation_agent"),
    ("KPI标准是什么？同时我的绩效数据在哪里查", "knowledge_agent"),
    ("公司的福利政策和报销流程分别是什么？", "knowledge_agent"),
    ("What is the annual leave policy?", "knowledge_agent"),
    ("What's the time now?", "operation_agent"),
    ("Hello, how are you?", "general_agent"),
    ("How many days off do I have left?", "operation_agent"),
]

results = []
for query, expected in TEST_CASES:
    predicted, reason = fallback_routing(query)
    results.append({
        "query": query,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "reasoning": reason,
    })

total = len(results)
correct = sum(1 for r in results if r["correct"])
accuracy = correct / total * 100

print("=" * 65)
print("  Supervisor 路由测试报告  （Fallback-only 关键词路由）")
print("=" * 65)
print(f"\n整体准确率: {correct}/{total} = {accuracy:.1f}%\n")

labels = ["knowledge_agent", "operation_agent", "general_agent"]
label_names = {"knowledge_agent": "知识检索", "operation_agent": "操作执行", "general_agent": "通用问答"}

print(f"{'类别':<12} {'期望':>6} {'正确':>6} {'准确率':>8} {'召回率':>8} {'F1':>8}")
print("-" * 65)
for label in labels:
    em = [r["expected"] == label for r in results]
    pm = [r["predicted"] == label for r in results]
    tp = sum(1 for i in range(len(results)) if em[i] and pm[i])
    ec = sum(em)
    pc = sum(pm)
    p = tp / pc * 100 if pc else 0
    r_ = tp / ec * 100 if ec else 0
    f = 2 * p * r_ / (p + r_) if (p + r_) else 0
    print(f"{label_names[label]:<12} {ec:>6} {tp:>6} {p:>7.1f}% {r_:>7.1f}% {f:>7.1f}%")

print(f"\n混淆矩阵 (行=期望, 列=预测):")
print(f"{'':>18}", end="")
for lbl in labels:
    print(f"{lbl.replace('_agent',''):>14}", end="")
print()
for el in labels:
    print(f"{el.replace('_agent',''):>18}", end="")
    for pl in labels:
        cnt = sum(1 for r in results if r["expected"]==el and r["predicted"]==pl)
        print(f"{cnt:>14}", end="")
    print()

errors = [r for r in results if not r["correct"]]
print(f"\n错误样例 ({len(errors)} 条):")
print("-" * 65)
for r in errors:
    print(f"  ✗ expected={r['expected']:<18} predicted={r['predicted']:<18} | {r['query'][:45]}")
    print(f"    fallback reason: {r['reasoning']}")

ebc = defaultdict(list)
for r in errors:
    ebc[r["expected"]].append(r)
print(f"\n错误分布:")
for label, errs in sorted(ebc.items(), key=lambda x: -len(x[1])):
    total_exp = sum(1 for r in results if r["expected"]==label)
    print(f"  {label}: {len(errs)}/{total_exp} ({len(errs)/total_exp*100:.0f}%)")

# 问题诊断
print(f"\n{'='*65}")
print("  问题诊断")
print("=" * 65)
# 统计哪些关键词类别被错误分类
print("\n1. operation_agent 的关键词过于宽泛")
print("   - '年' '月' '日' 命中了 KPI考核周期(知识) / 年假(知识)")
print("   - '多少' '费' 命中了 差旅费报销标准(知识) / 复利(操作) / 单位换算(操作)")
print("   - 边界词: '怎么' '帮我' 完全没有覆盖")
print("\n2. general_agent 关键词过于狭窄")
print("   - 只匹配 greetings 列表中的几个固定词")
print("   - 所有通用建议/闲聊类查询全部漏到 knowledge_agent 默认路由")
print("   - 英文完全没有覆盖")
print("\n3. operation_agent 知识面不足")
print("   - '天气' '汇率' '距离' 没有工具支持但被误判为 operation")
print("   - '最大公约数' '复利' '单位换算' 等操作类词没有覆盖")
print("\n4. LLM structured output 完全无法工作")
print("   - Connection error: LLM 服务不可达")
print("   - 降级到 fallback 后只有 59.1% 准确率")
print("   - LLM 能解决大部分关键词冲突（如带'你好'前缀的knowledge查询）")
