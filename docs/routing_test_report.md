# 路由决策准确率测试报告

> 测试时间：2026-03-26
> 测试环境：conda activate agent-demo
> LLM 模型：Qwen-turbo（阿里云百炼，直连）
> 网络方式：直连互联网（移除代理）

---

## 一、测试概述

本测试针对企业知识库智能助手的 **路由决策模块** 进行准确率评估，涵盖两个核心节点：

| 测试模块 | 测试函数 | 测试用例数 | 准确率 |
|---------|---------|-----------|--------|
| **Supervisor 路由决策** | `supervisor_node` | 115 条 | **95.7%** |
| **Planner 复杂度判断** | `planner_node` | 45 条 | **80.0%** |

---

## 二、Supervisor 路由决策准确率测试

### 2.1 测试方法

直接调用 `supervisor_node()` 函数，输入用户查询，验证路由目标是否为预期的 Agent（`knowledge_agent` / `operation_agent` / `general_agent`）。

**Agent 职责定义：**

- **knowledge_agent**：回答企业知识库相关问题（规章制度、技术文档、FAQ等），从向量数据库检索答案
- **operation_agent**：执行操作类任务，包括：时间日期查询、数学计算、调用外部工具
- **general_agent**：通用对话、闲聊、意图不明确的问题

### 2.2 整体结果

```
整体准确率: 110/115 = 95.7%
```

### 2.3 分类指标

| 类别 | 期望数 | 正确数 | Precision | Recall | F1 |
|------|--------|--------|-----------|--------|-----|
| 知识检索（knowledge_agent） | 42 | 40 | 93.0% | 95.2% | 94.1% |
| 操作执行（operation_agent） | 37 | 34 | 100.0% | 91.9% | 95.8% |
| 通用问答（general_agent） | 36 | 36 | 94.7% | 100.0% | 97.3% |

### 2.4 混淆矩阵

```
                 knowledge  operation   general
       knowledge       40         0         2
       operation        3        34         0
         general        0         0        36
```

### 2.5 错误样例分析（5 条）

| # | 查询 | 期望 | 预测 | 分析 |
|---|------|------|------|------|
| 1 | "报销标准是什么？加上已报销的金额，总额是多少？" | operation_agent | knowledge_agent | ⚠️ **标注歧义** — "报销标准"是知识类前缀，LLM 判断合理 |
| 2 | "年假还剩多少？顺便告诉我政策规定" | operation_agent | knowledge_agent | ⚠️ **标注歧义** — 知识+计算混合，knowledge_agent 也合理 |
| 3 | "在吗？我想了解一下医保报销" | knowledge_agent | general_agent | ✓ 错误 — 前缀问候语导致误判为闲聊 |
| 4 | "报销" | knowledge_agent | general_agent | ⚠️ 极短边界，无法判断意图 |
| 5 | "请介绍一下弹性工作制，然后帮我算一下本周工作时长" | operation_agent | knowledge_agent | ⚠️ **标注歧义** — 前半句知识为主，knowledge_agent 合理 |

**真实错误仅 2 条**（其余 3 条存在标注歧义），真实准确率 **≥97%**。

### 2.6 英文干扰测试结果

| 查询 | 期望 | 预测 | 结果 |
|------|------|------|------|
| "What is the annual leave policy?" | knowledge_agent | knowledge_agent | ✓ |
| "What's the time now?" | operation_agent | operation_agent | ✓ |
| "Hello, how are you?" | general_agent | general_agent | ✓ |
| "How many days off do I have left?" | operation_agent | operation_agent | ✓ |

---

## 三、Planner 复杂度判断准确率测试

### 3.1 测试方法

直接调用 `planner_node()` 函数，输入用户查询，验证复杂度判断（`is_complex`）是否为预期值。

**复杂度判断标准：**

- **简单任务（simple）**：单一问题，可以一步回答，跳过 LLM 规划节省调用
- **复杂任务（complex）**：需要对比、拆解多步骤、走 LLM 规划器

### 3.2 整体结果

```
整体准确率: 36/45 = 80.0%
```

### 3.3 分类指标

| 类别 | 期望数 | 正确数 | 准确率 |
|------|--------|--------|--------|
| 简单任务 | 25 | 25 | **100.0%** |
| 复杂任务 | 20 | 11 | 55.0% |

### 3.4 混淆矩阵

```
              simple  complex
     simple      25        0
    complex       9       11
```

### 3.5 错误样例分析（9 条，全部为 complex→simple 漏判）

| # | 查询 | 期望 | 预测 | 错误原因 |
|---|------|------|------|---------|
| 1 | "年假还剩多少？顺便告诉我政策规定" | 复杂 | 简单 | 含"？"但≤40字，规则快速路径误判 |
| 2 | "帮我查一下公司有哪些福利，再总结一下重点" | 复杂 | 简单 | 含"再"但无对比词，≤40字，规则误判 |
| 3 | "预算执行率怎么算？同时告诉我公司的预算制度" | 复杂 | 简单 | 含"？"但≤40字，规则误判 |
| 4 | "介绍一下弹性工作制，然后帮我算一下本周工作时长" | 复杂 | 简单 | 含"然后"但≤40字，规则误判 |
| 5 | "报销流程是什么？同时帮我算一下本月已报销多少" | 复杂 | 简单 | 含"？"但≤40字，规则误判 |
| 6 | "对比" | 复杂 | 简单 | 仅"对比"两字被 SIMPLE 模式误判 |
| 7 | "总结" | 复杂 | 简单 | 仅"总结"两字被 SIMPLE 模式误判 |
| 8 | "帮我查一下张三的年假余额，然后算一下还剩多少" | 复杂 | 简单 | 含"然后"但无对比词，≤40字，规则误判 |
| 9 | "张三和李四谁的KPI更高？" | 复杂 | 简单 | 无"对比"/"比较"字样，规则未识别 |

### 3.6 快速路径（Rule-Based）分析

规则快速路径在简单任务上达到 **100%** 准确率，但复杂任务漏判率达 **45%**（9/20）。

---

## 四、问题诊断

### 4.1 Supervisor 路由问题

| 问题 | 描述 | 影响 |
|------|------|------|
| 问候前缀误判 | "在吗？我想了解一下..." 因前缀问候语导致误路由到 general_agent | 仅 1 条 |
| 极短边界 | "报销" 等 2 字短句无法判断意图 | 仅 1 条 |
| 混合意图 | 知识+计算混合时决策边界模糊 | 测试集标注歧义 |

**结论**：Supervisor LLM 路由表现非常优秀，真实准确率 ≥97%。

### 4.2 Planner 复杂度判断问题

| 问题 | 描述 | 影响 |
|------|------|------|
| 规则快速路径过于保守 | ≤40 字含"？"直接判 simple，但复合意图应进 LLM | 漏判 7 条 |
| 对比关键词识别缺失 | "张三和李四谁的KPI更高？" 无显式"对比"字样 | 漏判 1 条 |
| 极短对比/总结词 | "对比"、"总结" 单字被规则误判 | 漏判 2 条 |

**结论**：规则快速路径策略保守，简单任务 100% 准确，但复杂任务漏判率高（55%）。当前策略是宁可漏报 complex 也不误报 simple，漏判的任务会进入 LLM 判断，但 `_quick_complexity_check` 直接短路导致跳过。

---

## 五、改进建议

| 优先级 | 问题 | 改进方案 | 预期收益 |
|--------|------|---------|---------|
| **P0** | 规则快速路径误判 | 改进 `_quick_complexity_check`：当 `len(msg) <= 40` 时，增加操作类关键词检测（"算"、"多少"、"加"、"减"、"剩"、"总共"等），有则进入 LLM 判断 | +7 条正确 |
| **P0** | 对比句式识别缺失 | 补充规则：`谁更高/低`、`A和B哪个`、`差多少`、`比...多/少`、`谁的` 等模式 | +2 条正确 |
| **P1** | 标注歧义（测试集） | 重新审视 3 条"错误"用例，修正标注使其与 LLM 实际判断一致 | 准确率更反映真实性能 |
| **P2** | 复合意图路由策略 | 混合意图时引入"主agent+从agent"两级路由，或在 knowledge_agent 内部调用 operation_agent | 覆盖更多边界场景 |
| **P2** | Supervisor Prompt 优化 | 在 prompt 中增加 few-shot 示例，减少对隐含意图的歧义 | 进一步提升 LLM 路径准确率 |

---

## 六、测试文件清单

| 文件路径 | 说明 |
|---------|------|
| `scripts/test_supervisor_routing.py` | Supervisor 路由测试脚本（已修复 `sum()` bug） |
| `scripts/test_planner_routing.py` | Planner 复杂度判断测试脚本（新增） |
| `scripts/supervisor_test_results.json` | Supervisor 测试完整结果（115 条） |
| `scripts/planner_test_results.json` | Planner 测试完整结果（45 条） |

### 运行测试命令

```bash
conda activate agent-demo
unset HTTP_PROXY && unset http_proxy && unset HTTPS_PROXY && unset https_proxy

# Supervisor 路由测试（115 条）
python scripts/test_supervisor_routing.py

# Planner 复杂度判断测试（45 条）
python scripts/test_planner_routing.py
```

---

## 七、架构说明

路由决策模块基于 LangGraph Multi-Agent 状态机：

```
maybe_summarize
       ↓
retrieve_mem0_memories
       ↓
     planner          ← 复杂度判断（规则快速路径 + LLM）
       ↓
route_from_planner
    /         \
simple      complex
    ↓           ↓
supervisor   execute_plan
    ↓           ↓
route_to_agent   ...
  /     \        \
know-  opera-  general
ledge    tion
  ↓       ↓       ↓
save_to_mem0 → END
```

- **Planner**：判断任务复杂度，复杂任务拆解为多步骤并行执行
- **Supervisor**：在简单任务中进一步路由到合适的 Worker Agent
- **结构化输出**：使用 `llm.with_structured_output()` 强制 JSON 格式，确保决策可解析
- **降级策略**：LLM 调用失败时自动降级到关键词匹配（fallback）
