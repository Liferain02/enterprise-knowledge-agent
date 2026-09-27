# 秋招优化方向与 Harness 评估

> 评估日期：2026-09-26。本文只把已经实现的能力写成事实，把外部框架写成可选方案，不把“接入框架”本身当成效果。

## 1. 当前 Harness 到什么程度

当前项目可以准确称为：**LangGraph 主图之上的业务执行 Harness v1**。

它已经具备四类核心能力：

| 能力 | 当前实现 | 面试中可以怎么说 |
|---|---|---|
| 能力边界 | 路由决定 skill，代码白名单决定工具，模型不能临时发明工具 | “模型只在本轮授权的工具集合内行动” |
| 预算边界 | 模型次数、工具次数、总时限、工具时限、输入/输出字符数、模型输出 token 上限 | “把提示词里的软约束下沉成执行器硬约束” |
| 过程验证 | 整批 tool call 先校验名称、ID 和参数 schema，再执行；工具失败会保留为 ToolMessage | “避免执行一半才发现第二个调用越权或参数错误” |
| 失败与证据 | `run_id`、事件序号、工具结果 hash、节点结束后的 checkpoint 报告 | “可以解释为什么停、调用了什么、消耗了多少” |

这已经超过“给 LLM 加几个工具”的 Agent Demo，但还不是完整的生产级 Agent Platform。当前缺少：

- 跨服务 trace：API、Planner、检索、模型、工具、MySQL/Qdrant/Redis 还没有统一 trace tree。
- 中途持久化：目前主图结束时保存报告，进程在工具中间崩溃时不能从任意步骤恢复。
- 强制取消：异步等待可取消，但同步工具不能被 Python 强制终止。
- 副作用策略：当前白名单主要是纯函数；文件写入、远程命令、邮件等能力还没有审批、幂等和沙箱。
- 独立效果评测：已有固定回归和故障注入，但还没有独立盲测集、真人标注一致性和真实模型任务完成率。

所以面试时不要说“实现了完整 Harness”或“保证 Agent 安全”，更准确的表述是：**实现了一个有界、可回归、可解释的 operation 执行 Harness v1，并明确了生产化缺口。**

## 2. Harness、LangGraph、LangSmith 的关系

可以用三层来解释：

```text
LangGraph：状态图、节点顺序、checkpoint、主流程
业务 Harness：工具白名单、预算、schema 校验、停止条件、回退、审计事件
LangSmith/Langfuse：trace、可视化、评测数据集、实验对比、线上反馈
```

LangSmith 将一次模型调用、工具调用或检索视为 run，把一次请求中的多个 run 组织成 trace，并支持 thread、trajectory、tag、metadata 和 feedback；这正好适合作为当前 `run_id`/`harness_report` 的外部观测层。LangSmith 也支持对 LangChain、LangGraph 等集成自动追踪，或使用 `@traceable`、trace context、RunTree 做手动埋点。它负责记录和分析，不替代本项目的工具白名单、ACL、预算和停止逻辑。

LangSmith 的 evaluation API 可以把应用函数绑定到数据集，运行 evaluator，并用实验结果做版本比较；大规模异步评测可以配置并发度。对本项目来说，它适合承载“固定问题集 + 真实模型运行 + 评分器 + 版本对比”，但当前离线环境仍应保留本地 JSONL/pytest 作为不依赖网络的基线。

Langfuse 更适合作为自托管观测备选：官方文档强调 trace 中记录 prompt、response、token、latency、工具和检索步骤，并提供 scores、成本、延迟、质量看板等能力。它也会引入自托管服务和数据治理成本，因此不建议为了秋招演示同时接入 LangSmith 和 Langfuse。选一个即可。

Ragas 的定位更偏 RAG/Agent 评估指标，例如 context precision、context recall、faithfulness、tool call accuracy、agent goal accuracy。它可以补充评分层，但不能替代运行时 Harness；评分器也不能替代 ACL 和工具权限。

DeepAgents 可以作为 Harness 设计参考。其公开仓库把长任务的上下文管理、持久化、子 Agent、人工介入和 LangSmith 观测放在一个可扩展 harness 中。当前项目不需要整体替换为 DeepAgents：主图和科研任务边界已经存在，直接引入会扩大依赖和行为面。面试时可以说“参考其长任务和上下文管理思想，保留本项目的显式预算和科研 ACL 约束”。

## 3. 推荐的优化顺序

### P0：补一份独立评测契约

优先级最高，因为它最能提升面试可信度，也不需要新服务。

把数据集分成三部分：

1. **路由集**：operation、knowledge、writing、general，包含粘贴正文、反例和多轮续接。
2. **检索集**：问题、允许来源、拒绝来源、用户角色、期望 top-k。
3. **工具集**：输入、期望结构化结果、越权/超时/错误工具调用。

每条记录固定：`case_id`、输入、用户上下文摘要、期望结果、数据集版本、评测脚本版本。报告至少包含：

- 路由 macro-F1 和混淆矩阵。
- Recall@k、MRR、引用来源命中率。
- ACL violation，目标为 0。
- tool-call 参数通过率、越权拒绝率、预算停止率。
- P50/P95 延迟、缓存命中率、模型调用次数。

当前 50 条路由回归仍属于自编验收集，不能冒充独立盲测。下一步只需加入独立时间切分的 30～50 条样例，就能把“代码通过”升级成“版本对比证据”。

### P1：给 Harness 加统一 trace context

不改主图，只统一一个 `trace_id`，并让以下节点传递它：

```text
API request
→ planner
→ retrieval
→ generation/operation
→ tool call
→ storage/cache
```

本地先记录 `trace_id`、`run_id`、节点名、状态、耗时、模型调用次数和错误类型，禁止记录完整密级文档和用户敏感内容。未来接 LangSmith 时，把这些字段映射为 run metadata/tags；这样迁移观测平台不会重写业务逻辑。

### P1：做一次“检索质量—成本”消融

只比较三组，不要同时试十种算法：

| 组别 | 组成 | 观察指标 |
|---|---|---|
| A | Qdrant 向量 | Recall@k、P95、embedding 次数 |
| B | 向量 + BM25 + RRF | Recall@k、MRR、P95 |
| C | B + reranker | Recall@k、引用命中、成本、P95 |

如果 C 只提升很少但延迟和百炼调用明显增加，就保留 B 为默认，C 作为显式高质量模式。这种有负结果的决策比“把所有检索组件都打开”更适合面试。

### P1：补数据质量检查

不增加数据库，只在入库完成时统计：空块率、chunk 长度分布、重复 chunk 比例、来源缺失率、staging 超时数、active 版本数。出现空块、来源为空、同一文档多个 active 版本时直接报警或失败。它比继续调 chunk overlap 更能证明入库可靠性。

### P2：论文写作功能从聊天升级为“可验收产物”

先做一个小闭环：用户提供原文、目标段落、实验数据和引用；系统输出修改稿、数字变化表、引用占用表和“需要作者确认”的修改项。先不做完整 Word/LaTeX 工作台，不做自动投稿和自动改全文。面试中可以展示“保留数字和引用的写作修改”，比展示更多按钮更有说服力。

## 4. LangSmith 是否可以在面试中引用

可以引用，但要区分三种说法：

**已经使用：** 只有在项目实际配置并产生 trace、数据集和实验报告后才能这样说。

**已设计兼容：** 当前项目已经有 `run_id`、事件、检索指标和 checkpoint，可以说“执行报告字段预留了接入 LangSmith/Langfuse 的位置”，但不能说已经接入。

**参考其思想：** 可以说“参考 LangSmith 的 run/trace、dataset/evaluator 和实验对比思想”，这是当前最准确的表述。

推荐的面试表达：

> 我没有把 LangSmith 当作 Agent 的安全边界。项目内部先实现了可测试的 Harness：限制工具集合、模型轮数、调用次数、超时和输入输出大小，并保存结构化执行报告。LangSmith 适合作为后续观测和评测平台，把 Planner、Retriever、Tool、Generator 组织成一条 trace，再用固定数据集比较不同 prompt、reranker 和模型版本。这样即使观测平台不可用，业务执行边界仍然有效。

如果面试官问“为什么不直接用 LangSmith 解决”：

> LangSmith 能帮助我看见和比较运行过程，但不会替我决定哪些工具有权限、用户是否能看到某个 Qdrant chunk，也不会自动保证跨 MySQL/Qdrant 更新一致性。因此运行边界留在代码里，观测和评测平台放在外层。

## 5. 暂时不要做的事情

为了避免过度设计，当前不建议：

- 同时接入 LangSmith、Langfuse、Arize、Phoenix。
- 为普通 operation 引入 Supervisor、动态多 Agent fan-out。
- 没有真实并发和恢复需求就增加分布式锁、outbox、消息总线。
- 没有真实数据就校准 Jev confidence 或宣称路由提升。
- 把所有工具都改成 MCP，纯本地函数没有必要经过远程协议。
- 只为了简历堆更多 skill，却没有对应数据集和失败用例。

## 6. 秋招最终交付清单

1. 一张主图：Planner → Retrieval/Operation → Generation → Checkpoint。
2. 一张 Harness 图：Allowlist → Budget → Schema validation → Tool → Event/Trace → Stop。
3. 一份 30～50 条独立检索/路由评测报告。
4. 一次 A/B/C 检索消融，带 Recall、MRR、P95 和成本。
5. 三个故障演示：embedding 失败保留旧版本、工具超时停止、ACL 文档不泄露。
6. 一段一分钟回答，明确“我实现了什么、借鉴了什么、还没证明什么”。

当前最合适的项目定位是：**有边界的科研 RAG + 论文写作工具 + 可评测执行 Harness**。不要把项目包装成通用 AGI 平台。
