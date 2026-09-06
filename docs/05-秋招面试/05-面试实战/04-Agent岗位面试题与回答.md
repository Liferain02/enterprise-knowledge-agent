# Agent 岗位面试题与回答

> 面向秋招的实验室科研助手面试材料。更新日期：2026-09-07。

## 1. 使用说明与证据边界

这份文档不是“背题库”，而是一套把通用 Agent 问题落到本项目的练习稿。每道题都按四步准备：先给结论，再讲实现，指出代码证据，最后主动说明限制。

题目根据公开技术资料、开源项目文档和常见岗位能力要求归纳，不是任何公司的真实题库。公开资料主要包括：

- [Anthropic：Building effective agents](https://www.anthropic.com/research/building-effective-agents)：区分工作流和 Agent，强调先用简单、可组合的模式，再根据任务需要选择路由、并行或评审循环。
- [LangGraph：Low-level concepts](https://langchain-ai.github.io/langgraph/concepts/low_level/)：用状态、节点、边和检查点组织可恢复的图执行。
- [Anthropic Cookbook：Tool use](https://github.com/anthropics/claude-cookbooks/tree/main/tool_use)：展示工具 schema、工具调用循环和工具结果如何回到模型上下文。
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)：总结提示注入、过度代理权限、不安全输出和敏感信息泄露等风险。

这些来源用于解释通用概念，不证明本项目已经解决所有生产级问题。项目结论仍以代码、冻结数据集和评测报告为准。

## 2. 能力地图

面试官通常从六条线考察 Agent 候选人：

| 能力线 | 常见追问 | 本项目证据 |
|---|---|---|
| 任务建模 | Agent 和工作流有什么区别？何时需要多 Agent？ | Normal 默认链、显式 Deep 固定链 |
| 检索与生成 | 为什么混合检索？如何防幻觉？ | BM25+向量、RRF、Rerank、引用校验 |
| 状态与编排 | 状态放哪里？失败如何恢复？ | `AgentState`、LangGraph checkpoint、统一 Finalizer |
| 工具与安全 | 工具 schema、越权、提示注入怎么处理？ | 计算、时间、统计和受限文件工具；ACL 前置过滤 |
| 评测与归因 | 指标是什么？如何证明优化有效？ | Retrieval-only、Development、Blind、Regression 分层 |
| 工程取舍 | 延迟、成本、降级、可观测性怎么办？ | Deep opt-in、Rerank 降级、SSE、调用与 token 统计 |

## 3. 基础概念题

### 问题：Agent 和固定工作流有什么区别？

**20 秒回答：**固定工作流的步骤、分支和边界由代码预先定义；Agent 会根据当前状态决定下一步行动，灵活性更高但更难测试和控制。本项目把稳定的 Normal 做成固定工作流，只在用户显式选择 Deep 时启用固定三角色链，保留必要的协作收益而不放大不确定性。

**深入解释：**Anthropic 将 workflow 理解为预定义代码路径，将 agent 理解为模型动态决定工具和步骤。判断标准不是“用了 LangGraph 就是 Agent”，而是下一步是否由模型在运行时选择。项目 Planner 只选择已有分支，不生成任意执行计划，因此它是规则路由器，不是自由规划 Agent。

**本项目证据：**`src/agent/agents/planner.py` 的 `planner_node()` 和 `route_from_planner()`；`src/agent/graph.py` 的 `create_multi_agent_graph()`。

**限制：**当前 Deep 仍是固定 Researcher→Analyst→Reviewer，不支持动态新增角色；这是可测试性和成本边界，不是能力缺失的掩饰。

### 问题：为什么不能“问题越复杂，Agent 越多”？

**20 秒回答：**复杂度增加不等于增加角色就会提升质量。角色越多，调用次数、状态协议和失败组合越多；如果失败来自缺文档或错误前提，增加 Agent 也不能创造证据。本项目用评测先定位问题，再决定是否增加复杂度。

**深入解释：**多 Agent 适合任务天然可拆、子任务可以并行或需要互相审查的场景。它不适合简单事实查询、低延迟交互或主要瓶颈在数据入库的场景。项目的 26 条 V2 对比中，Deep 文档召回提高，但 premise 从 0.667 降到 0.167，P50 从 6.6 秒升到 27.5 秒，所以不能自动升级。

**本项目证据：**`docs/05-秋招面试/03-评测与实验/04-实验结果解读与停止决策.md` 的 Blind V2 表格和停止条件。

**不要这样说：**“多 Agent 一定比单 Agent 更智能”或“模型自己会找到最佳流程”。

### 问题：ReAct、规划器和执行器分别解决什么问题？

**20 秒回答：**ReAct 是思考与行动交替的通用循环；规划器负责把目标拆成步骤；执行器负责调用检索或工具并返回结果。本项目没有把三者做成无限自由循环，而是用 Planner 做规则路由，Researcher 做有限拆题，后续角色只消费结构化状态。

**深入解释：**自由 ReAct 的优点是灵活，缺点是循环次数、工具权限和输出协议难以预测。对实验室资料问答，检索和引用边界比自由探索更重要，因此将可变部分限制在最多 4 个子问题和一次修订内。

**本项目证据：**`research_team.py` 中 `MAX_SUBQUESTIONS=4`、`MAX_EVIDENCES=12` 和 `route_after_reviewer()`。

### 问题：Tool Calling 是怎样工作的？

**20 秒回答：**先把工具名称、参数 schema 和用途交给模型；模型返回结构化工具调用；运行时校验参数并执行工具；工具结果以新的消息回到上下文；模型再生成最终答案。工具本身必须有权限、超时和错误返回边界。

**深入解释：**工具调用不是让模型直接执行 Python，而是模型输出调用意图，宿主程序决定是否执行。参数 schema 能减少格式错误，但不能代替授权。项目的计算和科研统计工具是确定性函数，文件工具通过 MCP 作用域限制在 `data/knowledge`。

**本项目证据：**`src/agent/skills/*/scripts/tools.py`、`src/agent/tools/mcp_adapter.py`、`chat_service.py` 的 `on_tool_start/on_tool_end` 事件处理。

**限制：**当前工具数量有意保持很少，没有把所有实验室系统都暴露给模型。

### 问题：为什么需要结构化输出后再做确定性校验？

**20 秒回答：**Pydantic 结构化输出只保证字段能解析，不保证事实、引用和权限正确。项目在 Analyst、Reviewer 和最终来源阶段继续检查 source ID、ACL、声明类型和空答案，避免“格式合法但内容越权或无证据”。

**本项目证据：**`research_team.py` 的 `EvidencePackage`、`Claim`、`ReviewReport`；`graph.py` 的 `finalize_response_node()`。

## 4. 本项目架构题

### 问题：为什么 Normal 默认、Deep 必须显式选择？

**20 秒回答：**Normal 对多数单文档问题已经足够，延迟和 token 成本低，且 premise 指标更好；Deep 的优势是搜齐多证据，但成本约高 4 倍且安全门槛失败。让用户显式选择可以把成本和适用场景说清楚。

**代码与数据：**请求 schema 默认 `research_mode="normal"`；`route_from_planner()` 只有显式 `deep` 才进入 Researcher；V2 Deep/Normal P50 为 27.5/6.6 秒。

**追问：**未来自动路由怎么办？先增加带真实用户意图的新 Holdout，并把 premise 作为不可退化门槛；没有证据不自动打开。

### 问题：为什么只有一个 Planner？

**20 秒回答：**路由是低熵决策，规则足以覆盖问候、时间、计算、知识查询和显式 Deep。多个 Planner 会产生冲突和额外调用，却不增加证据质量。复杂知识问题在 Normal 中由 Query Expansion 处理，而不是再开一个 Planner。

**代码证据：**`planner.py` 的 `_quick_route()`、`planner_node()`；生产图只注册一个 `planner` 节点。

### 问题：Reviewer 为什么最多触发一次修订？

**20 秒回答：**无限评审循环会让延迟和 token 无上界，也可能在相同错误上循环。当前 Reviewer 只决定通过或一次 Revision，之后强制进入既有 Generation；`research_revision_count` 是显式成本上界。

**追问：**如果一次修订仍失败怎么办？保留 Reviewer 的拒绝、缺口和限制，生成带不确定性说明的答案，而不是继续隐藏重试。

### 问题：为什么 Deep 复用 Generation，而不是再加 Writer？

**20 秒回答：**最终答案格式、引用和拒答规则在 Normal 与 Deep 相同，重复 Writer 只会增加协议和偏差。Deep 的差异应该集中在证据搜集、声明分析和审核，最终输出复用已有 Generation 能减少变量。

### 问题：`AgentState` 为什么比 Agent 之间自由聊天更适合这个项目？

**20 秒回答：**显式状态能固定字段、权限和生命周期，便于测试、checkpoint 恢复和失败归因。项目让 Retrieval 写 `retrieved_docs`，Analyst 写 `analysis_report`，Reviewer 写 `review_report`，每个节点不能随意改身份或最终答案。

**代码证据：**`src/agent/graph.py` 的 `AgentState` 字段；`research_team.py` 的三个协议模型。

### 问题：checkpoint、摘要和 Mem0 有什么区别？

**20 秒回答：**checkpoint 是当前会话状态的可恢复快照；摘要是为减少上下文长度而压缩旧消息；Mem0 是可选的长期语义记忆，用来提供用户偏好或历史线索。三者不能互相替代，Mem0 也不是自动学习。

**限制：**Mem0 初始化、检索或保存失败时降级；记忆内容不能绕过 ACL，也不能替代当前知识库证据。

## 5. RAG 与检索题

### 问题：为什么同时使用 BM25 和向量检索？

**20 秒回答：**BM25 擅长文件名、缩写和精确术语，向量检索擅长同义表达和语义改写。两路互补后用 RRF 融合，减少只依赖一种检索器的偏差。

**代码证据：**`src/rag/retrieval/hybrid_retriever.py` 的 `search_with_scores()`。

### 问题：RRF 为什么比直接相加分数更稳？

**20 秒回答：**向量相似度和 BM25 分数不在同一量纲，直接相加需要额外校准；RRF 只使用名次，当前实现为 `Σ weight/(60+rank)`，因此对分数尺度不敏感。缺点是忽略绝对分差，所以后面仍可用 Rerank 做精排。

### 问题：Rerank 和 Grader 有什么区别？

**20 秒回答：**Rerank 做候选之间的相对排序；Grader 判断每个候选是否足够相关，并决定保留、改写或拒答。Rerank 先缩小候选，再运行 Grader 可以减少调用。Rerank 不可用时保留 Hybrid 原排序降级。

### 问题：Query Expansion 和 Researcher 拆题有什么区别？

**20 秒回答：**Query Expansion 仍服务于一次 Normal 检索，通过同义改写或有限拆分提高召回；Researcher 是 Deep 的证据规划，会为每个子问题记录证据、缺口和冲突，供 Analyst 形成 Claim。两者目的不同，不能因为问题复杂就自动切到 Deep。

### 问题：如何处理无答案和错误前提？

**20 秒回答：**先区分“库中没有文档”和“文档存在但排序靠后”。检索阶段记录 `NO_RESULTS`、缺口和改写历史；生成阶段禁止把缺证据补成事实；Deep 还由 Reviewer 检查 `premise_assessment`。无答案正确拒答率在快速基线中为 100%，但这只是固定集结果。

### 问题：Chunking 为什么不是越大越好？

**20 秒回答：**块太小会丢失上下文，太大会稀释相似度并增加生成 token。当前默认目标约 500 tokens、overlap 100，并把父标题拼接到块前；参数必须通过检索和答案评测一起验证。

### 问题：为什么 ACL 必须在 Rerank 前做？

**20 秒回答：**如果先把受限文档发给外部 Rerank 或 Grader，再在最终答案中删除，内容已经越过边界。项目先做 Chroma 过滤，再做 Python 二次检查，只有授权候选才进入 Rerank、Claim 和引用。

**代码证据：**`src/rag/retrieval/acl_filter.py`、`retriever.py`、`finalize_response_node()`。

## 6. 记忆、状态与知识演化题

### 问题：长期记忆怎样避免污染？

**20 秒回答：**只保存明确有价值的用户偏好或对话事实，带用户和会话范围；检索到的记忆只作为辅助上下文，不直接当作知识证据。新知识必须经过 Evidence→Claim→Review，并保留来源、版本和撤销能力。

**不要这样说：**“Mem0 会自动学习实验室知识”或“记忆比知识库更可信”。

### 问题：Evidence、Claim、ReviewReport 的关系是什么？

**20 秒回答：**Evidence 是带 source ID 和 ACL 结果的原始证据；Claim 是 Analyst 基于证据形成的可审查声明；ReviewReport 逐条检查支持关系、冲突、前提和引用缺口。只有通过校验的声明才进入最终回答或后续知识沉淀。

### 问题：知识库更新时如何避免新版本写坏旧版本？

**20 秒回答：**先解析、切块和写入新 chunks，成功后再删除同来源旧 chunks；失败时旧版本仍可用。SQLite 队列记录状态、重试和错误，服务重启可恢复陈旧任务。文件用 SHA-256 去重。

## 7. 工具与安全题

### 问题：如何防提示注入？

**20 秒回答：**把文档内容当作不可信数据，不允许它改变系统规则或工具权限；工具参数做 schema 校验和路径范围校验；ACL 在外部模型调用前完成；最终输出做引用和敏感信息检查。OWASP 将提示注入和过度代理权限列为 LLM 应用的主要风险。

**本项目边界：**MCP 文件工具只允许 `data/knowledge`，不提供任意 shell；受限文档不会先送到 Rerank 再删除。

### 问题：工具调用失败、超时或模型不可用怎么办？

**20 秒回答：**区分可重试和不可重试错误，设置有限次数和总时限；失败时返回可解释的降级结果，不伪造工具输出。Rerank 失败保留 Hybrid 排序，Mem0 失败跳过记忆，Deep 开关关闭时确定性提示并继续 Normal。

### 问题：为什么不让 Agent 直接执行任意文件或命令？

**20 秒回答：**Agent 输出不能直接获得宿主机权限，否则提示注入就可能变成任意操作。当前文件工具限定目录和操作类型，计算工具使用受限环境，任何新工具都要先定义 schema、授权范围、超时和审计字段。

## 8. 评测题

### 问题：Hit@K、Recall@K、MRR、NDCG 分别表示什么？

**20 秒回答：**Hit@K 看前 K 是否命中至少一个目标；Recall@K 看目标文档覆盖比例；MRR 看第一条正确结果出现得有多早；NDCG 同时考虑多条结果的相关等级和位置。它们评估检索，不等于最终答案正确。

### 问题：Citation Coverage 为什么只是代理指标？

**20 秒回答：**它通常只检查答案是否引用了预期来源或出现引用格式，不能完整判断句子是否被证据支持，也不能发现错误前提。因此项目把 citation、support、premise 和人工质量分开报告。

### 问题：为什么不直接使用 LLM-as-a-Judge？

**20 秒回答：**LLM Judge 容易偏好更长、更像自己的答案，且可能与被评模型共享偏差。当前模型模拟评分只用于失败定位，不冒充真人产品结论；若未来使用，先用独立人工样本校准相关性和重复运行方差。

### 问题：如何设计一次可靠的消融实验？

**20 秒回答：**一次只改一个因素，冻结代码、数据、模型和 ACL 语义；同时报告质量、安全、延迟、token 和调用次数；在 Development 修复，在 Blind 验证，在 Regression 防回退。若改动没有稳定收益，就默认关闭或删除。

### 问题：如何做失败归因？

**20 秒回答：**保存原始候选、RRF 结果、Rerank 结果、最终文档、阶段 trace 和答案指标，再按 query planning、lexical、semantic、rerank、ACL、missing document、generation 分类。这样不会用 Prompt 去修复本应补齐知识库的问题。

## 9. 性能与工程题

### 问题：SSE 为什么适合这个项目？

**20 秒回答：**问答是服务器向浏览器单向推送阶段状态和 token，SSE 比双向 WebSocket 更简单，浏览器原生支持重连。最终业务事实仍从 checkpoint 读取，不能把 token 流当作完整状态。

### 问题：如何降低 Deep 成本？

**20 秒回答：**先限制适用场景和调用上限，再优化单次提示和候选数；只有评测证明并行能降低墙钟时间时才引入并行。当前 Deep 已固定 4 个子问题、12 条证据和一次修订，Normal 仍是默认。

### 问题：SQLite、Chroma 和 Redis 的边界是什么？

**20 秒回答：**当前数据规模是实验室原型，SQLite 适合会话、任务和反馈，Chroma 适合本地向量持久化。Redis 只有在跨进程队列、共享限流或缓存成为真实瓶颈时才有必要；不能为了“看起来像生产系统”提前引入。

## 10. 编码与白板题

### 题目：手写 RRF

```python
def rrf(rank_lists, weights=None, k=60):
    weights = weights or [1.0] * len(rank_lists)
    scores = {}
    for weight, items in zip(weights, rank_lists):
        for rank, doc_id in enumerate(items, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
```

说明时补充：实际实现还要做文档身份去重、ACL 过滤和保留来源元数据；`k=60` 是稳定的平滑常数，不是当前小数据集上过拟合出来的最优值。

### 题目：设计 ACL 过滤函数

回答顺序：先定义 `UserContext`，再按 `visibility/project/department/role` 生成存储层过滤条件，最后对返回对象做 Python 二次校验；测试 anonymous、普通用户、项目成员、授权用户和拒绝用户的矩阵，并验证拒绝文档不会进入 Rerank、Claim、Trace 和 Final。

### 题目：设计有限重试

```text
调用
  → 判断错误类型
  → 可重试且未超过次数/总时限？
      是：指数退避后重试
      否：记录错误并走降级分支
```

必须说明：参数校验、权限拒绝和缺文档不是重试错误；重试次数是成本上界；每次重试要写入 trace，避免排障时只看到最后一次错误。

### 题目：设计工具调用状态机

```text
READY → CALLING → SUCCEEDED → ANSWER
             └→ FAILED → RETRYING（有限）→ FALLBACK
```

状态中至少保存工具名、规范化参数、开始/结束时间、错误类别和是否改变外部状态。读文件等只读工具可以自动重试，写操作必须有更严格的确认和幂等设计；本项目目前没有高风险写工具。

## 11. 连续追问清单

面试官可能沿以下顺序深入：

1. 你说 Deep 召回更高，为什么不默认？答：premise 退化、延迟和 token 成本更高。
2. 你说 ACL violation 为 0，能证明绝对安全吗？答：只能证明冻结矩阵和污染样本中未观察到违规。
3. 你说引用支持率 97.9%，是不是 Faithfulness？答：不是，当前是自动代理指标。
4. H08/H16 为什么不是检索器失败？答：目标文档不在冻结索引中，归因是 missing document。
5. 为什么不加第四个 Agent？答：已知失败不是角色缺失，增加角色不能修复数据和 premise 问题。
6. 你会如何继续优化？答：只处理真实 ACL、入库、SSE 故障，或在获得新标注后重开评测；不凭感觉扩张架构。

## 12. 最容易说错的表述

| 不要说 | 建议说 |
|---|---|
| 企业级知识库平台 | 实验室科研助手，当前是小规模原型 |
| Deep 全面提升 | Deep 在固定集上搜齐证据更好，但 premise 和成本退化 |
| 独立双盲已证明 | 已生成匿名评分包；当前没有新的独立真人产品结论 |
| 解决了幻觉 | 无证据时确定性拒答，并用引用校验降低风险 |
| Mem0 自动学习知识 | Mem0 提供辅助会话记忆，不替代知识库证据 |
| 使用了 LangGraph 所以是自主 Agent | LangGraph 承载显式状态图，Planner 和 Deep 的自由度是受限的 |
| Redis/GraphRAG 已规划必须上线 | 当前没有证据证明需要，保持 SQLite/Chroma 简洁边界 |

## 13. 面试前自测

- 能在 20 秒内说清项目定位、Normal/Deep 边界和最大负结果。
- 能手画请求链路，并指出 `AgentState`、EvidencePackage、Claim、ReviewReport 的写入节点。
- 能解释 RRF、Rerank、ACL 前置过滤、无答案拒答和失败归因。
- 能说出 Deep/Normal 的召回、premise、P50 和调用次数对照，并主动说明快照限制。
- 能用一个真实故障讲出“现象—假设—证据—修复—验证—剩余风险”。
- 能回答为什么没有增加 Agent、Redis、GraphRAG 和自动路由。
