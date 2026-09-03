# Enterprise Knowledge Agent：Codex 优化任务书

> 目标仓库：`Liferain02/enterprise-knowledge-agent`
>
> 参考项目：`dianayyds/research_copilot`
>
> 本文不是要求照抄参考项目，而是要求在**保持当前可信 RAG + 有界 Deep Research 主架构不变**的前提下，吸收参考项目中值得借鉴的设计：Research Run / Trace、分层记忆、Standalone / Step-back Query Rewrite、外部 Research Tool 抽象。

---

## 0. Codex 执行原则

在修改任何代码前，请先完整阅读并理解当前仓库，至少检查：

- `src/agent/graph.py`
- `src/agent/agents/planner.py`
- `src/agent/agents/research_team.py`
- `src/agent/memory/mem0_manager.py`
- `src/api/services/chat_service.py`
- `src/api/services/research_service.py`
- `src/api/controllers/research_controller.py`
- `src/rag/retrieval/query_expander.py`
- `src/rag/retrieval/hybrid_retriever.py`
- `src/rag/retrieval/acl_filter.py`
- `frontend/src/App.vue`
- Deep Research、ACL、Retrieval、Research Workspace 相关测试
- `tests/eval/` 下现有评测脚本与冻结数据

**以当前代码为最终事实来源。**  
如果本文中的文件名、字段名与最新代码有轻微变化，请按当前实现适配，不要机械照搬。

修改过程中必须遵守：

1. 不重写整个项目。
2. 不增加无必要的 Agent。
3. 不恢复 Supervisor / 动态 Worker / 自由协商式 Multi-Agent。
4. 不把 Normal 模式改成 Plan-ReAct。
5. 不让 Planner 自动进入 Deep Research。
6. 不恢复无限 CRAG / Rewrite / Re-search 循环。
7. 不为了“架构漂亮”引入新的中间件。
8. 不破坏已有 ACL fail-closed 语义。
9. 不删除已有冻结评测、历史结果和回归测试。
10. 每一项新能力必须有测试，并且最好能进入现有评测体系。

---

# 1. 当前架构必须保留的核心

当前项目已经完成过一次重要的架构收敛，这些设计是正确的，不能因为参考了 `research_copilot` 而推翻。

## 1.1 Normal 模式

保持：

```text
User Query
    ↓
Deterministic Planner
    ↓
Optional Query Expansion
    ↓
Hybrid Retrieval
(Vector + BM25 + RRF)
    ↓
Exact ACL Check
    ↓
Rerank
    ↓
Single Generation
    ↓
Citation
```

Planner 只做：

- 请求进入哪个已有分支；
- Knowledge Query 是否需要 Query Expansion。

Normal 模式不允许出现自由规划、多 Agent fan-out 或循环式 ReAct。

## 1.2 Deep Research

保持固定、有界链路：

```text
Researcher
    ↓
Analyst
    ↓
Reviewer
    ↓
最多一次 Revision
    ↓
Generation
```

继续保留：

- `MAX_SUBQUESTIONS`
- `MAX_TARGETED_QUERIES`
- `MAX_EVIDENCES`
- 固定 Claim schema
- Reviewer issue taxonomy
- 最多一次 revision
- research metrics
- failure attribution
- 显式 `research_mode=deep`

**不要增加 Supervisor，不要增加第四、第五个角色。**

---

# 2. 本轮优化总目标

本轮只做四个方向，其中 P0 必须优先完成。

```text
P0-1  Deep Research → 可持久化、可查看的 Research Run
P0-2  现有 Mem0 → Evidence-backed Layered Memory
P1    Query Expansion → 增加 Standalone + Step-back
P2    Deep Research → 增加统一 External Research Tool / Evidence Adapter
```

最终希望系统形成：

```text
可信检索
   ↓
有界 Deep Research
   ↓
可审计 Research Run
   ↓
经过验证的知识沉淀
   ↓
未来任务可信复用
```

---

# 3. P0-1：Research Run / Deep Research 可审计执行记录

## 3.1 为什么要改

当前 `research_team.py` 已经存在：

- `EvidencePackage`
- `AnalysisReport`
- `ReviewReport`
- `research_team_metrics`
- `research_trace`
- stage latency
- failure attribution

也就是说，**不要重新设计一套 Trace 系统**。

当前真正缺少的是：

> 把这些运行时 state 从“测试 / 调试数据”提升成正式产品数据：持久化、查询、前端展示。

---

## 3.2 新增 ResearchRun

优先复用当前 `research_service.py` 的 SQLite / Research Workspace 存储体系，不要额外引入 PostgreSQL、Redis Stream、OpenTelemetry 等。

建议增加：

```text
research_runs
```

字段建议：

```text
id
project_id              nullable
session_id
user_id
question
mode                     deep
status                   running / completed / failed

evidence_package_json
analysis_report_json
review_report_json
research_trace_json
metrics_json

final_answer
source_cards_json

created_at
completed_at
```

如现有 Research Workspace 已有更合适的数据模型，请融入现有设计。

---

## 3.3 Research Run 保存内容

一次 Deep Research 至少保存：

### Input

```text
question
user_id
session_id
project_id（如果能解析）
```

### Researcher

```text
subquestions
targeted queries
evidence ids
missing evidence
conflicts
```

### Analyst

```text
claims
claim_type
source_ids
confidence
limitations
premise_assessment
```

### Reviewer

```text
decision
issue_type
revision_instruction
targeted_queries
false_premise_detected
acl_verified
```

### Revision

```text
revision_count
是否触发 revision
为什么触发 / 为什么跳过
```

### Final

```text
final_answer
citations/source_cards
```

### Metrics

```text
llm_calls
retrieval_calls
input_tokens
output_tokens
elapsed_ms
stage_latency_ms
failure_attribution
```

---

## 3.4 接入位置

不要侵入每个 Agent 节点单独写数据库。

推荐：

```text
research_team.py
    ↓
继续只维护 state / trace

graph.py / chat_service.py
    ↓
Deep Research 完成
    ↓
统一构建 ResearchRun
    ↓
research_service.persist_run(...)
```

写入失败不能影响用户拿到最终答案：

```text
ResearchRun persistence failure
→ log warning
→ answer still returns
```

---

## 3.5 API

增加最小接口即可：

```text
GET /api/.../research-runs
GET /api/.../research-runs/{run_id}
```

如果 Research Workspace 已按 project 组织：

```text
GET /projects/{project_id}/research-runs
GET /projects/{project_id}/research-runs/{run_id}
```

不要一次设计复杂 CRUD。

Research Run 是系统生成的审计记录，第一版不需要用户手动创建、编辑。

---

## 3.6 ACL

Research Run 可能包含受限 Evidence，所以它本身也必须受权限保护。

要求：

1. 用户只能读取自己有权访问的 project run。
2. 读取 run 时不能仅信任历史缓存的 source card。
3. 如要返回 Evidence 详情，应再次执行当前用户的 `check_doc_access`。
4. 无法验证来源权限时，隐藏 Evidence 内容而不是 fail-open。
5. 不允许一个曾经有权限、现在已无权限的用户继续通过旧 Research Run 看受限原文。

---

## 3.7 前端

在 Deep Research 回答附近增加一个可折叠：

```text
Research Trace
```

第一版不需要复杂时间轴组件。

建议展示：

```text
Researcher
- 4 subquestions
- 8 evidences

Analyst
- 7 claims
- 1 conflict
- premise: supported

Reviewer
- REVISE
- citation_gap: 1
- missing_evidence: 1

Revision
- 1 round

Metrics
- Retrieval: ...
- LLM calls: ...
- Latency: ...
```

点击后再显示详细 Evidence / Claim / Reviewer 内容。

**默认收起，不干扰普通用户阅读答案。**

---

# 4. P0-2：把当前 Mem0 改造成 Evidence-backed Layered Memory

## 4.1 现状问题

当前仓库已经有 `Mem0MemoryManager`，并且聊天完成后异步调用 Mem0 保存对话。

这个能力不要被描述成“完全没有 Memory”。

但当前设计更接近：

```text
Conversation
    ↓
Mem0 自动提取
    ↓
User / Session semantic memory
```

对普通聊天可用，但对科研知识有一个核心风险：

> 模型生成的回答可能被直接沉淀成“长期事实”，而这些事实未必带证据、版本或 ACL。

因此本轮不是简单“新增 Memory”，而是：

> **把项目事实类记忆从 generic conversational memory 升级成 evidence-backed structured memory。**

---

## 4.2 三层定义

### A. Working Memory

用途：

- 当前会话上下文；
- 最近几轮问题；
- 用户当前正在讨论什么。

优先复用：

- 现有 session history；
- LangGraph checkpointer；
- 已有 Mem0 session 能力。

**不要为了 Working Memory 再建重复数据库。**

---

### B. Episodic Memory

定义：

> 一次真实发生的科研事件 / Research Run。

主要来源：

```text
Deep Research completed
Experiment completed
Meeting / project event（如果现有结构支持）
```

第一版优先从 Research Run 自动生成。

建议字段：

```text
episode_id
project_id
user_id
event_type = research_run
summary
run_id
question
final_answer_summary
source_ids
created_at
```

Episodic Memory 是“发生过什么”，不是“永远正确的事实”。

---

### C. Semantic Memory

只允许保存结构化类型：

```text
fact
decision
open_question
progress
preference
```

其中 `fact / decision / progress` 必须有来源。

建议模型：

```text
memory_id
project_id
memory_type

statement

source_type
source_ids
research_run_id

created_by
created_at

confidence
verified

acl_snapshot / source metadata
```

---

## 4.3 Memory Promotion Gate

这是本轮最重要的设计。

**禁止把 Final Answer 整段直接写成 Semantic Memory。**

Deep Research 完成以后，从 `AnalysisReport.claims` 中选择候选。

只有满足以下条件的 Claim 才允许自动进入事实类 Semantic Memory：

```text
claim.source_ids 非空
AND
Reviewer 对该 claim supported == true
AND
issue_type == none
AND
不存在 ACL 问题
AND
source ids 在当前 EvidencePackage 中真实存在
```

推荐：

```text
fact        → 可自动 promotion
comparison  → 可保存为 episode，默认不提升为永久 fact
inference   → 默认不自动提升
recommendation → 默认不自动提升
```

`decision` 最好只来自：

- 用户明确确认；
- 项目 / 实验结构化数据；
- 有明确 evidence 的项目结论。

---

## 4.4 读取 Memory 时再次做权限检查

不能因为 Memory 在过去被成功写入，就永久绕开 ACL。

Memory recall 时：

```text
query
  ↓
memory candidate retrieval
  ↓
resolve source_ids
  ↓
check_doc_access(current_user)
  ↓
只保留当前仍可见的 memory
```

对于没有可验证 source 的项目事实：

```text
默认不注入事实上下文
```

用户偏好类 memory 可以走单独逻辑，不需要文档 citation，但必须 user-scoped。

---

## 4.5 Mem0 的处理方式

不要为了重构而强行删除 Mem0。

推荐拆成两类：

```text
Mem0
→ user preference / conversational hints

Structured Evidence-backed Memory
→ project facts / decisions / research history
```

如果最终发现 Mem0 和新体系重复严重，可以在测试通过后逐步收敛。

第一阶段不要做大规模删除。

---

# 5. P1：Query Expansion 增加 Standalone Query 与 Step-back Query

## 5.1 保留现有能力

当前 `query_expander.py` 已有：

- Rule decomposition
- LLM decomposition
- HyDE
- keyword expansion

这些不要重写。

新增两个窄能力即可。

---

## 5.2 Standalone Query

解决多轮指代问题。

例如：

```text
Turn 1:
RDMA 项目最近一次延迟实验发现了什么？

Turn 2:
第二个问题后来怎么解决的？
```

第二轮检索前可重写：

```text
RDMA 项目最近一次延迟实验中发现的第二个问题后来如何解决？
```

触发条件建议：

- 当前 query 有“这个 / 那个 / 第二个 / 上面 / 刚才 / 它 / 这个方案”等明显指代；
- 存在可用会话历史；
- Normal knowledge query 才触发。

不要所有 query 都调用 LLM rewrite。

---

## 5.3 Step-back Query

解决“具体问题缺背景召回”的情况。

例如：

```text
为什么实验 B 的延迟突然恶化？
```

可以额外生成：

```text
影响该实验延迟的主要系统因素有哪些？
```

触发条件保持克制：

- why / 原因分析；
- failure diagnosis；
- 复杂技术机制；
- 原查询直接检索可能过窄。

---

## 5.4 Retrieval 流程

仍然必须是**单轮检索**：

```text
Original Query
   ↓
Optional Variants
   ├─ standalone
   ├─ decomposition
   ├─ step-back
   └─ hyde（已有）
   ↓
一次并行/批量 Retrieval
   ↓
RRF / merge
   ↓
Exact ACL
   ↓
Rerank
```

禁止：

```text
retrieve
→ judge
→ rewrite
→ retrieve
→ judge
→ rewrite ...
```

---

## 5.5 去重与预算

需要控制 Query Explosion。

建议：

```text
MAX_QUERY_VARIANTS <= 4 或 5
```

优先级：

```text
original query 必须保留
standalone 优先于 step-back
明确 decomposition 时不必再生成大量泛化 query
HyDE 按现有策略使用
```

对完全重复或高度相似 query 做 normalized dedup。

---

# 6. P1 评测：必须做 Query Rewrite Ablation

不要只是把功能写进去。

在现有 retrieval dev set 基础上新增多轮 / 指代 / 原因型 case。

至少比较：

```text
A: Original only
B: Original + existing decomposition
C: Original + standalone
D: Original + step-back
E: Full selective rewrite
```

指标优先使用：

```text
document recall
hit@k
MRR
query count
retrieval latency
```

不要直接复制参考项目的 0.80 → 0.93 数字。

只能报告本仓库自己的真实实验结果。

---

# 7. P2：External Research Tool + Evidence Adapter

这一阶段低于 P0/P1。

不要为了 MCP 而 MCP。

核心目标只有一个：

> 让 Deep Research 能在内部知识库以外获取研究证据，同时不改变 Analyst / Reviewer 协议。

---

## 7.1 统一 Evidence

当前已有 `EvidenceItem`。

扩展时不要建立第二套完全不同的 Evidence schema。

可以兼容增加：

```text
source_type:
    internal_doc
    arxiv
    github
    web

source_uri
retrieved_at
external
```

所有工具输出最终都转成 `EvidenceItem`。

因此后续流程继续是：

```text
Internal Retriever ─┐
Arxiv Tool ─────────┤
GitHub Tool ────────┤
Web Search ─────────┘
        ↓
EvidencePackage
        ↓
Analyst
        ↓
Reviewer
```

Analyst / Reviewer 不关心工具细节。

---

## 7.2 Tool 抽象

建议定义很薄的接口，例如：

```python
class ResearchTool(Protocol):
    name: str
    read_only: bool

    async def search(self, query: str, limit: int = 5) -> list[EvidenceItem]:
        ...
```

第一版只需要 1~2 个可靠只读工具即可。

不要一口气做几十个 MCP Tool。

---

## 7.3 安全边界

Deep Research 第一版外部工具：

```text
只读
```

不要让科研助手：

- 创建 GitHub issue
- 修改仓库
- 删除资源
- 发送邮件
- 自动提交外部内容

如果以后引入副作用工具，必须走明确 allowlist / approval。

---

# 8. 明确不要照搬 research_copilot 的内容

参考项目值得借鉴，但下面几件事不要复制。

## 8.1 不把 Plan-ReAct-MCP 变成默认执行路径

当前项目已经证明需要控制 Agent 复杂度。

因此：

```text
Normal ≠ Plan-ReAct
Deep ≠ 无限 ReAct
```

继续保持固定协议。

---

## 8.2 不新增一堆角色

不要新增：

```text
Supervisor
Search Agent
Tool Agent
Memory Agent
Citation Agent
Manager Agent
```

可以有 Service / Tool / Adapter，但不要都包装成 Agent。

---

## 8.3 不做 Tool-use SFT / DPO

除非以后有独立、高质量数据和明确实验目标，否则本轮完全不要加入。

这与当前项目主线无关。

---

## 8.4 不为了上传功能引入 MinIO / Redis Bitmap

如果当前项目的资料规模没有真实断点续传需求，不做。

---

# 9. 建议代码组织

以下只是建议，优先遵循当前仓库风格。

```text
src/
├── agent/
│   ├── agents/
│   │   └── research_team.py          # 保持 Agent 协议
│   ├── memory/
│   │   ├── mem0_manager.py           # 保留偏好/会话用途
│   │   ├── models.py                 # structured memory schema
│   │   ├── promotion.py              # Evidence → Memory gate
│   │   └── service.py                # recall / save
│   └── research_tools/
│       ├── base.py
│       ├── arxiv.py
│       └── github.py
│
├── api/
│   ├── services/
│   │   ├── chat_service.py
│   │   ├── research_service.py       # ResearchRun persistence
│   │   └── memory_service.py
│   └── controllers/
│       └── research_controller.py
│
└── rag/
    └── retrieval/
        └── query_expander.py          # standalone + step-back
```

如果当前代码规模不需要拆这么细，可以合并文件。

**不要为了符合这张目录图而做无意义重构。**

---

# 10. 测试要求

本轮修改完成后，至少新增以下测试。

## 10.1 Research Run

### 保存

```text
Deep Research 成功
→ ResearchRun completed
→ trace / reports / metrics 均存在
```

### 失败

```text
Deep Research 异常
→ 可以记录 failed run（如实现）
→ 不能破坏原聊天错误处理
```

### ACL

```text
User A 有权限创建/读取
User B 无项目权限
→ User B 无法读取 run 内受限 evidence
```

---

## 10.2 Memory Promotion

必须覆盖：

```text
supported + valid citation
→ promote

unsupported
→ reject

citation_gap
→ reject

invalid_source
→ reject

acl issue
→ reject

recommendation
→ default reject from semantic fact
```

---

## 10.3 Memory Recall ACL

测试：

```text
写入时可见
→ 用户权限改变
→ 再 recall
→ 不得返回无权限 source 对应 memory
```

---

## 10.4 Standalone Rewrite

至少：

```text
有明确指代 + 有历史
→ rewrite

普通独立问题
→ 不 rewrite

问候 / operation
→ 不 rewrite
```

---

## 10.5 Step-back

至少：

```text
原因 / 机制型复杂问题
→ 可生成一个 step-back

简单 lookup
→ 不生成
```

---

## 10.6 架构回归

确保已有测试继续证明：

```text
Normal 不自动进 Deep
Deep 只能显式进入
Reviewer 最多一次 revision
ACL fail-closed
Hybrid retrieval 正常
```

---

# 11. 评测与数据边界

不要改写历史评测结论。

当前 V3 属于：

```text
claim-level development evaluation
```

不是新的 blind holdout。

本轮新增功能请建立新的 development / ablation 结果，明确标识：

```text
development
regression
ablation
```

除非真正重新冻结并执行 blind set，否则不要写“blind improvement”。

---

# 12. 推荐执行顺序

Codex 请按以下顺序执行，不要四条主线同时大改。

## Phase 0：Baseline

1. 运行当前测试。
2. 记录 baseline。
3. 确认当前 Deep Research、ACL、Memory、Query Expansion 行为。

---

## Phase 1：Research Run

1. 增加 persistence。
2. Deep Research 完成时统一写入。
3. 增加只读 API。
4. 增加 ACL。
5. 增加前端 Trace 折叠面板。
6. 测试。

**先完成这一阶段再继续。**

---

## Phase 2：Evidence-backed Memory

1. 明确 Mem0 保留边界。
2. 增加 structured memory。
3. 实现 promotion gate。
4. 实现 ACL-aware recall。
5. Research Run → Episodic Memory。
6. Validated Claim → Semantic Memory。
7. 测试。

---

## Phase 3：Selective Query Rewrite

1. Standalone。
2. Step-back。
3. query variant budget / dedup。
4. retrieval ablation。
5. 不满意就回滚某一策略，不强留功能。

---

## Phase 4：External Research Tool

只在前三阶段稳定后做。

1. Evidence Adapter。
2. 一个只读外部工具。
3. Deep Research integration。
4. 工具错误 fail-soft。
5. Reviewer 继续检查 source validity。
6. 测试。

---

# 13. 完成标准

这次优化不是以“代码写完”为完成。

至少满足：

### Architecture

- [ ] Normal 主链没有变复杂
- [ ] Deep 仍为显式模式
- [ ] 没新增无意义 Agent
- [ ] Reviewer 仍最多一次 revision

### Research Run

- [ ] Deep run 可持久化
- [ ] 可通过 API 查看
- [ ] 前端可查看简洁 trace
- [ ] run evidence 受 ACL 保护

### Memory

- [ ] 不再把普通 LLM answer 当项目事实直接沉淀
- [ ] Claim promotion 有 deterministic gate
- [ ] Semantic memory 有 source
- [ ] Memory recall 再次检查 ACL

### Query Rewrite

- [ ] Standalone 是 selective
- [ ] Step-back 是 selective
- [ ] query variants 有预算
- [ ] 没有 iterative CRAG loop
- [ ] 有 retrieval ablation

### Quality

- [ ] 原测试通过
- [ ] 新测试通过
- [ ] 没有明显 dead code
- [ ] 没有为了抽象而抽象
- [ ] README / docs 与代码实际行为一致

---

# 14. Codex 最终必须输出的改造报告

完成修改后，请生成一份简短报告，例如：

```markdown
# Optimization Result

## 1. Changed
- ...
- ...

## 2. Architecture decisions
- 为什么这么改
- 哪些参考 research_copilot
- 哪些明确没有采用

## 3. Files changed
- path: purpose

## 4. Tests
- total
- passed
- failed

## 5. Evaluation
- baseline
- new variant
- metric differences

## 6. Remaining risks
- ...

## 7. Explicitly not implemented
- ...
```

如果某个功能没有实验数据支持，不要在 README 或简历材料中写成“效果提升”。

---

# 15. 这轮改造最重要的一句话

不要把项目从一个已经收敛的可信科研助手重新改成“功能很多的通用 Agent”。

这轮真正要解决的是：

> **如何让一次有证据约束的科研分析，被完整复盘，并把其中经过验证的知识安全地沉淀下来，在未来继续复用。**

因此优先级永远是：

```text
Evidence
> Auditability
> Memory correctness
> Evaluation
> More Agents / More Tools
```
