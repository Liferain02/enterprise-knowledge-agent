# Enterprise Knowledge Agent 项目分析与优化建议

> 项目：`Liferain02/enterprise-knowledge-agent`  
> 当前定位：实验室科研知识与协作智能助手  
> 本文目的：供 Codex 从现有代码质量、Agent 架构、RAG 工程、权限安全、评测体系和生产化方向进一步评估。

---

# 1. 当前项目定位

当前项目表面上很容易被理解成：

> “LangGraph + RAG + Vue 的实验室知识库助手。”

但从实际仓库结构看，这个判断明显低估了项目。

当前已经存在：

- 文档 ingestion
- processing
- vector storage
- BM25
- vector retrieval
- RRF fusion
- query expansion
- reranking
- query cache
- table QA
- ACL filter
- retrieval evaluation
- conflict detection
- Agent Planner
- Deep Research
- Researcher
- Analyst
- Reviewer
- Revision
- Memory
- Checkpointer
- FastAPI
- SSE
- 多用户认证
- 项目 / 实验 / 任务管理
- 测试体系

因此项目更适合定位成：

> **面向多用户科研知识场景的权限感知 RAG 与可控 Deep Research Agent 系统。**

如果后续继续优化，应该朝：

> **Production-oriented RAG / Agent System**

发展，而不是继续堆 Agent 数量。

---

# 2. 当前项目中最有价值的部分

## 2.1 RAG Pipeline 已经明显超出普通 Demo

当前 `src/rag/` 已拆分为：

```text
cache
evaluation
ingestion
processing
retrieval
storage
```

说明已经不是所有 RAG 逻辑都写在一个 service 文件中。

`src/rag/retrieval/` 中还存在：

- `acl_filter.py`
- `hybrid_retriever.py`
- `query_cache.py`
- `query_expander.py`
- `reranker.py`
- `retriever.py`
- `table_qa.py`

这已经形成了相对完整的 retrieval subsystem。

---

## 2.2 Hybrid Retrieval 有真实工程细节

`hybrid_retriever.py` 目前已经实现：

- BM25
- Vector Retrieval
- RRF
- jieba domain dictionary
- BM25 index 复用
- bm25s / rank_bm25 fallback
- Query 长度自适应 BM25 / Vector 权重
- Candidate over-fetch
- ACL filter
- 检索结果二次 ACL check

这比简单：

```text
BM25 + Embedding + Reranker
```

更有实际工程设计。

其中最值得强化的是：

> **ACL 权限过滤贯穿 Retrieval，而不是生成完成以后才做。**

这是企业知识库场景中很真实的问题。

---

## 2.3 权限控制是当前项目最有差异化潜力的能力

当前检索过程中已经存在：

- build ACL filter
- vector search filter
- result-level `check_doc_access`
- Deep Research ACL 测试

这意味着项目可以进一步强化为：

> **Security-aware RAG**

相比普通个人项目常见的：

- PDF 入库
- 向量检索
- Chat UI

权限安全更容易体现生产系统意识。

---

## 2.4 Agent Graph 的“克制”是优点

当前 `src/agent/graph.py` 明确采用：

```text
Summary
  ↓
Memory Retrieval
  ↓
Planner
  ├── RAG
  ├── Tool
  ├── General
  └── Deep Research
```

Deep Research 使用：

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

并且代码中明确说明：

- 不使用 Supervisor
- 不使用动态 Worker
- 不使用 Send fan-out
- 不做无限 Reviewer loop

这其实是很好的工程取舍。

说明系统目标不是“展示 Agent 数量”，而是：

- 可控
- 可调试
- 可评测
- 控制成本
- 控制时延

这一点建议在 README 和简历里重点讲。

---

## 2.5 Evaluation 已经是独立 subsystem

`src/rag/evaluation/` 当前存在：

- `evaluator.py`
- `retrieval_grader.py`
- `retrieval_metrics.py`
- `grade_cache.py`
- `conflict_detector.py`

说明项目已经有评测意识，而不是只看 UI 是否能回答问题。

这应该成为后续项目升级的核心。

---

## 2.6 测试已经覆盖多个生产问题

现有 tests 可以看到：

- Agent architecture
- Auth config
- Streaming
- Controller binding
- Deep Research ACL matrix
- Embedding config
- Feedback workflow
- Ingestion pipeline
- Input security
- Knowledge catalog
- LLM config
- Password security

项目本身的工程成熟度已经比 README 展示出的更高。

---

# 3. 为什么现在仍然容易被认为是 Demo

## 3.1 README 太像典型 RAG 项目介绍

README 当前主要在列：

- 混合检索
- 深度研究
- PDF / MD / TXT / DOCX
- 引用
- 项目管理
- 多用户

这些能力单独看都已经非常常见。

因此面试官很容易快速归类为：

> “一个做得比较完整的 RAG Demo。”

但 README 没突出真正差异化的部分：

- ACL Retrieval
- Query Rewrite
- Conflict Detection
- Evaluation
- Deep Research Routing
- Cost / Latency Trade-off
- Revision Limit
- Retrieval Decision
- Version Source
- Memory Compression

---

## 3.2 项目功能很多，但“核心问题”没有定义清楚

现在项目同时包含：

- 知识问答
- Deep Research
- 项目管理
- 实验记录
- 任务管理
- Memory
- Tools
- 多用户
- 权限

容易产生：

> “功能很多，但不知道作者真正解决了什么核心难题。”

建议明确项目主问题：

> **如何让内部知识 Agent 在多用户权限约束下，对多来源科研资料完成可信检索、引用、综合和冲突分析，同时控制 Deep Research 的时延与成本。**

后续所有模块围绕这个问题组织。

---

## 3.3 当前最大的可信度问题仍然是 Evaluation 数据规模

简历当前写：

- 41 条开发评测
- 26 条冻结 Blind Test
- Deep 文档召回 0.811
- 引用覆盖 0.727

虽然已经比很多个人项目好，但 26 条 Blind Test 仍然偏小。

尤其仓库已经拥有比较复杂的 Evaluation Infrastructure。

如果最终 Blind Set 只有 26 条，会造成：

> “评测系统比评测数据还重。”

的问题。

---

# 4. 推荐的最终目标形态

建议升级为：

# **权限感知、可评测、可观测的科研知识 Agent 平台**

目标架构：

```text
Documents
   ↓
Ingestion
   ↓
Chunk / Metadata / Version
   ↓
Vector + BM25
   ↓
ACL Pre-filter
   ↓
Hybrid Retrieval
   ↓
Query Expansion
   ↓
Rerank
   ↓
ACL Re-check
   ↓
Retrieval Decision
   ├── Normal RAG
   └── Deep Research
           ↓
      Researcher
           ↓
       Analyst
           ↓
       Reviewer
           ↓
      One Revision
           ↓
       Final Answer
```

外层增加：

```text
Evaluation
Observability
Cost Tracking
Latency Tracking
Feedback
```

---

# 5. 第一优先级：扩大 Evaluation

建议冻结至少：

- 150～300 条 Query

并按照场景划分。

---

## 5.1 Basic Retrieval

测试：

- 单文档事实
- 明确关键词
- 同义改写
- 长问题
- 极短问题
- 专有名词

---

## 5.2 Multi-document Synthesis

测试：

- 两份文档联合回答
- 三份以上资料综合
- 不同实验结果汇总
- 多次组会记录总结

这是 Deep Research 的核心价值所在。

---

## 5.3 Conflict Cases

专门构造：

- 两份文档结论不一致
- 新旧版本不一致
- 实验结果不同
- 论文观点冲突
- 组会记录覆盖旧决定

指标：

- Conflict Detection Recall
- Conflict Explanation Accuracy
- Newer Version Selection Accuracy

---

## 5.4 ACL / Security Cases

建议把 ACL 单独作为正式 Benchmark。

例如：

```text
public document
project document
restricted document
```

不同用户：

```text
anonymous
normal member
project member
document owner
admin
```

构造 ACL Matrix。

核心指标：

> Unauthorized Retrieval Rate

理想：

```text
0
```

这个指标非常适合写简历。

---

## 5.5 No-answer Cases

测试系统是否能：

- 判断资料不足
- 不 hallucinate
- 不使用无权限资料
- 不强行生成答案

指标：

- Abstention Precision
- Abstention Recall

---

## 5.6 Table QA

如果已有 `table_qa.py`，应该专门设计：

- 实验参数表
- 指标对比表
- 项目进度表
- 会议任务表

否则该模块很容易变成“有代码但没有证据”。

---

# 6. 第一优先级：做 Normal RAG vs Deep Research 对照

这是当前项目最值得强化的能力。

当前已经同时有：

- Normal RAG
- Deep Research

但需要回答：

> Deep Research 到底什么时候值得调用？

---

## 6.1 建立 Complexity Label

给 Blind Test 标：

```text
Simple
Medium
Complex
```

例如：

### Simple

“实验室请假流程是什么？”

### Medium

“当前项目 A 的最新实验结果是什么？与上周有什么变化？”

### Complex

“综合最近三次组会、实验记录和论文笔记，分析模型效果下降的可能原因并给出下一步实验建议。”

---

## 6.2 做 Quality / Latency / Cost 对照

建议指标：

| Metric | Normal RAG | Deep Research |
|---|---:|---:|
| Retrieval Recall | | |
| Citation Coverage | | |
| Answer Correctness | | |
| Conflict Detection | | |
| P50 Latency | | |
| P95 Latency | | |
| LLM Calls | | |
| Input Tokens | | |
| Output Tokens | | |
| Estimated Cost | | |

最终应该形成一个结论：

```text
Simple Query
→ Normal RAG

Complex Multi-document Query
→ Deep Research
```

然后让 Planner 根据 complexity 自动路由。

这会让 Planner 的存在变得有明确价值。

---

# 7. 第一优先级：增加全链路 Observability

这是降低“Demo 感”非常有效的一步。

一次问答建议生成一个：

```text
trace_id
```

记录：

```text
request
↓
planner
↓
query expansion
↓
bm25
↓
vector
↓
rrf
↓
reranker
↓
acl
↓
retrieval decision
↓
agent graph
↓
llm
↓
citation
```

每个节点记录：

- start_time
- end_time
- latency
- input count
- output count
- token
- model
- cache hit
- error

---

## 建议最终可以看到

```text
Query Expansion       320 ms
BM25                   28 ms
Vector Retrieval       86 ms
RRF                      2 ms
Rerank                 240 ms
ACL                     11 ms
Generation            2.8 s

Total                 3.5 s
```

Deep Research 则显示：

```text
Researcher            ...
Analyst               ...
Reviewer              ...
Revision              ...
Generation            ...
```

这会极大提升生产系统感。

---

# 8. 权限系统建议继续强化

当前 ACL 已经是很好的基础。

建议进一步确认以下问题。

---

## 8.1 ACL 是否真正发生在 Retrieval 前

需要 Codex 重点确认：

- Vector DB filter 是否真正生效
- BM25 是否先加载了无权限全文
- Query Cache 是否可能返回其他用户结果
- Rerank 前是否可能存在敏感文档
- Deep Research evidence_package 是否可能混入无权限资料

---

## 8.2 Cache 必须 ACL-aware

如果当前 query cache key 只是：

```text
query
```

那可能有风险。

建议至少包含：

```text
query
user_id
project_scope
permission_version
```

或者缓存 retrieval candidate 时，只缓存 public/global candidate，再在用户层重新过滤。

---

## 8.3 Citation 必须二次校验

最终生成引用时，再确认：

```text
citation.document_id
```

属于当前用户有权限读取的 document。

最好有一个：

```text
assert_authorized_citations()
```

作为 finalizer 前最后一道 guard。

---

# 9. Deep Research 建议：不要继续增加 Agent

当前：

```text
Researcher
Analyst
Reviewer
Revision
```

已经足够。

不建议增加：

- Supervisor
- Debate Agent
- Critic Agent
- Executor Agent
- Dynamic Worker Pool
- Infinite Reflection

这些会导致：

- Token 爆炸
- Latency 上升
- Debug 困难
- Evaluation 不稳定

当前限制最多一次 Revision 是正确方向。

后续重点应该放在：

> **什么时候进入 Deep Research，以及 Deep Research 是否真正带来收益。**

---

# 10. Agent Planner 建议强化

Planner 当前可以进一步输出结构化决策：

```json
{
  "route": "deep_research",
  "complexity": "complex",
  "needs_expansion": true,
  "reason": "...",
  "estimated_cost_level": "high"
}
```

然后纳入 Evaluation。

评测：

- Route Accuracy
- Over-routing Rate
- Under-routing Rate

特别值得关注：

> 有多少简单问题被错误送进 Deep Research？

这直接决定成本和延迟。

---

# 11. Ingestion 可以进一步体现工程能力

如果当前已经支持：

- PDF
- MD
- TXT
- DOCX

建议强化：

- content hash
- dedup
- ingestion status
- retry
- failure reason
- version
- source metadata
- chunk version

构造：

```text
Document
├── document_id
├── source
├── version
├── content_hash
├── permission
├── ingestion_status
├── created_at
└── chunks
```

这样资料更新后可以做到：

```text
旧 chunk 删除
新 chunk 写入
索引更新
cache invalidation
```

比“支持多格式上传”更有技术含量。

---

# 12. Production Profile 建议

当前项目使用 SQLite + Chroma 完成本地运行没有问题。

但建议增加一个：

```text
production profile
```

例如：

```text
FastAPI
PostgreSQL
Redis
Qdrant / Chroma
Worker
Vue
```

通过：

```text
docker compose up
```

启动。

不需要 Kubernetes。

对于个人项目，Docker Compose 已经足够体现部署意识。

---

# 13. 建议增加异步 Ingestion Worker

如果文档 ingestion 当前发生在 API 请求中，建议改成：

```text
Upload
  ↓
Create Ingestion Job
  ↓
Queue
  ↓
Worker
  ↓
Parse
  ↓
Chunk
  ↓
Embedding
  ↓
Index
  ↓
Update Status
```

状态：

```text
PENDING
PROCESSING
SUCCESS
FAILED
```

支持：

- Retry
- Failure Reason
- Progress

这非常符合真实知识库系统。

---

# 14. README 应该如何重写

建议 README 第一屏直接展示：

## Problem

> 内部科研知识库不仅需要“能检索”，还需要解决权限隔离、多文档综合、版本冲突、回答引用、复杂任务成本与时延控制。

## Architecture

给出完整链路。

## Key Engineering Decisions

重点写：

- ACL-aware Retrieval
- Hybrid Retrieval + RRF
- Query Expansion
- Rerank
- Controlled Deep Research
- One Revision Limit
- Evaluation
- Tracing

## Benchmark

展示：

```text
Normal RAG vs Deep Research
ACL Benchmark
Ablation
Latency / Cost
```

## Security

展示：

```text
ACL
Input Security
Citation Validation
```

## Demo

最后再展示前端。

---

# 15. 不建议继续投入的方向

不建议：

- 增加更多 Agent
- 为了“高级”加入 AutoGen / CrewAI
- 加复杂 Supervisor
- 无限 Reflection
- 大量 Prompt 花活
- 继续增加普通 CRUD
- 花大量时间优化聊天 UI
- 硬上 Kubernetes
- 做与核心科研知识场景无关的插件

因为这些不会解决项目当前最核心的问题：

> **真实效果和工程价值没有被系统性证明。**

---

# 16. 最终简历应该能够讲出的故事

理想状态：

> 面向科研团队构建权限感知的知识与协作 Agent，围绕内部资料检索中的专有名词、语义查询、多文档综合和权限泄漏问题，设计 BM25 + Vector + RRF + Rerank 混合检索，并在向量检索、融合结果和最终引用阶段贯穿 ACL 校验；基于 LangGraph 将普通 RAG 与 Deep Research 分离，仅对复杂查询进入 Researcher → Analyst → Reviewer → 单次 Revision 流程，并建立 Blind Evaluation 对比普通 RAG 与 Deep Research 在质量、引用覆盖、时延和 Token 成本上的收益。

这个故事比：

> “做了一个 LangGraph 多 Agent RAG 助手。”

强很多。

---

# 17. 希望 Codex 重点评估的问题

请 Codex基于当前仓库实际代码回答：

1. 当前 RAG Pipeline 是否真的按照 README / 本文描述运行？
2. `hybrid_retriever.py` 是否存在架构重复或历史代码未清理？
3. BM25 / bm25s / LangChain BM25Retriever 是否存在重复实现？
4. Query 长度自适应权重目前是否有实验依据，还是经验 hardcode？
5. RRF 实现是否正确？
6. Vector 和 BM25 candidate 的 score / rank 是否处理合理？
7. ACL 是否真正发生在所有 retrieval path？
8. 是否存在 ACL bypass？
9. Query Cache 是否存在跨用户泄漏风险？
10. Reranker 前后 ACL 顺序是否合理？
11. Deep Research evidence package 是否可能混入未授权文档？
12. Citation 是否有 final authorization check？
13. Planner 是否真的有必要，当前是否存在 over-engineering？
14. Deep Research 当前流程是否稳定、可复现、可测试？
15. Revision 最多一次的限制是否合理？
16. 当前 Memory / Mem0 是否带来了实际价值，还是增加了复杂度？
17. 当前 Agent State 是否过重？
18. `graph.py` 是否应该继续拆分？
19. 当前 evaluation framework 是否适合扩展到 150～300 条 Blind Test？
20. 应如何实现 Normal RAG vs Deep Research 的正式对照实验？
21. 当前哪些指标最适合写入简历？
22. 如何设计 Unauthorized Retrieval Rate Benchmark？
23. 当前 SQLite + Chroma 在项目定位下是否足够？
24. 是否值得增加 PostgreSQL / Redis / Worker？
25. 哪些改造最能降低 Demo 感？
26. 如果只给 1～2 周，最值得完成哪些事项？
27. 如果目标岗位是 AI 应用开发 / Agent 开发，哪些模块应该重点保留和深化？

---

# 18. 建议实施顺序

```text
P0
├── Blind Test 扩展到 150～300
├── Normal RAG vs Deep Research Benchmark
├── ACL Benchmark
├── 全链路 Trace
├── Latency / Token / Cost 统计
└── 重写 README

P1
├── Planner Routing Evaluation
├── Citation Final Authorization
├── ACL-aware Cache
├── Document Version / Dedup
├── Async Ingestion
└── Docker Compose Production Profile

P2
├── PostgreSQL
├── Redis
├── Worker Queue
├── 更完整 Monitoring
└── Benchmark CI
```

---

# 19. 核心原则

后续应该减少：

> “再增加一个 Agent / 再增加一个功能。”

转而强化：

> **权限是否真的安全？检索是否真的更好？Deep Research 是否真的值得？错误是否能定位？成本和时延是否可观测？**

如果这些问题能够用 Benchmark 和系统设计回答清楚，这个项目就会从普通“RAG Demo”升级为一个有明显生产意识的 **Agent Engineering Project**。
