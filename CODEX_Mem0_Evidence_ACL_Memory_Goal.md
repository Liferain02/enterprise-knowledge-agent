# Goal：完善 Mem0 科研长期记忆，形成 Evidence-backed + ACL-aware Memory

> 目标仓库：`Liferain02/enterprise-knowledge-agent`
>
> 本 Goal 基于当前最新实现继续优化，不重构现有主架构。
>
> 核心原则：**继续使用 Mem0 做语义记忆检索，但科研事实是否可信、是否仍有权限使用，由 Research Run + Evidence + Reviewer + ACL 决定。**

---

## 0. 当前已经具备的能力

当前项目已经实现：

- Mem0 普通对话长期记忆；
- Deep Research 固定链路：
  - Researcher
  - Analyst
  - Reviewer
  - 最多一次 Revision
- Research Run 持久化；
- Research Run 读取时会根据当前用户权限重新执行 ACL 检查；
- Reviewer PASS + `claim_type=fact` + 有效 `source_ids` 的 Claim 可以由用户点击“确认并记住”；
- 确认后的科研事实使用 `infer=False` 精确写入 Mem0，并携带：
  - `research_run_id`
  - `claim_id`
  - `source_ids`
  - `review_decision`
  - `user_confirmed`

以上能力全部保留。

---

# 1. 本 Goal 要解决的问题

当前存在两个记忆安全问题。

## 问题 1：Deep Research 仍可能绕过 Promotion Gate

当前 Deep Research 完成后，除了保存 Research Run 和用户确认后的 fact 外，整个问答仍可能进入普通 Mem0 自动提取链路。

这可能导致：

```text
Reviewer 认为某条内容只是 inference / recommendation
        ↓
不能通过“确认并记住”进入科研事实记忆
        ↓
但整个 Deep Research Final Answer 又被普通 Mem0 自动抽取
        ↓
绕过 Memory Promotion Gate
```

也就是说，当前可能存在两条并行写入路径：

```text
A. Research Run
   → Reviewer PASS
   → 用户确认 Fact
   → Mem0 infer=False

B. Deep Research Final Answer
   → 普通 save_to_mem0
   → Mem0 自动提取
```

B 路径必须被收口。

---

## 问题 2：confirmed research fact 召回时没有重新做 ACL

当前 Mem0 搜索结果主要按 `user_id` 召回后直接格式化注入上下文。

对于以下普通记忆没有问题：

- preference
- conversation
- 普通历史上下文

但对于：

```text
memory_type = confirmed_research_fact
```

不能仅仅因为过去保存成功，就永久允许使用。

典型场景：

```text
用户保存科研事实时属于 Project A
        ↓
事实写入 Mem0
        ↓
之后用户被移出 Project A / role downgrade
        ↓
原 Evidence 已经无权访问
        ↓
Mem0 仍可能按 user_id 命中历史事实
```

Research Run 已经实现了“权限变化后重新鉴权”，Memory 也必须遵循同样原则。

---

# 2. 目标架构

不要删除 Mem0。

Mem0 继续负责：

- semantic retrieval
- preference memory
- conversational memory
- confirmed research fact 的向量召回

但 Mem0 不负责决定科研事实是否可信、是否仍有权限使用。

最终流程：

```text
Normal Conversation
    ↓
Mem0
    ↓
preference / conversation
    ↓
正常使用
```

Deep Research：

```text
Deep Research
    ↓
Research Run
    ↓
AnalysisReport
    ↓
Reviewer
    ↓
用户确认 Fact
    ↓
Mem0 infer=False
```

未来召回：

```text
Mem0.search
    ↓
判断 memory_type
    ↓
┌────────────────────────────┐
│ 普通 preference/conversation │
└────────────────────────────┘
    ↓
正常注入

confirmed_research_fact
    ↓
读取 research_run_id / claim_id / source_ids
    ↓
根据当前 user_context 重新验证 Research Run / Evidence ACL
    ↓
通过 → 注入
失败 → 丢弃
```

一句话：

> **Mem0 做 Retrieval；Research Run + Evidence + Reviewer + ACL 做 Trust。**

---

# 3. Task 1：禁止 Deep Research 绕过 Promotion Gate

请检查：

- `src/api/services/chat_service.py`
- `src/agent/graph.py`
- `save_to_mem0_node`
- 当前普通回答结束后的 Mem0 自动保存逻辑

修改要求如下。

## 3.1 Normal 模式

继续允许：

```text
conversation
→ Mem0 auto inference
```

不要影响普通用户偏好和聊天记忆。

## 3.2 Deep Research

不要再把 Deep Research assistant final answer 通过普通 Mem0 自动事实提取保存。

Deep Research 的科研事实只能通过：

```text
Research Run
→ Reviewer PASS
→ Evidence-backed Fact
→ 用户确认
→ Mem0 infer=False
```

进入科研事实长期记忆。

如果为了兼容用户偏好，确实需要保存 Deep Research 对话中的 preference，可以只保存明确属于用户输入的 preference。

但禁止：

```text
Deep Research Final Answer
→ 自动提升为科研事实
```

优先采用简单、明确、可测试的方案。

不要为了这一点增加新的 Agent 或复杂 LLM 分类链。

---

# 4. Task 2：给科研事实补齐统一 metadata

用户确认科研事实写入 Mem0 时，metadata 至少包含：

```json
{
  "memory_type": "confirmed_research_fact",
  "scope": "research",
  "project_id": "...",
  "research_run_id": "...",
  "claim_id": "...",
  "source_ids": ["S1", "S2"],
  "review_decision": "PASS",
  "user_confirmed": true,
  "verified": true
}
```

要求：

- 如果该 Research Run 没有关联 project：
  - `project_id = ""`
  - 或 `null`
- 不要伪造 project
- 保持 `infer=False`
- 不要让 Mem0 再调用 LLM 改写经过确认的 Claim

---

# 5. Task 3：增加 Research Memory Recall Gate

修改 Mem0 recall 流程。

当前大致为：

```text
Mem0.search
→ format_memories_for_context
→ 注入 AgentState
```

改成：

```text
Mem0.search
→ validate_memory_candidates
→ format allowed memories
→ 注入 AgentState
```

建议增加类似接口：

```python
async def filter_memories_for_current_user(
    memories,
    user_context,
) -> list:
    ...
```

具体行为如下。

## 5.1 普通 Memory

例如：

- preference
- conversation
- 普通历史上下文

继续按照原逻辑使用。

## 5.2 confirmed_research_fact

必须满足：

1. metadata 存在；
2. `research_run_id` 存在；
3. `claim_id` 存在；
4. Research Run 当前仍可被该用户读取；
5. 对应 Claim 当前仍存在；
6. Claim 的 `source_ids` 当前仍然能够解析到可访问 Evidence；
7. 当前用户对这些 Evidence 仍通过 ACL。

全部满足才允许注入。

任意一步无法确认：

```text
fail closed
→ 丢弃该 Memory
```

不能因为 Mem0 中已经存在这条记录就默认可信。

---

# 6. Task 4：复用 ResearchService，不重复实现 ACL

Research Run 已经实现了当前权限重新校验和历史 Evidence 隐藏逻辑。

优先复用：

```python
research_service.get_research_run(...)
```

或者在 ResearchService 中增加一个窄接口，例如：

```python
def validate_confirmed_research_memory(
    self,
    run_id: str,
    claim_id: str,
    source_ids: list[str],
    user: dict,
) -> bool:
    ...
```

要求：

- 不要在 `graph.py` 重新复制一整套 project ACL；
- 不要在 Memory 模块重新实现 document ACL；
- ACL source of truth 仍保持统一；
- 无法验证时默认拒绝。

---

# 7. Task 5：继续使用 Mem0，不新建第二套向量 Memory Engine

本轮明确不要：

- 删除 Mem0；
- 新增新的向量数据库；
- 自己重新实现 embedding memory search；
- 增加 Memory Agent；
- 增加 Supervisor；
- 改 Deep Research Agent 数量；
- 重写整个 Memory 子系统。

Mem0 负责 Retrieval。

Research Run + Evidence + Reviewer + ACL 负责 Trust。

---

# 8. 可选改进：区分 User Memory 与 Research Memory

如果在不显著增加复杂度的前提下容易实现，可以在 metadata 中增加：

```text
scope = user / research
```

建议语义：

```text
scope=user
- preference
- conversation
- personal context

scope=research
- confirmed_research_fact
- 后续可能扩展的 verified decision / progress
```

当前 Goal 只强制：

```text
confirmed_research_fact → scope=research
```

不要在这一轮扩展过多 memory type。

---

# 9. 必须增加的测试

至少增加以下测试。

## Test 1：Normal Memory 不受影响

```text
normal mode
→ 普通对话
→ Mem0 自动保存仍执行
```

验证：

- 原普通记忆路径仍正常；
- 不因为新增 Research Recall Gate 破坏普通聊天体验。

## Test 2：Deep Research 不自动保存 Final Answer

```text
deep mode
→ Research Answer
→ 普通 save_to_mem0
→ 不得把 assistant final answer 自动写入科研事实 Memory
```

重点验证：

- Deep Research 不再绕过 Promotion Gate。

## Test 3：Confirmed Fact 可以正常召回

```text
Research Run PASS
→ fact + source
→ 用户确认
→ Mem0
→ 当前 ACL 合法
→ recall
→ memory 可以进入上下文
```

## Test 4：权限变化后禁止召回

这是本 Goal 最重要的测试。

```text
用户确认 Fact 时有权限
→ Memory 已经写入 Mem0
→ 用户被移出项目 / role downgrade
→ 再次查询
→ Mem0 虽然命中该 Memory
→ Recall Gate 必须过滤
```

最终结果：

```text
该 Research Memory 不得进入 Agent 上下文
```

## Test 5：伪造 / 损坏 metadata 必须 fail closed

至少覆盖：

```text
research_run_id 不存在
claim_id 不存在
source_ids 不匹配
Research Run 当前无权访问
Evidence 被隐藏
metadata 缺失
```

全部不得注入。

## Test 6：Recommendation / Inference 仍无法通过确认接口

保持现有行为：

```text
claim_type = recommendation
→ 不允许 confirmed_research_fact

claim_type = inference
→ 不允许 confirmed_research_fact
```

## Test 7：普通 Preference 不受 Research ACL 影响

例如：

```text
memory_type = preference
scope = user
```

即使没有 `research_run_id / source_ids`：

```text
仍应按原普通 Memory 路径允许使用
```

不要把 Research Memory 的严格校验误伤普通用户记忆。

---

# 10. 可观测性

给 Memory Recall 增加简单统计，但不要记录敏感文本。

建议：

```text
memory_candidates
memory_allowed
memory_acl_filtered
memory_invalid_metadata
memory_research_verified
```

可以：

- 写 debug log；
- 或写入已有 trace / diagnostics。

不要记录：

- 完整 memory 文本；
- Evidence 原文；
- 敏感项目内容。

---

# 11. 不要做的事情

本 Goal 暂时不要实现：

- Step-back Query
- arXiv
- Semantic Scholar
- MCP
- External Research Tool
- 新 Agent
- Tool-use SFT / DPO
- Memory Graph
- Knowledge Graph
- 新的向量数据库
- 大规模重构前端

先把现有 Memory 链路做正确。

---

# 12. 验收标准

完成后必须满足：

- [ ] Normal Chat 的 Mem0 行为保持正常
- [ ] Deep Research Final Answer 不再绕过 Promotion Gate
- [ ] 用户确认后的科研 fact 仍使用 Mem0 保存与检索
- [ ] confirmed research fact 召回时会重新检查当前 ACL
- [ ] 用户权限变化后，历史科研 Memory 无法泄漏
- [ ] 无效 metadata 一律 fail closed
- [ ] Research Run ACL 逻辑没有重复实现
- [ ] Recommendation / Inference 仍不可作为 confirmed research fact
- [ ] Preference / Conversation memory 不被误伤
- [ ] 不增加新的 Agent
- [ ] 原有测试全部通过
- [ ] 新增测试全部通过

---

# 13. 建议实现顺序

按下面顺序完成，不要一次大改：

## Phase 1：梳理当前写入路径

确认：

```text
Normal → Mem0 auto save
Deep → Mem0 auto save
Deep Fact confirm → Mem0 infer=False
```

明确所有入口后再改。

## Phase 2：关闭 Deep Research 普通自动事实写入

做到：

```text
Normal → 保留
Deep → 不允许 Final Answer 自动进入科研事实 Memory
```

先补测试。

## Phase 3：补 metadata

统一 confirmed research fact metadata。

重点补：

```text
scope
project_id
verified
```

## Phase 4：实现 Recall Gate

```text
Mem0.search
→ classify memory type
→ research memory validation
→ allowed memories only
→ format context
```

优先复用 ResearchService。

## Phase 5：权限变化回归测试

重点跑：

```text
有权限保存
→ 降权
→ Mem0 命中
→ Recall Gate 拦截
```

这个必须通过。

---

# 14. Codex 完成后必须输出的报告

完成后请生成一份简短报告：

```markdown
# Memory Optimization Result

## 1. Changed
- 修改了哪些文件
- 每个文件的目的

## 2. Normal vs Deep Memory behavior
- Normal 现在怎么保存
- Deep 现在怎么保存
- 哪些内容必须用户确认

## 3. Recall Gate
- confirmed_research_fact 如何验证
- 如何复用 Research Run / ACL

## 4. ACL downgrade verification
- 测试场景
- 测试结果

## 5. Tests
- 原测试通过数量
- 新增测试数量
- 是否有失败

## 6. Not implemented
- Step-back
- External Research Tool
- MCP
- 其他本 Goal 明确排除内容
```

---

# 15. 本 Goal 最重要的一句话

不要把 Mem0 删除，也不要让 Mem0 成为科研事实可信性的裁判。

正确分工是：

```text
Mem0
→ 负责“我能不能找到这条记忆”

Research Run + Evidence + Reviewer + ACL
→ 负责“这条记忆今天还能不能被信任、被使用”
```
