# 拒答与降级策略设计

> 定位：企业内部制度问答与流程检索系统 → 可信回答

## 1. 背景与问题

**核心原则**：答不准时不如不答，乱答比拒答危害大 10 倍。

**现状问题**：
- 检索结果 LOW 时虽有 rewrite/re-expand，但仍可能返回次优结果
- 多篇文档冲突（文档A说15天，文档B说5天）时无感知，直接生成
- 下游 LLM 超时/429 时无降级策略
- 无证据置信度评估，回答总是很"自信"

---

## 2. 拒答触发矩阵

| 触发条件 | 行为 | 返回内容 |
|---------|------|---------|
| 检索结果为空（NO_RESULTS） | 拒答 | "知识库中未找到相关内容。建议联系 HR/行政获取帮助。" |
| CRAG 评估 avg_score < `crag_medium_threshold` 且 rewrite 失败 | 拒答 | "根据现有知识库，无法给出准确答案。请提供更多线索或联系管理员。" |
| 文档间冲突（同主题不同内容） | 返回冲突摘要 | "找到相互冲突的制度内容：A 规定 X，B 规定 Y。请以 HR 确认为准。" |
| 下游 LLM 超时/5xx | 返回检索片段（不生成） | 直接返回 top-3 检索结果文本，不走生成 |
| MCP 工具不可用 | 降级不影响主链路 | 工具调用失败 → 返回友好提示，知识检索链路不受影响 |
| Vision LLM 超时 | 跳过图片理解，继续文本检索 | 警告日志 + 返回文本检索结果 |
| CRAG grading 全部失败 | 拒答或返回 rerank 结果 | 若 rerank 有结果则返回，否则拒答 |

---

## 3. 冲突检测实现

```python
# src/rag/evaluation/conflict_detector.py — 新文件
class ConflictDetector:
    """
    检测检索结果中的文档冲突。

    冲突类型：
    1. 数值冲突：A说15天年假，B说5天
    2. 状态冲突：A说有效，B说已废止
    3. 范围冲突：A说全公司，B说仅技术部
    """

    def detect(self, docs: List[Document], query: str) -> Optional[ConflictReport]:
        # 1. 提取关键数值实体
        entities = self._extract_numeric_claims(docs, query)

        # 2. 检测矛盾
        conflicts = []
        for claim_type, claims in entities.items():
            unique_values = set(claims.values())
            if len(unique_values) > 1:
                # 有多个不同值，可能是冲突
                sources = {v: claims[v] for v in unique_values}
                conflicts.append(Conflict(
                    type="numeric_conflict",
                    claim_type=claim_type,
                    sources=sources,
                ))

        if not conflicts:
            return None

        return ConflictReport(
            conflicts=conflicts,
            severity="high" if len(conflicts) > 1 else "medium",
            suggested_action="conflict_summary",  # 而非直接生成
        )

    def format_conflict_summary(self, report: ConflictReport) -> str:
        """格式化为用户友好的冲突摘要"""
        lines = ["**⚠️  发现制度内容存在冲突，请以 HR 或制度管理员确认为准：**\n"]

        for c in report.conflicts:
            if c.type == "numeric_conflict":
                lines.append(f"**{c.claim_type}** 存在不同规定：")
                for value, source in c.sources.items():
                    lines.append(f"  - **{value}**：依据 {source}")
                lines.append("")

        lines.append("> 如需最终确认，请联系 HR 部门或制度管理员。")
        return "\n".join(lines)
```

---

## 4. 分层降级策略

### 4.1 降级链路图

```
用户查询
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 检索阶段                                              │
│  ├─► 检索成功 → CRAG 评估                            │
│  │      ├─► HIGH → 生成答案                          │
│  │      ├─► MEDIUM → 生成答案（注明"证据一般"）        │
│  │      ├─► LOW + rewrite 成功 → 生成答案              │
│  │      ├─► LOW + rewrite 失败 → 拒答 ❌              │
│  │      └─► NO_RESULTS → 拒答 ❌                      │
│  └─► 检索失败（向量库空/损坏）→ 拒答 ❌                │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 生成阶段                                              │
│  ├─► LLM 成功 → 流式输出答案（带版本溯源）           │
│  ├─► LLM 超时 → 返回 top-3 检索片段（不生成）⚠️       │
│  ├─► LLM 429 → 重试 1 次，仍失败 → 返回检索片段 ⚠️    │
│  └─► LLM 5xx → 返回检索片段 ⚠️                       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 工具阶段                                              │
│  ├─► MCP 工具不可用 → 降级提示（不影响知识检索）       │
│  └─► 工具调用异常 → 返回错误，不阻断主链路             │
└─────────────────────────────────────────────────────┘
```

### 4.2 拒答 Prompt 注入

```python
# 在生成 Agent 的 system prompt 中注入拒答指令
REFUSAL_INJECTION = """
【重要：可信回答原则】
1. 如果检索结果不足以回答用户问题，你应该**明确拒答**，
   而非猜测或模糊回答。
2. 拒答格式：「根据现有知识库，我无法准确回答这个问题。
   建议您：① 联系 HR/行政获取帮助；② 提供更多线索。」
3. 如果检索结果之间存在冲突，**不要自己判断哪个正确**，
   应列出冲突内容，并注明"请以 HR 确认为准"。
4. 永远不要编造知识库中没有的信息。
5. 如果对某条规定不确定，应使用「根据XXX制度（版本X）」，
   而非模糊表述。
"""
```

---

## 5. 置信度标注

```python
def annotate_answer_confidence(grade_result: GradeResult, answer: str) -> str:
    """
    根据 CRAG 评估结果，在答案中标注置信度。
    """
    if grade_result.decision == GradeLevel.HIGH:
        confidence_tag = "✅ 高置信（依据充分）"
    elif grade_result.decision == GradeLevel.MEDIUM:
        confidence_tag = "⚠️  中等置信（依据部分，建议核实）"
    else:
        confidence_tag = "❌ 低置信（依据不足，请勿直接引用）"

    return f"""{answer}

---

**置信度评估**：{confidence_tag}
- 高相关文档：{grade_result.high_count} 篇
- 中等相关文档：{grade_result.medium_count} 篇
- 低相关文档：{grade_result.low_count} 篇
- 平均相关分：{grade_result.avg_score:.2f}
- 评估理由：{grade_result.decision_reason}

> 本答案仅供参考，请以公司正式发布的制度文件为准。
"""
```

---

## 6. 实施计划

| 阶段 | 内容 | 改动文件 |
|------|------|---------|
| Phase 1 | `ConflictDetector` | `src/rag/evaluation/conflict_detector.py` (新) |
| Phase 2 | 生成 Agent 注入 `REFUSAL_INJECTION` | `src/agent/prompts.py` |
| Phase 3 | `annotate_answer_confidence()` | `src/agent/skills/knowledge/scripts/tools.py` |
| Phase 4 | LLM 超时降级：返回检索片段 | `src/api/services/chat_service.py` |
| Phase 5 | CRAG LOW → 拒答（而非次优结果）| `src/rag/evaluation/retrieval_grader.py` |
| Phase 6 | MCP 降级：工具失败不阻断主链路 | `src/agent/tools/mcp_adapter.py` |
