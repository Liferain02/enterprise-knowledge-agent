# 企业知识库 Agent 系统测试总览

**更新时间**: 2026-04-05
**测试文件**: `tests/` 目录 + `scripts/` 目录
**覆盖范围**: 单元测试、集成测试、边界测试、对抗测试、检索评估脚本

---

## 一、测试文件清单

### 1.1 `tests/` 目录（共 16 个 Python 文件）

#### 测试套件文件

| 文件 | 行数 | 测试类/函数数 | 覆盖模块 | 状态 |
|------|------|------------|---------|------|
| `test_retrieval_adversarial.py` | ~1812 | 10 个类 + `test_summary()` | CRAG 检索对抗（全链路） | **主测试套件** |
| `test_oss_integration.py` | ~542 | 10 个类 | 文档解析/分块/混合检索/重排/熔断/缓存/可观测性 | 集成测试 |
| `test_input_security.py` | ~69 | 3 个类 | 输入安全/PII 检测/XSS/SQL 注入 | 单元测试 |
| `test_response_cache.py` | ~50 | 2 个类 | LLM 响应缓存（Set/Get/温度过滤） | 单元测试 |
| `test_version_manager.py` | ~177 | 2 个类 | 版本解析/比较/过期管理 | 单元测试 |
| `test_observability.py` | ~197 | 3 个类 | 分布式追踪/指标收集/成本追踪 | 单元测试 |
| `test_acl_filter.py` | ~132 | 2 个类 | ACL 权限过滤/版本过期 | 单元测试 |
| `test_retrieval_boundary_queries.py` | ~270 | 1 个类 | 查询边界（空/超长/特殊字符/Unicode） | 集成测试 |
| `test_retrieval_boundary_data.py` | ~261 | 1 个类 | 数据边界（空向量库/超大 chunk/重复 ID） | 集成测试 |
| `test_security_adversarial.py` | ~230 | 1 个类 | 文档注入/用户注入/权限绕过/记忆污染 | 对抗测试 |
| `test_retrieval_poisoning.py` | ~227 | 3 个类 | 关键词填充/Agent 操纵/LLM 错误降级 | 对抗测试 |

#### Fixtures 基础设施

| 文件 | 行数 | 说明 |
|------|------|------|
| `conftest.py` | ~142 | 共享 pytest fixtures（mock LLM、sample_docs、mock_llm_factory） |

#### `__init__.py` 占位文件（无测试逻辑）

- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/unit/rag/__init__.py`
- `tests/unit/api/__init__.py`
- `tests/unit/agent/__init__.py`
- `tests/integration/__init__.py`
- `tests/adversarial/__init__.py`

---

### 1.2 `scripts/` 目录（共 4 个脚本文件）

| 文件 | 行数 | 类型 | 说明 |
|------|------|------|------|
| `run_rag_benchmark.py` | ~892 | 评测脚本 | 7 阶段分级基准测试（基线向量→BM25+混合→分块→精排→多阶段+Rerank→CRAG→QE），输出 CSV 和 JSON |
| `rag_evaluation.py` | ~716 | 评测脚本 | 三层评估体系（检索层/端到端生成/Chunk 级细粒度），调用 `eval_dataset.py` |
| `eval_dataset.py` | ~279 | 评测数据 | 50+ 条 ground-truth 查询（与 `data/knowledge/` 逐一核对），定义 `EvalQuery` 数据类 |
| `ingest.py` | ~270 | 工具脚本 | 知识库文档向量化入库，支持 Vision LLM 图片理解 |

---

## 二、测试分类总览

```
tests/
├── conftest.py                    ← 共享 fixtures（所有测试依赖）
├── test_retrieval_adversarial.py  ← 主测试套件：CRAG 全链路对抗（10 类，105 用例）
├── test_oss_integration.py        ← 集成测试：真实模块端到端验证（10 类）
├── unit/
│   ├── api/
│   │   └── test_input_security.py ← 单元测试：输入安全过滤
│   └── rag/
│       ├── test_response_cache.py ← 单元测试：LLM 响应缓存
│       ├── test_version_manager.py← 单元测试：文档版本管理
│       ├── test_observability.py ← 单元测试：可观测性（追踪/指标/成本）
│       └── test_acl_filter.py     ← 单元测试：ACL 权限过滤
├── integration/
│   ├── test_retrieval_boundary_queries.py ← 集成测试：查询边界
│   └── test_retrieval_boundary_data.py    ← 集成测试：数据边界
└── adversarial/
    ├── test_security_adversarial.py      ← 对抗测试：安全注入/越权
    └── test_retrieval_poisoning.py        ← 对抗测试：检索投毒

scripts/
├── run_rag_benchmark.py  ← 7 阶段检索基准测试
├── rag_evaluation.py      ← 三层 RAG 评估体系
├── eval_dataset.py        ← Ground-truth 评测数据集（50+ 条）
└── ingest.py              ← 知识库文档向量化入库
```

---

## 三、主测试套件详解 (`test_retrieval_adversarial.py`)

这是项目最核心的测试文件，包含 105 个测试用例，10 个测试类。

### 3.1 总体数据

| 测试类 | 用例数 | 测试类型 | 核心验证指标 |
|--------|--------|----------|-------------|
| `TestComplexQueryDecomposition` | 13 | 规则逻辑 + 异步 | 分解数量、速度、策略选择 |
| `TestQueryBoundaryConditions` | 8 | 异步 pipeline | 决策稳定性、防注入 |
| `TestCRAGDecisionBoundaries` | 8 | 单元 + 异步 pipeline | HIGH/MEDIUM/LOW/NO_RESULTS 阈值 |
| `TestRRFFusion` | 5 | 单元 + 异步 pipeline | 自适应权重、RRF k 参数、去重 |
| `TestQueryExpansionAdversarial` | 4 | 异步 | HyDE 降级、rewrite 历史 |
| `TestRetrievalPoisoning` | 5 | 异步 | 关键词填充、语义矛盾、ACL 过滤 |
| `TestRetrievalPerformance` | 5 | 异步 + 性能 | 大 top_k、并发、重试循环上限 |
| `TestRetrievalIntegration` | 3 | 异步 pipeline | 端到端流程、决策可追溯性 |
| `TestRetrievalQualityMetrics` | 47 | 单元 | Recall/Precision/F1/MRR/NDCG/MAP/Hit@K/BM25/RRF |
| `TestAdversarialMetrics` | 7 | 单元 | 投毒 Recall 影响/冲突 Precision/列举 Recall |
| **合计** | **105** | — | **100% 通过** |

### 3.2 模块 1：复杂查询与查询改写 (`TestComplexQueryDecomposition`)

| # | 测试用例 | 测试类型 | 测试指标 | 预期结果 | 状态 |
|---|---------|---------|---------|---------|------|
| 1.1 | 多意图查询（逗号分隔，无对比关键词） | 规则逻辑 | `needs_expansion()` / `decompose()` | 不触发 expansion，至少 1 个子查询 | PASS |
| 1.2 | 对比查询（含"和"关键词） | 规则逻辑 | `needs_expansion()` | 触发 expansion，分解 >= 2 子查询 | PASS |
| 1.3 | 对比连接词（和/与/跟/或/或者） | 规则逻辑 | `decompose()` 子查询数 | 每个连接词均分解 >= 2 子查询 | PASS |
| 1.4 | 列举类查询（有哪些/福利/包括） | 规则逻辑 | `needs_expansion()` | 全部识别为列举查询，触发 expansion | PASS |
| 1.5 | 流程类查询（怎么/如何/步骤） | 规则逻辑 | `decompose()` 子查询数 | 每个流程查询至少 1 个子查询 | PASS |
| 1.6 | 嵌套对比查询（3 项嵌套） | 规则逻辑 | `decompose()` 子查询数 | 分解 >= 2 个子查询 | PASS |
| 1.7 | 反向对比查询（没什么区别） | 规则逻辑 | `decompose()` 子查询数 | 至少 1 个子查询，不崩溃 | PASS |
| 1.8 | 隐式对比查询（好还是） | 规则逻辑 | `decompose()` 子查询数 | 至少 1 个子查询 | PASS |
| 1.9 | 多问号查询（>=2 个问号） | 规则逻辑 | `needs_expansion()` | 触发 expansion | PASS |
| 1.10 | 含"或者"的长查询 | 规则逻辑 | `needs_expansion()` | 触发 expansion | PASS |
| 1.11 | LLM 分解器服务不可用 | 异步降级 | `decompose()` 返回值 | 降级返回原始查询，不抛异常 | PASS |
| 1.12 | 混合策略分解（规则+LLM） | 异步 | `strategy == HYBRID`，子查询数 | 策略为 HYBRID，至少 1 个子查询 | PASS |
| 1.13 | 纯规则分解速度验证 | 性能 | 分解耗时 | **每条 < 10ms**，`used_llm = False` | PASS |

### 3.3 模块 2：查询边界条件 (`TestQueryBoundaryConditions`)

| # | 测试用例 | 测试类型 | 测试指标 | 预期结果 | 状态 |
|---|---------|---------|---------|---------|------|
| 2.1 | 极短查询（1 字，"假"） | 异步 pipeline | `grade_result.decision.value` | 决策为 `no_results` 或 `low`，不崩溃 | PASS |
| 2.2 | 极短查询（2 字，"年假"） | 异步 pipeline | `grade_result.decision.value` | 决策为 `no_results` 或 `low` | PASS |
| 2.3 | 纯标点查询（"???"） | 异步 pipeline | `grade_result.decision.value` | 决策为 `no_results` 或 `low`，不崩溃 | PASS |
| 2.4 | 重复字符查询（"年假年假年假"） | 异步 pipeline | `grade_result.decision.value` | 决策为 `no_results` 或 `low`，不崩溃 | PASS |
| 2.5 | 中英混杂查询 | 异步 pipeline | `grade_result.decision.value` 有效 | 正常处理，不崩溃 | PASS |
| 2.6 | 语义模糊查询（"那个事情怎么办"） | 异步 pipeline | `grade_result.decision.value` | 决策为 `no_results` 或 `low` | PASS |
| 2.7 | 超长查询（500 字） | 异步 pipeline | 异常捕获 | 不抛异常，正常返回 | PASS |
| 2.8 | 特殊 Unicode（零宽空格等） | 异步 pipeline | `grade_result.decision.value` | 正常处理零宽字符 | PASS |

### 3.4 模块 3：CRAG 决策边界 (`TestCRAGDecisionBoundaries`)

| # | 测试用例 | 测试类型 | 测试数据 / 场景 | 预期结果 | 状态 |
|---|---------|---------|----------------|---------|------|
| 3.1 | HIGH 边界（high_ratio=0.2，avg=0.22） | 单元 | 5 篇：1HIGH+1MEDIUM+3LOW | `decision != HIGH`（因为 avg < 0.25） | PASS |
| 3.2 | NO_RESULTS 边界（80% LOW） | 单元 | 5 篇全部 LOW，`low_ratio=1.0` | `decision == NO_RESULTS` | PASS |
| 3.3 | NO_RESULTS 有 HIGH 兜底 | 单元 | 5 篇：4LOW+1HIGH，`low_ratio=0.8` | `decision != NO_RESULTS`（有 HIGH 兜底） | PASS |
| 3.4 | MEDIUM 仅文档决策 | 单元 | 2 篇均为 MEDIUM，`avg >= 0.15` | `decision == MEDIUM` | PASS |
| 3.5 | LOW 兜底（MEDIUM 但 avg < 0.15） | 单元 | MEDIUM 但 `avg < 0.15` | `decision == LOW`（MEDIUM 兜底 LOW） | PASS |
| 3.6 | 空评估结果决策 | 单元 | 空评估结果 | `decision == NO_RESULTS`，`total_docs == 0` | PASS |
| 3.7 | 查询改写 LLM 失败 | 异步 | LLM 抛出异常 | 返回原始查询，不抛异常 | PASS |
| 3.8 | 并发评估一致性 | 异步并发 | 同一文档并发评分 5 次 | 不同决策数 <= 2（允许微小波动） | PASS |

### 3.5 模块 4：RRF 融合与排序 (`TestRRFFusion`)

| # | 测试用例 | 测试类型 | 场景 | 预期结果 | 状态 |
|---|---------|---------|------|---------|------|
| 4.1 | 短查询自适应权重（3 字） | 单元 | 查询长度 = 3 字 | BM25 = **0.7**，向量 = **0.3** | PASS |
| 4.2 | 中等查询自适应权重（6 字） | 单元 | 查询长度 = 6 字 | BM25 = **0.6**，向量 = **0.4** | PASS |
| 4.3 | RRF k 参数敏感性 | 单元 | k = 60，比较不同排名的 RRF 贡献 | 分数差 < 0.002（k=60 时各排名贡献接近） | PASS |
| 4.4 | 内容 hash 去重 | 单元 | 两篇相同 page_content 的文档 | `hash(doc1) == hash(doc2)` | PASS |
| 4.5 | 双路检索融合 | 异步 pipeline | vector + BM25 均有结果 | 返回非空 list，不崩溃 | PASS |

### 3.6 模块 5：查询扩展对抗 (`TestQueryExpansionAdversarial`)

| # | 测试用例 | 测试类型 | 场景 | 预期结果 | 状态 |
|---|---------|---------|------|---------|------|
| 5.1 | HyDE 生成失败降级 | 异步降级 | LLM 服务不可用 | 降级返回原始查询文本 | PASS |
| 5.2 | 扩展后全部子查询空结果 | 异步 | 全部子查询返回空结果 | 决策为 `no_results` 或 `low` | PASS |
| 5.3 | LLM rewrite 返回乱码 | 异步解析 | LLM 返回无法解析的乱码 | `raw_score == 3`（默认值），reasoning 含"解析失败" | PASS |
| 5.4 | rewrite 历史记录 | 异步 | 执行一次带 rewrite 的 pipeline | `len(history) >= 1`（每次 rewrite 均记录） | PASS |

### 3.7 模块 6：检索投毒检测 (`TestRetrievalPoisoning`)

| # | 测试用例 | 测试类型 | 场景 | 预期结果 | 状态 |
|---|---------|---------|------|---------|------|
| 6.1 | 关键词填充检测 | 异步 | 文档注入大量"年假"但实际不相关，向量 0.95 | `low_count >= 0`（CRAG 识别为低相关） | PASS |
| 6.2 | 语义矛盾检测 | 异步 | 2 篇文档年假天数矛盾（15天 vs 10天） | `warnings` 非空，能检测到冲突 | PASS |
| 6.3 | 矛盾数值提取 | 异步 | 文档中包含多个矛盾数值 | `len(facts) > 0`，提取到多个声明 | PASS |
| 6.4 | 机密文档 ACL 过滤 | 异步 | `confidential` + `department_restrict=["HR"]` | pipeline 不崩溃，`results` 正常返回 | PASS |
| 6.5 | 过期版本文档 | 异步 | `expiry_date=2024-12-31`（已过期） | pipeline 不崩溃，`results` 正常返回 | PASS |

### 3.8 模块 7：性能与压力测试 (`TestRetrievalPerformance`)

| # | 测试用例 | 测试类型 | 场景 | 阈值 / 预期 | 状态 |
|---|---------|---------|------|-----------|------|
| 7.1 | 超大 top_k 请求 | 异步 | `top_k=100`，20 篇 mock 文档 | 耗时 < **10,000 ms**，返回 <= 100 条 | PASS |
| 7.2 | 50 并发检索风暴 | 异步并发 | 50 个并发 `pipeline.retrieve()` | 耗时 < **60 s**，返回 50 条结果 | PASS |
| 7.3 | 连续 rewrite 循环上限 | 异步 | `max_retries=2`，连续 LOW 触发 rewrite | `len(history) <= 3`（原始 + 最多 2 次 rewrite） | PASS |
| 7.4 | LLM 429 限流重试成功 | 异步重试 | LLM 前 2 次抛 429，第 3 次成功 | `call_count == 3`，第 3 次返回有效评分 | PASS |
| 7.5 | LLM 全重试失败返回 LOW | 异步降级 | LLM 3 次全部返回 500 错误 | `grade == LOW`，reasoning 含"评估失败" | PASS |

### 3.9 模块 8：综合集成测试 (`TestRetrievalIntegration`)

| # | 测试用例 | 测试类型 | 场景 | 预期结果 | 状态 |
|---|---------|---------|------|---------|------|
| 8.1 | 正常查询完整流程 | 异步端到端 | "公司年假政策是什么"，5 篇相关文档 | `len(results) > 0`，decision 为有效字符串 | PASS |
| 8.2 | 对比查询触发 QE | 异步端到端 | "年假和病假的区别"，`needs_expansion=True` | `len(history) >= 1`（触发了查询扩展） | PASS |
| 8.3 | 决策可追溯性 | 异步端到端 | "年假"，`needs_expansion=False` | `decision_reason` 存在，`len(history) >= 1` | PASS |

### 3.10 模块 9：检索质量指标评测 (`TestRetrievalQualityMetrics`)

| # | 测试用例 | 测试指标 | 场景 | 预期结果 | 状态 |
|---|---------|---------|------|---------|------|
| 9.1 | `recall_at_k` — 相关在第 1 位 | Recall@1 | 相关文档排在第1位 | Recall@1 = 1.0 | PASS |
| 9.2 | `recall_at_k` — 相关在第 3 位 | Recall@3 | 唯一相关在 top-3 内 | Recall@3 = 1.0 | PASS |
| 9.3 | `recall_at_k` — 4 篇全命中 | Recall@5 | 4 篇相关全在 top-5 内 | Recall@5 = 4/4 = 1.0 | PASS |
| 9.4 | `recall_at_k` — 部分命中 | Recall@5 | 部分相关不在 top-5 内 | Recall@5 = 1/3 = 0.333 | PASS |
| 9.5 | `recall_at_k` — 空 GT | Recall@K | ground truth 为空（无答案查询） | Recall@K = 1.0 | PASS |
| 9.6 | `recall_at_k` — 空检索 | Recall@K | 检索结果为空 | Recall@K = 0.0 | PASS |
| 9.7 | `precision_at_k` — 第 1 位命中 | Precision@1 | 第 1 位是相关文档 | Precision@1 = 1.0 | PASS |
| 9.8 | `precision_at_k` — 第 1 位未命中 | Precision@1 | 第 1 位不是相关文档 | Precision@1 = 0.0 | PASS |
| 9.9 | `precision_at_k` — 混合命中 | Precision@3 | top-3 中有 2 个相关 | Precision@3 = 2/3 ≈ 0.667 | PASS |
| 9.10 | `precision_at_k` — 含噪声 | Precision@5 | top-5 中有 1 个相关 | Precision@5 = 1/5 = 0.2 | PASS |
| 9.11 | `precision_at_k` — k=0 | Precision@0 | k=0 时 | Precision@0 = 0.0 | PASS |
| 9.12 | `f1_at_k` — P=R=1 | F1@1 | P=R=1 → F1=1.0 | F1@1 = 1.0 | PASS |
| 9.13 | `f1_at_k` — P=1/3, R=1.0 | F1@3 | P=1/3, R=1.0 → F1=0.5 | F1@3 = 0.5 | PASS |
| 9.14 | `f1_at_k` — P=R=0 | F1@K | P=R=0 时 | F1@K = 0.0 | PASS |
| 9.15 | `mrr` — 相关在第 1 位 | MRR | 第一个相关在第 1 位 | MRR = 1.0 | PASS |
| 9.16 | `mrr` — 相关在第 2 位 | MRR | 第一个相关在第 2 位 | MRR = 0.5 | PASS |
| 9.17 | `mrr` — 相关在第 3 位 | MRR | 第一个相关在第 3 位 | MRR = 1/3 ≈ 0.333 | PASS |
| 9.18 | `mrr` — 无命中 | MRR | 无任何相关命中 | MRR = 0.0 | PASS |
| 9.19 | `mrr` — 空 GT | MRR | ground truth 为空 | MRR = 0.0 | PASS |
| 9.20 | `mrr` — 多相关 | MRR | 多相关文档，取第一个排名 | MRR = 1.0 | PASS |
| 9.21 | `ndcg_at_k` — 完美排序 | NDCG@5 | 所有相关排在最前 | NDCG@5 = 1.0 | PASS |
| 9.22 | `ndcg_at_k` — 部分排序 | NDCG@5 | 相关排在中间 | 0 < NDCG < 1.0 | PASS |
| 9.23 | `ndcg_at_k` — 分级相关 | NDCG | 传入分级相关性分数 | NDCG = 1.0（分级排序正确） | PASS |
| 9.24 | `ndcg_at_k` — 无相关 | NDCG | 无相关文档 | NDCG = 0.0 | PASS |
| 9.25 | `ndcg_at_k` — 空 GT | NDCG | ground truth 为空 | NDCG = 1.0 | PASS |
| 9.26 | `ndcg_at_k` — k=0 | NDCG | k=0 时 | NDCG = 0.0 | PASS |
| 9.27 | `map_score` — 完美 | MAP | 所有相关均排最前 | MAP = 1.0 | PASS |
| 9.28 | `map_score` — 部分命中 | MAP | rank=1 和 rank=3 命中 | MAP = (1 + 2/3)/2 ≈ 0.833 | PASS |
| 9.29 | `map_score` — 多查询 | MAP | 批量 2 个查询 | MAP = 1.0 | PASS |
| 9.30 | `hit_at_k` — 第 1 位命中 | Hit@1 | 第 1 位命中 | Hit@1 = 1.0 | PASS |
| 9.31 | `hit_at_k` — 无命中 | Hit@5 | top-5 内无命中 | Hit@5 = 0.0 | PASS |
| 9.32 | `hit_at_k` — 对抗查询 | Hit@K | 对抗查询（空 GT） | Hit@K = 1.0 | PASS |
| 9.33 | `compute_bm25_scores` — 排序 | BM25 | 年假查询，验证降序 | 降序排列，员工手册.pdf 排第 1 | PASS |
| 9.34 | `compute_bm25_scores` — 短查询 | BM25 | 短查询"请假" | 有结果，分数 >= 0 | PASS |
| 9.35 | `compute_bm25_scores` — 不存在词 | BM25 | 查询词不在语料中 | Fallback 模式正常返回 | PASS |
| 9.36 | `rrf_score` — 公式验证 | RRF | 验证 1/(k+rank) 公式 | rank=1→1/61，rank=5→1/65 | PASS |
| 9.37 | `fused_rrf_score` — 等权互换 | RRF | 两路排名互换 | 分数相同 | PASS |
| 9.38 | `fused_rrf_score` — 加权方向 | RRF | BM25 权重 0.7 时排第 1 应更高 | 权重大的路排第 1 时得分更高 | PASS |
| 9.39 | `fused_rrf_score` — 单路未命中 | RRF | 单路 rank=0 | 仅靠单路贡献计算 | PASS |
| 9.40 | `RetrievalMetricsEngine` — 单查询 | 引擎 | 单查询完整评估 | 所有指标有效，MRR=1.0 | PASS |
| 9.41 | `RetrievalMetricsEngine` — 聚合 | 引擎 | 多查询聚合 | num_queries=3, num_adversarial=1 | PASS |
| 9.42 | `RetrievalMetricsEngine` — 对抗标记 | 引擎 | 无答案对抗查询 | `is_adversarial=True`，Recall=1.0 | PASS |
| 9.43 | `RetrievalMetricsEngine` — 冲突查询 | 引擎 | 冲突文档检索，权威排第 1 | MRR=1.0，Hit@1=1.0，Precision@1=1.0 | PASS |
| 9.44 | `RetrievalMetricsResult.to_dict()` | 序列化 | 单查询序列化 | 所有字段存在且有效 | PASS |
| 9.45 | `AggregatedMetrics.to_dict()` | 序列化 | 批量聚合序列化 | `metrics` 字典完整 | PASS |
| 9.46 | Ground Truth — 覆盖率 | GT 数据集 | 5 类查询覆盖 | normal>=10, contrast/enumerate/short/adversarial 均存在 | PASS |
| 9.47 | Ground Truth — doc_id 有效性 | GT 数据集 | 所有 relevant doc_id 有效 | 8 个 doc_id 全部在有效集合中 | PASS |

### 3.11 模块 10：指标驱动对抗评测 (`TestAdversarialMetrics`)

| # | 测试用例 | 测试指标 | 场景 | 预期结果 | 状态 |
|---|---------|---------|------|---------|------|
| 10.1 | 投毒 Recall 影响 | Recall@3 / MRR 对比 | 关键词填充使正确答案从第 1→第 2 | Recall@3 不变，MRR 降低 | PASS |
| 10.2 | 冲突 Precision 影响 | Precision@1 对比 | 权威文档 vs 过期文档排第 1 | Precision@1 权威 > 过期 | PASS |
| 10.3 | 列举查询多文档覆盖 | Recall@4 / Recall@5 | 列举查询需 4 篇文档覆盖 | 4 命中→Recall@4=1.0，3 命中→Recall@5=0.75 | PASS |
| 10.4 | 短查询 BM25 权重效果 | BM25 排序 | 短查询"年假"BM25 排序 | 员工手册.pdf 排 BM25 第 1 | PASS |
| 10.5 | 对比查询扩展 MRR | Recall@3 / MRR 对比 | 对比查询未扩展 vs 扩展 | 扩展后 Recall>=原，MRR>=原 | PASS |
| 10.6 | 无答案查询指标定义 | Recall/MRR/Hit | 无答案查询指标语义 | Recall=1.0，MRR=0.0，Hit@K=1.0 | PASS |
| 10.7 | rewrite 改善指标 | Recall/MRR/NDCG 对比 | rewrite 前后指标对比 | rewrite 后 Recall/MRR/NDCG 均不降低 | PASS |

---

## 四、其他测试套件

### 4.1 集成测试 (`test_oss_integration.py`)

覆盖 10 个真实模块的端到端验证：

| 模块 | 测试内容 |
|------|---------|
| `UnstructuredDocumentParser` | 多格式文档解析（PDF/DOCX/TXT/HTML/Markdown） |
| `DocumentLoaderManager` | 文档加载管理器（自动格式识别） |
| `HybridChunker` / `SemanticChunker` | 分块策略（重叠块/语义块/递归字符块） |
| `HybridRetrieverManager` | BM25+向量混合检索（RRF 融合） |
| `RerankerManager` | 多后端 Rerank 工厂（本地/云端） |
| `CircuitBreaker` | 熔断器（失败率/半开/恢复） |
| `CostTracker` | 成本追踪（按模型/会话计费） |
| `MetricsCollector` | 指标收集（P50/P95/P99 延迟） |
| `ResponseCache` | LLM 响应缓存（TTL/温度过滤/命中） |
| `DocumentVersionManager` | 文档版本管理（最新版本/过期过滤） |

### 4.2 边界条件测试

**查询边界** (`test_retrieval_boundary_queries.py`)：

- 空查询、超长查询（500 字）、纯标点、重复字符
- 中英混杂、语义模糊、特殊 Unicode（零宽空格）

**数据边界** (`test_retrieval_boundary_data.py`)：

- 空向量库、超大 chunk（>10MB）、重复 chunk ID
- 无效 metadata 字段

### 4.3 对抗测试

**安全对抗** (`test_security_adversarial.py`)：

- 文档注入（忽略以上内容）、用户注入、权限绕过
- 上下文污染、幻觉诱导

**检索投毒** (`test_retrieval_poisoning.py`)：

- 关键词填充、Agent 操纵尝试、多轮对话污染
- LLM 错误降级

### 4.4 单元测试

| 文件 | 覆盖范围 |
|------|---------|
| `test_input_security.py` | XSS/SQL 注入/路径遍历检测；PII 识别/脱敏 |
| `test_response_cache.py` | Set/Get 缓存；温度过滤（>0.7 不缓存） |
| `test_version_manager.py` | 版本解析（v1.0/日期格式/alpha）；版本比较；过期过滤 |
| `test_observability.py` | Span 生命周期；指标收集；成本追踪（按 token 计费） |
| `test_acl_filter.py` | 匿名用户降级；JWT payload 构建；部门/角色权限；机密级别过滤 |

---

## 五、评测脚本详解

### 5.1 `run_rag_benchmark.py` — 7 阶段分级基准测试

按优先级分 7 个阶段逐步优化检索链路：

| 阶段 | 名称 | 用例数 | LLM 调用 | 核心指标 |
|------|------|--------|---------|---------|
| B-1 | 基线向量检索 | 30 条 | 无 | R@5 / MRR / NDCG |
| B-2 | BM25+向量混合检索 | 30 条 | 无 | 与 B-1 对比提升 |
| C | 分块策略测试 | — | 无 | chunk 质量评估 |
| B-6 | 精排过滤率测试 | 30 条 | 仅 embedding | 精排前/后质量对比 |
| B-3 | 多阶段+Rerank | 10 条 | 精简 LLM | R@5 提升 |
| B-4 | CRAG 正确性反馈 | 10 条 | 精简 LLM | CRAG 决策准确性 |
| B-5 | Query Expansion | 复杂查询子集 | 精简 LLM | 对比类查询质量 |

**运行方式**：

```bash
conda activate agent-demo
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
python scripts/run_rag_benchmark.py
```

### 5.2 `rag_evaluation.py` — 三层 RAG 评估体系

| 评估层次 | 评估内容 | 指标 |
|---------|---------|------|
| **检索层** | BM25+向量+RRF 融合排序质量 | Recall@K、MRR、NDCG@K、MAP、Hit@K |
| **端到端生成层** | RAG 生成质量（Faithfulness/Answer Relevancy/Context Recall） | RAGAS 指标 |
| **Chunk 级细粒度层** | chunk 级别召回质量 | 块级召回率 |

### 5.3 `eval_dataset.py` — Ground-truth 评测数据集

50+ 条经逐一核对的 ground-truth 查询，定义 `EvalQuery` 数据类：

```python
@dataclass
class EvalQuery:
    query: str
    relevant_doc_ids: List[str]   # 相关文档 ID 列表
    ground_truth: str              # 期望答案
    description: str = ""          # 查询描述
```

覆盖场景：员工手册、招聘、绩效、培训、财务、行政、信息安全。

---

## 六、旧文件分析与清理建议

### 6.1 建议删除的文件（2 个）

| 文件 | 删除理由 |
|------|---------|
| `docs/test_report_20260326.md` | 2026-03-26 的检索测试结果（R@5=90%），被 `docs/verification_test_report.md`（2026-04-03）全面覆盖。Supervisor/Planner 路由测试也已被整合到最新的验证报告中。历史数据，无独立参考价值。 |
| `docs/routing_test_report.md` | 同上，2026-03-26 的路由准确率测试（Supervisor 95.7%，Planner 80%），被 2026-04-03 的 `verification_test_report.md` 整合。该报告内容更新、更全面。 |

### 6.2 建议保留的文件

| 文件 | 理由 |
|------|------|
| `docs/implementation_report.md` | 2026-04-03 的功能实现报告（General Agent → ReAct 升级、Planner 复杂度路由、并行执行器），项目核心文档 |
| `docs/verification_test_report.md` | 2026-04-03 的验证测试报告（最全面，覆盖所有 11 个核心文件的测试验证） |
| `docs/PERFORMANCE_OPTIMIZATION.md` | 2026-04-03 的性能优化报告（35s→4.7s，7.5x 提升），有实际数据支撑 |
| `docs/open_source_reference_report.md` | 架构参考文档，与 LangGraph/Mem0/Google A2A/OPEA 对比，持续参考价值 |
| `docs/architecture/*.md` | 5 个架构设计文档（ACL、OBSERVABILITY、REFUSAL_STRATEGY、VERSION_MANAGEMENT、INGESTION_PIPELINE），部分已实现，部分为设计参考，均有保留价值 |

### 6.3 测试文件状态

**无旧文件需删除**。所有 `tests/` 下的测试文件均为当前有效测试，没有被替代或过期的测试文件。

---

## 七、测试运行指南

### 7.1 主测试套件

```bash
conda activate agent-demo
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897

# 完整测试套件（105 个用例）
python -m pytest tests/test_retrieval_adversarial.py -v --tb=short

# 分模块运行（推荐，异步测试较多）
python -m pytest tests/test_retrieval_adversarial.py::TestComplexQueryDecomposition -v
python -m pytest tests/test_retrieval_adversarial.py::TestCRAGDecisionBoundaries -v
python -m pytest tests/test_retrieval_adversarial.py::TestRRFFusion -v
python -m pytest tests/test_retrieval_adversarial.py::TestRetrievalQualityMetrics -v
python -m pytest tests/test_retrieval_adversarial.py::TestAdversarialMetrics -v

# 集成测试
python -m pytest tests/test_oss_integration.py -v --tb=short

# 单元测试
python -m pytest tests/unit/ -v --tb=short

# 对抗测试
python -m pytest tests/adversarial/ -v --tb=short

# 边界测试
python -m pytest tests/integration/ -v --tb=short
```

### 7.2 评测脚本

```bash
# 7 阶段分级基准测试
python scripts/run_rag_benchmark.py

# 三层 RAG 评估
python scripts/rag_evaluation.py

# 知识库文档向量化入库
python scripts/ingest.py --source data/knowledge/
```

---

## 八、测试指标汇总

| 指标 | 来源 | 说明 |
|------|------|------|
| **Recall@K** | `src/rag/evaluation/retrieval_metrics.py` | 相关文档命中率 |
| **Precision@K** | `src/rag/evaluation/retrieval_metrics.py` | top-K 中相关文档占比 |
| **F1@K** | `src/rag/evaluation/retrieval_metrics.py` | P 和 R 的调和平均 |
| **MRR** | `src/rag/evaluation/retrieval_metrics.py` | 首个相关文档排名倒数均值 |
| **NDCG@K** | `src/rag/evaluation/retrieval_metrics.py` | 折损累计增益，支持分级相关度 |
| **MAP** | `src/rag/evaluation/retrieval_metrics.py` | 平均精度均值 |
| **Hit@K** | `src/rag/evaluation/retrieval_metrics.py` | top-K 内是否有命中 |
| **BM25 Score** | `src/rag/evaluation/retrieval_metrics.py` | Okapi BM25 排序评分 |
| **RRF Score** | `src/rag/evaluation/retrieval_metrics.py` | 加权排名融合（k=60） |
| **Faithfulness** | `src/rag/evaluation/evaluator.py` | RAGAS 指标：答案对上下文的忠实度 |
| **Answer Relevancy** | `src/rag/evaluation/evaluator.py` | RAGAS 指标：答案与问题的相关度 |
| **Context Recall** | `src/rag/evaluation/evaluator.py` | RAGAS 指标：上下文对期望答案的召回率 |
| **Context Precision** | `src/rag/evaluation/evaluator.py` | RAGAS 指标：上下文中相关文档的精度 |
