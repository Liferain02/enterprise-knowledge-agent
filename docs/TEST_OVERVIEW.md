# 企业知识库检索系统测试总览

**项目**: enterprise-knowledge-agent
**生成日期**: 2026-04-05
**测试套件版本**: v2.0（含检索质量指标评测）

---

## 一、测试文件总目录

### 1.1 测试文件清单

| # | 文件路径 | 类型 | 主要测试内容 | 状态 |
|---|---------|------|------------|------|
| 1 | `tests/conftest.py` | Fixture | 共享 pytest fixtures（mock LLM、sample documents、user context） | 活跃 |
| 2 | `tests/test_retrieval_adversarial.py` | Integration | **主测试套件**：CRAG 完整对抗测试（10 个测试类，105 个用例） | 活跃 |
| 3 | `tests/test_oss_integration.py` | Integration | 真实模块端到端验证（文档解析、分块、混合检索、重排、断路器、成本追踪） | 活跃 |
| 4 | `tests/unit/api/test_input_security.py` | Unit | API 输入安全过滤（消毒、PII 检测/脱敏、XSS/SQL 注入防护） | 活跃 |
| 5 | `tests/unit/rag/test_response_cache.py` | Unit | LLM 响应缓存层（TTL、命中率、并发写入） | 活跃 |
| 6 | `tests/unit/rag/test_version_manager.py` | Unit | 文档版本解析与比较（版本号解析、过期检测） | 活跃 |
| 7 | `tests/unit/rag/test_observability.py` | Unit | 分布式追踪（span）、指标采集、成本追踪 | 活跃 |
| 8 | `tests/unit/rag/test_acl_filter.py` | Unit | ACL 权限过滤与版本过期逻辑 | 活跃 |
| 9 | `tests/integration/test_retrieval_boundary_queries.py` | Integration | 查询端边界条件（空查询、极长查询、特殊字符） | 活跃 |
| 10 | `tests/integration/test_retrieval_boundary_data.py` | Integration | 数据/分块边界条件（空向量库、超大分块、重复 ID） | 活跃 |
| 11 | `tests/adversarial/test_security_adversarial.py` | Adversarial | 安全对抗（文档注入、用户注入、权限提升、上下文污染） | 活跃 |
| 12 | `tests/adversarial/test_retrieval_poisoning.py` | Adversarial | 检索投毒（关键词填充、Agent 操纵、LLM 错误响应） | 活跃 |
| 13 | `scripts/run_rag_benchmark.py` | 评测脚本 | 7 阶段检索 pipeline 基准评测（基线→BM25→分块→重排→CRAG→QE） | 活跃 |
| 14 | `scripts/rag_evaluation.py` | 评测脚本 | 3 层 RAG 评测系统（检索层指标、端到端生成质量、分块级细粒度召回） | 活跃 |
| 15 | `scripts/eval_dataset.py` | 评测数据 | 50+ 查询的 Ground Truth 数据集（EvalQuery 数据类 + EVAL_DATASET） | 活跃 |
| 16 | `scripts/ingest.py` | 工具脚本 | 知识库文档向量化入库脚本 | 活跃 |

### 1.2 测试套件分布

```
tests/
├── conftest.py                         # 全局 fixtures
├── test_retrieval_adversarial.py       # 主对抗测试套件（105 用例）
├── test_oss_integration.py             # 真实模块集成测试（10 类）
├── unit/
│   ├── api/
│   │   └── test_input_security.py      # 输入安全单元测试
│   └── rag/
│       ├── test_response_cache.py       # 响应缓存单元测试
│       ├── test_version_manager.py      # 版本管理单元测试
│       ├── test_observability.py        # 可观测性单元测试
│       └── test_acl_filter.py           # ACL 过滤单元测试
├── integration/
│   ├── test_retrieval_boundary_queries.py  # 查询边界集成测试
│   └── test_retrieval_boundary_data.py     # 数据边界集成测试
└── adversarial/
    ├── test_security_adversarial.py    # 安全对抗测试
    └── test_retrieval_poisoning.py      # 检索投毒对抗测试

scripts/
├── run_rag_benchmark.py                # 7 阶段基准评测
├── rag_evaluation.py                    # 3 层评测系统
├── eval_dataset.py                      # Ground Truth 数据集
└── ingest.py                           # 文档入库工具
```

---

## 二、主测试套件详情 (`tests/test_retrieval_adversarial.py`)

### 2.1 概览

**文件**: `tests/test_retrieval_adversarial.py`
**测试类**: 10 个
**用例总数**: 105 个
**通过率**: 100%

| 模块 | 测试类 | 用例数 | 测试类型 | 核心验证指标 |
|------|--------|--------|----------|-------------|
| 1. 复杂查询与查询改写 | `TestComplexQueryDecomposition` | 13 | 规则逻辑 + 异步 | 分解数量、速度、策略选择 |
| 2. 查询边界条件 | `TestQueryBoundaryConditions` | 8 | 异步 pipeline | 决策结果稳定性、防注入 |
| 3. CRAG 决策边界 | `TestCRAGDecisionBoundaries` | 8 | 单元 + 异步 pipeline | HIGH/MEDIUM/LOW/NO_RESULTS 阈值 |
| 4. RRF 融合与排序 | `TestRRFFusion` | 5 | 单元 + 异步 pipeline | 自适应权重、RRF k 参数、去重 |
| 5. 查询扩展对抗 | `TestQueryExpansionAdversarial` | 4 | 异步 | HyDE 降级、rewrite 历史 |
| 6. 检索投毒检测 | `TestRetrievalPoisoning` | 5 | 异步 | 关键词填充、语义矛盾、ACL 过滤 |
| 7. 性能与压力测试 | `TestRetrievalPerformance` | 5 | 异步 + 性能 | 大 top_k、并发、重试循环上限 |
| 8. 综合集成测试 | `TestRetrievalIntegration` | 3 | 异步 pipeline | 端到端流程、决策可追溯性 |
| 9. 检索质量指标评测 | `TestRetrievalQualityMetrics` | 47 | 单元 | Recall/Precision/F1/MRR/NDCG/MAP/Hit@K/BM25/RRF |
| 10. 指标驱动对抗评测 | `TestAdversarialMetrics` | 7 | 单元 | 投毒 Recall 影响/冲突 Precision/列举 Recall |

### 2.2 核心指标说明

#### CRAG 决策指标

| 指标 | 默认阈值 | 说明 |
|------|---------|------|
| `high_threshold` (`crag_grade_threshold`) | 0.25 | 平均相关分 >= 此值才可能为 HIGH |
| `medium_threshold` (`crag_medium_threshold`) | 0.15 | 平均相关分 >= 此值才可能为 MEDIUM |
| `min_high_ratio` (`crag_min_high_ratio`) | 0.2 (20%) | HIGH 文档占比 >= 此值才可能为 HIGH |
| `no_results_low_ratio` (`crag_no_results_low_ratio`) | 0.8 (80%) | LOW 文档占比 >= 此值才触发 NO_RESULTS |

#### RRF 融合指标

| 查询长度 | BM25 权重 | 向量权重 | 依据 |
|---------|----------|----------|------|
| <= 4 字 | **0.7** | 0.3 | 短查询依赖精确关键词匹配 |
| 5-8 字 | **0.6** | 0.4 | 短查询偏重 BM25 |
| > 8 字 | 配置权重 | 配置权重 | 语义检索占主导 |

- **RRF k 参数**: k = 60，各排名贡献均衡（1/(k+rank)）
- **去重方式**: 内容 hash 去重（`hash(doc.page_content)`）

#### 检索质量指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Recall@K** | \|relevant_in_topk\| / \|all_relevant\| | 命中相关文档的比例 |
| **Precision@K** | \|relevant_in_topk\| / K | top-K 中相关文档的比例 |
| **F1@K** | 2·P·R / (P+R) | Precision 和 Recall 的调和平均 |
| **MRR** | 1 / first_relevant_rank | 首个相关文档排名的倒数均值 |
| **NDCG@K** | DCG@K / IDCG@K | 折损累计增益归一化值 |
| **MAP** | mean(AP_i) | 平均精度均值 |
| **Hit@K** | 1 if any_hit_in_topk else 0 | top-K 内是否有命中文档 |
| **BM25 Score** | Okapi BM25 公式 | 词频-逆文档频率排序 |
| **RRF Score** | 1 / (k + rank) | 排名融合倒数排序（k=60） |

#### 性能指标

| 测试场景 | 阈值 |
|---------|------|
| `top_k=100` 大请求 | < 10,000 ms |
| 50 并发检索风暴 | < 60 s |
| LLM 重试策略 | 最多 3 次（2 次 429 后第 3 次成功） |
| 查询改写循环上限 | <= 3 次（原始查询 + 最多 2 次 rewrite） |

---

## 三、其他测试文件详情

### 3.1 `tests/test_oss_integration.py` — 真实模块集成测试

覆盖 10 个真实模块（无 Mock），验证端到端正确性：

| 测试类 | 覆盖模块 | 关键验证点 |
|--------|---------|-----------|
| `TestUnstructuredDocumentParser` | `src/rag/ingestion/parser.py` | 多格式解析（PDF/Word/Excel/PPT） |
| `TestDocumentLoaderManager` | `src/rag/ingestion/` | 统一加载接口、异常处理 |
| `TestChunkers` | 分块策略 | 固定窗口/语义/递归分块，overlap |
| `TestHybridRetrieverManager` | `hybrid_retriever.py` | BM25 + 向量双路融合 |
| `TestRerankerManager` | `reranker.py` | 重排质量、截断逻辑 |
| `TestCircuitBreaker` | 断路器 | 超时触发、恢复机制 |
| `TestCostTracker` | 成本追踪 | Token 计数、并发安全 |
| `TestMetrics` | 可观测性 | span 创建、指标上报 |
| `TestResponseCache` | 响应缓存 | TTL、LRU 驱逐、并发写入 |
| `TestVersionManager` | 版本管理 | 版本解析、过期检测 |

### 3.2 `tests/unit/api/test_input_security.py` — 输入安全单元测试

| 测试类 | 覆盖内容 |
|--------|---------|
| `TestInputSanitizer` | SQL 注入、XSS、路径遍历、命令注入的消毒处理 |
| `TestPIIFilter` | 姓名/手机号/身份证/邮箱/银行卡号的检测与脱敏 |
| `TestConvenienceFunctions` | 快捷函数的组合使用 |

### 3.3 `tests/unit/rag/` — RAG 单元测试

| 文件 | 测试类 | 覆盖内容 |
|------|--------|---------|
| `test_response_cache.py` | `TestResponseCache` | LLM 响应缓存（TTL、命中率、并发写入安全） |
| `test_version_manager.py` | `TestVersionParsing` + `TestVersionManager` | 语义版本解析（v1.2.3）、日期版本、过期文档过滤 |
| `test_observability.py` | `TestTracer` + `TestMetrics` + `TestCostTracker` | OpenTelemetry span、指标 P50/P95、Token 计数 |
| `test_acl_filter.py` | `TestACLFilter` + `TestVersionExpiration` | 用户/部门/角色权限矩阵、过期版本文档过滤 |

### 3.4 `tests/integration/` — 边界集成测试

| 文件 | 测试类 | 覆盖场景 |
|------|--------|---------|
| `test_retrieval_boundary_queries.py` | `TestQueryBoundary` | 空查询、超长查询（2000字）、Unicode 特殊字符、纯数字查询、表情符号查询 |
| `test_retrieval_boundary_data.py` | `TestDataBoundary` | 空向量库、单分块、多分块边界（499字/500字）、重复 chunk_id |

### 3.5 `tests/adversarial/` — 对抗测试

| 文件 | 测试类 | 对抗场景 |
|------|--------|---------|
| `test_security_adversarial.py` | `TestSecurityAdversarial` | 文档注入（权限绕过）、用户注入（身份伪造）、权限提升（越权访问）、上下文污染、幻觉诱导 |
| `test_retrieval_poisoning.py` | `TestRetrievalPoisoning` + `TestAgentManipulation` + `TestLLMErrors` | 关键词填充、语义矛盾溯源限制、上下文窗口耗尽、多轮对话污染、LLM 500/解析错误降级 |

---

## 四、评测脚本

### 4.1 `scripts/run_rag_benchmark.py` — 7 阶段检索基准评测

评测 7 个递进阶段，从基线到完整 CRAG pipeline：

| 阶段 | 名称 | 核心指标 |
|------|------|---------|
| B-1 | 基线向量检索 | BM25+向量各 0.5 融合 |
| B-2 | BM25+向量混合 | 自适应权重调优 |
| B-3 | 分块策略对比 | 不同 chunk_size 的影响 |
| B-4 | 重排过滤 | 重排后截断对指标的影响 |
| B-5 | 多阶段+重排 | 综合最优配置 |
| C-1 | CRAG 评估 | CRAG HIGH/MEDIUM/LOW 决策对质量的影响 |
| C-2 | 查询扩展 | QE 对比查询的效果 |

输出：每阶段 `recall@1/3/5/10`、`precision@1/3/5`、`mrr`、`ndcg@3/5`、`hit@1/3/5` 及 CSV 报告。

### 4.2 `scripts/rag_evaluation.py` — 3 层评测系统

| 层次 | 评估内容 | 方法 |
|------|---------|------|
| 检索层 | top-K 召回质量 | Recall@K / Precision@K / MRR / NDCG |
| 端到端生成层 | 生成答案质量 | RAGAS（Faithfulness / Answer Relevancy / Context Recall / Precision）|
| 分块级 | 分块边界对召回的影响 | 关键词跨 chunk 的召回率 |

### 4.3 `scripts/eval_dataset.py` — Ground Truth 数据集

定义 `EvalQuery` 数据类和 `EVAL_DATASET` 列表，包含 50+ 查询，涵盖：
- 正常查询（年假/病假/请假流程/福利）
- 对比查询（年假 vs 病假）
- 列举查询（有哪些福利）
- 边界查询（空/超长/特殊字符）
- 对抗查询（注入/投毒）

---

## 五、测试结果汇总

### 5.1 通过率

| 测试类 | 用例数 | 通过率 |
|--------|--------|--------|
| `TestComplexQueryDecomposition` | 13 | 100% |
| `TestQueryBoundaryConditions` | 8 | 100% |
| `TestCRAGDecisionBoundaries` | 8 | 100% |
| `TestRRFFusion` | 5 | 100% |
| `TestQueryExpansionAdversarial` | 4 | 100% |
| `TestRetrievalPoisoning` | 5 | 100% |
| `TestRetrievalPerformance` | 5 | 100% |
| `TestRetrievalIntegration` | 3 | 100% |
| `TestRetrievalQualityMetrics` | 47 | 100% |
| `TestAdversarialMetrics` | 7 | 100% |
| **总计（主套件）** | **105** | **100%** |

### 5.2 核心验证维度

| 维度 | 覆盖测试 | 验证结论 |
|------|---------|---------|
| CRAG 决策边界 | 模块 3 | HIGH 需 `high_ratio >= 0.2` 且 `avg >= 0.25`；NO_RESULTS 需 `low_ratio >= 0.8` |
| 规则分解覆盖度 | 模块 1 | 对比/列举/流程/嵌套/反向/隐式/多问号 7 类全覆盖 |
| 分解性能 | 1.13 | 纯规则分解稳定 < 10ms |
| 自适应权重 | 模块 4 | 短查询 BM25=0.7/0.6，符合设计预期 |
| LLM 降级链路 | 模块 5, 7 | HyDE 失败、解析失败、429/500 错误均正确降级 |
| 重写循环保护 | 7.3 | `max_retries=2` 时 rewrite 历史 <= 3 次 |
| 检索投毒防御 | 模块 6, adversarial | 关键词填充/语义矛盾/ACL/过期文档 完整链路 |
| 并发一致性 | 3.8, 7.2 | 5 次并发评分决策一致；50 并发 < 60s |
| 决策可追溯性 | 8.3 | 每步决策均有 `decision_reason` 和 `history` 记录 |
| 指标正确性 | 模块 9 | Recall/Precision/F1/MRR/NDCG/MAP/Hit 全场景覆盖 |
| 对抗量化评估 | 模块 10 | 投毒 Recall 影响/冲突 Precision/无答案指标 定义正确 |
| 真实模块集成 | `test_oss_integration.py` | 10 类真实模块端到端验证 |
| 输入安全 | `test_input_security.py` | PII 脱敏、XSS/SQL 注入防护 |
| 响应缓存 | `test_response_cache.py` | TTL、LRU、并发安全 |
| 版本管理 | `test_version_manager.py` | 语义版本解析、过期过滤 |
| ACL 过滤 | `test_acl_filter.py` | 用户/部门/角色权限矩阵 |

---

## 六、文档清单与清理建议

### 6.1 当前 docs/ 目录结构

```
docs/
├── implementation_report.md          # 功能实现报告（2026-04-03）
├── verification_test_report.md       # 验证测试报告（2026-04-03）
├── PERFORMANCE_OPTIMIZATION.md       # 性能优化报告（2026-04-03）
├── open_source_reference_report.md  # 开源参考对比报告
├── test_report_20260326.md          # [建议删除] 旧版检索测试报告（2026-03-26）
├── routing_test_report.md           # [建议删除] 旧版路由测试报告（2026-03-26）
└── architecture/
    ├── ACL_DESIGN.md                # ACL 权限隔离设计（部分实现）
    ├── OBSERVABILITY.md             # 可观测性设计（部分实现）
    ├── REFUSAL_STRATEGY.md          # 拒绝策略设计（未实现）
    ├── VERSION_MANAGEMENT.md        # 版本管理设计（部分实现）
    └── INGESTION_PIPELINE.md        # 异步入库 pipeline 设计（部分实现）
```

### 6.2 清理建议

#### 建议删除（2 个，已被新报告覆盖）

| 文件 | 删除原因 |
|------|---------|
| `docs/test_report_20260326.md` | 2026-03-26 的检索测试报告，内容已被 `verification_test_report.md`（2026-04-03）完全覆盖。该文件距今 8 天，是旧版本测试结果的记录，无独立参考价值。 |
| `docs/routing_test_report.md` | 2026-03-26 的路由测试报告，内容同样已被 `verification_test_report.md`（2026-04-03）覆盖。2026-04-03 报告包含 supervisor 路由准确率（95.7%）和 planner 路由准确率（80.0%）的完整测试结果。 |

#### 建议保留（9 个）

| 文件 | 保留理由 |
|------|---------|
| `docs/implementation_report.md` | 2026-04-03 功能实现报告，记录了 General Agent → ReAct 升级、Planner 复杂度路由、并行执行器的完整实现，当前项目状态的核心文档。 |
| `docs/verification_test_report.md` | 2026-04-03 验证测试报告，综合了语法检查、导入检查、Skill 加载、Planner 路由、并行执行依赖分析、图创建、端到端对话测试的完整验证结果。是最新、最权威的项目验证文档。 |
| `docs/PERFORMANCE_OPTIMIZATION.md` | 2026-04-03 性能优化报告，记录了简单查询从 35s 优化到 4.7s（7.5x 提速）的串行执行瓶颈修复过程，有工程参考价值。 |
| `docs/open_source_reference_report.md` | 将项目架构与 LangGraph、Mem0、Google A2A、OPEA 进行对比，是架构决策的参考文档，适合新成员了解项目定位。 |
| `docs/architecture/ACL_DESIGN.md` | ACL 权限隔离设计的完整文档，虽然完整设计尚未完全落地，但 `src/rag/retrieval/acl_filter.py` 中已有部分实现，适合指导后续开发。 |
| `docs/architecture/OBSERVABILITY.md` | 可观测性设计的 SLO 目标文档（延迟 P50/P95/P99、成本追踪），`src/observability/` 模块已有实现，是设计意图和实现之间的桥梁文档。 |
| `docs/architecture/REFUSAL_STRATEGY.md` | 拒绝与降级策略的设计文档，虽未完全实现，但定义了系统在低置信度时的行为边界，是安全合规的重要参考。 |
| `docs/architecture/VERSION_MANAGEMENT.md` | 文档版本管理的完整设计，`src/rag/storage/version_manager.py` 中有对应实现，设计文档和代码相互印证。 |
| `docs/architecture/INGESTION_PIPELINE.md` | 异步入库 pipeline 的设计文档，`src/rag/ingestion/` 模块已有实现，设计文档有助于理解系统的文档生命周期管理。 |

### 6.3 执行删除

```bash
# 删除旧的测试报告
rm docs/test_report_20260326.md
rm docs/routing_test_report.md
```

---

## 七、运行指南

### 7.1 运行主测试套件

```bash
conda activate agent-demo
export set_proxy
conda run -n agent-demo python -m pytest tests/test_retrieval_adversarial.py -v --tb=short
```

### 7.2 分模块运行（推荐）

```bash
# 单元测试（快速）
conda run -n agent-demo python -m pytest tests/unit/ -v --tb=short -k "not async"

# 集成测试
conda run -n agent-demo python -m pytest tests/integration/ -v --tb=short

# 对抗测试
conda run -n agent-demo python -m pytest tests/adversarial/ -v --tb=short

# 指标测试（仅指标验证，无需 LLM）
conda run -n agent-demo python -m pytest tests/test_retrieval_adversarial.py::TestRetrievalQualityMetrics -v
conda run -n agent-demo python -m pytest tests/test_retrieval_adversarial.py::TestAdversarialMetrics -v

# CRAG 核心逻辑（快速）
conda run -n agent-demo python -m pytest tests/test_retrieval_adversarial.py::TestCRAGDecisionBoundaries -v
conda run -n agent-demo python -m pytest tests/test_retrieval_adversarial.py::TestRRFFusion -v
```

### 7.3 运行基准评测

```bash
conda run -n agent-demo python scripts/run_rag_benchmark.py
```

### 7.4 运行完整 RAG 评测

```bash
conda run -n agent-demo python scripts/rag_evaluation.py
```

---

## 八、更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-04-05 | 新增检索质量指标评测模块（Recall/Precision/MRR/NDCG/BM25/RRF），新增 `src/rag/evaluation/retrieval_metrics.py`，测试用例从 51 扩充至 105 个，新增 2 个测试类（`TestRetrievalQualityMetrics`、`TestAdversarialMetrics`），本测试总览文档首次创建 |
| 2026-04-03 | 基础测试套件搭建完成（8 个测试类，51 个用例），CRAG 决策边界、RRF 融合、查询分解、投毒检测全覆盖 |
