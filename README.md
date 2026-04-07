# Enterprise Knowledge Agent

企业级 RAG 智能问答系统，支持多 Agent 协作、长期记忆、流式输出、多用户认证。

## 核心架构

### Multi-Agent 工作流（LangGraph）

```
用户输入 → maybe_summarize → retrieve_mem0_memories → Planner
    ├─ 简单任务 → Supervisor → retrieval_agent_node → generation_agent_node → save_to_mem0 → END
    └─ 复杂任务 → Execute Plan → 各子步骤 → save_to_mem0 → END
```

- **retrieval_agent_node**：Hybrid 检索 → Rerank → CRAG LLM 评估 → 查询改写/分解（条件触发）
- **generation_agent_node**：基于评估后的文档独立生成，强制引用来源
- **冲突检测**：`conflict_detector` 对多文档矛盾信息告警
- **并行执行**：`parallel_executor` 通过 LangGraph Send 原语实现子任务并行
- **Planner**：任务复杂度判断，决定走简单路由还是复杂规划
- **Supervisor**：负责任务分发和结果汇总

### 技术栈

| 组件 | 用途 | 来源 |
|------|------|------|
| LangChain / LangGraph | Agent 编排框架 | `langchain`, `langgraph` |
| bm25s | BM25 关键词检索 | `bm25s` |
| FlashRank | 轻量级本地重排序 | `flashrank` |
| Mem0 | 跨会话语义记忆 | `mem0ai` |
| Arize Phoenix | RAG 可观测性 | `arize-phoenix` |
| OpenTelemetry | 分布式追踪 | `opentelemetry-*` |
| Redis | 评估缓存持久化 | `redis[hiredis]` |
| Unstructured | 多格式文档解析 | `unstructured` |
| ChromaDB | 向量数据库 | `chromadb` |
| Mem0 | MCP 服务器 | 模型上下文协议扩展 |

## 功能特性

- **多用户认证**：JWT + SQLite 用户数据库，支持注册/登录，密码 salted SHA-256
- **流式输出 SSE**：后端 `StreamingResponse`，前端 `ReadableStream` 实时渲染
- **会话隔离**：`{username}_{session_id}` 前缀保证用户间数据隔离
- **Mem0 语义记忆**：跨会话用户偏好持久化
- **Corrective RAG**：LLM-as-judge 评估检索结果，低质量触发查询改写
- **Hybrid 检索**：BM25 + 向量相似度 + RRF 融合排序
- **文档冲突检测**：多文档数值/描述矛盾告警
- **语义摘要压缩**：对话超过阈值自动压缩历史
- **多 Agent 协作**：Planner 任务复杂度判断 + Supervisor 路由 + Worker Agents
- **OTEL 可观测性**：Span/Trace 全链路追踪 + Prometheus 指标
- **动态 Skill 加载**：Skill Loader 自动扫描 `skills/` 目录注册工具
- **MCP 集成**：通过 MCP 协议连接外部工具服务器
- **多模态**：支持上传图片，Vision LLM 理解后入库或提问
- **文档版本管理**：向量库文档版本追踪与过期检测
- **异步入库**：启动不阻塞，文档通过队列异步处理
- **输入安全**：SQL 注入、XSS、Prompt 注入、PII 脱敏检测
- **熔断器**：LLM 调用失败自动熔断，防止级联故障
- **成本追踪**：Token 消耗与 API 调用成本监控

## 快速开始

```bash
# 1. 安装依赖
conda activate agent-demo
pip install -r requirements.txt

# 2. 配置环境变量
cp config/env.template config/.env
# 编辑 config/.env 填入 LLM API Key 等

# 3. 启动服务
python main.py
# 或 Docker Compose（包含 Redis + Prometheus + OTEL Collector + Jaeger）
docker-compose up -d
```

## 项目结构

```
.
├── main.py                              # FastAPI 入口
│
├── config/                              # 配置
│   ├── settings.py                      # Pydantic Settings
│   ├── environment.py                   # 多环境配置加载器（base/staging/production）
│   ├── env.template                     # 环境变量模板
│   ├── base.env / staging.env / production.env
│   ├── mcp_servers.json                 # MCP 服务器配置
│   ├── otel-collector-config.yaml       # OpenTelemetry 配置
│   ├── prometheus.yml                  # Prometheus 抓取配置
│   └── prometheus_alerts.yml           # Prometheus 告警规则
│
├── src/                                 # 源代码
│   ├── agent/                           # LangGraph 多 Agent 编排
│   │   ├── graph.py                     # StateGraph 定义 + 节点注册
│   │   ├── prompts.py                   # Prompt 模板
│   │   ├── checkpointer.py             # SQLite 状态持久化
│   │   ├── agents/
│   │   │   ├── supervisor.py           # Supervisor 路由决策
│   │   │   ├── planner.py             # 任务复杂度判断
│   │   │   ├── knowledge.py           # retrieval_agent + generation_agent 节点
│   │   │   ├── general.py             # 通用 Agent
│   │   │   ├── operation.py           # 操作类 Agent
│   │   │   ├── parallel_executor.py   # 并行执行（Send 原语）
│   │   │   ├── _schemas.py            # 内部数据模型
│   │   │   └── _utils.py              # 内部工具函数
│   │   ├── memory/
│   │   │   └── mem0_manager.py       # Mem0 语义记忆管理
│   │   ├── skills/                    # 动态 Skill Loader
│   │   │   ├── skill_loader.py       # Skill 注册与发现
│   │   │   ├── calculator/           # 计算器 Skill
│   │   │   ├── datetime/             # 日期时间 Skill
│   │   │   ├── file_operation/       # 文件操作 Skill
│   │   │   ├── general/              # 通用 Skill
│   │   │   └── knowledge/            # 知识检索 Skill
│   │   └── tools/
│   │       └── mcp_adapter.py        # MCP 工具适配器
│   │
│   ├── rag/                            # RAG 管线
│   │   ├── retrieval/
│   │   │   ├── retriever.py          # 检索器管理器
│   │   │   ├── hybrid_retriever.py   # BM25 + 向量混合检索 + RRF
│   │   │   ├── reranker.py           # FlashRank / Cohere / BGE 重排序
│   │   │   ├── query_expander.py     # 复杂查询分解 + HyDE
│   │   │   ├── query_cache.py       # 查询缓存
│   │   │   ├── acl_filter.py        # 访问控制列表过滤
│   │   │   └── table_qa.py          # 表格问答
│   │   ├── evaluation/
│   │   │   ├── retrieval_grader.py   # Corrective RAG 评估 + GradeResult
│   │   │   ├── conflict_detector.py # 多文档冲突检测
│   │   │   ├── grade_cache.py       # Redis 评估缓存
│   │   │   └── evaluator.py         # RAGAs 评估集成
│   │   ├── processing/
│   │   │   ├── document_loader.py    # 多格式文档加载
│   │   │   ├── chunker.py           # 语义 + Hybrid 分块
│   │   │   ├── unstructured_loader.py # Unstructured 库集成
│   │   │   └── multimodal.py        # Vision LLM 图片理解
│   │   ├── storage/
│   │   │   ├── vectorstore.py       # ChromaDB 向量存储管理
│   │   │   └── version_manager.py    # 文档版本管理
│   │   ├── ingestion/
│   │   │   ├── document_processor.py # 文档处理流水线
│   │   │   ├── worker.py            # 异步入库 Worker
│   │   │   └── job_queue.py        # 任务队列管理
│   │   └── cache/
│   │       └── response_cache.py    # LLM + 检索响应缓存
│   │
│   ├── api/                            # FastAPI 接口层
│   │   ├── controllers/               # 路由聚合（chat / knowledge / auth / vision）
│   │   ├── routes/
│   │   │   ├── a2a_routes.py        # A2A 协议端点 + Agent Card
│   │   │   └── websocket_routes.py  # WebSocket 实时对话
│   │   ├── services/
│   │   │   ├── chat_service.py      # 聊天业务逻辑
│   │   │   ├── knowledge_service.py # 知识库业务逻辑
│   │   │   └── session_service.py  # 会话管理
│   │   ├── repositories/
│   │   │   └── dao/session_dao.py  # Session / Message DAO
│   │   ├── schemas/                 # Pydantic 请求/响应模型
│   │   ├── middleware/
│   │   │   ├── __init__.py         # 统一异常处理 + 限流中间件
│   │   │   └── input_security.py   # SQL注入/XSS/Prompt注入/PII 检测
│   │   ├── security.py             # JWT 鉴权
│   │   ├── security_user.py        # SQLite 用户管理
│   │   ├── rate_limiter.py         # 限流器
│   │   └── audit.py               # 审计日志
│   │
│   ├── models/                       # LLM 模型封装
│   │   ├── llm.py                  # 主 LLM（DashScope / OpenAI）
│   │   ├── llm_cache.py           # LLM 响应缓存
│   │   ├── embeddings.py          # Embedding 模型
│   │   ├── vision.py             # Vision 模型
│   │   └── mcp_client.py        # MCP 客户端管理器
│   │
│   ├── observability/               # 可观测性
│   │   ├── tracer.py             # 自定义分布式追踪（ContextVar Span）
│   │   ├── otel_tracer.py       # OpenTelemetry OTLP 导出
│   │   ├── metrics.py           # Prometheus 指标
│   │   ├── structured_logging.py # 结构化 JSON 日志
│   │   ├── circuit_breaker.py  # 熔断器模式
│   │   └── cost_tracker.py     # LLM 成本追踪
│   │
│   └── optimization/
│       └── performance.py       # 性能优化工具
│
├── frontend/                         # Vue 3 前端（SSE 流式聊天）
├── tests/                           # 测试套件（pytest）
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/                        # 单元测试
│   ├── integration/                 # 集成测试
│   ├── adversarial/                # 对抗性测试（注入攻击）
│   ├── load/                       # 压力测试
│   └── eval/                       # 评估数据集
│
├── scripts/                         # 工具脚本
│   ├── ingest.py                   # 文档向量入库脚本
│   ├── rag_evaluation.py           # RAG 全链路评估
│   ├── run_rag_benchmark.py        # RAG 基准测试
│   ├── test_rag_retrieval.py      # 检索链路测试
│   ├── test_rag_no_llm.py        # 无 LLM 链路测试
│   ├── test_qa_single.py         # 单条 QA 测试
│   ├── test_qa_accuracy.py       # QA 准确率测试
│   ├── eval_dataset.py           # 评估数据集
│   └── reports/                   # 评估报告输出
│
├── docs/                            # 文档
│   ├── implementation_report.md    # 实施报告
│   ├── open_source_reference_report.md
│   ├── verification_test_report.md
│   ├── PERFORMANCE_OPTIMIZATION.md
│   └── architecture/              # 架构设计
│       ├── VERSION_MANAGEMENT.md
│       ├── REFUSAL_STRATEGY.md
│       ├── INGESTION_PIPELINE.md
│       ├── OBSERVABILITY.md
│       └── ACL_DESIGN.md
│
├── grafana/                         # Grafana 仪表盘
├── docker-compose.yml               # Docker 全家桶
└── Dockerfile
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/register` | 用户注册 |
| POST | `/api/v1/auth/login` | 用户登录，返回 JWT |
| POST | `/api/v1/chat` | 同步聊天（非流式） |
| POST | `/api/v1/chat/stream` | 流式聊天（SSE） |
| GET | `/api/v1/chat/history/{session_id}` | 获取聊天历史 |
| GET | `/api/v1/chat/sessions` | 列出用户所有会话 |
| DELETE | `/api/v1/chat/sessions/{session_id}` | 删除会话 |
| PATCH | `/api/v1/chat/sessions/{session_id}` | 更新会话标题 |
| POST | `/api/v1/knowledge/ingest` | 上传文档入库 |
| GET | `/api/v1/knowledge/search` | 知识库检索 |
| POST | `/api/v1/a2a/send` | A2A 协议消息 |
| GET | `/api/v1/a2a/skills` | 列出 Agent 支持的技能 |
| GET | `/.well-known/agent.json` | Agent Card（A2A 协议） |
| GET | `/health` | 健康检查（含组件状态） |
| GET | `/health/live` | K8s Liveness Probe |
| GET | `/health/ready` | K8s Readiness Probe |
| GET | `/metrics` | Prometheus 指标端点 |
