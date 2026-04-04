# Enterprise Knowledge Agent

企业级 RAG 智能问答系统，支持多 Agent 协作、长期记忆、流式输出、多用户认证。

## 核心架构

### Multi-Agent 工作流（LangGraph）

```
用户输入 → maybe_summarize → retrieve_mem0_memories → Planner
    ├─ 简单任务 → Supervisor → Worker Agents → save_to_mem0 → END
    └─ 复杂任务 → Execute Plan → 各子步骤 → save_to_mem0 → END
```

**检索管线（重构成两阶段）：**

```
Supervisor → retrieval_agent_node → generation_agent_node → save_to_mem0 → END
```

- `retrieval_agent_node`：Hybrid 检索 → Rerank → CRAG LLM 评估 → 查询改写/分解（条件触发），无 ReAct 循环
- `generation_agent_node`：基于评估后的文档独立生成，强制引用来源
- 冲突检测：`conflict_detector` 对多文档矛盾信息告警

### Tier 1 开源组件

| 组件 | 用途 | 来源 |
|------|------|------|
| LangChain / LangGraph | Agent 编排框架 | `langchain`, `langgraph` |
| bm25s | BM25 关键词检索 | `bm25s` |
| FlashRank | 轻量级本地重排序 | `flashrank` |
| Arize Phoenix | RAG 可观测性 | `arize-phoenix` |
| OpenTelemetry | 分布式追踪 | `opentelemetry-*` |
| Redis | 评估缓存持久化 | `redis[hiredis]` |
| Unstructured | 多格式文档解析 | `unstructured` |
| LlamaIndex | 向量检索底层 | `llama-index` |

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

## 快速开始

```bash
# 1. 安装依赖
conda activate agent-demo
pip install -r requirements.txt

# 2. 配置环境变量
cp config/.env.example config/.env
# 编辑 config/.env 填入 LLM API Key 等

# 3. 启动服务
python main.py
# 或 Docker Compose（包含 Redis + Prometheus + OTEL Collector）
docker-compose up -d
```

## 项目结构

```
.
├── main.py                          # FastAPI 入口
├── config/
│   ├── settings.py                  # Pydantic Settings 配置
│   └── .env                         # 环境变量（需创建）
├── src/
│   ├── agent/                       # LangGraph 多 Agent 编排
│   │   ├── graph.py                 # StateGraph 定义 + 节点注册
│   │   ├── agents/
│   │   │   ├── supervisor.py        # 路由决策
│   │   │   ├── knowledge.py         # retrieval_agent + generation_agent 节点
│   │   │   ├── operation.py        # 操作类 Agent
│   │   │   ├── general.py          # 通用 Agent
│   │   │   ├── planner.py          # 任务复杂度判断
│   │   │   └── parallel_executor.py # 并行执行（Send 原语）
│   │   ├── memory/                  # Mem0 语义记忆
│   │   └── skills/                  # Skill Loader 动态 Agent
│   ├── rag/                         # RAG 管线
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py # BM25 + 向量混合检索 + RRF
│   │   │   ├── reranker.py         # FlashRank / Cohere / BGE 重排序
│   │   │   └── query_expander.py   # 复杂查询分解 + HyDE
│   │   ├── evaluation/
│   │   │   ├── retrieval_grader.py  # Corrective RAG 评估 + GradeResult
│   │   │   ├── conflict_detector.py # 多文档冲突检测
│   │   │   └── grade_cache.py      # Redis 评估缓存
│   │   └── processing/
│   │       └── unstructured_loader.py # 多格式文档解析
│   ├── api/                         # FastAPI 接口层
│   │   ├── controllers/             # 路由聚合
│   │   ├── routes/                 # API 路由定义
│   │   ├── services/               # 业务逻辑（chat_service, session_service）
│   │   ├── schemas/               # Pydantic 请求/响应模型
│   │   ├── security.py            # JWT 鉴权
│   │   └── security_user.py       # SQLite 用户管理
│   ├── observability/              # 可观测性
│   │   ├── tracer.py              # 自定义分布式追踪（ContextVar Span）
│   │   └── otel_tracer.py        # OpenTelemetry OTLP 导出
│   └── models/                     # LLM 模型封装
├── frontend/                        # Vue 3 前端（SSE 流式聊天）
├── tests/                          # 测试套件
├── docs/                           # 文档
│   └── architecture/              # 架构设计
├── docker-compose.yml              # Docker 全家桶（Redis/Prometheus/OTEL）
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
| POST | `/api/v1/knowledge/ingest` | 上传文档入库 |
| POST | `/api/v1/a2a/message` | A2A 协议消息 |
