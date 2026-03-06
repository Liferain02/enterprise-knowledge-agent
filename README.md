# 企业知识库智能助手 (Enterprise Knowledge Agent)

一个基于 LangChain、LangGraph、ReAct 和 MCP 的企业级 RAG Agent 系统，支持多轮对话、长效记忆、会话持久化。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│                    (http://localhost:8000)                      │
└─────────────────────────────────────────────────────────────────┘
                              │ 每条消息
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  maybe_summarize（语义总结记忆）                  │
│  消息数 > 阈值时：LLM 压缩旧消息为摘要，删除旧消息，保留近 6 条   │
│  消息数 ≤ 阈值时：透传（零开销）                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Planner（任务规划 + 轻量化快速路径）              │
│  规则预判（< 1ms，无 LLM）：简单 → 直接跳过，复杂/不确定 → LLM   │
│  LLM 判断：简单任务 → Supervisor；复杂任务 → Execute Plan        │
└─────────────────────────────────────────────────────────────────┘
           │ 简单                              │ 复杂
           ▼                                   ▼
┌─────────────────────────┐       ┌────────────────────────────┐
│      Supervisor          │       │       Execute Plan          │
│   (路由到专业 Agent)      │       │  (逐步执行各子任务，汇总)   │
└─────────────────────────┘       └────────────────────────────┘
     │           │          │
     ▼           ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│Knowledge│ │Operation│ │ General │
│ Agent   │ │ Agent   │ │ Agent   │
│(知识检索)│ │(工具执行)│ │(通用问答)│
└─────────┘ └─────────┘ └─────────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────────┐
│Chroma DB│ │ MCP Servers │
│(向量存储)│ │(文件系统等)  │
└─────────┘ └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         SQLite 持久化层                          │
│  sessions.db              → 会话 / 消息历史                      │
│  langgraph_checkpoints.db → Agent 推理状态 + 对话摘要            │
└─────────────────────────────────────────────────────────────────┘
```

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| LangChain | ≥1.0 | LLM 框架 |
| LangGraph | ≥1.0 | 工作流编排 |
| LangGraph Prebuilt | ≥1.0 | 预构建 ReAct Agent |
| langgraph-checkpoint-sqlite | ≥3.0 | Agent 状态 + 摘要持久化 |
| aiosqlite | ≥0.22 | 异步 SQLite 驱动 |
| ChromaDB | ≥1.5 | 向量数据库 |
| FastAPI | ≥0.115 | REST API + SSE |
| Vue 3 | ≥3.4 | 前端框架 |
| MCP | ≥1.6 | 工具协议 |

## 功能特性

1. **多 Agent 路由** — Planner + Supervisor 智能分发任务到专业 Agent
2. **任务规划与拆解** — Planner 识别复杂任务，拆解为多步骤顺序执行
3. **轻量化 Planner** — 规则预判（< 1ms）跳过简单任务的 LLM 调用，降低约 60% 的 Planner 开销
4. **语义总结记忆** — 对话过长时自动压缩旧消息为滚动摘要，防止 Context Window 溢出
5. **会话状态持久化** — LangGraph 推理状态（含摘要）写入 SQLite，服务重启后恢复
6. **知识库问答** — 基于 RAG 的企业知识库检索（PDF / Word / Markdown / TXT）
7. **混合检索** — BM25 + 向量混合检索，提升召回率
8. **Reranker 重排序** — 基于 LLM 的检索结果重排序
9. **Agent Skills** — 声明式技能定义（Skill.md），支持热加载
10. **MCP 工具集成** — 文件系统、外部搜索等工具通过 MCP 协议接入
11. **流式响应** — Server-Sent Events 实时输出
12. **JWT 鉴权** — 单用户登录保护

## 快速开始（Linux）

### 1. 环境准备

- Conda（Miniconda / Anaconda）
- Python 3.10+
- Node.js 18+ / npm
- Git

### 2. 克隆项目

```bash
git clone https://your.git.repo.url/enterprise-knowledge-agent.git
cd enterprise-knowledge-agent
```

### 3. 创建 Conda 环境

```bash
conda create -n agent-demo python=3.11 -y
conda activate agent-demo
```

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 5. 安装 MCP 服务器依赖

```bash
npm install
```

### 6. 配置环境变量

```bash
cp config/env.template config/.env
vim config/.env
```

关键变量：

```bash
# LLM（二选一）
DASHSCOPE_API_KEY=你的千问APIKey
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus

# OPENAI_API_KEY=...
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini

# 服务地址
API_HOST=0.0.0.0
API_PORT=8000

# 鉴权
AUTH_ENABLED=true
ADMIN_USERNAME=你的用户名
ADMIN_PASSWORD=你的密码
JWT_SECRET_KEY=请换成一个长随机字符串
JWT_EXPIRE_MINUTES=720

# 持久化（推荐开启）
USE_SQLITE_CHECKPOINTER=true

# 语义总结记忆（可选调整）
SUMMARY_THRESHOLD=20      # 消息数超过此值时触发摘要
SUMMARY_KEEP_RECENT=6     # 触发摘要后保留的最近消息条数
```

### 7. 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

### 8. 启动后端

```bash
python main.py
```

首次启动会自动将 `data/knowledge/` 中的文档嵌入到向量库。

### 9. 启动前端（新终端）

```bash
cd frontend
npm run dev
```

### 10. 访问

| 地址 | 说明 |
|------|------|
| `http://localhost:3000` | 前端聊天界面 |
| `http://localhost:8000` | 后端 API |
| `http://localhost:8000/docs` | Swagger API 文档 |

## 项目结构

```
enterprise-knowledge-agent/
├── main.py                        # FastAPI 入口（自动嵌入知识库）
├── requirements.txt               # Python 依赖
├── package.json                   # MCP 服务器依赖
│
├── config/
│   ├── settings.py                # Pydantic Settings（含摘要/持久化配置）
│   ├── mcp_servers.json           # MCP 服务器配置
│   └── .env                       # 环境变量（不提交 Git）
│
├── src/
│   ├── models/                    # 模型层
│   │   ├── llm.py                 # LLM 初始化
│   │   ├── embeddings.py          # 向量嵌入模型
│   │   └── mcp_client.py          # MCP 客户端
│   │
│   ├── rag/                       # RAG 模块
│   │   ├── storage/vectorstore.py # Chroma 向量存储
│   │   ├── retrieval/
│   │   │   ├── retriever.py       # 基础检索器
│   │   │   ├── hybrid_retriever.py# 混合检索（BM25 + 向量）
│   │   │   └── reranker.py        # LLM 重排序
│   │   └── processing/
│   │       ├── document_loader.py # 文档加载（PDF/Word/MD/TXT）
│   │       └── chunker.py         # 文档分块
│   │
│   ├── agent/                     # Agent 模块
│   │   ├── graph.py               # LangGraph 图（含 maybe_summarize 节点）
│   │   ├── checkpointer.py        # Checkpointer 工厂（MemorySaver / AsyncSqliteSaver）
│   │   ├── prompts.py             # 系统提示词
│   │   ├── agents/
│   │   │   ├── planner.py         # Planner（规则快速路径 + LLM 任务拆解）
│   │   │   ├── supervisor.py      # Supervisor 路由决策
│   │   │   ├── knowledge.py       # 知识检索 Agent
│   │   │   ├── operation.py       # 操作执行 Agent（工具调用）
│   │   │   ├── general.py         # 通用问答 Agent
│   │   │   └── _utils.py          # 共享工具函数（摘要注入等）
│   │   ├── skills/                # Agent Skills（声明式技能）
│   │   │   ├── skill_loader.py
│   │   │   ├── knowledge/
│   │   │   ├── calculator/
│   │   │   ├── datetime/
│   │   │   └── file_operation/
│   │   └── tools/
│   │       ├── __init__.py
│   │       └── mcp_adapter.py
│   │
│   └── api/                       # API 层（Controller / Service / DAO）
│       ├── controllers/
│       │   ├── chat_controller.py
│       │   ├── knowledge_controller.py
│       │   └── auth_controller.py
│       ├── services/
│       │   ├── chat_service.py
│       │   ├── session_service.py
│       │   └── knowledge_service.py
│       ├── repositories/dao/
│       │   └── session_dao.py
│       ├── schemas/
│       └── security.py
│
├── scripts/
│   └── ingest.py                  # 知识库批量嵌入脚本
│
├── data/knowledge/                # 知识库文档（MD / PDF / DOCX）
├── chroma_db/                     # 数据库目录（自动生成，已 gitignore）
│   ├── chroma.sqlite3             # 向量存储
│   ├── sessions.db                # 会话 / 消息历史
│   └── langgraph_checkpoints.db  # Agent 状态 + 对话摘要持久化
│
└── frontend/                      # Vue 3 前端
    ├── src/
    │   ├── App.vue
    │   ├── main.ts
    │   └── style.css
    ├── package.json
    └── vite.config.ts
```

## 工作流程

```
用户消息
  │
  ▼
maybe_summarize ──── 消息数 ≤ 阈值 ──→ 透传（无开销）
  │ 消息数 > 阈值
  │ LLM 压缩旧消息 → 更新摘要 → 删除旧消息
  ▼
Planner
  ├── 规则预判 → 简单（≈60% 请求跳过 LLM）────────────────┐
  └── LLM 判断 → 简单 ──────────────────────────────────┐  │
               └── 复杂 → Execute Plan（多步执行）→ END  │  │
                                                         ▼  ▼
                                                      Supervisor
                                                    ↙    ↓    ↘
                                             Knowledge Operation General
                                              Agent    Agent    Agent
                                                └────────────────┘
                                                        │
                                                       END
```

## 数据库说明

所有数据库统一位于 `chroma_db/` 目录：

| 文件 | 用途 |
|------|------|
| `chroma.sqlite3` | 知识库向量存储 |
| `sessions.db` | 会话列表 + 聊天消息历史 |
| `langgraph_checkpoints.db` | LangGraph 推理状态（含对话摘要）持久化 |

## 语义总结记忆说明

当一个 session 的消息数超过 `SUMMARY_THRESHOLD`（默认 20）时，系统自动：

1. 将旧消息（保留最近 `SUMMARY_KEEP_RECENT` 条之外的所有消息）送入 LLM 生成摘要
2. 摘要滚动累积：新摘要 = 旧摘要 + 本批旧消息的总结
3. 从 LangGraph state 中删除旧消息，只保留摘要 + 最近几条原始消息
4. 摘要写入 `langgraph_checkpoints.db`，服务重启后不丢失
5. 所有 Agent 在生成回答时自动感知摘要上下文

## Planner 快速路径说明

`planner_node` 在调用 LLM 之前先做纯规则预判（< 1ms）：

| 判断 | 条件 | 行为 |
|------|------|------|
| `simple` | 极短消息 / 问候 / 单一查询（≤40字且无复杂信号）| 跳过 LLM，直接进 Supervisor |
| `complex` | 含对比/多信息/顺序/汇总关键词，或多个问号 | 走 LLM 拆步骤 |
| `uncertain` | 无法确定 | 走 LLM 精确判断 |

约 60% 的典型企业知识问答请求（单一查询、闲聊）走快速路径，节省 5-10 秒延迟。

## API 接口

### 聊天

```bash
POST /api/v1/chat
Authorization: Bearer <token>
{
    "message": "公司的年假政策是什么？",
    "session_id": "可选的会话ID"
}
```

### 流式聊天

```bash
GET /api/v1/chat/stream?message=你好&session_id=xxx
Authorization: Bearer <token>
```

### 健康检查

```bash
GET /health
```

完整接口文档见 `http://localhost:8000/docs`。

## 知识库管理

```bash
# 重新嵌入知识库（清空后重新导入）
python scripts/ingest.py --reset

# 增量添加文档
# 1. 将文档放入 data/knowledge/ 目录（支持 .md .txt .pdf .docx）
# 2. 重启服务或调用 /api/v1/knowledge/ingest 接口
```

## 部署

### 后台运行

```bash
conda activate agent-demo

# 后端
nohup python main.py > server.log 2>&1 &

# 前端
cd frontend && nohup npm run dev > frontend.log 2>&1 &
```

### 访问地址

```
用户浏览器 → Frontend (Vite:3000) → Backend (uvicorn:8000) → AI API
```

## 注意事项

1. 推荐在 **Linux** 环境运行。
2. 需在 `config/.env` 中配置阿里千问或 OpenAI 的 API Key。
3. 首次启动自动嵌入知识库，向量数据保存在 `./chroma_db/`（已 gitignore）。
4. 开启 `USE_SQLITE_CHECKPOINTER=true` 后，Agent 推理状态和对话摘要均持久化到 SQLite，服务重启后对话上下文完整恢复。
5. 若需调试或测试，可设置 `USE_SQLITE_CHECKPOINTER=false` 切换为内存模式（重启后状态清空）。
