# 企业知识库智能助手 (Enterprise Knowledge Agent)

一个基于 LangChain、LangGraph、ReAct 和 MCP 的企业级 RAG Agent 系统，支持多轮对话、长效记忆、会话持久化。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│                    (http://localhost:8000)                     │
└─────────────────────────────────────────────────────────────────┘
                              │ 每条消息
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  maybe_summarize（语义总结记忆）                  │
│  消息数 > 阈值时：LLM 压缩旧消息为摘要，删除旧消息，保留近 6 条   │
│  消息数 ≤ 阈值时：透传（零开销）                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           retrieve_mem0_memories（语义记忆检索）                  │
│  检索用户相关记忆：当前会话 + 跨会话记忆                          │
│  格式化后注入 Agent 上下文                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Planner（任务规划 + 轻量化快速路径）             │
│  规则预判（< 1ms，无 LLM）：简单 → 直接跳过，复杂/不确定 → LLM  │
│  LLM 判断：简单任务 → Supervisor；复杂任务 → Execute Plan        │
└─────────────────────────────────────────────────────────────────┘
           │ 简单                              │ 复杂
           ▼                                   ▼
┌─────────────────────────┐       ┌────────────────────────────┐
│      Supervisor         │       │       Execute Plan         │
│   (路由到专业 Agent)     │       │  (逐步执行各子任务，汇总)   │
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
     │
     ▼
┌──────────────────────────┐     ┌──────────────────────────────┐
│  Multimodal Pipeline     │     │      Vision LLM (qwen-vl)    │
│  (文档图片理解入库)       │────▶│  图片 → bytes → 理解 → 入库 │
└──────────────────────────┘     └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         双层记忆持久化                            │
├───────────────────────────────────────────────────────────────────┤
│  SQLite Checkpointer                                              │
│  ├── sessions.db              → 会话 / 消息历史                   │
│  └── langgraph_checkpoints.db → Agent 推理状态 + 对话摘要       │
├───────────────────────────────────────────────────────────────────┤
│  Mem0 语义记忆（Chroma）                                         │
│  └── mem0_chroma/              → 用户级语义记忆，跨会话共享      │
└───────────────────────────────────────────────────────────────────┘
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
| Mem0 | ≥1.0 | 语义记忆层 |
| FastAPI | ≥0.115 | REST API + SSE |
| Vue 3 | ≥3.4 | 前端框架 |
| MCP | ≥1.6 | 工具协议 |
| PyMuPDF | ≥1.24 | PDF 文本/图片/表格提取 |
| Pillow | ≥10.0 | 图片处理 + 尺寸检查 |
| python-docx | ≥1.0 | Word 文档解析 |
| openpyxl | ≥3.1 | Excel 表格提取 |

## 功能特性

1. **多 Agent 路由** — Planner + Supervisor 智能分发任务到专业 Agent
2. **任务规划与拆解** — Planner 识别复杂任务，拆解为多步骤顺序执行
3. **轻量化 Planner** — 规则预判（< 1ms）跳过简单任务的 LLM 调用，降低约 60% 的 Planner 开销
4. **Corrective RAG（检索自我纠错）** — 检索后 LLM 评估相关性，低质量时自动 rewrite 查询重试，最多重试 2 次
5. **Rerank 评估前置** — 评估前先 Rerank 精排候选文档，LLM 评估量从 15 篇减至 5 篇，开销减少 2/3
6. **Query Expansion 前置** — 复杂查询（对比/列举/多实体）在 CRAG 评估前主动分解，多查询并行检索后 RRF 合并
7. **Planner 与检索策略统一** — Planner 复杂度判断透传给下游 Knowledge Agent，避免 Planner 判简单但检索仍触发 Expansion 的不一致
8. **混合检索** — BM25 + 向量分数级融合，同一文档两路信号同时生效（不同于旧版硬去重二选一）
9. **语义总结记忆** — 对话过长时自动压缩旧消息为滚动摘要，防止 Context Window 溢出
10. **会话状态持久化** — LangGraph 推理状态（含摘要）写入 SQLite，服务重启后恢复
11. **长期记忆（Mem0）** — 用户级语义记忆存储，跨会话记忆共享，语义检索增强
12. **知识库问答** — 基于 RAG 的企业知识库检索（PDF / Word / Markdown / TXT）
13. **文档图片 Vision LLM 理解** — PDF/DOCX 中的图片自动通过 qwen-vl-plus 理解，内容入库向量库
14. **Reranker 重排序** — 基于 LLM 的检索结果重排序
15. **对话图片理解** — 聊天时上传图片，Vision LLM 理解后传给 Agent
16. **Agent Skills** — 声明式技能定义（Skill.md），支持热加载
17. **MCP 工具集成** — 文件系统、外部搜索等工具通过 MCP 协议接入
18. **流式响应** — Server-Sent Events 实时输出
19. **JWT 鉴权** — 单用户登录保护

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

# Vision LLM（对话图片理解）
VISION_MODEL=qwen-vl-plus
VISION_ENABLED=true

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

# Mem0 长期记忆（可选开启）
MEM0_ENABLED=true
MEM0_MAX_CONTEXT_CHARS=500

# Vision LLM 入库（文档图片理解，默认开启）
VISION_INGESTION_ENABLED=true
VISION_INGESTION_MODEL=qwen-vl-plus
VISION_INGESTION_MAX_IMAGES_PER_DOC=20   # 单文档最多处理图片数
VISION_INGESTION_SKIP_SMALL=64           # 小于此像素的图片跳过（图标/水印）
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
│   │   ├── vision.py             # Vision LLM（qwen-vl / GPT-4o）图片理解
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
│   │       ├── chunker.py         # 文档分块
│   │       └── multimodal.py      # 多模态处理（表格提取 / Vision LLM 图片理解）
│   │
│   ├── agent/                     # Agent 模块
│   │   ├── graph.py               # LangGraph 图（含 maybe_summarize 节点）
│   │   ├── checkpointer.py        # Checkpointer 工厂（MemorySaver / AsyncSqliteSaver）
│   │   ├── prompts.py             # 系统提示词
│   │   ├── memory/                # 记忆模块
│   │   │   ├── __init__.py
│   │   │   └── mem0_manager.py    # Mem0 语义记忆管理器
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
│       │   ├── vision_controller.py  # 对话图片理解 API
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

### 知识库入库流程（scripts/ingest.py）

```
知识库目录 (data/knowledge/)
  │
  ▼
[1/6] 加载文档 ──────── .md .txt .pdf .docx .xlsx
  │                              （PNG 等不支持格式跳过）
  ▼
[2/6] Vision LLM 理解文档图片（可选，默认启用）
  │  PDF/DOCX ── 提取图片 bytes
  │    │
  │    ├── xref 去重（同图跨页不重复调用）
  │    ├── 跳过 < 64px 的小图（图标/水印）
  │    ├── 单文档最多 20 张（防止超长调用）
  │    └── qwen-vl-plus 理解 → 图片描述文字
  │
[3/6] 分割文档（recursive / markdown / semantic / hybrid）
  │
[4/6] 广播图片描述到同一文档的所有 chunk
  │  （第一个 chunk 含完整描述，其他 chunk 追加引用）
  ▼
[5/6] 向量化 → ChromaDB
  │
[6/6] 验证结果
```

### 文档分块策略

项目提供 4 种分块策略，通过 `chunking_strategy` 配置或 `scripts/ingest.py --strategy` 指定：

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `recursive`（默认） | Token-based 递归分割：按分隔符层级逐级拆分，最终以 token 数控制块大小 | 通用场景，稳定可靠 |
| `semantic` | 基于 Embedding 变化率识别语义边界 + token 硬切兜底 | 需要语义连贯性的长文 |
| `hybrid` | 粗分割（token 递归）+ 语义边界精调 + token 兜底 | 兼顾稳定性和语义质量 |
| `markdown` | 按 Markdown 标题层级分割（字符数兜底） | 结构清晰的文档 |

**Token-based 核心机制**：分块大小以 token 数为控制目标（推荐 500），而非字符数。在分割器的最终兜底（无可用分隔符时），采用句子粒度分割，避免在句子中间截断。

**语义 Overlap**：块之间保留前后句作为重叠内容，提升跨块上下文的连贯性。

**Title + Content 拼接**：每个 chunk 开头自动拼接父级 Markdown 标题，帮助 LLM 感知章节上下文。

关键配置（`config/settings.py` / `config/.env`）：

```bash
CHUNK_TOKEN_SIZE=500        # 分块目标 token 数（推荐 300-800）
CHUNK_TOKEN_OVERLAP=100     # overlap token 数
CHUNK_SEMANTIC_OVERLAP=true # 是否启用语义 overlap
CHUNK_CONCAT_TITLE=true     # 是否拼接父级标题
SEMANTIC_THRESHOLD=0.3      # 语义分块敏感度（越高越敏感）
```

### Agent 问答流程

```
用户消息（文字 / 带图片）
  │
  ▼
maybe_summarize ──── 消息数 ≤ 阈值 ──→ 透传（无开销）
  │ 消息数 > 阈值
  │ LLM 压缩旧消息 → 更新摘要 → 删除旧消息
  ▼
retrieve_mem0_memories
  │ 检索当前会话记忆 + 跨会话记忆
  │ 格式化后注入上下文
  ▼
[可选] Vision LLM 理解用户上传的图片
  │ 图片 bytes → base64 → qwen-vl → 描述文字
  ▼
Planner
  ├── 规则预判（<1ms）→ simple ─────────────────────────────┐
  └── LLM 精确判断 → complex ─→ Execute Plan（多步骤执行）     │
                              │                                  │
                              └── 简单任务 ─→ Supervisor ──────┘
                                              │
                                              ▼
                                     ┌────────┴────────┐
                                     │ Knowledge Agent  │
                                     │ Operation Agent │
                                     │ General Agent   │
                                     └────────┬────────┘
                                              │
                                              ▼
                                    CorrectiveRAGPipeline
                                    ┌──────────────────────────┐
                                    │  阶段0: Query Expansion   │
                                    │  复杂查询（对比/列举/多实体）│
                                    │  → 分解子查询 → 并行检索    │
                                    │  → RRF 合并 → 评估 → 返回  │
                                    │  （若已由 Planner 拆解，     │
                                    │   此阶段跳过）             │
                                    └────────────┬─────────────┘
                                                 │
                                    ┌────────────▼─────────────┐
                                    │  阶段1: CRAG 主循环        │
                                    │  for attempt in 0..max:   │
                                    │    ① 混合检索（向量+BM25）  │
                                    │       candidate_k = k×3    │
                                    │    ② Rerank 精排 → top5   │
                                    │       （评估量 -67%）      │
                                    │    ③ LLM 评估（H/M/L）     │
                                    │    ④ 决策：                │
                                    │       HIGH → 返回          │
                                    │       MEDIUM → 返回        │
                                    │       LOW → rewrite → 重试 │
                                    └────────────┬─────────────┘
                                                 │
                                    ┌────────────▼─────────────┐
                                    │  阶段2: 兜底（rewrite全失败）│
                                    │  → QueryExpander 分解兜底  │
                                    │  → 返回次优结果            │
                                    └────────────────────────────┘
                                              │
                                              ▼
                                         save_to_mem0
                                       （保存对话到记忆）
                                              │
                                              ▼
                                             END
```

### CorrectiveRAGPipeline 详解

**CRAG（Corrective Retrieval-Augmented Generation）是什么**：CRAG 是一种检索后处理机制，在向量检索完成后、交给 LLM 生成答案之前，增加一个"检索质量评估 + 自我纠错"的环节。核心思想是——检索结果不一定完美，如果相关性低，就主动改写查询重新检索，而不是把劣质结果喂给 LLM 导致误导性回答。

CRAG 在本项目中的定位：Knowledge Agent 在执行检索后，先评估检索结果的相关性，若质量过低则 rewrite 查询重试，最多重试 2 次；重试仍失败则触发 QueryExpander 兜底分解。相比"直接返回检索结果"，CRAG 能显著提升低相关或歧义查询的答案质量。

完整流程分为三个阶段：

**阶段 0：Query Expansion 前置（复杂查询主动分解）**

适用于对比类（"年假和病假的区别"）、列举类（"有哪些福利"）、多实体类查询。主动将查询分解为多个子查询，并行检索后通过 RRF（Reciprocal Rank Fusion）合并结果，再经 LLM 评估后返回。注意：若任务已由 Planner 拆解为简单子查询，此阶段自动跳过。

**阶段 1：CRAG 主循环（检索 → 评估 → 决策）**

最多重试 `max_retries` 次（默认 2）。每次循环：① 混合检索候选 k×3 篇文档；② Rerank 精排至 top5（减少 67% 的 LLM 评估量）；③ LLM-as-Judge 评估每篇文档的相关性；④ 根据评估决策：HIGH 直接返回，MEDIUM 返回 HIGH+MEDIUM，LOW 则 rewrite 查询重试。

**阶段 2：兜底机制（两次 rewrite 均失败后）**

作为最后保障，再次调用 QueryExpander 进行更彻底的分解和检索，避免直接返回空结果。

### 混合检索详解

向量检索与 BM25 词项匹配互补：向量检索依赖 embedding 语义理解，BM25 擅长关键词精确命中。

**标准分数级融合（不同于旧版硬去重）**：向量和 BM25 分别检索，各自独立归一化到 [0,1]，然后按权重融合：`score = (1-α)×vec_norm + α×bm25_norm`。关键改进：同一文档在两个检索器中的信号同时生效，而不是后检索的 BM25 丢弃前路已命中的文档。

融合效果示例：

| 文档 | 向量分（归一化） | BM25分（归一化） | 融合分（α=0.5） |
|------|--------------|--------------|--------------|
| Doc_A | 0.0（语义模糊） | 1.0（关键词命中） | **0.50** |
| Doc_B | 1.0（语义强） | 0.0（无关键词） | 0.50 |
| Doc_C | 0.5 | 0.8 | **0.65**（最高） |

Doc_C 因为两路都有信号，融合后反而排到最前——这才是混合检索应有的效果。

## 数据库说明

所有数据库统一位于 `chroma_db/` 目录：

| 文件 | 用途 |
|------|------|
| `chroma.sqlite3` | 知识库向量存储 |
| `sessions.db` | 会话列表 + 聊天消息历史 |
| `langgraph_checkpoints.db` | LangGraph 推理状态（含对话摘要）持久化 |
| `mem0_chroma/` | Mem0 语义记忆向量存储 |

## 语义总结记忆说明

当一个 session 的消息数超过 `SUMMARY_THRESHOLD`（默认 20）时，系统自动：

1. 将旧消息（保留最近 `SUMMARY_KEEP_RECENT` 条之外的所有消息）送入 LLM 生成摘要
2. 摘要滚动累积：新摘要 = 旧摘要 + 本批旧消息的总结
3. 从 LangGraph state 中删除旧消息，只保留摘要 + 最近几条原始消息
4. 摘要写入 `langgraph_checkpoints.db`，服务重启后不丢失
5. 所有 Agent 在生成回答时自动感知摘要上下文


## Mem0 长期记忆说明

系统采用双层记忆机制：

### 短期记忆（会话级）
- LangGraph State 消息列表
- SQLite Checkpointer 持久化
- 滚动摘要压缩

### 长期记忆（用户级）- Mem0
- **自动记忆提取**：LLM 从对话中自动提取关键信息（用户偏好、身份信息、重要事项等）
- **跨会话共享**：用户维度的记忆在不同会话间共享
- **语义检索**：基于向量相似度的精准记忆召回
- **上下文注入**：Agent 执行前自动检索相关记忆并注入上下文

核心工作流程：

```
用户消息
    │
    ▼
retrieve_mem0_memories_node
    │ 1. 检索当前会话记忆（session_id 过滤）
    │ 2. 检索跨会话记忆（用户维度）
    │ 3. 格式化后注入上下文
    ▼
Agent 执行
    │
    ▼
save_to_mem0_node
    │ 1. 保存当前会话记忆
    │ 2. 保存跨会话记忆（用于后续跨会话检索）
    ▼
END
```

## Planner 快速路径说明

`planner_node` 在调用 LLM 之前先做纯规则预判（< 1ms）：

**高优先级 pattern**（优先于长度短路检查）：列举类关键词（"有哪些"、"有什么"）、多实体职责/区别关键词。命中这些 pattern 直接判定为 `complex`，不受 `len ≤ 40` 短路影响——防止 "公司假期都有哪些"（12字）这类列举类查询被误判为 simple。

| 判断 | 条件 | 行为 |
|------|------|------|
| `simple` | 极短消息（≤8字）/ 问候 / 简单信号且无复杂信号 | 跳过 LLM，直接进 Supervisor |
| `complex` | 高优先级 pattern 命中（列举类/多实体）或含对比/顺序/汇总关键词或多问号 | 走 LLM 拆步骤 |
| `uncertain` | 无法确定 | 走 LLM 精确判断 |

约 60% 的典型企业知识问答请求走快速路径，节省 5-10 秒延迟。

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

### 对话图片理解（聊天时附带图片）

```bash
POST /api/v1/vision/understand
Authorization: Bearer <token>
Content-Type: multipart/form-data

image: <file>   # 支持 jpg / png / webp
```

### 知识库搜索

```bash
POST /api/v1/knowledge/search
Authorization: Bearer <token>
{
    "query": "年假政策",
    "top_k": 5,
    "filter": {}           # 可选，元数据过滤条件
    "enable_rerank": true  # 可选，是否启用重排序
}
```

### 触发知识库嵌入

```bash
POST /api/v1/knowledge/ingest
Authorization: Bearer <token>
{
    "collection_name": "enterprise_knowledge",  # 可选，默认值
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "chunking_strategy": "recursive"
}
```

### 健康检查

```bash
GET /health
```

完整接口文档见 `http://localhost:8000/docs`。

## 知识库管理

### 批量嵌入

```bash
# 完整嵌入（启用 Vision LLM 图片理解）
python scripts/ingest.py --reset --vision

# 纯文本嵌入（跳过图片理解，速度更快）
python scripts/ingest.py --reset --no-vision

# 增量添加文档
# 1. 将文档放入 data/knowledge/ 目录（支持 .md .txt .pdf .docx）
# 2. 重启服务或调用 /api/v1/knowledge/ingest 接口
```

### Vision LLM 图片理解

嵌入时自动识别 PDF/DOCX 中的图片，通过 `qwen-vl-plus` 生成文字描述并入库。相关配置见上方「环境变量」中的 `VISION_INGESTION_*` 系列变量。

### 支持的文件格式

| 格式 | 文本提取 | 图片理解 | 表格提取 |
|------|--------|--------|--------|
| .md   | ✅ | — | — |
| .txt  | ✅ | — | — |
| .pdf  | ✅ | ✅ (Vision LLM) | ✅ |
| .docx | ✅ | ✅ (Vision LLM) | ✅ |
| .xlsx | — | — | ✅ |

## 测试

运行测试前需激活 conda 环境并配置代理：

```bash
conda activate agent-demo
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
```

### 测试文件说明

| 文件 | 说明 |
|------|------|
| `tests/test_chunking.py` | 文档分块策略测试：Token 估算、句子分割、各策略分块效果 |
| `tests/test_crags.py` | CRAG 模块测试：评估决策、查询改写、端到端流程 |
| `tests/test_retrieval_eval.py` | 检索评估测试 |
| `tests/test_evaluation.py` | 综合评估测试 |
| `tests/test_integration_e2e.py` | 端到端集成测试 |
| `tests/test_reranker.py` | Reranker 测试 |
| `tests/test_hybrid_reranker.py` | 混合检索 + Rerank 测试 |
| `tests/test_query_expander.py` | 查询扩展测试 |

运行测试：

```bash
# 运行单个测试文件
python tests/test_chunking.py
python tests/test_crags.py

# 运行所有测试（需 pytest）
pytest tests/ -v
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
6. 知识库入库默认启用 Vision LLM 图片理解（`VISION_INGESTION_ENABLED=true`），如需加速可加 `--no-vision` 跳过图片处理。
7. PDF 图片提取依赖 **PyMuPDF** (`pip install pymupdf`)，Word 图片提取依赖 **Pillow** (`pip install pillow`)，首次使用请确认已安装。
