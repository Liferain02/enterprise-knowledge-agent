# 🤖 企业知识库智能助手 (Enterprise Knowledge Agent)

一个基于 LangChain 1.0+、LangGraph、ReAct 和 MCP 的企业级 RAG Agent 系统。

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│                    (http://localhost:8000)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Supervisor Agent                           │
│  (路由决策: knowledge_agent / operation_agent / general_agent) │
└─────────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Knowledge Agent │  │ Operation Agent │  │  General Agent  │
│   (知识检索)     │  │  (工具执行)      │  │   (通用问答)     │
│   + RAG         │  │  + MCP Tools    │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
           │                    │
           ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│  Chroma DB      │  │  MCP Servers   │
│  (向量存储)      │  │ (文件系统等)    │
└─────────────────┘  └─────────────────┘
```

## 🛠️ 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| LangChain | ≥1.0 | LLM框架 |
| LangGraph | ≥1.0 | 工作流编排 |
| LangGraph Prebuilt | ≥1.0 | 预构建Agent |
| ChromaDB | ≥0.4 | 向量数据库 |
| FastAPI | ≥0.100 | REST API |
| Vue 3 | ≥3.4 | 前端框架 |
| MCP | ≥1.0 | 工具协议 |

## 📦 功能特性

1. **多Agent路由** - Supervisor智能分发任务到专业Agent
2. **知识库问答** - 基于RAG的企业知识库检索
3. **Agent Skills** - 声明式技能定义 (Skill.md)
4. **MCP工具集成** - 文件系统、搜索等外部工具
5. **流式响应** - 支持Server-Sent Events
6. **Vue 3前端** - 现代化聊天界面

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2. 配置环境变量

```bash
# 编辑 config/.env 文件，填入你的 API Key
```

### 3. 启动服务

```bash
# 启动后端 (会自动嵌入知识库)
python main.py

# 启动前端 (新终端)
cd frontend && npm run dev
```

### 4. 访问

- 后端: http://localhost:8000
- 前端: http://localhost:3000
- API文档: http://localhost:8000/docs

## 📁 项目结构

```
enterprise-knowledge-agent/
├── main.py                      # 主入口 (FastAPI + 启动时自动嵌入知识库)
├── requirements.txt             # Python 依赖
├── package.json                 # MCP 服务器依赖
│
├── config/                      # 配置
│   ├── settings.py              # Pydantic Settings 配置
│   ├── mcp_servers.json         # MCP 服务器配置
│   └── .env                     # 环境变量
│
├── core/                        # 核心模块
│   ├── llm.py                   # LLM 初始化 (千问/OpenAI)
│   ├── embeddings.py            # 向量嵌入模型
│   └── mcp_client.py            # MCP 客户端
│
├── rag/                         # RAG 模块
│   ├── vectorstore.py           # Chroma 向量存储
│   ├── retriever.py             # 检索器
│   ├── document_loader.py       # 文档加载 (PDF/Word/MD/TXT)
│   └── pipeline.py              # RAG 管道
│
├── tools/                       # 工具模块 (Operation Agent 用)
│   ├── __init__.py              # 基础工具 (知识搜索/计算器/时间)
│   └── mcp_adapter.py           # MCP 工具适配器
│
├── agents/                      # Agent 模块
│   ├── graph.py                 # LangGraph 工作流定义
│   ├── prompts.py               # Agent 系统提示词
│   ├── nodes/                   # Agent 节点
│   │   ├── supervisor.py        # 路由决策节点
│   │   ├── knowledge.py          # 知识库问答节点
│   │   ├── operation.py         # 操作执行节点
│   │   ├── general.py            # 通用问答节点
│   │   └── utils.py              # 节点工具函数
│   └── skills/                  # Agent Skills (声明式技能)
│       ├── skill_loader.py       # 技能加载器
│       ├── knowledge/            # 知识检索技能
│       │   ├── Skill.md          # 技能定义
│       │   └── scripts/tools.py  # 技能工具
│       ├── calculator/           # 计算器技能
│       ├── datetime/             # 日期时间技能
│       └── file_operation/      # 文件操作技能
│
├── api/                         # API 模块
│   └── routes/
│       ├── chat.py               # 聊天 API
│       └── knowledge.py          # 知识管理 API
│
├── scripts/                     # 脚本
│   └── ingest.py                # 知识库嵌入脚本
│
├── data/                        # 数据目录
│   └── knowledge/               # 知识库文档 (MD/PDF/DOCX)
│
├── chroma_db/                   # Chroma 向量数据库 (自动生成)
│
└── frontend/                    # Vue 3 前端
    ├── src/
    │   ├── App.vue               # 主组件
    │   ├── main.ts               # 入口
    │   └── style.css             # 样式
    ├── package.json             # 前端依赖
    └── vite.config.ts           # Vite 配置
```

## 🔧 使用说明

### 知识库管理

```bash
# 重新嵌入知识库 (清空后重新导入)
python scripts/ingest.py --reset

# 查看知识库状态
# 启动后会自动显示知识库文档数量
```

### 添加新知识

1. 将文档放入 `data/knowledge/` 目录
2. 支持格式: `.md`, `.txt`, `.pdf`, `.docx`
3. 重启服务会自动更新知识库

### 自定义 Agent Skill

1. 在 `agents/skills/` 下创建新目录
2. 添加 `Skill.md` 定义技能
3. 在 `scripts/tools.py` 实现工具函数

## 📡 API 接口

### 聊天

```bash
POST /api/v1/chat
{
    "message": "公司的年假政策是什么？",
    "session_id": "可选的会话ID"
}
```

### 健康检查

```bash
GET /health
```

## 🔄 工作流程

```
用户请求 → Supervisor 路由 → [Knowledge/Operation/General] Agent → 返回结果
                              │
                              ▼
                    ┌─────────────────┐
                    │  ReAct Loop     │
                    │ (推理 → 工具 →  │
                    │   执行 → 观察)  │
                    └─────────────────┘
```

## 📝 注意

1. 需要阿里千问 API Key (config/.env)
2. 首次启动会自动嵌入知识库
3. Chroma 数据库保存在 `./chroma_db`

## 📄 License

MIT
