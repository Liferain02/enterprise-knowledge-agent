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

## 🚀 快速开始（Linux + Git）

下面以 Linux 环境为例，说明从 `git clone` 到可以访问系统的完整步骤。

### 1. 环境准备

- **Conda**: 建议使用 Miniconda / Anaconda 管理 Python 环境
- **Python**: 建议 Python 3.10+（通过 Conda 创建）
- **Node.js & npm**: 建议 Node 18+/20+（如 `node --version`、`npm --version`）
- **Git**: 已安装 `git`

### 2. 克隆项目

```bash
git clone https://your.git.repo.url/enterprise-knowledge-agent.git
cd enterprise-knowledge-agent
```

将上面的地址替换成你自己的 Git 仓库地址。

### 3. 使用 Conda 创建并激活环境（推荐）

```bash
# 创建名为 enterprise-agent 的 Conda 环境（Python 版本可按需调整）
conda create -n enterprise-agent python=3.11 -y

# 激活环境
conda activate enterprise-agent


> 提示：如果你仍然选择使用 `python -m venv`，`.venv/` 等虚拟环境目录也已经在 `.gitignore` 中忽略。

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 5. 安装 MCP 服务器依赖（npm）

```bash
# 在项目根目录安装 MCP 服务器所需的 Node 依赖
npm install
```

### 6. 配置环境变量

```bash
# 复制模板
cp config/env.template config/.env

# 编辑 .env，填入你的 API Key（至少配置千问或 OpenAI 其中一种）
vim config/.env
```

关键变量示例（在 `config/.env` 中）：

```bash
DASHSCOPE_API_KEY=你的千问APIKey
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus

# 或者使用 OpenAI：
# OPENAI_API_KEY=...
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini

# 对外提供服务（远程机器可访问）
API_HOST=0.0.0.0
API_PORT=8000

# 登录鉴权（单用户）
AUTH_ENABLED=true
ADMIN_USERNAME=你的用户名
ADMIN_PASSWORD=你的密码
JWT_SECRET_KEY=请换成一个长随机字符串
JWT_EXPIRE_MINUTES=720
```

> `.env` 文件已经在 `.gitignore` 中配置为忽略，**不要**把真实 Key 提交到 Git。

### 6. 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

### 7. 启动后端服务（Linux）

```bash
# 确保在项目根目录，虚拟环境已激活
python main.py
```

- 默认监听地址：`http://0.0.0.0:8000`
- 如需其他机器访问，请确保 `API_HOST=0.0.0.0`，并在服务器防火墙/安全组放通端口 `8000`（后端）和 `3000`（前端）。
- 首次启动会自动检查并嵌入 `data/knowledge/` 中的文档到 `chroma_db/`

### 8. 启动前端（新终端窗口）

```bash
cd frontend
npm run dev
```

默认访问地址：`http://localhost:3000`

### 9. 访问与调试

- **后端**: `http://localhost:8000`
- **前端**: `http://localhost:3000`
- **API 文档 (Swagger)**: `http://localhost:8000/docs`

## 📁 项目结构

```
enterprise-knowledge-agent/
├── main.py                      # 主入口 (FastAPI + 启动时自动嵌入知识库)
├── requirements.txt             # Python 依赖
├── package.json                  # MCP 服务器依赖 (npm install)
│
├── config/                       # 配置
│   ├── settings.py               # Pydantic Settings 配置
│   ├── mcp_servers.json         # MCP 服务器配置
│   └── .env                     # 环境变量
│
├── core/                        # 核心模块
│   ├── llm.py                   # LLM 初始化 (千问/OpenAI)
│   ├── embeddings.py            # 向量嵌入模型
│   └── mcp_client.py             # MCP 客户端
│
├── rag/                         # RAG 模块
│   ├── vectorstore.py           # Chroma 向量存储
│   ├── retriever.py             # 检索器
│   ├── document_loader.py       # 文档加载 (PDF/Word/MD/TXT)
│   └── pipeline.py              # RAG 管道
│
├── tools/                       # 工具模块 (Operation Agent 用)
│   ├── __init__.py              # 基础工具 (知识搜索/计算器/时间)
│   └── mcp_adapter.py            # MCP 工具适配器
│
├── agents/                      # Agent 模块
│   ├── graph.py                 # LangGraph 工作流定义
│   ├── prompts.py               # Agent 系统提示词
│   ├── nodes/                   # Agent 节点
│   │   ├── supervisor.py       # 路由决策节点
│   │   ├── knowledge.py         # 知识库问答节点
│   │   ├── operation.py         # 操作执行节点
│   │   ├── general.py           # 通用问答节点
│   │   └── utils.py             # 节点工具函数
│   └── skills/                  # Agent Skills (声明式技能)
│       ├── skill_loader.py      # 技能加载器
│       ├── knowledge/           # 知识检索技能
│       │   ├── Skill.md         # 技能定义
│       │   └── scripts/tools.py  # 技能工具
│       ├── calculator/          # 计算器技能
│       ├── datetime/            # 日期时间技能
│       └── file_operation/      # 文件操作技能
│
├── api/                         # API 模块 (Controller + Service + DAO 分层)
│   ├── controllers/             # Controller 层 (接收请求)
│   │   ├── chat_controller.py   # 聊天 API
│   │   └── knowledge_controller.py  # 知识管理 API
│   ├── services/                 # Service 层 (业务逻辑)
│   │   ├── chat_service.py
│   │   ├── session_service.py
│   │   └── knowledge_service.py
│   ├── dao/                     # DAO 层 (数据访问)
│   │   └── session_dao.py
│   └── dependencies.py          # 依赖注入
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
    │   ├── App.vue              # 主组件
    │   ├── main.ts              # 入口
    │   └── style.css            # 样式
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

1. 推荐在 **Linux** 环境下运行（当前说明默认以 Linux/Bash 为准）。
2. 需要在 `config/.env` 中配置阿里千问或 OpenAI 的 API Key。
3. 首次启动会自动嵌入知识库，向量数据保存在 `./chroma_db` 目录下（已在 `.gitignore` 中忽略）。

## 🚢 部署指南

### 直接部署

```bash
# 1. 进入项目目录
cd /share/home/lifr/workspace/code/enterprise-knowledge-agent

# 2. 激活 Python 环境
conda activate enterprise-agent

# 3. 安装依赖
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 4. 配置环境变量
cp config/env.template config/.env
vim config/.env   # 填入 DASHSCOPE_API_KEY / ADMIN_USERNAME / ADMIN_PASSWORD / JWT_SECRET_KEY

# 5. 启动后端（后台运行）
nohup python main.py > server.log 2>&1 &
echo $! > backend.pid

# 6. 启动前端
cd frontend && nohup npm run dev > frontend.log 2>&1 &

# 7. 访问
# 前端: http://your-server-ip:3000
# 后端: http://your-server-ip:8000
```

### 部署架构

```
用户浏览器 → Frontend (Vite:3000) → Backend (uvicorn:8000) → AI API
```

### 生产环境注意事项

1. **API Key 安全**: 使用环境变量管理，不要提交到代码仓库
2. **防火墙**: 开放 3000（前端）和 8000（后端）端口
3. **数据持久化**: `chroma_db` 和 `data` 目录建议定期备份

## 📄 License

MIT
