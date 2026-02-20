# 🤖 企业知识库智能助手 (Enterprise Knowledge Assistant)

一个基于 LangChain、LangGraph、ReAct 和 MCP 的企业级 RAG Agent 系统。

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│                    (http://localhost:8000)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  规划    │───▶│  记忆    │───▶│  工具    │───▶│  执行    │  │
│  │ Planner  │    │ Memory  │    │ Tools    │    │ Executor │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                    │                   │
│       ▼                                    ▼                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ReAct Loop (推理 + 行动)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   RAG Pipeline  │  │   MCP Servers   │  │   Chroma DB     │
│   (知识检索)     │  │ (文件系统/搜索)  │  │  (向量存储)      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 🛠️ 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| LangChain | ≥1.0 | LLM框架 |
| LangGraph | ≥0.1 | 工作流编排 |
| LangChain MCP | - | MCP工具集成 |
| ChromaDB | ≥0.4 | 向量数据库 |
| FastAPI | ≥0.100 | REST API |
| ReAct | - | 推理框架 |
| pydantic | ≥2.0 | 数据验证 |

## 📦 功能特性

1. **智能问答** - 基于RAG的企业知识库问答
2. **多工具协作** - 文件搜索、数据库查询、API调用
3. **对话记忆** - 短期记忆(会话) + 长期记忆(向量存储)
4. **ReAct推理** - 思考-行动-观察循环
5. **MCP协议** - 标准化的工具协议
6. **流式响应** - 支持Server-Sent Events

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制并编辑配置文件
cp .env.example .env
# 编辑 .env 文件，填入你的API Key
```

### 3. 初始化向量数据库

```bash
python scripts/init_knowledge_base.py
```

### 4. 启动服务

```bash
# 启动FastAPI服务
uvicorn main:app --reload --port 8000
```

### 5. 测试

```bash
# API文档: http://localhost:8000/docs

# 方式1: 使用curl测试
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "公司的年假政策是什么？", "session_id": "test001"}'

# 方式2: 使用Web界面
# 访问 http://localhost:8000
```

## 📁 项目结构

```
enterprise-knowledge-agent/
├── config/
│   ├── __init__.py
│   ├── settings.py          # 配置管理
│   └── prompts.py          # Prompt模板
├── core/
│   ├── __init__.py
│   ├── llm.py              # LLM初始化
│   ├── embeddings.py       # 向量嵌入
│   └── chat_history.py    # 对话历史管理
├── rag/
│   ├── __init__.py
│   ├── vectorstore.py      # Chroma向量存储
│   ├── retriever.py        # 检索器
│   ├── document_loader.py  # 文档加载
│   └── pipeline.py         # RAG管道
├── tools/
│   ├── __init__.py
│   ├── base.py             # 工具基类
│   ├── mcp_tools.py        # MCP工具实现
│   ├── search_tools.py     # 搜索工具
│   └── calculator.py       # 计算工具
├── agent/
│   ├── __init__.py
│   ├── state.py            # Agent状态定义
│   ├── nodes.py            # LangGraph节点
│   ├── graph.py            # LangGraph图定义
│   └── react.py            # ReAct实现
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py         # 聊天API
│   │   └── knowledge.py    # 知识管理API
│   └── dependencies.py     # API依赖
├── data/
│   └── knowledge/          # 知识库文档
├── scripts/
│   ├── init_knowledge_base.py  # 初始化脚本
│   └── seed_data.py            # 种子数据
├── main.py                 # 主入口
├── requirements.txt        # 依赖
├── .env.example           # 环境变量示例
└── README.md
```

## 🔧 API 接口

### 聊天接口

```
POST /api/v1/chat
{
    "message": "问题内容",
    "session_id": "会话ID(可选)",
    "use_rag": true,
    "stream": false
}
```

### 知识管理

```
POST /api/v1/knowledge/add     # 添加文档
GET  /api/v1/knowledge/search # 搜索知识
DELETE /api/v1/knowledge/{id} # 删除文档
```

## 🔄 工作流程

```
用户问题 → 规划(Plan) → 记忆(Recall) → 工具选择(Tools) → 执行(Execute)
                ↑                                              │
                └──────────────── 结果评估 ←───────────────────┘
```

## 📝 注意事项

1. 需要提前准备好 OpenAI API Key 或其他兼容的LLM API
2. 首次运行需要初始化知识库
3. Chroma数据库文件会保存在 `./chroma_db` 目录

## 📄 License

MIT License

