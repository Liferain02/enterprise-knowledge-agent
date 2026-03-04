"""
企业知识库智能助手 - 主入口
"""
import asyncio
import sys
import os

# 设置代理（如果环境变量中没有配置）
if not os.environ.get("http_proxy") and not os.environ.get("HTTP_PROXY"):
    # 可以在这里硬编码代理，或者从配置文件读取
    os.environ["http_proxy"] = "http://127.0.0.1:7897"
    os.environ["https_proxy"] = "http://127.0.0.1:7897"
    pass

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config.settings import get_settings
from api.controllers import chat_router, knowledge_router
from core.mcp_client import mcp_manager as global_mcp_manager
from rag.vectorstore import get_vectorstore_manager

# 初始化设置
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 在 uvicorn 的 event loop 中初始化 MCP"""
    # 在 uvicorn 的 event loop 中初始化 MCP
    print("==================================================")
    print("正在初始化 MCP 服务器...")
    await global_mcp_manager.initialize()
    print(f"MCP 服务器初始化完成，获取 {len(global_mcp_manager.get_tools())} 个工具")
    print("==================================================")

    # 检查知识库
    try:
        vectorstore_manager = get_vectorstore_manager()
        info = vectorstore_manager.get_collection_info()
        doc_count = info.get("count", 0)

        if doc_count == 0:
            print("=" * 50)
            print("⚠️  知识库为空，正在自动嵌入文档...")
            print("=" * 50)

            # 导入并运行嵌入脚本
            from scripts.ingest import ingest_knowledge_base
            ingest_knowledge_base(reset=False)

            print("✅ 知识库嵌入完成！")
        else:
            print(f"✅ 知识库已就绪，当前包含 {doc_count} 个文档块")
    except Exception as e:
        print(f"⚠️  检查知识库时出错: {e}")

    try:
        yield
    finally:
        # 关闭时清理 MCP 连接 - 使用 try-except 忽略 anyio 取消错误
        print("正在关闭 MCP 服务器连接...")
        try:
            await global_mcp_manager.close()
            print("MCP 服务器连接已关闭")
        except Exception as e:
            print(f"关闭 MCP 连接时出错 (可忽略): {e}")

# 创建FastAPI应用
app = FastAPI(
    title="企业知识库智能助手",
    description="基于 LangChain、LangGraph、ReAct 和 MCP 的企业级 RAG Agent 系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(knowledge_router)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "enterprise-knowledge-assistant",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    # 启动 uvicorn 服务器
    # MCP 和知识库的初始化将在 lifespan 中完成
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False
    )


