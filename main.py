"""
企业知识库智能助手 - 主入口
"""
import asyncio
import sys

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config.settings import get_settings
from api.controllers import chat_router, knowledge_router
from core.mcp_client import init_mcp, close_mcp
from rag.vectorstore import get_vectorstore_manager

# 初始化设置
settings = get_settings()


def check_and_ingest_knowledge_base():
    """启动时检查并嵌入知识库"""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时检查并嵌入知识库
    print("=" * 50)
    print("正在检查知识库状态...")
    check_and_ingest_knowledge_base()
    print("=" * 50)
    
    # 启动时初始化 MCP 服务器
    print("=" * 50)
    print("正在初始化 MCP 服务器...")
    
    try:
        await init_mcp()
    except asyncio.CancelledError:
        print("MCP 初始化被取消")
    except Exception as e:
        print(f"MCP 初始化出错（不影响运行）: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 50)
    
    yield
    
    # 注意：不主动断开 MCP 连接，让进程自然退出
    # 避免 uvicorn 关闭时因任务取消导致的 asyncio cancel scope 冲突
    print("应用关闭，MCP 连接将在进程退出时自动关闭")


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
    # Linux 下禁用 reload 模式，避免 MCP stdio 连接在重载时出现 asyncio 取消作用域冲突
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False
    )


