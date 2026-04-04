"""
企业知识库智能助手 - 主入口

定位：企业内部制度问答与流程检索系统
优先服务 HR / 行政 / IT 支持三大高频场景
核心能力：
- Corrective RAG：检索结果 LLM 评估 + 自我纠错
- 多租户权限隔离（部门/角色/密级）
- 文档版本管理与时效性
- 冲突检测与拒答策略
- 异步入库 Pipeline（启动不阻塞）
- 流式 SSE 输出
- 可观测性（Prometheus metrics + 链路追踪）
"""
import asyncio
import sys
import os
import threading
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import uvicorn

from config import get_settings

# 初始化设置
settings = get_settings()

# 设置代理（如果配置了代理地址）
if settings.http_proxy:
    os.environ["http_proxy"] = settings.http_proxy
if settings.https_proxy:
    os.environ["https_proxy"] = settings.https_proxy

# 清空 LLM 缓存，确保代理配置生效
from src.models import reset_llm
reset_llm()

# 导入路由
from src.api.controllers import (
    chat_router, knowledge_router, auth_router, vision_router, _a2a_router
)

# 导入核心组件
from src.models.mcp_client import mcp_manager as global_mcp_manager
from src.rag import get_vectorstore_manager

# 可观测性
from src.observability.metrics import get_metrics, get_content_type

# 启动日志
logger = logging.getLogger(__name__)


# ==================== 后台 Worker 管理 ====================

_workers: list = []


def _start_ingestion_worker():
    """在独立线程中启动异步入库 Worker"""
    from src.rag.ingestion import IngestionWorker
    worker = IngestionWorker()
    t = threading.Thread(
        target=worker.run,
        args=("lifespan-worker",),
        kwargs={"poll_interval": 3.0},
        daemon=True,
    )
    t.start()
    return worker, t


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("=" * 60)
    print("启动企业知识库智能助手...")
    print("定位：企业内部制度问答与流程检索系统")
    print("场景：HR / 行政 / IT 支持")
    print("=" * 60)

    # ── 1. 初始化 MCP ─────────────────────────────────────────
    print("初始化 MCP 服务器...")
    mcp_timeout = settings.mcp_init_timeout
    await global_mcp_manager.initialize(timeout=mcp_timeout)
    print(f"MCP 初始化完成，获取 {len(global_mcp_manager.get_tools())} 个工具")

    # ── 2. 检查知识库（不阻塞）────────────────────────────────
    try:
        vectorstore_manager = get_vectorstore_manager()
        info = vectorstore_manager.get_collection_info()
        doc_count = info.get("count", 0)
        if doc_count == 0:
            print("⚠️  知识库为空，文档将异步入库")
        else:
            print(f"✅ 知识库就绪，当前包含 {doc_count} 个文档块")
    except Exception as e:
        print(f"⚠️  知识库检查出错: {e}")

    # ── 3. 启动异步入库 Worker（不阻塞启动）─────────────────────
    print("启动异步入库 Worker...")
    worker, t = _start_ingestion_worker()
    _workers.append((worker, t))
    print("✅ 入库 Worker 已启动（后台运行）")

    # ── OpenTelemetry（可选，环境变量 OTEL_ENABLED=true）────────────────
    from src.observability.otel_tracer import init_otel, instrument_fastapi_app
    if await init_otel():
        instrument_fastapi_app(app)

    print("=" * 60)
    print("服务就绪！")
    print(f"API: http://{settings.api_host}:{settings.api_port}")
    print(f"文档: http://{settings.api_host}:{settings.api_port}/docs")
    print("=" * 60)

    yield

    # ── 关闭 ─────────────────────────────────────────────────
    from src.observability.otel_tracer import shutdown_otel
    shutdown_otel()

    print("正在关闭...")
    for w, _ in _workers:
        w.stop()
    print("入库 Worker 已停止")

    try:
        await global_mcp_manager.close()
        print("MCP 连接已关闭")
    except Exception as e:
        print(f"关闭 MCP 连接时出错 (可忽略): {e}")


# 创建FastAPI应用
app = FastAPI(
    title="企业知识库智能助手",
    description=(
        "企业内部制度问答与流程检索系统\n\n"
        "**定位**：服务 HR / 行政 / IT 支持三大高频场景\n\n"
        "**核心能力**：\n"
        "- Corrective RAG：检索结果 LLM 评估 + 自我纠错\n"
        "- 多租户权限隔离（部门/角色/密级）\n"
        "- 文档版本管理与时效性\n"
        "- 冲突检测与拒答策略\n"
        "- 异步入库 Pipeline\n"
        "- 流式 SSE 输出\n"
        "- Prometheus 可观测性"
    ),
    version="1.0.0",
    lifespan=lifespan
)

# 注册统一异常处理器（中间件 + 全局 handlers）
from src.api.middleware import register_exception_handlers
register_exception_handlers(app, debug=settings.debug)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载前端静态文件
import os
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIST), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))


# 注册路由
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(auth_router)
app.include_router(vision_router)
app.include_router(_a2a_router)  # A2A Agent Card 暴露


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "enterprise-knowledge-assistant",
        "version": "1.0.0",
        "定位": "企业内部制度问答与流程检索系统",
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics 端点
    暴露所有业务指标（延迟分布、决策计数、token 消耗等）
    """
    return Response(
        content=get_metrics(),
        media_type=get_content_type(),
    )


if __name__ == "__main__":
    # 启动 uvicorn 服务器
    # MCP 和知识库的初始化将在 lifespan 中完成
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False
    )
