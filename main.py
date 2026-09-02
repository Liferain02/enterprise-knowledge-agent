"""
实验室科研智能助手 - 主入口

定位：服务计算机专业研究生实验室课题组的科研知识管理与协作检索
核心能力：
- Corrective RAG：检索结果 LLM 评估 + 自我纠错
- 资料分类检索（论文/项目/组会/实验记录/FAQ）
- 结构化来源展示与版本追踪
- 简化权限控制（公共 / 项目组内 / 负责人可见）
- 异步入库 Pipeline（启动不阻塞）
- 流式 SSE 输出
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

# 配置结构化日志
import logging as _logging
from pathlib import Path

_log_level = _logging.DEBUG if settings.debug else _logging.INFO
_log_env = "development" if settings.debug else "production"
_log_file = str(Path(__file__).parent / "logs" / "agent.log")

# 简单日志配置（无外部依赖）
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        _logging.StreamHandler(),
        _logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

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
    chat_router, knowledge_router, research_router, auth_router, feedback_router, vision_router, _a2a_router
)
from src.api.routes.websocket_routes import ws_router

# 导入核心组件
from src.models.mcp_client import mcp_manager as global_mcp_manager
from src.rag import get_vectorstore_manager

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
    print("启动实验室科研智能助手...")
    print("定位：科研知识管理与协作检索系统")
    print("场景：论文 / 项目 / 组会 / 环境配置 / 实验记录")
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

    print("=" * 60)
    print("服务就绪！")
    print(f"API: http://{settings.api_host}:{settings.api_port}")
    print(f"文档: http://{settings.api_host}:{settings.api_port}/docs")
    print("=" * 60)

    yield

    # ── 关闭（优雅关闭 Graceful Shutdown）─────────────────────
    import asyncio

    print("[Shutdown] 收到关闭信号，开始优雅关闭...")

    # 1. 停止接收新请求（不再创建新 session）
    print("[Shutdown] 停止接收新请求...")

    # 2. 等待现有请求完成（最多等待 30 秒）
    print("[Shutdown] 等待现有请求完成（最多 30s）...")
    await asyncio.sleep(0.5)  # 给 SSE 连接一点时间完成

    # 3. 停止入库 Worker（处理完队列中已有的任务）
    for w, _ in _workers:
        print("[Shutdown] 停止入库 Worker...")
        w.stop()
    print("[Shutdown] 入库 Worker 已停止")

    # 4. 关闭 Redis 连接（缓存持久化）
    try:
        from src.rag.evaluation import grade_cache
        await grade_cache.close_redis()
        print("[Shutdown] Redis 连接已关闭")
    except Exception as e:
        print(f"[Shutdown] Redis 关闭时出错 (可忽略): {e}")

    # 5. 关闭 MCP 连接
    try:
        await global_mcp_manager.close()
        print("[Shutdown] MCP 连接已关闭")
    except Exception as e:
        print(f"[Shutdown] MCP 连接关闭出错 (可忽略): {e}")

    print("[Shutdown] 优雅关闭完成！")


# 创建FastAPI应用
app = FastAPI(
    title="实验室科研智能助手",
    description=(
        "面向计算机专业研究生实验室课题组的知识管理与协作检索系统\n\n"
        "**定位**：服务论文、项目、组会、实验记录、环境配置等场景\n\n"
        "**核心能力**：\n"
        "- Corrective RAG：检索结果 LLM 评估 + 自我纠错\n"
        "- 资料分层检索与结构化来源展示\n"
        "- 简化权限隔离（公共 / 项目组内 / 负责人可见）\n"
        "- 文档版本管理与时效性\n"
        "- 冲突检测与拒答策略\n"
        "- 异步入库 Pipeline\n"
        "- 流式 SSE 输出\n"
    ),
    version="1.0.0",
    lifespan=lifespan
)

# 注册统一异常处理器（中间件 + 全局 handlers）
from src.api.middleware import register_exception_handlers
register_exception_handlers(app, debug=settings.debug)

# 注册限流中间件
try:
    from src.api.middleware import register_rate_limit_middleware
    register_rate_limit_middleware(app, enabled=getattr(settings, "rate_limit_enabled", True))
except Exception:
    pass

# 添加输入安全中间件（SQL注入/XSS/Prompt注入/PII检测）
from src.api.middleware.input_security import InputSecurityMiddleware
app.add_middleware(InputSecurityMiddleware, strict=settings.debug)

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
app.include_router(research_router)
app.include_router(auth_router)
app.include_router(feedback_router)
app.include_router(vision_router)
app.include_router(_a2a_router)  # A2A Agent Card 暴露
app.include_router(ws_router)  # WebSocket 实时对话


@app.get("/health")
async def health_check():
    """
    生产级健康检查
    包含核心组件状态，用于 K8s/负载均衡器 存活探针和就绪探针
    """
    import time
    start = time.time()
    components = {}
    overall_status = "healthy"

    # ── 1. 知识库（向量存储）──────────
    try:
        vs_manager = get_vectorstore_manager()
        info = vs_manager.get_collection_info()
        components["vectorstore"] = {
            "status": "healthy",
            "doc_count": info.get("count", 0),
        }
    except Exception as e:
        components["vectorstore"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "degraded"

    # ── 2. Redis（可选的评估缓存）──────────
    # 未配置 Redis 时系统明确使用进程内缓存，这不是服务降级。
    redis_host = (settings.redis_host or "").strip().lower()
    if redis_host in ("", "disabled"):
        components["redis"] = {
            "status": "disabled",
            "fallback": "memory",
        }
    else:
        try:
            from src.rag.evaluation import grade_cache
            redis_ok = await grade_cache.health_check()
            components["redis"] = {
                "status": "healthy" if redis_ok else "unavailable",
            }
            if not redis_ok:
                overall_status = "degraded"
        except Exception as e:
            components["redis"] = {"status": "unavailable", "error": str(e)}
            overall_status = "degraded"

    # ── 3. MCP 工具可用性 ──────────
    try:
        tools = global_mcp_manager.get_tools()
        components["mcp"] = {
            "status": "healthy",
            "tool_count": len(tools),
        }
    except Exception as e:
        components["mcp"] = {"status": "degraded", "error": str(e)}
        overall_status = "degraded"

    # ── 4. LLM 连通性 ───────────────
    try:
        from src.models.llm import get_llm
        llm = get_llm()
        components["llm"] = {
            "status": "healthy",
            "provider": settings.llm_provider,
            "model": settings.dashscope_model if settings.llm_provider == "qwen" else settings.openai_model,
        }
    except Exception as e:
        components["llm"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "unhealthy"

    response = {
        "status": overall_status,
        "service": "lab-knowledge-assistant",
        "version": "1.0.0",
        "uptime_ms": round((time.time() - start) * 1000, 1),
        "components": components,
    }

    from fastapi import status
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=response,
        status_code=status.HTTP_200_OK if overall_status != "unhealthy" else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/health/live")
async def liveness_check():
    """
    K8s Liveness Probe - 进程存活检查
    只检查进程是否存活，不检查依赖
    """
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_check():
    """
    K8s Readiness Probe - 就绪探针
    检查核心依赖（向量存储 + LLM），不通过则 K8s 停止转发流量
    """
    try:
        # 快速检查向量存储
        vs_manager = get_vectorstore_manager()
        vs_manager.get_collection_info()
        return {"status": "ready"}
    except Exception:
        from fastapi import status
        return {"status": "not_ready"}, status.HTTP_503_SERVICE_UNAVAILABLE


if __name__ == "__main__":
    # 启动 uvicorn 服务器
    # MCP 和知识库的初始化将在 lifespan 中完成
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False
    )
