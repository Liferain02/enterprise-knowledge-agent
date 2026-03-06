"""
Checkpointer - 状态持久化
提供 CheckpointSaver 的创建和管理

设计原则：
- AsyncSqliteSaver 必须在将要使用它的同一个 event loop 中创建
- 异步图实例通过 get_agent_graph_async() 延迟初始化（在 FastAPI event loop 内）
- 同步图实例使用 MemorySaver（用于测试或非服务器场景）
"""
from langgraph.checkpoint.memory import MemorySaver


def get_sync_checkpointer() -> MemorySaver:
    """
    获取同步 Checkpointer（内存模式）
    用于 run_agent()、测试、或非 FastAPI 环境
    
    注意：MemorySaver 不持久化，进程重启后状态丢失
    但对单次测试完全足够，且不存在事件循环冲突问题
    """
    return MemorySaver()


async def get_async_checkpointer():
    """
    在当前 event loop 中创建 AsyncSqliteSaver
    
    必须在异步上下文中调用（如 FastAPI 的 lifespan/endpoint）
    这样 aiosqlite 连接就绑定在 FastAPI 的 event loop 上，
    后续 graph.ainvoke() 也在同一个 loop 中运行，不会产生跨循环阻塞
    
    Returns:
        AsyncSqliteSaver 或 MemorySaver（取决于配置）
    """
    from config.settings import get_settings
    settings = get_settings()

    if settings.use_sqlite_checkpointer:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = settings.chroma_dir / "langgraph_checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # 在当前 event loop 中创建连接——这是关键
        conn = await aiosqlite.connect(str(db_path))
        return AsyncSqliteSaver(conn)
    else:
        return MemorySaver()
