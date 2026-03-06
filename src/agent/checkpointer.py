"""
Checkpointer - 状态持久化
提供 AsyncSqliteSaver 的创建和管理
"""
import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# 全局单例
_checkpointer = None


async def _create_async_sqlite_checkpointer() -> AsyncSqliteSaver:
    """创建 AsyncSqliteSaver 异步检查点"""
    import aiosqlite
    from config.settings import get_settings
    
    settings = get_settings()
    db_path = settings.chroma_dir / "langgraph_checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    return AsyncSqliteSaver(conn)


def get_checkpointer():
    """
    获取 Checkpointer（单例模式）
    使用 AsyncSqliteSaver 持久化 LangGraph 的运行状态
    """
    global _checkpointer
    
    if _checkpointer is not None:
        return _checkpointer
    
    from config.settings import get_settings
    settings = get_settings()
    
    # 使用配置决定是否使用 SQLite 持久化
    if settings.use_sqlite_checkpointer:
        # 尝试获取已有事件循环
        try:
            loop = asyncio.get_running_loop()
            # 如果有运行中的循环，需要创建任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _create_async_sqlite_checkpointer())
                _checkpointer = future.result()
        except RuntimeError:
            # 没有运行中的循环，可以直接使用 asyncio.run
            _checkpointer = asyncio.run(_create_async_sqlite_checkpointer())
    else:
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
    
    return _checkpointer
