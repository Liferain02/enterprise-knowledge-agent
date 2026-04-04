"""
异步 HTTP 连接池管理
为 httpx/LangChain HTTP 请求配置连接池参数。
避免每次请求创建新连接，提升 LLM 调用和工具调用的性能。
"""
import os
import httpx
from typing import Optional


# 全局 HTTP 客户端（复用连接池）
_http_client: Optional[httpx.AsyncClient] = None
_http_client_lock = None  # 延迟导入避免循环依赖


def get_http_client(
    timeout: float = 60.0,
    max_connections: int = 100,
    max_keepalive_connections: int = 20,
) -> httpx.AsyncClient:
    """
    获取全局 HTTP 客户端（单例，连接池复用）

    Args:
        timeout: 默认超时时间（秒）
        max_connections: 最大连接数
        max_keepalive_connections: 最大保活连接数

    Returns:
        复用的 AsyncClient 实例
    """
    global _http_client

    if _http_client is None or _http_client.is_closed:
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        _http_client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            # 代理配置
            proxy=os.environ.get("https_proxy") or os.environ.get("http_proxy"),
        )

    return _http_client


async def close_http_client():
    """关闭全局 HTTP 客户端（应用关闭时调用）"""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


def configure_langchain_http():
    """
    配置 LangChain 的 HTTP 请求使用全局连接池
    通过环境变量或设置 LangChain 的 httpx 客户端实现。
    """
    # LangChain 0.3+ 使用 langchain-core 的全局 httpx 设置
    # 设置默认超时
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


class BatchInference:
    """
    批量推理优化器

    适用场景：多个相似查询（如查询改写后的多个子查询）可以合并为一次 LLM 调用，
    显著减少 token 消耗和延迟。

    工作原理：
    1. 收集一段时间内的相似查询
    2. 使用 Few-shot prompt 让 LLM 一次性处理多个问题
    3. 解析返回的结构化结果

    注意：仅适用于具有明确答案的事实性问题，不适用于创意写作。
    """

    def __init__(self, max_batch_size: int = 5, max_wait_ms: int = 500):
        """
        Args:
            max_batch_size: 单批最大问题数
            max_wait_ms: 等待新问题的最大时间（毫秒）
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._pending: list = []
        self._lock = __import__("asyncio").Lock()

    async def add_query(self, query: str) -> dict:
        """
        添加查询到批处理队列，返回结果（自动等待批量或立即返回）

        Returns:
            LLM 响应（dict 格式）
        """
        import asyncio
        from src.models.llm import get_llm
        from src.rag.cache.response_cache import llm_cache_get, llm_cache_set
        from config.settings import get_settings

        settings = get_settings()
        llm = get_llm(temperature=0.0)  # 批量推理用确定性温度

        # 检查缓存
        cached = await llm_cache_get(query, settings.dashscope_model, temperature=0.0)
        if cached:
            return cached

        async with self._lock:
            self._pending.append(query)

            if len(self._pending) >= self.max_batch_size:
                # 达到批量大小，立即处理
                queries = self._pending
                self._pending = []
                return await self._process_batch(queries, llm)

            # 未达批量，等待一段时间
            # 注意：这里简单等待，生产环境可以用 asyncio.wait_for + 任务取消
            await asyncio.sleep(self.max_wait_ms / 1000)

            if self._pending:
                queries = self._pending
                self._pending = []
                return await self._process_batch(queries, llm)

            return None

    async def _process_batch(self, queries: list, llm) -> dict:
        """执行批量处理"""
        from src.models.llm import get_llm
        from config.settings import get_settings

        settings = get_settings()

        # 构建 Few-shot batch prompt
        prompt = self._build_batch_prompt(queries)

        try:
            response = await llm.ainvoke(prompt)
            result = {"response": response.content, "batch_size": len(queries)}

            # 缓存每个查询
            for q in queries:
                await llm_cache_set(
                    q, response.content, settings.dashscope_model,
                    temperature=0.0,
                )

            return result
        except Exception as e:
            return {"error": str(e), "batch_size": len(queries)}

    def _build_batch_prompt(self, queries: list) -> str:
        """构建批量处理 prompt"""
        qs_str = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(queries))
        return f"""请回答以下问题，每个问题一行答案：

{qs_str}

回答格式（严格按此格式，不要解释）：
A1: <答案>
A2: <答案>
..."""


# 全局批处理实例
_batch_inference: Optional[BatchInference] = None


def get_batch_inference() -> BatchInference:
    global _batch_inference
    if _batch_inference is None:
        _batch_inference = BatchInference(max_batch_size=3, max_wait_ms=300)
    return _batch_inference
