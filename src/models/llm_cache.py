"""
LLM 响应缓存

基于 Redis 的 LLM 响应缓存，通过 prompt hash 键直接返回缓存结果。
适用于完全相同的 prompt 重复调用场景（如客服 FAQ、文档问答等）。
Redis 不可用时降级到内存缓存。
"""
import hashlib
import json
import logging
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from config.settings import get_settings

logger = logging.getLogger(__name__)

_LLM_CACHE_PREFIX = "ekb:llmcache:"
_DEFAULT_LLM_CACHE_TTL = 1800  # 30 分钟


@dataclass
class LLMCacheEntry:
    """LLM 缓存条目"""
    content: str
    model: str
    created_at: float
    hit_count: int = 0


class LLMCache:
    """
    LLM 响应缓存

    基于 prompt 的 MD5 hash 作为 key，缓存完整的 LLM 响应。
    支持：
    - TTL 过期
    - 命中率统计
    - Redis / 内存双层降级
    """

    def __init__(self, ttl: int = _DEFAULT_LLM_CACHE_TTL):
        self._settings = None
        self.ttl = ttl
        self._redis_client = None
        self._redis_available = False
        self._memory_cache: Dict[str, LLMCacheEntry] = {}
        self._lock = threading.Lock()
        self._initialized = False
        self._stats = {"hits": 0, "misses": 0}

    @property
    def settings(self):
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    async def _init_redis(self):
        """初始化 Redis 连接"""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                import redis.asyncio as aioredis
                settings = self.settings
                host = getattr(settings, "redis_host", "disabled")
                if not host or host == "disabled":
                    self._redis_available = False
                else:
                    self._redis_client = aioredis.Redis(
                        host=host,
                        port=getattr(settings, "redis_port", 6379),
                        password=getattr(settings, "redis_password", None) or None,
                        db=getattr(settings, "redis_db", 0),
                        decode_responses=True,
                        socket_connect_timeout=2,
                        socket_timeout=3,
                    )
                    await self._redis_client.ping()
                    self._redis_available = True
                    logger.info("[LLMCache] Redis 连接成功")
            except Exception as e:
                logger.warning(f"[LLMCache] Redis 不可用，降级到内存: {e}")
                self._redis_available = False
                self._redis_client = None
            self._initialized = True

    def _make_key(self, prompt: str, model: str) -> str:
        """生成缓存 key：prompt hash + model"""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:32]
        return f"{_LLM_CACHE_PREFIX}{model}:{prompt_hash}"

    async def get(self, prompt: str, model: str) -> Optional[LLMCacheEntry]:
        """获取缓存的 LLM 响应"""
        await self._init_redis()
        key = self._make_key(prompt, model)

        if self._redis_available and self._redis_client:
            try:
                raw = await self._redis_client.get(key)
                if raw:
                    data = json.loads(raw)
                    self._stats["hits"] += 1
                    return LLMCacheEntry(**data)
            except Exception as e:
                logger.warning(f"[LLMCache] Redis GET 失败: {e}")
        else:
            with self._lock:
                entry = self._memory_cache.get(key)
                if entry and time.time() - entry.created_at < self.ttl:
                    self._stats["hits"] += 1
                    entry.hit_count += 1
                    return entry

        self._stats["misses"] += 1
        return None

    async def set(self, prompt: str, model: str, content: str):
        """缓存 LLM 响应"""
        await self._init_redis()
        key = self._make_key(prompt, model)
        entry = LLMCacheEntry(
            content=content,
            model=model,
            created_at=time.time(),
            hit_count=0,
        )

        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.setex(
                    key,
                    self.ttl,
                    json.dumps(asdict(entry), ensure_ascii=False)
                )
            except Exception as e:
                logger.warning(f"[LLMCache] Redis SET 失败: {e}")
                # 降级到内存
                with self._lock:
                    self._memory_cache[key] = entry
        else:
            with self._lock:
                self._memory_cache[key] = entry

    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "total": total,
            "hit_rate": round(hit_rate, 4),
            "storage": "redis" if self._redis_available else "memory",
            "cache_size": len(self._memory_cache),
            "ttl_seconds": self.ttl,
        }

    async def clear(self):
        """清空缓存"""
        if self._redis_available and self._redis_client:
            try:
                cursor = 0
                deleted = 0
                while True:
                    cursor, keys = await self._redis_client.scan(
                        cursor=cursor,
                        match=f"{_LLM_CACHE_PREFIX}*",
                        count=100
                    )
                    if keys:
                        await self._redis_client.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
                logger.info(f"[LLMCache] 清空完成，删除了 {deleted} 个缓存条目")
            except Exception as e:
                logger.warning(f"[LLMCache] 清空失败: {e}")
        with self._lock:
            self._memory_cache.clear()
        self._stats = {"hits": 0, "misses": 0}

    def close(self):
        """关闭"""
        if self._redis_client:
            try:
                import redis.asyncio as aioredis
                asyncio.create_task(self._redis_client.aclose())
            except Exception:
                pass


# 全局实例
_llm_cache: Optional[LLMCache] = None


def get_llm_cache() -> LLMCache:
    """获取 LLM Cache 实例"""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMCache()
    return _llm_cache
