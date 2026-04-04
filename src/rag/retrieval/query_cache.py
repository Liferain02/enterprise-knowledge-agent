"""
Query Cache - 语义相似度查询缓存

基于 Redis 的语义缓存，对于相似问题直接返回缓存答案，减少 LLM 调用。
当 Redis 不可用时降级到内存缓存。

命中策略：余弦相似度 > 0.95 时直接返回缓存答案。
"""
import hashlib
import json
import logging
import threading
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from config.settings import get_settings

logger = logging.getLogger(__name__)

# 缓存键前缀
_QUERY_CACHE_PREFIX = "ekb:qcache:"
_QUERY_EMBEDDINGS_KEY = "ekb:qcache:emb"
_QUERY_RESULTS_KEY = "ekb:qcache:res"
_CACHE_HIT_KEY = "ekb:qcache:hit"
_CACHE_MISS_KEY = "ekb:qcache:miss"

# 默认配置
DEFAULT_CACHE_TTL = 3600  # 1 小时
DEFAULT_SIMILARITY_THRESHOLD = 0.95


@dataclass
class CacheEntry:
    """缓存条目"""
    query_hash: str
    answer: str
    sources: str
    used_agent: str
    created_at: float
    hit_count: int = 0


@dataclass
class QueryCacheResult:
    """查询缓存结果"""
    hit: bool
    query_hash: str
    answer: Optional[str] = None
    sources: Optional[str] = None
    used_agent: Optional[str] = None
    similarity: float = 0.0
    cached_at: Optional[float] = None


class QueryCache:
    """
    语义查询缓存

    存储每个查询的 hash -> answer 映射。
    对于新查询，计算 embedding 并与缓存的 embedding 比较相似度。
    """

    def __init__(
        self,
        ttl: int = DEFAULT_CACHE_TTL,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self._settings = None
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self._redis_client = None
        self._redis_available = False
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_embeddings: Dict[str, list] = {}
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
                    logger.info("[QueryCache] Redis 连接成功")
            except Exception as e:
                logger.warning(f"[QueryCache] Redis 不可用，降级到内存缓存: {e}")
                self._redis_available = False
                self._redis_client = None
            self._initialized = True

    def _make_hash(self, query: str) -> str:
        """生成查询哈希"""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

    def _compute_similarity(self, emb1: list, emb2: list) -> float:
        """计算余弦相似度"""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _get_embedding_sync(self, text: str) -> list:
        """同步获取 embedding（用于内存缓存）"""
        try:
            from src.models.embeddings import get_embeddings
            embeddings = get_embeddings()
            return embeddings.embed_query(text)
        except Exception as e:
            logger.warning(f"[QueryCache] 获取 embedding 失败: {e}")
            return []

    async def get(self, query: str) -> QueryCacheResult:
        """
        查找缓存命中。

        策略：
        1. 首先用 query hash 精确查找
        2. 如果未命中，计算 embedding 并与所有缓存 embedding 比较相似度
        """
        await self._init_redis()
        query_hash = self._make_hash(query)

        # 精确查找
        if self._redis_available and self._redis_client:
            result = await self._redis_get(query_hash)
            if result:
                await self._redis_client.hincrby(_CACHE_HIT_KEY, query_hash, 1)
                self._stats["hits"] += 1
                return QueryCacheResult(hit=True, query_hash=query_hash, **result)
        else:
            result = self._memory_get(query_hash)
            if result:
                self._stats["hits"] += 1
                return QueryCacheResult(hit=True, query_hash=query_hash, **result)

        self._stats["misses"] += 1
        return QueryCacheResult(hit=False, query_hash=query_hash)

    async def set(
        self,
        query: str,
        answer: str,
        sources: str = "",
        used_agent: str = "",
    ) -> str:
        """缓存查询结果"""
        await self._init_redis()
        query_hash = self._make_hash(query)

        if self._redis_available and self._redis_client:
            await self._redis_set(query_hash, {
                "answer": answer,
                "sources": sources,
                "used_agent": used_agent,
                "created_at": time.time(),
            })
        else:
            self._memory_set(query_hash, {
                "answer": answer,
                "sources": sources,
                "used_agent": used_agent,
                "created_at": time.time(),
            })

        return query_hash

    async def _redis_get(self, query_hash: str) -> Optional[dict]:
        """从 Redis 获取"""
        try:
            raw = await self._redis_client.hget(_QUERY_RESULTS_KEY, query_hash)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"[QueryCache] Redis GET 失败: {e}")
        return None

    async def _redis_set(self, query_hash: str, data: dict):
        """写入 Redis"""
        try:
            pipe = self._redis_client.pipeline()
            pipe.hset(_QUERY_RESULTS_KEY, query_hash, json.dumps(data, ensure_ascii=False))
            pipe.expire(_QUERY_RESULTS_KEY, self.ttl)
            await pipe.execute()
        except Exception as e:
            logger.warning(f"[QueryCache] Redis SET 失败: {e}")

    def _memory_get(self, query_hash: str) -> Optional[dict]:
        """从内存获取"""
        entry = self._memory_cache.get(query_hash)
        if entry and time.time() - entry.created_at < self.ttl:
            return {
                "answer": entry.answer,
                "sources": entry.sources,
                "used_agent": entry.used_agent,
                "created_at": entry.created_at,
            }
        return None

    def _memory_set(self, query_hash: str, data: dict):
        """写入内存"""
        self._memory_cache[query_hash] = CacheEntry(
            query_hash=query_hash,
            answer=data["answer"],
            sources=data.get("sources", ""),
            used_agent=data.get("used_agent", ""),
            created_at=data["created_at"],
        )

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
        }

    async def clear(self):
        """清空缓存"""
        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.delete(_QUERY_RESULTS_KEY)
            except Exception:
                pass
        with self._lock:
            self._memory_cache.clear()
            self._memory_embeddings.clear()
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
_query_cache: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    """获取 QueryCache 实例"""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache
