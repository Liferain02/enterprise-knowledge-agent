"""
通用缓存层 - 基于 Redis（生产）/ 内存（降级）
提供两层缓存：
1. LLM 响应缓存（`llm_cache_get/set`）
2. 检索结果缓存（`retrieval_cache_get/set`）
3. CRUD 缓存（通用 `cache_get/set`）

特性：
- Redis 优先，内存降级（`grade_cache` 同款策略）
- 自动 TTL 过期
- LRU 淘汰（内存层）
- 缓存统计（hit/miss/eviction）
- 分布式一致性：支持 cache-aside 模式
"""
import asyncio
import hashlib
import json
import logging
import time
import threading
from typing import Optional, Any, Callable, Awaitable

import redis.asyncio as aioredis
from redis.asyncio.client import Redis

from config.settings import get_settings


logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

_SETTINGS_LOADED = False
_LLM_CACHE_TTL = 300      # LLM 响应缓存：5 分钟
_RETRIEVAL_CACHE_TTL = 60  # 检索结果缓存：1 分钟（知识库变化快，不宜太长）
_GENERAL_CACHE_TTL = 600   # 通用缓存：10 分钟

_KEY_PREFIX = {
    "llm": "ekb:llm:",
    "retrieval": "ekb:ret:",
    "general": "ekb:cache:",
    "metric": "ekb:metric:",
}


# ============================================================
# Redis 连接
# ============================================================

_redis_client: Optional[Redis] = None
_redis_available = False


async def get_redis_client() -> Optional[Redis]:
    global _redis_client, _redis_available

    if _redis_available and _redis_client is not None:
        return _redis_client

    settings = get_settings()
    host = getattr(settings, "redis_host", "localhost")

    if not host or host in ("", "disabled"):
        _redis_available = False
        return None

    try:
        _redis_client = aioredis.Redis(
            host=host,
            port=getattr(settings, "redis_port", 6379),
            password=getattr(settings, "redis_password", None) or None,
            db=getattr(settings, "redis_db", 0),
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5,
            retry_on_timeout=False,
        )
        await _redis_client.ping()
        _redis_available = True
        logger.info(f"[Cache] Redis 连接成功: {host}")
        return _redis_client
    except Exception as e:
        logger.warning(f"[Cache] Redis 不可用: {e}，降级到内存缓存")
        _redis_available = False
        _redis_client = None
        return None


async def close_redis():
    global _redis_client, _redis_available
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        finally:
            _redis_client = None
            _redis_available = False


def is_redis_available() -> bool:
    return _redis_available


# ============================================================
# 内存缓存（降级用）
# ============================================================

_memory_cache: dict = {}
_memory_timestamps: dict = {}
_memory_access_order: list = []  # 用于 LRU
_memory_lock = threading.RLock()
_MEMORY_MAX_SIZE = 200  # 最多缓存条目数（防止内存无限增长）


def _memory_get(key: str) -> Optional[dict]:
    with _memory_lock:
        if key in _memory_cache:
            result, timestamp, ttl = _memory_cache[key]
            if time.time() - timestamp < ttl:
                # LRU: 移到末尾（最久未使用）
                if key in _memory_access_order:
                    _memory_access_order.remove(key)
                _memory_access_order.append(key)
                return result
            else:
                # 过期，删除
                del _memory_cache[key]
                if key in _memory_access_order:
                    _memory_access_order.remove(key)
        return None


def _memory_set(key: str, value: dict, ttl: int):
    with _memory_lock:
        # LRU 淘汰：超过最大容量时移除最老的
        while len(_memory_cache) >= _MEMORY_MAX_SIZE and _memory_access_order:
            oldest = _memory_access_order.pop(0)
            _memory_cache.pop(oldest, None)

        _memory_cache[key] = (value, time.time(), ttl)
        if key not in _memory_access_order:
            _memory_access_order.append(key)


def _memory_delete(key: str):
    with _memory_lock:
        _memory_cache.pop(key, None)
        if key in _memory_access_order:
            _memory_access_order.remove(key)


def _memory_stats() -> dict:
    with _memory_lock:
        return {
            "size": len(_memory_cache),
            "max_size": _MEMORY_MAX_SIZE,
            "utilization": round(len(_memory_cache) / _MEMORY_MAX_SIZE, 3),
        }


# ============================================================
# 缓存统计
# ============================================================

_cache_stats = {"hit": 0, "miss": 0, "eviction": 0}
_stats_lock = threading.Lock()


def _inc_stat(name: str):
    with _stats_lock:
        _cache_stats[name] = _cache_stats.get(name, 0) + 1


async def cache_stats() -> dict:
    """获取缓存统计"""
    redis_ok = await get_redis_client()
    result = {
        "redis_available": redis_ok is not None,
        "redis_available_sync": _redis_available,
        "stats": dict(_cache_stats),
    }
    if not redis_ok:
        result["memory"] = _memory_stats()
    return result


# ============================================================
# LLM 响应缓存
# ============================================================

def _make_llm_key(query: str, model: str, temperature: float) -> str:
    """生成 LLM 缓存 key（对 query 取 MD5）"""
    key_str = f"{model}:{temperature}:{query[:200]}"
    return _KEY_PREFIX["llm"] + hashlib.md5(key_str.encode()).hexdigest()


async def llm_cache_get(
    query: str,
    model: str,
    temperature: float = 0.7,
) -> Optional[dict]:
    """
    获取 LLM 缓存（异步）

    Returns:
        {"response": "...", "model": "...", "cached": true} 或 None
    """
    key = _make_llm_key(query, model, temperature)

    redis_client = await get_redis_client()
    if redis_client:
        try:
            raw = await redis_client.get(key)
            if raw:
                _inc_stat("hit")
                data = json.loads(raw)
                logger.debug(f"[Cache] LLM HIT: {key[:32]}...")
                return data
        except Exception as e:
            logger.warning(f"[Cache] Redis LLM GET 失败: {e}")

    result = _memory_get(key)
    if result:
        _inc_stat("hit")
        return result

    _inc_stat("miss")
    return None


async def llm_cache_set(
    query: str,
    response: str,
    model: str,
    temperature: float = 0.7,
    metadata: Optional[dict] = None,
):
    """
    写入 LLM 响应缓存（异步）

    注意：仅缓存 temperature=0.0~0.3 的确定性请求
    """
    # 高温度（创意写作/闲聊）不缓存
    if temperature > 0.3:
        return

    key = _make_llm_key(query, model, temperature)
    data = {
        "response": response,
        "model": model,
        "temperature": temperature,
        "cached": True,
        "cached_at": time.time(),
        **(metadata or {}),
    }

    redis_client = await get_redis_client()
    if redis_client:
        try:
            await redis_client.setex(key, _LLM_CACHE_TTL, json.dumps(data, ensure_ascii=False))
            return
        except Exception as e:
            logger.warning(f"[Cache] Redis LLM SET 失败: {e}")

    _memory_set(key, data, _LLM_CACHE_TTL)


# ============================================================
# 检索结果缓存
# ============================================================

def _make_retrieval_key(query: str, top_k: int, user_id: Optional[str]) -> str:
    """生成检索缓存 key"""
    q_key = query[:150].lower()
    ident = f"{top_k}:{user_id or 'anonymous'}"
    return _KEY_PREFIX["retrieval"] + hashlib.md5(f"{ident}:{q_key}".encode()).hexdigest()


async def retrieval_cache_get(
    query: str,
    top_k: int = 5,
    user_id: Optional[str] = None,
) -> Optional[dict]:
    """
    获取检索结果缓存

    Returns:
        {"results": [...], "score_type": "..."} 或 None
    """
    key = _make_retrieval_key(query, top_k, user_id)

    redis_client = await get_redis_client()
    if redis_client:
        try:
            raw = await redis_client.get(key)
            if raw:
                _inc_stat("hit")
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"[Cache] Redis retrieval GET 失败: {e}")

    result = _memory_get(key)
    if result:
        _inc_stat("hit")
        return result

    _inc_stat("miss")
    return None


async def retrieval_cache_set(
    query: str,
    results: list,
    top_k: int = 5,
    user_id: Optional[str] = None,
    score_type: str = "hybrid",
):
    """写入检索结果缓存"""
    key = _make_retrieval_key(query, top_k, user_id)
    data = {
        "results": results,
        "score_type": score_type,
        "cached_at": time.time(),
        "count": len(results),
    }

    redis_client = await get_redis_client()
    if redis_client:
        try:
            await redis_client.setex(key, _RETRIEVAL_CACHE_TTL, json.dumps(data, ensure_ascii=False))
            return
        except Exception as e:
            logger.warning(f"[Cache] Redis retrieval SET 失败: {e}")

    _memory_set(key, data, _RETRIEVAL_CACHE_TTL)


async def retrieval_cache_invalidate(query_prefix: str = None):
    """
    失效检索缓存
    - query_prefix: 可选，仅失效包含此前缀的缓存
    - 空：失效全部
    """
    redis_client = await get_redis_client()
    prefix = _KEY_PREFIX["retrieval"]

    if redis_client:
        try:
            cursor = 0
            deleted = 0
            match = f"{prefix}*" if not query_prefix else f"{prefix}*{query_prefix}*"
            while True:
                cursor, keys = await redis_client.scan(cursor=cursor, match=match, count=100)
                if keys:
                    await redis_client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(f"[Cache] 失效检索缓存: {deleted} 条")
            return
        except Exception as e:
            logger.warning(f"[Cache] Redis 失效失败: {e}")

    # 内存：清空全部（或按前缀过滤）
    with _memory_lock:
        if query_prefix:
            to_del = [k for k in _memory_cache if query_prefix in k]
            for k in to_del:
                _memory_cache.pop(k, None)
                _memory_access_order.remove(k) if k in _memory_access_order else None
        else:
            _memory_cache.clear()
            _memory_access_order.clear()


# ============================================================
# 通用缓存（Cache-Aside 模式封装）
# ============================================================

async def cache_get(key: str, cache_type: str = "general") -> Optional[Any]:
    """通用 GET"""
    redis_client = await get_redis_client()
    full_key = _KEY_PREFIX.get(cache_type, _KEY_PREFIX["general"]) + key

    if redis_client:
        try:
            raw = await redis_client.get(full_key)
            if raw:
                _inc_stat("hit")
                return json.loads(raw)
        except Exception:
            pass

    result = _memory_get(full_key)
    if result:
        _inc_stat("hit")
    else:
        _inc_stat("miss")
    return result


async def cache_set(key: str, value: Any, cache_type: str = "general", ttl: int = None):
    """通用 SET"""
    redis_client = await get_redis_client()
    full_key = _KEY_PREFIX.get(cache_type, _KEY_PREFIX["general"]) + key
    ttl = ttl or _GENERAL_CACHE_TTL

    if redis_client:
        try:
            await redis_client.setex(full_key, ttl, json.dumps(value, ensure_ascii=False))
            return
        except Exception as e:
            logger.warning(f"[Cache] Redis SET 失败: {e}")

    _memory_set(full_key, value, ttl)


async def cache_get_or_set(
    key: str,
    factory: Callable[[], Awaitable[Any]],
    cache_type: str = "general",
    ttl: int = None,
) -> Any:
    """
    Cache-Aside 模式：先查缓存，未命中则调用 factory 获取并缓存

    这是缓存的标准用法，推荐在业务代码中使用此方法。
    """
    cached = await cache_get(key, cache_type)
    if cached is not None:
        return cached

    value = await factory()
    if value is not None:
        await cache_set(key, value, cache_type, ttl)
    return value


# ============================================================
# 健康检查
# ============================================================

async def health_check() -> bool:
    """检查缓存层健康状态"""
    client = await get_redis_client()
    if client is None:
        return True  # 内存缓存也算健康
    try:
        await client.ping()
        return True
    except Exception:
        return False
