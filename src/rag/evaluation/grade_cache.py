"""
评估结果缓存层 - Redis 持久化

提供两层缓存：
1. Redis（生产环境）：跨进程共享，支持 TTL，进程重启后缓存不丢失
2. 内存（开发/无 Redis）：作为降级，不支持跨进程共享

使用方式：
- 优先尝试连接 Redis（通过 settings.redis_* 配置）
- 若 Redis 不可用，自动降级到内存缓存
- 支持 TTL 过期、LRU 淘汰
"""
import asyncio
import hashlib
import json
import logging
import time
from typing import Optional, Any

import redis.asyncio as aioredis
from redis.asyncio.client import Redis

from config.settings import get_settings

logger = logging.getLogger(__name__)


# ============================================================
# 缓存键前缀
# ============================================================

_GRADE_KEY_PREFIX = "ekb:grade:"
_METRIC_KEY_PREFIX = "ekb:metric:"


# ============================================================
# Redis 连接管理器（懒加载 + 单例）
# ============================================================

_redis_client: Optional[Redis] = None
_redis_available: bool = False


async def get_redis_client() -> Optional[Redis]:
    """
    获取 Redis 客户端（异步，懒加载）。

    Returns:
        Redis 客户端实例（连接成功时）
        None（Redis 不可用，降级到内存缓存）
    """
    global _redis_client, _redis_available

    if _redis_available and _redis_client is not None:
        return _redis_client

    try:
        settings = get_settings()
        host = getattr(settings, "redis_host", "localhost")
        port = getattr(settings, "redis_port", 6379)
        password = getattr(settings, "redis_password", None)
        db = getattr(settings, "redis_db", 0)

        # 如果 host 为空或 localhost 且未配置，跳过 Redis
        if not host or host in ("", "disabled"):
            logger.info("[Redis Cache] Redis 未配置，跳过")
            _redis_available = False
            return None

        # 连接 Redis
        _redis_client = aioredis.Redis(
            host=host,
            port=port,
            password=password if password else None,
            db=db,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5,
            retry_on_timeout=False,
        )

        # 测试连接
        await _redis_client.ping()
        _redis_available = True
        logger.info(f"[Redis Cache] 连接成功: {host}:{port}/{db}")
        return _redis_client

    except Exception as e:
        logger.warning(f"[Redis Cache] Redis 连接失败: {e}，降级到内存缓存")
        _redis_available = False
        _redis_client = None
        return None


async def close_redis():
    """关闭 Redis 连接（应用关闭时调用）"""
    global _redis_client, _redis_available
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
            logger.info("[Redis Cache] 连接已关闭")
        except Exception as e:
            logger.warning(f"[Redis Cache] 关闭时出错: {e}")
        finally:
            _redis_client = None
            _redis_available = False


def is_redis_available() -> bool:
    """检查 Redis 是否可用（同步，仅反映上次连接状态）"""
    return _redis_available


# ============================================================
# 内存缓存（降级用）
# ============================================================

_memory_cache: dict = {}
_memory_timestamps: dict = {}
_MEMORY_TTL = 300  # 与 Redis 默认 TTL 一致


def _memory_get(key: str) -> Optional[tuple]:
    """从内存缓存读取（同步）"""
    if key in _memory_cache:
        result, timestamp = _memory_cache[key]
        if time.time() - timestamp < _MEMORY_TTL:
            return result
        del _memory_cache[key]
        if key in _memory_timestamps:
            del _memory_timestamps[key]
    return None


def _memory_set(key: str, value: tuple):
    """写入内存缓存（同步）"""
    _memory_cache[key] = (value, time.time())


def _memory_delete(key: str):
    """从内存缓存删除"""
    _memory_cache.pop(key, None)
    _memory_timestamps.pop(key, None)


def _memory_clear():
    """清空内存缓存"""
    global _memory_cache, _memory_timestamps
    _memory_cache.clear()
    _memory_timestamps.clear()


# ============================================================
# 统一缓存接口（自动选择 Redis 或内存）
# ============================================================

_GRADE_TTL = 300  # 5 分钟


def _make_grade_key(query: str, doc_content: str) -> str:
    """生成评估缓存 key（MD5 哈希）"""
    key_str = f"{query}|{doc_content[:500]}"
    return _GRADE_KEY_PREFIX + hashlib.md5(key_str.encode()).hexdigest()


async def grade_cache_get(query: str, doc_content: str) -> Optional[tuple]:
    """
    获取缓存的评估结果（异步，自动选择存储层）

    Returns:
        (score, reasoning) 元组（缓存命中）
        None（缓存未命中）
    """
    key = _make_grade_key(query, doc_content)

    # 优先尝试 Redis
    redis_client = await get_redis_client()
    if redis_client is not None:
        try:
            raw = await redis_client.get(key)
            if raw:
                data = json.loads(raw)
                # 反序列化时附上 "[缓存]" 标记
                score, reasoning = data["score"], data["reasoning"]
                logger.debug(f"[Redis Cache] HIT: {key[:32]}...")
                return (score, f"[缓存]{reasoning}")
            return None
        except Exception as e:
            logger.warning(f"[Redis Cache] 读取失败: {e}，降级到内存")
            pass

    # 降级到内存
    return _memory_get(key)


async def grade_cache_set(query: str, doc_content: str, score: float, reasoning: str):
    """
    写入评估缓存（异步，自动选择存储层）

    Args:
        query: 查询文本
        doc_content: 文档内容（前 500 字符参与 key 计算）
        score: 评估分数
        reasoning: 评估理由
    """
    key = _make_grade_key(query, doc_content)
    data = {"score": score, "reasoning": reasoning}

    # 优先写入 Redis（设置 TTL）
    redis_client = await get_redis_client()
    if redis_client is not None:
        try:
            await redis_client.setex(key, _GRADE_TTL, json.dumps(data, ensure_ascii=False))
            logger.debug(f"[Redis Cache] SET: {key[:32]}... (TTL={_GRADE_TTL}s)")
            return
        except Exception as e:
            logger.warning(f"[Redis Cache] 写入失败: {e}，降级到内存")
            pass

    # 降级到内存
    _memory_set(key, (score, reasoning))


async def grade_cache_clear():
    """
    清空评估缓存（异步）
    生产环境中推荐使用 Redis FLUSHDB，内存则逐个删除
    """
    redis_client = await get_redis_client()
    if redis_client is not None:
        try:
            # 只清空本应用前缀的 key，避免误删其他数据
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await redis_client.scan(
                    cursor=cursor,
                    match=f"{_GRADE_KEY_PREFIX}*",
                    count=100
                )
                if keys:
                    await redis_client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(f"[Redis Cache] 清空完成，删除了 {deleted} 个缓存条目")
            return
        except Exception as e:
            logger.warning(f"[Redis Cache] 清空失败: {e}")
            pass

    # 降级到内存
    _memory_clear()
    logger.info("[Memory Cache] 评估缓存已清空")


async def grade_cache_stats() -> dict:
    """
    获取缓存统计信息（用于监控）

    Returns:
        {"storage": "redis"|"memory", "count": int, "ttl": int}
    """
    redis_client = await get_redis_client()
    if redis_client is not None:
        try:
            cursor = 0
            count = 0
            while True:
                cursor, keys = await redis_client.scan(
                    cursor=cursor,
                    match=f"{_GRADE_KEY_PREFIX}*",
                    count=100
                )
                count += len(keys)
                if cursor == 0:
                    break
            return {
                "storage": "redis",
                "count": count,
                "ttl_seconds": _GRADE_TTL,
                "redis_available": True,
            }
        except Exception:
            pass

    return {
        "storage": "memory",
        "count": len(_memory_cache),
        "ttl_seconds": _MEMORY_TTL,
        "redis_available": False,
    }


# ============================================================
# 指标缓存（用于 rate limiting、熔断等）
# ============================================================

_METRIC_TTL = 60  # 1 分钟


async def metric_cache_incr(key: str, ttl: int = _METRIC_TTL) -> int:
    """
    递增计数器（原子操作，用于 rate limiting）

    Args:
        key: 指标 key（不含前缀）
        ttl: 过期时间（秒）

    Returns:
        递增后的值
    """
    full_key = _METRIC_KEY_PREFIX + key
    redis_client = await get_redis_client()

    if redis_client is not None:
        try:
            pipe = redis_client.pipeline()
            pipe.incr(full_key)
            pipe.expire(full_key, ttl)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.warning(f"[Redis Cache] metric incr 失败: {e}")

    # 降级：内存计数器
    import threading
    if not hasattr(metric_cache_incr, "_counters"):
        metric_cache_incr._counters = {}
        metric_cache_incr._lock = threading.Lock()
    with metric_cache_incr._lock:
        val = metric_cache_incr._counters.get(key, 0) + 1
        metric_cache_incr._counters[key] = val
        return val


async def metric_cache_get(key: str) -> int:
    """获取计数器当前值"""
    full_key = _METRIC_KEY_PREFIX + key
    redis_client = await get_redis_client()

    if redis_client is not None:
        try:
            val = await redis_client.get(full_key)
            return int(val) if val else 0
        except Exception:
            pass

    import threading
    if hasattr(metric_cache_incr, "_counters"):
        with metric_cache_incr._lock:
            return metric_cache_incr._counters.get(key, 0)
    return 0
