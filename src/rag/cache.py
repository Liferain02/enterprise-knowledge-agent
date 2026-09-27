"""Best-effort Redis cache; never serialize user/session state or cached Documents.

Keys hash complete inputs and model identity. Values are embeddings or rerank
indices/scores, so hits are always applied to the current authorized candidates.
Redis outages open a short circuit instead of stalling every chunk/request.
"""
import hashlib
import json
import threading
import time

from config.settings import get_settings


def cache_key(kind, identity):
    value = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "eka:v1:" + kind + ":" + hashlib.sha256(value.encode()).hexdigest()


class RedisCache:
    def __init__(self):
        self._client = None
        self._retry_after = 0.0
        self._lock = threading.Lock()
        self._flight_lock = threading.Lock()
        self._inflight = {}
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def _connection(self):
        if time.monotonic() < self._retry_after:
            return None
        settings = get_settings()
        if not getattr(settings, "rag_cache_enabled", False):
            return None
        if settings.redis_host in ("", "disabled"):
            return None
        with self._lock:
            if self._client is None:
                import redis
                self._client = redis.Redis(
                    host=settings.redis_host, port=settings.redis_port,
                    password=settings.redis_password or None, db=settings.redis_db,
                    socket_connect_timeout=0.3, socket_timeout=0.3,
                    decode_responses=True,
                )
        return self._client

    def _failed(self):
        self.errors += 1
        self._retry_after = time.monotonic() + 10

    def get(self, key):
        try:
            client = self._connection()
            raw = client.get(key) if client is not None else None
            if raw is not None:
                value = json.loads(raw)
                self.hits += 1
                return value
        except Exception:
            self._failed()
        self.misses += 1
        return None

    def get_or_set(self, key, loader, ttl):
        """Read-through cache with in-process single-flight protection.

        Concurrent cold requests for the same embedding/rerank input wait for
        one producer instead of issuing duplicate paid or CPU-heavy work. A
        failed producer wakes waiters; they may retry locally, so cache failure
        never becomes a correctness dependency.
        """
        hit = self.get(key)
        if hit is not None:
            return hit
        with self._flight_lock:
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False
        if not owner:
            event.wait(timeout=30.0)
            hit = self.get(key)
            return hit if hit is not None else loader()
        try:
            # Double-check after becoming the producer. Another caller may
            # have populated the value between the first read and lock claim.
            hit = self.get(key)
            if hit is not None:
                return hit
            value = loader()
            self.set(key, value, ttl)
            return value
        finally:
            with self._flight_lock:
                self._inflight.pop(key, None)
                event.set()

    def set(self, key, value, ttl):
        try:
            client = self._connection()
            if client is not None:
                client.setex(key, ttl, json.dumps(value, allow_nan=False))
        except Exception:
            self._failed()


cache = RedisCache()
