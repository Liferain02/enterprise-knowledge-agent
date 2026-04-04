"""
缓存层单元测试
"""
import pytest


class TestResponseCache:
    @pytest.mark.asyncio
    async def test_llm_cache_set_and_get(self):
        from src.rag.cache.response_cache import llm_cache_set, llm_cache_get
        await llm_cache_set(
            query="什么是年假",
            response="年假是...",
            model="qwen3.5-flash",
            temperature=0.0,
        )
        cached = await llm_cache_get(
            query="什么是年假",
            model="qwen3.5-flash",
            temperature=0.0,
        )
        assert cached is not None
        assert cached["response"] == "年假是..."

    @pytest.mark.asyncio
    async def test_llm_cache_not_stored_high_temperature(self):
        from src.rag.cache.response_cache import llm_cache_set, llm_cache_get
        await llm_cache_set(
            query="写一首诗",
            response="诗的内容...",
            model="qwen3.5-flash",
            temperature=0.9,
        )
        cached = await llm_cache_get(
            query="写一首诗",
            model="qwen3.5-flash",
            temperature=0.9,
        )
        assert cached is None

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        from src.rag.cache.response_cache import cache_stats
        stats = await cache_stats()
        assert "redis_available" in stats
        assert "stats" in stats
        assert "hit" in stats["stats"]
        assert "miss" in stats["stats"]
