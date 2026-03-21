"""Redis caching layer for query results, retrieval results, and embeddings."""

import json
import hashlib
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisCache:
    """Async Redis cache with typed key prefixes and TTL."""

    _instance: Optional["RedisCache"] = None
    _client: Optional[aioredis.Redis] = None

    PREFIX_QUERY = "rag:query:"
    PREFIX_RETRIEVAL = "rag:retrieval:"
    PREFIX_EMBEDDING = "rag:embedding:"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self):
        if self._client is None:
            self._client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                max_connections=20,
            )
            logger.info("Redis cache connected")

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client

    @staticmethod
    def _hash_key(*parts: str) -> str:
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ─── Query cache ─────────────────────────────────────────────────────

    async def get_query_result(self, query: str, tenant_id: str) -> Optional[dict]:
        if not settings.CACHE_ENABLED:
            return None
        key = f"{self.PREFIX_QUERY}{self._hash_key(query.lower().strip(), tenant_id)}"
        data = await self.client.get(key)
        if data:
            logger.debug(f"Query cache HIT: {query[:50]}")
            return json.loads(data)
        return None

    async def set_query_result(self, query: str, tenant_id: str, result: dict):
        if not settings.CACHE_ENABLED:
            return
        key = f"{self.PREFIX_QUERY}{self._hash_key(query.lower().strip(), tenant_id)}"
        await self.client.setex(key, settings.CACHE_QUERY_TTL, json.dumps(result, default=str))

    # ─── Retrieval cache ─────────────────────────────────────────────────

    async def get_retrieval_result(self, query: str, tenant_id: str, top_k: int) -> Optional[list]:
        if not settings.CACHE_ENABLED:
            return None
        key = f"{self.PREFIX_RETRIEVAL}{self._hash_key(query.lower().strip(), tenant_id, str(top_k))}"
        data = await self.client.get(key)
        if data:
            logger.debug(f"Retrieval cache HIT: {query[:50]}")
            return json.loads(data)
        return None

    async def set_retrieval_result(self, query: str, tenant_id: str, top_k: int, results: list):
        if not settings.CACHE_ENABLED:
            return
        key = f"{self.PREFIX_RETRIEVAL}{self._hash_key(query.lower().strip(), tenant_id, str(top_k))}"
        await self.client.setex(key, settings.CACHE_RETRIEVAL_TTL, json.dumps(results, default=str))

    # ─── Invalidation ────────────────────────────────────────────────────

    async def invalidate_tenant(self, tenant_id: str):
        """Invalidate all cached data for a tenant (called after ingestion)."""
        for prefix in [self.PREFIX_QUERY, self.PREFIX_RETRIEVAL]:
            async for key in self.client.scan_iter(match=f"{prefix}*"):
                await self.client.delete(key)
        logger.info(f"Cache invalidated for tenant: {tenant_id}")

    async def health_check(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False


_cache: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    global _cache
    if _cache is None:
        _cache = RedisCache()
        await _cache.connect()
    return _cache
