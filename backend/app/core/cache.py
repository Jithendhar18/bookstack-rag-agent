"""In-memory caching layer for query results and retrieval results."""

import hashlib
import logging
import time
from typing import Optional

from cachetools import TTLCache

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class InMemoryCache:
    """Simple in-memory TTL cache replacing Redis."""

    _instance: Optional["InMemoryCache"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._query_cache = TTLCache(
                maxsize=1000, ttl=300,
            )
            cls._instance._retrieval_cache = TTLCache(
                maxsize=1000, ttl=300,
            )
            logger.info("In-memory cache initialized")
        return cls._instance

    @staticmethod
    def _hash_key(*parts: str) -> str:
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ─── Query cache ─────────────────────────────────────────────────────

    async def get_query_result(self, query: str, tenant_id: str) -> Optional[dict]:
        if not settings.CACHE_ENABLED:
            return None
        key = self._hash_key(query.lower().strip(), tenant_id)
        result = self._query_cache.get(key)
        if result:
            logger.debug(f"Query cache HIT: {query[:50]}")
        return result

    async def set_query_result(self, query: str, tenant_id: str, result: dict):
        if not settings.CACHE_ENABLED:
            return
        key = self._hash_key(query.lower().strip(), tenant_id)
        self._query_cache[key] = result

    # ─── Retrieval cache ─────────────────────────────────────────────────

    async def get_retrieval_result(self, query: str, tenant_id: str, top_k: int) -> Optional[list]:
        if not settings.CACHE_ENABLED:
            return None
        key = self._hash_key(query.lower().strip(), tenant_id, str(top_k))
        result = self._retrieval_cache.get(key)
        if result:
            logger.debug(f"Retrieval cache HIT: {query[:50]}")
        return result

    async def set_retrieval_result(self, query: str, tenant_id: str, top_k: int, results: list):
        if not settings.CACHE_ENABLED:
            return
        key = self._hash_key(query.lower().strip(), tenant_id, str(top_k))
        self._retrieval_cache[key] = results

    # ─── Invalidation ────────────────────────────────────────────────────

    async def invalidate_tenant(self, tenant_id: str):
        """Clear all cached data."""
        self._query_cache.clear()
        self._retrieval_cache.clear()
        logger.info(f"Cache invalidated for tenant: {tenant_id}")

    async def health_check(self) -> bool:
        return True

    async def close(self):
        pass


_cache: Optional[InMemoryCache] = None


async def get_cache() -> InMemoryCache:
    global _cache
    if _cache is None:
        _cache = InMemoryCache()
    return _cache
