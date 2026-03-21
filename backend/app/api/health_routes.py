"""Health check routes."""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/detailed")
async def health_detailed():
    """Detailed health check including subsystems."""
    checks = {"api": "ok"}

    # Redis
    try:
        from app.core.cache import get_cache
        cache = await get_cache()
        checks["redis"] = "ok" if await cache.health_check() else "unhealthy"
    except Exception:
        checks["redis"] = "unavailable"

    # Vector store
    try:
        from app.retrieval.vector_store import VectorStoreManager
        vs = VectorStoreManager()
        checks["vector_store"] = f"ok ({vs.store_type}, {vs.count} vectors)"
    except Exception:
        checks["vector_store"] = "unavailable"

    overall = "ok" if all(v.startswith("ok") for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
