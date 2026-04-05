"""Health check API routes."""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    """
    Basic health check endpoint.
    
    Returns:
        Simple status response
    """
    return {"status": "ok"}


@router.get("/health/detailed")
async def health_detailed():
    """
    Detailed health check including subsystems.
    
    Checks status of cache, vector store, and other critical components.
    
    Returns:
        Dictionary with overall status and individual subsystem checks
    """
    checks = {"api": "ok"}

    # Cache
    try:
        from app.core.cache import get_cache

        cache = await get_cache()
        checks["cache"] = "ok" if await cache.health_check() else "unhealthy"
    except Exception:
        checks["cache"] = "unavailable"

    # Vector store
    try:
        from app.retrieval.vector_store import VectorStoreManager

        vs = VectorStoreManager()
        info = vs.client.get_collection(vs.collection_name)
        checks["vector_store"] = f"ok ({info.points_count} vectors)"
    except Exception:
        checks["vector_store"] = "unavailable"

    overall = (
        "ok"
        if all(v.startswith("ok") for v in checks.values())
        else "degraded"
    )
    return {"status": overall, "checks": checks}
