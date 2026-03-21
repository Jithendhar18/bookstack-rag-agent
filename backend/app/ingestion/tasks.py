"""Celery tasks for ingestion pipeline."""

import asyncio
import logging
from typing import Optional, List

from app.ingestion.celery_app import celery_app
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _run_async(coro):
    """Run an async coroutine in a new event loop (for Celery sync workers)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    name="ingestion.ingest_pages",
    max_retries=3,
    default_retry_delay=30,
    track_started=True,
)
def ingest_pages_task(
    self,
    tenant_id: str = "default",
    page_ids: Optional[List[int]] = None,
    force_reindex: bool = False,
):
    """Celery task: run the ingestion pipeline asynchronously."""
    task_id = self.request.id
    logger.info(f"Ingestion task {task_id} started for tenant '{tenant_id}'")

    try:
        self.update_state(state="PROGRESS", meta={"status": "initializing"})

        stats = _run_async(
            _do_ingest(tenant_id, page_ids, force_reindex, self)
        )

        logger.info(f"Ingestion task {task_id} completed: {stats}")
        return {"status": "completed", "stats": stats}

    except Exception as exc:
        logger.error(f"Ingestion task {task_id} failed: {exc}")
        raise self.retry(exc=exc)


async def _do_ingest(
    tenant_id: str,
    page_ids: Optional[List[int]],
    force_reindex: bool,
    task,
):
    """Async ingestion logic called by Celery task."""
    from app.db.session import AsyncSessionLocal
    from app.ingestion.pipeline import IngestionPipeline

    async with AsyncSessionLocal() as db:
        pipeline = IngestionPipeline(db)
        stats = await pipeline.ingest_pages(
            tenant_id=tenant_id,
            page_ids=page_ids,
            force_reindex=force_reindex,
        )

    # Invalidate cache after ingestion
    try:
        from app.core.cache import get_cache
        cache = await get_cache()
        await cache.invalidate_tenant(tenant_id)
    except Exception as e:
        logger.warning(f"Cache invalidation failed: {e}")

    return stats


@celery_app.task(
    name="ingestion.health_check",
)
def ingestion_health_check():
    """Simple health check task."""
    return {"status": "ok"}
