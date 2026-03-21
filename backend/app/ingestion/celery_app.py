"""Celery worker and task definitions for async ingestion."""

import logging
from celery import Celery
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "bookstack_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_retry_delay=30,
    task_max_retries=3,
    result_expires=3600,
)
