"""Ingestion API routes — Celery-based queue system."""

import uuid
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import Document, Chunk, AuditLog, AuditAction
from app.auth.dependencies import require_roles, CurrentUser
from app.ingestion.tasks import ingest_pages_task
from app.ingestion.celery_app import celery_app
from app.schemas.schemas import IngestRequest, IngestResponse, DocumentResponse, IngestionStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
    db: AsyncSession = Depends(get_db),
):
    """Trigger ingestion from BookStack via Celery queue. Requires admin or developer role."""
    # Send to Celery queue
    task = ingest_pages_task.apply_async(
        kwargs={
            "tenant_id": current_user.tenant_id,
            "page_ids": request.bookstack_ids,
            "force_reindex": request.force_reindex,
        }
    )

    task_id = task.id

    # Audit log
    db.add(AuditLog(
        id=uuid.uuid4(),
        user_id=current_user.user_id,
        action=AuditAction.INGEST,
        resource="ingestion",
        resource_id=task_id,
        details={"bookstack_type": request.bookstack_type, "ids": request.bookstack_ids},
        tenant_id=current_user.tenant_id,
    ))
    await db.commit()

    return IngestResponse(
        task_id=task_id,
        status="queued",
        documents_queued=len(request.bookstack_ids) if request.bookstack_ids else -1,
        message="Ingestion task queued via Celery",
    )


@router.get("/status/{task_id}", response_model=IngestionStatusResponse)
async def get_ingestion_status(
    task_id: str,
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
):
    """Check status of an ingestion task."""
    result = celery_app.AsyncResult(task_id)

    info = {}
    if result.info and isinstance(result.info, dict):
        info = result.info

    return IngestionStatusResponse(
        task_id=task_id,
        status=result.state,
        progress=info.get("status", None),
        result=info if result.ready() else None,
    )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
    db: AsyncSession = Depends(get_db),
):
    """List ingested documents."""
    query = select(Document).where(Document.tenant_id == current_user.tenant_id)
    if status:
        query = query.where(Document.status == status)

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    docs = result.scalars().all()

    responses = []
    for doc in docs:
        chunk_count_result = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
        )
        chunk_count = chunk_count_result.scalar() or 0

        responses.append(DocumentResponse(
            id=doc.id,
            bookstack_id=doc.bookstack_id,
            bookstack_type=doc.bookstack_type,
            title=doc.title,
            status=doc.status.value if doc.status else "unknown",
            chunk_count=chunk_count,
            ingested_at=doc.ingested_at,
            created_at=doc.created_at,
        ))

    return responses
