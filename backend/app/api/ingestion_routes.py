"""Ingestion API routes — direct async execution."""

import uuid
import logging
import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db, AsyncSessionLocal
from app.db.models import Document, Chunk, AuditLog
from app.auth.dependencies import require_roles, CurrentUser
from app.schemas.schemas import (
    IngestRequest, IngestResponse, DocumentResponse, IngestionStatusResponse,
    BookSummaryResponse, BookHierarchyResponse, ChapterGroupResponse, PageSummaryResponse,
)
from config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

# In-memory task tracking for background ingestion
_ingestion_tasks: dict[str, dict] = {}


async def _run_ingestion_background(
    task_id: str,
    tenant_id: str,
    page_ids: Optional[List[int]],
    force_reindex: bool,
):
    """Run ingestion pipeline in the background."""
    _ingestion_tasks[task_id] = {"status": "PROGRESS", "progress": "initializing", "result": None}
    try:
        from app.ingestion.pipeline import IngestionPipeline

        async with AsyncSessionLocal() as db:
            pipeline = IngestionPipeline(db)
            stats = await pipeline.ingest_pages(
                tenant_id=tenant_id,
                page_ids=page_ids,
                force_reindex=force_reindex,
                task_id=task_id,
            )

        # Invalidate cache after ingestion
        try:
            from app.core.cache import get_cache
            cache = await get_cache()
            await cache.invalidate_tenant(tenant_id)
        except Exception as e:
            logger.warning(f"Cache invalidation failed (non-fatal): {e}")

        _ingestion_tasks[task_id] = {
            "status": "SUCCESS",
            "progress": "completed",
            "result": {"status": "completed", "stats": stats},
        }
        logger.info("Background ingestion completed", extra={
            "task_id": task_id, "stats": stats,
        })
    except Exception as exc:
        logger.error(f"Background ingestion failed: {exc}", exc_info=True)
        _ingestion_tasks[task_id] = {
            "status": "FAILURE",
            "progress": "failed",
            "result": {"status": "failed", "error": str(exc)},
        }


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
    db: AsyncSession = Depends(get_db),
):
    """Trigger ingestion from BookStack."""
    task_id = str(uuid.uuid4())

    logger.info("Ingestion request received", extra={
        "stage": "api",
        "task_id": task_id,
        "tenant_id": current_user.tenant_id,
        "user_id": str(current_user.user_id),
        "bookstack_type": request.bookstack_type,
        "page_ids": request.bookstack_ids,
        "force_reindex": request.force_reindex,
    })

    # Launch ingestion in the background
    background_tasks.add_task(
        _run_ingestion_background,
        task_id=task_id,
        tenant_id=current_user.tenant_id,
        page_ids=request.bookstack_ids,
        force_reindex=request.force_reindex,
    )

    # Audit log
    db.add(AuditLog(
        id=uuid.uuid4(),
        user_id=current_user.user_id,
        action="ingest",
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
        message="Ingestion task started",
    )


@router.get("/status/{task_id}", response_model=IngestionStatusResponse)
async def get_ingestion_status(
    task_id: str,
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
):
    """Check status of an ingestion task."""
    task_info = _ingestion_tasks.get(task_id)
    if not task_info:
        return IngestionStatusResponse(
            task_id=task_id,
            status="PENDING",
            progress=None,
            result=None,
        )

    return IngestionStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info.get("progress"),
        result=task_info.get("result") if task_info["status"] in ("SUCCESS", "FAILURE") else None,
    )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    book_id: int | None = None,
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
    db: AsyncSession = Depends(get_db),
):
    """List ingested documents, optionally filtered by book_id and/or status.

    Results are ordered book → chapter → title for a predictable hierarchical view.
    """
    query = select(Document).where(Document.tenant_id == current_user.tenant_id)
    if status:
        query = query.where(Document.status == status)
    if book_id is not None:
        query = query.where(Document.book_id == book_id)

    # Hierarchical ordering: book first, then chapter, then page title
    query = query.order_by(Document.book_id, Document.chapter_id, Document.title)
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    docs = result.scalars().all()

    # Batch chunk counts
    doc_ids = [d.id for d in docs]
    chunk_counts: dict = {}
    if doc_ids:
        cc_result = await db.execute(
            select(Chunk.document_id, func.count(Chunk.id).label("cnt"))
            .where(Chunk.document_id.in_(doc_ids))
            .group_by(Chunk.document_id)
        )
        chunk_counts = {str(row[0]): row[1] for row in cc_result.all()}

    responses = []
    for doc in docs:
        responses.append(DocumentResponse(
            id=doc.id,
            bookstack_id=doc.bookstack_id,
            bookstack_type=doc.bookstack_type,
            title=doc.title,
            status=doc.status or "unknown",
            chunk_count=chunk_counts.get(str(doc.id), 0),
            book_id=doc.book_id,
            book_name=doc.book_name,
            chapter_id=doc.chapter_id,
            chapter_name=doc.chapter_name,
            ingested_at=doc.ingested_at,
            created_at=doc.created_at,
        ))

    return responses


@router.get("/books", response_model=List[BookSummaryResponse])
async def list_books(
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
    db: AsyncSession = Depends(get_db),
):
    """List all distinct books that have ingested pages, with page and chunk counts."""
    tenant = current_user.tenant_id

    # Page counts per book
    page_result = await db.execute(
        select(Document.book_id, Document.book_name, func.count(Document.id).label("page_count"))
        .where(
            Document.tenant_id == tenant,
            Document.bookstack_type == "page",
            Document.book_id.isnot(None),
        )
        .group_by(Document.book_id, Document.book_name)
        .order_by(Document.book_id)
    )
    book_rows = page_result.all()

    if not book_rows:
        return []

    book_ids = [row[0] for row in book_rows]

    # Chunk counts per book — single batched query
    chunk_result = await db.execute(
        select(Document.book_id, func.count(Chunk.id).label("chunk_count"))
        .join(Chunk, Chunk.document_id == Document.id)
        .where(Document.tenant_id == tenant, Document.book_id.in_(book_ids))
        .group_by(Document.book_id)
    )
    chunk_by_book = {row[0]: row[1] for row in chunk_result.all()}

    # Coalesce duplicate book_id rows (same book_id, different book_name due to NULL)
    seen: dict[int, BookSummaryResponse] = {}
    for book_id, book_name, page_count in book_rows:
        if book_id in seen:
            seen[book_id].page_count += page_count
        else:
            seen[book_id] = BookSummaryResponse(
                book_id=book_id,
                book_name=book_name,
                page_count=page_count,
                chunk_count=chunk_by_book.get(book_id, 0),
            )

    return list(seen.values())


@router.get("/books/{book_id}", response_model=BookHierarchyResponse)
async def get_book_hierarchy(
    book_id: int,
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
    db: AsyncSession = Depends(get_db),
):
    """Return a full Book → Chapter → Page hierarchy for the requested book_id."""
    tenant = current_user.tenant_id

    docs_result = await db.execute(
        select(Document)
        .where(
            Document.tenant_id == tenant,
            Document.book_id == book_id,
            Document.bookstack_type == "page",
        )
        .order_by(Document.chapter_id, Document.title)
    )
    docs = docs_result.scalars().all()

    if not docs:
        raise HTTPException(status_code=404, detail=f"No ingested pages found for book_id={book_id}")

    # Batch chunk counts for all docs in this book
    doc_ids = [d.id for d in docs]
    cc_result = await db.execute(
        select(Chunk.document_id, func.count(Chunk.id).label("cnt"))
        .where(Chunk.document_id.in_(doc_ids))
        .group_by(Chunk.document_id)
    )
    chunk_counts = {str(row[0]): row[1] for row in cc_result.all()}

    # Total chunks for this book
    total_chunks_result = await db.execute(
        select(func.count(Chunk.id))
        .join(Document)
        .where(Document.tenant_id == tenant, Document.book_id == book_id)
    )
    total_chunks = total_chunks_result.scalar() or 0

    # Resolve book_name from the first doc that has it
    book_name = next((d.book_name for d in docs if d.book_name), None)

    # Group pages by chapter
    chapter_groups: dict[Optional[int], dict] = {}
    for doc in docs:
        ch_id = doc.chapter_id
        if ch_id not in chapter_groups:
            chapter_groups[ch_id] = {"chapter_name": doc.chapter_name, "pages": []}
        chapter_groups[ch_id]["pages"].append(
            PageSummaryResponse(
                id=doc.id,
                bookstack_id=doc.bookstack_id,
                title=doc.title,
                slug=doc.slug,
                chapter_id=doc.chapter_id,
                chapter_name=doc.chapter_name,
                status=doc.status or "unknown",
                chunk_count=chunk_counts.get(str(doc.id), 0),
                source_url=(doc.metadata_ or {}).get("source_url"),
                ingested_at=doc.ingested_at,
                created_at=doc.created_at,
            )
        )

    # Sort: un-chaptered pages (None) last; chapters ordered by chapter_id
    ordered_chapters = sorted(
        chapter_groups.items(),
        key=lambda x: (x[0] is None, x[0] or 0),
    )
    chapters = [
        ChapterGroupResponse(
            chapter_id=ch_id,
            chapter_name=ch_data["chapter_name"],
            page_count=len(ch_data["pages"]),
            pages=ch_data["pages"],
        )
        for ch_id, ch_data in ordered_chapters
    ]

    return BookHierarchyResponse(
        book_id=book_id,
        book_name=book_name,
        total_pages=len(docs),
        total_chunks=total_chunks,
        chapters=chapters,
    )
