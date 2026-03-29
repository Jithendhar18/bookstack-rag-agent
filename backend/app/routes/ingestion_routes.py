"""Ingestion API routes - HTTP handlers with dependency injection."""

import uuid
import logging
from typing import Optional, List, Annotated

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db, AsyncSessionLocal
from app.auth.dependencies import require_roles, CurrentUser
from app.services.ingestion_service import IngestionService
from app.schemas.schemas import (
    IngestRequest,
    IngestResponse,
    IngestionStatusResponse,
    DocumentResponse,
    BookSummaryResponse,
    BookHierarchyResponse,
)
from app.ingestion.pipeline import IngestionPipeline
from app.core.cache import get_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

# In-memory task tracking for background ingestion
_ingestion_tasks: dict[str, dict] = {}


async def get_ingestion_service(
    db: AsyncSession = Depends(get_db),
) -> IngestionService:
    """Dependency injection for IngestionService."""
    return IngestionService(db)


async def _run_ingestion_background(
    task_id: str,
    tenant_id: str,
    bookstack_type: str,
    page_ids: Optional[List[int]],
    force_reindex: bool,
):
    """
    Run ingestion pipeline in the background.
    
    Args:
        task_id: Unique task identifier
        tenant_id: Tenant identifier
        bookstack_type: Type of BookStack content (pages, books, chapters, shelves)
        page_ids: Specific items to ingest (None = all)
        force_reindex: Whether to force reindex existing documents
    """
    _ingestion_tasks[task_id] = {
        "status": "PROGRESS",
        "progress": "initializing",
        "result": None,
    }
    try:
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
            cache = await get_cache()
            await cache.invalidate_tenant(tenant_id)
        except Exception as e:
            logger.warning(
                f"Cache invalidation failed (non-fatal): {e}"
            )

        _ingestion_tasks[task_id] = {
            "status": "SUCCESS",
            "progress": "completed",
            "result": {"status": "completed", "stats": stats},
        }
        logger.info(
            "Background ingestion completed",
            extra={"task_id": task_id, "stats": stats},
        )
    except Exception as exc:
        logger.error(
            f"Background ingestion failed: {exc}",
            exc_info=True,
        )
        _ingestion_tasks[task_id] = {
            "status": "FAILURE",
            "progress": "failed",
            "result": {"status": "failed", "error": str(exc)},
        }


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[
        CurrentUser, Depends(require_roles(["admin", "developer"]))
    ],
    ingestion_service: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """
    Trigger ingestion from BookStack.
    
    Validates the ingestion request and queues the ingestion task to run
    in the background. Returns a task_id to track progress.
    
    Args:
        request: IngestRequest with BookStack type and optional item IDs
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user (admin or developer)
        ingestion_service: Injected ingestion service
        
    Returns:
        IngestResponse with task_id and initial status
    """
    task_id = str(uuid.uuid4())

    logger.info(
        "Ingestion request received",
        extra={
            "stage": "api",
            "task_id": task_id,
            "tenant_id": current_user.tenant_id,
            "user_id": str(current_user.user_id),
            "bookstack_type": request.bookstack_type,
            "page_ids": request.bookstack_ids,
            "force_reindex": request.force_reindex,
        },
    )

    # Validate request
    validation = await ingestion_service.validate_ingestion_request(
        current_user.tenant_id,
        request.force_reindex,
    )

    if not validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ingestion request",
        )

    # Queue background task
    background_tasks.add_task(
        _run_ingestion_background,
        task_id=task_id,
        tenant_id=current_user.tenant_id,
        bookstack_type=request.bookstack_type,
        page_ids=request.bookstack_ids,
        force_reindex=request.force_reindex,
    )

    return IngestResponse(
        task_id=task_id,
        status="queued",
        documents_queued=0,
    )


@router.get("/status/{task_id}")
async def get_ingestion_status(task_id: str):
    """
    Get the status of an ingestion task.
    
    Args:
        task_id: Ingestion task ID
        
    Returns:
        Dictionary with task status and progress
    """
    if task_id not in _ingestion_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    return _ingestion_tasks[task_id]


@router.get("/status", response_model=IngestionStatusResponse)
async def get_ingestion_tenant_status(
    current_user: Annotated[CurrentUser, Depends(require_roles(["admin"]))],
    ingestion_service: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """
    Get ingestion status for the current tenant.
    
    Returns counts of documents by status (pending, processing, completed, failed).
    Admin access required.
    
    Args:
        current_user: Current authenticated user
        ingestion_service: Injected ingestion service
        
    Returns:
        IngestionStatusResponse with status counts
    """
    status_info = await ingestion_service.get_ingestion_status(
        current_user.tenant_id
    )

    return IngestionStatusResponse(**status_info)


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    current_user: Annotated[CurrentUser, Depends(require_roles(["admin", "developer"]))],
    ingestion_service: Annotated[IngestionService, Depends(get_ingestion_service)],
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    book_id: Optional[int] = None,
):
    """
    List ingested documents with pagination and filtering.
    
    Results ordered by book → chapter → title.
    
    Args:
        current_user: Current authenticated user
        ingestion_service: Injected ingestion service
        page: Page number (1-indexed)
        page_size: Documents per page
        status: Filter by document status (optional)
        book_id: Filter by book ID (optional)
        
    Returns:
        List of DocumentResponse objects
    """
    return await ingestion_service.list_documents(
        tenant_id=current_user.tenant_id,
        page=page,
        page_size=page_size,
        status=status,
        book_id=book_id,
    )


@router.get("/books", response_model=List[BookSummaryResponse])
async def list_books(
    current_user: Annotated[CurrentUser, Depends(require_roles(["admin", "developer"]))],
    ingestion_service: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """
    List all distinct books with ingested pages and their page/chunk counts.
    
    Args:
        current_user: Current authenticated user
        ingestion_service: Injected ingestion service
        
    Returns:
        List of BookSummaryResponse objects
    """
    return await ingestion_service.list_books(current_user.tenant_id)


@router.get("/books/{book_id}", response_model=BookHierarchyResponse)
async def get_book_hierarchy(
    book_id: int,
    current_user: Annotated[CurrentUser, Depends(require_roles(["admin", "developer"]))],
    ingestion_service: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """
    Get full Book → Chapter → Page hierarchy for a book.
    
    Args:
        book_id: BookStack book ID
        current_user: Current authenticated user
        ingestion_service: Injected ingestion service
        
    Returns:
        BookHierarchyResponse with complete book structure
        
    Raises:
        HTTPException: 404 if no ingested pages found for book
    """
    try:
        return await ingestion_service.get_book_hierarchy(
            tenant_id=current_user.tenant_id,
            book_id=book_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
