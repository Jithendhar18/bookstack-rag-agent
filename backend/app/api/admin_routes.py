"""Admin API routes — metrics, user management."""

import uuid
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import User, Role, Document, Chunk, EmbeddingMetadata, AuditLog
from app.auth.dependencies import require_roles, CurrentUser
from app.schemas.schemas import SystemMetrics, UserResponse, UserUpdateRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics(
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    db: AsyncSession = Depends(get_db),
):
    """Get system metrics. Admin only."""
    tenant = current_user.tenant_id

    total_docs = (await db.execute(
        select(func.count(Document.id)).where(Document.tenant_id == tenant)
    )).scalar() or 0

    total_chunks = (await db.execute(
        select(func.count(Chunk.id))
        .join(Document)
        .where(Document.tenant_id == tenant)
    )).scalar() or 0

    total_embeddings = (await db.execute(
        select(func.count(EmbeddingMetadata.id))
        .join(Chunk)
        .join(Document)
        .where(Document.tenant_id == tenant)
    )).scalar() or 0

    total_users = (await db.execute(
        select(func.count(User.id)).where(User.tenant_id == tenant)
    )).scalar() or 0

    total_queries = (await db.execute(
        select(func.count(AuditLog.id)).where(
            AuditLog.tenant_id == tenant,
            AuditLog.action == "query",
        )
    )).scalar() or 0

    # Documents by status
    status_result = await db.execute(
        select(Document.status, func.count(Document.id))
        .where(Document.tenant_id == tenant)
        .group_by(Document.status)
    )
    docs_by_status = {str(row[0]) if row[0] else "unknown": row[1] for row in status_result}

    # Documents grouped by book_id
    book_result = await db.execute(
        select(Document.book_id, func.count(Document.id))
        .where(Document.tenant_id == tenant, Document.book_id.isnot(None))
        .group_by(Document.book_id)
        .order_by(Document.book_id)
    )
    docs_by_book = {str(row[0]): row[1] for row in book_result}
    total_books = len(docs_by_book)

    return SystemMetrics(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_embeddings=total_embeddings,
        total_users=total_users,
        total_queries=total_queries,
        total_books=total_books,
        documents_by_status=docs_by_status,
        documents_by_book=docs_by_book,
        avg_query_latency_ms=None,
    )


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    page: int = 1,
    page_size: int = 20,
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    db: AsyncSession = Depends(get_db),
):
    """List users in the tenant. Admin only."""
    result = await db.execute(
        select(User, Role.name)
        .join(Role)
        .where(User.tenant_id == current_user.tenant_id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = result.all()

    return [
        UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            role=role_name.value,
            tenant_id=user.tenant_id,
            created_at=user.created_at,
        )
        for user, role_name in rows
    ]


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    db: AsyncSession = Depends(get_db),
):
    """Update user details. Admin only."""
    result = await db.execute(
        select(User).where(User.id == user_id, User.tenant_id == current_user.tenant_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if request.full_name is not None:
        user.full_name = request.full_name
    if request.is_active is not None:
        user.is_active = request.is_active
    if request.role is not None:
        role_result = await db.execute(select(Role).where(Role.name == request.role))
        role = role_result.scalar_one_or_none()
        if role is None:
            raise HTTPException(status_code=400, detail="Invalid role")
        user.role_id = role.id

    # Audit
    db.add(AuditLog(
        id=uuid.uuid4(),
        user_id=current_user.user_id,
        action="update_user",
        resource="users",
        resource_id=str(user_id),
        tenant_id=current_user.tenant_id,
    ))

    await db.commit()

    role_result = await db.execute(select(Role).where(Role.id == user.role_id))
    role = role_result.scalar_one()

    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        is_active=user.is_active,
        role=role.name,
        tenant_id=user.tenant_id,
        created_at=user.created_at,
    )


@router.get("/cache/health")
async def cache_health(
    current_user: CurrentUser = Depends(require_roles(["admin"])),
):
    """Check cache health."""
    from app.core.cache import get_cache
    try:
        cache = await get_cache()
        healthy = await cache.health_check()
        return {"status": "ok" if healthy else "unhealthy"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
