"""Admin service - business logic for admin operations."""

import logging
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import select, func, cast, Float
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, Document, Chunk, EmbeddingMetadata, AuditLog
from app.repositories.user_repository import UserRepository
from app.repositories.role_repository import RoleRepository
from app.repositories.document_repository import DocumentRepository, ChunkRepository
from app.repositories.audit_log_repository import AuditLogRepository
from app.services.base import BaseService

logger = logging.getLogger(__name__)


class AdminService(BaseService):
    """
    Admin service handling metrics, user management, system statistics.
    
    Contains business logic for admin operations.
    """

    def __init__(self, db: AsyncSession):
        """Initialize admin service with repositories."""
        super().__init__(db)
        self.user_repo = UserRepository(db)
        self.role_repo = RoleRepository(db)
        self.document_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.audit_log_repo = AuditLogRepository(db)

    async def get_system_metrics(self, tenant_id: str) -> dict:
        """
        Calculate and return system metrics for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with metrics (documents, chunks, users, queries, etc.)
        """
        # Count documents
        total_docs = await self.document_repo.count_by_tenant(tenant_id)

        # Count chunks
        total_chunks = await self.chunk_repo.count_by_tenant(tenant_id)

        # Count embeddings
        embedding_result = await self.db.execute(
            select(func.count(EmbeddingMetadata.id))
            .join(Chunk)
            .join(Document)
            .where(Document.tenant_id == tenant_id)
        )
        total_embeddings = embedding_result.scalar() or 0

        # Count users
        user_result = await self.db.execute(
            select(func.count(User.id)).where(User.tenant_id == tenant_id)
        )
        total_users = user_result.scalar() or 0

        # Count queries
        query_result = await self.db.execute(
            select(func.count(AuditLog.id)).where(
                (AuditLog.tenant_id == tenant_id) & (AuditLog.action == "query")
            )
        )
        total_queries = query_result.scalar() or 0

        # Documents by status
        status_result = await self.db.execute(
            select(Document.status, func.count(Document.id))
            .where(Document.tenant_id == tenant_id)
            .group_by(Document.status)
        )
        docs_by_status = {
            str(row[0]) if row[0] else "unknown": row[1] for row in status_result
        }

        # Documents grouped by book_id
        book_result = await self.db.execute(
            select(Document.book_id, func.count(Document.id))
            .where(
                (Document.tenant_id == tenant_id) & (Document.book_id.isnot(None))
            )
            .group_by(Document.book_id)
            .order_by(Document.book_id)
        )
        docs_by_book = {str(row[0]): row[1] for row in book_result}
        total_books = len(docs_by_book)

        # Average query latency
        try:
            avg_latency_result = await self.db.execute(
                select(func.avg(cast(AuditLog.details["latency_ms"].astext, Float)))
                .where(
                    (AuditLog.tenant_id == tenant_id)
                    & (AuditLog.action == "query")
                    & (AuditLog.details["latency_ms"].isnot(None))
                )
            )
            avg_latency = avg_latency_result.scalar()
        except Exception as e:
            logger.warning(f"Failed to calculate average latency: {e}")
            avg_latency = None

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "total_users": total_users,
            "total_queries": total_queries,
            "total_books": total_books,
            "documents_by_status": docs_by_status,
            "documents_by_book": docs_by_book,
            "avg_query_latency_ms": round(avg_latency, 1) if avg_latency else None,
        }

    async def list_users(
        self, tenant_id: str, skip: int = 0, limit: int = 100
    ) -> list[User]:
        """
        List users for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of User instances
        """
        result = await self.db.execute(
            select(User)
            .where(User.tenant_id == tenant_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def update_user(
        self,
        user_id: UUID,
        full_name: Optional[str] = None,
        is_active: Optional[bool] = None,
        role: Optional[str] = None,
        tenant_id: Optional[str] = None,
        updated_by: Optional[UUID] = None,
    ) -> User:
        """
        Update multiple user fields.
        
        Args:
            user_id: User UUID
            full_name: New full name (optional)
            is_active: New active status (optional)
            role: New role name (optional)
            tenant_id: User's tenant (for authorization)
            updated_by: User performing the update (for audit)
            
        Returns:
            Updated User instance
        """
        # Get user
        user = await self.user_repo.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        
        # Verify tenant authorization
        if tenant_id and user.tenant_id != tenant_id:
            raise PermissionError("Cannot update user from different tenant")
        
        # Build update data
        update_data = {}
        if full_name is not None:
            update_data["full_name"] = full_name
        if is_active is not None:
            update_data["is_active"] = is_active
        
        # Handle role update
        if role is not None:
            role_obj = await self.role_repo.get_by_name(role)
            if role_obj is None:
                raise ValueError(f"Role {role} not found")
            update_data["role_id"] = role_obj.id
        
        # Update user
        updated_user = await self.user_repo.update(user_id, update_data)
        
        # Log audit event
        if updated_by:
            audit_log = AuditLog(
                id=uuid4(),
                user_id=updated_by,
                action="update_user",
                resource="users",
                resource_id=str(user_id),
                tenant_id=tenant_id or user.tenant_id,
            )
            self.db.add(audit_log)
            await self.db.commit()
        
        return updated_user

    async def get_user_role(self, user_id: UUID) -> Optional[object]:
        """
        Get role for a user by user ID.
        
        Args:
            user_id: User UUID
            
        Returns:
            Role instance or None
        """
        user = await self.user_repo.get_by_id(user_id)
        if user is None:
            return None
        return await self.role_repo.get_by_id(user.role_id)
