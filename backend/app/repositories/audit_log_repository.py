"""Audit log data repository."""

from typing import Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog
from app.repositories.base import BaseRepository


class AuditLogRepository(BaseRepository[AuditLog]):
    """Repository for AuditLog model."""

    model_class = AuditLog

    async def get_by_tenant(
        self, tenant_id: str, skip: int = 0, limit: int = 100
    ) -> list[AuditLog]:
        """
        Get audit logs for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of AuditLog instances
        """
        result = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.tenant_id == tenant_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_by_user(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> list[AuditLog]:
        """
        Get audit logs for a user.
        
        Args:
            user_id: User UUID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of AuditLog instances
        """
        result = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
