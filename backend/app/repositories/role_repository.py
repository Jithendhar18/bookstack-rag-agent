"""Role data repository - handles all role database operations."""

from typing import Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Role
from app.repositories.base import BaseRepository


class RoleRepository(BaseRepository[Role]):
    """
    Repository for Role model providing role-specific database operations.
    """

    model_class = Role

    async def get_by_name(self, name: str) -> Optional[Role]:
        """
        Get role by name.
        
        Args:
            name: Role name to search for
            
        Returns:
            Role instance or None if not found
        """
        result = await self.db.execute(
            select(Role).where(Role.name == name)
        )
        return result.scalar_one_or_none()

    async def get_all_roles(self) -> list[Role]:
        """
        Get all available roles.
        
        Returns:
            List of all Role instances
        """
        result = await self.db.execute(select(Role))
        return result.scalars().all()
