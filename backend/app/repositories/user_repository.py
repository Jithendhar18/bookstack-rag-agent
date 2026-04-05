"""User data repository - handles all user database operations."""

from typing import Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, Role
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """
    Repository for User model providing user-specific database operations.
    """

    model_class = User

    async def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            User instance or None if not found
        """
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: Email to search for
            
        Returns:
            User instance or None if not found
        """
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_username_or_email(self, username: str, email: str) -> Optional[User]:
        """
        Get user by either username or email (for registration checks).
        
        Args:
            username: Username to search for
            email: Email to search for
            
        Returns:
            User instance or None if not found
        """
        result = await self.db.execute(
            select(User).where(
                (User.username == username) | (User.email == email)
            )
        )
        return result.scalar_one_or_none()

    async def get_active_by_username(self, username: str) -> Optional[User]:
        """
        Get active user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            User instance or None if not found or not active
        """
        result = await self.db.execute(
            select(User).where(
                (User.username == username) & (User.is_active == True)
            )
        )
        return result.scalar_one_or_none()

    async def get_by_tenant_id(self, tenant_id: str) -> list[User]:
        """
        Get all users for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of User instances
        """
        result = await self.db.execute(
            select(User).where(User.tenant_id == tenant_id)
        )
        return result.scalars().all()

    async def get_active_by_identifier(self, identifier: str) -> Optional[User]:
        """
        Get active user by username OR email.
        
        Args:
            identifier: Username or email to search for
            
        Returns:
            User instance or None if not found or not active
        """
        result = await self.db.execute(
            select(User).where(
                ((User.username == identifier) | (User.email == identifier))
                & (User.is_active == True)
            )
        )
        return result.scalar_one_or_none()
