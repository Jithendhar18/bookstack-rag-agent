"""Base repository class with common CRUD operations."""

from typing import TypeVar, Generic, Type, Optional, List, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import DeclarativeBase

T = TypeVar("T", bound=DeclarativeBase)


class BaseRepository(Generic[T]):
    """
    Generic base repository providing common CRUD operations.
    
    Subclasses should set model_class to the SQLAlchemy model.
    """

    model_class: Type[T]

    def __init__(self, db: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            db: SQLAlchemy AsyncSession instance
        """
        self.db = db

    async def create(self, obj: T) -> T:
        """
        Create and persist a new object.
        
        Args:
            obj: Model instance to create
            
        Returns:
            Created model instance
        """
        self.db.add(obj)
        await self.db.commit()
        await self.db.refresh(obj)
        return obj

    async def get_by_id(self, obj_id: UUID) -> Optional[T]:
        """
        Get object by ID.
        
        Args:
            obj_id: UUID of the object
            
        Returns:
            Model instance or None if not found
        """
        result = await self.db.execute(
            select(self.model_class).where(self.model_class.id == obj_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """
        Get all objects with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of model instances
        """
        result = await self.db.execute(
            select(self.model_class).offset(skip).limit(limit)
        )
        return result.scalars().all()

    async def update(self, obj_id: UUID, data: dict[str, Any]) -> Optional[T]:
        """
        Update an object.
        
        Args:
            obj_id: UUID of the object to update
            data: Dictionary of fields to update
            
        Returns:
            Updated model instance or None if not found
        """
        stmt = (
            update(self.model_class)
            .where(self.model_class.id == obj_id)
            .values(**data)
            .returning(self.model_class)
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.scalar_one_or_none()

    async def delete(self, obj_id: UUID) -> bool:
        """
        Delete an object.
        
        Args:
            obj_id: UUID of the object to delete
            
        Returns:
            True if deleted, False if not found
        """
        stmt = delete(self.model_class).where(self.model_class.id == obj_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount > 0
