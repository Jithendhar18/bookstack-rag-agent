"""Base service class providing common business logic utilities."""

from sqlalchemy.ext.asyncio import AsyncSession


class BaseService:
    """
    Base service class providing common utilities.
    
    Subclasses should implement domain-specific business logic.
    Services are framework-independent and handle all business logic.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize service with database session.
        
        Args:
            db: SQLAlchemy AsyncSession instance
        """
        self.db = db
