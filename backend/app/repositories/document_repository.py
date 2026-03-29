"""Document and Chunk data repositories."""

from typing import Optional, List
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Chunk
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document model providing document-specific operations."""

    model_class = Document

    async def get_by_bookstack_id(
        self, bookstack_id: int, bookstack_type: str, tenant_id: str
    ) -> Optional[Document]:
        """
        Get document by BookStack ID and type.
        
        Args:
            bookstack_id: BookStack document ID
            bookstack_type: BookStack content type (pages, books, chapters, etc.)
            tenant_id: Tenant identifier
            
        Returns:
            Document instance or None if not found
        """
        result = await self.db.execute(
            select(Document).where(
                (Document.bookstack_id == bookstack_id)
                & (Document.bookstack_type == bookstack_type)
                & (Document.tenant_id == tenant_id)
            )
        )
        return result.scalar_one_or_none()

    async def get_by_tenant(
        self, tenant_id: str, skip: int = 0, limit: int = 100
    ) -> List[Document]:
        """Get documents for a tenant with pagination."""
        result = await self.db.execute(
            select(Document)
            .where(Document.tenant_id == tenant_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def count_by_tenant(self, tenant_id: str) -> int:
        """Count total unique books for a tenant (distinct book_id)."""
        result = await self.db.execute(
            select(func.count(func.distinct(Document.book_id))).where(
                (Document.tenant_id == tenant_id) & (Document.book_id.isnot(None))
            )
        )
        return result.scalar() or 0

    async def count_by_status(self, tenant_id: str, status: str) -> int:
        """Count documents by status for a tenant."""
        result = await self.db.execute(
            select(func.count(Document.id)).where(
                (Document.tenant_id == tenant_id) & (Document.status == status)
            )
        )
        return result.scalar() or 0

    async def update_status(self, doc_id: UUID, status: str) -> Optional[Document]:
        """Update document status."""
        return await self.update(doc_id, {"status": status})

    async def get_documents_paginated(
        self,
        tenant_id: str,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None,
        book_id: Optional[int] = None,
    ) -> List[Document]:
        """
        Get paginated documents for a tenant with optional filters.
        Ordered by book → chapter → title.
        
        Args:
            tenant_id: Tenant identifier
            skip: Records to skip
            limit: Maximum records to return
            status: Filter by status (optional)
            book_id: Filter by book ID (optional)
            
        Returns:
            List of Document instances
        """
        query = select(Document).where(Document.tenant_id == tenant_id)
        
        if status:
            query = query.where(Document.status == status)
        if book_id is not None:
            query = query.where(Document.book_id == book_id)
        
        query = query.order_by(
            Document.book_id, Document.chapter_id, Document.title
        )
        query = query.offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_books_with_counts(self, tenant_id: str) -> List[tuple]:
        """
        Get distinct books with page counts.
        
        Returns list of tuples: (book_id, book_name, page_count)
        """
        result = await self.db.execute(
            select(
                Document.book_id,
                Document.book_name,
                func.count(Document.id).label("page_count"),
            )
            .where(
                (Document.tenant_id == tenant_id)
                & (Document.bookstack_type == "page")
                & (Document.book_id.isnot(None))
            )
            .group_by(Document.book_id, Document.book_name)
            .order_by(Document.book_id)
        )
        return result.all()

    async def get_documents_by_book(
        self, tenant_id: str, book_id: int
    ) -> List[Document]:
        """
        Get all pages for a book, ordered by chapter then title.
        
        Args:
            tenant_id: Tenant identifier
            book_id: BookStack book ID
            
        Returns:
            List of Document instances for the book
        """
        result = await self.db.execute(
            select(Document)
            .where(
                (Document.tenant_id == tenant_id)
                & (Document.book_id == book_id)
                & (Document.bookstack_type == "page")
            )
            .order_by(Document.chapter_id, Document.title)
        )
        return result.scalars().all()


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for Chunk model providing chunk-specific operations."""

    model_class = Chunk

    async def get_by_document(self, document_id: UUID) -> List[Chunk]:
        """Get all chunks for a document."""
        result = await self.db.execute(
            select(Chunk).where(Chunk.document_id == document_id)
        )
        return result.scalars().all()

    async def count_by_tenant(self, tenant_id: str) -> int:
        """Count total chunks for a tenant."""
        result = await self.db.execute(
            select(func.count(Chunk.id))
            .join(Document)
            .where(Document.tenant_id == tenant_id)
        )
        return result.scalar() or 0

    async def count_by_document_ids(self, document_ids: List[UUID]) -> dict[str, int]:
        """
        Get chunk counts by document ID.
        
        Args:
            document_ids: List of document IDs
            
        Returns:
            Dictionary mapping document_id (str) -> chunk count
        """
        if not document_ids:
            return {}
        
        result = await self.db.execute(
            select(Chunk.document_id, func.count(Chunk.id).label("cnt"))
            .where(Chunk.document_id.in_(document_ids))
            .group_by(Chunk.document_id)
        )
        return {str(row[0]): row[1] for row in result.all()}

    async def count_by_book(self, tenant_id: str, book_ids: List[int]) -> dict[int, int]:
        """
        Get total chunk counts per book.
        
        Args:
            tenant_id: Tenant identifier
            book_ids: List of book IDs
            
        Returns:
            Dictionary mapping book_id -> total chunk count
        """
        if not book_ids:
            return {}
        
        result = await self.db.execute(
            select(Document.book_id, func.count(Chunk.id).label("chunk_count"))
            .join(Chunk, Chunk.document_id == Document.id)
            .where(
                (Document.tenant_id == tenant_id)
                & (Document.book_id.in_(book_ids))
            )
            .group_by(Document.book_id)
        )
        return {row[0]: row[1] for row in result.all()}

    async def count_total_for_book(self, tenant_id: str, book_id: int) -> int:
        """
        Get total chunk count for a specific book.
        
        Args:
            tenant_id: Tenant identifier
            book_id: BookStack book ID
            
        Returns:
            Total chunk count for the book
        """
        result = await self.db.execute(
            select(func.count(Chunk.id))
            .join(Document)
            .where(
                (Document.tenant_id == tenant_id)
                & (Document.book_id == book_id)
            )
        )
        return result.scalar() or 0
