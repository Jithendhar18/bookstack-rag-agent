"""Ingestion service - business logic for data ingestion pipeline."""

import logging
from typing import Optional, List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from app.repositories.document_repository import DocumentRepository, ChunkRepository
from app.repositories.chat_repository import ChatSessionRepository
from app.services.base import BaseService
from app.schemas.schemas import (
    DocumentResponse,
    BookSummaryResponse,
    BookHierarchyResponse,
    ChapterGroupResponse,
    PageSummaryResponse,
)

logger = logging.getLogger(__name__)


class IngestionService(BaseService):
    """
    Ingestion service handling document ingestion, indexing, and synchronization.
    
    Contains business logic for ingestion workflows (delegates to pipeline for execution).
    """

    def __init__(self, db: AsyncSession):
        """Initialize ingestion service with repositories."""
        super().__init__(db)
        self.document_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)

    async def validate_ingestion_request(
        self, tenant_id: str, force_reindex: bool = False
    ) -> dict:
        """
        Validate ingestion request and gather pre-ingestion metrics.
        
        Args:
            tenant_id: Tenant identifier
            force_reindex: Whether to force reindex
            
        Returns:
            Dictionary with validation info and metrics
        """
        total_docs = await self.document_repo.count_by_tenant(tenant_id)

        return {
            "is_valid": True,
            "tenant_id": tenant_id,
            "force_reindex": force_reindex,
            "existing_documents": total_docs,
        }

    async def get_ingestion_status(self, tenant_id: str) -> dict:
        """
        Get current ingestion status for tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with tenant ingestion status
        """
        pending = await self.document_repo.count_by_status(tenant_id, "pending")
        processing = await self.document_repo.count_by_status(
            tenant_id, "processing"
        )
        completed = await self.document_repo.count_by_status(tenant_id, "completed")
        failed = await self.document_repo.count_by_status(tenant_id, "failed")

        return {
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "total": pending + processing + completed + failed,
        }

    async def get_documents(
        self, tenant_id: str, skip: int = 0, limit: int = 50
    ) -> list[Document]:
        """
        Get documents for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of Document instances
        """
        return await self.document_repo.get_by_tenant(tenant_id, skip, limit)

    async def mark_document_status(
        self, document_id: UUID, status: str
    ) -> Optional[Document]:
        """
        Update document processing status.
        
        Args:
            document_id: Document UUID
            status: New status (pending, processing, completed, failed)
            
        Returns:
            Updated Document instance or None if not found
        """
        return await self.document_repo.update_status(document_id, status)

    async def list_documents(
        self,
        tenant_id: str,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        book_id: Optional[int] = None,
    ) -> List[DocumentResponse]:
        """
        Get paginated documents with chunk counts.
        
        Args:
            tenant_id: Tenant identifier
            page: Page number (1-indexed)
            page_size: Documents per page
            status: Filter by status (optional)
            book_id: Filter by book ID (optional)
            
        Returns:
            List of DocumentResponse objects
        """
        skip = (page - 1) * page_size
        
        # Fetch documents
        docs = await self.document_repo.get_documents_paginated(
            tenant_id=tenant_id,
            skip=skip,
            limit=page_size,
            status=status,
            book_id=book_id,
        )
        
        # Batch fetch chunk counts
        doc_ids = [d.id for d in docs]
        chunk_counts = await self.chunk_repo.count_by_document_ids(doc_ids)
        
        # Build responses
        responses = []
        for doc in docs:
            responses.append(
                DocumentResponse(
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
                )
            )
        
        return responses

    async def list_books(self, tenant_id: str) -> List[BookSummaryResponse]:
        """
        Get all distinct books with page and chunk counts.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of BookSummaryResponse objects
        """
        # Get all books with page counts
        book_rows = await self.document_repo.get_books_with_counts(tenant_id)
        
        if not book_rows:
            return []
        
        book_ids = [row[0] for row in book_rows]
        
        # Batch fetch chunk counts per book
        chunk_by_book = await self.chunk_repo.count_by_book(tenant_id, book_ids)
        
        # Coalesce rows (same book_id may appear with different book_name due to NULL)
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

    async def get_book_hierarchy(
        self, tenant_id: str, book_id: int
    ) -> BookHierarchyResponse:
        """
        Get complete Book → Chapter → Page hierarchy for a book.
        
        Args:
            tenant_id: Tenant identifier
            book_id: BookStack book ID
            
        Returns:
            BookHierarchyResponse with full structure
            
        Raises:
            ValueError: If no pages found for the book
        """
        # Fetch all pages for the book
        docs = await self.document_repo.get_documents_by_book(tenant_id, book_id)
        
        if not docs:
            raise ValueError(f"No ingested pages found for book_id={book_id}")
        
        # Batch fetch chunk counts for all docs
        doc_ids = [d.id for d in docs]
        chunk_counts = await self.chunk_repo.count_by_document_ids(doc_ids)
        
        # Get total chunks for the book
        total_chunks = await self.chunk_repo.count_total_for_book(tenant_id, book_id)
        
        # Extract book name from first doc that has it
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
                    source_url=(doc.metadata_ or {}).get("source_url")
                    if hasattr(doc, "metadata_")
                    else None,
                    ingested_at=doc.ingested_at,
                    created_at=doc.created_at,
                )
            )
        
        # Sort chapters: un-chaptered (None) last, others by chapter_id
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
