"""Ingestion pipeline: orchestrates fetching, parsing, chunking, embedding, and storing."""

import logging
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Chunk, EmbeddingMetadata, DocumentStatus
from app.ingestion.bookstack_client import BookStackClient
from app.ingestion.content_parser import ContentParser
from app.ingestion.chunker import SemanticChunker
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.vector_store import VectorStoreManager
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionPipeline:
    """Full ingestion pipeline from BookStack to vector store."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.client = BookStackClient()
        self.parser = ContentParser()
        self.chunker = SemanticChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()

    async def ingest_pages(
        self,
        tenant_id: str = "default",
        page_ids: Optional[List[int]] = None,
        force_reindex: bool = False,
    ) -> dict:
        """Ingest pages from BookStack."""
        stats = {"processed": 0, "skipped": 0, "failed": 0, "chunks_created": 0}

        if page_ids:
            pages = [{"id": pid} for pid in page_ids]
        else:
            pages = await self.client.get_all_pages()

        for page_summary in pages:
            try:
                page_id = page_summary["id"]
                page = await self.client.get_page(page_id)

                # Check for existing document (dedup)
                html = page.get("html", "") or ""
                plain = self.parser.html_to_text(html)
                plain = self.parser.normalize_text(plain)
                content_hash = self.parser.compute_hash(plain)

                existing = await self._get_existing_document(page_id, "page", tenant_id)

                if existing and not force_reindex:
                    if existing.content_hash == content_hash:
                        stats["skipped"] += 1
                        continue
                    # Content changed – delete old chunks and re-ingest
                    await self._delete_document_data(existing)

                doc = await self._upsert_document(page, plain, html, content_hash, tenant_id, existing)

                # Chunk
                chunks_text = self.chunker.chunk_text(plain)
                if not chunks_text:
                    doc.status = DocumentStatus.COMPLETED
                    stats["processed"] += 1
                    continue

                # Create chunk records
                chunk_records = []
                for idx, chunk_text in enumerate(chunks_text):
                    chunk = Chunk(
                        id=uuid.uuid4(),
                        document_id=doc.id,
                        chunk_index=idx,
                        content=chunk_text,
                        content_hash=self.parser.compute_hash(chunk_text),
                        char_count=len(chunk_text),
                        metadata_={"page_id": page_id, "title": page.get("name", ""), "book_id": page.get("book_id")},
                    )
                    chunk_records.append(chunk)

                self.db.add_all(chunk_records)
                await self.db.flush()

                # Embed
                texts = [c.content for c in chunk_records]
                embeddings = self.embedding_service.embed_batch(texts)

                # Store in vector store
                ids = [str(c.id) for c in chunk_records]
                metadatas = [
                    {"document_id": str(doc.id), "chunk_index": c.chunk_index, "title": page.get("name", ""), "tenant_id": tenant_id}
                    for c in chunk_records
                ]
                self.vector_store.add_embeddings(ids, embeddings, metadatas, texts)

                # Create embedding metadata records
                for chunk_rec, emb_id in zip(chunk_records, ids):
                    emb_meta = EmbeddingMetadata(
                        id=uuid.uuid4(),
                        chunk_id=chunk_rec.id,
                        vector_store_id=emb_id,
                        model_name=settings.EMBEDDING_MODEL,
                        dimension=settings.EMBEDDING_DIMENSION,
                    )
                    self.db.add(emb_meta)

                doc.status = DocumentStatus.COMPLETED
                doc.ingested_at = datetime.utcnow()
                stats["processed"] += 1
                stats["chunks_created"] += len(chunk_records)

                await self.db.commit()
                logger.info(f"Ingested page {page_id}: {len(chunk_records)} chunks")

            except Exception as e:
                logger.error(f"Failed to ingest page {page_summary.get('id')}: {e}")
                stats["failed"] += 1
                await self.db.rollback()

        await self.client.close()
        self.vector_store.save()
        return stats

    async def _get_existing_document(self, bookstack_id: int, btype: str, tenant_id: str) -> Optional[Document]:
        result = await self.db.execute(
            select(Document).where(
                Document.bookstack_id == bookstack_id,
                Document.bookstack_type == btype,
                Document.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def _upsert_document(self, page: dict, plain: str, html: str, content_hash: str, tenant_id: str, existing: Optional[Document]) -> Document:
        if existing:
            existing.title = page.get("name", "")
            existing.slug = page.get("slug", "")
            existing.html_content = html
            existing.plain_content = plain
            existing.content_hash = content_hash
            existing.status = DocumentStatus.PROCESSING
            existing.updated_at = datetime.utcnow()
            return existing

        doc = Document(
            id=uuid.uuid4(),
            bookstack_id=page["id"],
            bookstack_type="page",
            title=page.get("name", ""),
            slug=page.get("slug", ""),
            book_id=page.get("book_id"),
            chapter_id=page.get("chapter_id"),
            content_hash=content_hash,
            html_content=html,
            plain_content=plain,
            status=DocumentStatus.PROCESSING,
            tenant_id=tenant_id,
        )
        self.db.add(doc)
        await self.db.flush()
        return doc

    async def _delete_document_data(self, doc: Document):
        """Delete chunks and vector store entries for a document."""
        result = await self.db.execute(
            select(Chunk).where(Chunk.document_id == doc.id)
        )
        chunks = result.scalars().all()
        chunk_ids = [str(c.id) for c in chunks]

        if chunk_ids:
            self.vector_store.delete_embeddings(chunk_ids)

        for chunk in chunks:
            await self.db.delete(chunk)
        await self.db.flush()
