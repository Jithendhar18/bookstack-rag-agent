"""Ingestion pipeline: orchestrates fetching, parsing, chunking, embedding, and storing."""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Chunk, EmbeddingMetadata
from app.ingestion.bookstack_client import BookStackClient
from app.ingestion.content_parser import ContentParser
from app.ingestion.chunker import SemanticChunker
from app.providers.factory import get_embedding
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
        self.embedding_service = get_embedding()
        self.vector_store = VectorStoreManager()

    async def ingest_pages(
        self,
        tenant_id: str = "default",
        page_ids: Optional[List[int]] = None,
        force_reindex: bool = False,
        task_id: str = "",
    ) -> dict:
        """Ingest pages from BookStack."""
        job_id = task_id or str(uuid.uuid4())
        stats = {"processed": 0, "skipped": 0, "failed": 0, "chunks_created": 0}
        pipeline_start = time.monotonic()

        _ctx = {"job_id": job_id, "task_id": task_id, "tenant_id": tenant_id}

        logger.info("Pipeline started", extra={**_ctx, "stage": "pipeline",
                    "mode": "specific_pages" if page_ids else "all_pages",
                    "force_reindex": force_reindex})

        # ── Stage: Fetch ──────────────────────────────────────────────────
        fetch_start = time.monotonic()
        try:
            if page_ids:
                pages = [{"id": pid} for pid in page_ids]
                logger.info("Using provided page IDs", extra={**_ctx, "stage": "fetch",
                            "page_count": len(pages), "page_ids": page_ids})
            else:
                logger.info("Fetching all pages from BookStack", extra={**_ctx, "stage": "fetch"})
                pages = await self.client.get_all_pages()
                logger.info("BookStack pages fetched", extra={**_ctx, "stage": "fetch",
                            "page_count": len(pages),
                            "elapsed_s": round(time.monotonic() - fetch_start, 2)})
        except Exception as exc:
            logger.error("Failed to fetch pages from BookStack", exc_info=True, extra={
                **_ctx, "stage": "fetch", "error": str(exc)})
            raise

        logger.info("Beginning per-page ingestion", extra={**_ctx, "stage": "pipeline",
                    "total_pages": len(pages)})

        # ── Pre-fetch hierarchy caches in one pass ───────────────────────────
        # build_hierarchy_caches walks every book's ``contents`` response.
        # Book contents pages carry the canonical ``url`` field that the
        # page-detail API does not return — caching it here ensures
        # resolve_page_url always resolves to the real URL (priority 1).
        book_name_cache: dict[int, str] = {}
        book_slug_cache: dict[int, str] = {}
        page_url_cache: dict[int, str] = {}
        chapter_name_cache: dict[int, str] = {}
        try:
            (
                book_name_cache,
                book_slug_cache,
                page_url_cache,
                chapter_name_cache,
            ) = await self.client.build_hierarchy_caches()
            logger.info("Hierarchy caches ready", extra={
                **_ctx, "stage": "fetch",
                "books": len(book_name_cache),
                "chapters": len(chapter_name_cache),
                "pages_with_url": len(page_url_cache),
            })
        except Exception as exc:
            logger.warning(
                "Could not build hierarchy caches; URLs/names may be incomplete",
                extra={**_ctx, "stage": "fetch", "error": str(exc)},
            )

        for page_summary in pages:
            page_id = page_summary.get("id")
            page_ctx = {**_ctx, "page_id": page_id}

            try:
                # ── Stage: Fetch individual page ──────────────────────────
                logger.debug("Fetching page detail", extra={**page_ctx, "stage": "fetch"})
                page = await self.client.get_page(page_id)
                # Inject the canonical URL from the book-contents cache.
                # The page-detail API (GET /api/pages/{id}) does not return
                # a ``url`` field; the book-contents API does.  Injecting it
                # here lets resolve_page_url hit its priority-1 path and
                # always produce the correct canonical BookStack URL.
                if page_id in page_url_cache:
                    page["url"] = page_url_cache[page_id]
                elif page.get("book_id") in book_slug_cache:
                    # Fallback: inject book_slug so priority-2 path works.
                    page["book_slug"] = book_slug_cache[page.get("book_id")]
                book_id_for_page = page.get("book_id")
                page_title = page.get("name", "")
                source_url = self.client.resolve_page_url(page)
                book_name = book_name_cache.get(book_id_for_page or 0, "")
                chapter_name = chapter_name_cache.get(page.get("chapter_id") or 0, "")

                logger.info("Page fetched", extra={**page_ctx, "stage": "fetch",
                            "title": page_title,
                            "html_length": len(page.get("html", "") or "")})

                # ── Stage: Parse & dedup ──────────────────────────────────
                html = page.get("html", "") or ""
                plain = self.parser.html_to_text(html)
                plain = self.parser.normalize_text(plain)
                content_hash = self.parser.compute_hash(plain)

                logger.debug("Content parsed", extra={**page_ctx, "stage": "parse",
                             "plain_length": len(plain), "content_hash": content_hash[:8] + "..."})

                existing = await self._get_existing_document(page_id, "page", tenant_id)

                if existing and not force_reindex:
                    if existing.content_hash == content_hash:
                        logger.info("Page unchanged — skipped", extra={**page_ctx, "stage": "dedup",
                                    "title": page_title, "document_id": str(existing.id)})
                        stats["skipped"] += 1
                        continue
                    logger.info("Page content changed — re-ingesting", extra={
                        **page_ctx, "stage": "dedup",
                        "title": page_title, "document_id": str(existing.id),
                        "old_hash": existing.content_hash[:8] + "...",
                        "new_hash": content_hash[:8] + "...",
                    })
                    await self._delete_document_data(existing)

                doc = await self._upsert_document(page, plain, html, content_hash, tenant_id, existing, book_name, chapter_name)
                logger.info("Document upserted → PROCESSING", extra={
                    **page_ctx, "stage": "db",
                    "document_id": str(doc.id), "title": page_title,
                    "action": "update" if existing else "insert",
                })

                # ── Stage: Chunk ──────────────────────────────────────────
                chunk_start = time.monotonic()
                chunks_text = self.chunker.chunk_text(plain)

                if not chunks_text:
                    logger.info("No chunks produced — marking COMPLETED", extra={
                        **page_ctx, "stage": "chunk",
                        "plain_length": len(plain), "document_id": str(doc.id),
                    })
                    doc.status = "completed"
                    stats["processed"] += 1
                    continue

                logger.info("Chunks created", extra={**page_ctx, "stage": "chunk",
                            "chunk_count": len(chunks_text),
                            "avg_chunk_chars": round(sum(len(c) for c in chunks_text) / len(chunks_text)),
                            "elapsed_s": round(time.monotonic() - chunk_start, 3)})

                chunk_records = []
                for idx, chunk_text in enumerate(chunks_text):
                    chunk = Chunk(
                        id=uuid.uuid4(),
                        document_id=doc.id,
                        chunk_index=idx,
                        content=chunk_text,
                        content_hash=self.parser.compute_hash(chunk_text),
                        char_count=len(chunk_text),
                        metadata_={
                            "page_id": page_id,
                            "title": page_title,
                            "book_id": page.get("book_id"),
                            "book_name": book_name,
                            "chapter_id": page.get("chapter_id"),
                            "source_url": source_url,
                        },
                    )
                    chunk_records.append(chunk)

                self.db.add_all(chunk_records)
                await self.db.flush()

                # ── Stage: Embed ──────────────────────────────────────────
                embed_start = time.monotonic()
                texts = [c.content for c in chunk_records]

                logger.info("Embedding chunks", extra={**page_ctx, "stage": "embed",
                            "model": settings.EMBEDDING_MODEL,
                            "chunk_count": len(texts),
                            "document_id": str(doc.id)})
                try:
                    embeddings = self.embedding_service.embed_batch(texts)
                except Exception as exc:
                    logger.error("Embedding failed", exc_info=True, extra={
                        **page_ctx, "stage": "embed",
                        "model": settings.EMBEDDING_MODEL,
                        "chunk_count": len(texts),
                        "error": str(exc),
                    })
                    raise

                embed_elapsed = round(time.monotonic() - embed_start, 2)
                logger.info("Embedding completed", extra={**page_ctx, "stage": "embed",
                            "model": settings.EMBEDDING_MODEL,
                            "vectors_produced": len(embeddings),
                            "elapsed_s": embed_elapsed})

                # ── Stage: Vector Store ───────────────────────────────────
                ids = [str(c.id) for c in chunk_records]
                metadatas = [
                    {
                        "document_id": str(doc.id),
                        "chunk_index": c.chunk_index,
                        "title": page_title,
                        "tenant_id": tenant_id,
                        "book_id": page.get("book_id"),
                        "book_name": book_name,
                        "chapter_id": page.get("chapter_id"),
                        "source_url": source_url,
                    }
                    for c in chunk_records
                ]

                store_start = time.monotonic()
                logger.info("Inserting vectors into Qdrant", extra={
                    **page_ctx, "stage": "vector_store",
                    "collection": self.vector_store.collection_name,
                    "vector_count": len(ids),
                })
                try:
                    self.vector_store.add_embeddings(ids, embeddings, metadatas, texts)
                except Exception as exc:
                    logger.error("Vector store insert failed", exc_info=True, extra={
                        **page_ctx, "stage": "vector_store",
                        "vector_count": len(ids),
                        "error": str(exc),
                    })
                    raise

                logger.info("Vectors inserted successfully", extra={
                    **page_ctx, "stage": "vector_store",
                    "vector_count": len(ids),
                    "elapsed_s": round(time.monotonic() - store_start, 2),
                })

                # ── Stage: Embedding metadata DB ──────────────────────────
                for chunk_rec, emb_id in zip(chunk_records, ids):
                    emb_meta = EmbeddingMetadata(
                        id=uuid.uuid4(),
                        chunk_id=chunk_rec.id,
                        vector_store_id=emb_id,
                        model_name=settings.EMBEDDING_MODEL,
                        dimension=settings.EMBEDDING_DIMENSION,
                    )
                    self.db.add(emb_meta)

                # ── Stage: DB update → COMPLETED ──────────────────────────
                doc.status = "completed"
                doc.ingested_at = datetime.utcnow()
                stats["processed"] += 1
                stats["chunks_created"] += len(chunk_records)

                await self.db.commit()

                logger.info("Page ingested successfully", extra={
                    **page_ctx, "stage": "db",
                    "status": "COMPLETED",
                    "document_id": str(doc.id),
                    "title": page_title,
                    "chunks": len(chunk_records),
                    "page_elapsed_s": round(time.monotonic() - fetch_start, 2),
                })

            except Exception as e:
                logger.error("Page ingestion failed — marking FAILED", exc_info=True, extra={
                    **page_ctx, "stage": "pipeline",
                    "error": str(e),
                })
                stats["failed"] += 1
                try:
                    # Best-effort: mark document FAILED in DB
                    existing_on_error = await self._get_existing_document(page_id, "page", tenant_id)
                    if existing_on_error:
                        existing_on_error.status = "failed"
                        existing_on_error.updated_at = datetime.utcnow()
                    await self.db.rollback()
                    if existing_on_error:
                        await self.db.merge(existing_on_error)
                        await self.db.commit()
                        logger.info("Document status updated to FAILED in DB", extra={
                            **page_ctx, "stage": "db",
                            "document_id": str(existing_on_error.id),
                        })
                except Exception as db_err:
                    logger.warning("Could not update document status to FAILED", extra={
                        **page_ctx, "stage": "db", "error": str(db_err),
                    })
                    await self.db.rollback()

        await self.client.close()
        self.vector_store.save()

        total_elapsed = round(time.monotonic() - pipeline_start, 2)
        logger.info("Pipeline completed", extra={
            **_ctx, "stage": "pipeline",
            "total_elapsed_s": total_elapsed,
            "processed": stats["processed"],
            "skipped": stats["skipped"],
            "failed": stats["failed"],
            "chunks_created": stats["chunks_created"],
        })
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

    async def _upsert_document(
        self,
        page: dict,
        plain: str,
        html: str,
        content_hash: str,
        tenant_id: str,
        existing: Optional[Document],
        book_name: str = "",
        chapter_name: str = "",
    ) -> Document:
        source_url = self.client.resolve_page_url(page)

        if existing:
            existing.title = page.get("name", "")
            existing.slug = page.get("slug", "")
            existing.html_content = html
            existing.plain_content = plain
            existing.content_hash = content_hash
            existing.status = "processing"
            existing.updated_at = datetime.utcnow()
            if book_name:
                existing.book_name = book_name
            if chapter_name:
                existing.chapter_name = chapter_name
            meta = dict(existing.metadata_ or {})
            meta["source_url"] = source_url
            existing.metadata_ = meta
            return existing

        doc = Document(
            id=uuid.uuid4(),
            bookstack_id=page["id"],
            bookstack_type="page",
            title=page.get("name", ""),
            slug=page.get("slug", ""),
            book_id=page.get("book_id"),
            chapter_id=page.get("chapter_id"),
            book_name=book_name or None,
            chapter_name=chapter_name or None,
            content_hash=content_hash,
            html_content=html,
            plain_content=plain,
            status="processing",
            tenant_id=tenant_id,
            metadata_={"source_url": source_url},
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
            logger.info("Deleting old vectors from store", extra={
                "stage": "vector_store",
                "document_id": str(doc.id),
                "chunk_count": len(chunk_ids),
            })
            self.vector_store.delete_embeddings(chunk_ids)

        for chunk in chunks:
            await self.db.delete(chunk)
        await self.db.flush()

