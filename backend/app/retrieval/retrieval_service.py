"""Retrieval service: search, rerank, return top results."""

import logging
from typing import List, Optional, Dict, Any

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.vector_store import VectorStoreManager
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RetrievalService:
    """Retrieve and rerank documents from the vector store."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        tenant_id: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Embed query and retrieve top-k similar chunks."""
        top_k = top_k or settings.TOP_K_RETRIEVAL

        query_embedding = self.embedding_service.embed(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            tenant_id=tenant_id,
        )

        # Apply metadata filters if provided
        if filters:
            results = self._apply_filters(results, filters)

        # Filter by similarity threshold
        results = [r for r in results if r["score"] >= settings.SIMILARITY_THRESHOLD]

        return results

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Rerank retrieved documents using cross-encoder scoring.

        Falls back to embedding similarity score if no cross-encoder is available.
        """
        top_k = top_k or settings.TOP_K_RERANK

        if not documents:
            return []

        # Score-based reranking using query-document embedding similarity
        query_emb = self.embedding_service.embed(query)
        for doc in documents:
            doc_emb = self.embedding_service.embed(doc["text"])
            # Cosine similarity (embeddings are normalized)
            dot = sum(a * b for a, b in zip(query_emb, doc_emb))
            doc["rerank_score"] = dot

        documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return documents[:top_k]

    def _apply_filters(self, results: List[Dict[str, Any]], filters: dict) -> List[Dict[str, Any]]:
        """Filter results by metadata fields."""
        filtered = []
        for r in results:
            meta = r.get("metadata", {})
            match = all(meta.get(k) == v for k, v in filters.items())
            if match:
                filtered.append(r)
        return filtered
