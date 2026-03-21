"""Retrieval service: hybrid search, cross-encoder reranking, and caching."""

import logging
import time
import asyncio
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from langsmith import traceable
from sentence_transformers import CrossEncoder

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.vector_store import VectorStoreManager
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_executor = ThreadPoolExecutor(max_workers=4)


class RetrievalService:
    """Retrieve and rerank documents using hybrid search + cross-encoder."""

    _reranker_model: Optional[CrossEncoder] = None

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreManager()
        self._ensure_reranker()

    @classmethod
    def _ensure_reranker(cls):
        if cls._reranker_model is None:
            logger.info(f"Loading cross-encoder reranker: {settings.RERANKER_MODEL}")
            cls._reranker_model = CrossEncoder(
                settings.RERANKER_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )
            logger.info("Cross-encoder reranker loaded")

    @traceable(name="dense_retrieval")
    def dense_retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector retrieval."""
        query_embedding = self.embedding_service.embed(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            tenant_id=tenant_id,
        )

    @traceable(name="keyword_retrieval")
    def keyword_retrieve(
        self,
        query: str,
        top_k: int,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Keyword / BM25-like retrieval."""
        return self.vector_store.keyword_search(
            query_text=query,
            top_k=top_k,
            tenant_id=tenant_id,
        )

    @traceable(name="hybrid_retrieval")
    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = None,
        tenant_id: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval: merge dense + sparse results with score normalization."""
        top_k = top_k or settings.TOP_K_RETRIEVAL

        # Run both retrievals
        dense_results = self.dense_retrieve(query, top_k=top_k, tenant_id=tenant_id)
        keyword_results = self.keyword_retrieve(query, top_k=top_k, tenant_id=tenant_id)

        # Merge and deduplicate
        merged = self._merge_results(
            dense_results,
            keyword_results,
            dense_weight=settings.DENSE_WEIGHT,
            sparse_weight=settings.BM25_WEIGHT,
        )

        # Apply metadata filters
        if filters:
            merged = self._apply_filters(merged, filters)

        # Apply similarity threshold
        merged = [r for r in merged if r["score"] >= settings.SIMILARITY_THRESHOLD]

        return merged[:top_k]

    def _merge_results(
        self,
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) merge of dense and sparse results."""
        k = 60  # RRF constant
        scores: Dict[str, float] = {}
        docs: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)
            docs[doc_id] = doc

        for rank, doc in enumerate(sparse):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc

        # Normalize scores to 0-1
        if scores:
            max_score = max(scores.values())
            min_score = min(scores.values())
            score_range = max_score - min_score if max_score != min_score else 1.0
            for doc_id in scores:
                scores[doc_id] = (scores[doc_id] - min_score) / score_range

        # Build sorted result list
        result = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            doc = docs[doc_id]
            doc["score"] = score
            result.append(doc)

        return result

    @traceable(name="cross_encoder_rerank")
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder model for precise relevance scoring."""
        top_k = top_k or settings.TOP_K_RERANK

        if not documents:
            return []

        start = time.time()

        # Prepare pairs for cross-encoder
        pairs = [[query, doc["text"]] for doc in documents]

        # Score in batches
        scores = self._reranker_model.predict(
            pairs,
            batch_size=settings.RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )

        # Assign scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        latency = (time.time() - start) * 1000
        logger.info(f"Reranking {len(documents)} docs took {latency:.1f}ms")

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
