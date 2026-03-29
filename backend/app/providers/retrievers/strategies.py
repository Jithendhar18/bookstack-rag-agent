"""Retriever implementations — dense, keyword, and hybrid strategies."""

import logging
from typing import List, Dict, Any, Optional

from langsmith import traceable

from app.providers.base import BaseRetriever, BaseEmbedding
from app.retrieval.vector_store import VectorStoreManager
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DenseRetriever(BaseRetriever):
    """Dense vector retrieval using embedding similarity."""

    def __init__(self, embedding: BaseEmbedding, vector_store: VectorStoreManager):
        self._embedding = embedding
        self._vector_store = vector_store

    @traceable(name="dense_retrieval")
    def retrieve(self, query: str, top_k: int, tenant_id: Optional[str] = None,
                 filters: Optional[dict] = None) -> List[Dict[str, Any]]:
        query_embedding = self._embedding.embed(query)
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            tenant_id=tenant_id,
        )
        if filters:
            results = _apply_filters(results, filters)
        # Filter by similarity threshold — cosine scores are already 0-1
        results = [r for r in results if r.get("score", 0) >= settings.SIMILARITY_THRESHOLD]
        return results[:top_k]


class KeywordRetriever(BaseRetriever):
    """Keyword / full-text retrieval."""

    def __init__(self, vector_store: VectorStoreManager):
        self._vector_store = vector_store

    @traceable(name="keyword_retrieval")
    def retrieve(self, query: str, top_k: int, tenant_id: Optional[str] = None,
                 filters: Optional[dict] = None) -> List[Dict[str, Any]]:
        results = self._vector_store.keyword_search(
            query_text=query,
            top_k=top_k,
            tenant_id=tenant_id,
        )
        if filters:
            results = _apply_filters(results, filters)
        return results[:top_k]


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval: merge dense + keyword results via Reciprocal Rank Fusion."""

    def __init__(self, embedding: BaseEmbedding, vector_store: VectorStoreManager,
                 dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self._dense = DenseRetriever(embedding, vector_store)
        self._keyword = KeywordRetriever(vector_store)
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    @traceable(name="hybrid_retrieval")
    def retrieve(self, query: str, top_k: int, tenant_id: Optional[str] = None,
                 filters: Optional[dict] = None) -> List[Dict[str, Any]]:
        dense_results = self._dense.retrieve(query, top_k=top_k, tenant_id=tenant_id)
        keyword_results = self._keyword.retrieve(query, top_k=top_k, tenant_id=tenant_id)

        logger.info(f"Hybrid: {len(dense_results)} dense + {len(keyword_results)} keyword results")

        merged = self._rrf_merge(dense_results, keyword_results)

        if filters:
            merged = _apply_filters(merged, filters)

        return merged[:top_k]

    def _rrf_merge(self, dense: List[Dict], sparse: List[Dict]) -> List[Dict]:
        """Reciprocal Rank Fusion merge with normalized 0-1 output scores."""
        k = 60  # RRF constant
        scores: Dict[str, float] = {}
        docs: Dict[str, Dict] = {}

        for rank, doc in enumerate(dense):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + self._dense_weight / (k + rank + 1)
            docs[doc_id] = doc

        for rank, doc in enumerate(sparse):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + self._sparse_weight / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc

        # Normalize to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                for doc_id in scores:
                    scores[doc_id] = scores[doc_id] / max_score

        result = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            doc = docs[doc_id].copy()
            doc["score"] = round(score, 4)
            result.append(doc)

        return result


def _apply_filters(results: List[Dict[str, Any]], filters: dict) -> List[Dict[str, Any]]:
    """Filter results by metadata fields."""
    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        if all(meta.get(k) == v for k, v in filters.items()):
            filtered.append(r)
    return filtered
