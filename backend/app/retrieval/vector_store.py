"""Qdrant vector store manager."""

import logging
import time
import threading
from typing import List, Optional, Dict, Any

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStoreManager:
    """Qdrant vector store interface (singleton)."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        from qdrant_client import QdrantClient

        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=30,
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection if it does not exist, or verify dimension matches."""
        from qdrant_client.models import Distance, VectorParams, TextIndexParams, TokenizerType

        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema="keyword",
            )
            logger.info(f"Qdrant collection created: {self.collection_name} (dim={settings.EMBEDDING_DIMENSION})")
        else:
            info = self.client.get_collection(self.collection_name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    "Qdrant dimension mismatch: collection=%d, config=%d",
                    existing_dim, settings.EMBEDDING_DIMENSION,
                )
            logger.info(f"Qdrant collection exists: {self.collection_name} ({info.points_count} vectors)")

    # ─── Public API ──────────────────────────────────────────────────────

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        texts: List[str],
    ):
        """Upsert embeddings into Qdrant in batches."""
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=doc_id, vector=emb, payload={**meta, "text": text})
            for doc_id, emb, meta, text in zip(ids, embeddings, metadatas, texts)
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            last_err = None
            for attempt in range(1, 4):
                try:
                    self.client.upsert(collection_name=self.collection_name, points=batch)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                        logger.warning("Qdrant upsert retry %d/3: %s", attempt, e)
                    else:
                        logger.error("Qdrant upsert failed after 3 attempts", exc_info=True)
            if last_err is not None:
                raise last_err

        logger.info(f"Qdrant upserted {len(points)} vectors into {self.collection_name}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if tenant_id:
            query_filter = Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            }
            for hit in results.points
        ]

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 10,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search using Qdrant's text index.

        Scores results by word overlap ratio instead of returning flat 1.0.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

        must_conditions = [
            FieldCondition(key="text", match=MatchText(text=query_text))
        ]
        if tenant_id:
            must_conditions.append(
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
            )

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        # Score by word overlap ratio so keyword results have meaningful ranking
        query_words = set(query_text.lower().split())
        scored = []
        for point in results[0]:
            text = point.payload.get("text", "")
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            score = overlap / max(len(query_words), 1)
            scored.append({
                "id": str(point.id),
                "score": min(score, 1.0),
                "text": text,
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def delete_embeddings(self, ids: List[str]):
        """Remove embeddings by ID."""
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def save(self):
        """No-op: Qdrant persists automatically."""
        pass

    @property
    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count
