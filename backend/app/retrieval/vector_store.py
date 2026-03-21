"""Vector store abstraction supporting FAISS, PGVector, and Qdrant."""

import os
import logging
import json
from typing import List, Optional, Dict, Any

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStoreManager:
    """Unified interface for vector stores (FAISS, PGVector, or Qdrant)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.store_type = settings.VECTOR_STORE_TYPE

        if self.store_type == "qdrant":
            self._init_qdrant()
        elif self.store_type == "faiss":
            self._init_faiss()
        else:
            self._init_pgvector()

    # ─── Qdrant ───────────────────────────────────────────────────────────

    def _init_qdrant(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, TextIndexParams, TokenizerType

        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
        )
        self.collection_name = settings.QDRANT_COLLECTION

        # Create collection if not exists
        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            # Create text payload index for full-text / BM25-like search
            self.qdrant_client.create_payload_index(
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
            # Create tenant_id keyword index for filtering
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema="keyword",
            )
            logger.info(f"Qdrant collection '{self.collection_name}' created with text index")
        else:
            logger.info(f"Qdrant collection '{self.collection_name}' already exists")

    # ─── FAISS ────────────────────────────────────────────────────────────

    def _init_faiss(self):
        import faiss

        self.index_path = settings.FAISS_INDEX_PATH
        self.metadata_path = f"{self.index_path}_metadata.json"

        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)

        if os.path.exists(self.index_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            self._load_metadata()
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
            self.id_map: List[str] = []
            self.metadata_store: Dict[str, dict] = {}
            self.text_store: Dict[str, str] = {}

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                data = json.load(f)
                self.id_map = data.get("id_map", [])
                self.metadata_store = data.get("metadata_store", {})
                self.text_store = data.get("text_store", {})
        else:
            self.id_map = []
            self.metadata_store = {}
            self.text_store = {}

    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump({
                "id_map": self.id_map,
                "metadata_store": self.metadata_store,
                "text_store": self.text_store,
            }, f)

    # ─── PGVector ─────────────────────────────────────────────────────────

    def _init_pgvector(self):
        logger.info("Using PGVector for vector storage")
        self.id_map = []
        self.metadata_store = {}
        self.text_store = {}

    # ─── Public API ──────────────────────────────────────────────────────

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        texts: List[str],
    ):
        """Add embeddings to the vector store."""
        if self.store_type == "qdrant":
            self._qdrant_add(ids, embeddings, metadatas, texts)
        elif self.store_type == "faiss":
            vectors = np.array(embeddings, dtype="float32")
            self.index.add(vectors)
            for doc_id, meta, text in zip(ids, metadatas, texts):
                self.id_map.append(doc_id)
                self.metadata_store[doc_id] = meta
                self.text_store[doc_id] = text

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search for similar vectors."""
        if self.store_type == "qdrant":
            return self._qdrant_search(query_embedding, top_k, tenant_id)
        elif self.store_type == "faiss":
            return self._faiss_search(query_embedding, top_k, tenant_id)
        return []

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 10,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Keyword / full-text search (BM25-like). Qdrant only for now."""
        if self.store_type == "qdrant":
            return self._qdrant_keyword_search(query_text, top_k, tenant_id)
        return []

    def delete_embeddings(self, ids: List[str]):
        """Remove embeddings by ID."""
        if self.store_type == "qdrant":
            self._qdrant_delete(ids)
        elif self.store_type == "faiss":
            for doc_id in ids:
                self.metadata_store.pop(doc_id, None)
                self.text_store.pop(doc_id, None)
                if doc_id in self.id_map:
                    self.id_map.remove(doc_id)

    def save(self):
        """Persist the index to disk."""
        if self.store_type == "faiss":
            import faiss
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()
            logger.info(f"FAISS index saved: {self.index.ntotal} vectors")
        # Qdrant persists automatically

    @property
    def count(self) -> int:
        if self.store_type == "qdrant":
            info = self.qdrant_client.get_collection(self.collection_name)
            return info.points_count
        elif self.store_type == "faiss":
            return self.index.ntotal
        return 0

    # ─── Qdrant internals ────────────────────────────────────────────────

    def _qdrant_add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        texts: List[str],
    ):
        from qdrant_client.models import PointStruct

        points = []
        for doc_id, emb, meta, text in zip(ids, embeddings, metadatas, texts):
            payload = {**meta, "text": text}
            points.append(PointStruct(
                id=doc_id,
                vector=emb,
                payload=payload,
            ))

        # Batch upsert in chunks of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + batch_size],
            )

    def _qdrant_search(
        self,
        query_embedding: List[float],
        top_k: int,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if tenant_id:
            query_filter = Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
            )

        results = self.qdrant_client.query_points(
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

    def _qdrant_keyword_search(
        self,
        query_text: str,
        top_k: int,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search using Qdrant's text index."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

        must_conditions = [
            FieldCondition(key="text", match=MatchText(text=query_text))
        ]
        if tenant_id:
            must_conditions.append(
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
            )

        results = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        points = results[0]
        return [
            {
                "id": str(point.id),
                "score": 1.0,  # full-text match, normalized later
                "text": point.payload.get("text", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
            }
            for point in points
        ]

    def _qdrant_delete(self, ids: List[str]):
        from qdrant_client.models import PointIdsList
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    # ─── FAISS internals ─────────────────────────────────────────────────

    def _faiss_search(
        self,
        query_embedding: List[float],
        top_k: int,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype="float32")
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue

            doc_id = self.id_map[idx]
            meta = self.metadata_store.get(doc_id, {})

            if tenant_id and meta.get("tenant_id") != tenant_id:
                continue

            results.append({
                "id": doc_id,
                "score": float(score),
                "text": self.text_store.get(doc_id, ""),
                "metadata": meta,
            })

            if len(results) >= top_k:
                break

        return results
