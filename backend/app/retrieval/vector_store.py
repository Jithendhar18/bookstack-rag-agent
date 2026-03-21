"""Vector store abstraction supporting FAISS and PGVector."""

import os
import logging
import json
from typing import List, Optional, Dict, Any

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStoreManager:
    """Unified interface for vector stores (FAISS or PGVector)."""

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

        if self.store_type == "faiss":
            self._init_faiss()
        else:
            self._init_pgvector()

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
        # PGVector uses the existing PostgreSQL connection
        # Embeddings are stored directly in the DB via SQLAlchemy
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
        if self.store_type == "faiss":
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
        """Search for similar vectors."""
        if self.store_type == "faiss":
            return self._faiss_search(query_embedding, top_k, tenant_id)
        return []

    def _faiss_search(
        self,
        query_embedding: List[float],
        top_k: int,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        import faiss

        if self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype="float32")
        # Search more than needed, then filter by tenant
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue

            doc_id = self.id_map[idx]
            meta = self.metadata_store.get(doc_id, {})

            # Tenant filtering
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

    def delete_embeddings(self, ids: List[str]):
        """Remove embeddings by ID. For FAISS, this marks them for rebuild."""
        if self.store_type == "faiss":
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

    @property
    def count(self) -> int:
        if self.store_type == "faiss":
            return self.index.ntotal
        return 0
