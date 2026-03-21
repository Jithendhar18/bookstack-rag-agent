"""Embedding service with singleton model loading, batch support, and caching."""

import logging
import hashlib
from typing import List

from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Singleton-based embedding service using SentenceTransformers."""

    _instance = None
    _model = None
    _cache: LRUCache = LRUCache(maxsize=10000)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if EmbeddingService._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            EmbeddingService._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )
            logger.info("Embedding model loaded")

    @property
    def model(self) -> SentenceTransformer:
        return EmbeddingService._model

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> List[float]:
        """Embed a single text, with caching."""
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        self._cache[key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Uses cache for previously seen texts."""
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                normalize_embeddings=True,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
            ).tolist()

            for idx, emb, text in zip(uncached_indices, embeddings, uncached_texts):
                key = self._cache_key(text)
                self._cache[key] = emb
                results[idx] = emb

        return results

    @property
    def dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION
