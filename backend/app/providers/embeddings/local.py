"""Local embedding provider using SentenceTransformers."""

import logging
import hashlib
from typing import List

from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

from app.providers.base import BaseEmbedding

logger = logging.getLogger(__name__)


class LocalEmbedding(BaseEmbedding):
    """Embedding provider using locally loaded SentenceTransformer models."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32, dimension: int = 1024):
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._dimension = dimension
        self._cache: LRUCache = LRUCache(maxsize=10000)

        logger.info(f"Loading local embedding model: {model_name} on {device}")
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(f"Local embedding model loaded: {model_name}")

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> List[float]:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        embedding = self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False,
        ).tolist()

        self._cache[key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
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
            embeddings = self._model.encode(
                uncached_texts,
                normalize_embeddings=True,
                batch_size=self._batch_size,
                show_progress_bar=False,
            ).tolist()

            for idx, emb, text in zip(uncached_indices, embeddings, uncached_texts):
                key = self._cache_key(text)
                self._cache[key] = emb
                results[idx] = emb

        return results

    @property
    def dimension(self) -> int:
        return self._dimension
