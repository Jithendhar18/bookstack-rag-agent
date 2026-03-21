"""Embedding service — thin wrapper that delegates to the configured provider.

This module maintains backward compatibility for code that imports EmbeddingService
directly (e.g., the ingestion pipeline), while routing through the factory.
"""

import logging
from typing import List

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Singleton embedding service backed by the pluggable provider system."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_provider"):
            from app.providers.factory import get_embedding
            self._provider = get_embedding()
            logger.info(f"EmbeddingService initialized via factory: {settings.EMBEDDING_PROVIDER}/{settings.EMBEDDING_MODEL}")

    def embed(self, text: str) -> List[float]:
        """Embed a single text, with caching."""
        return self._provider.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        return self._provider.embed_batch(texts)

    @property
    def dimension(self) -> int:
        return self._provider.dimension
