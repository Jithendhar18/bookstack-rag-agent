"""Cross-encoder reranker provider."""

import logging
import time
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from app.providers.base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranker using a locally loaded CrossEncoder model."""

    _instances: Dict[str, "CrossEncoderReranker"] = {}

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16):
        self._model_name = model_name
        self._batch_size = batch_size
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        self._model = CrossEncoder(model_name, device=device)
        logger.info(f"Cross-encoder reranker loaded: {model_name}")

    @classmethod
    def get_instance(cls, model_name: str, device: str = "cpu", batch_size: int = 16) -> "CrossEncoderReranker":
        """Singleton per model name to avoid reloading."""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name, device, batch_size)
        return cls._instances[model_name]

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []

        start = time.time()
        pairs = [[query, doc.get("text", "")] for doc in documents]

        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        latency = (time.time() - start) * 1000
        logger.info(f"Reranking {len(documents)} docs took {latency:.1f}ms")

        return documents[:top_k]
