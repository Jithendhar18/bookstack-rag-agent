"""OpenAI embedding provider."""

import logging
from typing import List

import httpx

from app.providers.base import BaseEmbedding

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """Embedding provider using OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = "",
                 dimension: int = 1536):
        self._model_name = model_name
        self._api_key = api_key
        self._dimension = dimension
        self._base_url = "https://api.openai.com/v1/embeddings"
        logger.info(f"OpenAI embedding provider initialized: {model_name}")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        response = httpx.post(
            self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self._model_name},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def embed(self, text: str) -> List[float]:
        return self._call_api([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # OpenAI has a limit, batch in groups of 100
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension
