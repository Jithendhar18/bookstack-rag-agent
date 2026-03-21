"""Abstract base classes for all pluggable pipeline components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator


class BaseLLM(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from a list of messages."""
        ...

    @abstractmethod
    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """Stream a response token-by-token."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...


class BaseEmbedding(ABC):
    """Interface for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class BaseReranker(ABC):
    """Interface for reranker providers."""

    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to query. Returns top_k results with rerank_score."""
        ...


class NoOpReranker(BaseReranker):
    """Pass-through reranker used when reranking is disabled."""

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        return documents[:top_k]


class BaseRetriever(ABC):
    """Interface for retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int, tenant_id: Optional[str] = None,
                 filters: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        ...
