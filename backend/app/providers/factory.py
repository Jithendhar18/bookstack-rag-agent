"""Factory functions — return provider instances based on config."""

import logging
from typing import Optional

from config import get_settings
from app.providers.base import BaseLLM, BaseEmbedding, BaseReranker, BaseRetriever, NoOpReranker

logger = logging.getLogger(__name__)

_llm: Optional[BaseLLM] = None
_embedding: Optional[BaseEmbedding] = None
_reranker: Optional[BaseReranker] = None
_retriever: Optional[BaseRetriever] = None


def get_llm() -> BaseLLM:
    """Get the configured LLM provider (singleton)."""
    global _llm
    if _llm is not None:
        return _llm

    settings = get_settings()

    if settings.LLM_PROVIDER == "ollama" or not settings.LLM_API_KEY:
        # Use Ollama for local inference if provider is explicitly set or if no API key is configured
        from app.providers.llm.ollama import OllamaLLM
        _llm = OllamaLLM(
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL or (settings.OLLAMA_BASE_URL + "/v1"),
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    else:
        # Use OpenAI-compatible provider
        from app.providers.llm.openai_compatible import OpenAICompatibleLLM
        _llm = OpenAICompatibleLLM(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            provider_name=settings.LLM_PROVIDER,
        )

    logger.info(f"LLM: provider={settings.LLM_PROVIDER or 'ollama'}, model={settings.LLM_MODEL}")
    return _llm


def get_embedding() -> BaseEmbedding:
    """Get the local embedding provider (singleton)."""
    global _embedding
    if _embedding is not None:
        return _embedding

    settings = get_settings()
    from app.providers.embeddings.local import LocalEmbedding
    _embedding = LocalEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        dimension=settings.EMBEDDING_DIMENSION,
    )

    logger.info(f"Embedding: model={settings.EMBEDDING_MODEL}")
    return _embedding


def get_reranker() -> BaseReranker:
    """Get the reranker (singleton). Returns NoOpReranker if disabled."""
    global _reranker
    if _reranker is not None:
        return _reranker

    settings = get_settings()

    if not settings.RERANKER_ENABLED:
        _reranker = NoOpReranker()
        logger.info("Reranker: DISABLED")
    else:
        from app.providers.rerankers.cross_encoder import CrossEncoderReranker
        _reranker = CrossEncoderReranker.get_instance(model_name=settings.RERANKER_MODEL)
        logger.info(f"Reranker: model={settings.RERANKER_MODEL}")

    return _reranker


def get_retriever() -> BaseRetriever:
    """Get the configured retriever strategy (singleton)."""
    global _retriever
    if _retriever is not None:
        return _retriever

    settings = get_settings()
    embedding = get_embedding()
    from app.retrieval.vector_store import VectorStoreManager
    vector_store = VectorStoreManager()

    if settings.RETRIEVAL_MODE == "keyword":
        from app.providers.retrievers.strategies import KeywordRetriever
        _retriever = KeywordRetriever(vector_store)
    elif settings.RETRIEVAL_MODE == "dense":
        from app.providers.retrievers.strategies import DenseRetriever
        _retriever = DenseRetriever(embedding, vector_store)
    else:
        from app.providers.retrievers.strategies import HybridRetriever
        _retriever = HybridRetriever(
            embedding, vector_store,
            dense_weight=settings.DENSE_WEIGHT,
            sparse_weight=settings.BM25_WEIGHT,
        )

    logger.info(f"Retriever: mode={settings.RETRIEVAL_MODE}")
    return _retriever
