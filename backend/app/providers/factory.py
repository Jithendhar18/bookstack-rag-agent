"""Factory functions — read env config and return the correct provider instance.

Usage:
    llm = get_llm()          # Returns BaseLLM
    emb = get_embedding()    # Returns BaseEmbedding
    rer = get_reranker()     # Returns BaseReranker (or NoOpReranker)
    ret = get_retriever()    # Returns BaseRetriever
"""

import logging
from typing import Optional

from config import get_settings
from app.providers.base import BaseLLM, BaseEmbedding, BaseReranker, BaseRetriever, NoOpReranker

logger = logging.getLogger(__name__)

# ─── Singletons ──────────────────────────────────────────────────────────
_llm: Optional[BaseLLM] = None
_fallback_llm: Optional[BaseLLM] = None
_embedding: Optional[BaseEmbedding] = None
_reranker: Optional[BaseReranker] = None
_retriever: Optional[BaseRetriever] = None


def _build_llm(provider: str, model: str, api_key: str, base_url: str,
               temperature: float, max_tokens: int) -> BaseLLM:
    """Build an LLM instance for the given provider."""
    if provider == "ollama":
        from app.providers.llm.ollama import OllamaLLM
        return OllamaLLM(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        from app.providers.llm.openai_compatible import OpenAICompatibleLLM
        return OpenAICompatibleLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name=provider,
        )


def get_llm(force_new: bool = False) -> BaseLLM:
    """Get the configured LLM provider (singleton)."""
    global _llm
    if _llm is not None and not force_new:
        return _llm

    settings = get_settings()

    _llm = _build_llm(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY or "",
        base_url=settings.LLM_BASE_URL or "",
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    logger.info(f"LLM factory: provider={settings.LLM_PROVIDER}, model={settings.LLM_MODEL}")
    return _llm


def get_fallback_llm() -> Optional[BaseLLM]:
    """Get the fallback LLM provider, if configured."""
    global _fallback_llm
    if _fallback_llm is not None:
        return _fallback_llm

    settings = get_settings()
    if not settings.LLM_FALLBACK_PROVIDER or not settings.LLM_FALLBACK_MODEL:
        return None

    # Resolve fallback base URL
    fallback_base_url = settings.LLM_FALLBACK_BASE_URL
    if not fallback_base_url:
        base_urls = {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "groq": "https://api.groq.com/openai/v1",
            "ollama": settings.OLLAMA_BASE_URL + "/v1",
        }
        fallback_base_url = base_urls.get(settings.LLM_FALLBACK_PROVIDER, "")

    # Resolve fallback API key
    fallback_api_key = settings.LLM_FALLBACK_API_KEY
    if not fallback_api_key:
        key_map = {
            "openai": settings.OPENAI_API_KEY,
            "openrouter": settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY,
            "groq": settings.GROQ_API_KEY,
            "ollama": "not-needed",
        }
        fallback_api_key = key_map.get(settings.LLM_FALLBACK_PROVIDER, "")

    _fallback_llm = _build_llm(
        provider=settings.LLM_FALLBACK_PROVIDER,
        model=settings.LLM_FALLBACK_MODEL,
        api_key=fallback_api_key,
        base_url=fallback_base_url,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    logger.info(f"Fallback LLM: provider={settings.LLM_FALLBACK_PROVIDER}, model={settings.LLM_FALLBACK_MODEL}")
    return _fallback_llm


def get_embedding(force_new: bool = False) -> BaseEmbedding:
    """Get the configured embedding provider (singleton)."""
    global _embedding
    if _embedding is not None and not force_new:
        return _embedding

    settings = get_settings()

    if settings.EMBEDDING_PROVIDER == "openai":
        from app.providers.embeddings.openai import OpenAIEmbedding
        _embedding = OpenAIEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
            dimension=settings.EMBEDDING_DIMENSION,
        )
    else:
        # "local" and "huggingface" both use SentenceTransformers locally
        from app.providers.embeddings.local import LocalEmbedding
        _embedding = LocalEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            dimension=settings.EMBEDDING_DIMENSION,
        )

    logger.info(f"Embedding factory: provider={settings.EMBEDDING_PROVIDER}, model={settings.EMBEDDING_MODEL}")
    return _embedding


def get_fallback_embedding() -> Optional[BaseEmbedding]:
    """Build a fallback embedding provider using EMBEDDING_FALLBACK_* settings."""
    settings = get_settings()
    if not settings.EMBEDDING_FALLBACK_MODEL:
        return None

    from app.providers.embeddings.local import LocalEmbedding
    return LocalEmbedding(
        model_name=settings.EMBEDDING_FALLBACK_MODEL,
        device=settings.EMBEDDING_DEVICE,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        dimension=384,  # small model dimension
    )


def get_reranker(force_new: bool = False) -> BaseReranker:
    """Get the configured reranker (singleton). Returns NoOpReranker if disabled."""
    global _reranker
    if _reranker is not None and not force_new:
        return _reranker

    settings = get_settings()

    if not settings.RERANKER_ENABLED:
        _reranker = NoOpReranker()
        logger.info("Reranker: DISABLED (using NoOpReranker)")
    else:
        from app.providers.rerankers.cross_encoder import CrossEncoderReranker
        _reranker = CrossEncoderReranker.get_instance(
            model_name=settings.RERANKER_MODEL,
            device=settings.EMBEDDING_DEVICE,
            batch_size=settings.RERANKER_BATCH_SIZE,
        )
        logger.info(f"Reranker factory: model={settings.RERANKER_MODEL}")

    return _reranker


def get_retriever(force_new: bool = False) -> BaseRetriever:
    """Get the configured retriever strategy (singleton)."""
    global _retriever
    if _retriever is not None and not force_new:
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
    else:  # "hybrid" (default)
        from app.providers.retrievers.strategies import HybridRetriever
        _retriever = HybridRetriever(
            embedding, vector_store,
            dense_weight=settings.DENSE_WEIGHT,
            sparse_weight=settings.BM25_WEIGHT,
        )

    logger.info(f"Retriever factory: mode={settings.RETRIEVAL_MODE}")
    return _retriever


def log_active_configuration():
    """Log the current active configuration at startup."""
    settings = get_settings()
    modules = settings.get_active_modules()
    logger.info("=" * 60)
    logger.info("ACTIVE AI PIPELINE CONFIGURATION")
    logger.info("=" * 60)
    for module_name, details in modules.items():
        if isinstance(details, dict):
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            logger.info(f"  {module_name:.<25s} {detail_str}")
        else:
            logger.info(f"  {module_name:.<25s} {details}")
    logger.info("=" * 60)
