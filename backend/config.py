"""Application configuration loaded from environment variables."""

import logging
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from functools import lru_cache

logger = logging.getLogger(__name__)


# ─── AI Profile Defaults ─────────────────────────────────────────────────
# Profiles provide smart defaults. Explicit env vars always override.

AI_PROFILES = {
    "cheap": {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "llama3",
        "EMBEDDING_PROVIDER": "local",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_DIMENSION": 384,
        "RERANKER_ENABLED": False,
        "QUERY_REWRITER_ENABLED": False,
        "CONTEXT_COMPRESSION_ENABLED": True,
        "RETRIEVAL_MODE": "dense",
    },
    "balanced": {
        "LLM_PROVIDER": "openrouter",
        "LLM_MODEL": "mistralai/mistral-7b-instruct",
        "EMBEDDING_PROVIDER": "local",
        "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
        "EMBEDDING_DIMENSION": 768,
        "RERANKER_ENABLED": True,
        "QUERY_REWRITER_ENABLED": True,
        "CONTEXT_COMPRESSION_ENABLED": True,
        "RETRIEVAL_MODE": "hybrid",
    },
    "best": {
        "LLM_PROVIDER": "openrouter",
        "LLM_MODEL": "openai/gpt-4o",
        "EMBEDDING_PROVIDER": "local",
        "EMBEDDING_MODEL": "BAAI/bge-large-en-v1.5",
        "EMBEDDING_DIMENSION": 1024,
        "RERANKER_ENABLED": True,
        "QUERY_REWRITER_ENABLED": True,
        "CONTEXT_COMPRESSION_ENABLED": True,
        "RETRIEVAL_MODE": "hybrid",
    },
}


class Settings(BaseSettings):
    """Central application settings — fully configurable via .env."""

    # ─── Global ───────────────────────────────────────────────────────
    APP_NAME: str = "BookStack RAG Agent"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development | production
    AI_PROFILE: str = ""  # cheap | balanced | best (empty = manual config)

    # ─── Server ───────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ─── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5435/bookstack_rag"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5435/bookstack_rag"

    # ─── BookStack ────────────────────────────────────────────────────
    BOOKSTACK_BASE_URL: str = ""
    BOOKSTACK_TOKEN_ID: str = ""
    BOOKSTACK_TOKEN_SECRET: str = ""

    # ─── LLM ──────────────────────────────────────────────────────────
    LLM_ENABLED: bool = True
    LLM_PROVIDER: str = "openai"  # openai | openrouter | groq | ollama
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_BASE_URL: Optional[str] = None  # custom base URL for provider
    LLM_API_KEY: Optional[str] = None  # provider-specific key (falls back to OPENAI_API_KEY)

    # Fallback LLM (used if primary fails)
    LLM_FALLBACK_PROVIDER: str = ""  # empty = no fallback
    LLM_FALLBACK_MODEL: str = ""
    LLM_FALLBACK_API_KEY: Optional[str] = None
    LLM_FALLBACK_BASE_URL: Optional[str] = None

    # Legacy / shared key
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ─── Embeddings ───────────────────────────────────────────────────
    EMBEDDING_ENABLED: bool = True
    EMBEDDING_PROVIDER: str = "local"  # local | huggingface | openai
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_FALLBACK_PROVIDER: str = "local"
    EMBEDDING_FALLBACK_MODEL: str = "BAAI/bge-small-en-v1.5"

    # ─── Reranker ─────────────────────────────────────────────────────
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    RERANKER_BATCH_SIZE: int = 16

    # ─── Retrieval ────────────────────────────────────────────────────
    RETRIEVAL_MODE: str = "hybrid"  # dense | hybrid | keyword
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    BM25_WEIGHT: float = 0.3
    DENSE_WEIGHT: float = 0.7

    # ─── Query Rewriter ──────────────────────────────────────────────
    QUERY_REWRITER_ENABLED: bool = True

    # ─── Context Compression ─────────────────────────────────────────
    CONTEXT_COMPRESSION_ENABLED: bool = True
    MAX_CONTEXT_TOKENS: int = 4096
    MMR_LAMBDA: float = 0.7

    # ─── Cache ────────────────────────────────────────────────────────
    CACHE_ENABLED: bool = True
    CACHE_QUERY_TTL: int = 600
    CACHE_RETRIEVAL_TTL: int = 300

    # ─── Guardrails ──────────────────────────────────────────────────
    GUARDRAILS_ENABLED: bool = True
    MIN_SUPPORTING_CHUNKS: int = 1
    HALLUCINATION_THRESHOLD: float = 0.5

    # ─── Vector Store ─────────────────────────────────────────────────
    VECTOR_STORE_TYPE: str = "qdrant"
    FAISS_INDEX_PATH: str = "./data/faiss_index"

    # ─── Qdrant ───────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "bookstack_chunks"
    QDRANT_API_KEY: str = ""

    # ─── Chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100

    # ─── Auth / JWT ───────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-this-to-a-secure-random-string"
    JWT_ALGORITHM: str = "HS256"
    ADMIN_DEFAULT_PASSWORD: str = "admin1234"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ─── LangSmith ───────────────────────────────────────────────────
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "bookstack-rag-agent"
    LANGSMITH_TRACING_V2: bool = True
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "bookstack-rag-agent"

    # ─── Redis ────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ─── CORS ─────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # ─── Rate Limiting ────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60

    # ─── Celery ───────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ─── Streaming ────────────────────────────────────────────────────
    STREAMING_ENABLED: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @model_validator(mode="after")
    def _apply_profile_and_validate(self) -> "Settings":
        """Apply AI profile defaults, then validate production settings."""
        # Apply AI profile defaults (only for fields not explicitly set in env)
        if self.AI_PROFILE and self.AI_PROFILE in AI_PROFILES:
            import os
            profile = AI_PROFILES[self.AI_PROFILE]
            for key, value in profile.items():
                # Only apply profile default if the env var is NOT explicitly set
                if key not in os.environ:
                    setattr(self, key, value)
            logger.info(f"AI profile '{self.AI_PROFILE}' applied")

        # Resolve effective LLM API key
        if not self.LLM_API_KEY:
            provider_key_map = {
                "openai": self.OPENAI_API_KEY,
                "openrouter": self.OPENROUTER_API_KEY or self.OPENAI_API_KEY,
                "groq": self.GROQ_API_KEY,
                "ollama": "not-needed",
            }
            self.LLM_API_KEY = provider_key_map.get(self.LLM_PROVIDER, self.OPENAI_API_KEY)

        # Resolve default base URLs per provider
        if not self.LLM_BASE_URL:
            base_urls = {
                "openai": "https://api.openai.com/v1",
                "openrouter": "https://openrouter.ai/api/v1",
                "groq": "https://api.groq.com/openai/v1",
                "ollama": self.OLLAMA_BASE_URL + "/v1",
            }
            self.LLM_BASE_URL = base_urls.get(self.LLM_PROVIDER)

        # Production validations
        if self.ENVIRONMENT == "production":
            if self.JWT_SECRET_KEY == "change-this-to-a-secure-random-string":
                raise ValueError(
                    "JWT_SECRET_KEY must be changed from the default in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                )
            if self.LLM_PROVIDER != "ollama" and not self.LLM_API_KEY:
                raise ValueError(f"LLM_API_KEY is required for provider '{self.LLM_PROVIDER}' in production.")
            if self.DEBUG:
                logger.warning("DEBUG=true in production environment — not recommended.")
        else:
            if self.JWT_SECRET_KEY == "change-this-to-a-secure-random-string":
                logger.warning("Using default JWT_SECRET_KEY — not suitable for production.")
            if self.LLM_PROVIDER != "ollama" and not self.LLM_API_KEY:
                logger.warning("LLM API key not set — LLM queries will fail.")

        return self

    def get_active_modules(self) -> dict:
        """Return a summary of which modules are active."""
        return {
            "llm": {"enabled": self.LLM_ENABLED, "provider": self.LLM_PROVIDER, "model": self.LLM_MODEL},
            "embedding": {"enabled": self.EMBEDDING_ENABLED, "provider": self.EMBEDDING_PROVIDER, "model": self.EMBEDDING_MODEL},
            "reranker": {"enabled": self.RERANKER_ENABLED, "model": self.RERANKER_MODEL},
            "retrieval": {"mode": self.RETRIEVAL_MODE},
            "query_rewriter": {"enabled": self.QUERY_REWRITER_ENABLED},
            "context_compression": {"enabled": self.CONTEXT_COMPRESSION_ENABLED},
            "guardrails": {"enabled": self.GUARDRAILS_ENABLED},
            "cache": {"enabled": self.CACHE_ENABLED},
            "profile": self.AI_PROFILE or "custom",
        }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
