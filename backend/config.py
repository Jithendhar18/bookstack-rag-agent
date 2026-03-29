"""Application configuration loaded from environment variables."""

import logging
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import model_validator
from functools import lru_cache

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Central application settings — fully configurable via .env."""

    # ─── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5435/bookstack_rag"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5435/bookstack_rag"

    # ─── BookStack ────────────────────────────────────────────────────
    BOOKSTACK_BASE_URL: str = ""
    BOOKSTACK_TOKEN_ID: str = ""
    BOOKSTACK_TOKEN_SECRET: str = ""

    # ─── LLM ──────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "groq"  # openai | openrouter | groq | ollama
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ─── Embeddings ───────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768

    # ─── Vector Store (Qdrant) ────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "bookstack_documents"

    # ─── Retrieval ────────────────────────────────────────────────────
    RETRIEVAL_MODE: str = "hybrid"  # dense | hybrid | keyword
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 8
    SIMILARITY_THRESHOLD: float = 0.4
    DENSE_WEIGHT: float = 0.8
    BM25_WEIGHT: float = 0.2

    # ─── Reranker ─────────────────────────────────────────────────────
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ─── Chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100

    # ─── Auth / JWT ───────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-this-to-a-secure-random-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ADMIN_DEFAULT_PASSWORD: str = "admin1234"

    # ─── Optional toggles ────────────────────────────────────────────
    DEBUG: bool = False
    GUARDRAILS_ENABLED: bool = True
    QUERY_REWRITER_ENABLED: bool = True
    CONTEXT_COMPRESSION_ENABLED: bool = True
    CACHE_ENABLED: bool = True
    STREAMING_ENABLED: bool = True
    MAX_CONTEXT_TOKENS: int = 4096
    MMR_LAMBDA: float = 0.7
    HALLUCINATION_THRESHOLD: float = 0.3
    MIN_SUPPORTING_CHUNKS: int = 1
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> "Settings":
        """Resolve LLM API key and base URL based on provider."""
        if not self.LLM_BASE_URL:
            base_urls = {
                "openai": "https://api.openai.com/v1",
                "openrouter": "https://openrouter.ai/api/v1",
                "groq": "https://api.groq.com/openai/v1",
                "ollama": self.OLLAMA_BASE_URL + "/v1",
            }
            self.LLM_BASE_URL = base_urls.get(self.LLM_PROVIDER)

        if not self.LLM_API_KEY and self.LLM_PROVIDER == "ollama":
            self.LLM_API_KEY = "not-needed"

        if self.JWT_SECRET_KEY == "change-this-to-a-secure-random-string":
            logger.warning("Using default JWT_SECRET_KEY — not suitable for production.")

        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()
