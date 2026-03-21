"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Central application settings."""

    # App
    APP_NAME: str = "BookStack RAG Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/bookstack_rag"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5432/bookstack_rag"

    # BookStack
    BOOKSTACK_BASE_URL: str = ""
    BOOKSTACK_TOKEN_ID: str = ""
    BOOKSTACK_TOKEN_SECRET: str = ""

    # LLM
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"

    # Vector store
    VECTOR_STORE_TYPE: str = "faiss"  # "faiss" or "pgvector"
    FAISS_INDEX_PATH: str = "./data/faiss_index"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100

    # Auth / JWT
    JWT_SECRET_KEY: str = "change-this-to-a-secure-random-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # LangSmith
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "bookstack-rag-agent"
    LANGSMITH_TRACING_V2: bool = True
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "bookstack-rag-agent"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # Retrieval
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
