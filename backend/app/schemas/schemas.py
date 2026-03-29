"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class RoleEnum(str, Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"


# ─── Auth ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)


class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., max_length=255)
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=255)
    tenant_id: str = Field(default="default", max_length=100)


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: UUID
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    role: RoleEnum
    tenant_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None
    role: Optional[RoleEnum] = None


# ─── Query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[UUID] = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None


class SourceDocument(BaseModel):
    chunk_id: str
    document_title: str
    content: str
    score: float
    source_url: Optional[str] = None
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: UUID
    trace_id: Optional[str] = None
    latency_ms: float


# ─── Ingestion ───────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    bookstack_type: str = Field(default="pages", pattern="^(pages|books|chapters|shelves)$")
    bookstack_ids: Optional[List[int]] = None  # None = ingest all
    force_reindex: bool = False


class IngestResponse(BaseModel):
    task_id: str
    status: str
    documents_queued: int
    message: Optional[str] = None


class IngestionStatusResponse(BaseModel):
    pending: int
    processing: int
    completed: int
    failed: int
    total: int


class DocumentResponse(BaseModel):
    id: UUID
    bookstack_id: int
    bookstack_type: str
    title: str
    status: str
    chunk_count: int = 0
    book_id: Optional[int] = None
    book_name: Optional[str] = None
    chapter_id: Optional[int] = None
    chapter_name: Optional[str] = None
    ingested_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Book / Chapter hierarchy schemas ──────────────────────────────────────────

class PageSummaryResponse(BaseModel):
    """Lightweight page node within a book hierarchy response."""
    id: UUID
    bookstack_id: int
    title: str
    slug: Optional[str] = None
    chapter_id: Optional[int] = None
    chapter_name: Optional[str] = None
    status: str
    chunk_count: int = 0
    source_url: Optional[str] = None
    ingested_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


class ChapterGroupResponse(BaseModel):
    """Group of pages sharing the same chapter (chapter_id=None = direct book pages)."""
    chapter_id: Optional[int]
    chapter_name: Optional[str]
    page_count: int
    pages: List[PageSummaryResponse]


class BookSummaryResponse(BaseModel):
    """Per-book aggregate — used in the /ingestion/books listing."""
    book_id: int
    book_name: Optional[str] = None
    page_count: int
    chunk_count: int


class BookHierarchyResponse(BaseModel):
    """Full Book → Chapter → Page hierarchy for a single book."""
    book_id: int
    book_name: Optional[str] = None
    total_pages: int
    total_chunks: int
    chapters: List[ChapterGroupResponse]


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatMessageSchema(BaseModel):
    role: str
    content: str
    sources: List[SourceDocument] = []
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    id: UUID
    title: Optional[str]
    messages: List[ChatMessageSchema] = []
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ChatSessionListItem(BaseModel):
    id: UUID
    title: Optional[str]
    message_count: int
    last_message_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


class FrequentQuestion(BaseModel):
    query: str
    count: int
    last_asked_at: datetime


# ─── Admin / Metrics ────────────────────────────────────────────────────────

class SystemMetrics(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int
    total_users: int
    total_queries: int
    total_books: int
    documents_by_status: dict[str, int]
    documents_by_book: dict[str, int]   # book_id (str) -> page count
    avg_query_latency_ms: Optional[float]


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int


# ─── Evaluation ──────────────────────────────────────────────────────────────

class EvalQuerySchema(BaseModel):
    query: str
    expected_answer: Optional[str] = None
    expected_sources: Optional[List[str]] = None


class EvalResultSchema(BaseModel):
    query: str
    answer: str
    expected_answer: Optional[str] = None
    sources_count: int
    latency_ms: float
    grounding_confidence: Optional[float] = None
    passed: bool
