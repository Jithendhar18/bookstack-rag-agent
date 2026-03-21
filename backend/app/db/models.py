"""SQLAlchemy ORM models for the full database schema."""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, UniqueConstraint, Index, Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from app.db.session import Base
import enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class RoleName(str, enum.Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AuditAction(str, enum.Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    QUERY = "query"
    INGEST = "ingest"
    CREATE_USER = "create_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    UPDATE_ROLE = "update_role"


# ─── Users & RBAC ───────────────────────────────────────────────────────────

class Role(Base):
    __tablename__ = "roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(SAEnum(RoleName), unique=True, nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    permissions = relationship("Permission", back_populates="role", cascade="all, delete-orphan")
    users = relationship("User", back_populates="role")


class Permission(Base):
    __tablename__ = "permissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_id = Column(UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
    resource = Column(String(100), nullable=False)  # e.g. "ingestion", "query", "admin"
    action = Column(String(50), nullable=False)  # e.g. "read", "write", "delete"
    created_at = Column(DateTime, default=datetime.utcnow)

    role = relationship("Role", back_populates="permissions")

    __table_args__ = (
        UniqueConstraint("role_id", "resource", "action", name="uq_role_resource_action"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    tenant_id = Column(String(100), nullable=False, default="default", index=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey("roles.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    role = relationship("Role", back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")

    __table_args__ = (
        Index("ix_users_tenant_email", "tenant_id", "email"),
    )


# ─── Documents & Chunks ─────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bookstack_id = Column(Integer, nullable=False, index=True)
    bookstack_type = Column(String(50), nullable=False)  # "page", "book", "chapter", "shelf"
    title = Column(String(500), nullable=False)
    slug = Column(String(500))
    book_id = Column(Integer, index=True)
    chapter_id = Column(Integer, index=True)
    content_hash = Column(String(64), nullable=False)  # SHA-256 for dedup
    html_content = Column(Text)
    plain_content = Column(Text)
    status = Column(SAEnum(DocumentStatus), default=DocumentStatus.PENDING)
    tenant_id = Column(String(100), nullable=False, default="default", index=True)
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ingested_at = Column(DateTime)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("bookstack_id", "bookstack_type", "tenant_id", name="uq_bookstack_doc"),
        Index("ix_documents_tenant_status", "tenant_id", "status"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    token_count = Column(Integer)
    char_count = Column(Integer)
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
    embedding_metadata = relationship("EmbeddingMetadata", back_populates="chunk", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_index"),
        Index("ix_chunks_content_hash", "content_hash"),
    )


class EmbeddingMetadata(Base):
    __tablename__ = "embeddings_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), unique=True, nullable=False)
    vector_store_id = Column(String(255), nullable=False)  # ID in FAISS/PGVector
    model_name = Column(String(255), nullable=False)
    dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunk = relationship("Chunk", back_populates="embedding_metadata")


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255))
    tenant_id = Column(String(100), nullable=False, default="default", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB, default={})
    sources = Column(JSONB, default=[])
    token_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


# ─── Audit ───────────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(SAEnum(AuditAction), nullable=False)
    resource = Column(String(100))
    resource_id = Column(String(255))
    details = Column(JSONB, default={})
    ip_address = Column(String(45))
    tenant_id = Column(String(100), nullable=False, default="default", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_logs_tenant_action", "tenant_id", "action"),
        Index("ix_audit_logs_created_at", "created_at"),
    )
