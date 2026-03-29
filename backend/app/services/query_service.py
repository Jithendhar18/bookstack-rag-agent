"""Query service - business logic for RAG queries and chat sessions."""

import logging
import uuid as uuid_module
from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatSession, ChatMessage, AuditLog
from app.repositories.chat_repository import ChatSessionRepository, ChatMessageRepository
from app.services.base import BaseService

logger = logging.getLogger(__name__)


class QueryService(BaseService):
    """
    Query service handling chat sessions, messages, and query operations.
    
    Contains business logic for query/chat workflows.
    """

    def __init__(self, db: AsyncSession):
        """Initialize query service with repositories."""
        super().__init__(db)
        self.session_repo = ChatSessionRepository(db)
        self.message_repo = ChatMessageRepository(db)

    async def get_or_create_session(
        self, session_id: Optional[UUID], user_id: UUID, tenant_id: str, title: str
    ) -> ChatSession:
        """
        Get existing chat session or create a new one.
        
        Args:
            session_id: Session ID (if continuing existing session)
            user_id: User UUID
            tenant_id: Tenant identifier
            title: Session title
            
        Returns:
            ChatSession instance
        """
        if session_id:
            session = await self.session_repo.get_by_id(session_id)
            if session and session.user_id == user_id:
                return session

        # Create new session
        new_session = ChatSession(
            id=uuid_module.uuid4(),
            user_id=user_id,
            title=title,
            tenant_id=tenant_id,
        )
        created_session = await self.session_repo.create(new_session)
        return created_session

    async def save_chat_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        sources: Optional[list] = None,
        token_count: Optional[int] = None,
    ) -> ChatMessage:
        """
        Save a chat message.
        
        Args:
            session_id: Chat session ID
            role: Message role (user, assistant, system)
            content: Message content
            sources: Source documents (optional)
            token_count: Token count for the message (optional)
            
        Returns:
            Saved ChatMessage instance
        """
        message = ChatMessage(
            id=uuid_module.uuid4(),
            session_id=session_id,
            role=role,
            content=content,
            sources=sources or [],
            token_count=token_count,
        )
        
        created_message = await self.message_repo.create(message)
        return created_message

    async def get_session_history(
        self, session_id: UUID, user_id: UUID, limit: int = 50
    ) -> list[ChatMessage]:
        """
        Get chat message history for a session.
        
        Args:
            session_id: Chat session ID
            user_id: User UUID (for authorization check)
            limit: Maximum number of messages to return
            
        Returns:
            List of ChatMessage instances
        """
        # First verify session belongs to user
        session = await self.session_repo.get_by_id(session_id)
        if session is None or session.user_id != user_id:
            return []
        
        # Get messages, limited
        return await self.message_repo.get_by_session_paginated(
            session_id, skip=0, limit=limit
        )

    async def get_user_sessions(
        self, user_id: UUID, skip: int = 0, limit: int = 50
    ) -> list[ChatSession]:
        """
        Get all chat sessions for a user.
        
        Args:
            user_id: User UUID
            skip: Number of sessions to skip
            limit: Maximum number of sessions to return
            
        Returns:
            List of ChatSession instances
        """
        return await self.session_repo.get_by_user(user_id, skip=skip, limit=limit)

    async def log_query_audit(
        self,
        user_id: UUID,
        tenant_id: str,
        query: str,
        latency_ms: float,
    ) -> AuditLog:
        """
        Log query execution for audit trail.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant identifier
            query: Query text (truncated to 200 chars)
            latency_ms: Query execution time in milliseconds
            
        Returns:
            Created AuditLog instance
        """
        audit_log = AuditLog(
            id=uuid_module.uuid4(),
            user_id=user_id,
            action="query",
            resource="query",
            details={"query": query[:200], "latency_ms": latency_ms},
            tenant_id=tenant_id,
        )
        self.db.add(audit_log)
        await self.db.commit()
        return audit_log
