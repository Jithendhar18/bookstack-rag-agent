"""Chat session and message data repositories."""

from typing import Optional, List
from uuid import UUID
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatSession, ChatMessage
from app.repositories.base import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    """Repository for ChatSession model providing chat session operations."""

    model_class = ChatSession

    async def get_by_user(
        self, user_id: UUID, skip: int = 0, limit: int = 50
    ) -> List[ChatSession]:
        """
        Get chat sessions for a user.
        
        Args:
            user_id: User UUID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of ChatSession instances
        """
        result = await self.db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(desc(ChatSession.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_by_user_and_tenant(
        self, user_id: UUID, tenant_id: str
    ) -> Optional[ChatSession]:
        """
        Get recent session for user in tenant.
        
        Args:
            user_id: User UUID
            tenant_id: Tenant identifier
            
        Returns:
            ChatSession instance or None
        """
        result = await self.db.execute(
            select(ChatSession)
            .where(
                (ChatSession.user_id == user_id)
                & (ChatSession.tenant_id == tenant_id)
            )
            .order_by(desc(ChatSession.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()


class ChatMessageRepository(BaseRepository[ChatMessage]):
    """Repository for ChatMessage model providing message operations."""

    model_class = ChatMessage

    async def get_by_session(self, session_id: UUID) -> List[ChatMessage]:
        """
        Get all messages in a chat session.
        
        Args:
            session_id: Chat session UUID
            
        Returns:
            List of ChatMessage instances
        """
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        return result.scalars().all()

    async def get_by_session_paginated(
        self, session_id: UUID, skip: int = 0, limit: int = 50
    ) -> List[ChatMessage]:
        """
        Get messages in a chat session with pagination.
        
        Args:
            session_id: Chat session UUID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of ChatMessage instances
        """
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
