"""Query API routes — uses LangGraph agent."""

import uuid
import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import ChatSession, ChatMessage, AuditLog, AuditAction
from app.auth.dependencies import get_current_user, CurrentUser
from app.agents.graph import run_agent_query
from app.schemas.schemas import QueryRequest, QueryResponse, SourceDocument

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit a query to the RAG agent."""
    start = time.time()

    # Get or create chat session
    session_id = request.session_id
    if session_id:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == current_user.user_id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        session = ChatSession(
            id=uuid.uuid4(),
            user_id=current_user.user_id,
            title=request.query[:100],
            tenant_id=current_user.tenant_id,
        )
        db.add(session)
        await db.flush()
        session_id = session.id

    # Run agent
    result = await run_agent_query(
        query=request.query,
        tenant_id=current_user.tenant_id,
        session_id=str(session_id),
    )

    latency_ms = (time.time() - start) * 1000

    # Store messages
    db.add(ChatMessage(
        id=uuid.uuid4(),
        session_id=session_id,
        role="user",
        content=request.query,
    ))
    db.add(ChatMessage(
        id=uuid.uuid4(),
        session_id=session_id,
        role="assistant",
        content=result["answer"],
        sources=result.get("sources", []),
    ))

    # Audit log
    db.add(AuditLog(
        id=uuid.uuid4(),
        user_id=current_user.user_id,
        action=AuditAction.QUERY,
        resource="query",
        details={"query": request.query[:200], "latency_ms": latency_ms},
        tenant_id=current_user.tenant_id,
    ))

    await db.commit()

    sources = [
        SourceDocument(
            chunk_id=s.get("chunk_id", ""),
            document_title=s.get("document_title", ""),
            content=s.get("content", ""),
            score=s.get("score", 0),
            metadata=s.get("metadata", {}),
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        session_id=session_id,
        trace_id=result.get("metadata", {}).get("trace_id"),
        latency_ms=latency_ms,
    )
