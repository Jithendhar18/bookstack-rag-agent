"""Query API routes — uses LangGraph agent with streaming support."""

import json
import uuid
import time
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import ChatSession, ChatMessage, AuditLog
from app.auth.dependencies import get_current_user, require_roles, CurrentUser
from app.agents.graph import run_agent_query, stream_agent_query
from app.schemas.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    ChatSessionResponse, ChatSessionListItem, ChatMessageSchema, FrequentQuestion,
)

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

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
    try:
        logger.info(f"Running agent query: {request.query[:100]}")
        result = await run_agent_query(
            query=request.query,
            tenant_id=current_user.tenant_id,
            session_id=str(session_id),
        )
        logger.info("Agent query completed successfully")
    except Exception as e:
        logger.error(f"Agent query failed: {type(e).__name__}: {e}", exc_info=True)
        raise

    latency_ms = (time.time() - start) * 1000

    # Store messages
    try:
        db.add(ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="user",
            content=request.query,
            token_count=len(request.query.split()),
        ))
        db.add(ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="assistant",
            content=result["answer"],
            sources=result.get("sources", []),
            token_count=len(result["answer"].split()),
        ))

        # Audit log
        db.add(AuditLog(
            id=uuid.uuid4(),
            user_id=current_user.user_id,
            action="query",
            resource="query",
            details={"query": request.query[:200], "latency_ms": latency_ms},
            tenant_id=current_user.tenant_id,
        ))

        await db.commit()
        logger.info("Database commit successful")
    except Exception as e:
        logger.error(f"Database save failed: {type(e).__name__}: {e}", exc_info=True)
        raise

    sources = [
        SourceDocument(
            chunk_id=s.get("chunk_id", ""),
            document_title=s.get("document_title", ""),
            content=s.get("content", ""),
            score=s.get("score", 0),
            source_url=s.get("source_url") or s.get("metadata", {}).get("source_url"),
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



@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Stream query results via SSE (Server-Sent Events)."""
    if not settings.STREAMING_ENABLED:
        raise HTTPException(status_code=400, detail="Streaming is disabled")

    async def event_generator():
        try:
            async for event in stream_agent_query(
                query=request.query,
                tenant_id=current_user.tenant_id,
                session_id=str(request.session_id) if request.session_id else None,
            ):
                node = event.get("node", "unknown")
                data = event.get("data", {})

                # Yield progress events for each node
                payload = {
                    "node": node,
                    "answer": data.get("answer", ""),
                    "sources": data.get("sources", []),
                    "metadata": data.get("metadata", {}),
                }
                yield f"data: {json.dumps(payload, default=str)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Chat History ─────────────────────────────────────────────────────────────

@router.get("/history", response_model=List[ChatSessionListItem])
async def list_chat_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List the current user's chat sessions, newest first."""
    result = await db.execute(
        select(
            ChatSession.id,
            ChatSession.title,
            ChatSession.created_at,
            ChatSession.updated_at,
            func.count(ChatMessage.id).label("message_count"),
            func.max(ChatMessage.created_at).label("last_message_at"),
        )
        .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
        .where(ChatSession.user_id == current_user.user_id)
        .group_by(ChatSession.id)
        .order_by(desc(ChatSession.updated_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = result.all()
    return [
        ChatSessionListItem(
            id=row.id,
            title=row.title,
            message_count=row.message_count,
            last_message_at=row.last_message_at,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get("/history/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific chat session with all messages and source links."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.user_id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = msg_result.scalars().all()

    def _clean_sources(raw: list) -> List[SourceDocument]:
        out = []
        for s in (raw or []):
            meta = s.get("metadata", {})
            out.append(SourceDocument(
                chunk_id=s.get("chunk_id", ""),
                document_title=s.get("document_title", ""),
                content=s.get("content", ""),
                score=s.get("score", 0.0),
                source_url=s.get("source_url") or meta.get("source_url"),
                metadata=meta,
            ))
        return out

    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[
            ChatMessageSchema(
                role=msg.role,
                content=msg.content,
                sources=_clean_sources(msg.sources or []),
                created_at=msg.created_at,
            )
            for msg in messages
        ],
    )


@router.delete("/history/{session_id}", status_code=204)
async def delete_chat_session(
    session_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat session and all its messages."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.user_id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    await db.delete(session)
    await db.commit()


# ─── Popular Questions ────────────────────────────────────────────────────────

@router.get("/popular", response_model=List[FrequentQuestion])
async def popular_questions(
    limit: int = Query(default=10, ge=1, le=50),
    current_user: CurrentUser = Depends(require_roles(["admin", "developer"])),
    db: AsyncSession = Depends(get_db),
):
    """Return the most frequently asked questions in this tenant. Admin/developer only."""
    from sqlalchemy import literal_column
    q_col = AuditLog.details["query"].astext
    count_col = func.count(AuditLog.id)
    result = await db.execute(
        select(
            q_col.label("query"),
            count_col.label("count"),
            func.max(AuditLog.created_at).label("last_asked_at"),
        )
        .where(
            AuditLog.action == "query",
            AuditLog.tenant_id == current_user.tenant_id,
            q_col.isnot(None),
        )
        .group_by(q_col)
        .order_by(count_col.desc())
        .limit(limit)
    )
    rows = result.all()
    return [
        FrequentQuestion(query=row.query, count=row.count, last_asked_at=row.last_asked_at)
        for row in rows
    ]
