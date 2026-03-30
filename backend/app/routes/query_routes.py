"""Query API routes - HTTP handlers with dependency injection."""

import json
import time
import logging
from typing import Optional, Annotated, AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.auth.dependencies import get_current_user, require_roles, CurrentUser
from app.services.query_service import QueryService
from app.schemas.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    ChatSessionResponse,
    ChatSessionListItem,
    FrequentQuestion,
)
from app.agents.graph import run_agent_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


async def get_query_service(db: AsyncSession = Depends(get_db)) -> QueryService:
    """Dependency injection for QueryService."""
    return QueryService(db)


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    query_service: Annotated[QueryService, Depends(get_query_service)],
):
    """
    Submit a query to the RAG agent.
    
    Processes a user query through the LangGraph agent, manages chat sessions,
    stores messages, and logs the query execution.
    
    Args:
        request: QueryRequest with query text and optional session_id
        current_user: Current authenticated user
        query_service: Injected query service
        
    Returns:
        QueryResponse with answer, sources, session_id, and latency
    """
    start = time.time()

    # Get or create chat session
    session = await query_service.get_or_create_session(
        request.session_id,
        current_user.user_id,
        current_user.tenant_id,
        request.query[:100],
    )

    try:
        logger.info(f"Running agent query: {request.query[:100]}")
        
        # Run agent
        result = await run_agent_query(
            query=request.query,
            tenant_id=current_user.tenant_id,
            session_id=str(session.id),
        )
        
        logger.info("Agent query completed successfully")
        
    except Exception as e:
        logger.error(
            f"Agent query failed: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing failed",
        )

    latency_ms = (time.time() - start) * 1000

    # Store messages in database
    try:
        await query_service.save_chat_message(
            session.id,
            "user",
            request.query,
            token_count=len(request.query.split()),
        )
        
        await query_service.save_chat_message(
            session.id,
            "assistant",
            result["answer"],
            sources=result.get("sources", []),
            token_count=len(result["answer"].split()),
        )

        # Log audit event
        await query_service.log_query_audit(
            current_user.user_id,
            current_user.tenant_id,
            request.query,
            latency_ms,
        )
        
        logger.info("Database commit successful")
        
    except Exception as e:
        logger.error(
            f"Database save failed: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save query results",
        )

    # Format sources
    sources = [
        SourceDocument(
            chunk_id=s.get("chunk_id", ""),
            document_title=s.get("document_title", ""),
            content=s.get("content", ""),
            score=s.get("score", 0.0),
            source_url=s.get("source_url")
            or s.get("metadata", {}).get("source_url"),
            metadata=s.get("metadata", {}),
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        session_id=session.id,
        trace_id=result.get("metadata", {}).get("trace_id"),
        latency_ms=latency_ms,
    )


@router.get("/history", response_model=list[ChatSessionListItem])
async def list_sessions(
    page: int = 1,
    page_size: int = 20,
    current_user: Annotated[CurrentUser, Depends(get_current_user)] = None,
    query_service: Annotated[QueryService, Depends(get_query_service)] = None,
):
    """
    List all chat sessions for the current user.
    
    Paginated list of sessions ordered by most recent first.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of sessions per page
        current_user: Current authenticated user
        query_service: Injected query service
        
    Returns:
        List of ChatSessionListItem (id, title, message_count, last_message_at, created_at)
    """
    skip = (page - 1) * page_size
    
    sessions = await query_service.get_user_sessions(
        current_user.user_id,
        skip=skip,
        limit=page_size
    )
    
    # Build response with message counts
    result = []
    for session in sessions:
        messages = await query_service.get_session_history(
            session.id, current_user.user_id
        )
        
        # Get last message timestamp
        last_message_at = None
        if messages:
            last_message_at = messages[-1].created_at
        
        result.append(
            ChatSessionListItem(
                id=session.id,
                title=session.title,
                message_count=len(messages),
                last_message_at=last_message_at,
                created_at=session.created_at,
            )
        )
    
    return result


@router.get("/history/{session_id}", response_model=ChatSessionResponse)
async def get_session(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)] = None,
    query_service: Annotated[QueryService, Depends(get_query_service)] = None,
):
    """
    Get a specific chat session with all messages.
    
    Args:
        session_id: Chat session UUID
        current_user: Current authenticated user
        query_service: Injected query service
        
    Returns:
        ChatSessionResponse with session details and messages
    """
    # Get session (authorization check done in get_session_by_id)
    session = await query_service.get_session_by_id(session_id, current_user.user_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    
    # Get messages
    messages = await query_service.get_session_history(
        session_id, current_user.user_id
    )
    
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        messages=[
            {
                "role": msg.role,
                "content": msg.content,
                "sources": msg.sources or [],
                "created_at": msg.created_at,
            }
            for msg in messages
        ],
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.delete("/history/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)] = None,
    query_service: Annotated[QueryService, Depends(get_query_service)] = None,
):
    """
    Delete a chat session and all its messages.
    
    Args:
        session_id: Chat session UUID
        current_user: Current authenticated user
        query_service: Injected query service
        
    Returns:
        204 No Content
    """
    deleted = await query_service.delete_session(session_id, current_user.user_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )


async def stream_agent_query(
    query: str,
    tenant_id: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream agent query results as Server-Sent Events (SSE).
    
    Yields JSON events from the LangGraph agent for real-time progress.
    
    Args:
        query: User query text
        tenant_id: Tenant identifier
        session_id: Chat session ID
        
    Yields:
        JSON-formatted SSE events with node updates
    """
    try:
        # Run agent and stream results
        result = await run_agent_query(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id,
        )
        
        # Emit final result
        sources = [
            {
                "chunk_id": s.get("chunk_id", ""),
                "document_title": s.get("document_title", ""),
                "content": s.get("content", ""),
                "score": s.get("score", 0.0),
                "source_url": s.get("source_url") or s.get("metadata", {}).get("source_url"),
                "metadata": s.get("metadata", {}),
            }
            for s in result.get("sources", [])
        ]
        
        yield f'data: {json.dumps({"node": "response", "answer": result["answer"], "sources": sources})}\n\n'
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Streaming query failed: {type(e).__name__}: {e}", exc_info=True)
        yield f'data: {json.dumps({"error": "Query processing failed"})}\n\n'
        yield "data: [DONE]\n\n"


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)] = None,
    query_service: Annotated[QueryService, Depends(get_query_service)] = None,
):
    """
    Stream a query to the RAG agent in real-time.
    
    Returns Server-Sent Events (SSE) showing agent progress.
    Note: Streaming queries are NOT saved to history.
    
    Args:
        request: QueryRequest with query text
        current_user: Current authenticated user
        query_service: Injected query service
        
    Returns:
        StreamingResponse with text/event-stream media type
    """
    # Get or create session (for tracking purposes only)
    session = await query_service.get_or_create_session(
        request.session_id,
        current_user.user_id,
        current_user.tenant_id,
        request.query[:100],
    )
    
    return StreamingResponse(
        stream_agent_query(
            query=request.query,
            tenant_id=current_user.tenant_id,
            session_id=str(session.id),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/popular", response_model=list[FrequentQuestion])
async def get_popular_queries(
    limit: int = 10,
    days: int = 30,
    current_user: Annotated[CurrentUser, Depends(require_roles(["admin", "developer"]))] = None,
    query_service: Annotated[QueryService, Depends(get_query_service)] = None,
):
    """
    Get popular/trending queries for the tenant.
    
    Requires admin or developer role.
    
    Args:
        limit: Maximum number of queries to return
        days: Number of days to look back
        current_user: Current authenticated user (admin/developer only)
        query_service: Injected query service
        
    Returns:
        List of FrequentQuestion (query, count, last_asked_at)
    """
    queries = await query_service.get_popular_queries(
        current_user.tenant_id,
        limit=limit,
        days=days,
    )
    
    return [
        FrequentQuestion(
            query=q["query"],
            count=q["count"],
            last_asked_at=q["last_asked_at"],
        )
        for q in queries
    ]
