"""Query API routes - HTTP handlers with dependency injection."""

import time
import logging
from typing import Optional, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.auth.dependencies import get_current_user, CurrentUser
from app.services.query_service import QueryService
from app.schemas.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
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
