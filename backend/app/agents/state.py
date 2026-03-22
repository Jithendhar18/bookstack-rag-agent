"""LangGraph agent state definition."""

from typing import List, Optional, Any, TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """State that flows through the LangGraph agent."""
    # User query
    query: str
    # Rewritten query for better retrieval
    rewritten_query: Optional[str]
    # Tenant context for RBAC filtering
    tenant_id: str
    # Session ID for conversation continuity
    session_id: Optional[str]
    # Chat history
    messages: Annotated[List[BaseMessage], operator.add]
    # Retrieved documents from vector store (hybrid)
    retrieved_documents: List[dict]
    # Reranked documents
    reranked_documents: List[dict]
    # Compressed context documents
    compressed_documents: List[dict]
    # Generated answer
    answer: str
    # Source citations
    sources: List[dict]
    # Response validation result
    validation_result: Optional[dict]
    # Error tracking
    error: Optional[str]
    # Metadata for tracing
    metadata: dict
