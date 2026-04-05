"""LangGraph agent graph definition — optimized pipeline.

Optimized flow (conditional rewrite saves an LLM call on most queries):
    Input → Retriever → [ConditionalRewrite → Re-retrieve?] → Reranker
        → ContextCompressor → LLM → ResponseValidator → Response → END

Key change from the original:
- Retrieval runs FIRST with the original query.
- Query rewrite only fires when initial retrieval is poor (< MIN_RETRIEVAL_RESULTS
  results or low similarity scores), saving ~300-500ms per well-formed query.
- A second retrieval pass runs only when a rewrite actually changed the query.
"""

import logging
from typing import Dict, Any, Optional, AsyncIterator, List

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langsmith import traceable

from app.agents.state import AgentState
from app.agents.nodes import AgentNodes
from app.core.cache import get_cache

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def is_blocked(state: AgentState) -> str:
    """Conditional edge after input: skip pipeline if query was blocked."""
    if state.get("error"):
        return "response"
    return "retriever"


def has_documents_after_initial(state: AgentState) -> str:
    """After initial retrieval: proceed to conditional rewrite."""
    if state.get("error"):
        return "response"
    docs = state.get("retrieved_documents", [])
    if not docs:
        # No results at all — still try a rewrite before giving up
        return "query_rewrite"
    return "query_rewrite"


def needs_re_retrieval(state: AgentState) -> str:
    """After query rewrite: re-retrieve only if the query was actually changed."""
    if state.get("error"):
        return "response"

    rewritten = state.get("rewritten_query")
    original = state.get("query")

    # If the rewrite produced a different query, do a second retrieval pass
    if rewritten and rewritten != original:
        return "re_retriever"

    # Otherwise, go straight to reranking with existing results
    docs = state.get("retrieved_documents", [])
    if not docs:
        return "response"
    return "reranker"


def has_documents_after_rerank(state: AgentState) -> str:
    """After reranking: skip LLM if no documents survived."""
    if state.get("error"):
        return "response"
    docs = state.get("retrieved_documents", [])
    if not docs:
        return "response"
    return "reranker"


def build_agent_graph() -> StateGraph:
    """Construct the optimized LangGraph RAG agent workflow."""
    nodes = AgentNodes()

    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("input", nodes.input_node)
    graph.add_node("retriever", nodes.retriever_node)
    graph.add_node("query_rewrite", nodes.query_rewrite_node)
    graph.add_node("re_retriever", nodes.retriever_node)  # Same node, second pass
    graph.add_node("reranker", nodes.reranker_node)
    graph.add_node("context_compressor", nodes.context_compressor_node)
    graph.add_node("llm_reasoning", nodes.llm_reasoning_node)
    graph.add_node("response_validator", nodes.response_validator_node)
    graph.add_node("response", nodes.response_node)

    # Entry point
    graph.set_entry_point("input")

    # Input → blocked check → retriever or response
    graph.add_conditional_edges(
        "input",
        is_blocked,
        {"retriever": "retriever", "response": "response"},
    )

    # Retriever → conditional query rewrite
    graph.add_edge("retriever", "query_rewrite")

    # Query rewrite → re-retrieve (if changed) or reranker (if unchanged)
    graph.add_conditional_edges(
        "query_rewrite",
        needs_re_retrieval,
        {
            "re_retriever": "re_retriever",
            "reranker": "reranker",
            "response": "response",
        },
    )

    # Re-retriever → reranker (or response if still empty)
    graph.add_conditional_edges(
        "re_retriever",
        has_documents_after_rerank,
        {"reranker": "reranker", "response": "response"},
    )

    # Reranker → context compression → LLM → validation → response → END
    graph.add_edge("reranker", "context_compressor")
    graph.add_edge("context_compressor", "llm_reasoning")
    graph.add_edge("llm_reasoning", "response_validator")
    graph.add_edge("response_validator", "response")
    graph.add_edge("response", END)

    return graph


# Compiled graph singleton
_compiled_graph = None


def get_agent():
    """Get or create the compiled agent graph."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_agent_graph()
        _compiled_graph = graph.compile()
        logger.info("LangGraph agent compiled successfully")
    return _compiled_graph


@traceable(name="rag_agent_query", run_type="chain")
async def run_agent_query(
    query: str,
    tenant_id: str = "default",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    messages: Optional[List[BaseMessage]] = None,
) -> Dict[str, Any]:
    """Execute a query through the RAG agent pipeline with caching."""
    # Check cache first
    try:
        cache = await get_cache()
        cached = await cache.get_query_result(query, tenant_id)
        if cached:
            logger.info("Returning cached result")
            cached["metadata"] = {**cached.get("metadata", {}), "cached": True}
            return cached
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")

    agent = get_agent()

    initial_state: AgentState = {
        "query": query,
        "rewritten_query": None,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "messages": messages or [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "compressed_documents": [],
        "answer": "",
        "sources": [],
        "validation_result": None,
        "error": None,
        "metadata": {},
    }

    try:
        logger.info(f"Starting agent invocation for query: {query[:100]}...")
        result = await agent.ainvoke(initial_state)
        logger.info(f"Agent invocation completed successfully")
    except Exception as e:
        logger.error(f"Agent invocation failed: {type(e).__name__}: {e}", exc_info=True)
        raise

    response = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
    }

    # Cache the result
    try:
        cache = await get_cache()
        await cache.set_query_result(query, tenant_id, response)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")

    return response


async def stream_agent_query(
    query: str,
    tenant_id: str = "default",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    messages: Optional[List[BaseMessage]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream query execution through the RAG agent, yielding node events."""
    agent = get_agent()

    initial_state: AgentState = {
        "query": query,
        "rewritten_query": None,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "messages": messages or [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "compressed_documents": [],
        "answer": "",
        "sources": [],
        "validation_result": None,
        "error": None,
        "metadata": {},
    }

    async for event in agent.astream(initial_state):
        for node_name, node_output in event.items():
            yield {"node": node_name, "data": node_output}
