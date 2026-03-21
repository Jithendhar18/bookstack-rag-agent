"""LangGraph agent graph definition and compilation — upgraded pipeline.

Flow:
    Input → QueryRewrite → HybridRetriever → Reranker
        → (Optional Tool Node) → ContextCompressor → LLMReasoning
        → ResponseValidator → Response → END
"""

import logging
from typing import Dict, Any, Optional, AsyncIterator
from uuid import UUID

from langgraph.graph import StateGraph, END
from langsmith import traceable

from app.agents.state import AgentState
from app.agents.nodes import AgentNodes
from app.core.cache import get_cache

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def should_use_tools(state: AgentState) -> str:
    """Conditional edge: decide whether to route through tool node."""
    if state.get("error"):
        return "context_compressor"
    # Future: inspect query or metadata for tool-requiring patterns
    return "context_compressor"


def has_documents(state: AgentState) -> str:
    """Conditional edge: check if retriever found documents."""
    if state.get("error"):
        return "response"
    docs = state.get("retrieved_documents", [])
    if not docs:
        return "response"
    return "reranker"


def is_blocked(state: AgentState) -> str:
    """Conditional edge after input: check if query was blocked by guardrails."""
    if state.get("error"):
        return "response"
    return "query_rewrite"


def build_agent_graph() -> StateGraph:
    """Construct the upgraded LangGraph RAG agent workflow.

    Graph structure:
        input → [is_blocked?]
            → query_rewrite → hybrid_retriever → [has_documents?]
                → reranker → [should_use_tools?]
                    → tool → context_compressor
                    → context_compressor
                → context_compressor → llm_reasoning → response_validator → response → END
            → response → END (blocked / no docs fallback)
    """
    nodes = AgentNodes()

    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("input", nodes.input_node)
    graph.add_node("query_rewrite", nodes.query_rewrite_node)
    graph.add_node("hybrid_retriever", nodes.hybrid_retriever_node)
    graph.add_node("reranker", nodes.reranker_node)
    graph.add_node("tool", nodes.tool_node)
    graph.add_node("context_compressor", nodes.context_compressor_node)
    graph.add_node("llm_reasoning", nodes.llm_reasoning_node)
    graph.add_node("response_validator", nodes.response_validator_node)
    graph.add_node("response", nodes.response_node)

    # Set entry point
    graph.set_entry_point("input")

    # Input → check if blocked → query_rewrite or response
    graph.add_conditional_edges(
        "input",
        is_blocked,
        {"query_rewrite": "query_rewrite", "response": "response"},
    )

    # query_rewrite → hybrid_retriever
    graph.add_edge("query_rewrite", "hybrid_retriever")

    # hybrid_retriever → [has_documents?] → reranker or response
    graph.add_conditional_edges(
        "hybrid_retriever",
        has_documents,
        {"reranker": "reranker", "response": "response"},
    )

    # reranker → [should_use_tools?] → context_compressor (or tool → context_compressor)
    graph.add_conditional_edges(
        "reranker",
        should_use_tools,
        {"context_compressor": "context_compressor"},
    )

    # tool → context_compressor
    graph.add_edge("tool", "context_compressor")

    # context_compressor → llm_reasoning
    graph.add_edge("context_compressor", "llm_reasoning")

    # llm_reasoning → response_validator
    graph.add_edge("llm_reasoning", "response_validator")

    # response_validator → response
    graph.add_edge("response_validator", "response")

    # response → END
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
        logger.info("LangGraph agent compiled successfully (v2 pipeline)")
    return _compiled_graph


@traceable(name="rag_agent_query", run_type="chain")
async def run_agent_query(
    query: str,
    tenant_id: str = "default",
    session_id: Optional[str] = None,
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
        "messages": [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "compressed_documents": [],
        "answer": "",
        "sources": [],
        "tool_results": None,
        "validation_result": None,
        "error": None,
        "metadata": {},
    }

    result = await agent.ainvoke(initial_state)

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
) -> AsyncIterator[Dict[str, Any]]:
    """Stream query execution through the RAG agent, yielding node events."""
    agent = get_agent()

    initial_state: AgentState = {
        "query": query,
        "rewritten_query": None,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "messages": [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "compressed_documents": [],
        "answer": "",
        "sources": [],
        "tool_results": None,
        "validation_result": None,
        "error": None,
        "metadata": {},
    }

    async for event in agent.astream(initial_state):
        for node_name, node_output in event.items():
            yield {"node": node_name, "data": node_output}
