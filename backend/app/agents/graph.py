"""LangGraph agent graph definition and compilation."""

import logging
from typing import Dict, Any, Optional
from uuid import UUID

from langgraph.graph import StateGraph, END
from langsmith import traceable

from app.agents.state import AgentState
from app.agents.nodes import AgentNodes

logger = logging.getLogger(__name__)


def should_use_tools(state: AgentState) -> str:
    """Conditional edge: decide whether to route through tool node."""
    if state.get("error"):
        return "response"
    # Future: inspect query for tool-requiring patterns
    return "llm_reasoning"


def has_documents(state: AgentState) -> str:
    """Conditional edge: check if retriever found documents."""
    if state.get("error"):
        return "response"
    docs = state.get("retrieved_documents", [])
    if not docs:
        return "response"
    return "reranker"


def build_agent_graph() -> StateGraph:
    """Construct the LangGraph RAG agent workflow.

    Graph structure:
        input → retriever → [has_documents?]
            → reranker → [should_use_tools?]
                → llm_reasoning → response → END
                → tool → llm_reasoning → response → END
            → response → END (no docs fallback)
    """
    nodes = AgentNodes()

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("input", nodes.input_node)
    graph.add_node("retriever", nodes.retriever_node)
    graph.add_node("reranker", nodes.reranker_node)
    graph.add_node("tool", nodes.tool_node)
    graph.add_node("llm_reasoning", nodes.llm_reasoning_node)
    graph.add_node("response", nodes.response_node)

    # Set entry point
    graph.set_entry_point("input")

    # Define edges
    graph.add_edge("input", "retriever")

    graph.add_conditional_edges(
        "retriever",
        has_documents,
        {"reranker": "reranker", "response": "response"},
    )

    graph.add_conditional_edges(
        "reranker",
        should_use_tools,
        {"llm_reasoning": "llm_reasoning", "response": "response"},
    )

    graph.add_edge("tool", "llm_reasoning")
    graph.add_edge("llm_reasoning", "response")
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
) -> Dict[str, Any]:
    """Execute a query through the RAG agent pipeline.

    This is the main entry point traced by LangSmith.
    """
    agent = get_agent()

    initial_state: AgentState = {
        "query": query,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "messages": [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "answer": "",
        "sources": [],
        "tool_results": None,
        "error": None,
        "metadata": {},
    }

    result = await agent.ainvoke(initial_state)

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
    }
