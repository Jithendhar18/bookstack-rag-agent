"""LangGraph agent node implementations."""

import logging
import time
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable

from app.agents.state import AgentState
from app.retrieval.retrieval_service import RetrievalService
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


SYSTEM_PROMPT = """You are a knowledgeable AI assistant that answers questions based on documentation from BookStack.

INSTRUCTIONS:
- Answer questions accurately using ONLY the provided context documents.
- If the context doesn't contain enough information, say so clearly.
- Cite your sources by referencing document titles.
- Be concise but thorough.
- If asked about something not in the provided context, state that the information is not available in the current documentation.

CONTEXT DOCUMENTS:
{context}
"""


class AgentNodes:
    """Node implementations for the RAG agent graph."""

    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
        )

    @traceable(name="input_node")
    def input_node(self, state: AgentState) -> Dict[str, Any]:
        """Process and validate input."""
        query = state["query"].strip()
        if not query:
            return {"error": "Empty query", "answer": "Please provide a question."}

        return {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "metadata": {**state.get("metadata", {}), "start_time": time.time()},
        }

    @traceable(name="retriever_node")
    def retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant documents from vector store."""
        if state.get("error"):
            return {}

        query = state["query"]
        tenant_id = state.get("tenant_id", "default")

        results = self.retrieval_service.retrieve(
            query=query,
            top_k=settings.TOP_K_RETRIEVAL,
            tenant_id=tenant_id,
        )

        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return {"retrieved_documents": results}

    @traceable(name="reranker_node")
    def reranker_node(self, state: AgentState) -> Dict[str, Any]:
        """Rerank retrieved documents."""
        if state.get("error"):
            return {}

        documents = state.get("retrieved_documents", [])
        if not documents:
            return {"reranked_documents": [], "error": "No documents found for this query."}

        reranked = self.retrieval_service.rerank(
            query=state["query"],
            documents=documents,
            top_k=settings.TOP_K_RERANK,
        )

        logger.info(f"Reranked to {len(reranked)} documents")
        return {"reranked_documents": reranked}

    @traceable(name="llm_reasoning_node")
    def llm_reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate answer using LLM with retrieved context."""
        if state.get("error"):
            return {"answer": state["error"], "sources": []}

        documents = state.get("reranked_documents", [])

        # Build context from retrieved documents
        context_parts = []
        sources = []
        for i, doc in enumerate(documents):
            title = doc.get("metadata", {}).get("title", f"Document {i+1}")
            text = doc.get("text", "")
            context_parts.append(f"[{i+1}] {title}\n{text}")
            sources.append({
                "chunk_id": doc.get("id", ""),
                "document_title": title,
                "content": text[:500],
                "score": doc.get("rerank_score", doc.get("score", 0)),
                "metadata": doc.get("metadata", {}),
            })

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context))
        human_msg = HumanMessage(content=state["query"])

        response = self.llm.invoke([system_msg] + state.get("messages", []))

        return {
            "answer": response.content,
            "sources": sources,
            "messages": [AIMessage(content=response.content)],
            "metadata": {
                **state.get("metadata", {}),
                "end_time": time.time(),
                "documents_used": len(documents),
            },
        }

    @traceable(name="tool_node")
    def tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools if needed (extensibility point)."""
        # Placeholder for future tool integrations (e.g., calculator, web search)
        return {"tool_results": None}

    @traceable(name="response_node")
    def response_node(self, state: AgentState) -> Dict[str, Any]:
        """Finalize the response."""
        metadata = state.get("metadata", {})
        start_time = metadata.get("start_time", time.time())
        latency = (time.time() - start_time) * 1000

        return {
            "metadata": {
                **metadata,
                "latency_ms": latency,
                "total_sources": len(state.get("sources", [])),
            }
        }
