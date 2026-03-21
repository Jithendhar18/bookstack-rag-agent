"""LangGraph agent node implementations — upgraded pipeline."""

import logging
import time
from typing import Dict, Any, List

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable

from app.agents.state import AgentState
from app.retrieval.retrieval_service import RetrievalService
from app.core.guardrails import GuardrailsService
from app.agents.tools import get_tool_registry
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

QUERY_REWRITE_PROMPT = """You are a query optimization assistant. Your task is to rewrite the user's query into a clear, specific search query optimized for document retrieval.

Rules:
- Expand abbreviations and acronyms
- Fix incomplete questions into full questions
- Preserve the original intent
- Make the query more specific and searchable
- Output ONLY the rewritten query, nothing else

User query: {query}

Rewritten query:"""

RESPONSE_VALIDATION_PROMPT = """You are a fact-checking assistant. Evaluate whether the given answer is factually grounded in the provided source documents.

Answer: {answer}

Source documents:
{sources}

Evaluate:
1. Is the answer supported by the sources? (yes/partially/no)
2. Are there any claims not found in the sources?
3. Confidence score (0.0 to 1.0)

Respond in this exact format:
GROUNDED: yes/partially/no
UNSUPPORTED_CLAIMS: <list or "none">
CONFIDENCE: <0.0-1.0>"""


class AgentNodes:
    """Node implementations for the upgraded RAG agent graph."""

    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.guardrails = GuardrailsService()
        self.tool_registry = get_tool_registry()
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
        )
        self._fast_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.0,
            max_tokens=256,
            api_key=settings.OPENAI_API_KEY,
        )
        try:
            self._tokenizer = tiktoken.encoding_for_model(settings.LLM_MODEL)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # ─── Input Node ──────────────────────────────────────────────────────

    @traceable(name="input_node")
    def input_node(self, state: AgentState) -> Dict[str, Any]:
        """Process, validate input, and check for prompt injection."""
        query = state["query"].strip()
        if not query:
            return {"error": "Empty query", "answer": "Please provide a question."}

        # Guardrails: prompt injection check
        injection_check = self.guardrails.check_prompt_injection(query)
        if not injection_check["safe"]:
            return {
                "error": "blocked",
                "answer": injection_check["reason"],
            }

        return {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "metadata": {**state.get("metadata", {}), "start_time": time.time()},
        }

    # ─── Query Rewrite Node ─────────────────────────────────────────────

    @traceable(name="query_rewrite_node")
    def query_rewrite_node(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite user query for better retrieval."""
        if state.get("error"):
            return {}

        query = state["query"]

        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        response = self._fast_llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()

        # Fallback to original if rewrite is empty or too different
        if not rewritten or len(rewritten) > len(query) * 5:
            rewritten = query

        logger.info(f"Query rewritten: '{query[:50]}' → '{rewritten[:50]}'")
        return {
            "rewritten_query": rewritten,
            "metadata": {
                **state.get("metadata", {}),
                "original_query": query,
                "rewritten_query": rewritten,
            },
        }

    # ─── Hybrid Retriever Node ───────────────────────────────────────────

    @traceable(name="hybrid_retriever_node")
    def hybrid_retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant documents using hybrid search (dense + sparse)."""
        if state.get("error"):
            return {}

        # Use rewritten query if available, else original
        search_query = state.get("rewritten_query") or state["query"]
        tenant_id = state.get("tenant_id", "default")

        results = self.retrieval_service.hybrid_retrieve(
            query=search_query,
            top_k=settings.TOP_K_RETRIEVAL,
            tenant_id=tenant_id,
        )

        logger.info(f"Hybrid retrieval returned {len(results)} documents")
        return {"retrieved_documents": results}

    # ─── Reranker Node ───────────────────────────────────────────────────

    @traceable(name="reranker_node")
    def reranker_node(self, state: AgentState) -> Dict[str, Any]:
        """Rerank retrieved documents with cross-encoder."""
        if state.get("error"):
            return {}

        documents = state.get("retrieved_documents", [])
        if not documents:
            return {"reranked_documents": [], "error": "No documents found for this query."}

        search_query = state.get("rewritten_query") or state["query"]

        reranked = self.retrieval_service.rerank(
            query=search_query,
            documents=documents,
            top_k=settings.TOP_K_RERANK,
        )

        logger.info(f"Reranked to {len(reranked)} documents")
        return {"reranked_documents": reranked}

    # ─── Tool Node ───────────────────────────────────────────────────────

    @traceable(name="tool_node")
    def tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools if needed (extensibility point)."""
        return {"tool_results": None}

    # ─── Context Compressor Node ─────────────────────────────────────────

    @traceable(name="context_compressor_node")
    def context_compressor_node(self, state: AgentState) -> Dict[str, Any]:
        """Compress context: remove redundancy, enforce token limits (MMR)."""
        if state.get("error"):
            return {}

        documents = state.get("reranked_documents", [])
        if not documents:
            return {"compressed_documents": []}

        # Step 1: Remove near-duplicate chunks
        unique_docs = self._deduplicate_chunks(documents)

        # Step 2: MMR-based diversity selection
        selected = self._mmr_select(unique_docs, max_docs=10)

        # Step 3: Trim to token budget
        compressed = self._trim_to_token_budget(selected, settings.MAX_CONTEXT_TOKENS)

        logger.info(f"Context compressed: {len(documents)} → {len(compressed)} documents")
        return {"compressed_documents": compressed}

    def _deduplicate_chunks(self, documents: List[dict]) -> List[dict]:
        """Remove chunks with very similar content."""
        seen_hashes = set()
        unique = []
        for doc in documents:
            text = doc.get("text", "")
            # Simple content fingerprint: first 200 chars normalized
            fingerprint = text[:200].lower().strip()
            if fingerprint not in seen_hashes:
                seen_hashes.add(fingerprint)
                unique.append(doc)
        return unique

    def _mmr_select(self, documents: List[dict], max_docs: int = 10) -> List[dict]:
        """Max Marginal Relevance selection for diversity."""
        if len(documents) <= max_docs:
            return documents

        selected = [documents[0]]
        remaining = documents[1:]

        while len(selected) < max_docs and remaining:
            best_score = -1
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # Relevance score from reranking
                relevance = candidate.get("rerank_score", candidate.get("score", 0))

                # Max similarity to already-selected docs (simple text overlap)
                max_sim = 0
                cand_words = set(candidate.get("text", "").lower().split())
                for sel in selected:
                    sel_words = set(sel.get("text", "").lower().split())
                    if cand_words or sel_words:
                        overlap = len(cand_words & sel_words) / max(len(cand_words | sel_words), 1)
                        max_sim = max(max_sim, overlap)

                # MMR score
                mmr = settings.MMR_LAMBDA * relevance - (1 - settings.MMR_LAMBDA) * max_sim

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _trim_to_token_budget(self, documents: List[dict], max_tokens: int) -> List[dict]:
        """Keep documents until token budget is exhausted."""
        result = []
        total_tokens = 0
        for doc in documents:
            text = doc.get("text", "")
            tokens = len(self._tokenizer.encode(text))
            if total_tokens + tokens > max_tokens:
                # Truncate last document to fit
                remaining = max_tokens - total_tokens
                if remaining > 50:
                    truncated_text = self._tokenizer.decode(
                        self._tokenizer.encode(text)[:remaining]
                    )
                    doc = {**doc, "text": truncated_text}
                    result.append(doc)
                break
            total_tokens += tokens
            result.append(doc)
        return result

    # ─── LLM Reasoning Node ─────────────────────────────────────────────

    @traceable(name="llm_reasoning_node")
    def llm_reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate answer using LLM with compressed context."""
        if state.get("error"):
            return {"answer": state["error"], "sources": []}

        # Use compressed docs if available, else reranked
        documents = state.get("compressed_documents") or state.get("reranked_documents", [])

        # Build context from documents
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

    # ─── Response Validator Node ─────────────────────────────────────────

    @traceable(name="response_validator_node")
    def response_validator_node(self, state: AgentState) -> Dict[str, Any]:
        """Validate response grounding and factual accuracy."""
        if state.get("error"):
            return {}

        answer = state.get("answer", "")
        sources = state.get("sources", [])

        # Source enforcement
        if not self.guardrails.enforce_source_requirement(sources):
            fallback = self.guardrails.build_fallback_response("No sources")
            return {
                "answer": fallback,
                "validation_result": {"grounded": False, "reason": "No supporting sources"},
            }

        # Output grounding validation
        grounding = self.guardrails.validate_output_grounding(answer, sources)

        if not grounding["grounded"]:
            # Attempt retry with stricter prompt
            logger.warning(f"Response failed grounding check: {grounding['reason']}")
            fallback = self.guardrails.build_fallback_response(grounding["reason"])
            return {
                "answer": fallback,
                "validation_result": grounding,
            }

        return {
            "validation_result": grounding,
            "metadata": {
                **state.get("metadata", {}),
                "grounding_confidence": grounding["confidence"],
            },
        }

    # ─── Response Node ───────────────────────────────────────────────────

    @traceable(name="response_node")
    def response_node(self, state: AgentState) -> Dict[str, Any]:
        """Finalize the response with latency and metadata."""
        metadata = state.get("metadata", {})
        start_time = metadata.get("start_time", time.time())
        latency = (time.time() - start_time) * 1000

        return {
            "metadata": {
                **metadata,
                "latency_ms": latency,
                "total_sources": len(state.get("sources", [])),
                "query_rewritten": bool(state.get("rewritten_query")),
                "validation": state.get("validation_result"),
            }
        }
