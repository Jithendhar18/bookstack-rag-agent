"""LangGraph agent node implementations — modular, configurable pipeline.

All pipeline components are loaded via factory functions and respect
enable/disable toggles from the environment configuration.
"""

import logging
import math
import time
from typing import Dict, Any, List

import tiktoken
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable

from app.agents.state import AgentState
from app.providers.factory import get_llm, get_reranker, get_retriever
from app.core.guardrails import GuardrailsService
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


class AgentNodes:
    """Node implementations for the configurable RAG agent graph.

    All components are injected via the factory pattern and can be
    toggled on/off via environment variables without code changes.
    """

    def __init__(self):
        self.guardrails = GuardrailsService()

        # Load tokenizer for context compression
        try:
            self._tokenizer = tiktoken.encoding_for_model(settings.LLM_MODEL)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # ─── Input Node ──────────────────────────────────────────────────────

    @traceable(name="input_node")
    def input_node(self, state: AgentState) -> Dict[str, Any]:
        """Process, validate input, and optionally check for prompt injection."""
        query = state["query"].strip()
        if not query:
            return {"error": "Empty query", "answer": "Please provide a question."}

        # Guardrails: prompt injection check (skipped if GUARDRAILS_ENABLED=false)
        injection_check = self.guardrails.check_prompt_injection(query)
        if not injection_check["safe"]:
            logger.warning(f"Query blocked by guardrails: {query[:80]}")
            return {
                "error": "blocked",
                "answer": injection_check["reason"],
            }

        return {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "metadata": {
                **state.get("metadata", {}),
                "start_time": time.time(),
                "pipeline_config": {"llm": settings.LLM_PROVIDER, "retrieval": settings.RETRIEVAL_MODE},
            },
        }

    # ─── Query Rewrite Node ─────────────────────────────────────────────

    @traceable(name="query_rewrite_node")
    def query_rewrite_node(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite user query for better retrieval.

        Skipped when QUERY_REWRITER_ENABLED=false — passes query through unchanged.
        """
        if state.get("error"):
            return {}

        query = state["query"]

        # Toggle check: skip rewriting if disabled
        if not settings.QUERY_REWRITER_ENABLED:
            logger.info("Query rewriter: DISABLED (passing through original query)")
            return {
                "rewritten_query": query,
                "metadata": {
                    **state.get("metadata", {}),
                    "query_rewriter_skipped": True,
                },
            }

        try:
            llm = get_llm()
            prompt = QUERY_REWRITE_PROMPT.format(query=query)

            # LangGraph nodes run synchronously — use the langchain client directly
            lc_client = llm.langchain_client
            response = lc_client.invoke([HumanMessage(content=prompt)])
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
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            return {
                "rewritten_query": query,
                "metadata": {**state.get("metadata", {}), "query_rewriter_error": str(e)},
            }

    # ─── Retriever Node ──────────────────────────────────────────────────

    @traceable(name="retriever_node")
    def retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant documents using the configured retrieval strategy.

        Strategy is set via RETRIEVAL_MODE: dense | hybrid | keyword.
        """
        if state.get("error"):
            return {}

        search_query = state.get("rewritten_query") or state["query"]
        tenant_id = state.get("tenant_id", "default")

        try:
            retriever = get_retriever()
            results = retriever.retrieve(
                query=search_query,
                top_k=settings.TOP_K_RETRIEVAL,
                tenant_id=tenant_id,
            )
            logger.info(f"Retrieval ({settings.RETRIEVAL_MODE}) returned {len(results)} documents")
            return {"retrieved_documents": results}
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {"retrieved_documents": [], "error": f"Retrieval failed: {e}"}

    # ─── Reranker Node ───────────────────────────────────────────────────

    @traceable(name="reranker_node")
    def reranker_node(self, state: AgentState) -> Dict[str, Any]:
        """Rerank retrieved documents.

        Skipped when RERANKER_ENABLED=false — passes documents through unchanged.
        Fails gracefully — continues with unreranked documents on error.
        """
        if state.get("error"):
            return {}

        documents = state.get("retrieved_documents", [])
        if not documents:
            return {"reranked_documents": []}

        search_query = state.get("rewritten_query") or state["query"]

        try:
            reranker = get_reranker()
            reranked = reranker.rerank(
                query=search_query,
                documents=documents,
                top_k=settings.TOP_K_RERANK,
            )
            skipped = not settings.RERANKER_ENABLED
            logger.info(f"Reranker: {'SKIPPED' if skipped else f'{len(reranked)} docs'}")
            return {
                "reranked_documents": reranked,
                "metadata": {
                    **state.get("metadata", {}),
                    "reranker_skipped": skipped,
                },
            }
        except Exception as e:
            # Failsafe: continue without reranking
            logger.warning(f"Reranker failed, continuing with unreranked docs: {e}")
            return {
                "reranked_documents": documents[:settings.TOP_K_RERANK],
                "metadata": {
                    **state.get("metadata", {}),
                    "reranker_error": str(e),
                    "reranker_skipped": True,
                },
            }

    # ─── Context Compressor Node ─────────────────────────────────────────

    @traceable(name="context_compressor_node")
    def context_compressor_node(self, state: AgentState) -> Dict[str, Any]:
        """Compress context: remove redundancy, enforce token limits (MMR).

        Skipped when CONTEXT_COMPRESSION_ENABLED=false — passes documents through.
        """
        if state.get("error"):
            return {}

        documents = state.get("reranked_documents", [])
        if not documents:
            return {"compressed_documents": []}

        # Toggle check: skip compression if disabled
        if not settings.CONTEXT_COMPRESSION_ENABLED:
            logger.info("Context compression: DISABLED (passing through all documents)")
            return {
                "compressed_documents": documents,
                "metadata": {
                    **state.get("metadata", {}),
                    "compression_skipped": True,
                },
            }

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
                raw_score = candidate.get("rerank_score", candidate.get("score", 0))
                relevance = raw_score if isinstance(raw_score, (int, float)) and not math.isnan(raw_score) else 0.0

                max_sim = 0
                cand_words = set(candidate.get("text", "").lower().split())
                for sel in selected:
                    sel_words = set(sel.get("text", "").lower().split())
                    if cand_words or sel_words:
                        overlap = len(cand_words & sel_words) / max(len(cand_words | sel_words), 1)
                        max_sim = max(max_sim, overlap)

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
        """Generate answer using the configured LLM provider."""
        if state.get("error"):
            return {"answer": state["error"], "sources": []}

        documents = state.get("compressed_documents") or state.get("reranked_documents", [])

        # Build context from documents
        context_parts = []
        sources = []
        for i, doc in enumerate(documents):
            title = doc.get("metadata", {}).get("title", f"Document {i+1}")
            text = doc.get("text", "")
            context_parts.append(f"[{i+1}] {title}\n{text}")
            score = doc.get("rerank_score", doc.get("score", 0))
            doc_meta = doc.get("metadata", {})
            sources.append({
                "chunk_id": doc.get("id", ""),
                "document_title": title,
                "content": text[:500],
                "score": score if isinstance(score, (int, float)) and not math.isnan(score) else 0.0,
                "source_url": doc_meta.get("source_url"),
                "metadata": doc_meta,
            })

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context))

        llm = get_llm()
        try:
            response = llm.langchain_client.invoke([system_msg] + state.get("messages", []))
        except Exception as e:
            logger.error(f"LLM failed ({llm.model_name}): {e}")
            return {
                "answer": "I'm temporarily unable to generate a response. Please try again later.",
                "sources": sources,
                "metadata": {
                    **state.get("metadata", {}),
                    "llm_error": str(e),
                },
            }

        return {
            "answer": response.content,
            "sources": sources,
            "messages": [AIMessage(content=response.content)],
            "metadata": {
                **state.get("metadata", {}),
                "end_time": time.time(),
                "documents_used": len(documents),
                "llm_provider": llm.model_name,
            },
        }

    # ─── Response Validator Node ─────────────────────────────────────────

    @traceable(name="response_validator_node")
    def response_validator_node(self, state: AgentState) -> Dict[str, Any]:
        """Validate response grounding and factual accuracy.

        Skipped when GUARDRAILS_ENABLED=false.
        """
        if state.get("error"):
            return {}

        # Skip validation if guardrails disabled
        if not settings.GUARDRAILS_ENABLED:
            return {
                "validation_result": {"grounded": True, "confidence": 1.0, "reason": None},
                "metadata": {**state.get("metadata", {}), "guardrails_skipped": True},
            }

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

        # Build summary of which modules were active/skipped
        modules_summary = {
            "query_rewriter": "skipped" if metadata.get("query_rewriter_skipped") else "active",
            "retrieval_mode": settings.RETRIEVAL_MODE,
            "reranker": "skipped" if metadata.get("reranker_skipped") else "active",
            "compression": "skipped" if metadata.get("compression_skipped") else "active",
            "guardrails": "skipped" if metadata.get("guardrails_skipped") else "active",
            "llm_provider": metadata.get("llm_provider", settings.LLM_PROVIDER),
        }

        return {
            "metadata": {
                **metadata,
                "latency_ms": latency,
                "total_sources": len(state.get("sources", [])),
                "query_rewritten": bool(state.get("rewritten_query")),
                "validation": state.get("validation_result"),
                "modules": modules_summary,
            }
        }
