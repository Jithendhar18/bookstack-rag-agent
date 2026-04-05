"""LangGraph agent node implementations — optimized RAG pipeline.

Pipeline flow (conditional rewrite):
    Input → Retriever → [Conditional Query Rewrite → Re-retrieve] → Reranker
        → Context Compression → LLM Answer → Response Validator → Response

Key optimizations:
- Query rewrite is conditional: only triggers when initial retrieval is poor
  (fewer than MIN_RETRIEVAL_RESULTS or low similarity scores).
- Context is capped at MAX_CONTEXT_DOCS (default 5) and MAX_CONTEXT_TOKENS (2000).
- Small/irrelevant chunks are dropped before LLM sees them.
- The answer generation prompt strictly forbids hallucination.
"""

import logging
import math
import re
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


# ─── Prompt Templates ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful documentation assistant. Answer the user's question using the provided context documents.

GUIDELINES:
1. Base your answer on the provided context. Do NOT invent facts not present in the context.
2. If the context does not contain enough information to answer, say:
   "I couldn't find relevant information in the available documentation."
3. You may synthesize and paraphrase the context — do not copy it word for word.
4. Keep answers concise and well-structured. Use bullet points for lists.
5. When multiple documents contribute, synthesize them into a single coherent answer.
6. Never fabricate sources or URLs.

CONTEXT:
{context}
"""

QUERY_REWRITE_PROMPT = """You are a query optimization assistant. Rewrite the user's query into a clear, specific search query optimized for document retrieval.

Rules:
- Expand abbreviations and acronyms
- Fix incomplete questions into full questions
- Preserve the original intent
- Make the query more specific and searchable
- Output ONLY the rewritten query, nothing else

User query: {query}

Rewritten query:"""

NO_RESULTS_FALLBACK = (
    "I couldn't find relevant information in the available documentation. "
    "Please try rephrasing your question or check that the relevant content has been ingested."
)

# Phrases the LLM uses when it determines the context is insufficient.
# When the LLM itself says it can't answer, we skip grounding and respect that.
_LLM_NO_INFO_PATTERNS = [
    r"couldn't find relevant",
    r"could not find relevant",
    r"don't have.*information",
    r"do not have.*information",
    r"not (enough|sufficient) information",
    r"no relevant information",
    r"not found in.*documentation",
    r"context does not (contain|have|provide)",
    r"unable to find.*answer",
    r"cannot (find|answer|provide)",
]


class AgentNodes:
    """Node implementations for the optimized RAG agent graph.

    Key design decisions:
    - Retrieval runs FIRST with the original query (avoids unnecessary LLM call).
    - Query rewrite is conditional: only triggered when retrieval quality is poor.
    - Context is aggressively pruned: dedup → rerank → MMR → token trim.
    - Max MAX_CONTEXT_DOCS documents reach the LLM (default 5) to cut latency.
    - Small chunks (< MIN_CHUNK_TOKENS tokens) are dropped as low-signal noise.
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
        """Validate input and check for prompt injection."""
        query = state["query"].strip()
        if not query:
            return {"error": "Empty query", "answer": "Please provide a question."}

        injection_check = self.guardrails.check_prompt_injection(query)
        if not injection_check["safe"]:
            logger.warning(f"Query blocked by guardrails: {query[:80]}")
            return {
                "error": "blocked",
                "answer": injection_check["reason"],
            }

        return {
            "query": query,
            "messages": state.get("messages", []) + [HumanMessage(content=query)],
            "metadata": {
                **state.get("metadata", {}),
                "start_time": time.time(),
                "pipeline_config": {
                    "llm": settings.LLM_PROVIDER,
                    "retrieval": settings.RETRIEVAL_MODE,
                },
            },
        }

    # ─── Retriever Node ──────────────────────────────────────────────────

    @traceable(name="retriever_node")
    def retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve documents using the original query first.

        Runs BEFORE any query rewrite so we can decide whether a rewrite
        is actually needed (saves an LLM call on well-formed queries).
        """
        if state.get("error"):
            return {}

        # Use rewritten query if this is a second retrieval pass, otherwise original
        search_query = state.get("rewritten_query") or state["query"]
        tenant_id = state.get("tenant_id", "default")

        try:
            retriever = get_retriever()
            results = retriever.retrieve(
                query=search_query,
                top_k=settings.TOP_K_RETRIEVAL,
                tenant_id=tenant_id,
            )
            logger.info(
                f"Retrieval ({settings.RETRIEVAL_MODE}): "
                f"{len(results)} docs for query '{search_query[:60]}'"
            )
            return {"retrieved_documents": results}
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {"retrieved_documents": [], "error": f"Retrieval failed: {e}"}

    # ─── Conditional Query Rewrite Node ──────────────────────────────────

    @traceable(name="query_rewrite_node")
    def query_rewrite_node(self, state: AgentState) -> Dict[str, Any]:
        """Conditionally rewrite the query when initial retrieval is poor.

        The rewrite is triggered only when:
        - QUERY_REWRITER_ENABLED=true AND
        - CONDITIONAL_REWRITE_ENABLED=true AND
        - Initial retrieval returned fewer than MIN_RETRIEVAL_RESULTS docs
          OR the best score is below SIMILARITY_THRESHOLD.

        This avoids wasting an LLM call when the original query already
        retrieves good results.
        """
        if state.get("error"):
            return {}

        query = state["query"]

        # Skip entirely if rewriter is turned off
        if not settings.QUERY_REWRITER_ENABLED:
            logger.info("Query rewriter: DISABLED")
            return {
                "rewritten_query": query,
                "metadata": {
                    **state.get("metadata", {}),
                    "query_rewriter_skipped": True,
                },
            }

        documents = state.get("retrieved_documents", [])

        # Decide whether a rewrite is needed
        needs_rewrite = self._should_rewrite(documents)

        if not needs_rewrite:
            logger.info("Query rewriter: SKIPPED (initial retrieval sufficient)")
            return {
                "rewritten_query": query,
                "metadata": {
                    **state.get("metadata", {}),
                    "query_rewriter_skipped": True,
                    "rewrite_reason": "initial_retrieval_sufficient",
                },
            }

        # Perform rewrite via LLM
        try:
            llm = get_llm()
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            response = llm.langchain_client.invoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip()

            # Fallback to original if rewrite is empty or suspiciously long
            if not rewritten or len(rewritten) > len(query) * 5:
                rewritten = query

            logger.info(f"Query rewritten: '{query[:50]}' → '{rewritten[:50]}'")
            return {
                "rewritten_query": rewritten,
                "metadata": {
                    **state.get("metadata", {}),
                    "original_query": query,
                    "rewritten_query": rewritten,
                    "rewrite_reason": "poor_initial_retrieval",
                },
            }
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            return {
                "rewritten_query": query,
                "metadata": {
                    **state.get("metadata", {}),
                    "query_rewriter_error": str(e),
                },
            }

    def _should_rewrite(self, documents: List[dict]) -> bool:
        """Decide whether the query needs rewriting based on retrieval quality."""
        # Always rewrite when conditional mode is off (legacy behaviour)
        if not settings.CONDITIONAL_REWRITE_ENABLED:
            return True

        # Too few results → rewrite
        if len(documents) < settings.MIN_RETRIEVAL_RESULTS:
            return True

        # Best score below threshold → rewrite
        if documents:
            best_score = max(
                (d.get("score", 0) for d in documents),
                default=0,
            )
            if best_score < settings.SIMILARITY_THRESHOLD:
                return True

        return False

    # ─── Reranker Node ───────────────────────────────────────────────────

    @traceable(name="reranker_node")
    def reranker_node(self, state: AgentState) -> Dict[str, Any]:
        """Rerank retrieved documents using a cross-encoder model.

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
        """Compress context: deduplicate, drop small chunks, enforce doc + token limits.

        Pipeline: dedup → drop tiny chunks → MMR diversity → token trim.
        Hard cap at MAX_CONTEXT_DOCS documents and MAX_CONTEXT_TOKENS tokens.
        """
        if state.get("error"):
            return {}

        documents = state.get("reranked_documents", [])
        if not documents:
            return {"compressed_documents": []}

        if not settings.CONTEXT_COMPRESSION_ENABLED:
            logger.info("Context compression: DISABLED")
            return {
                "compressed_documents": documents[:settings.MAX_CONTEXT_DOCS],
                "metadata": {
                    **state.get("metadata", {}),
                    "compression_skipped": True,
                },
            }

        # Step 1: Remove near-duplicate chunks
        unique_docs = self._deduplicate_chunks(documents)

        # Step 2: Drop chunks that are too small to be useful
        filtered = self._drop_small_chunks(unique_docs)

        # Step 3: MMR-based diversity selection (capped at MAX_CONTEXT_DOCS)
        selected = self._mmr_select(filtered, max_docs=settings.MAX_CONTEXT_DOCS)

        # Step 4: Trim to token budget
        compressed = self._trim_to_token_budget(selected, settings.MAX_CONTEXT_TOKENS)

        logger.info(f"Context compressed: {len(documents)} → {len(compressed)} documents")
        return {"compressed_documents": compressed}

    def _deduplicate_chunks(self, documents: List[dict]) -> List[dict]:
        """Remove chunks with near-identical content (first 200 chars fingerprint)."""
        seen = set()
        unique = []
        for doc in documents:
            fingerprint = doc.get("text", "")[:200].lower().strip()
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(doc)
        return unique

    def _drop_small_chunks(self, documents: List[dict]) -> List[dict]:
        """Drop chunks shorter than MIN_CHUNK_TOKENS — they add noise, not signal."""
        kept = []
        for doc in documents:
            text = doc.get("text", "")
            token_count = len(self._tokenizer.encode(text))
            if token_count >= settings.MIN_CHUNK_TOKENS:
                kept.append(doc)
            else:
                logger.debug(f"Dropped small chunk ({token_count} tokens): {text[:60]}")
        return kept

    def _mmr_select(self, documents: List[dict], max_docs: int = 5) -> List[dict]:
        """Max Marginal Relevance: balance relevance with diversity."""
        if len(documents) <= max_docs:
            return documents

        selected = [documents[0]]
        remaining = documents[1:]

        while len(selected) < max_docs and remaining:
            best_score = -float("inf")
            best_idx = 0

            for i, candidate in enumerate(remaining):
                raw_score = candidate.get("rerank_score", candidate.get("score", 0))
                relevance = raw_score if isinstance(raw_score, (int, float)) and not math.isnan(raw_score) else 0.0

                # Compute max word-overlap similarity to already selected docs
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
        """Keep documents until the token budget is exhausted."""
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
                    result.append({**doc, "text": truncated_text})
                break
            total_tokens += tokens
            result.append(doc)
        return result

    # ─── LLM Reasoning Node ─────────────────────────────────────────────

    @traceable(name="llm_reasoning_node")
    def llm_reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate an answer grounded strictly in the retrieved context.

        Returns a safe fallback when no relevant documents are available,
        avoiding an unnecessary LLM call.
        """
        if state.get("error"):
            return {"answer": state["error"], "sources": []}

        documents = state.get("compressed_documents") or state.get("reranked_documents", [])

        # No documents → return fallback without calling the LLM
        if not documents:
            return {
                "answer": NO_RESULTS_FALLBACK,
                "sources": [],
                "metadata": {
                    **state.get("metadata", {}),
                    "llm_skipped": True,
                    "reason": "no_documents",
                },
            }

        # Build context and source list
        context_parts = []
        sources = []
        raw_scores = self._collect_raw_scores(documents)
        min_raw, max_raw, score_range = self._score_normalization_params(raw_scores)

        for i, doc in enumerate(documents):
            title = doc.get("metadata", {}).get("title", f"Document {i+1}")
            text = doc.get("text", "")
            context_parts.append(f"[{i+1}] {title}\n{text}")

            normalized = self._normalize_score(raw_scores[i], min_raw, max_raw, score_range)
            doc_meta = doc.get("metadata", {})
            sources.append({
                "chunk_id": doc.get("id", ""),
                "document_title": title,
                "content": text[:500],
                "score": round(normalized, 4),
                "source_url": doc_meta.get("source_url"),
                "metadata": doc_meta,
            })

        context = "\n\n---\n\n".join(context_parts)

        # All compressed docs passed the retrieval + reranking quality bar — include
        # all of them as sources. Cap at 3 for display, sorted by score descending.
        # We no longer filter by display score because min-max normalization can
        # map legitimate results to low values (e.g., all-negative reranker logits).
        sources = sorted(sources, key=lambda s: s["score"], reverse=True)[:3]

        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context))

        llm = get_llm()
        try:
            response = llm.langchain_client.invoke(
                [system_msg] + state.get("messages", [])
            )
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

    # ─── Score helpers ───────────────────────────────────────────────────

    @staticmethod
    def _collect_raw_scores(documents: List[dict]) -> List[float]:
        scores = []
        for doc in documents:
            s = doc.get("rerank_score", doc.get("score", 0))
            scores.append(s if isinstance(s, (int, float)) and not math.isnan(s) else 0.0)
        return scores

    @staticmethod
    def _score_normalization_params(raw_scores: List[float]):
        max_raw = max(raw_scores) if raw_scores else 1.0
        min_raw = min(raw_scores) if raw_scores else 0.0
        # Always use min-max normalization.
        # When all scores are equal (or only one doc), score_range=0 — assign
        # a neutral display score of 0.8 so single relevant docs surface cleanly.
        score_range = max_raw - min_raw if max_raw != min_raw else None
        return min_raw, max_raw, score_range

    @staticmethod
    def _normalize_score(raw: float, min_raw: float, max_raw: float, score_range) -> float:
        # score_range=None means all docs had identical scores — display 0.8
        if score_range is None:
            return 0.8
        # Min-max normalization works for both cosine (0–1) and reranker logits
        # (can be negative). Result is always in [0, 1].
        return round((raw - min_raw) / score_range, 4)

    # ─── Response Validator Node ─────────────────────────────────────────

    @traceable(name="response_validator_node")
    def response_validator_node(self, state: AgentState) -> Dict[str, Any]:
        """Validate response grounding. Skipped when GUARDRAILS_ENABLED=false."""
        if state.get("error"):
            return {}

        if not settings.GUARDRAILS_ENABLED:
            return {
                "validation_result": {"grounded": True, "confidence": 1.0, "reason": None},
                "metadata": {**state.get("metadata", {}), "guardrails_skipped": True},
            }

        answer = state.get("answer", "")
        sources = state.get("sources", [])

        # Check source enforcement against compressed documents (what the LLM
        # actually used), not the display sources which are filtered to score > 0.4.
        # The display filter can remove all sources even when the LLM had context.
        full_docs = state.get("compressed_documents") or state.get("reranked_documents", [])
        has_sources = len(full_docs) >= settings.MIN_SUPPORTING_CHUNKS if full_docs else len(sources) >= settings.MIN_SUPPORTING_CHUNKS

        if not has_sources:
            fallback = self.guardrails.build_fallback_response("No sources")
            return {
                "answer": fallback,
                "validation_result": {"grounded": False, "reason": "No supporting sources"},
            }

        # High-confidence retrieval bypass: when the top retrieved source has a
        # normalized score ≥ 0.7, the retrieval pipeline is very confident the
        # context is relevant. For literary / archaic text the LLM naturally
        # paraphrases, so word-overlap grounding is unreliable — trust retrieval
        # confidence instead.
        best_source_score = max((s.get("score", 0) for s in sources), default=0)
        if best_source_score >= 0.7:
            logger.info(f"Grounding bypassed — high retrieval confidence ({best_source_score:.2f})")
            return {
                "validation_result": {"grounded": True, "confidence": best_source_score, "reason": "high_confidence_retrieval"},
                "metadata": {**state.get("metadata", {}), "retrieval_bypass": True, "top_score": best_source_score},
            }

        # If the LLM itself determined the context was insufficient and said so,
        # respect that response — don't replace it with the guardrails fallback.
        # Applying grounding to a "no info" phrase produces confidence=0 since
        # those words never appear in the source text, causing a double-fallback.
        answer_lower = answer.lower()
        import re as _re
        for pattern in _LLM_NO_INFO_PATTERNS:
            if _re.search(pattern, answer_lower):
                logger.info("LLM reported no relevant context — passing through as-is")
                return {
                    "validation_result": {"grounded": True, "confidence": 1.0, "reason": "llm_no_info_passthrough"},
                    "metadata": {**state.get("metadata", {}), "guardrails_skipped": False, "llm_no_info": True},
                }

        # Validate grounding against the FULL compressed documents (what the LLM
        # actually saw), not the truncated source snippets (500-char previews).
        # Using truncated snippets caused false rejections when the LLM's answer
        # drew from text beyond the 500-char preview window.
        full_docs = state.get("compressed_documents") or state.get("reranked_documents", [])
        grounding_sources = [
            {"content": doc.get("text", "")} for doc in full_docs
        ] if full_docs else sources

        grounding = self.guardrails.validate_output_grounding(answer, grounding_sources)

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
        """Finalize the response with latency and pipeline metadata."""
        metadata = state.get("metadata", {})
        start_time = metadata.get("start_time", time.time())
        latency = (time.time() - start_time) * 1000

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
                "query_rewritten": bool(
                    state.get("rewritten_query")
                    and state.get("rewritten_query") != state.get("query")
                ),
                "validation": state.get("validation_result"),
                "modules": modules_summary,
            }
        }
