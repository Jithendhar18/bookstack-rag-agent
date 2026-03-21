"""Guardrails: prompt injection detection, output validation, source enforcement."""

import re
import logging
from typing import Dict, Any, List, Optional

from langsmith import traceable

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Known prompt injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior",
    r"forget\s+(everything|all|your\s+instructions)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"pretend\s+you\s+are",
    r"act\s+as\s+(a|an|if)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*override",
    r"ignore\s+the\s+above",
    r"do\s+not\s+follow\s+(the\s+)?(above|previous)",
    r"\[system\]",
    r"\[INST\]",
    r"<\|im_start\|>",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


class GuardrailsService:
    """Safety layer for input validation and output verification."""

    @staticmethod
    @traceable(name="check_prompt_injection")
    def check_prompt_injection(query: str) -> Dict[str, Any]:
        """Detect potential prompt injection attempts."""
        if not settings.GUARDRAILS_ENABLED:
            return {"safe": True, "reason": None}

        for pattern in _COMPILED_PATTERNS:
            if pattern.search(query):
                logger.warning(f"Prompt injection detected: {query[:100]}")
                return {
                    "safe": False,
                    "reason": "Potential prompt injection detected. Please rephrase your question.",
                }
        return {"safe": True, "reason": None}

    @staticmethod
    @traceable(name="validate_output_grounding")
    def validate_output_grounding(
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate that the answer is grounded in provided sources."""
        if not settings.GUARDRAILS_ENABLED:
            return {"grounded": True, "confidence": 1.0, "reason": None}

        if not sources:
            return {
                "grounded": False,
                "confidence": 0.0,
                "reason": "No supporting sources found.",
            }

        if len(sources) < settings.MIN_SUPPORTING_CHUNKS:
            return {
                "grounded": False,
                "confidence": 0.2,
                "reason": f"Insufficient supporting sources (need at least {settings.MIN_SUPPORTING_CHUNKS}).",
            }

        # Check if key phrases from the answer appear in sources
        answer_lower = answer.lower()
        source_texts = " ".join(s.get("content", s.get("text", "")) for s in sources).lower()

        # Extract significant words from answer (> 4 chars, not common words)
        stop_words = {"this", "that", "with", "from", "have", "been", "were", "will",
                      "would", "could", "should", "there", "their", "about", "which",
                      "when", "what", "where", "does", "also", "more", "than", "into",
                      "most", "some", "such", "only", "very", "just", "they", "your"}
        answer_words = set(
            w for w in re.findall(r'\b[a-z]{4,}\b', answer_lower)
            if w not in stop_words
        )

        if not answer_words:
            return {"grounded": True, "confidence": 0.8, "reason": None}

        matched = sum(1 for w in answer_words if w in source_texts)
        confidence = matched / len(answer_words) if answer_words else 0

        grounded = confidence >= settings.HALLUCINATION_THRESHOLD

        if not grounded:
            logger.warning(f"Low grounding confidence: {confidence:.2f}")

        return {
            "grounded": grounded,
            "confidence": round(confidence, 3),
            "reason": None if grounded else f"Low grounding confidence ({confidence:.2f}). Answer may not be fully supported by sources.",
        }

    @staticmethod
    @traceable(name="enforce_source_requirement")
    def enforce_source_requirement(
        sources: List[Dict[str, Any]],
    ) -> bool:
        """Ensure at least MIN_SUPPORTING_CHUNKS sources exist."""
        return len(sources) >= settings.MIN_SUPPORTING_CHUNKS

    @staticmethod
    def build_fallback_response(reason: str) -> str:
        """Generate a safe fallback response when validation fails."""
        return (
            "I wasn't able to find a well-supported answer to your question "
            "based on the available documentation. Please try rephrasing your "
            "question or contact your administrator for more help."
        )
