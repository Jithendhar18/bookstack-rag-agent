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

        # Check if the answer is grounded in source content using both
        # unigram and bigram overlap. Bigrams catch paraphrased content where
        # the LLM rephrases poetic/archaic source text into modern language
        # but preserves named entities and key phrases.
        answer_lower = answer.lower()
        source_texts = " ".join(s.get("content", s.get("text", "")) for s in sources).lower()

        stop_words = {"this", "that", "with", "from", "have", "been", "were", "will",
                      "would", "could", "should", "there", "their", "about", "which",
                      "when", "what", "where", "does", "also", "more", "than", "into",
                      "most", "some", "such", "only", "very", "just", "they", "your",
                      "based", "provided", "context", "documentation", "information",
                      "according", "following", "answer", "question"}

        answer_words = [
            w for w in re.findall(r'\b[a-z]{4,}\b', answer_lower)
            if w not in stop_words
        ]

        if not answer_words:
            return {"grounded": True, "confidence": 0.8, "reason": None}

        # Build a set of source words for efficient lookup
        source_word_set = set(re.findall(r'\b[a-z]{3,}\b', source_texts))

        # Unigram overlap — includes prefix matching for morphological variants
        # (e.g., "ravana"/"ravan", "destroyed"/"destroying", "ancient"/"anciently")
        unique_words = set(answer_words)
        word_matches = 0
        for w in unique_words:
            if w in source_texts:
                word_matches += 1
            else:
                # Prefix match: if a source word shares a stem (≥5 char prefix), count as partial match
                prefix = w[:5] if len(w) >= 5 else w[:4]
                if any(sw.startswith(prefix) for sw in source_word_set if len(sw) >= len(prefix)):
                    word_matches += 0.7  # partial credit for morphological variants
        word_confidence = word_matches / len(unique_words) if unique_words else 0

        # Bigram overlap — catches phrases like "faithful wife", "demon king"
        answer_bigrams = set(
            f"{answer_words[i]} {answer_words[i+1]}"
            for i in range(len(answer_words) - 1)
        )
        bigram_matches = sum(1 for bg in answer_bigrams if bg in source_texts) if answer_bigrams else 0
        # Also check individual bigram words — if both words exist independently in source, partial credit
        if answer_bigrams:
            for bg in answer_bigrams:
                w1, w2 = bg.split(" ", 1)
                if bg not in source_texts and w1 in source_texts and w2 in source_texts:
                    bigram_matches += 0.5
        bigram_confidence = bigram_matches / len(answer_bigrams) if answer_bigrams else 0

        # Combined confidence: weight unigrams more but reward bigram matches
        confidence = 0.7 * word_confidence + 0.3 * bigram_confidence

        grounded = confidence >= settings.HALLUCINATION_THRESHOLD

        if not grounded:
            logger.warning(
                f"Low grounding confidence: {confidence:.2f} "
                f"(words={word_confidence:.2f}, bigrams={bigram_confidence:.2f})"
            )

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
