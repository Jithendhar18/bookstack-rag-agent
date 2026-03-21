"""Evaluation framework for RAG pipeline quality metrics with LangSmith integration."""

import time
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from langsmith import traceable

from app.agents.graph import run_agent_query
from app.core.guardrails import GuardrailsService
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class EvalCase:
    """A single evaluation test case."""
    query: str
    expected_answer: Optional[str] = None
    expected_sources: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of a single evaluation case."""
    query: str
    answer: str
    expected_answer: Optional[str]
    sources_count: int
    latency_ms: float
    grounding_confidence: float
    retrieval_hit: bool  # did expected sources appear?
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Summary of an evaluation run."""
    total_cases: int
    passed: int
    failed: int
    avg_latency_ms: float
    avg_grounding_confidence: float
    retrieval_accuracy: float
    results: List[EvalResult] = field(default_factory=list)


# Default test dataset
DEFAULT_EVAL_DATASET: List[EvalCase] = [
    EvalCase(
        query="How do I configure backups in BookStack?",
        expected_sources=["backup"],
        tags=["configuration"],
    ),
    EvalCase(
        query="What are the user roles and permissions?",
        expected_sources=["role", "permission"],
        tags=["rbac"],
    ),
    EvalCase(
        query="How to install BookStack?",
        expected_sources=["install", "setup"],
        tags=["installation"],
    ),
    EvalCase(
        query="How do I set up LDAP authentication?",
        expected_sources=["ldap", "auth"],
        tags=["authentication"],
    ),
    EvalCase(
        query="What is the API rate limiting configuration?",
        expected_sources=["api", "rate"],
        tags=["api"],
    ),
]


class EvaluationRunner:
    """Run evaluation suites against the RAG pipeline."""

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.guardrails = GuardrailsService()

    @traceable(name="evaluation_run", run_type="chain")
    async def run_evaluation(
        self,
        cases: Optional[List[EvalCase]] = None,
    ) -> EvalSummary:
        """Execute an evaluation suite and return metrics."""
        cases = cases or DEFAULT_EVAL_DATASET
        results: List[EvalResult] = []

        for case in cases:
            result = await self._evaluate_case(case)
            results.append(result)
            logger.info(
                f"Eval: '{case.query[:40]}...' | "
                f"passed={result.passed} | "
                f"latency={result.latency_ms:.0f}ms | "
                f"grounding={result.grounding_confidence:.2f}"
            )

        # Compute summary
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_latency = sum(r.latency_ms for r in results) / total if total else 0
        avg_grounding = sum(r.grounding_confidence for r in results) / total if total else 0
        retrieval_hits = sum(1 for r in results if r.retrieval_hit)

        summary = EvalSummary(
            total_cases=total,
            passed=passed,
            failed=total - passed,
            avg_latency_ms=round(avg_latency, 2),
            avg_grounding_confidence=round(avg_grounding, 3),
            retrieval_accuracy=round(retrieval_hits / total, 3) if total else 0,
            results=results,
        )

        logger.info(
            f"Evaluation complete: {passed}/{total} passed | "
            f"avg_latency={avg_latency:.0f}ms | "
            f"retrieval_accuracy={summary.retrieval_accuracy:.1%}"
        )

        return summary

    @traceable(name="evaluate_single_case")
    async def _evaluate_case(self, case: EvalCase) -> EvalResult:
        """Run a single evaluation case."""
        start = time.time()

        try:
            result = await run_agent_query(
                query=case.query,
                tenant_id=self.tenant_id,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return EvalResult(
                query=case.query,
                answer=f"ERROR: {e}",
                expected_answer=case.expected_answer,
                sources_count=0,
                latency_ms=latency,
                grounding_confidence=0,
                retrieval_hit=False,
                passed=False,
                details={"error": str(e)},
            )

        latency = (time.time() - start) * 1000
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Check grounding
        grounding = self.guardrails.validate_output_grounding(answer, sources)
        confidence = grounding.get("confidence", 0)

        # Check retrieval accuracy — did expected source keywords appear?
        retrieval_hit = self._check_retrieval_hit(sources, case.expected_sources)

        passed = (
            confidence >= settings.HALLUCINATION_THRESHOLD
            and len(sources) >= settings.MIN_SUPPORTING_CHUNKS
            and bool(answer)
        )

        return EvalResult(
            query=case.query,
            answer=answer[:500],
            expected_answer=case.expected_answer,
            sources_count=len(sources),
            latency_ms=round(latency, 2),
            grounding_confidence=round(confidence, 3),
            retrieval_hit=retrieval_hit,
            passed=passed,
            details={
                "metadata": result.get("metadata", {}),
                "grounding": grounding,
            },
        )

    def _check_retrieval_hit(
        self,
        sources: List[Dict[str, Any]],
        expected_keywords: Optional[List[str]],
    ) -> bool:
        """Check if at least one expected keyword appears in source titles/content."""
        if not expected_keywords:
            return True  # No expected sources = pass by default

        source_text = " ".join(
            f"{s.get('document_title', '')} {s.get('content', '')}"
            for s in sources
        ).lower()

        return any(kw.lower() in source_text for kw in expected_keywords)

    def to_dict(self, summary: EvalSummary) -> Dict[str, Any]:
        """Serialize evaluation summary to dict."""
        return asdict(summary)
