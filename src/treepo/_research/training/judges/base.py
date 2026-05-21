"""
Base classes and protocols for supervision backends.

Backends may compare multiple summaries, compare two summaries, or provide
scalar judgments over single responses. The canonical stored data surface lives
under ``src.training.supervision``; this module defines backend protocols and
result objects that supervision collectors can normalize.

Available judge types:
- DSPyJudge: Optimizable LLM judge backend
- GenRMJudge: NVIDIA Qwen3-Nemotron GenRM backend
- OracleJudge: Oracle scoring backend

Usage:
    from treepo._research.training.judges import get_judge, JudgeConfig

    # Get a judge by name
    config = JudgeConfig(type="genrm", base_url="http://localhost:8001/v1")
    judge = get_judge("genrm", config)

    # Compare summaries or candidates
    result = judge.compare(
        context="Preserve key information",
        original_text="...",
        summary_a="...",
        summary_b="...",
    )
    print(result.preferred)  # "A", "B", or "tie"
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class JudgeResult:
    """
    Result from a binary comparison backend.

    This is the unified result type returned by all judge implementations.
    """
    preferred: Literal["A", "B", "tie"]
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""

    # Optional score estimates (if the judge provides them)
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None

    # Raw data from the underlying judge (for debugging)
    raw_result: Optional[Any] = None

    def is_tie(self) -> bool:
        """Check if the result is a tie."""
        return self.preferred == "tie"

    def winner(self) -> Optional[Literal["A", "B"]]:
        """Return the winner, or None if tie."""
        if self.preferred in ("A", "B"):
            return self.preferred
        return None


@dataclass
class JudgeError:
    """
    Error result from a judge (distinct from ties).

    This type exists to distinguish real comparison failures
    (network errors, parse failures) from legitimate preference ties.
    Callers should filter these out of training data.
    """
    error_type: Literal["network", "timeout", "parse_error", "server_error"]
    error_message: str
    raw_response: str = ""

    def is_error(self) -> bool:
        """Always True for error results."""
        return True


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class JudgeConfig:
    """Configuration for judge instantiation."""

    # Common settings
    type: str = "dspy"  # "dspy", "genrm", "oracle"

    # DSPy judge settings
    use_cot: bool = True

    # GenRM judge settings
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 16384

    # Oracle judge settings
    oracle_fn: Optional[Callable[[str], float]] = None
    tie_margin: float = 0.05  # Normalized error margin for ties

    # Extra config for custom judges
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Protocol
# =============================================================================

@runtime_checkable
class BaseJudge(Protocol):
    """
    Protocol that all judge implementations must follow.

    Backends compare two summaries and determine which better preserves
    the specified information. This protocol defines the minimal compatibility
    interface for binary callers.
    """

    def compare(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
        **kwargs,
    ) -> JudgeResult:
        """
        Compare two summaries and return preference.

        Args:
            context: Description of what information to preserve (rubric)
            original_text: Original text being summarized
            summary_a: First candidate summary
            summary_b: Second candidate summary
            law_type: OPS law type ("sufficiency", "idempotence", "merge")
            extra_context: Additional context for the comparison
            **kwargs: Additional arguments for specific judge types

        Returns:
            JudgeResult with preference and confidence
        """
        ...


class CompilableJudge(BaseJudge, Protocol):
    """
    Extended protocol for judges that can be optimized with DSPy.

    Some judges (like DSPyJudge) can be optimized using DSPy's
    compile() method to improve their prompts/examples.
    """

    def compile(
        self,
        trainset: Any,
        metric: Optional[Callable] = None,
        optimizer_name: str = "bootstrap_random_search",
        **kwargs,
    ) -> "CompilableJudge":
        """
        Optimize the judge using DSPy.

        Args:
            trainset: Training examples
            metric: Evaluation metric
            optimizer_name: DSPy optimizer to use
            **kwargs: Additional optimizer arguments

        Returns:
            Optimized judge instance
        """
        ...


class AsyncJudge(BaseJudge, Protocol):
    """
    Extended protocol for judges with async support.

    Some judges (like GenRMJudge) support async operations
    for better throughput in tournament-style comparisons.
    """

    async def compare_async(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
        **kwargs,
    ) -> JudgeResult:
        """Async version of compare()."""
        ...


# =============================================================================
# Utility Functions
# =============================================================================

def is_judge_error(result: Any) -> bool:
    """Check if a result is an error (not a valid preference)."""
    return isinstance(result, JudgeError)


def judge_result_from_dict(data: Dict[str, Any]) -> JudgeResult:
    """Create JudgeResult from a dictionary."""
    preferred = str(data.get("preferred", "tie")).upper()
    if preferred not in ("A", "B", "TIE"):
        preferred = "tie"
    elif preferred == "TIE":
        preferred = "tie"

    return JudgeResult(
        preferred=preferred,
        confidence=float(data.get("confidence", 0.5)),
        reasoning=str(data.get("reasoning", "")),
        score_estimate_a=data.get("score_estimate_a"),
        score_estimate_b=data.get("score_estimate_b"),
        raw_result=data,
    )
