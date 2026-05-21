"""
FeedbackCollector protocol and registry.

Generalizes PreferenceDeriver for type-agnostic feedback collection.
The protocol supports pairwise comparison, scalar rating, written critique,
and arbitrary combinations through the FeedbackRequest/FeedbackResponse pair.

Existing PreferenceDeriver implementations (JudgeDeriver, GenRMDeriver,
OracleDeriver) work unchanged via the PreferenceDeriverAdapter bridge.

Usage:
    from treepo._research.feedback import get_collector, register_collector, FeedbackRequest

    # Use an existing PreferenceDeriver through the adapter
    from treepo._research.training.supervision import get_deriver
    deriver = get_deriver("oracle", oracle_predict=my_fn)
    collector = PreferenceDeriverAdapter(deriver)

    # Or get a registered collector by name
    collector = get_collector("oracle", oracle_predict=my_fn)

    # Collect feedback
    request = FeedbackRequest(request_id="r1", text_a="...", text_b="...", rubric="...")
    response = collector.collect(request)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Protocol, Type, runtime_checkable

from treepo._research.feedback.types import FeedbackRequest, FeedbackResponse
from treepo._research.core.async_utils import to_thread

logger = logging.getLogger(__name__)


# =============================================================================
# FeedbackCollector Protocol
# =============================================================================

@runtime_checkable
class FeedbackCollector(Protocol):
    """Generalized feedback collection protocol.

    Subsumes PreferenceDeriver for pairwise tasks, and extends to
    scalar rating, written critique, and arbitrary feedback.

    Implementations inspect ``request.dimensions`` to determine what
    feedback to provide. They may provide more dimensions than requested
    (e.g., a judge that always produces both preference and scores).
    """

    def collect(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Collect feedback for the given request.

        Args:
            request: FeedbackRequest with context and requested dimensions
            **kwargs: Collector-specific arguments

        Returns:
            FeedbackResponse with collected feedback
        """
        ...

    async def collect_async(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Async version of collect(). Default implementations wrap sync."""
        ...


# =============================================================================
# Collector Registry
# =============================================================================

_COLLECTOR_REGISTRY: Dict[str, Type] = {}


def register_collector(name: str):
    """Decorator to register a feedback collector class.

    Usage:
        @register_collector("my_collector")
        class MyCollector:
            def collect(self, request, **kwargs): ...
            async def collect_async(self, request, **kwargs): ...
    """
    def decorator(cls: Type) -> Type:
        _COLLECTOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_collector(name: str, **kwargs: Any) -> FeedbackCollector:
    """Get a feedback collector by name.

    Args:
        name: Collector name (case-insensitive)
        **kwargs: Arguments passed to collector constructor

    Returns:
        Configured collector instance

    Raises:
        ValueError: If collector name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _COLLECTOR_REGISTRY:
        available = list(_COLLECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown collector: '{name}'. Available: {available}")
    return _COLLECTOR_REGISTRY[name_lower](**kwargs)


def list_collectors() -> List[str]:
    """Return list of registered collector names."""
    return list(_COLLECTOR_REGISTRY.keys())


# =============================================================================
# PreferenceDeriverAdapter
# =============================================================================

@register_collector("deriver_adapter")
class PreferenceDeriverAdapter:
    """Wraps any existing PreferenceDeriver as a FeedbackCollector.

    This is the primary backward-compatibility bridge. All existing derivers
    (JudgeDeriver, GenRMDeriver, OracleDeriver) work through this adapter
    without modification.

    Usage:
        from treepo._research.training.supervision import get_deriver

        deriver = get_deriver("oracle", oracle_predict=my_fn)
        collector = PreferenceDeriverAdapter(deriver)
        response = collector.collect(pairwise_request)
    """

    def __init__(self, deriver: Any):
        """
        Args:
            deriver: Any PreferenceDeriver instance (JudgeDeriver, GenRMDeriver, etc.)
        """
        self.deriver = deriver

    def collect(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Collect pairwise feedback by delegating to the wrapped deriver.

        Raises:
            ValueError: If the request is not pairwise (text_b is None)
        """
        if not request.is_pairwise or request.text_b is None:
            raise ValueError(
                "PreferenceDeriverAdapter requires a pairwise request "
                "(text_b must be set)"
            )

        result = self.deriver.derive(
            summary_a=request.text_a,
            summary_b=request.text_b,
            context=request.rubric,
            original_text=request.original_text,
            reference_score=request.reference_score,
            law_type=request.law_type,
            **kwargs,
        )

        return FeedbackResponse(
            request_id=request.request_id,
            preferred=result.preferred,
            confidence=result.confidence,
            reasoning=result.reasoning,
            score_estimate_a=result.score_estimate_a,
            score_estimate_b=result.score_estimate_b,
            extra={
                "comparison_signal_value": getattr(result, "comparison_signal_value", None),
                "comparison_signal_name": getattr(result, "comparison_signal_name", None),
                "comparison_signal_min": getattr(result, "comparison_signal_min", None),
                "comparison_signal_max": getattr(result, "comparison_signal_max", None),
                "response_signal_name": getattr(result, "response_signal_name", None),
                "response_signal_min": getattr(result, "response_signal_min", None),
                "response_signal_max": getattr(result, "response_signal_max", None),
            },
            source="deriver",
            judge_model=getattr(self.deriver, "judge_model", ""),
            raw_result=result.raw_result if hasattr(result, "raw_result") else result,
        )

    async def collect_async(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Async wrapper around sync collect()."""
        return await to_thread(self.collect, request, **kwargs)
