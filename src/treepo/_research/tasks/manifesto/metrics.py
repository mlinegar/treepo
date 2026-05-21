"""
RILE-specific metrics for manifesto training and evaluation.
"""

from typing import Callable, Optional, Any

from treepo._research.core.scoring import UNIT_SCALE
from treepo._research.training.metrics import summarization, create_merge_metric

from .constants import RILE_RANGE


def _resolve_rile_predictor(predictor: Any) -> Callable[[str], Any]:
    if hasattr(predictor, "predict_rile"):
        return predictor.predict_rile
    if callable(predictor):
        return predictor
    raise ValueError("predictor must be callable or implement predict_rile")


def create_rile_summarization_metric(
    oracle_classifier: Any,
    human_weight: float = 0.3,
    oracle_weight: float = 0.7,
    threshold: float = 0.05,
    min_summary_length: int = 50,
    max_error: Optional[float] = None,
) -> Callable:
    """
    Create a RILE-specific summarization metric.

    Args:
        oracle_classifier: Predictor with predict_rile(text) or any callable
        human_weight: Weight for human feedback score (0.0-1.0)
        oracle_weight: Weight for oracle-based score (should sum to 1.0 with human_weight)
        threshold: Maximum acceptable score drift (normalized units)
        min_summary_length: Minimum acceptable summary length
        max_error: Optional error scale (defaults to 1.0)
    """
    if threshold > 1.0:
        threshold = threshold / RILE_RANGE
    max_error = max_error or 1.0
    predict_fn = _resolve_rile_predictor(oracle_classifier)
    return summarization(
        oracle_classifier=predict_fn,
        human_weight=human_weight,
        oracle_weight=oracle_weight,
        threshold=threshold,
        min_summary_length=min_summary_length,
        max_error=max_error,
        scale=UNIT_SCALE,
        label_name="RILE",
    )


def create_rile_merge_metric(
    oracle_classifier: Any,
    threshold: float = 0.05,
) -> Callable:
    """Create a RILE-specific merge metric."""
    predict_fn = _resolve_rile_predictor(oracle_classifier)
    if threshold > 1.0:
        threshold = threshold / RILE_RANGE
    return create_merge_metric(
        oracle_classifier=predict_fn,
        scale=UNIT_SCALE,
        threshold=threshold,
    )
