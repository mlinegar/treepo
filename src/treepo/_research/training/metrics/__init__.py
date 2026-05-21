"""
Training metrics for OPS optimization.

Provides DSPy-compatible metrics for evaluating oracle approximation
and summarization quality.
"""

from treepo._research.training.metrics.metrics import (
    metric,
    calibration_error,
    law_compliance_rate,
    overall_compliance_rate,
    create_cached_oracle_metric,
    create_oracle_metric,
    create_merge_metric,
    oracle_as_metric,
    oracle_as_metric_with_feedback,
    summarization,
)

__all__ = [
    "metric",
    "calibration_error",
    "law_compliance_rate",
    "overall_compliance_rate",
    "create_cached_oracle_metric",
    "create_oracle_metric",
    "create_merge_metric",
    "oracle_as_metric",
    "oracle_as_metric_with_feedback",
    "summarization",
]
