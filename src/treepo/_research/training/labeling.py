"""
Shared labeling utilities for the training framework.

This module provides pluggable labeling strategies for converting
prediction errors into training labels. Different strategies can be
used depending on the task requirements.

Available strategies:
    - ThresholdLabeler: Fixed thresholds (default)
    - PercentileLabeler: Data-driven percentile thresholds
    - BinaryLabeler: Simple above/below threshold

Usage:
    from treepo._research.training.labeling import (
        get_labeler,
        ThresholdLabeler,
        PercentileLabeler,
    )

    # Get a labeler by name
    labeler = get_labeler("threshold", threshold_high=0.3, threshold_low=0.1)

    # Use directly
    labeler = ThresholdLabeler(threshold_high=0.2, threshold_low=0.05)
    label = labeler.label_from_error(error=15.0, scale=my_scale)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Type, runtime_checkable

from treepo._research.training.core import TrainingExampleLabel
from treepo._research.tasks.base import ScaleDefinition


# =============================================================================
# Protocol
# =============================================================================

@runtime_checkable
class LabelingStrategy(Protocol):
    """
    Protocol for error-to-label conversion strategies.

    Labeling strategies convert prediction errors (or scores) into
    training labels (POSITIVE/NEGATIVE/None). Different strategies
    can be used for different training scenarios.
    """

    def label_from_error(
        self,
        error: float,
        scale: Optional[ScaleDefinition] = None,
    ) -> Optional[TrainingExampleLabel]:
        """
        Convert error to label.

        Args:
            error: Raw error (absolute difference between predicted and actual)
            scale: If provided, normalize error by scale.range before comparison

        Returns:
            TrainingExampleLabel.POSITIVE: Error is high (likely violation)
            TrainingExampleLabel.NEGATIVE: Error is low (good preservation)
            None: Error is in ambiguous range (should be skipped)
        """
        ...


# =============================================================================
# Registry
# =============================================================================

_LABELER_REGISTRY: Dict[str, Type["LabelingStrategy"]] = {}


def register_labeler(name: str):
    """Decorator to register a labeler class."""
    def decorator(cls: Type[LabelingStrategy]):
        _LABELER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_labeler(name: str, **kwargs) -> LabelingStrategy:
    """
    Get a labeler by name from the registry.

    Args:
        name: Labeler name ("threshold", "percentile", "binary")
        **kwargs: Arguments passed to labeler constructor

    Returns:
        Configured labeler instance

    Raises:
        ValueError: If labeler name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _LABELER_REGISTRY:
        available = list(_LABELER_REGISTRY.keys())
        raise ValueError(f"Unknown labeler: '{name}'. Available: {available}")

    return _LABELER_REGISTRY[name_lower](**kwargs)


def list_labelers() -> List[str]:
    """Return list of registered labeler names."""
    return list(_LABELER_REGISTRY.keys())


# =============================================================================
# Implementations
# =============================================================================

@register_labeler("threshold")
@dataclass
class ThresholdLabeler:
    """
    Shared logic for error-based label assignment.

    Uses normalized thresholds (0-1 scale representing percentage of scale range)
    to classify prediction errors into POSITIVE (violation) or NEGATIVE (good) labels.

    Labels only - confidence calculation remains per-class for flexibility.

    Example:
        labeler = ThresholdLabeler(threshold_high=0.3, threshold_low=0.1)

        # With scale normalization
        scale = ScaleDefinition("score", -1, 1)
        label = labeler.label_from_error(error=25.0, scale=scale)  # 25/200 = 0.125 → None (ambiguous)

        # Without scale (assumes already normalized)
        label = labeler.label_from_error(error=0.4)  # 0.4 >= 0.3 → POSITIVE
    """

    threshold_high: float = 0.3  # Normalized: 30% of scale range → violation
    threshold_low: float = 0.1   # Normalized: 10% of scale range → good

    def label_from_error(
        self,
        error: float,
        scale: Optional[ScaleDefinition] = None
    ) -> Optional[TrainingExampleLabel]:
        """
        Convert error to label.

        Args:
            error: Raw error (absolute difference between predicted and actual)
            scale: If provided, normalize error by scale.range before comparison

        Returns:
            TrainingExampleLabel.POSITIVE: Error is high (likely violation)
            TrainingExampleLabel.NEGATIVE: Error is low (good preservation)
            None: Error is in ambiguous middle range (should be skipped)
        """
        # Normalize error to 0-1 range using scale if available
        if scale is not None:
            normalized = error / scale.range
        else:
            normalized = error  # Assume already normalized

        if normalized >= self.threshold_high:
            return TrainingExampleLabel.POSITIVE
        elif normalized <= self.threshold_low:
            return TrainingExampleLabel.NEGATIVE
        else:
            return None  # Ambiguous - skip this example

    def is_violation(
        self,
        error: float,
        scale: Optional[ScaleDefinition] = None
    ) -> bool:
        """
        Check if error indicates a violation.

        Convenience method for cases where we only care about violations.

        Args:
            error: Raw error
            scale: Optional scale for normalization

        Returns:
            True if error exceeds threshold_high
        """
        normalized = error / scale.range if scale else error
        return normalized >= self.threshold_high

    def is_good(
        self,
        error: float,
        scale: Optional[ScaleDefinition] = None
    ) -> bool:
        """
        Check if error indicates good preservation.

        Convenience method for cases where we only care about good results.

        Args:
            error: Raw error
            scale: Optional scale for normalization

        Returns:
            True if error is below threshold_low
        """
        normalized = error / scale.range if scale else error
        return normalized <= self.threshold_low


# Default labeler instance with standard thresholds
DEFAULT_LABELER = ThresholdLabeler()


@register_labeler("binary")
@dataclass
class BinaryLabeler:
    """
    Simple binary labeling with single threshold.

    All errors above threshold are POSITIVE (violations),
    all below are NEGATIVE (good). No ambiguous middle range.

    Example:
        labeler = BinaryLabeler(threshold=0.15)
        label = labeler.label_from_error(error=0.2)  # POSITIVE
        label = labeler.label_from_error(error=0.1)  # NEGATIVE
    """

    threshold: float = 0.15  # Normalized: 15% of scale range

    def label_from_error(
        self,
        error: float,
        scale: Optional[ScaleDefinition] = None,
    ) -> Optional[TrainingExampleLabel]:
        """
        Convert error to binary label.

        Args:
            error: Raw error (absolute difference)
            scale: If provided, normalize error by scale.range

        Returns:
            POSITIVE if error >= threshold, NEGATIVE otherwise
        """
        normalized = error / scale.range if scale else error

        if normalized >= self.threshold:
            return TrainingExampleLabel.POSITIVE
        else:
            return TrainingExampleLabel.NEGATIVE


@register_labeler("percentile")
@dataclass
class PercentileLabeler:
    """
    Data-driven labeling based on error distribution percentiles.

    Instead of fixed thresholds, uses percentiles of the error
    distribution to determine labels. Must be fitted on a set of
    errors before use.

    Example:
        labeler = PercentileLabeler(high_percentile=80, low_percentile=20)
        labeler.fit(errors=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        label = labeler.label_from_error(error=0.55)  # POSITIVE (above 80th percentile)
    """

    high_percentile: float = 80.0  # Above this percentile → POSITIVE
    low_percentile: float = 20.0   # Below this percentile → NEGATIVE

    # Computed thresholds (after fitting)
    _threshold_high: float = field(default=float('inf'), init=False, repr=False)
    _threshold_low: float = field(default=0.0, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, errors: List[float]) -> "PercentileLabeler":
        """
        Fit the labeler on a distribution of errors.

        Args:
            errors: List of error values to compute percentiles from

        Returns:
            Self for chaining
        """
        if not errors:
            return self

        sorted_errors = sorted(errors)
        n = len(sorted_errors)

        # Compute percentile indices
        high_idx = min(int(n * self.high_percentile / 100), n - 1)
        low_idx = max(int(n * self.low_percentile / 100), 0)

        self._threshold_high = sorted_errors[high_idx]
        self._threshold_low = sorted_errors[low_idx]
        self._fitted = True

        return self

    def label_from_error(
        self,
        error: float,
        scale: Optional[ScaleDefinition] = None,
    ) -> Optional[TrainingExampleLabel]:
        """
        Convert error to label based on fitted percentiles.

        Args:
            error: Raw error (absolute difference)
            scale: If provided, normalize error by scale.range

        Returns:
            POSITIVE if above high percentile, NEGATIVE if below low percentile,
            None if in middle range

        Raises:
            RuntimeError: If fit() has not been called
        """
        if not self._fitted:
            raise RuntimeError(
                "PercentileLabeler must be fitted before use. Call fit() first."
            )

        normalized = error / scale.range if scale else error

        if normalized >= self._threshold_high:
            return TrainingExampleLabel.POSITIVE
        elif normalized <= self._threshold_low:
            return TrainingExampleLabel.NEGATIVE
        else:
            return None


@register_labeler("adaptive")
@dataclass
class AdaptiveLabeler:
    """
    Adaptive labeling that adjusts thresholds based on task scale.

    Automatically adjusts thresholds relative to the scale's range,
    making configuration more intuitive across different tasks.

    Example:
        labeler = AdaptiveLabeler(high_percent=20, low_percent=5)

        # For a wide scale (range=200):
        # - high_threshold = 200 * 0.20 = 40 points
        # - low_threshold = 200 * 0.05 = 10 points

        # For a narrow scale (range=2):
        # - high_threshold = 2 * 0.20 = 0.4 points
        # - low_threshold = 2 * 0.05 = 0.1 points
    """

    high_percent: float = 20.0  # 20% of scale range → violation
    low_percent: float = 5.0    # 5% of scale range → good

    def label_from_error(
        self,
        error: float,
        scale: Optional[ScaleDefinition] = None,
    ) -> Optional[TrainingExampleLabel]:
        """
        Convert error to label with adaptive thresholds.

        Args:
            error: Raw error (not normalized)
            scale: Required - used to compute adaptive thresholds

        Returns:
            POSITIVE if error exceeds high threshold,
            NEGATIVE if error below low threshold,
            None if in middle range

        Raises:
            ValueError: If scale is not provided
        """
        if scale is None:
            raise ValueError(
                "AdaptiveLabeler requires a scale. "
                "Use ThresholdLabeler for pre-normalized errors."
            )

        high_threshold = scale.range * (self.high_percent / 100.0)
        low_threshold = scale.range * (self.low_percent / 100.0)

        if error >= high_threshold:
            return TrainingExampleLabel.POSITIVE
        elif error <= low_threshold:
            return TrainingExampleLabel.NEGATIVE
        else:
            return None


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Protocol
    "LabelingStrategy",
    # Registry
    "get_labeler",
    "list_labelers",
    "register_labeler",
    # Implementations
    "ThresholdLabeler",
    "BinaryLabeler",
    "PercentileLabeler",
    "AdaptiveLabeler",
    # Default instance
    "DEFAULT_LABELER",
]
