"""
Score-centric oracle types for the OPS engine.

This module provides the core types for score-based oracles that align with
DSPy's metric optimization paradigm. The key insight is:
- Score is the PRIMARY output (0.0-1.0, where 1.0 = good/similar)
- Classification is DERIVED (threshold the score when needed)

Usage:
    from treepo._research.core.scoring import OracleScore, ScoringOracle, oracle_as_metric

    class MyScorer:
        def score(self, input_a: str, input_b: str, rubric: str) -> OracleScore:
            similarity = compute_similarity(input_a, input_b)
            return OracleScore(score=similarity, reasoning="...")

    # Use directly as DSPy metric
    metric = oracle_as_metric(MyScorer())
    optimizer.compile(student, trainset, metric=metric)
"""

import threading
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

from treepo._research.core.conditional_memory import canonical_hash

if TYPE_CHECKING:
    from treepo._research.core.conditional_memory import ConditionalMemory


# =============================================================================
# Score Normalization Utility
# =============================================================================

def normalize_error_to_score(error: float, max_error: float) -> float:
    """
    Convert an error value to a normalized score where 1.0 = best, 0.0 = worst.

    This is the standard pattern for converting distance/error to similarity/score:
    - score = 1.0 when error = 0
    - score = 0.0 when error >= max_error
    - Linear interpolation between

    Args:
        error: The absolute error/distance (will use abs(error) internally)
        max_error: The maximum error value (defines the scale)

    Returns:
        Normalized score in [0.0, 1.0] where higher is better

    Raises:
        ValueError: If max_error <= 0

    Examples:
        # Bounded scale similarity (e.g., scale from -100 to +100, range=200)
        score = normalize_error_to_score(abs(value_a - value_b), max_error=200.0)
        # value_a=-50, value_b=+50 → error=100 → score=0.5

        # Configurable scale (e.g., 100-point scale)
        score = normalize_error_to_score(abs(pred - gt), max_error=100.0)
        # pred=45, gt=50 → error=5 → score=0.95

        # Generic distance
        score = normalize_error_to_score(distance, max_distance)
    """
    if max_error <= 0:
        raise ValueError(f"max_error must be positive, got {max_error}")
    return max(0.0, 1.0 - abs(error) / max_error)


def score_to_error(score: float, max_error: float) -> float:
    """
    Inverse of normalize_error_to_score.

    Given a normalized score (0-1 where 1=good), compute the error.

    Args:
        score: Normalized score in [0.0, 1.0] where 1.0 = perfect
        max_error: The maximum error value (defines the scale)

    Returns:
        Error value: max_error * (1 - score)

    Examples:
        # Convert score back to error
        error = score_to_error(0.5, max_error=200.0)  # → 100.0 points

        # Perfect score
        error = score_to_error(1.0, max_error=200.0)  # → 0.0 points

        # Worst score
        error = score_to_error(0.0, max_error=200.0)  # → 200.0 points
    """
    return max_error * (1.0 - score)


# =============================================================================
# Bounded Scale
# =============================================================================

@dataclass(frozen=True)
class BoundedScale:
    """
    A bounded linear scale for score normalization.

    Represents a continuous range with defined bounds.
    Handles the math of converting distances to normalized scores.

    This class provides compatibility with the SimilarityScorer and Oracle
    classes in this module. For new code, consider using ScaleDefinition
    from treepo._research.tasks.base which provides additional metadata like description,
    higher_is_better, and neutral_value.

    Examples:
        # Political positioning (-100 to +100)
        political = BoundedScale(-100.0, 100.0)
        score = political.values_to_score(pred, gt)  # 0.0-1.0

        # Percentage scale (0 to 100)
        pct = BoundedScale(0.0, 100.0)
    """
    min_value: float
    max_value: float

    @property
    def range(self) -> float:
        """Get the range of the scale."""
        return self.max_value - self.min_value

    def normalize(self, value: float) -> float:
        """Normalize a value to 0-1 range."""
        return (value - self.min_value) / self.range

    def denormalize(self, normalized: float) -> float:
        """Convert from 0-1 range back to scale range."""
        return normalized * self.range + self.min_value

    def clamp(self, value: float) -> float:
        """Clamp a value to the valid range."""
        return max(self.min_value, min(self.max_value, value))

    def distance_to_score(self, distance: float) -> float:
        """
        Convert distance to normalized similarity score.

        Args:
            distance: Absolute distance between two values

        Returns:
            Score in [0.0, 1.0] where 1.0 = no distance (identical)
        """
        return max(0.0, 1.0 - abs(distance) / self.range)

    def values_to_score(self, value_a: float, value_b: float) -> float:
        """
        Compute similarity score between two values on this scale.

        Args:
            value_a: First value
            value_b: Second value

        Returns:
            Score in [0.0, 1.0] where 1.0 = identical values
        """
        distance = abs(value_a - value_b)
        return self.distance_to_score(distance)


# Common scale constants
UNIT_SCALE = BoundedScale(0.0, 1.0)        # [0, 1] - normalized
PERCENT_SCALE = BoundedScale(0.0, 100.0)   # [0, 100] - percentages
SYMMETRIC_SCALE = BoundedScale(-1.0, 1.0)  # [-1, +1] - centered


# =============================================================================
# Generic Oracle (Single-Text Prediction)
# =============================================================================

@dataclass
class OraclePrediction:
    """
    Result from an Oracle predicting a value from a single text.

    Unlike OracleScore (which compares two texts), OraclePrediction
    represents a prediction about a single piece of text.

    Attributes:
        value: Predicted value on the oracle's scale
        confidence: 0-1 confidence in the prediction
        reasoning: Human-readable explanation

    Example:
        # Oracle predicting a value
        pred = oracle.predict("Some text to analyze...")
        print(pred.value)       # 45.2 (on the oracle's scale)
        print(pred.confidence)  # 0.85
        print(pred.reasoning)   # "Explanation of prediction..."
    """
    value: float
    confidence: float
    reasoning: str

    def __post_init__(self):
        """Validate confidence is in [0, 1]."""
        self.confidence = max(0.0, min(1.0, self.confidence))


class Oracle(ABC):
    """
    Abstract base class for single-text prediction oracles.

    An Oracle takes text and predicts a numeric value on a bounded scale.
    This is the generic foundation for domain-specific oracles.

    Unlike ScoringOracle (which compares two texts), Oracle predicts
    a value from a single text - useful for classification and regression.

    Example:
        class SentimentOracle(Oracle):
            def __init__(self):
                super().__init__(SYMMETRIC_SCALE)  # -1 to +1

            def predict(self, text: str) -> OraclePrediction:
                sentiment = self._analyze(text)
                return OraclePrediction(
                    value=sentiment,
                    confidence=0.9,
                    reasoning="Positive language detected"
                )

        oracle = SentimentOracle()
        pred = oracle.predict("I love this product!")
        accuracy = oracle.score_accuracy(pred.value, ground_truth=0.8)
    """

    def __init__(self, scale: BoundedScale):
        """
        Initialize oracle with its output scale.

        Args:
            scale: BoundedScale defining the range of predicted values
        """
        self.scale = scale

    @abstractmethod
    def predict(self, text: str) -> OraclePrediction:
        """
        Predict a value for the given text.

        Args:
            text: Input text to analyze

        Returns:
            OraclePrediction with value, confidence, and reasoning
        """
        pass

    def score_error(self, predicted: float, ground_truth: float) -> float:
        """
        Compute normalized error between prediction and ground truth.

        Returns error in [0, 1] where 0 = perfect, 1 = max error.

        Args:
            predicted: Predicted value
            ground_truth: True value

        Returns:
            Normalized error (0 = perfect)
        """
        error = abs(predicted - ground_truth)
        return min(1.0, error / self.scale.range)

    def score_accuracy(self, predicted: float, ground_truth: float) -> float:
        """
        Compute normalized accuracy between prediction and ground truth.

        Returns accuracy in [0, 1] where 1 = perfect, 0 = max error.
        This is the inverse of score_error and suitable for DSPy metrics.

        Args:
            predicted: Predicted value
            ground_truth: True value

        Returns:
            Normalized accuracy (1 = perfect)
        """
        return 1.0 - self.score_error(predicted, ground_truth)

    def predict_and_score(self, text: str, ground_truth: float) -> Tuple[OraclePrediction, float]:
        """
        Predict value and compute accuracy against ground truth.

        Convenience method combining predict() and score_accuracy().

        Args:
            text: Input text to analyze
            ground_truth: True value to compare against

        Returns:
            Tuple of (prediction, accuracy_score)
        """
        pred = self.predict(text)
        accuracy = self.score_accuracy(pred.value, ground_truth)
        return pred, accuracy


# =============================================================================
# Generic Similarity Scorer
# =============================================================================

class SimilarityScorer:
    """
    Generic similarity scorer for comparing two texts via extracted values.

    This is the generic foundation for domain-specific scorers. It:
    1. Extracts a numeric value from each text using value_extractor
    2. Computes similarity using BoundedScale.values_to_score()
    3. Returns an OracleScore

    Example:
        # Domain-specific similarity scorer
        def extract_value(text: str) -> float:
            # Call LLM or model to extract a numeric value
            return model.predict(text=text)['score']

        scale = BoundedScale(-100.0, 100.0)
        scorer = SimilarityScorer(extract_value, scale, name="metric")
        result = scorer.score(original, summary, rubric)
        print(result.score)  # 0.95 (similarity)

        # Sentiment similarity example
        scorer = SimilarityScorer(sentiment_fn, SYMMETRIC_SCALE)
    """

    def __init__(
        self,
        value_extractor: Callable[[str], float],
        scale: BoundedScale,
        name: str = "value",
        cache_size: int = 1024,
        memory: Optional["ConditionalMemory"] = None,
    ):
        """
        Initialize the similarity scorer.

        Args:
            value_extractor: Function (text) -> float that extracts a value
            scale: BoundedScale for normalizing the comparison
            name: Name for the extracted value (used in reasoning)
            cache_size: Max entries to cache (0 = no caching). Caching avoids
                        redundant LLM calls when scoring same texts repeatedly.
            memory: Optional ConditionalMemory instance. When provided, extracted
                    values are cached persistently (L1+SQLite L2) under a
                    namespaced key, enabling cross-run reuse.
        """
        self._value_extractor_raw = value_extractor
        self.scale = scale
        self.name = name
        self.cache_size = cache_size
        self._cache: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.Lock()  # Thread safety for cache operations
        self._memory = memory

    def _text_hash(self, text: str) -> str:
        """Compute deterministic cache key via shared canonical hash utility."""
        return canonical_hash(text)

    def value_extractor(self, text: str) -> float:
        """Cached value extraction. Avoids redundant LLM calls for same text.

        Thread-safe implementation using a lock to protect cache operations.
        When a ConditionalMemory instance is available, it is checked first
        (providing cross-run persistence and unified stats).
        """
        if self.cache_size == 0 and self._memory is None:
            return self._value_extractor_raw(text)

        # Check ConditionalMemory first (cross-run persistent tier).
        if self._memory is not None:
            namespace = f"similarity:{self.name}:{self._memory.namespace_version}"
            mem_key = canonical_hash(text)
            cached = self._memory.get_json(namespace, mem_key)
            if cached is not None:
                try:
                    return float(cached)
                except (TypeError, ValueError):
                    pass

        key = self._text_hash(text)

        # Check local cache with lock
        with self._cache_lock:
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1

        # Call extractor outside lock to avoid holding lock during LLM call
        value = self._value_extractor_raw(text)

        # Add to local cache with lock
        with self._cache_lock:
            # Double-check in case another thread added it
            if key in self._cache:
                return self._cache[key]

            # Add to cache (with simple LRU eviction if full)
            if self.cache_size > 0 and len(self._cache) >= self.cache_size:
                # Remove oldest entry (first key in dict - Python 3.7+ preserves order)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            if self.cache_size > 0:
                self._cache[key] = value

        # Store in ConditionalMemory for cross-run persistence.
        if self._memory is not None:
            namespace = f"similarity:{self.name}:{self._memory.namespace_version}"
            mem_key = canonical_hash(text)
            self._memory.set_json(namespace, mem_key, float(value))

        return value

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics (thread-safe)."""
        with self._cache_lock:
            total = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": hit_rate,
                "entries": len(self._cache),
                "max_entries": self.cache_size,
            }

    def score(
        self,
        input_a: str,
        input_b: str,
        rubric: str = "",
    ) -> 'OracleScore':
        """
        Score similarity between two texts.

        Args:
            input_a: First text
            input_b: Second text
            rubric: Optional context (not used in base implementation)

        Returns:
            OracleScore with similarity (1.0 = identical values)
        """
        try:
            val_a = self.value_extractor(input_a)
            val_b = self.value_extractor(input_b)

            similarity = self.scale.values_to_score(val_a, val_b)
            diff = abs(val_a - val_b)

            return OracleScore(
                score=similarity,
                reasoning=f"{self.name}: {val_a:.1f} vs {val_b:.1f}, diff={diff:.1f}",
                metadata={
                    f'{self.name}_a': val_a,
                    f'{self.name}_b': val_b,
                    'difference': diff,
                },
            )

        except Exception as e:
            return OracleScore(
                score=0.0,
                reasoning=f"Scorer error: {str(e)}",
            )


# =============================================================================
# Oracle Score
# =============================================================================

@dataclass(frozen=True)
class OracleScore:
    """
    Primary output of a ScoringOracle.

    Convention: score uses SIMILARITY (1.0 = good, 0.0 = bad)
    This aligns with DSPy metrics and most ML conventions.

    Attributes:
        score: Primary output, 0.0-1.0 where higher = better match
        reasoning: Human-readable explanation of the score
        metadata: Optional domain-specific details (e.g., {'value_a': 45, 'value_b': 52})
    """

    score: float
    reasoning: str
    metadata: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        """Validate and clamp score to [0.0, 1.0] range.

        From BoundedMetricSpace.lean: The OPS proofs require bounded metrics
        where dist(x,y) <= diameterBound. For scores, we require 0 <= score <= 1.
        """
        if not 0.0 <= self.score <= 1.0:
            clamped = max(0.0, min(1.0, self.score))
            # Warn user about clamping - this may indicate a bug in their oracle
            warnings.warn(
                f"OracleScore.score={self.score} is outside [0.0, 1.0] bounds, "
                f"clamped to {clamped}. Bounded scores are required for theoretical "
                "guarantees (see BoundedMetricSpace.lean).",
                UserWarning,
                stacklevel=2
            )
            object.__setattr__(self, 'score', clamped)

    def passes_threshold(self, threshold: float = 0.9) -> bool:
        """
        Derive classification from score.

        Use this when you need a binary decision (e.g., audit flagging).

        Args:
            threshold: Minimum score to pass (default 0.9 = 90% similar)

        Returns:
            True if score >= threshold
        """
        return self.score >= threshold

    def as_metric(self) -> float:
        """
        Return score directly usable as DSPy metric.

        Since OracleScore.score is already 0.0-1.0 with 1.0 = good,
        it can be used directly as a DSPy metric return value.
        """
        return self.score

    @classmethod
    def from_error(
        cls,
        error: float,
        max_error: float,
        reasoning: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'OracleScore':
        """
        Create OracleScore from error value using standard normalization.

        This is a convenience constructor for the common pattern of converting
        an error/distance to a normalized score.

        Args:
            error: The absolute error/distance value (will be abs()'d)
            max_error: The maximum error value (defines the scale)
            reasoning: Explanation of the score
            metadata: Optional domain-specific details

        Returns:
            OracleScore with normalized score (1.0 = no error, 0.0 = max error)

        Example:
            # From prediction error (using scale range)
            score = OracleScore.from_error(
                error=abs(predicted - ground_truth),
                max_error=200.0,  # Use scale.range for your domain
                reasoning=f"Prediction error: {abs(predicted - ground_truth):.0f}"
            )
        """
        score = normalize_error_to_score(error, max_error)
        return cls(score=score, reasoning=reasoning, metadata=metadata)


@runtime_checkable
class ScoringOracle(Protocol):
    """
    Protocol for score-centric oracles.

    Unlike the legacy OracleJudge which returns classification first,
    ScoringOracle returns a continuous score as the primary output.

    Implementations should return OracleScore with:
    - score: 0.0-1.0, where 1.0 means perfect match/similarity
    - reasoning: Human-readable explanation
    - metadata: Optional domain-specific details

    Example:
        class MyScorer:
            def score(self, input_a: str, input_b: str, rubric: str) -> OracleScore:
                value_a = self._compute_value(input_a)
                value_b = self._compute_value(input_b)
                similarity = normalize_error_to_score(abs(value_a - value_b), self.scale.range)
                return OracleScore(
                    score=similarity,
                    reasoning=f"Values: {value_a} vs {value_b}",
                    metadata={'value_a': value_a, 'value_b': value_b}
                )
    """

    def score(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> OracleScore:
        """
        Score similarity between two inputs according to rubric.

        Args:
            input_a: First input (typically original/source text)
            input_b: Second input (typically summary/target text)
            rubric: Criteria/context for comparison

        Returns:
            OracleScore with score (1.0 = perfect match, 0.0 = no match)
        """
        ...


# =============================================================================
# Metric Integration
# =============================================================================

def oracle_as_metric(
    oracle: ScoringOracle,
    original_field: str = 'original',
    summary_field: str = 'summary',
    rubric_field: str = 'rubric',
) -> Callable:
    """
    Convert a ScoringOracle to a DSPy-compatible metric function.

    The oracle's score IS the metric - no complex wrapping needed.
    This is the primary way to use oracles for DSPy optimization.

    Args:
        oracle: ScoringOracle implementation
        original_field: Attribute name for original text on gold example
        summary_field: Attribute name for summary on prediction
        rubric_field: Attribute name for rubric on gold example

    Returns:
        DSPy metric function: (gold, pred, trace?) -> float

    Example:
        scorer = MyScorer()
        metric = oracle_as_metric(scorer)

        # Use in optimization
        optimizer = dspy.BootstrapFewShot(metric=metric)
        compiled = optimizer.compile(student, trainset)
    """

    def metric(gold, pred, trace=None) -> float:
        # Extract texts from example/prediction objects
        original = getattr(gold, original_field, '') or getattr(gold, 'text', '') or str(gold)
        summary = getattr(pred, summary_field, '') or str(pred)
        rubric = getattr(gold, rubric_field, '')

        # Score and return directly
        result = oracle.score(original, summary, rubric)
        return result.as_metric()

    return metric


def oracle_as_metric_with_feedback(
    oracle: ScoringOracle,
    original_field: str = 'original',
    summary_field: str = 'summary',
    rubric_field: str = 'rubric',
) -> Callable:
    """
    Convert a ScoringOracle to a GEPA-compatible metric with feedback.

    GEPA can use feedback strings for reflection-based optimization.

    Args:
        oracle: ScoringOracle implementation
        original_field: Attribute name for original text on gold example
        summary_field: Attribute name for summary on prediction
        rubric_field: Attribute name for rubric on gold example

    Returns:
        GEPA metric function: (gold, pred, trace?, pred_name?, pred_trace?) -> dict
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dict:
        original = getattr(gold, original_field, '') or getattr(gold, 'text', '') or str(gold)
        summary = getattr(pred, summary_field, '') or str(pred)
        rubric = getattr(gold, rubric_field, '')

        result = oracle.score(original, summary, rubric)

        return {
            'score': result.score,
            'feedback': result.reasoning,
        }

    return metric


# =============================================================================
# Oracle Scorer Factory
# =============================================================================

def create_oracle_scorer(
    scorer_module,
    task_context: str,
    score_field: str = "score",
    scale: Optional[BoundedScale] = None,
) -> Callable[[str], float]:
    """
    Factory for creating oracle scorer functions from DSPy modules.

    This creates a simple function that takes text and returns a score,
    suitable for use in tournament-of-tournaments or preference collection.

    Args:
        scorer_module: A DSPy module with a forward(text, task_context) method
                       that returns a dict with the score
        task_context: Context/instructions for the scoring task
        score_field: Name of the field containing the score in the result dict
        scale: Optional BoundedScale to normalize the score to 0-1

    Returns:
        Function(text: str) -> float

    Example:
        from treepo._research.core.scorers import ScaleScorer
        from treepo._research.core.signatures import MetricScore

        scorer = ScaleScorer(MetricScore)
        oracle_fn = create_oracle_scorer(
            scorer_module=scorer,
            task_context="Rate quality 0-10",
            score_field="score",
        )
        score = oracle_fn("Some text to score")  # Returns 0.0-1.0
    """
    import logging
    logger = logging.getLogger(__name__)

    def oracle_predict(text: str) -> float:
        """Predict score for text."""
        try:
            result = scorer_module(text=text, task_context=task_context)

            # Extract raw score
            if isinstance(result, dict):
                raw_score = float(result.get(score_field, 0.0))
            elif hasattr(result, score_field):
                raw_score = float(getattr(result, score_field))
            else:
                raw_score = 0.0

            # Normalize if scale provided
            if scale:
                return scale.normalize(raw_score)
            return raw_score

        except Exception as e:
            logger.warning(f"Oracle prediction failed: {e}")
            return 0.5 if scale else 0.0  # Return neutral on error

    return oracle_predict
