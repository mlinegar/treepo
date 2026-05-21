"""
Evaluation Metrics for Oracle Approximation.

This module provides metrics for evaluating:
- Continuous score prediction (using BoundedScale)
- OPS law compliance rates
- DSPy-compatible metric functions

New API (Preferred):
    from treepo._research.core.scoring import oracle_as_metric, ScoringOracle

    # ScoringOracle -> DSPy metric in one line
    metric = oracle_as_metric(my_scorer)
    optimizer.compile(student, trainset, metric=metric)

Summarization metrics:
    from treepo._research.training.metrics import summarization

    # When you need specialized summarization metrics with quality checks
    metric = summarization(oracle_classifier, ...)
"""

from dataclasses import dataclass
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Callable, Any
import math
import threading
import warnings

if TYPE_CHECKING:
    from treepo._research.core.conditional_memory import ConditionalMemory

from treepo._research.core.conditional_memory import canonical_hash, hash_payload

from treepo._research.training.core import Prediction, LawCheckResult

# Re-export score-centric metric converters for convenience
from treepo._research.core.scoring import (
    oracle_as_metric,
    oracle_as_metric_with_feedback,
    ScoringOracle,
    OracleScore,
    normalize_error_to_score,
    BoundedScale,
    PERCENT_SCALE,
    OraclePrediction,
)


# =============================================================================
# Utilities
# =============================================================================

def _extract_text_from_pred(pred, *attr_names: str, default: str = "") -> str:
    """Extract text from a prediction, checking multiple attribute names.

    Handles both string predictions and object predictions with various
    attribute names (summary, merged_summary, etc).

    Args:
        pred: Prediction value (str or object)
        *attr_names: Attribute names to check in order (e.g., 'summary', 'merged_summary')
        default: Default value if no text found (defaults to empty string)

    Returns:
        Extracted text string
    """
    if isinstance(pred, str):
        return pred

    for attr in attr_names:
        text = getattr(pred, attr, None)
        if text:
            return text

    return str(pred) if pred else default


def _resolve_single_text_predictor(predictor: Any) -> Callable[[str], Any]:
    """Normalize predictor interfaces into a callable (text) -> prediction."""
    if hasattr(predictor, "predict_score"):
        return predictor.predict_score
    if hasattr(predictor, "predict"):
        return predictor.predict
    if callable(predictor):
        return predictor
    raise ValueError("predictor must be callable or implement predict_score/predict")


def _coerce_prediction_result(result: Any) -> Tuple[float, float, str]:
    """Coerce prediction results into (value, confidence, reasoning)."""
    if isinstance(result, OraclePrediction):
        return float(result.value), float(result.confidence), str(result.reasoning)
    if isinstance(result, tuple):
        value = result[0] if len(result) > 0 else 0.0
        confidence = result[1] if len(result) > 1 else 1.0
        reasoning = result[2] if len(result) > 2 else ""
        return float(value), float(confidence), str(reasoning)
    if isinstance(result, dict):
        if "value" in result:
            value = result.get("value", 0.0)
        else:
            value = result.get("score", result.get("prediction", 0.0))
        confidence = result.get("confidence", 1.0)
        reasoning = result.get("reasoning", "")
        return float(value), float(confidence), str(reasoning)
    if hasattr(result, "value"):
        return (
            float(getattr(result, "value")),
            float(getattr(result, "confidence", 1.0)),
            str(getattr(result, "reasoning", "")),
        )
    if hasattr(result, "score"):
        return (
            float(getattr(result, "score")),
            float(getattr(result, "confidence", 1.0)),
            str(getattr(result, "reasoning", "")),
        )
    return float(result), 1.0, ""


# =============================================================================
# Generic Metric Factory (Scale-Based)
# =============================================================================

def metric(
    oracle_fn: Callable,
    scale: BoundedScale,
    ground_truth_field: str = 'ground_truth',
    prediction_field: str = 'label',
    with_feedback: bool = False,
) -> Callable:
    """
    Create a DSPy metric returning a normalized 0-1 score.

    Score computation (euclidean distance normalized by scale range):
        score = 1 - abs(predicted - ground_truth) / scale.range

    Examples:
        - pred=50, gt=50   → score = 1.0 (perfect match)
        - pred=50, gt=-50  → score = 1 - 100/200 = 0.5
        - pred=100, gt=-100 → score = 1 - 200/200 = 0.0 (max error)

    Args:
        oracle_fn: Function that scores text, returns value on the scale.
            Signature: (text: str) -> float  OR
            Signature: (text: str) -> Tuple[float, float, str] (value, confidence, reasoning)
        scale: BoundedScale defining the value range for normalization
        ground_truth_field: Field name for ground truth on gold example
        prediction_field: Field name for predicted value on prediction
        with_feedback: Return dict with feedback for GEPA

    Returns:
        DSPy-compatible metric function returning pure score (0-1).
        Use create_combined_metric() to combine multiple metrics with weights.

    Example:
        scale = BoundedScale(-1.0, 1.0)
        metric = create_metric(oracle_fn=my_oracle.predict, scale=scale)
    """
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Get ground truth
        ground_truth = getattr(gold, ground_truth_field, 0.0)

        # Get prediction text
        text_to_score = _extract_text_from_pred(pred, 'summary', prediction_field)

        # Call oracle function
        try:
            result = oracle_fn(text_to_score)
            # Handle different return types
            if isinstance(result, tuple):
                predicted_value = result[0]  # (value, confidence, reasoning)
                reasoning = result[2] if len(result) > 2 else ""
            else:
                predicted_value = float(result)
                reasoning = ""
        except Exception as e:
            if with_feedback:
                return {'score': 0.0, 'feedback': f"Oracle error: {str(e)}"}
            return 0.0

        # Compute score using scale
        score = scale.values_to_score(predicted_value, ground_truth)

        if with_feedback:
            feedback = f"Predicted {predicted_value:.1f}, expected {ground_truth:.1f}"
            if reasoning:
                feedback += f" ({reasoning})"
            return {'score': score, 'feedback': feedback}

        return score

    return metric



# Note: cached_oracle has been removed.
# Use create_cached_oracle_metric() or @functools.lru_cache for oracle prediction caching.

# Note: path_aggregate_score and probabilistic_audit have been removed.
# These were complex tree-path-based metrics that are no longer used.
# Use oracle_score_prediction for simpler oracle-based scoring.

# =============================================================================
# Calibration Metrics
# =============================================================================

def calibration_error(
    predictions: List[Prediction],
    ground_truth: List[str],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration error metrics.

    A well-calibrated classifier should have predictions with confidence X
    being correct X% of the time.

    Args:
        predictions: List of Prediction objects with confidence scores
        ground_truth: List of true labels
        n_bins: Number of confidence bins

    Returns:
        Dict with ECE (expected calibration error) and MCE (max calibration error)
    """
    if not predictions or len(predictions) != len(ground_truth):
        return {'ece': 0.0, 'mce': 0.0}

    # Bin predictions by confidence
    bins = [[] for _ in range(n_bins)]
    for pred, true in zip(predictions, ground_truth):
        bin_idx = min(int(pred.confidence * n_bins), n_bins - 1)
        correct = 1 if pred.label == true else 0
        bins[bin_idx].append((pred.confidence, correct))

    # Compute per-bin accuracy and confidence
    ece = 0.0
    mce = 0.0
    total = len(predictions)

    for bin_data in bins:
        if not bin_data:
            continue

        n = len(bin_data)
        avg_confidence = sum(c for c, _ in bin_data) / n
        accuracy = sum(corr for _, corr in bin_data) / n
        gap = abs(accuracy - avg_confidence)

        # Weighted contribution to ECE
        ece += (n / total) * gap
        mce = max(mce, gap)

    return {'ece': ece, 'mce': mce}


def calibration_diagram_data(
    predictions: List[Prediction],
    ground_truth: List[str],
    n_bins: int = 10,
) -> List[Dict]:
    """
    Compute data for a calibration diagram.

    Args:
        predictions: List of Prediction objects
        ground_truth: List of true labels
        n_bins: Number of confidence bins

    Returns:
        List of dicts with bin_center, accuracy, confidence, and count
    """
    if not predictions:
        return []

    # Bin predictions
    bins = [[] for _ in range(n_bins)]
    for pred, true in zip(predictions, ground_truth):
        bin_idx = min(int(pred.confidence * n_bins), n_bins - 1)
        correct = 1 if pred.label == true else 0
        bins[bin_idx].append((pred.confidence, correct))

    # Compute per-bin stats
    data = []
    for i, bin_data in enumerate(bins):
        bin_center = (i + 0.5) / n_bins
        if bin_data:
            accuracy = sum(corr for _, corr in bin_data) / len(bin_data)
            confidence = sum(c for c, _ in bin_data) / len(bin_data)
            count = len(bin_data)
        else:
            accuracy = None
            confidence = None
            count = 0

        data.append({
            'bin_center': bin_center,
            'accuracy': accuracy,
            'confidence': confidence,
            'count': count,
        })

    return data


# =============================================================================
# OPS Law Compliance Metrics
# =============================================================================

def law_compliance_rate(
    check_results: List[LawCheckResult],
) -> Dict[str, float]:
    """
    Compute compliance rate for each OPS law.

    Args:
        check_results: List of LawCheckResult objects

    Returns:
        Dict mapping law name to pass rate [0, 1]
    """
    by_law: Dict[str, List[bool]] = {}

    for result in check_results:
        if result.law not in by_law:
            by_law[result.law] = []
        by_law[result.law].append(result.passed)

    return {
        law: sum(passes) / len(passes) if passes else 1.0
        for law, passes in by_law.items()
    }


def overall_compliance_rate(check_results: List[LawCheckResult]) -> float:
    """
    Compute overall OPS law compliance rate.

    Args:
        check_results: List of LawCheckResult objects

    Returns:
        Overall pass rate [0, 1]
    """
    if not check_results:
        return 1.0

    passed = sum(r.passed for r in check_results)
    return passed / len(check_results)


def average_discrepancy(
    check_results: List[LawCheckResult],
) -> Dict[str, float]:
    """
    Compute average discrepancy for each OPS law.

    Args:
        check_results: List of LawCheckResult objects

    Returns:
        Dict mapping law name to average discrepancy
    """
    by_law: Dict[str, List[float]] = {}

    for result in check_results:
        if result.law not in by_law:
            by_law[result.law] = []
        by_law[result.law].append(result.discrepancy)

    return {
        law: sum(discs) / len(discs) if discs else 0.0
        for law, discs in by_law.items()
    }


# =============================================================================
# DSPy Metric Functions
# =============================================================================

# Note: classification() and violation() have been removed.
# Use continuous score prediction with oracle_as_metric() or metric() instead.
# Use compliance() for law compliance metrics.


def compliance(
    tolerance: float = 0.0,
) -> Callable:
    """
    Create a DSPy metric based on OPS law compliance.

    Args:
        tolerance: Allowed discrepancy before counting as failure

    Returns:
        Metric function
    """
    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
        """DSPy metric for law compliance. Compatible with GEPA's 5-argument signature."""
        # Get discrepancy from prediction if available
        discrepancy = getattr(prediction, 'discrepancy', None)
        if discrepancy is not None:
            return 1.0 if discrepancy <= tolerance else 0.0

        # Otherwise, compare labels directly
        true_label = getattr(example, 'label', None)
        pred_label = getattr(prediction, 'label', None)

        if true_label is None or pred_label is None:
            return 0.0

        return 1.0 if true_label == pred_label else 0.0

    return metric


# =============================================================================
# Advanced DSPy Metrics (LLM Judge)
# =============================================================================

# Note: classification_trace() has been removed. Use continuous score prediction
# with oracle_as_metric() or metric() instead.


def llm_judge(judge_lm=None) -> Callable:
    """
    Create a metric that uses an LLM as judge for nuanced evaluation.

    This follows DSPy best practices for using DSPy programs as metrics.
    Useful when:
    - Simple distance metrics miss important nuances
    - Need to evaluate reasoning quality
    - Want domain-expert-like feedback

    Args:
        judge_lm: Optional LM to use for judging (uses default if None)

    Returns:
        Metric function that uses LLM as judge
    """
    try:
        import dspy
        from contextlib import nullcontext
    except ImportError:
        # Return a fallback metric if DSPy not available
        def fallback_metric(gold, pred, trace=None):
            return 0.5  # Neutral score
        return fallback_metric

    # Define the judge signature
    class MetricJudge(dspy.Signature):
        """Evaluate the quality of a classification prediction."""

        original_text: str = dspy.InputField(desc="The original text")
        summary: str = dspy.InputField(desc="The summary being classified")
        predicted_label: str = dspy.InputField(desc="The predicted label")
        predicted_reasoning: str = dspy.InputField(desc="The model's reasoning for the prediction")
        true_label: str = dspy.InputField(desc="The ground truth label")

        score: float = dspy.OutputField(desc="Score from 0.0 to 1.0 based on prediction quality")
        feedback: str = dspy.OutputField(desc="Specific feedback for improvement")

    # Create the judge module
    judge = dspy.ChainOfThought(MetricJudge)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """
        LLM-as-judge metric for nuanced evaluation.

        Compatible with GEPA's 5-argument signature.

        Args:
            gold: Ground truth example
            pred: Prediction
            trace: Optional trace (not used by this metric)
            pred_name: Optional predictor name (for GEPA)
            pred_trace: Optional predictor trace (for GEPA)

        Returns:
            {'score': float, 'feedback': str}
        """
        # Get prediction details
        original_text = getattr(gold, 'original_content', '')
        if not original_text:
            original_text = getattr(gold, 'text', '')
        # Use full text - truncation corrupts evaluation

        summary = getattr(gold, 'summary', '')
        pred_label = str(getattr(pred, 'label', ''))
        pred_reasoning = getattr(pred, 'reasoning', '')
        true_label = str(getattr(gold, 'label', ''))

        if not true_label:
            return {'score': 0.0, 'feedback': 'No ground truth label.'}

        try:
            # Use judge LM if different from main LM
            context = dspy.context(lm=judge_lm) if judge_lm else nullcontext()

            with context:
                result = judge(
                    original_text=original_text,
                    summary=summary,
                    predicted_label=pred_label,
                    predicted_reasoning=pred_reasoning,
                    true_label=true_label,
                )

            # Parse score
            try:
                score = float(result.score)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except (ValueError, TypeError, AttributeError):
                score = 0.5  # Default if parsing fails

            feedback = getattr(result, 'feedback', 'No feedback provided.')

            return {'score': score, 'feedback': feedback}

        except Exception as e:
            # Fallback on any error
            return {
                'score': 0.5,
                'feedback': f'Judge evaluation failed: {str(e)}'
            }

    return metric


# Note: composite() has been removed. Use combine_metrics() or combine_feedback()
# with oracle_as_metric() for composing metrics.


# =============================================================================
# Summarization Metrics (for Two-Step Iterative Optimization)
# =============================================================================

def summarization(
    oracle_classifier,
    human_weight: float = 0.3,
    oracle_weight: float = 0.7,
    threshold: Optional[float] = None,
    min_summary_length: int = 50,
    max_error: Optional[float] = None,
    scale: Optional["BoundedScale"] = None,
    label_name: str = "score",
) -> Callable:
    """
    Create a metric for evaluating summary quality against score preservation.

    This metric is used to optimize summarization prompts in the two-step
    iterative process:
    1. Train oracle classifier on current summaries
    2. Optimize summarizers using oracle + human feedback as metric

    The metric combines:
    - Oracle score: Does the summary preserve the target score?
    - Human feedback: Historical corrections from review queue
    - Quality checks: Summary length, compression, clarity

    Args:
        oracle_classifier: Score predictor with predict_score(text), predict(text),
            or any callable returning a numeric score
        human_weight: Weight for human feedback score (0.0-1.0)
        oracle_weight: Weight for oracle-based score (should sum to 1.0 with human_weight)
        threshold: Maximum acceptable score drift (defaults to 5% of max_error)
        min_summary_length: Minimum acceptable summary length
        max_error: Scale range for error normalization. If None, uses scale.range
            when provided, otherwise defaults to 1.0.
        scale: Optional BoundedScale for deriving max_error
        label_name: Label used in feedback messages (e.g., "score")

    Returns:
        Metric function compatible with DSPy optimizers (including GEPA)

    Example:
        scale = BoundedScale(-1.0, 1.0)
        metric = summarization(oracle_classifier=oracle, scale=scale)
    """

    if max_error is None:
        max_error = scale.range if scale is not None else 1.0
    if threshold is None:
        threshold = max_error * 0.05

    predict_fn = _resolve_single_text_predictor(oracle_classifier)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """
        Evaluate summary quality for score preservation.

        Compatible with GEPA's 5-argument signature.

        Args:
            gold: Training example with:
                - original_text: The original chunk/text being summarized
                - reference_score: The true score for the full document
                - rubric: The preservation rubric
                - human_score: Optional human feedback score (0.0-1.0)
            pred: Prediction with:
                - summary: The generated summary
            trace: Optional DSPy trace (for debugging)
            pred_name: Optional predictor name (for GEPA)
            pred_trace: Optional predictor trace (for GEPA)

        Returns:
            {'score': float, 'feedback': str} for GEPA compatibility
        """
        # Extract inputs
        original_text = getattr(gold, 'original_text', '')
        if not original_text:
            original_text = getattr(gold, 'content', '')

        reference_score = getattr(gold, 'reference_score', 0.0)
        human_score = getattr(gold, 'human_score', None)

        # Extract prediction text
        summary = _extract_text_from_pred(pred, 'summary', 'merged_summary')

        # Initialize feedback
        feedback_parts = []
        scores = {}

        # 1. Oracle score: Does summary preserve the target score?
        try:
            predicted_value, confidence, _ = _coerce_prediction_result(
                predict_fn(summary)
            )
            score_diff = abs(predicted_value - reference_score)

            # Normalize to 0-1: perfect preservation = 1.0, max_error points off = 0.0
            oracle_score = normalize_error_to_score(score_diff, max_error=max_error)
            scores['oracle'] = oracle_score

            if score_diff > threshold:
                feedback_parts.append(
                    f"{label_name} drift detected: oracle predicted {predicted_value:.3f}, "
                    f"expected ~{reference_score:.3f} (diff={score_diff:.3f}). "
                    "Preserve more task-relevant content."
                )
            if confidence < 0.5:
                feedback_parts.append(
                    "Oracle uncertain about score prediction - "
                    "summary may be too vague or miss key indicators."
                )
        except Exception as e:
            oracle_score = 0.5  # Default on error
            scores['oracle'] = oracle_score
            feedback_parts.append(f"Oracle evaluation failed: {str(e)}")

        # 2. Quality checks
        quality_penalty = 0.0

        # Check summary length
        if len(summary) < min_summary_length:
            quality_penalty += 0.1
            feedback_parts.append(
                f"Summary too short ({len(summary)} chars < {min_summary_length}). "
                "May have lost important information."
            )

        # Check compression ratio
        if original_text:
            compression = len(summary) / max(len(original_text), 1)
            if compression < 0.05:  # Less than 5% of original
                quality_penalty += 0.05
                feedback_parts.append(
                    f"Extreme compression ({compression:.1%}) - may lose detail."
                )
            elif compression > 0.8:  # More than 80% of original
                quality_penalty += 0.05
                feedback_parts.append(
                    f"Low compression ({compression:.1%}) - summary should be more concise."
                )

        # Check for empty or minimal content
        if not summary.strip():
            quality_penalty = 0.5
            feedback_parts.append("Summary is empty.")
        elif len(summary.split()) < 10:
            quality_penalty += 0.1
            feedback_parts.append("Summary has very few words - may lack substance.")

        scores['quality'] = max(0.0, 1.0 - quality_penalty)

        # 3. Combine scores
        if human_score is not None:
            # Convex combination of oracle and human scores
            base_score = oracle_weight * oracle_score + human_weight * human_score
            scores['human'] = human_score
        else:
            # Oracle only
            base_score = oracle_score

        # Apply quality penalty
        final_score = max(0.0, base_score - quality_penalty)
        scores['final'] = final_score

        # Generate feedback
        if not feedback_parts:
            feedback = "Good score preservation and summary quality."
        else:
            feedback = ' '.join(feedback_parts)

        return {
            'score': final_score,
            'feedback': feedback,
            'details': scores,  # Additional detail for debugging
        }

    return metric


def create_merge_metric(
    oracle_classifier,
    scale: "BoundedScale",
    threshold: float = 10.0,
) -> Callable:
    """
    Create a metric for evaluating merge quality.

    Similar to summarization metric but specifically for merge operations
    where two summaries are combined.

    This is the preferred API - it requires an explicit scale parameter.

    Args:
        oracle_classifier: Score predictor with predict_score(), predict(), or callable
        scale: BoundedScale defining the value range for normalization
        threshold: Maximum acceptable score drift

    Returns:
        Metric function for merge evaluation

    Example:
        scale = BoundedScale(-1.0, 1.0)
        metric = create_merge_metric(oracle_classifier=oracle, scale=scale, threshold=0.1)
    """
    max_error = scale.range if hasattr(scale, 'range') else (scale.max_value - scale.min_value)
    predict_fn = _resolve_single_text_predictor(oracle_classifier)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """Evaluate merge quality for score preservation."""
        # Extract inputs
        left_summary = getattr(gold, 'left_summary', '')
        right_summary = getattr(gold, 'right_summary', '')
        ground_truth = getattr(gold, 'reference_score', 0.0)

        # Extract prediction text
        merged = _extract_text_from_pred(pred, 'merged_summary', 'summary')

        feedback_parts = []

        # Check merged preserves score
        try:
            merged_score, _, _ = _coerce_prediction_result(predict_fn(merged))
            score_diff = abs(merged_score - ground_truth)
            oracle_score = normalize_error_to_score(score_diff, max_error=max_error)

            if score_diff > threshold:
                feedback_parts.append(
                    f"Merge lost signal: {merged_score:.1f} vs expected {ground_truth:.1f}"
                )
        except Exception:
            oracle_score = 0.5

        # Check merge didn't lose too much content
        input_length = len(left_summary) + len(right_summary)
        output_length = len(merged)

        if output_length < input_length * 0.3:
            oracle_score *= 0.9
            feedback_parts.append(
                "Merge compressed significantly - may have lost information."
            )

        feedback = ' '.join(feedback_parts) if feedback_parts else "Good merge quality."

        return {
            'score': oracle_score,
            'feedback': feedback,
        }

    return metric


# =============================================================================
# Simple Metric Factories (DSPy-style)
# =============================================================================

def exact_match(
    gold_field: str = "answer",
    pred_field: str = "answer",
) -> Callable:
    """
    Create a simple exact match metric.

    This follows DSPy's pattern of simple boolean metrics.

    Args:
        gold_field: Field name to get from gold example
        pred_field: Field name to get from prediction

    Returns:
        Metric function returning bool

    Example:
        metric = create_exact_match_metric("label", "label")
        score = metric(gold_example, prediction)  # True/False
    """
    def metric(gold, pred, trace=None) -> bool:
        gold_val = getattr(gold, gold_field, None)
        pred_val = getattr(pred, pred_field, None)
        return gold_val == pred_val

    return metric


def numeric_match(
    gold_field: str,
    pred_field: str,
    tolerance: float = 0.0,
) -> Callable:
    """
    Create a numeric comparison metric with optional tolerance.

    Useful for ordinal or numeric predictions where close-enough is acceptable.

    Args:
        gold_field: Field name for gold value
        pred_field: Field name for predicted value
        tolerance: Maximum allowed difference

    Returns:
        Metric function returning bool

    Example:
        metric = create_numeric_match_metric("score", "score", tolerance=5.0)
        score = metric(gold, pred)  # True if within 5 points
    """
    def metric(gold, pred, trace=None) -> bool:
        try:
            gold_val = float(getattr(gold, gold_field, 0))
            pred_val = float(getattr(pred, pred_field, 0))
            return abs(gold_val - pred_val) <= tolerance
        except (ValueError, TypeError):
            return False

    return metric


def create_oracle_metric(
    oracle_fn: Callable,
    scale: "BoundedScale",
    input_field: str = "summary",
    gold_field: str = "reference_score",
    with_feedback: bool = False,
) -> Callable:
    """
    Create a unified oracle-as-metric function.

    This is the preferred API for creating oracle-based DSPy metrics.
    It consolidates the functionality of oracle() and oracle_score_prediction()
    into a single, clean interface.

    Args:
        oracle_fn: Callable that takes text and returns a score.
                   Signature: oracle_fn(text) -> float OR
                   Signature: oracle_fn(text) -> Tuple[float, float, str]
        scale: BoundedScale defining the value range for normalization.
        input_field: Field name to extract from prediction for oracle input
        gold_field: Field name for ground truth score
        with_feedback: Return dict with feedback for GEPA compatibility

    Returns:
        DSPy-compatible metric function (supports both 3-arg and 5-arg signatures)

    Example:
        scale = BoundedScale(-1.0, 1.0)
        metric = create_oracle_metric(oracle_fn=my_oracle.predict, scale=scale)
    """
    def metric_fn(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Get ground truth
        ground_truth = getattr(gold, gold_field, None)
        if ground_truth is None:
            ground_truth = getattr(gold, 'label', 0.0)
        try:
            ground_truth = float(ground_truth)
        except (ValueError, TypeError):
            ground_truth = (scale.min_value + scale.max_value) / 2

        # Get input from prediction
        if isinstance(pred, str):
            text_to_score = pred
        else:
            text_to_score = getattr(pred, input_field, None)
            if text_to_score is None:
                text_to_score = getattr(pred, 'summary', None)
            if text_to_score is None:
                text_to_score = str(pred)

        # Call oracle
        try:
            result = oracle_fn(text_to_score)
            if isinstance(result, tuple):
                predicted_value = result[0]
                reasoning = result[2] if len(result) > 2 else ""
            else:
                predicted_value = float(result)
                reasoning = ""
        except Exception as e:
            if with_feedback:
                return {'score': 0.0, 'feedback': f"Oracle error: {str(e)}"}
            return 0.0

        # Compute score using scale
        score = scale.values_to_score(predicted_value, ground_truth)

        if with_feedback:
            error = abs(predicted_value - ground_truth)
            feedback = f"Predicted {predicted_value:.1f}, expected {ground_truth:.1f} (error: {error:.1f})"
            if reasoning:
                feedback += f" - {reasoning}"
            return {'score': score, 'feedback': feedback}

        return score

    return metric_fn


class OraclePredictionCache:
    """Thread-safe LRU cache for oracle predictions.

    When an optional ``ConditionalMemory`` instance is provided, lookups and
    stores are delegated to it under the ``"oracle"`` score head.  This gives
    cross-run persistence (via L2/SQLite) and unified statistics without
    changing any callers.  The local ``OrderedDict`` is kept as an L0 hot
    tier for minimum-latency lookups within a single run.
    """

    def __init__(
        self,
        max_entries: int = 10000,
        memory: Optional["ConditionalMemory"] = None,
        *,
        oracle_id: str = "oracle",
        scale: Optional["BoundedScale"] = None,
    ):
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._memory = memory
        self._oracle_id = str(oracle_id or "oracle")
        self._scale = scale

    def _memory_namespace(self) -> Optional[str]:
        if self._memory is None:
            return None
        return f"oracle:{self._oracle_id}:{self._memory.namespace_version}"

    def _memory_key(self, text: str) -> Optional[str]:
        if self._memory is None or self._scale is None:
            return None
        text_hash = canonical_hash(text)
        payload = {
            "text": text_hash,
            "oracle_id": self._oracle_id,
            "scale": {"min": float(self._scale.min_value), "max": float(self._scale.max_value)},
        }
        return hash_payload(payload)

    @staticmethod
    def _serialize(value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, tuple):
            val = value[0] if len(value) > 0 else 0.0
            conf = value[1] if len(value) > 1 else 1.0
            reasoning = value[2] if len(value) > 2 else ""
            return {
                "type": "tuple",
                "value": [float(val), float(conf), str(reasoning)],
            }
        try:
            return {"type": "float", "value": float(value)}
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _deserialize(payload: Any) -> Optional[Any]:
        if isinstance(payload, dict):
            kind = str(payload.get("type", "")).strip().lower()
            if kind == "float":
                try:
                    return float(payload.get("value"))
                except (TypeError, ValueError):
                    return None
            if kind == "tuple":
                value = payload.get("value")
                if isinstance(value, list) and value:
                    val = float(value[0])
                    conf = float(value[1]) if len(value) > 1 else 1.0
                    reasoning = str(value[2]) if len(value) > 2 else ""
                    return (val, conf, reasoning)
        return None

    def get(self, key: str):
        """Return cached value or None, tracking hit/miss stats."""
        l0_key = canonical_hash(key)
        with self._lock:
            if l0_key in self._cache:
                self._cache.move_to_end(l0_key)
                self.hits += 1
                return self._cache[l0_key]

        # Check ConditionalMemory (cross-run persistent tier)
        if self._memory is not None:
            namespace = self._memory_namespace()
            mem_key = self._memory_key(key)
            if namespace and mem_key:
                cached = self._memory.get_json(namespace, mem_key)
                value = self._deserialize(cached)
            else:
                value = None
            if value is not None:
                with self._lock:
                    self.hits += 1
                    self._cache[l0_key] = value
                    self._cache.move_to_end(l0_key)
                return value

        with self._lock:
            self.misses += 1
        return None

    def set(self, key: str, value) -> None:
        """Insert value with LRU eviction."""
        l0_key = canonical_hash(key)
        with self._lock:
            if l0_key in self._cache:
                self._cache[l0_key] = value
                self._cache.move_to_end(l0_key)
                return

            if self.max_entries and len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)

            self._cache[l0_key] = value

        # Persist to ConditionalMemory for cross-run reuse
        if self._memory is not None:
            namespace = self._memory_namespace()
            mem_key = self._memory_key(key)
            payload = self._serialize(value)
            if namespace and mem_key and payload is not None:
                self._memory.set_json(namespace, mem_key, payload)

    def seed(self, values: Dict[str, Tuple[float, float, str]]) -> None:
        """Seed the cache with precomputed oracle predictions."""
        with self._lock:
            for key, value in values.items():
                if key in self._cache:
                    continue
                if self.max_entries and len(self._cache) >= self.max_entries:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        result = {
            "cache_size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "max_entries": self.max_entries,
        }
        if self._memory is not None:
            result["memory_stats"] = self._memory.report()
        return result


def _resolve_oracle_fn(oracle_classifier: Callable) -> Callable:
    """Normalize oracle interfaces into a callable (text) -> score/tuple."""
    return _resolve_single_text_predictor(oracle_classifier)


def create_cached_oracle_metric(
    oracle_classifier: Callable,
    scale: "BoundedScale",
    ground_truth_field: str = "reference_score",
    summary_field: str = "summary",
    cache_size: int = 10000,
    with_feedback: bool = False,
    memory: Optional["ConditionalMemory"] = None,
    oracle_id: Optional[str] = None,
) -> Tuple[Callable, Optional[OraclePredictionCache]]:
    """
    Create an oracle metric with memoized oracle predictions.

    Args:
        memory: Optional ConditionalMemory for cross-run persistence.

    Returns:
        (metric_fn, cache_obj) where cache_obj can be passed to get_cache_stats().
    """
    oracle_fn = _resolve_oracle_fn(oracle_classifier)
    if oracle_id is None:
        oracle_id = str(getattr(oracle_classifier, "name", "") or "").strip() or getattr(oracle_classifier, "__name__", None)
        if not oracle_id:
            oracle_id = type(oracle_classifier).__name__
    cache = (
        OraclePredictionCache(
            max_entries=cache_size,
            memory=memory,
            oracle_id=str(oracle_id),
            scale=scale,
        )
        if cache_size > 0
        else None
    )

    def cached_oracle_fn(text: str):
        if cache is None:
            return oracle_fn(text)
        cached = cache.get(text)
        if cached is not None:
            return cached
        result = oracle_fn(text)
        cache.set(text, result)
        return result

    metric_fn = create_oracle_metric(
        oracle_fn=cached_oracle_fn,
        scale=scale,
        input_field=summary_field,
        gold_field=ground_truth_field,
        with_feedback=with_feedback,
    )
    return metric_fn, cache


def get_cache_stats(cache: Optional[OraclePredictionCache]) -> Dict[str, Any]:
    """Return cache statistics for logging."""
    if cache is None:
        return {
            "cache_size": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "max_entries": 0,
        }
    if hasattr(cache, "stats"):
        return cache.stats()
    return {
        "cache_size": len(cache),
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0,
        "max_entries": len(cache),
    }


def combine_metrics(
    metrics: List[Tuple[Callable, float]],
    normalize_weights: bool = True,
) -> Callable:
    """
    Combine multiple metrics with weights.

    Creates a weighted combination of multiple metric functions.

    Args:
        metrics: List of (metric_fn, weight) tuples
        normalize_weights: If True, normalize weights to sum to 1.0

    Returns:
        Combined metric function

    Example:
        combined = combine_metrics([
            (oracle_metric, 0.6),
            (exact_match_metric, 0.3),
            (custom_metric, 0.1),
        ])

        score = combined(gold, pred)  # Weighted average
    """
    if not metrics:
        raise ValueError("At least one metric required")

    # Normalize weights if requested
    if normalize_weights:
        total_weight = sum(w for _, w in metrics)
        if total_weight > 0:
            metrics = [(m, w / total_weight) for m, w in metrics]

    def combined_metric(gold, pred, trace=None) -> float:
        total_score = 0.0

        for metric_fn, weight in metrics:
            try:
                score = metric_fn(gold, pred, trace)
                # Convert bool to float
                if isinstance(score, bool):
                    score = 1.0 if score else 0.0
                # Extract score from dict if needed
                elif isinstance(score, dict):
                    score = score.get('score', 0.0)
                total_score += weight * float(score)
            except Exception:
                # Skip failed metrics
                continue

        return total_score

    return combined_metric


def combine_feedback(
    metrics: List[Tuple[Callable, float, str]],
    normalize_weights: bool = True,
) -> Callable:
    """
    Combine multiple metrics with weights and feedback aggregation.

    Like combine_metrics but returns {'score': float, 'feedback': str}
    for GEPA compatibility.

    Args:
        metrics: List of (metric_fn, weight, name) tuples
        normalize_weights: If True, normalize weights to sum to 1.0

    Returns:
        Combined metric function with feedback

    Example:
        combined = combine_metrics_with_feedback([
            (oracle_metric, 0.6, "oracle"),
            (exact_match, 0.3, "exact"),
            (custom_metric, 0.1, "custom"),
        ])
    """
    if not metrics:
        raise ValueError("At least one metric required")

    # Normalize weights if requested
    if normalize_weights:
        total_weight = sum(w for _, w, _ in metrics)
        if total_weight > 0:
            metrics = [(m, w / total_weight, n) for m, w, n in metrics]

    def combined_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        total_score = 0.0
        feedback_parts = []
        component_scores = {}

        for metric_fn, weight, name in metrics:
            try:
                result = metric_fn(gold, pred, trace)

                # Extract score and feedback
                if isinstance(result, dict):
                    score = result.get('score', 0.0)
                    feedback = result.get('feedback', '')
                elif isinstance(result, bool):
                    score = 1.0 if result else 0.0
                    feedback = ''
                else:
                    score = float(result)
                    feedback = ''

                total_score += weight * score
                component_scores[name] = score

                if feedback:
                    feedback_parts.append(f"{name}: {feedback}")

            except Exception as e:
                component_scores[name] = 0.0
                feedback_parts.append(f"{name}: error ({str(e)})")

        return {
            'score': total_score,
            'feedback': ' | '.join(feedback_parts) if feedback_parts else 'OK',
            'components': component_scores,
        }

    return combined_metric


# Note: EvaluationResult dataclass has been removed.
# Use continuous score evaluation with oracle_as_metric() or metric() instead.


# =============================================================================
# General Oracle Score Prediction (Scale-Agnostic)
# =============================================================================

def oracle_score_prediction(
    oracle_fn: Callable,
    scale: "ScaleDefinition",
    ground_truth_field: str = 'reference_score',
    summary_field: str = 'summary',
    with_feedback: bool = True,
) -> Callable:
    """
    Create a DSPy metric that compares oracle predictions to ground truth.

    This is a general-purpose metric that works with any bounded scale.
    It uses the scale's `values_to_score()` method to convert prediction
    errors to normalized 0-1 scores.

    Args:
        oracle_fn: Oracle with predict_score(text), predict(text), or any callable returning a score.
        scale: ScaleDefinition defining the value range for normalization.
               Uses `scale.values_to_score(predicted, ground_truth)` for scoring.
        ground_truth_field: Field name for ground truth on gold example
        summary_field: Field name for summary text on prediction
        with_feedback: Return dict with feedback for GEPA compatibility

    Returns:
        DSPy-compatible metric function (5-arg signature)

    Example:
        from treepo._research.tasks.base import ScaleDefinition

        # Any bounded scale works
        my_scale = ScaleDefinition("sentiment", -1.0, 1.0)
        metric = oracle_score_prediction(
            oracle_fn=sentiment_oracle,
            scale=my_scale,
        )
    """
    # Lazy import to avoid circular dependency
    from treepo._research.tasks.base import ScaleDefinition

    def metric_fn(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Get ground truth
        ground_truth = getattr(gold, ground_truth_field, None)
        if ground_truth is None:
            ground_truth = getattr(gold, 'label', 0.0)
        try:
            ground_truth = float(ground_truth)
        except (ValueError, TypeError):
            ground_truth = scale.neutral_value or ((scale.min_value + scale.max_value) / 2)

        # Get summary from prediction
        if isinstance(pred, str):
            summary = pred
        else:
            summary = getattr(pred, summary_field, None)
            if summary is None:
                summary = getattr(pred, 'merged_summary', None)
            if summary is None:
                summary = str(pred)

        # Call oracle - support multiple interfaces
        try:
            predict_fn = _resolve_single_text_predictor(oracle_fn)
            predicted_value, confidence, reasoning = _coerce_prediction_result(
                predict_fn(summary)
            )
        except Exception as e:
            if with_feedback:
                return {'score': 0.0, 'feedback': f"Oracle error: {str(e)}"}
            return 0.0

        # Compute score using scale's built-in method
        score = scale.values_to_score(predicted_value, ground_truth)

        if with_feedback:
            error = abs(predicted_value - ground_truth)
            feedback = f"Predicted {predicted_value:.1f}, expected {ground_truth:.1f} (error: {error:.1f})"
            if reasoning:
                feedback += f" - {reasoning}"
            return {'score': score, 'feedback': feedback}

        return score

    return metric_fn


def pairwise_consistency_metric(
    oracle_fn: Callable,
    with_feedback: bool = True,
) -> Callable:
    """
    Create a metric measuring pairwise preference consistency.

    This metric evaluates whether predictions are consistent with oracle
    pairwise comparisons. Useful for GenRM-style preference learning.

    Args:
        oracle_fn: Oracle that can score text. Used to compare two summaries.
        with_feedback: Return dict with feedback for GEPA compatibility

    Returns:
        DSPy-compatible metric function (5-arg signature)

    Example:
        metric = pairwise_consistency_metric(oracle)
        # gold has 'summary_a', 'summary_b', 'preference' (1 or -1)
        score = metric(gold, pred)
    """
    def metric_fn(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # Get the two summaries to compare
        summary_a = getattr(gold, 'summary_a', None)
        summary_b = getattr(gold, 'summary_b', None)
        preference = getattr(gold, 'preference', None)  # 1 if A better, -1 if B better, 0 if equal

        if summary_a is None or summary_b is None:
            if with_feedback:
                return {'score': 0.0, 'feedback': 'Missing summary_a or summary_b'}
            return 0.0

        try:
            predict_fn = _resolve_single_text_predictor(oracle_fn)
            score_a, _, _ = _coerce_prediction_result(predict_fn(summary_a))
            score_b, _, _ = _coerce_prediction_result(predict_fn(summary_b))

            # Determine oracle's preference
            oracle_preference = 1 if score_a > score_b else (-1 if score_b > score_a else 0)

            # Compare with expected preference
            if preference is None:
                # No ground truth preference - return neutral
                score = 0.5
            elif preference == 0 or oracle_preference == 0:
                # Either is neutral - partial credit
                score = 0.75
            elif preference == oracle_preference:
                # Preferences match
                score = 1.0
            else:
                # Preferences conflict
                score = 0.0

            if with_feedback:
                feedback = f"Oracle scores: A={score_a:.1f}, B={score_b:.1f}"
                return {'score': score, 'feedback': feedback}
            return score

        except Exception as e:
            if with_feedback:
                return {'score': 0.0, 'feedback': f"Comparison error: {str(e)}"}
            return 0.0

    return metric_fn


# =============================================================================
# Domain-Specific Metrics
# =============================================================================
# Task-specific metrics live alongside their task modules.
# Import them directly from the task package (e.g., src.tasks.manifesto.metrics).

# =============================================================================
# MetricBuilder (Fluent API)
# =============================================================================

class MetricBuilder:
    """
    Fluent builder for composing DSPy metrics.

    Provides a clean, chainable interface for creating metrics with
    optional caching, feedback, thresholds, and multi-metric composition.

    Example:
        # Simple oracle metric
        metric = (MetricBuilder()
            .with_oracle(my_oracle.predict)
            .with_scale(scale)
            .build())

        # With caching and feedback for GEPA
        metric = (MetricBuilder()
            .with_oracle(oracle_fn)
            .with_scale(scale)
            .with_caching(max_entries=5000)
            .with_feedback()
            .build())

        # Composed metrics
        metric = (MetricBuilder()
            .add_component("oracle", oracle_metric, weight=0.7)
            .add_component("quality", quality_metric, weight=0.3)
            .build())

    Configuration via constructor:
        builder = MetricBuilder(
            ground_truth_field="reference_score",
            summary_field="summary",
        )
    """

    def __init__(
        self,
        ground_truth_field: str = "reference_score",
        summary_field: str = "summary",
    ):
        """
        Initialize the metric builder.

        Args:
            ground_truth_field: Default field name for ground truth
            summary_field: Default field name for summary text
        """
        self._ground_truth_field = ground_truth_field
        self._summary_field = summary_field
        self._oracle_fn: Optional[Callable] = None
        self._scale: Optional["BoundedScale"] = None
        self._cache_size: Optional[int] = None
        self._with_feedback: bool = False
        self._thresholds: Dict[str, float] = {}
        self._components: List[Tuple[str, Callable, float]] = []
        self._cache: Optional[OraclePredictionCache] = None
        self._memory: Optional["ConditionalMemory"] = None

    def with_oracle(self, oracle_fn: Callable) -> "MetricBuilder":
        """
        Set the oracle function for scoring.

        Args:
            oracle_fn: Function (text) -> float or (text) -> (float, float, str)

        Returns:
            Self for chaining
        """
        self._oracle_fn = oracle_fn
        return self

    def with_scale(self, scale: "BoundedScale") -> "MetricBuilder":
        """
        Set the scale for normalization.

        Args:
            scale: BoundedScale or ScaleDefinition defining value range

        Returns:
            Self for chaining
        """
        self._scale = scale
        return self

    def with_caching(self, max_entries: int = 10000) -> "MetricBuilder":
        """
        Enable oracle prediction caching.

        Args:
            max_entries: Maximum cache size (0 to disable)

        Returns:
            Self for chaining
        """
        self._cache_size = max_entries
        return self

    def with_memory(self, memory: "ConditionalMemory") -> "MetricBuilder":
        """Attach a ConditionalMemory instance for cross-run persistence.

        Returns:
            Self for chaining
        """
        self._memory = memory
        return self

    def with_feedback(self) -> "MetricBuilder":
        """
        Return dict with feedback for GEPA compatibility.

        When enabled, metric returns {'score': float, 'feedback': str}
        instead of just float.

        Returns:
            Self for chaining
        """
        self._with_feedback = True
        return self

    def with_thresholds(
        self,
        good: float = 0.05,
        acceptable: float = 0.10,
        bad: float = 0.20,
    ) -> "MetricBuilder":
        """
        Set error thresholds for feedback messages.

        Args:
            good: Normalized error threshold for "good" (default 5%)
            acceptable: Normalized error threshold for "acceptable" (default 10%)
            bad: Normalized error threshold for "bad" (default 20%)

        Returns:
            Self for chaining
        """
        self._thresholds = {
            "good": good,
            "acceptable": acceptable,
            "bad": bad,
        }
        return self

    def add_component(
        self,
        name: str,
        metric_fn: Callable,
        weight: float = 1.0,
    ) -> "MetricBuilder":
        """
        Add a metric component for composition.

        Args:
            name: Component name (for feedback)
            metric_fn: Metric function
            weight: Weight for this component

        Returns:
            Self for chaining
        """
        self._components.append((name, metric_fn, weight))
        return self

    def build(self) -> Tuple[Callable, Optional[OraclePredictionCache]]:
        """
        Build the final metric function.

        Returns:
            Tuple of (metric_fn, cache) where cache is None if caching disabled

        Raises:
            ValueError: If required configuration is missing
        """
        # If components are set, build composed metric
        if self._components:
            return self._build_composed(), None

        # Otherwise build oracle metric
        return self._build_oracle()

    def build_metric(self) -> Callable:
        """
        Build and return just the metric function.

        Convenience method when you don't need the cache reference.

        Returns:
            Metric function
        """
        metric_fn, _ = self.build()
        return metric_fn

    def _build_oracle(self) -> Tuple[Callable, Optional[OraclePredictionCache]]:
        """Build oracle-based metric."""
        if self._oracle_fn is None:
            raise ValueError("Oracle function required. Call with_oracle() first.")
        if self._scale is None:
            raise ValueError("Scale required. Call with_scale() first.")

        # Use cached version if caching enabled
        if self._cache_size is not None and self._cache_size > 0:
            metric_fn, cache = create_cached_oracle_metric(
                oracle_classifier=self._oracle_fn,
                scale=self._scale,
                ground_truth_field=self._ground_truth_field,
                summary_field=self._summary_field,
                cache_size=self._cache_size,
                with_feedback=self._with_feedback,
                memory=self._memory,
            )
            self._cache = cache
            return metric_fn, cache

        # Non-cached version
        metric_fn = create_oracle_metric(
            oracle_fn=self._oracle_fn,
            scale=self._scale,
            input_field=self._summary_field,
            gold_field=self._ground_truth_field,
            with_feedback=self._with_feedback,
        )
        return metric_fn, None

    def _build_composed(self) -> Callable:
        """Build composed metric from components."""
        if self._with_feedback:
            return combine_feedback(
                [(fn, w, name) for name, fn, w in self._components],
                normalize_weights=True,
            )
        else:
            return combine_metrics(
                [(fn, w) for _, fn, w in self._components],
                normalize_weights=True,
            )

    def get_cache(self) -> Optional[OraclePredictionCache]:
        """Get the cache after building (if caching was enabled)."""
        return self._cache


# =============================================================================
# Legacy Aliases
# =============================================================================

# Note: create_classification_metric, evaluate_classifier, and create_violation_metric
# have been removed. Use oracle_as_metric() or metric() for continuous score evaluation.
