"""
Judge Optimization Module - Tournament of Tournaments.

This module provides functionality for optimizing GenRM judge prompts using DSPy.
It implements the "tournament of tournaments" concept where we optimize the judge
itself to improve comparison accuracy.

The optimized judge can then be used in TournamentStrategy for better
preference collection and summary selection.

Usage:
    from treepo._research.training.judge_optimization import (
        JudgeOptimizer,
        create_judge_trainset,
        derive_ground_truth_preference,
    )

    # Create optimizer
    optimizer = JudgeOptimizer(budget='medium', num_threads=4)

    # Optimize judge from preference pairs
    optimized_judge = optimizer.optimize(preference_pairs)

    # Use in TournamentStrategy
    strategy = TournamentStrategy(base=..., judge=optimized_judge)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Callable, Sequence, Union

import dspy

from treepo._research.core.prompting import clean_summary_text
from treepo._research.training.judges.genrm_dspy import GenRMComparisonModule
from treepo._research.training.supervision import (
    BinaryComparison,
    SupervisionDataset,
    coerce_supervision_dataset,
)
from treepo._research.training.supervision.adapters import (
    prepare_binary_optimizer_dataset,
)
from treepo._research.training.supervision.judge_capabilities import invoke_pairwise_judgment_sync
from treepo._research.training.supervision.timing import (
    ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC,
    ACTIVATION_IMMEDIATE,
    CONSUMER_JUDGE_GEPA_OPTIMIZER,
    supervision_timing_contract,
)

if TYPE_CHECKING:
    from treepo._research.training.supervision import ComparativeDataset, PreferenceDataset
    from treepo._research.training.supervision.comparative_types import ComparativeJudgmentRecord

PreferenceLabeler = Callable[[BinaryComparison, float], Optional[str]]
OptimizerSupervision = Union[
    SupervisionDataset,
    "PreferenceDataset",
    "ComparativeDataset",
    Sequence[BinaryComparison],
    Sequence["ComparativeJudgmentRecord"],
]

logger = logging.getLogger(__name__)

try:
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
except Exception:  # pragma: no cover
    ScoreWithFeedback = None  # type: ignore[assignment]

def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + " ... (truncated)"


@dataclass
class SkippedReasons:
    """Track skip reasons for training data preparation.

    Each counter is mutually exclusive - a pair is only counted once
    in the first applicable category.
    """
    missing_oracle_no_labeler: int = 0  # No oracle_error and no preference_labeler
    labeler_returned_none: int = 0       # preference_labeler was provided but returned None
    oracle_tie_margin_none: int = 0      # Oracle-based derivation returned None (shouldn't happen)

    @property
    def total(self) -> int:
        return (
            self.missing_oracle_no_labeler +
            self.labeler_returned_none +
            self.oracle_tie_margin_none
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            'missing_oracle_no_labeler': self.missing_oracle_no_labeler,
            'labeler_returned_none': self.labeler_returned_none,
            'oracle_tie_margin_none': self.oracle_tie_margin_none,
            'total': self.total,
        }

# =============================================================================
# Constants
# =============================================================================

# Ranking scale constants for GenRM judge output
# The judge outputs scores on a 1-6 Likert scale where:
# - 1 = Strongly prefer A
# - 6 = Strongly prefer B
# - 3-4 = Neutral/uncertain
RANKING_SCALE_MIN = 1.0
RANKING_SCALE_MAX = 6.0
RANKING_SCALE_CENTER = 3.5  # Midpoint representing maximum uncertainty
RANKING_SCALE_HALF_RANGE = 2.5  # Distance from center to extremes: (6-1)/2

# NOTE: We no longer truncate original text for training examples.
# Truncation corrupts training signal - models should see full context.
# DSPy and GenRM can handle longer contexts natively.


# =============================================================================
# Training Data Preparation
# =============================================================================

def derive_ground_truth_preference(
    pair: BinaryComparison,
    tie_margin: float = 0.5,
    preference_labeler: Optional[PreferenceLabeler] = None,
) -> Optional[str]:
    """
    Derive ground truth preference from oracle scores.

    By default:
    - If oracle_error_a/b are present, lower error is better.
    - Otherwise, requires a preference_labeler to provide ground truth.

    Args:
        pair: BinaryComparison with optional score estimates
        tie_margin: Score difference below this is considered a tie
        preference_labeler: Optional override for custom metrics

    Returns:
        'A', 'B', 'tie', or None if no ground truth available
    """
    if preference_labeler is not None:
        return preference_labeler(pair, tie_margin)

    if pair.oracle_error_a is not None and pair.oracle_error_b is not None:
        diff = pair.oracle_error_a - pair.oracle_error_b
        if abs(diff) < tie_margin:
            return 'tie'
        return 'A' if diff < 0 else 'B'  # Lower error is better

    return None


def make_preference_labeler(
    metric_name: str,
    prefer_lower: bool = False,
) -> PreferenceLabeler:
    """
    Create a preference labeler from a metric stored on a binary supervision row.

    Args:
        metric_name: Base metric name (e.g., "oracle_error", "score_estimate")
        prefer_lower: If True, lower metric value is better
    """
    metric_a_field = f"{metric_name}_a"
    metric_b_field = f"{metric_name}_b"

    def _label(pair: BinaryComparison, tie_margin: float = 0.5) -> Optional[str]:
        value_a = getattr(pair, metric_a_field, None)
        value_b = getattr(pair, metric_b_field, None)
        if value_a is None or value_b is None:
            return None

        diff = value_a - value_b
        if abs(diff) < tie_margin:
            return 'tie'

        if prefer_lower:
            return 'A' if diff < 0 else 'B'
        return 'A' if diff > 0 else 'B'

    return _label


def create_judge_trainset(
    pairs: List[BinaryComparison],
    tie_margin: float = 0.5,
    use_oracle_as_ground_truth: bool = True,
    preference_labeler: Optional[PreferenceLabeler] = None,
) -> Tuple[List[dspy.Example], SkippedReasons]:
    """
    Create DSPy training examples for judge optimization.

    Args:
        pairs: List of binary supervision rows
        tie_margin: Score difference below this is considered a tie
        use_oracle_as_ground_truth: If True, derive ground truth from oracle scores.
                                   If False, use the existing 'preferred' field.
        preference_labeler: Optional override for custom preference labeling

    Returns:
        Tuple of (examples, skipped_reasons) for full visibility into data quality
    """
    examples = []
    skipped = SkippedReasons()

    for pair in pairs:
        if use_oracle_as_ground_truth:
            # Check if we can derive ground truth
            has_oracle = (
                pair.oracle_error_a is not None and pair.oracle_error_b is not None
            )

            if preference_labeler is None and not has_oracle:
                # No way to get ground truth
                skipped.missing_oracle_no_labeler += 1
                continue

            ground_truth = derive_ground_truth_preference(
                pair,
                tie_margin=tie_margin,
                preference_labeler=preference_labeler,
            )

            if ground_truth is None:
                # Labeler was provided but returned None
                if preference_labeler is not None:
                    skipped.labeler_returned_none += 1
                else:
                    # This shouldn't happen if has_oracle is True
                    skipped.oracle_tie_margin_none += 1
                continue
        else:
            ground_truth = pair.preferred

        judge_reasoning_raw = str(getattr(pair, "reasoning", "") or "").strip()
        judge_reasoning = clean_summary_text(judge_reasoning_raw)
        if use_oracle_as_ground_truth:
            error_a = getattr(pair, "oracle_error_a", None)
            error_b = getattr(pair, "oracle_error_b", None)
            oracle_reason: Optional[str] = None
            if error_a is not None and error_b is not None:
                try:
                    error_a_f = float(error_a)
                    error_b_f = float(error_b)
                except (TypeError, ValueError):
                    error_a_f = None
                    error_b_f = None
                if error_a_f is not None and error_b_f is not None:
                    diff = error_a_f - error_b_f
                    if ground_truth == "tie":
                        oracle_reason = (
                            "Oracle label: tie "
                            f"(oracle_error_a={error_a_f:.4f}, oracle_error_b={error_b_f:.4f}, "
                            f"diff={diff:+.4f}, tie_margin={tie_margin:.4f})."
                        )
                    else:
                        oracle_reason = (
                            f"Oracle label: {ground_truth} "
                            f"(oracle_error_a={error_a_f:.4f}, oracle_error_b={error_b_f:.4f}, "
                            f"diff={diff:+.4f}, tie_margin={tie_margin:.4f})."
                        )

            if oracle_reason is None:
                oracle_reason = f"Gold label: {ground_truth!r} (no oracle errors available)."

        else:
            oracle_reason = None

        parts: List[str] = []
        if oracle_reason:
            parts.append(oracle_reason)
        if judge_reasoning:
            judge_pref = getattr(pair, "preferred", None)
            judge_conf = getattr(pair, "confidence", None)
            alignment = None
            if judge_pref in {"A", "B", "tie"} and ground_truth in {"A", "B", "tie"}:
                alignment = "aligned" if judge_pref == ground_truth else "disagrees"
            header = "Judge rationale:"
            if judge_pref in {"A", "B", "tie"}:
                conf_part = ""
                if judge_conf is not None:
                    try:
                        conf_part = f", confidence={float(judge_conf):.3f}"
                    except (TypeError, ValueError):
                        conf_part = ""
                if alignment:
                    header = f"Judge said {judge_pref}{conf_part} ({alignment}):"
                else:
                    header = f"Judge said {judge_pref}{conf_part}:"
            parts.append(f"{header} {judge_reasoning}")

        ground_truth_reasoning = _truncate_text("\n".join(parts).strip(), max_chars=2400)

        # Use full original_text - no truncation
        example = dspy.Example(
            context=pair.rubric,
            original_text=pair.original_text,
            summary_a=pair.summary_a,
            summary_b=pair.summary_b,
            law_type=pair.law_type,
            ground_truth_preference=ground_truth,
            ground_truth_reasoning=ground_truth_reasoning,
        ).with_inputs('context', 'original_text', 'summary_a', 'summary_b', 'law_type')

        examples.append(example)

    logger.info(f"Created {len(examples)} training examples for judge optimization")

    if skipped.total > 0:
        logger.warning(
            "Skipped %d pairs: %s",
            skipped.total,
            skipped.to_dict(),
        )

    return examples, skipped


# =============================================================================
# Metrics
# =============================================================================

def judge_accuracy_metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """
    Metric for judge accuracy: does the judge predict the correct preference?

    Returns 1.0 for correct, 0.0 for incorrect, 0.5 for tie mismatches.
    """
    try:
        predicted = getattr(prediction, "preference", None)
        ground_truth = example.ground_truth_preference
    except Exception:
        return 0.0

    if predicted == ground_truth:
        return 1.0

    score = 0.5 if (predicted == "tie" or ground_truth == "tie") else 0.0

    if ScoreWithFeedback is None:
        return float(score)

    gold_reasoning = str(getattr(example, "ground_truth_reasoning", "") or "").strip()
    pred_reasoning = str(getattr(prediction, "reasoning", "") or "").strip()
    feedback = f"Incorrect preference: predicted={predicted!r}, expected={ground_truth!r}."
    if pred_reasoning:
        feedback = f"{feedback}\nYour rationale: {_truncate_text(pred_reasoning, max_chars=800)}"
    if gold_reasoning:
        feedback = f"{feedback}\nGold rationale: {_truncate_text(gold_reasoning, max_chars=1600)}"
    return ScoreWithFeedback(score=float(score), feedback=feedback)


def judge_accuracy_with_confidence(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """
    Metric that weights accuracy by confidence.

    Rewards confident correct predictions, penalizes confident wrong predictions.
    """
    try:
        predicted = getattr(prediction, "preference", None)
        ground_truth = example.ground_truth_preference

        # Get confidence from ranking_score if available
        try:
            ranking_score = float(prediction.ranking_score)
            # Convert ranking_score to confidence (0-1):
            # - Extremes (1 or 6) = high confidence (1.0)
            # - Center (3.5) = low confidence (0.0)
            confidence = abs(ranking_score - RANKING_SCALE_CENTER) / RANKING_SCALE_HALF_RANGE
        except (ValueError, TypeError, AttributeError):
            confidence = 0.5

        if predicted == ground_truth:
            return 0.5 + 0.5 * confidence  # 0.5 to 1.0

        score = 0.5 if (predicted == "tie" or ground_truth == "tie") else (0.5 - 0.5 * confidence)
        score = float(max(0.0, min(1.0, score)))

        if ScoreWithFeedback is None:
            return score

        gold_reasoning = str(getattr(example, "ground_truth_reasoning", "") or "").strip()
        pred_reasoning = str(getattr(prediction, "reasoning", "") or "").strip()
        feedback = (
            f"Incorrect preference: predicted={predicted!r}, expected={ground_truth!r}. "
            f"(confidence={confidence:.3f})"
        )
        if pred_reasoning:
            feedback = f"{feedback}\nYour rationale: {_truncate_text(pred_reasoning, max_chars=800)}"
        if gold_reasoning:
            feedback = f"{feedback}\nGold rationale: {_truncate_text(gold_reasoning, max_chars=1600)}"
        return ScoreWithFeedback(score=score, feedback=feedback)

    except (AttributeError, TypeError):
        return 0.0


# =============================================================================
# Judge Optimizer
# =============================================================================

@dataclass
class JudgeOptimizationConfig:
    """Configuration for judge optimization."""
    budget: str = 'light'  # 'light', 'medium', 'heavy', 'superheavy'
    num_threads: int = 4
    tie_margin: float = 0.05  # In metric units (normalized errors => 0-1)
    test_split: float = 0.2
    use_confidence_metric: bool = False
    checkpoint_dir: Optional[Path] = None
    preference_labeler: Optional[PreferenceLabeler] = None
    use_propensity_weighting: bool = True
    propensity_weight_clip: Optional[float] = None
    propensity_random_seed: int = 42


class JudgeOptimizer:
    """
    Optimizer for GenRM judge prompts.

    Implements tournament of tournaments by training GenRMComparisonModule
    with optimizable DSPy prompts.
    """

    def __init__(
        self,
        config: Optional[JudgeOptimizationConfig] = None,
        budget: str = 'light',
        num_threads: int = 4,
    ):
        """
        Initialize judge optimizer.

        Args:
            config: Full configuration (if provided, overrides other args)
            budget: GEPA budget ('light', 'medium', 'heavy', 'superheavy')
            num_threads: Number of parallel evaluation threads
        """
        if config is not None:
            self.config = config
        else:
            self.config = JudgeOptimizationConfig(
                budget=budget,
                num_threads=num_threads,
            )

    def optimize(
        self,
        supervision: OptimizerSupervision,
        use_oracle_as_ground_truth: bool = True,
        initial_judge: Optional[GenRMComparisonModule] = None,
    ) -> Tuple[GenRMComparisonModule, dict]:
        """
        Optimize GenRMComparisonModule using GEPA.

        Args:
            supervision: Canonical supervision records for training
            use_oracle_as_ground_truth: Derive ground truth from oracle scores
            initial_judge: Optional starting judge (used for baseline + warm start)

        Returns:
            Tuple of (optimized_judge, evaluation_results)
        """
        from treepo._research.training.supervision import compute_propensity_diagnostics

        input_supervision = coerce_supervision_dataset(supervision)
        input_dataset = input_supervision.project_binary(projection="adjacent")
        optimizer_dataset = prepare_binary_optimizer_dataset(
            input_dataset,
            projection="adjacent",
            keep_existing=True,
        )
        optimizer_pairs = list(optimizer_dataset.pairs)

        weighted_dataset = optimizer_dataset
        if self.config.use_propensity_weighting and optimizer_pairs:
            weighted_dataset = optimizer_dataset.resample_by_propensity(
                target_size=len(optimizer_pairs),
                seed=self.config.propensity_random_seed,
                max_weight=self.config.propensity_weight_clip,
            )
        weighted_pairs = list(weighted_dataset.pairs)

        propensity_input = compute_propensity_diagnostics(
            optimizer_pairs, include_ties=False, max_weight=self.config.propensity_weight_clip
        )
        propensity_weighted = compute_propensity_diagnostics(
            weighted_pairs, include_ties=False, max_weight=self.config.propensity_weight_clip
        )

        # Create training examples
        all_examples, skipped_reasons = create_judge_trainset(
            weighted_pairs,
            tie_margin=self.config.tie_margin,
            use_oracle_as_ground_truth=use_oracle_as_ground_truth,
            preference_labeler=self.config.preference_labeler,
        )

        judge_module = initial_judge
        if judge_module is None:
            judge_module = GenRMComparisonModule(use_dspy_predictor=True)
        elif not (
            getattr(judge_module, "use_dspy_predictor", False)
            or getattr(judge_module, "use_dspy_prompt", False)
        ):
            logger.warning("Initial judge does not support DSPy prompts; starting from a fresh DSPy judge")
            judge_module = GenRMComparisonModule(use_dspy_predictor=True)

        if len(all_examples) < 10:
            logger.warning(f"Only {len(all_examples)} examples, returning unoptimized judge")
            return judge_module, {
                'error': 'insufficient_data',
                'total_pairs_input': len(optimizer_pairs),
                'total_pairs_weighted': len(weighted_pairs),
                'total_comparative_input': len(input_supervision.comparative_judgments),
                'propensity_diagnostics': {
                    'input': propensity_input,
                    'weighted': propensity_weighted,
                },
                'supervision_timing': supervision_timing_contract(
                    acquisition_policy=ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC,
                    activation_barrier=ACTIVATION_IMMEDIATE,
                    consumer=CONSUMER_JUDGE_GEPA_OPTIMIZER,
                    producer="judge_accuracy_metric",
                    delivery_mode="dspy_gepa_metric",
                    blocking=True,
                    notes=(
                        "Judge GEPA optimization would consume metric feedback synchronously, but optimization was skipped for insufficient data.",
                    ),
                    metadata={
                        "examples_available": len(all_examples),
                        "budget": str(self.config.budget),
                    },
                ),
            }

        # Split train/test
        import random
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * (1 - self.config.test_split))
        trainset = all_examples[:split_idx]
        testset = all_examples[split_idx:]

        logger.info(f"Judge optimization: {len(trainset)} train, {len(testset)} test examples")

        # Select metric
        metric_fn = (
            judge_accuracy_with_confidence if self.config.use_confidence_metric
            else judge_accuracy_metric
        )
        supervision_timing = supervision_timing_contract(
            acquisition_policy=ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC,
            activation_barrier=ACTIVATION_IMMEDIATE,
            consumer=CONSUMER_JUDGE_GEPA_OPTIMIZER,
            producer="judge_accuracy_metric",
            delivery_mode="dspy_gepa_metric",
            blocking=True,
            notes=(
                "Judge GEPA optimization consumes metric feedback synchronously during compile().",
                "Feedback is active immediately for GEPA candidate ranking and reflection.",
            ),
            metadata={
                "train_size": len(trainset),
                "test_size": len(testset),
                "budget": str(self.config.budget),
                "use_confidence_metric": bool(self.config.use_confidence_metric),
            },
        )

        # Evaluate baseline
        baseline_results = self._evaluate(judge_module, testset)
        logger.info(f"Baseline accuracy: {baseline_results['accuracy']:.3f}")

        # Create optimizer
        optimizer = dspy.GEPA(
            metric=metric_fn,
            auto=self.config.budget,
            num_threads=self.config.num_threads,
            use_wandb=False,
            use_mlflow=False,
        )

        # Run optimization
        logger.info(f"Starting GEPA optimization (budget={self.config.budget})...")
        optimized_judge = optimizer.compile(
            judge_module,
            trainset=trainset,
        )

        # Evaluate optimized
        optimized_results = self._evaluate(optimized_judge, testset)
        logger.info(f"Optimized accuracy: {optimized_results['accuracy']:.3f}")
        logger.info(f"Improvement: {optimized_results['accuracy'] - baseline_results['accuracy']:+.3f}")

        results = {
            'baseline': baseline_results,
            'optimized': optimized_results,
            'improvement': optimized_results['accuracy'] - baseline_results['accuracy'],
            'train_size': len(trainset),
            'test_size': len(testset),
            'budget': self.config.budget,
            'skipped_pairs': skipped_reasons.to_dict(),
            'total_pairs_input': len(optimizer_pairs),
            'total_pairs_weighted': len(weighted_pairs),
            'total_comparative_input': len(input_supervision.comparative_judgments),
            'propensity_diagnostics': {
                'input': propensity_input,
                'weighted': propensity_weighted,
            },
            'supervision_timing': supervision_timing,
        }

        return optimized_judge, results

    def _evaluate(
        self,
        judge: GenRMComparisonModule,
        testset: List[dspy.Example],
    ) -> dict:
        """Evaluate judge accuracy on test set."""
        correct = 0
        total = 0

        for example in testset:
            try:
                result = invoke_pairwise_judgment_sync(
                    context=example.context,
                    original_text=example.original_text,
                    summary_a=example.summary_a,
                    summary_b=example.summary_b,
                    judge=judge,
                    law_type=example.law_type,
                )

                predicted = result.preferred
                ground_truth = example.ground_truth_preference

                total += 1
                if predicted == ground_truth:
                    correct += 1

            except Exception as e:
                logger.debug(f"Evaluation error: {e}")
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }

    def save(self, judge: GenRMComparisonModule, path: Path) -> None:
        """Save optimized judge to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        judge.save(str(path))
        logger.info(f"Saved optimized judge to {path}")

    def load(
        self,
        path: Path,
        use_dspy_prompt: bool = True,
        prompt_lm: Optional[dspy.LM] = None,
    ) -> GenRMComparisonModule:
        """Load optimized judge from file."""
        judge = GenRMComparisonModule(use_dspy_prompt=use_dspy_prompt, prompt_lm=prompt_lm)
        try:
            judge.load(str(path))
        except Exception as e:
            if use_dspy_prompt:
                logger.warning(f"Prompt-tuned judge load failed ({e}); retrying as DSPy predictor")
                judge = GenRMComparisonModule(use_dspy_predictor=True)
                judge.load(str(path))
            else:
                raise
        logger.info(f"Loaded optimized judge from {path}")
        return judge


# =============================================================================
# Convenience Functions
# =============================================================================

def optimize_judge_from_preferences(
    preferences: OptimizerSupervision,
    budget: str = 'light',
    num_threads: int = 4,
    output_path: Optional[Path] = None,
) -> Tuple[GenRMComparisonModule, dict]:
    """
    Convenience function to optimize judge from preference pairs.

    Args:
        preferences: List of PreferencePair
        budget: GEPA budget
        num_threads: Parallel threads
        output_path: Optional path to save optimized judge

    Returns:
        Tuple of (optimized_judge, evaluation_results)
    """
    optimizer = JudgeOptimizer(budget=budget, num_threads=num_threads)
    judge, results = optimizer.optimize(preferences)

    if output_path:
        optimizer.save(judge, output_path)

    return judge, results


def load_optimized_judge(
    path: Path,
    use_dspy_prompt: bool = True,
    prompt_lm: Optional[dspy.LM] = None,
) -> GenRMComparisonModule:
    """Load optimized judge from file."""
    optimizer = JudgeOptimizer()
    return optimizer.load(path, use_dspy_prompt=use_dspy_prompt, prompt_lm=prompt_lm)
