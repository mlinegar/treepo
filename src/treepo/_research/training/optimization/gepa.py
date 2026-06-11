"""
GEPA (Gradient-free Efficient Prompt Adaptation) optimizer wrapper.

GEPA is the default optimizer for complex optimization tasks. It uses
reflection and merge capabilities for sophisticated prompt optimization.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import dspy

from treepo._research.training.gepa_defaults import GEPA_STRONG_DEFAULT_KWARGS

from treepo._research.training.supervision.timing import (
    ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC,
    ACTIVATION_IMMEDIATE,
    CONSUMER_GEPA_OPTIMIZER,
    supervision_timing_contract,
)

from .base import AbstractOptimizer, OptimizationResult
from .registry import register_optimizer

logger = logging.getLogger(__name__)

try:
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
except Exception:  # pragma: no cover
    ScoreWithFeedback = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Strong GEPA defaults — single source of truth.
#
# These are the kwargs the paper relies on for every GEPA optimization run.
# Both ``OptimizationConfig.gepa_kwargs`` and
# ``DSPyFamilyConfig.gepa_kwargs`` use this dict as their field-default
# factory, so the two configs cannot drift apart. The methods drift
# test pins ``canonical_defaults.GEPA_STRONG_DEFAULTS`` against this dict.
#
# Kwargs that change per-run (metric, reflection_lm, auto/max_metric_calls,
# num_threads) are layered on top imperatively by the optimizer builder.
# ---------------------------------------------------------------------------


@register_optimizer("gepa")
class GEPAOptimizer(AbstractOptimizer):
    """
    GEPA optimizer wrapper.

    GEPA (Gradient-free Efficient Prompt Adaptation) is a sophisticated
    optimizer that uses reflection and instruction merging to optimize
    DSPy modules.

    Key features:
    - Reflection-based optimization with teacher LM
    - Instruction merging for combining good prompts
    - Budget control via auto modes or max_metric_calls
    - Track statistics for optimization analysis
    """

    @property
    def name(self) -> str:
        return "gepa"

    @property
    def supports_parallel(self) -> bool:
        # GEPA uses num_threads internally for parallel evaluation
        return True

    def compile(
        self,
        student: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
        metric: Optional[Callable] = None,
        teacher: Optional[dspy.Module] = None,
        **kwargs,
    ) -> dspy.Module:
        """
        Compile using GEPA optimizer.

        Args:
            student: Module to optimize
            trainset: Training examples
            valset: Validation examples (GEPA uses these for validation)
            metric: DSPy metric function
            teacher: Not used by GEPA (uses reflection LM instead)
            **kwargs: Additional arguments (e.g., override config)

        Returns:
            Optimized module
        """
        valset = valset or trainset
        self._log_compile_start(len(trainset), len(valset))
        supervision_timing = self._supervision_timing_contract(
            metric_attached=bool(metric is not None),
            trainset_size=len(trainset),
            valset_size=len(valset),
        )
        self._update_compile_audit(supervision_timing=supervision_timing)

        # Wrap metric for GEPA's 5-argument signature
        wrapped_metric = self._wrap_metric_gepa(metric) if metric else None

        # Build GEPA kwargs
        gepa_kwargs = self._build_gepa_kwargs(wrapped_metric)

        # Override with any kwargs passed in
        gepa_kwargs.update(kwargs.get('gepa_kwargs', {}))

        try:
            optimizer = dspy.GEPA(**gepa_kwargs)
            compiled = optimizer.compile(
                student=student,
                trainset=trainset,
                valset=valset,
            )
            self._update_compile_audit(
                compile_status="completed",
                optimizer_used=self.name,
                fallback_reason="none",
                supervision_timing=supervision_timing,
                gepa_kwargs={
                    k: str(v)
                    if not isinstance(v, (str, int, float, bool, type(None)))
                    else v
                    for k, v in gepa_kwargs.items()
                },
            )
            return compiled

        except Exception as e:
            self._update_compile_audit(
                compile_status="failed",
                optimizer_used=self.name,
                fallback_reason="none",
                supervision_timing=supervision_timing,
                exception_summary=f"{type(e).__name__}: {e}",
            )
            logger.error(f"GEPA compilation failed: {e}")
            raise

    def estimate_budget(self, dataset_size: int) -> int:
        """
        Estimate metric calls based on GEPA budget setting.

        Args:
            dataset_size: Number of training examples

        Returns:
            Estimated metric call count
        """
        if self.config is None:
            return 1000  # Default estimate

        # Check for explicit max_metric_calls
        max_calls = getattr(self.config, 'max_metric_calls', None)
        if max_calls:
            return max_calls

        # Estimate based on budget mode
        budget = getattr(self.config, 'gepa_auto', 'medium')
        budget_estimates = {
            'light': 300,
            'medium': 1000,
            'heavy': 3000,
            'superheavy': 5000,
        }
        return budget_estimates.get(budget, 1000)

    def _wrap_metric_gepa(self, metric: Callable) -> Callable:
        """
        Wrap metric for GEPA's 5-argument signature.

        GEPA expects: (gold, pred, trace, pred_name, pred_trace) -> float

        Args:
            metric: Original metric function

        Returns:
            GEPA-compatible metric function
        """
        def wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
            result = metric(gold, pred, trace, pred_name, pred_trace)
            if isinstance(result, dict):
                score = result.get("score", 0.0)
                feedback = result.get("feedback", None)
                if feedback is not None and ScoreWithFeedback is not None:
                    try:
                        score_value = float(score)
                    except (TypeError, ValueError):
                        score_value = 0.0
                    return ScoreWithFeedback(score=score_value, feedback=str(feedback))
                try:
                    return float(score)
                except (TypeError, ValueError):
                    return 0.0
            return result

        wrapped.supervision_timing = self._supervision_timing_contract(  # type: ignore[attr-defined]
            metric_attached=True,
        )
        return wrapped

    def _supervision_timing_contract(
        self,
        *,
        metric_attached: bool,
        trainset_size: Optional[int] = None,
        valset_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        return supervision_timing_contract(
            acquisition_policy=ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC,
            activation_barrier=ACTIVATION_IMMEDIATE,
            consumer=CONSUMER_GEPA_OPTIMIZER,
            producer="metric_callback",
            delivery_mode="dspy_gepa_metric",
            blocking=True,
            notes=(
                "GEPA must receive metric scores and optional feedback during optimizer search.",
                "Feedback is active immediately for candidate ranking and reflection inside compile().",
            ),
            metadata={
                "metric_attached": bool(metric_attached),
                "trainset_size": None if trainset_size is None else int(trainset_size),
                "valset_size": None if valset_size is None else int(valset_size),
            },
        )

    def _build_gepa_kwargs(self, metric: Optional[Callable]) -> Dict[str, Any]:
        """Build kwargs for GEPA constructor.

        Seeds from ``GEPA_STRONG_DEFAULT_KWARGS`` (the single source of truth
        for paper-canonical GEPA kwargs) and layers per-call / per-config
        overrides on top. The seed keeps strong defaults consistent across
        ``OptimizationConfig`` and ``DSPyFamilyConfig`` users.
        """
        kwargs: Dict[str, Any] = dict(GEPA_STRONG_DEFAULT_KWARGS)
        kwargs['metric'] = metric
        kwargs['reflection_lm'] = dspy.settings.lm  # bound at call time

        if self.config is None:
            kwargs['auto'] = 'heavy'  # paper canonical (was 'medium')
            kwargs['num_threads'] = GEPA_STRONG_DEFAULT_KWARGS.get('num_threads', 64) if 'num_threads' in GEPA_STRONG_DEFAULT_KWARGS else 64
            return kwargs

        # Per-field overrides from OptimizationConfig (config wins over strong defaults).
        kwargs['use_merge'] = getattr(self.config, 'enable_merge', kwargs['use_merge'])
        kwargs['max_merge_invocations'] = getattr(self.config, 'max_merge_invocations', kwargs['max_merge_invocations'])
        kwargs['track_stats'] = getattr(self.config, 'track_stats', kwargs['track_stats'])
        kwargs['add_format_failure_as_feedback'] = bool(
            getattr(self.config, 'gepa_add_format_failure_as_feedback', kwargs['add_format_failure_as_feedback'])
        )

        requested_threads = max(1, int(getattr(self.config, 'num_threads', 64)))
        raw_thread_cap = getattr(self.config, 'gepa_max_threads', None)
        if raw_thread_cap is None:
            kwargs['num_threads'] = requested_threads
        else:
            thread_cap = max(1, int(raw_thread_cap))
            kwargs['num_threads'] = min(requested_threads, thread_cap)
            if kwargs['num_threads'] < requested_threads:
                logger.info(
                    "GEPA: Capping num_threads from %d to %d for stability",
                    requested_threads,
                    kwargs['num_threads'],
                )

        # Log directory
        log_dir = getattr(self.config, 'log_dir', None)
        if log_dir:
            kwargs['log_dir'] = str(log_dir)

        # Budget control
        max_metric_calls = getattr(self.config, 'max_metric_calls', None)
        gepa_auto = getattr(self.config, 'gepa_auto', 'heavy')

        if max_metric_calls:
            logger.info(f"GEPA: Using explicit max_metric_calls={max_metric_calls}")
            kwargs['max_metric_calls'] = max_metric_calls
        elif gepa_auto == 'superheavy':
            # 'superheavy' uses max_metric_calls instead of auto
            logger.info("GEPA: Using superheavy budget (max_metric_calls=5000)")
            kwargs['max_metric_calls'] = 5000
        else:
            logger.info(f"GEPA: Using auto='{gepa_auto}' budget")
            kwargs['auto'] = gepa_auto

        return kwargs
