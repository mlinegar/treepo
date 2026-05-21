"""
Bootstrap-based optimizer wrappers.

This module provides wrappers for DSPy's bootstrap-based optimizers:
- BootstrapFewShot: Basic few-shot prompt optimization
- BootstrapFewShotWithRandomSearch: Parallelizable random search variant

Bootstrap optimizers are best for:
- Small to medium datasets
- Quick iteration and experimentation
- When parallel evaluation is desired
"""

import logging
import copy
from typing import Callable, List, Optional

import dspy

from .base import AbstractOptimizer
from .registry import register_optimizer

logger = logging.getLogger(__name__)


def _prepare_iterative_bootstrap_student(student: dspy.Module, optimizer_name: str) -> dspy.Module:
    """
    Return a student module acceptable to DSPy bootstrap optimizers.

    DSPy's bootstrap-based teleprompters assert that the input student is
    "uncompiled". In multi-iteration pipeline runs we intentionally pass the
    previously optimized module back in, so we defensively clear the compiled
    marker (on a copied module when possible) before each compile step.
    """
    prepared = student
    try:
        if hasattr(student, "deepcopy"):
            prepared = student.deepcopy()
        else:
            prepared = copy.deepcopy(student)
    except Exception:
        prepared = student

    if getattr(prepared, "_compiled", False):
        try:
            setattr(prepared, "_compiled", False)
        except Exception:
            pass
        logger.info(
            "%s: cleared pre-existing _compiled flag for iterative optimization",
            optimizer_name,
        )

    if hasattr(prepared, "predictors"):
        try:
            for predictor in list(prepared.predictors()):
                if getattr(predictor, "_compiled", False):
                    try:
                        setattr(predictor, "_compiled", False)
                    except Exception:
                        continue
        except Exception:
            pass

    return prepared


@register_optimizer("bootstrap")
class BootstrapOptimizer(AbstractOptimizer):
    """
    BootstrapFewShot optimizer wrapper.

    Basic bootstrap optimizer that generates few-shot demonstrations
    from the training set using a teacher model.

    Best for:
    - Very small datasets (~10 examples)
    - Quick experiments
    - When minimal computational budget available
    """

    @property
    def name(self) -> str:
        return "bootstrap"

    @property
    def supports_parallel(self) -> bool:
        # Basic bootstrap is sequential
        return False

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
        Compile using BootstrapFewShot.

        Args:
            student: Module to optimize
            trainset: Training examples
            valset: Not directly used (metric evaluated on trainset)
            metric: DSPy metric function
            teacher: Optional teacher module (uses student copy if None)
            **kwargs: Additional arguments

        Returns:
            Optimized module with few-shot demonstrations
        """
        self._log_compile_start(len(trainset), len(valset or trainset))

        # Wrap metric to extract score from dict returns
        wrapped_metric = self.wrap_metric(metric) if metric else None

        # Build optimizer kwargs
        opt_kwargs = self._build_kwargs()
        student_for_compile = _prepare_iterative_bootstrap_student(student, self.name)

        try:
            optimizer = dspy.BootstrapFewShot(
                metric=wrapped_metric,
                **opt_kwargs,
            )

            # BootstrapFewShot uses student/teacher/trainset signature
            if teacher is None:
                teacher = (
                    student_for_compile.deepcopy()
                    if hasattr(student_for_compile, 'deepcopy')
                    else None
                )

            compiled = optimizer.compile(
                student_for_compile,
                teacher=teacher,
                trainset=trainset,
            )
            self._update_compile_audit(
                compile_status="completed",
                optimizer_used=self.name,
                fallback_reason="none",
            )
            return compiled

        except Exception as e:
            self._update_compile_audit(
                compile_status="failed",
                optimizer_used=self.name,
                fallback_reason="none",
                exception_summary=f"{type(e).__name__}: {e}",
            )
            logger.error(f"BootstrapFewShot compilation failed: {e}")
            raise

    def estimate_budget(self, dataset_size: int) -> int:
        """Estimate metric calls (roughly trainset_size * max_rounds)."""
        if self.config is None:
            return dataset_size * 1

        max_rounds = getattr(self.config, 'max_rounds', 1)
        return dataset_size * max_rounds

    def _build_kwargs(self) -> dict:
        """Build kwargs for BootstrapFewShot constructor."""
        return {
            'max_bootstrapped_demos': getattr(self.config, 'max_bootstrapped_demos', 2) if self.config else 2,
            'max_labeled_demos': getattr(self.config, 'max_labeled_demos', 4) if self.config else 4,
            'max_rounds': getattr(self.config, 'max_rounds', 1) if self.config else 1,
        }


@register_optimizer("bootstrap_random_search")
class BootstrapRandomSearchOptimizer(AbstractOptimizer):
    """
    BootstrapFewShotWithRandomSearch optimizer wrapper.

    Enhanced bootstrap optimizer that generates multiple candidate programs
    and evaluates them in parallel to find the best one.

    Best for:
    - Medium datasets (50+ examples)
    - When parallel evaluation available
    - Balanced computational budget

    Key advantage over basic bootstrap:
    - Explores more of the optimization space
    - Parallel evaluation of candidate programs
    - Often finds better solutions than single bootstrap
    """

    @property
    def name(self) -> str:
        return "bootstrap_random_search"

    @property
    def supports_parallel(self) -> bool:
        # Random search evaluates candidates in parallel
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
        Compile using BootstrapFewShotWithRandomSearch.

        Args:
            student: Module to optimize
            trainset: Training examples
            valset: Validation set for selecting best program
            metric: DSPy metric function
            teacher: Optional teacher module
            **kwargs: Additional arguments

        Returns:
            Best optimized module from random search
        """
        valset = valset or trainset
        self._log_compile_start(len(trainset), len(valset))

        # Wrap metric to extract score from dict returns
        wrapped_metric = self.wrap_metric(metric) if metric else None

        # Build optimizer kwargs
        opt_kwargs = self._build_kwargs()
        student_for_compile = _prepare_iterative_bootstrap_student(student, self.name)

        try:
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch

            optimizer = BootstrapFewShotWithRandomSearch(
                metric=wrapped_metric,
                **opt_kwargs,
            )

            # RandomSearch uses student/trainset signature
            compiled = optimizer.compile(
                student_for_compile,
                trainset=trainset,
            )
            self._update_compile_audit(
                compile_status="completed",
                optimizer_used=self.name,
                fallback_reason="none",
            )
            return compiled

        except ImportError:
            logger.warning(
                "BootstrapFewShotWithRandomSearch not available, "
                "falling back to basic BootstrapFewShot"
            )
            # Fall back to basic bootstrap
            basic = BootstrapOptimizer(self.config)
            compiled = basic.compile(student, trainset, valset, metric, teacher, **kwargs)
            self._update_compile_audit(
                compile_status="fallback",
                optimizer_used=basic.name,
                fallback_reason="teleprompter_unavailable",
                fallback_metadata=basic.last_compile_audit,
            )
            return compiled

        except Exception as e:
            self._update_compile_audit(
                compile_status="failed",
                optimizer_used=self.name,
                fallback_reason="none",
                exception_summary=f"{type(e).__name__}: {e}",
            )
            logger.error(f"BootstrapFewShotWithRandomSearch compilation failed: {e}")
            raise

    def estimate_budget(self, dataset_size: int) -> int:
        """
        Estimate metric calls.

        RandomSearch evaluates num_candidate_programs on the full trainset.
        """
        if self.config is None:
            return dataset_size * 10  # Default 10 candidates

        num_candidates = getattr(self.config, 'num_candidate_programs', 10)
        return dataset_size * num_candidates

    def _build_kwargs(self) -> dict:
        """Build kwargs for BootstrapFewShotWithRandomSearch constructor."""
        return {
            'max_bootstrapped_demos': getattr(self.config, 'max_bootstrapped_demos', 2) if self.config else 2,
            'max_labeled_demos': getattr(self.config, 'max_labeled_demos', 4) if self.config else 4,
            'max_rounds': getattr(self.config, 'max_rounds', 1) if self.config else 1,
            'num_candidate_programs': getattr(self.config, 'num_candidate_programs', 10) if self.config else 10,
            'num_threads': getattr(self.config, 'num_threads', 64) if self.config else 64,
        }


@register_optimizer("labeled_fewshot")
class LabeledFewShotOptimizer(AbstractOptimizer):
    """
    LabeledFewShot optimizer wrapper.

    Simplest optimizer that just uses labeled examples directly as
    few-shot demonstrations without any bootstrapping.

    Best for:
    - Extremely limited data
    - When you have high-quality labeled examples
    - Quick baseline experiments
    """

    @property
    def name(self) -> str:
        return "labeled_fewshot"

    @property
    def supports_parallel(self) -> bool:
        return False

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
        Compile using LabeledFewShot.

        Simply uses labeled examples as demonstrations.
        """
        self._log_compile_start(len(trainset), len(valset or trainset))

        try:
            from dspy.teleprompt import LabeledFewShot

            k = getattr(self.config, 'max_labeled_demos', 4) if self.config else 4

            optimizer = LabeledFewShot(k=k)
            compiled = optimizer.compile(
                student,
                trainset=trainset,
            )
            self._update_compile_audit(
                compile_status="completed",
                optimizer_used=self.name,
                fallback_reason="none",
            )
            return compiled

        except ImportError:
            logger.warning("LabeledFewShot not available, returning student unchanged")
            self._update_compile_audit(
                compile_status="noop",
                optimizer_used=self.name,
                fallback_reason="teleprompter_unavailable",
                noop_reason="teleprompter_unavailable",
            )
            return student

        except Exception as e:
            self._update_compile_audit(
                compile_status="failed",
                optimizer_used=self.name,
                fallback_reason="none",
                exception_summary=f"{type(e).__name__}: {e}",
            )
            logger.error(f"LabeledFewShot compilation failed: {e}")
            raise

    def estimate_budget(self, dataset_size: int) -> int:
        """LabeledFewShot doesn't use metric evaluation."""
        return 0
