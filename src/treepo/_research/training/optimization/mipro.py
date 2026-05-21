"""
MIPROv2 optimizer wrapper.

MIPRO (Multi-stage Instruction Proposal and Optimization) is a sophisticated
optimizer for large datasets that combines instruction optimization with
few-shot demonstration selection.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set

import dspy

from .base import AbstractOptimizer
from .registry import register_optimizer

logger = logging.getLogger(__name__)


@register_optimizer("mipro")
class MIPROOptimizer(AbstractOptimizer):
    """
    MIPROv2 optimizer wrapper.

    MIPRO (Multi-stage Instruction Proposal and Optimization) uses a
    multi-stage approach to optimize both instructions and demonstrations.

    Best for:
    - Large datasets (200+ examples)
    - When instruction optimization is important
    - Production-quality optimization

    Budget modes:
    - light: Quick optimization (~300 metric calls)
    - medium: Balanced (~1000 metric calls)
    - heavy: Thorough (~3000 metric calls)
    """

    @property
    def name(self) -> str:
        return "mipro"

    @property
    def supports_parallel(self) -> bool:
        # MIPROv2 uses num_threads for parallel evaluation
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
        Compile using MIPROv2.

        Args:
            student: Module to optimize
            trainset: Training examples
            valset: Not directly used by MIPROv2
            metric: DSPy metric function
            teacher: Not used by MIPROv2
            **kwargs: Additional arguments

        Returns:
            Optimized module
        """
        self._log_compile_start(len(trainset), len(valset or trainset))

        # Wrap metric to extract score from dict returns
        wrapped_metric = self.wrap_metric(metric) if metric else None

        # Build optimizer kwargs
        opt_kwargs = self._build_kwargs()
        compile_kwargs = self._build_compile_kwargs(trainset, kwargs)

        compact_trainset, train_compaction = self._compact_trainset(student, trainset)
        compact_valset, val_compaction = self._compact_trainset(student, valset or [])
        if compact_valset:
            compile_kwargs.setdefault("valset", compact_valset)

        try:
            optimizer = dspy.MIPROv2(
                metric=wrapped_metric,
                **opt_kwargs,
            )

            # MIPROv2 uses student/trainset signature
            compiled = optimizer.compile(
                student=student,
                trainset=compact_trainset,
                **compile_kwargs,
            )
            self._update_compile_audit(
                compile_status="completed",
                optimizer_used=self.name,
                fallback_reason="none",
                input_mutation_flags={
                    "train_compaction": train_compaction,
                    "val_compaction": val_compaction,
                },
            )
            return compiled

        except Exception as e:
            self._update_compile_audit(
                compile_status="failed",
                optimizer_used=self.name,
                fallback_reason="none",
                input_mutation_flags={
                    "train_compaction": train_compaction,
                    "val_compaction": val_compaction,
                },
                exception_summary=f"{type(e).__name__}: {e}",
            )
            logger.error(f"MIPROv2 compilation failed: {e}")
            raise

    def estimate_budget(self, dataset_size: int) -> int:
        """Estimate metric calls based on budget mode."""
        if self.config is None:
            return 1000  # Default medium

        budget = getattr(self.config, 'mipro_auto', 'medium')
        budget_estimates = {
            'light': 300,
            'medium': 1000,
            'heavy': 3000,
        }
        return budget_estimates.get(budget, 1000)

    def _build_kwargs(self) -> dict:
        """Build kwargs for MIPROv2 constructor."""
        if self.config is None:
            return {
                'auto': 'medium',
                'num_threads': 64,
            }

        budget = getattr(self.config, 'mipro_auto', 'medium')
        num_threads = getattr(self.config, 'num_threads', 64)

        return {
            'auto': budget,
            'num_threads': num_threads,
        }

    def _build_compile_kwargs(
        self,
        trainset: List[dspy.Example],
        extra_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build kwargs for MIPROv2.compile()."""
        compile_kwargs: Dict[str, Any] = dict(extra_kwargs or {})

        if self.config is None:
            data_aware = True
            view_batch = 3
        else:
            data_aware = bool(getattr(self.config, "mipro_data_aware_proposer", True))
            view_batch = int(getattr(self.config, "mipro_view_data_batch_size", 3))

        if trainset:
            view_batch = max(1, min(view_batch, len(trainset)))
        else:
            view_batch = 1

        compile_kwargs.setdefault("data_aware_proposer", data_aware)
        compile_kwargs.setdefault("view_data_batch_size", view_batch)
        return compile_kwargs

    def _compact_trainset(
        self,
        student: dspy.Module,
        examples: List[dspy.Example],
    ) -> tuple[List[dspy.Example], Dict[str, Any]]:
        """Drop optional heavy fields before MIPRO proposer stages."""
        if not examples:
            return examples, {
                "examples_before": 0,
                "examples_after": 0,
                "chars_before": 0,
                "chars_after": 0,
                "max_chars": 0,
                "keep_original_content": True,
                "truncated": False,
                "dropped_optional_original_content": False,
            }

        if self.config is None:
            max_chars = 0
            drop_optional_original = True
        else:
            max_chars = int(getattr(self.config, "mipro_max_example_chars", 0))
            drop_optional_original = bool(
                getattr(self.config, "mipro_drop_optional_original_content", True)
            )

        required_inputs, accepts_kwargs = self._required_forward_inputs(student)
        keep_original = (
            "original_content" in required_inputs
            or "original_text" in required_inputs
            or accepts_kwargs
            or not drop_optional_original
        )

        chars_before = self._estimate_text_chars(examples)
        compacted: List[dspy.Example] = []
        for example in examples:
            compacted.append(
                self._compact_example(
                    example,
                    max_chars=max_chars,
                    keep_original_content=keep_original,
                )
            )
        chars_after = self._estimate_text_chars(compacted)

        if max_chars > 0:
            logger.warning(
                "MIPRO example truncation is enabled (max_chars=%d). This can alter prompts.",
                max_chars,
            )
        if chars_after < chars_before:
            logger.info(
                "Compacted MIPRO examples: chars %d -> %d (max_chars=%d, keep_original=%s)",
                chars_before,
                chars_after,
                max_chars,
                keep_original,
            )
        return compacted, {
            "examples_before": int(len(examples)),
            "examples_after": int(len(compacted)),
            "chars_before": int(chars_before),
            "chars_after": int(chars_after),
            "max_chars": int(max_chars),
            "keep_original_content": bool(keep_original),
            "truncated": bool(chars_after < chars_before),
            "dropped_optional_original_content": bool(
                drop_optional_original and not keep_original
            ),
        }

    def _required_forward_inputs(self, student: dspy.Module) -> tuple[Set[str], bool]:
        """Infer required forward parameters for the student module."""
        forward = getattr(student, "forward", None)
        if forward is None:
            return set(), False

        try:
            parameters = inspect.signature(forward).parameters.values()
        except (TypeError, ValueError):
            return set(), False

        required: Set[str] = set()
        accepts_kwargs = False
        for param in parameters:
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_kwargs = True
                continue
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ) and param.default is inspect.Parameter.empty:
                required.add(param.name)

        return required, accepts_kwargs

    def _compact_example(
        self,
        example: dspy.Example,
        *,
        max_chars: int,
        keep_original_content: bool,
    ) -> dspy.Example:
        """Return a compacted copy of a DSPy example."""
        try:
            payload: Dict[str, Any] = dict(example.toDict())
        except Exception:
            payload = dict(example)

        input_keys = set(getattr(example, "_input_keys", set()))
        if not keep_original_content:
            if "original_content" in input_keys and (
                "summary" in input_keys or "text" in input_keys
            ):
                input_keys.remove("original_content")
                payload.pop("original_content", None)
            if "original_text" in input_keys and (
                "summary" in input_keys or "text" in input_keys
            ):
                input_keys.remove("original_text")
                payload.pop("original_text", None)

        if max_chars > 0:
            for key, value in list(payload.items()):
                if isinstance(value, str) and len(value) > max_chars:
                    payload[key] = self._truncate_text(value, max_chars)

        compact = dspy.Example(**payload)
        if input_keys:
            ordered_inputs = [key for key in payload.keys() if key in input_keys]
            if not ordered_inputs:
                ordered_inputs = sorted(input_keys)
            compact = compact.with_inputs(*ordered_inputs)
        return compact

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """Trim long text while keeping both prefix and suffix."""
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        if max_chars < 64:
            return text[:max_chars]

        marker = "\n...[truncated]...\n"
        room = max_chars - len(marker)
        if room <= 0:
            return text[:max_chars]

        head = int(room * 0.7)
        tail = room - head
        if tail <= 0:
            return text[:max_chars]
        return f"{text[:head]}{marker}{text[-tail:]}"

    @staticmethod
    def _estimate_text_chars(examples: List[dspy.Example]) -> int:
        """Estimate total text payload in chars across examples."""
        total = 0
        for example in examples:
            try:
                payload = dict(example.toDict())
            except Exception:
                payload = dict(example)
            for value in payload.values():
                if isinstance(value, str):
                    total += len(value)
        return total
