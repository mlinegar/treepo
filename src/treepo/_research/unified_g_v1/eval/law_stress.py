"""Gain-fraction utilities matching `src/ctreepo/sim/core/law_stress_common.py`.

The project-standard law-stress scoring expresses every metric as a
`gain_frac = 1 - (selected / baseline)` in (-∞, 1.0]. We mirror that shape
here for task-level MAE so RILE reports sit directly next to C1/C2/C3 gains
without per-metric rescaling.

Baseline strategy: **predict training-target mean** — the same constant for
every val example. Standard regression baseline; captures the "no signal"
pathology cleanly (a model with gain_frac ≈ 0 has learned nothing beyond
the training mean).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


DEFAULT_PRIMARY_GAIN_THRESHOLD = 0.10
"""Matches the `primary_gain_threshold` default in law_stress_common.py."""


def _default_target_getter(ex: Any) -> float:
    """Pull numeric RILE out of a TreeExample.

    Text-lane oracles store it in `extra["target_raw"]` (string `target`
    holds the LLM completion). Other oracles store a numeric `target`
    directly.
    """
    target_raw = None
    if hasattr(ex, "extra"):
        try:
            target_raw = ex.extra.get("target_raw")
        except AttributeError:
            target_raw = None
    if target_raw is not None:
        return float(target_raw)
    return float(getattr(ex, "target"))


def predict_train_mean_baseline_mae(
    train_items: Sequence[Any],
    val_items: Sequence[Any],
    *,
    target_getter: Callable[[Any], float] = _default_target_getter,
) -> float:
    """Constant-predictor val MAE: predict `mean(train_targets)` for every val item."""
    if not val_items:
        return 0.0
    train_targets = [target_getter(ex) for ex in train_items]
    if not train_targets:
        return 0.0
    constant = sum(train_targets) / float(len(train_targets))
    val_targets = [target_getter(ex) for ex in val_items]
    return float(sum(abs(constant - y) for y in val_targets) / float(len(val_targets)))


def gain_frac(*, model_mae: float, baseline_mae: float) -> float:
    """`1 - model_mae/baseline_mae`, with the same safe-ratio handling as law_stress_common."""
    base = float(baseline_mae)
    cur = float(model_mae)
    if base <= 0.0:
        # Mirrors _safe_ratio: if baseline is zero, any non-zero model MAE is "infinitely worse".
        if cur <= 0.0:
            return 1.0
        return float("-inf")
    return 1.0 - (cur / base)


@dataclass(frozen=True)
class LawStressReport:
    baseline_mae: float
    model_mae: float
    gain_frac: float
    passed: bool

    def to_metrics(self, *, prefix: str = "val_mae") -> dict[str, float]:
        """Flatten into a `dict[str, float]` suitable for `FitResult.metrics`."""
        return {
            f"baseline_{prefix}": float(self.baseline_mae),
            f"{prefix}_raw": float(self.model_mae),
            f"{prefix}_gain_frac": float(self.gain_frac),
            f"{prefix}_pass": 1.0 if self.passed else 0.0,
        }


def report(
    *,
    model_mae: float,
    baseline_mae: float,
    primary_gain_threshold: float = DEFAULT_PRIMARY_GAIN_THRESHOLD,
) -> LawStressReport:
    g = gain_frac(model_mae=model_mae, baseline_mae=baseline_mae)
    return LawStressReport(
        baseline_mae=float(baseline_mae),
        model_mae=float(model_mae),
        gain_frac=float(g),
        passed=bool(g >= float(primary_gain_threshold)),
    )


__all__ = [
    "DEFAULT_PRIMARY_GAIN_THRESHOLD",
    "LawStressReport",
    "gain_frac",
    "predict_train_mean_baseline_mae",
    "report",
]
