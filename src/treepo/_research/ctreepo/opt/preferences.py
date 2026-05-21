from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PreferenceOutcome:
    """A pairwise preference label plus minimal explainability metadata."""

    preferred: Literal["A", "B", "tie"]
    confidence: float
    reasoning: str = ""
    reference: Optional[float] = None
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    loss_a: Optional[float] = None
    loss_b: Optional[float] = None


def _bounded_confidence(raw: float) -> float:
    if not math.isfinite(raw):
        return 0.5
    return float(max(0.0, min(0.99, raw)))


def _confidence_from_margin(margin: float, *, scale: float) -> float:
    # Map 0 -> 0.5 and grow linearly, saturating. "scale" controls slope.
    if not math.isfinite(margin):
        return 0.5
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    return _bounded_confidence(0.5 + abs(float(margin)) / scale)


def derive_preference_from_utilities(
    utility_a: float,
    utility_b: float,
    *,
    tie_margin: float = 0.0,
    scale: Optional[float] = None,
) -> PreferenceOutcome:
    """Prefer the candidate with higher utility (ties within ``tie_margin``)."""
    utility_a = float(utility_a)
    utility_b = float(utility_b)
    if not (math.isfinite(utility_a) and math.isfinite(utility_b)):
        return PreferenceOutcome(preferred="tie", confidence=0.5, reasoning="Non-finite utilities.")

    diff = utility_a - utility_b
    if abs(diff) <= float(tie_margin):
        return PreferenceOutcome(
            preferred="tie",
            confidence=0.5,
            reasoning=f"Tie: |uA-uB|={abs(diff):.6g} <= {float(tie_margin):.6g}.",
        )

    preferred: Literal["A", "B"] = "A" if diff > 0 else "B"
    conf_scale = scale if scale is not None else max(1e-8, abs(utility_a), abs(utility_b), 1.0)
    confidence = _confidence_from_margin(diff, scale=conf_scale)
    reasoning = (
        f"{preferred} preferred by utility: uA={utility_a:.6g}, uB={utility_b:.6g}."
    )
    return PreferenceOutcome(
        preferred=preferred,
        confidence=confidence,
        reasoning=reasoning,
    )


def derive_preference_from_losses(
    loss_a: float,
    loss_b: float,
    *,
    tie_margin: float = 0.0,
    scale: Optional[float] = None,
) -> PreferenceOutcome:
    """Prefer the candidate with lower loss (ties within ``tie_margin``)."""
    loss_a = float(loss_a)
    loss_b = float(loss_b)
    if not (math.isfinite(loss_a) and math.isfinite(loss_b)):
        return PreferenceOutcome(preferred="tie", confidence=0.5, reasoning="Non-finite losses.")

    diff = loss_a - loss_b
    if abs(diff) <= float(tie_margin):
        return PreferenceOutcome(
            preferred="tie",
            confidence=0.5,
            reasoning=f"Tie: |lA-lB|={abs(diff):.6g} <= {float(tie_margin):.6g}.",
            loss_a=loss_a,
            loss_b=loss_b,
        )

    # diff < 0 => A has lower loss => A preferred.
    preferred: Literal["A", "B"] = "A" if diff < 0 else "B"
    conf_scale = scale if scale is not None else max(1e-8, abs(loss_a), abs(loss_b), 1.0)
    confidence = _confidence_from_margin(diff, scale=conf_scale)
    reasoning = f"{preferred} preferred by loss: lA={loss_a:.6g}, lB={loss_b:.6g}."
    return PreferenceOutcome(
        preferred=preferred,
        confidence=confidence,
        reasoning=reasoning,
        loss_a=loss_a,
        loss_b=loss_b,
    )


def derive_preference_from_scores(
    *,
    reference: float,
    score_a: float,
    score_b: float,
    tie_margin: float = 0.0,
    scale_range: Optional[float] = None,
) -> PreferenceOutcome:
    """Prefer the candidate whose score is closer to the reference score."""
    reference = float(reference)
    score_a = float(score_a)
    score_b = float(score_b)
    if not (math.isfinite(reference) and math.isfinite(score_a) and math.isfinite(score_b)):
        return PreferenceOutcome(preferred="tie", confidence=0.5, reasoning="Non-finite scores.")

    error_a = abs(score_a - reference)
    error_b = abs(score_b - reference)
    outcome = derive_preference_from_losses(
        error_a,
        error_b,
        tie_margin=tie_margin,
        scale=scale_range,
    )
    return PreferenceOutcome(
        preferred=outcome.preferred,
        confidence=outcome.confidence,
        reasoning=outcome.reasoning,
        reference=reference,
        score_a=score_a,
        score_b=score_b,
        loss_a=error_a,
        loss_b=error_b,
    )

