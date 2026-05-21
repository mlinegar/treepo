from __future__ import annotations

from dataclasses import asdict, dataclass


THEOREM_SCORE_SPREAD_WEIGHT = 0.25
DEFAULT_THEOREM_GAIN_THRESHOLD = 0.10
DEFAULT_SPREAD_GAIN_THRESHOLD = 0.10
DEFAULT_ROOT_RATIO_LIMIT = 1.05


@dataclass(frozen=True)
class CapabilityAssessment:
    capability_status: str
    theorem_pass: bool
    spread_pass: bool
    root_pass: bool
    theorem_gain_frac: float
    spread_gain_frac: float
    root_ratio: float
    theorem_margin: float
    spread_margin: float
    root_margin: float

    def to_dict(self) -> dict[str, float | bool | str]:
        return dict(asdict(self))


def markov_theorem_score(*, leaf: float, merge: float, spread: float) -> float:
    return float(leaf + merge + THEOREM_SCORE_SPREAD_WEIGHT * spread)


def _safe_relative_gain(*, baseline: float, current: float) -> float:
    base = float(baseline)
    cur = float(current)
    if base <= 0.0:
        return 0.0 if cur <= 0.0 else float("-inf")
    return float((base - cur) / base)


def _safe_ratio(*, numerator: float, denominator: float) -> float:
    num = float(numerator)
    den = float(denominator)
    if den <= 0.0:
        return 1.0 if num <= 0.0 else float("inf")
    return float(num / den)


def classify_capability(
    *,
    baseline_theorem_score: float,
    baseline_spread: float,
    baseline_root_mae: float,
    selected_theorem_score: float,
    selected_spread: float,
    selected_root_mae: float,
    theorem_gain_threshold: float = DEFAULT_THEOREM_GAIN_THRESHOLD,
    spread_gain_threshold: float = DEFAULT_SPREAD_GAIN_THRESHOLD,
    root_ratio_limit: float = DEFAULT_ROOT_RATIO_LIMIT,
) -> CapabilityAssessment:
    theorem_gain_frac = _safe_relative_gain(
        baseline=float(baseline_theorem_score),
        current=float(selected_theorem_score),
    )
    spread_gain_frac = _safe_relative_gain(
        baseline=float(baseline_spread),
        current=float(selected_spread),
    )
    root_ratio = _safe_ratio(
        numerator=float(selected_root_mae),
        denominator=float(baseline_root_mae),
    )

    theorem_pass = bool(theorem_gain_frac >= float(theorem_gain_threshold))
    spread_pass = bool(spread_gain_frac >= float(spread_gain_threshold))
    root_pass = bool(root_ratio <= float(root_ratio_limit))

    if theorem_pass and spread_pass and root_pass:
        status = "full_success"
    elif theorem_pass and spread_pass and not root_pass:
        status = "theorem_only"
    elif root_pass and not (theorem_pass and spread_pass):
        status = "root_only"
    else:
        status = "failure"

    return CapabilityAssessment(
        capability_status=str(status),
        theorem_pass=theorem_pass,
        spread_pass=spread_pass,
        root_pass=root_pass,
        theorem_gain_frac=float(theorem_gain_frac),
        spread_gain_frac=float(spread_gain_frac),
        root_ratio=float(root_ratio),
        theorem_margin=float(theorem_gain_frac - float(theorem_gain_threshold)),
        spread_margin=float(spread_gain_frac - float(spread_gain_threshold)),
        root_margin=float(float(root_ratio_limit) - root_ratio),
    )


__all__ = [
    "CapabilityAssessment",
    "DEFAULT_ROOT_RATIO_LIMIT",
    "DEFAULT_SPREAD_GAIN_THRESHOLD",
    "DEFAULT_THEOREM_GAIN_THRESHOLD",
    "THEOREM_SCORE_SPREAD_WEIGHT",
    "classify_capability",
    "markov_theorem_score",
]
