"""
DGP-agnostic local-law stress classification.

This module provides the shared vocabulary for evaluating whether a learned
summary function g improves on a baseline, and which local laws (C1/C2/C3)
contributed to that improvement.

Key distinction:
- **Primary metric** (root MAE / downstream task): Did learned g beat baseline
  on the actual task?  This is the success criterion.
- **Local laws** (C1/C2/C3): Diagnostic — which laws improved?  These explain
  *why* g works and serve as regularization, but aren't the success criterion.

The classification logic is metric-scale agnostic: it takes raw floats for
baseline and selected values and computes relative gains and pass/fail.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


DEFAULT_LAW_GAIN_THRESHOLD = 0.10
DEFAULT_ROOT_RATIO_LIMIT = 1.05
DEFAULT_SPREAD_GAIN_THRESHOLD = 0.10
DEFAULT_PRIMARY_GAIN_THRESHOLD = 0.10


@dataclass(frozen=True)
class LawStressAssessment:
    # Primary metric (downstream task)
    primary_pass: bool
    primary_gain_frac: float
    primary_margin: float

    # Backward compat: bundle_full_success = primary_pass
    bundle_status: str
    bundle_full_success: bool

    # Law diagnostics (regularization indicators)
    c1_pass: bool
    c2_pass: bool
    c3_pass: bool
    root_pass: bool
    spread_pass: bool

    # How many laws improved
    laws_improved: int
    all_laws_pass: bool

    # Raw gain fractions
    c1_gain_frac: float
    c2_gain_frac: float
    c3_gain_frac: float
    spread_gain_frac: float
    root_ratio: float

    # Margins (positive = passed by this much)
    c1_margin: float
    c2_margin: float
    c3_margin: float
    spread_margin: float
    root_margin: float

    def to_dict(self) -> dict[str, float | bool | str | int]:
        return dict(asdict(self))


def law_bundle_score(*, c1: float, c2: float, c3: float) -> float:
    """Combined theorem bundle score: sum of normalized C1 + C2 + C3 errors."""
    return float(c1 + c2 + c3)


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


def classify_law_stress(
    *,
    baseline_c1: float,
    baseline_c2: float,
    baseline_c3: float,
    baseline_spread: float,
    baseline_root_mae: float,
    selected_c1: float,
    selected_c2: float,
    selected_c3: float,
    selected_spread: float,
    selected_root_mae: float,
    law_gain_threshold: float = DEFAULT_LAW_GAIN_THRESHOLD,
    root_ratio_limit: float = DEFAULT_ROOT_RATIO_LIMIT,
    spread_gain_threshold: float = DEFAULT_SPREAD_GAIN_THRESHOLD,
    primary_gain_threshold: float = DEFAULT_PRIMARY_GAIN_THRESHOLD,
) -> LawStressAssessment:
    c1_gain = _safe_relative_gain(baseline=float(baseline_c1), current=float(selected_c1))
    c2_gain = _safe_relative_gain(baseline=float(baseline_c2), current=float(selected_c2))
    c3_gain = _safe_relative_gain(baseline=float(baseline_c3), current=float(selected_c3))
    spread_gain = _safe_relative_gain(
        baseline=float(baseline_spread),
        current=float(selected_spread),
    )
    root_ratio = _safe_ratio(
        numerator=float(selected_root_mae),
        denominator=float(baseline_root_mae),
    )

    # Primary metric: did we improve root MAE?
    # A ratio < 1.0 means improvement; primary_gain_frac is the fractional reduction.
    primary_gain_frac = float(1.0 - root_ratio)
    primary_pass = bool(primary_gain_frac >= float(primary_gain_threshold))
    primary_margin = float(primary_gain_frac - float(primary_gain_threshold))

    # Law diagnostics (informational / regularization story)
    c1_pass = bool(c1_gain >= float(law_gain_threshold))
    c2_pass = bool(c2_gain >= float(law_gain_threshold))
    c3_pass = bool(c3_gain >= float(law_gain_threshold))
    spread_pass = bool(spread_gain >= float(spread_gain_threshold))
    root_pass = bool(root_ratio <= float(root_ratio_limit))

    laws_improved = int(c1_pass) + int(c2_pass) + int(c3_pass)
    all_laws_pass = bool(c1_pass and c2_pass and c3_pass)

    # bundle_full_success = primary metric passed (backward compat name)
    bundle_full_success = bool(primary_pass)

    if primary_pass and all_laws_pass:
        bundle_status = "full_success"
    elif primary_pass and not all_laws_pass:
        bundle_status = "primary_only"
    elif not primary_pass and all_laws_pass:
        bundle_status = "laws_only"
    else:
        bundle_status = "failure"

    return LawStressAssessment(
        primary_pass=bool(primary_pass),
        primary_gain_frac=float(primary_gain_frac),
        primary_margin=float(primary_margin),
        bundle_status=str(bundle_status),
        bundle_full_success=bool(bundle_full_success),
        c1_pass=bool(c1_pass),
        c2_pass=bool(c2_pass),
        c3_pass=bool(c3_pass),
        root_pass=bool(root_pass),
        spread_pass=bool(spread_pass),
        laws_improved=int(laws_improved),
        all_laws_pass=bool(all_laws_pass),
        c1_gain_frac=float(c1_gain),
        c2_gain_frac=float(c2_gain),
        c3_gain_frac=float(c3_gain),
        spread_gain_frac=float(spread_gain),
        root_ratio=float(root_ratio),
        c1_margin=float(c1_gain - float(law_gain_threshold)),
        c2_margin=float(c2_gain - float(law_gain_threshold)),
        c3_margin=float(c3_gain - float(law_gain_threshold)),
        spread_margin=float(spread_gain - float(spread_gain_threshold)),
        root_margin=float(float(root_ratio_limit) - root_ratio),
    )


def infer_law_stress_failure_reason(row: dict) -> str:
    if bool(row.get("primary_pass", row.get("bundle_full_success"))):
        return ""
    if bool(row.get("all_laws_pass")) and not bool(row.get("primary_pass")):
        return "laws_without_downstream"
    if not bool(row.get("spread_pass")):
        return "schedule_instability"
    if float(row.get("audit_fraction", 1.0)) <= 0.05:
        return "insufficient_audit"
    if int(row.get("train_docs", 0)) <= 512:
        return "insufficient_data"
    if int(row.get("state_dim", 64)) < 64 or int(row.get("hidden_dim", 256)) < 256:
        return "insufficient_capacity"
    return "objective_conflict"


# Backward-compatible alias
markov_law_bundle_score = law_bundle_score


__all__ = [
    "DEFAULT_LAW_GAIN_THRESHOLD",
    "DEFAULT_PRIMARY_GAIN_THRESHOLD",
    "DEFAULT_ROOT_RATIO_LIMIT",
    "DEFAULT_SPREAD_GAIN_THRESHOLD",
    "LawStressAssessment",
    "classify_law_stress",
    "infer_law_stress_failure_reason",
    "law_bundle_score",
    "markov_law_bundle_score",
]
