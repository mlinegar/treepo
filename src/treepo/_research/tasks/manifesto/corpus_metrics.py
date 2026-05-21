"""
Corpus-level validity metrics for Benoit-aligned manifesto runs.

The primary metric is Pearson correlation of LLM-derived per-party scores
with expert ensemble means, matched to Benoit et al. (2026 AJPS) Figure 1
(baseline) and Table 6 (open-weight replication). We also report:

- Spearman rank correlation (diagnostic for monotone-but-nonlinear drift)
- Mean and median absolute deviation on the common 1-7 scale
- Sample size after NA exclusion and NA rate (Benoit Table 4 analogue)

Pearson r uses Fisher-z for confidence intervals. Affine invariance of r
means the LLM output scale does not have to match the benchmark scale —
both can be supplied on their native axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CorrelationReport:
    n: int
    n_na: int
    pearson_r: float
    pearson_ci_low: float
    pearson_ci_high: float
    spearman_r: float
    mae_rescaled: Optional[float]
    rmse_rescaled: Optional[float]
    pearson_defined: bool = True
    spearman_defined: bool = True
    undefined_reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "n_na": self.n_na,
            "pearson_r": self.pearson_r,
            "pearson_ci_low": self.pearson_ci_low,
            "pearson_ci_high": self.pearson_ci_high,
            "spearman_r": self.spearman_r,
            "mae_rescaled": self.mae_rescaled,
            "rmse_rescaled": self.rmse_rescaled,
            "pearson_defined": self.pearson_defined,
            "spearman_defined": self.spearman_defined,
            "undefined_reason": self.undefined_reason,
        }


def compute_corpus_pearson_r(
    pred: Sequence[Optional[float]],
    true: Sequence[Optional[float]],
    *,
    alpha: float = 0.05,
    pred_rescaled: Optional[Sequence[Optional[float]]] = None,
) -> CorrelationReport:
    """
    Pearson r with Fisher-z CI, computed on (pred, true) pairs where both
    values are not None.

    If `pred_rescaled` is provided (LLM scores rescaled onto the same axis
    as `true`), the report also includes MAE and RMSE on that axis. This is
    optional because Pearson r itself is affine-invariant and does not need
    either series rescaled.
    """
    pred_arr, true_arr, pred_rs_arr, n_na = _paired_finite(pred, true, pred_rescaled)

    n = len(pred_arr)
    if n < 3:
        raise ValueError(
            f"Need at least 3 non-NA pairs for correlation; got {n} from input of "
            f"length {len(pred)}."
        )

    r, pearson_defined, pearson_reason = _finite_corr_or_zero(pred_arr, true_arr)
    r = _clamp(r, -0.999999, 0.999999)

    if pearson_defined:
        z = math.atanh(r)
        se = 1.0 / math.sqrt(n - 3)
        z_crit = _z_crit(alpha)
        ci_low = math.tanh(z - z_crit * se)
        ci_high = math.tanh(z + z_crit * se)
    else:
        ci_low = 0.0
        ci_high = 0.0

    spearman, spearman_defined, spearman_reason = _spearman(pred_arr, true_arr)

    if pred_rs_arr is not None:
        diff = pred_rs_arr - true_arr
        mae = float(np.mean(np.abs(diff)))
        rmse = float(math.sqrt(np.mean(diff * diff)))
    else:
        mae = None
        rmse = None

    return CorrelationReport(
        n=n,
        n_na=n_na,
        pearson_r=r,
        pearson_ci_low=ci_low,
        pearson_ci_high=ci_high,
        spearman_r=spearman,
        mae_rescaled=mae,
        rmse_rescaled=rmse,
        pearson_defined=pearson_defined,
        spearman_defined=spearman_defined,
        undefined_reason=pearson_reason or spearman_reason,
    )


def _paired_finite(
    pred: Iterable[Optional[float]],
    true: Iterable[Optional[float]],
    pred_rescaled: Optional[Iterable[Optional[float]]],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    pred_list = list(pred)
    true_list = list(true)
    rs_list = list(pred_rescaled) if pred_rescaled is not None else None
    if len(pred_list) != len(true_list):
        raise ValueError(
            f"pred and true must be the same length: got {len(pred_list)} vs {len(true_list)}"
        )
    if rs_list is not None and len(rs_list) != len(pred_list):
        raise ValueError("pred_rescaled must match pred length")

    p: list[float] = []
    t: list[float] = []
    r: list[float] = []
    n_na = 0
    for i, (pi, ti) in enumerate(zip(pred_list, true_list)):
        ri = rs_list[i] if rs_list is not None else None
        if pi is None or ti is None or not math.isfinite(pi) or not math.isfinite(ti):
            n_na += 1
            continue
        p.append(float(pi))
        t.append(float(ti))
        if rs_list is not None and ri is not None and math.isfinite(ri):
            r.append(float(ri))
        elif rs_list is not None:
            n_na += 1
    p_arr = np.asarray(p, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    r_arr = np.asarray(r, dtype=float) if rs_list is not None and len(r) == len(p) else None
    return p_arr, t_arr, r_arr, n_na


def _finite_corr_or_zero(a: np.ndarray, b: np.ndarray) -> tuple[float, bool, Optional[str]]:
    if len(a) != len(b):
        raise ValueError("correlation arrays must be the same length")
    if len(a) == 0:
        return 0.0, False, "empty input"
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0, False, "non-finite input"
    if float(np.ptp(a)) == 0.0:
        return 0.0, False, "constant predictions"
    if float(np.ptp(b)) == 0.0:
        return 0.0, False, "constant targets"
    r = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(r):
        return 0.0, False, "non-finite correlation"
    return r, True, None


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, bool, Optional[str]]:
    a_rank = _rankdata(a)
    b_rank = _rankdata(b)
    value, defined, reason = _finite_corr_or_zero(a_rank, b_rank)
    if reason == "constant predictions":
        reason = "constant prediction ranks"
    elif reason == "constant targets":
        reason = "constant target ranks"
    return value, defined, reason


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_vals = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _clamp(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(lo, min(hi, x))


def _z_crit(alpha: float) -> float:
    if abs(alpha - 0.05) < 1e-9:
        return 1.959963984540054
    if abs(alpha - 0.01) < 1e-9:
        return 2.5758293035489004
    if abs(alpha - 0.10) < 1e-9:
        return 1.6448536269514722
    raise ValueError(f"Only alpha in {{0.01, 0.05, 0.10}} supported; got {alpha}")
