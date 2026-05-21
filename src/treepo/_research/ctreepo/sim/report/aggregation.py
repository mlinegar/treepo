"""Family-agnostic aggregation for unified reports."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean
from typing import Dict, List, Sequence, Tuple

from treepo._research.ctreepo.sim.core.law_stress_common import infer_law_stress_failure_reason
from treepo._research.ctreepo.sim.report.pdf_utils import safe_sem


def aggregate_law_stress(
    rows: Sequence[dict],
    *,
    group_keys: Sequence[str],
) -> List[dict]:
    """Group assessed rows and compute pass rates, gain fractions, and SEM.

    Each output row has:
      n_runs, c1_pass_rate, c2_pass_rate, c3_pass_rate,
      root_pass_rate, spread_pass_rate, bundle_full_success_rate,
      root_ratio (mean), root_ratio_sem,
      c1/c2/c3/spread_gain_frac, bundle_margin_mean,
      dominant_failure_reason,
      + all group_key values.
    """
    groups: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        groups[key].append(row)

    aggregated: List[dict] = []
    for key, group in sorted(groups.items()):
        payload = {name: value for name, value in zip(group_keys, key)}
        root_ratios = [float(row["root_ratio"]) for row in group]
        prim_gains = [1.0 - rr for rr in root_ratios]
        payload.update(
            {
                "n_runs": len(group),
                # Pass rates
                "c1_pass_rate": float(fmean(1.0 if bool(row["c1_pass"]) else 0.0 for row in group)),
                "c2_pass_rate": float(fmean(1.0 if bool(row["c2_pass"]) else 0.0 for row in group)),
                "c3_pass_rate": float(fmean(1.0 if bool(row["c3_pass"]) else 0.0 for row in group)),
                "root_pass_rate": float(fmean(1.0 if bool(row["root_pass"]) else 0.0 for row in group)),
                "spread_pass_rate": float(fmean(1.0 if bool(row["spread_pass"]) else 0.0 for row in group)),
                "bundle_full_success_rate": float(fmean(1.0 if bool(row["bundle_full_success"]) else 0.0 for row in group)),
                # Root ratio & PrimGain
                "root_ratio": float(fmean(root_ratios)),
                "root_ratio_sem": float(safe_sem(root_ratios)),
                "prim_gain_mean": float(fmean(prim_gains)),
                "prim_gain_sem": float(safe_sem(prim_gains)),
                # Gain fractions
                "c1_gain_frac": float(fmean(float(row["c1_gain_frac"]) for row in group)),
                "c2_gain_frac": float(fmean(float(row["c2_gain_frac"]) for row in group)),
                "c3_gain_frac": float(fmean(float(row["c3_gain_frac"]) for row in group)),
                "spread_gain_frac": float(fmean(float(row["spread_gain_frac"]) for row in group)),
                # Raw metric means
                "test_c1": float(fmean(float(row["test_c1"]) for row in group)),
                "test_c2": float(fmean(float(row["test_c2"]) for row in group)),
                "test_c3": float(fmean(float(row["test_c3"]) for row in group)),
                "test_spread": float(fmean(float(row["test_spread"]) for row in group)),
                "test_primary": float(fmean(float(row["test_primary"]) for row in group)),
                "test_bundle_score": float(fmean(float(row["test_bundle_score"]) for row in group)),
                # Bundle margin
                "bundle_margin_mean": float(fmean(
                    min(
                        float(row["c1_margin"]),
                        float(row["c2_margin"]),
                        float(row["c3_margin"]),
                        float(row["root_margin"]),
                    )
                    for row in group
                )),
                # Failure reason
                "dominant_failure_reason": (
                    Counter(
                        str(row.get("failure_reason", ""))
                        for row in group
                        if str(row.get("failure_reason", ""))
                    ).most_common(1)[0][0]
                    if any(str(row.get("failure_reason", "")) for row in group)
                    else ""
                ),
            }
        )
        # Infer failure reason for the aggregated row if not already set
        if not payload.get("failure_reason"):
            payload["failure_reason"] = infer_law_stress_failure_reason(payload)
        aggregated.append(payload)
    return aggregated


def build_downstream_table(
    aggregated_rows: Sequence[dict],
    *,
    packages_order: Sequence[str],
    package_labels: Dict[str, str],
    primary_label: str,
) -> List[str]:
    """Build a markdown table comparing downstream metrics across law packages."""
    lines = [
        "### Downstream Comparison Table",
        "",
        f"Each row averages across all configuration cells for a given law package.",
        f"**PrimGain** = 1 − {primary_label} ratio; positive means the learned g has lower held-out {primary_label} than the matched root-only baseline.",
        "",
        f"| Package | PrimGain | {primary_label} ratio | C1 pass% | C2 pass% | C3 pass% | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pkg in packages_order:
        pkg_rows = [row for row in aggregated_rows if str(row.get("law_package", "")) == pkg]
        if not pkg_rows:
            continue
        root_ratio = float(fmean(float(row["root_ratio"]) for row in pkg_rows))
        prim_gain = 1.0 - root_ratio
        c1_pass = float(fmean(float(row.get("c1_pass_rate", 0.0)) for row in pkg_rows))
        c2_pass = float(fmean(float(row.get("c2_pass_rate", 0.0)) for row in pkg_rows))
        c3_pass = float(fmean(float(row.get("c3_pass_rate", 0.0)) for row in pkg_rows))
        sem = float(safe_sem([1.0 - float(row["root_ratio"]) for row in pkg_rows]))
        if pkg == "root_only":
            interp = "baseline (no local laws)"
        elif prim_gain >= 0.10:
            interp = "improves downstream"
        elif prim_gain >= 0.0:
            interp = "neutral downstream"
        else:
            interp = "hurts downstream"
        label = package_labels.get(pkg, pkg)
        sem_str = f" ±{100.0 * sem:.1f}" if sem > 0.001 else ""
        lines.append(
            f"| `{label}` | {100.0 * prim_gain:+.1f}%{sem_str} | {root_ratio:.3f} "
            f"| {100.0 * c1_pass:.0f}% | {100.0 * c2_pass:.0f}% | {100.0 * c3_pass:.0f}% "
            f"| {interp} |"
        )
    lines.append("")
    return lines


def aggregate_learnability(
    rows: Sequence[dict],
    *,
    group_keys: Sequence[str],
    agg: str = "median",
) -> List[dict]:
    """Group learnability rows by config and reduce across seeds.

    Returns one dict per group with aggregated metrics and n_runs.
    """
    from statistics import median as stat_median
    import numpy as np

    def _reduce(xs: Sequence[float]) -> float:
        vals = [float(x) for x in xs if np.isfinite(float(x))]
        if not vals:
            return float("nan")
        if agg == "median":
            return float(stat_median(vals))
        if agg == "mean":
            return float(fmean(vals))
        raise ValueError(f"unsupported aggregate: {agg!r}")

    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        groups[key].append(row)

    # Identify all numeric fields to aggregate
    _config_keys = set(group_keys) | {"path", "seed", "effective_data_seed", "effective_model_seed",
                                       "c3_audit_strategy", "analysis_partition_mode"}
    aggregated: List[dict] = []
    for key, group in sorted(groups.items()):
        payload = {name: value for name, value in zip(group_keys, key)}
        payload["n_runs"] = len(group)

        # Aggregate all numeric fields not in group_keys
        for field in group[0].keys():
            if field in _config_keys or field == "n_runs":
                continue
            try:
                vals = [float(row[field]) for row in group]
            except (TypeError, ValueError):
                continue
            payload[field] = _reduce(vals)

        # Compute derived scores
        leaf = payload.get("learned_leaf_mae_n", float("nan"))
        merge = payload.get("learned_merge_mae_n", float("nan"))
        spread = payload.get("learned_spread_n", float("nan"))
        if np.isfinite(leaf) and np.isfinite(merge) and np.isfinite(spread):
            from treepo._research.ctreepo.sim.report.data_loading import _law_score
            payload["theorem_score"] = _law_score(leaf=leaf, merge=merge, spread=spread)
            payload["learned_law_score_n"] = payload["theorem_score"]
            payload["law_score"] = payload["theorem_score"]
        else:
            payload.setdefault("theorem_score", payload.get("learned_law_score_n", float("nan")))

        aggregated.append(payload)
    return aggregated


__all__ = [
    "aggregate_law_stress",
    "aggregate_learnability",
    "build_downstream_table",
]
