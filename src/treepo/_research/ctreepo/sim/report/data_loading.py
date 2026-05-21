"""Family-dispatched data loading for unified reports.

Each family's JSON schema differs, but the loaders here produce flat dicts
with canonical keys so the report scripts don't need to know about the family.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    canonical_law_component_weights,
)
from treepo._research.ctreepo.sim.core.law_stress_common import (
    classify_law_stress,
    law_bundle_score,
)
from treepo._research.ctreepo.sim.report.family_config import FamilyReportConfig
from treepo._research.ctreepo.sim.report.pdf_utils import normalize, safe_float, safe_float_key


# ═══════════════════════════════════════════════════════════════════════════
#  Law-stress records  (used by report_law_stress.py)
# ═══════════════════════════════════════════════════════════════════════════


def load_law_stress_records(
    input_root: Path,
    family: FamilyReportConfig,
) -> Tuple[List[dict], List[dict]]:
    """Load seed_*.json → (learned_records, exact_family_records).

    Each record is a flat dict with canonical keys:
      path, run_kind, law_package, exact_family, seed,
      test_c1, test_c2, test_c3, test_spread, test_primary,
      test_bundle_score,
      val_c1, val_c2, val_c3, val_spread, val_primary,
      val_bundle_score,
      + family-specific config fields kept as-is.
    """
    files = sorted(input_root.rglob("seed_*.json"))
    if family.family == "markov_ops_count":
        return _load_markov_stress(files)
    if family.family in ("tree_relevant_lda", "tree_relevant_lda_local_law"):
        return _load_lda_stress(input_root)
    raise ValueError(f"No law-stress loader for family {family.family!r}")


def baseline_package_for(package: str) -> str:  # noqa: ARG001
    """Return the baseline package for comparison.  Always root_only."""
    return "root_only"


def assess_law_stress_rows(
    learned_records: Sequence[dict],
) -> List[dict]:
    """Match each learned record to its root_only baseline and classify."""
    # Build baseline index
    baseline_map: Dict[Tuple[object, ...], dict] = {}
    for rec in learned_records:
        key = _baseline_key(rec, baseline_package=str(rec.get("law_package", "")))
        baseline_map[key] = rec

    out: List[dict] = []
    for rec in learned_records:
        bpkg = baseline_package_for(str(rec.get("law_package", "")))
        baseline = baseline_map.get(_baseline_key(rec, baseline_package=bpkg))
        if baseline is None:
            continue
        assessment = classify_law_stress(
            baseline_c1=float(baseline["test_c1"]),
            baseline_c2=float(baseline["test_c2"]),
            baseline_c3=float(baseline["test_c3"]),
            baseline_spread=float(baseline["test_spread"]),
            baseline_root_mae=float(baseline["test_primary"]),
            selected_c1=float(rec["test_c1"]),
            selected_c2=float(rec["test_c2"]),
            selected_c3=float(rec["test_c3"]),
            selected_spread=float(rec["test_spread"]),
            selected_root_mae=float(rec["test_primary"]),
        )
        assessed = {
            **rec,
            "baseline_package": bpkg,
            "baseline_test_c1": float(baseline["test_c1"]),
            "baseline_test_c2": float(baseline["test_c2"]),
            "baseline_test_c3": float(baseline["test_c3"]),
            "baseline_test_spread": float(baseline["test_spread"]),
            "baseline_test_primary": float(baseline["test_primary"]),
            "baseline_test_bundle_score": float(baseline["test_bundle_score"]),
            **assessment.to_dict(),
        }
        out.append(assessed)
    return out


# ── baseline key (config match ignoring law_package) ─────────────────────


def _baseline_key(rec: dict, *, baseline_package: str) -> Tuple[object, ...]:
    """Tuple that identifies the config cell for baseline matching."""
    return (
        rec.get("n_regimes", rec.get("tau")),
        rec.get("train_docs"),
        rec.get("val_docs", rec.get("test_docs")),
        rec.get("audit_fraction", rec.get("law_leaf_query_rate")),
        rec.get("state_dim", rec.get("hidden_dim")),
        rec.get("n_epochs"),
        rec.get("effective_data_seed", rec.get("seed")),
        rec.get("effective_model_seed", rec.get("seed")),
        str(baseline_package),
        rec.get("fixed_leaf_tokens", 0),
        rec.get("depth_discount_gamma", 1.0),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Markov-specific loading
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_markov_law_package(payload: dict) -> str:
    cfg = dict(payload.get("config", {}) or {})
    objective = dict(payload.get("objective", {}) or {})
    learnability = dict(payload.get("local_law_learnability", {}) or {})
    metadata = dict(learnability.get("metadata", {}) or {})
    metadata_objective = dict(metadata.get("objective", {}) or {})

    for mapping in (objective, metadata, metadata_objective, cfg):
        value = str(mapping.get("law_package", "") or "").strip()
        if value:
            return value

    objective_weights = dict(
        objective.get("local_law_component_weights", {})
        or metadata_objective.get("local_law_component_weights", {})
        or {}
    )
    if objective_weights:
        c1 = safe_float_key(objective_weights, "c1", 0.0)
        c2 = safe_float_key(objective_weights, "c2", 0.0)
        c3 = safe_float_key(objective_weights, "c3", 0.0)
        proxy_weight = safe_float(
            objective.get("proxy_weight_total", metadata_objective.get("proxy_weight_total", 0.0)),
            default=0.0,
        )
        active = tuple(
            name
            for name, value in (("c1", c1), ("c2", c2), ("c3", c3))
            if abs(float(value)) > 1e-12
        )
        if not active and abs(float(proxy_weight)) <= 1e-12:
            return "root_only"
        if active == ("c1",):
            return "c1_only"
        if active == ("c2",):
            return "c2_only"
        if active == ("c3",):
            return "c3_only"
        if active == ("c1", "c3"):
            return "c1c3"
        if active == ("c1", "c2", "c3"):
            return "all_laws_plus_sched" if abs(float(proxy_weight)) > 1e-12 else "all_laws"
        if not active and abs(float(proxy_weight)) > 1e-12:
            return "sched_only"
    return "unknown"


def _read_markov_learned(path: Path, payload: dict) -> Optional[dict]:
    cfg = payload.get("config", {}) or {}
    learned = ((payload.get("metrics", {}) or {}).get("learned", {}) or {})
    if not learned:
        return None
    scale = float(max(1, int(cfg.get("max_segments", 1)) - 1))
    law_package = _resolve_markov_law_package(payload)

    def _n(key: str, *fallbacks: str) -> float:
        """Try normalised key first, then fallback raw keys with normalisation."""
        val = safe_float_key(learned, key)
        if math.isfinite(val):
            return val
        for fb in fallbacks:
            raw = safe_float_key(learned, fb)
            if math.isfinite(raw):
                return normalize(raw, scale=scale)
        return float("nan")

    test_c1 = _n("test_c1_leaf_mae_n", "test_leaf_mae", "leaf_mae")
    test_c2 = _n(
        "test_c2_count_drift_r1_mae_n",
        "test_c2_idempotence_mae_n",
        "test_c2_count_drift_r1_mae",
        "test_c2_idempotence_mae",
        "c2_count_drift_r1_mae",
        "c2_idempotence_mae",
    )
    test_c3 = _n("test_c3_merge_mae_n", "test_merge_mae", "merge_mae")
    test_spread = _n("test_schedule_spread_mean_n", "test_schedule_spread_mean", "schedule_spread_mean")
    test_root = _n("test_root_mae_n", "test_root_mae", "root_mae")
    val_c1 = _n("val_c1_leaf_mae_n", "val_leaf_mae")
    val_c2 = _n(
        "val_c2_count_drift_r1_mae_n",
        "val_c2_idempotence_mae_n",
        "val_c2_count_drift_r1_mae",
        "val_c2_idempotence_mae",
    )
    val_c3 = _n("val_c3_merge_mae_n", "val_merge_mae")
    val_spread = _n("val_schedule_spread_mean_n", "val_schedule_spread_mean")
    val_root = _n("val_root_mae_n", "val_root_mae")

    return {
        "path": str(path),
        "run_kind": "learned",
        "law_package": str(law_package),
        "exact_family": str(cfg.get("exact_family", "")),
        "seed": int(cfg.get("effective_data_seed", cfg.get("data_seed", cfg.get("seed", 0)))),
        "effective_data_seed": int(cfg.get("effective_data_seed", cfg.get("data_seed", cfg.get("seed", 0)))),
        "effective_model_seed": int(cfg.get("effective_model_seed", cfg.get("model_seed", cfg.get("seed", 0)))),
        # Config fields
        "n_regimes": int(cfg.get("n_regimes", 0)),
        "fixed_leaf_tokens": int(cfg.get("fixed_leaf_tokens", 0)),
        "train_docs": int(cfg.get("train_docs", 0)),
        "val_docs": int(cfg.get("val_docs", 0)),
        "test_docs": int(cfg.get("test_docs", 0)),
        "audit_fraction": float(cfg.get("audit_fraction", 0.0)),
        "root_weight": float(cfg.get("root_weight", 1.0)),
        "state_dim": int(cfg.get("state_dim", 0)),
        "hidden_dim": int(cfg.get("hidden_dim", 0)),
        "n_epochs": int(cfg.get("n_epochs", 0)),
        "feature_mode": str(cfg.get("feature_mode", "")),
        # Supervision & geometry fields (required for correct baseline matching)
        "depth_discount_gamma": float(cfg.get("depth_discount_gamma", 1.0)),
        "package_semantics": str(cfg.get("package_semantics", "")),
        "leaf_label_rate": float(cfg.get("leaf_label_rate", 0.0)),
        "mass_target_per_doc": safe_float(cfg.get("mass_target_per_doc")),
        "budget_total_calls_per_doc": safe_float(cfg.get("budget_total_calls_per_doc")),
        # Test metrics (canonical names)
        "test_c1": float(test_c1),
        "test_c2": float(test_c2),
        "test_c3": float(test_c3),
        "test_spread": float(test_spread),
        "test_primary": float(test_root),
        "test_bundle_score": float(safe_float(
            learned.get("test_theorem_bundle_score_n"),
            default=law_bundle_score(c1=test_c1, c2=test_c2, c3=test_c3),
        )),
        # Val metrics
        "val_c1": float(val_c1),
        "val_c2": float(val_c2),
        "val_c3": float(val_c3),
        "val_spread": float(val_spread),
        "val_primary": float(val_root),
        "val_bundle_score": float(safe_float(
            learned.get("val_theorem_bundle_score_n"),
            default=law_bundle_score(c1=val_c1, c2=val_c2, c3=val_c3),
        )),
        # Markov-specific extra metrics
        "test_c2_r4": float(_n("test_c2_r4_mae_n", "test_c2_r4_mae", "c2_r4_mae")),
        "test_resummary_root_drift_r4": float(_n(
            "test_resummary_root_drift_r4_n", "test_resummary_root_drift_r4", "resummary_root_drift_r4"
        )),
    }


def _read_markov_exact_family(path: Path, payload: dict) -> Optional[dict]:
    cfg = payload.get("config", {}) or {}
    stress = ((payload.get("metrics", {}) or {}).get("stress_family", {}) or {})
    if not stress:
        return None
    scale = float(max(1, int(cfg.get("max_segments", 1)) - 1))
    fam = str(stress.get("stress_family_name", cfg.get("exact_family", "")))

    def _n(key: str, *fallbacks: str) -> float:
        val = safe_float_key(stress, key)
        if math.isfinite(val):
            return val
        for fb in fallbacks:
            raw = safe_float_key(stress, fb)
            if math.isfinite(raw):
                return normalize(raw, scale=scale)
        return float("nan")

    test_c1 = _n("test_c1_leaf_mae_n", "leaf_mae")
    test_c2 = _n(
        "test_c2_count_drift_r1_mae_n",
        "test_c2_idempotence_mae_n",
        "c2_count_drift_r1_mae",
        "c2_idempotence_mae",
    )
    test_c3 = _n("test_c3_merge_mae_n", "merge_mae")
    test_spread = _n("test_schedule_spread_mean_n", "schedule_spread_mean")
    test_root = _n("test_root_mae_n", "root_mae")

    return {
        "path": str(path),
        "run_kind": "exact_family",
        "law_package": "",
        "exact_family": str(fam),
        "seed": int(cfg.get("effective_data_seed", cfg.get("seed", 0))),
        "effective_data_seed": int(cfg.get("effective_data_seed", cfg.get("seed", 0))),
        "effective_model_seed": int(cfg.get("effective_model_seed", cfg.get("seed", 0))),
        "n_regimes": int(cfg.get("n_regimes", 0)),
        "fixed_leaf_tokens": int(cfg.get("fixed_leaf_tokens", 0)),
        "train_docs": int(cfg.get("train_docs", 0)),
        "val_docs": int(cfg.get("val_docs", 0)),
        "test_docs": int(cfg.get("test_docs", 0)),
        "audit_fraction": float(cfg.get("audit_fraction", 0.0)),
        "root_weight": float(cfg.get("root_weight", 1.0)),
        "state_dim": int(cfg.get("state_dim", 0)),
        "hidden_dim": int(cfg.get("hidden_dim", 0)),
        "n_epochs": int(cfg.get("n_epochs", 0)),
        "feature_mode": str(cfg.get("feature_mode", "")),
        # Supervision & geometry fields (required for correct baseline matching)
        "depth_discount_gamma": float(cfg.get("depth_discount_gamma", 1.0)),
        "package_semantics": str(cfg.get("package_semantics", "")),
        "leaf_label_rate": float(cfg.get("leaf_label_rate", 0.0)),
        "mass_target_per_doc": safe_float(cfg.get("mass_target_per_doc")),
        "budget_total_calls_per_doc": safe_float(cfg.get("budget_total_calls_per_doc")),
        "test_c1": float(test_c1),
        "test_c2": float(test_c2),
        "test_c3": float(test_c3),
        "test_spread": float(test_spread),
        "test_primary": float(test_root),
        "test_bundle_score": float(safe_float(
            stress.get("test_theorem_bundle_score_n"),
            default=law_bundle_score(c1=test_c1, c2=test_c2, c3=test_c3),
        )),
        # Val metrics not available for exact families
        "val_c1": float("nan"),
        "val_c2": float("nan"),
        "val_c3": float("nan"),
        "val_spread": float("nan"),
        "val_primary": float("nan"),
        "val_bundle_score": float("nan"),
        "test_c2_r4": float(_n("test_c2_r4_mae_n", "c2_r4_mae")),
        "test_resummary_root_drift_r4": float(_n(
            "test_resummary_root_drift_r4_n", "resummary_root_drift_r4"
        )),
    }


def _load_markov_stress(files: Sequence[Path]) -> Tuple[List[dict], List[dict]]:
    learned: List[dict] = []
    exact: List[dict] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec = _read_markov_learned(path, payload)
        if rec is not None:
            learned.append(rec)
        ex = _read_markov_exact_family(path, payload)
        if ex is not None:
            exact.append(ex)
    return learned, exact


# ═══════════════════════════════════════════════════════════════════════════
#  LDA-specific loading
# ═══════════════════════════════════════════════════════════════════════════


def _nested_get(payload: dict, path: Sequence[str], default: Any = float("nan")) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_lda_law_package(
    *,
    objective: Dict[str, Any],
    summary_metadata: Dict[str, Any],
    local_law_cfg: Dict[str, Any],
    top_cfg: Dict[str, Any],
) -> str:
    for mapping, key in (
        (objective, "law_package"),
        (summary_metadata, "law_package"),
        (local_law_cfg, "law_package"),
        (top_cfg, "law_package"),
    ):
        value = str(mapping.get(key, "") or "").strip()
        if value:
            return value
    return "unknown"


def _load_lda_stress(input_root: Path) -> Tuple[List[dict], List[dict]]:
    """Load LDA law-stress records from results/*.json."""
    results_root = input_root / "results"
    if not results_root.exists():
        results_root = input_root
    learned: List[dict] = []
    exact: List[dict] = []

    for path in sorted(results_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = dict(payload.get("config", {}) or {})
        local_law = dict(payload.get("local_law", {}) or {})
        local_law_cfg = dict(local_law.get("config", {}) or {})
        learnability = dict(payload.get("local_law_learnability", {}) or {})
        summary_metadata = dict(learnability.get("metadata", {}) or {})
        objective = dict(
            local_law.get("objective", {})
            or summary_metadata.get("objective", {})
            or {}
        )
        policy_metrics = dict(local_law.get("policy_metrics", {}) or {})
        law_package = _resolve_lda_law_package(
            objective=objective,
            summary_metadata=summary_metadata,
            local_law_cfg=local_law_cfg,
            top_cfg=cfg,
        )

        # Use stabilised IPW as selected policy (best calibrator)
        selected = dict(
            policy_metrics.get("law_calibrated_ipw_stabilized", {})
            or policy_metrics.get("law_calibrated_ipw", {})
            or policy_metrics.get("infer_identity", {})
            or {}
        )
        # Baseline = identity (no law calibration)
        baseline = dict(policy_metrics.get("infer_identity", {}) or {})

        if not selected and not baseline:
            continue

        rec = {
            "path": str(path),
            "run_kind": "learned",
            "law_package": str(law_package),
            "exact_family": str(local_law_cfg.get("exact_family", "")),
            "seed": int(cfg.get("seed", 0)),
            "effective_data_seed": int(cfg.get("seed", 0)),
            "effective_model_seed": int(cfg.get("seed", 0)),
            # Config
            "tau": safe_float(cfg.get("local_mixture_concentration")),
            "quadratic_utility_weight": safe_float(cfg.get("quadratic_utility_weight", cfg.get("lambda_multiplier"))),
            "lambda_multiplier": safe_float(cfg.get("lambda_multiplier")),
            "analysis_partition_mode": str(cfg.get("analysis_partition_mode", "")),
            "train_docs": int(cfg.get("train_docs", 0)),
            "audit_fraction": safe_float(cfg.get("law_leaf_query_rate", 1.0)),
            # Canonical test metrics (from selected policy)
            "test_c1": safe_float(selected.get("mean_c1")),
            "test_c2": safe_float(selected.get("mean_c2_proxy")),
            "test_c3": safe_float(selected.get("mean_c3")),
            "test_spread": safe_float(selected.get("schedule_spread", 0.0)),
            "test_primary": safe_float(selected.get("mean_aux_oracle_target_abs_error")),
            "test_bundle_score": float(
                safe_float(selected.get("mean_c1"), 0.0)
                + safe_float(selected.get("mean_c2_proxy"), 0.0)
                + safe_float(selected.get("mean_c3"), 0.0)
            ),
            # Baseline metrics (identity = no law calibration)
            "val_c1": safe_float(baseline.get("mean_c1")),
            "val_c2": safe_float(baseline.get("mean_c2_proxy")),
            "val_c3": safe_float(baseline.get("mean_c3")),
            "val_spread": safe_float(baseline.get("schedule_spread", 0.0)),
            "val_primary": safe_float(baseline.get("mean_aux_oracle_target_abs_error")),
            "val_bundle_score": float(
                safe_float(baseline.get("mean_c1"), 0.0)
                + safe_float(baseline.get("mean_c2_proxy"), 0.0)
                + safe_float(baseline.get("mean_c3"), 0.0)
            ),
        }
        learned.append(rec)

        # Exact family counterexamples
        for fam_key in ("exact_oracle", "exact_scrambled_topics", "exact_uniform_prior", "exact_adversarial_merge"):
            fam_pm = dict(policy_metrics.get(fam_key, {}) or {})
            if fam_pm:
                ex = {
                    "path": str(path),
                    "run_kind": "exact_family",
                    "law_package": "",
                    "exact_family": fam_key.replace("exact_", ""),
                    "seed": int(cfg.get("seed", 0)),
                    "effective_data_seed": int(cfg.get("seed", 0)),
                    "effective_model_seed": int(cfg.get("seed", 0)),
                    "tau": safe_float(cfg.get("local_mixture_concentration")),
                    "train_docs": int(cfg.get("train_docs", 0)),
                    "test_c1": safe_float(fam_pm.get("mean_c1")),
                    "test_c2": safe_float(fam_pm.get("mean_c2_proxy")),
                    "test_c3": safe_float(fam_pm.get("mean_c3")),
                    "test_spread": 0.0,
                    "test_primary": safe_float(fam_pm.get("mean_aux_oracle_target_abs_error")),
                    "test_bundle_score": float(
                        safe_float(fam_pm.get("mean_c1"), 0.0)
                        + safe_float(fam_pm.get("mean_c2_proxy"), 0.0)
                        + safe_float(fam_pm.get("mean_c3"), 0.0)
                    ),
                    "val_c1": float("nan"),
                    "val_c2": float("nan"),
                    "val_c3": float("nan"),
                    "val_spread": float("nan"),
                    "val_primary": float("nan"),
                    "val_bundle_score": float("nan"),
                }
                exact.append(ex)

    return learned, exact


# ═══════════════════════════════════════════════════════════════════════════
#  Learnability records  (used by report_learnability.py)
# ═══════════════════════════════════════════════════════════════════════════

THEOREM_SCORE_SPREAD_WEIGHT = 0.25


def _law_score(*, leaf: float, merge: float, spread: float) -> float:
    """Held-out theorem-facing score: leaf + merge + 0.25 * spread.

    Root MAE is intentionally excluded — it is the primary downstream metric
    and is reported separately.
    """
    return float(leaf + merge + THEOREM_SCORE_SPREAD_WEIGHT * spread)


def load_learnability_records(
    input_root: Path,
    family: FamilyReportConfig,
    *,
    do_normalize: bool = True,
) -> List[dict]:
    """Load seed_*.json → flat dicts with canonical learnability keys.

    Each dict has:
      path, seed, sweep_value (local_law_weight / tau / ...),
      learned_root_mae_n, learned_leaf_mae_n, learned_merge_mae_n,
      learned_spread_n, learned_law_score_n,
      train_root_mae_n, train_leaf_mae_n, train_merge_mae_n,
      train_spread_n, train_law_score_n,
      generalization_gap_*, exact_*, unders_*,
      heldout_objective_for_report, train_objective_for_report,
      + family-specific config fields.
    """
    files = sorted(input_root.rglob("seed_*.json"))
    if family.family == "markov_ops_count":
        return _load_markov_learnability(files, do_normalize=do_normalize)
    if family.family in ("tree_relevant_lda", "tree_relevant_lda_local_law"):
        return _load_lda_learnability(files, family=family, do_normalize=do_normalize)
    raise ValueError(f"No learnability loader for family {family.family!r}")


# ── Markov learnability loading ──────────────────────────────────────────


def _split_objective_metric_with_fallback(
    learned: dict,
    *,
    split: str,
    fallback_keys: Sequence[str],
    theorem_fallback: float,
) -> float:
    """Try exact weighted objective, then unweighted, then theorem proxy."""
    import numpy as np

    selection_metric_name = str(learned.get(f"{split}_objective_selection_metric_name", "") or "")
    if selection_metric_name:
        direct_key = f"{split}_{selection_metric_name}"
        direct_value = safe_float_key(learned, direct_key)
        if np.isfinite(direct_value):
            return float(direct_value)
        selected_value = safe_float_key(learned, f"{split}_objective_selection_metric_value")
        if np.isfinite(selected_value):
            return float(selected_value)
    for key in fallback_keys:
        value = safe_float_key(learned, str(key))
        if np.isfinite(value):
            return float(value)
    return float(theorem_fallback)


def _load_markov_learnability(
    files: Sequence[Path],
    *,
    do_normalize: bool,
) -> List[dict]:
    import numpy as np

    rows: List[dict] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        learned = metrics.get("learned", {}) or {}
        learned_train = metrics.get("learned_train", {}) or {}
        exact = metrics.get("exact", {}) or {}
        unders = metrics.get("undersupported", {}) or {}

        train_docs = int(cfg.get("train_docs", -1))
        audit_fraction = float(cfg.get("audit_fraction", float("nan")))
        max_segments = int(cfg.get("max_segments", -1))
        count_scale = float(max(1, max_segments - 1)) if max_segments > 0 else float("nan")
        if train_docs <= 0 or not np.isfinite(audit_fraction) or not np.isfinite(count_scale) or count_scale <= 0.0:
            continue

        def _norm(x: float) -> float:
            return float(x) / float(count_scale) if do_normalize else float(x)

        learned_root = _norm(safe_float_key(learned, "root_mae"))
        learned_leaf = _norm(safe_float_key(learned, "leaf_mae"))
        learned_merge = _norm(safe_float_key(learned, "merge_mae"))
        learned_spread = _norm(safe_float_key(learned, "schedule_spread_mean"))
        train_root = _norm(safe_float_key(learned_train, "root_mae"))
        train_leaf = _norm(safe_float_key(learned_train, "leaf_mae"))
        train_merge = _norm(safe_float_key(learned_train, "merge_mae"))
        train_spread = _norm(safe_float_key(learned_train, "schedule_spread_mean"))
        learned_law_score = _law_score(leaf=learned_leaf, merge=learned_merge, spread=learned_spread)
        train_law_score = _law_score(leaf=train_leaf, merge=train_merge, spread=train_spread)

        test_obj = safe_float_key(learned, "test_objective_full_labels")
        train_obj = safe_float_key(learned, "train_objective_full_labels")
        test_unw = safe_float_key(learned, "test_unweighted_objective_full_labels")
        train_unw = safe_float_key(learned, "train_unweighted_objective_full_labels")

        heldout_obj = _split_objective_metric_with_fallback(
            learned, split="test",
            fallback_keys=("test_objective_full_labels", "test_unweighted_objective_full_labels"),
            theorem_fallback=learned_law_score,
        )
        train_obj_report = _split_objective_metric_with_fallback(
            learned, split="train",
            fallback_keys=("train_objective_full_labels", "train_unweighted_objective_full_labels"),
            theorem_fallback=train_law_score,
        )

        objective = payload.get("objective", {}) or {}
        objective_metadata = dict(objective.get("metadata", {}) or {})
        legacy_objective_fields = sorted(
            field
            for field in (
                "lambda_local",
                "selected_lambda_local",
                "task_objective_weight",
                "configured_task_objective_weight",
                "optimization_root_weight",
                "root_weight",
                "law_package",
                "local_law_weights",
                "leaf_weight",
                "c1_weight",
                "c2_weight",
                "c3_weight",
            )
            if field in objective
        )
        if legacy_objective_fields:
            raise ValueError(
                f"{path} uses legacy objective report fields: "
                + ", ".join(legacy_objective_fields)
                + ". Regenerate with local_law_weight/root_share."
            )
        if "local_law_weight" not in objective and "local_law_weight" not in cfg:
            raise ValueError(f"{path} missing canonical local_law_weight")
        local_law_weight = float(objective.get("local_law_weight", cfg.get("local_law_weight", 0.0)))
        root_share = safe_float_key(objective, "root_share")
        component_weights = canonical_law_component_weights(
            dict(objective.get("local_law_component_weights", {}) or {}),
            allow_aliases=True,
        )
        local_law_leaf_weight = float(component_weights.get(LAW_ID_LEAF_PRESERVATION, 0.0))
        local_law_idempotence_weight = float(component_weights.get(LAW_ID_ON_RANGE_IDEMPOTENCE, 0.0))
        local_law_merge_weight = float(component_weights.get(LAW_ID_MERGE_PRESERVATION, 0.0))
        if "law_set_id" not in objective and "law_set_id" not in cfg:
            if "law_set_id" not in objective_metadata:
                raise ValueError(f"{path} missing canonical law_set_id")
        law_set_id = str(
            objective.get("law_set_id")
            or objective_metadata.get("law_set_id")
            or cfg.get("law_set_id")
        )
        method_id = str(
            objective.get("method_id")
            or objective_metadata.get("method_id")
            or cfg.get("method_id")
            or payload.get("method_id")
            or ""
        )
        if not method_id:
            raise ValueError(f"{path} missing canonical method_id")

        rows.append({
            "path": str(path),
            "method_id": method_id,
            "seed": int(cfg.get("effective_data_seed", cfg.get("seed", 0))),
            "effective_data_seed": int(cfg.get("effective_data_seed", cfg.get("seed", 0))),
            "effective_model_seed": int(cfg.get("effective_model_seed", cfg.get("seed", 0))),
            # Sweep variable
            "sweep_value": float(local_law_weight),
            "local_law_weight": float(local_law_weight),
            # Config fields
            "train_docs": int(train_docs),
            "audit_fraction": float(audit_fraction),
            "law_set_id": str(law_set_id),
            "schedule_consistency_weight": float(cfg.get("schedule_consistency_weight", 0.0)),
            "root_share": float(root_share),
            "state_dim": int(cfg.get("state_dim", 0)),
            "hidden_dim": int(cfg.get("hidden_dim", 0)),
            "n_epochs": int(cfg.get("n_epochs", 0)),
            "feature_mode": str(cfg.get("feature_mode", "")),
            "c3_audit_strategy": str(cfg.get("c3_audit_strategy", "")),
            # Supervision & geometry fields
            "fixed_leaf_tokens": int(cfg.get("fixed_leaf_tokens", 0)),
            "depth_discount_gamma": float(cfg.get("depth_discount_gamma", 1.0)),
            "package_semantics": str(cfg.get("package_semantics", "")),
            "leaf_label_rate": float(cfg.get("leaf_label_rate", 0.0)),
            "objective_weighting_scheme": str(objective.get("weighting_scheme", "") or ""),
            "root_share_source": str(objective.get("root_share_source", "") or ""),
            "objective_local_law_component_weights": {
                str(key): float(value)
                for key, value in dict(component_weights).items()
            },
            # Learned held-out metrics (canonical names)
            "learned_root_mae_n": float(learned_root),
            "learned_leaf_mae_n": float(learned_leaf),
            "learned_merge_mae_n": float(learned_merge),
            "learned_spread_n": float(learned_spread),
            "learned_law_score_n": float(learned_law_score),
            # Train metrics
            "train_root_mae_n": float(train_root),
            "train_leaf_mae_n": float(train_leaf),
            "train_merge_mae_n": float(train_merge),
            "train_spread_n": float(train_spread),
            "train_law_score_n": float(train_law_score),
            # Generalization gaps
            "generalization_gap_root_mae_n": float(_norm(safe_float_key(learned, "generalization_gap_root_mae"))),
            "generalization_gap_leaf_mae_n": float(_norm(safe_float_key(learned, "generalization_gap_leaf_mae"))),
            "generalization_gap_merge_mae_n": float(_norm(safe_float_key(learned, "generalization_gap_merge_mae"))),
            "generalization_gap_spread_n": float(_norm(safe_float_key(learned, "generalization_gap_schedule_spread_mean"))),
            "generalization_gap_law_score_n": float(learned_law_score - train_law_score),
            # Exact / undersupported baselines
            "exact_root_mae_n": float(_norm(safe_float_key(exact, "root_mae"))),
            "exact_leaf_mae_n": float(_norm(safe_float_key(exact, "leaf_mae"))),
            "exact_merge_mae_n": float(_norm(safe_float_key(exact, "merge_mae"))),
            "exact_spread_n": float(_norm(safe_float_key(exact, "schedule_spread_mean"))),
            "unders_root_mae_n": float(_norm(safe_float_key(unders, "root_mae"))),
            "unders_leaf_mae_n": float(_norm(safe_float_key(unders, "leaf_mae"))),
            "unders_merge_mae_n": float(_norm(safe_float_key(unders, "merge_mae"))),
            "unders_spread_n": float(_norm(safe_float_key(unders, "schedule_spread_mean"))),
            # Violation rates
            "learned_leaf_violation_rate": float(safe_float_key(learned, "leaf_violation_rate", 0.0)),
            "learned_merge_violation_rate": float(safe_float_key(learned, "merge_violation_rate", 0.0)),
            # Objective metrics
            "test_objective_full_labels": float(test_obj),
            "train_objective_full_labels": float(train_obj),
            "test_unweighted_objective_full_labels": float(test_unw),
            "train_unweighted_objective_full_labels": float(train_unw),
            "heldout_objective_for_report": float(heldout_obj),
            "train_objective_for_report": float(train_obj_report),
            "generalization_gap_objective_for_report": float(heldout_obj - train_obj_report) if np.isfinite(heldout_obj) and np.isfinite(train_obj_report) else float("nan"),
            "train_loss_final": float(safe_float_key(learned, "train_loss_final")),
        })
    return rows


# ── LDA learnability loading ─────────────────────────────────────────────


def _load_lda_learnability(
    files: Sequence[Path],
    *,
    family: FamilyReportConfig,
    do_normalize: bool,
) -> List[dict]:
    """Load LDA learnability records.

    LDA sweeps tau (mixture concentration) as the main x-axis variable.
    """
    rows: List[dict] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = payload.get("config", {}) or {}
        local_law = payload.get("local_law", {}) or {}
        local_law_cfg = dict(local_law.get("config", {}) or {})
        local_law_learnability = payload.get("local_law_learnability", {}) or {}
        learnability_meta = dict(local_law_learnability.get("metadata", {}) or {})
        resolved_weights = dict(learnability_meta.get("resolved_local_law_weights", {}) or {})
        objective = payload.get("objective", {}) or {}
        policy_metrics = dict(local_law.get("policy_metrics", {}) or {})

        # Use stabilised IPW as selected policy
        selected = dict(
            policy_metrics.get("law_calibrated_ipw_stabilized", {})
            or policy_metrics.get("law_calibrated_ipw", {})
            or policy_metrics.get("infer_identity", {})
            or {}
        )
        baseline = dict(policy_metrics.get("infer_identity", {}) or {})
        if not selected:
            continue

        tau = safe_float(cfg.get("local_mixture_concentration"))
        lambda_mult = safe_float(cfg.get("quadratic_utility_weight", cfg.get("lambda_multiplier", 1.0)))
        train_docs = int(cfg.get("train_docs", 0))
        law_package = str(
            learnability_meta.get(
                "law_package",
                local_law_cfg.get("law_package", cfg.get("law_package", "")),
            )
            or ""
        ).strip().lower()

        sel_c1 = safe_float(selected.get("mean_c1"))
        sel_c2 = safe_float(selected.get("mean_c2_proxy"))
        sel_c3 = safe_float(selected.get("mean_c3"))
        sel_spread = safe_float(selected.get("schedule_spread", 0.0))
        sel_primary = safe_float(selected.get("mean_aux_oracle_target_abs_error"))
        sel_law_score = _law_score(leaf=sel_c1, merge=sel_c3, spread=sel_spread)

        base_c1 = safe_float(baseline.get("mean_c1"))
        base_c2 = safe_float(baseline.get("mean_c2_proxy"))
        base_c3 = safe_float(baseline.get("mean_c3"))
        base_spread = safe_float(baseline.get("schedule_spread", 0.0))
        base_primary = safe_float(baseline.get("mean_aux_oracle_target_abs_error"))
        base_law_score = _law_score(leaf=base_c1, merge=base_c3, spread=base_spread)

        selection_metric_name = str(objective.get("selection_metric_name", "") or "").strip()
        if selection_metric_name:
            heldout_obj = safe_float(selected.get(selection_metric_name))
            train_obj = safe_float(baseline.get(selection_metric_name))
        else:
            heldout_obj = float("nan")
            train_obj = float("nan")
        if not math.isfinite(float(heldout_obj)):
            heldout_obj = float(sel_primary)
            selection_metric_name = "mean_aux_oracle_target_abs_error"
        if not math.isfinite(float(train_obj)):
            train_obj = float(base_primary)

        rows.append({
            "path": str(path),
            "seed": int(cfg.get("seed", 0)),
            "effective_data_seed": int(cfg.get("seed", 0)),
            "effective_model_seed": int(cfg.get("seed", 0)),
            # Sweep variable
            "sweep_value": float(tau),
            "tau": float(tau),
            "quadratic_utility_weight": float(lambda_mult),
            "lambda_multiplier": float(lambda_mult),
            # Config
            "train_docs": int(train_docs),
            "audit_fraction": safe_float(cfg.get("law_leaf_query_rate", 1.0)),
            "analysis_partition_mode": str(cfg.get("analysis_partition_mode", "")),
            "law_package": str(law_package or "unknown"),
            "objective_weighting_scheme": str(objective.get("weighting_scheme", "") or ""),
            "objective_lambda_interpretation": str(objective.get("interprets_lambda_as", "") or ""),
            "objective_selection_metric_name": str(selection_metric_name),
            "objective_local_law_weight_c1": safe_float(
                resolved_weights.get(
                    "c1",
                    local_law_cfg.get("law_c1_weight", cfg.get("law_c1_weight")),
                )
            ),
            "objective_local_law_weight_c2_proxy": safe_float(
                resolved_weights.get(
                    "c2_proxy",
                    local_law_cfg.get("law_c2_proxy_weight", cfg.get("law_c2_proxy_weight")),
                )
            ),
            "objective_local_law_weight_c3": safe_float(
                resolved_weights.get(
                    "c3",
                    local_law_cfg.get("law_c3_weight", cfg.get("law_c3_weight")),
                )
            ),
            # Use canonical learned_* naming (matches Markov)
            "learned_root_mae_n": float(sel_primary),
            "learned_leaf_mae_n": float(sel_c1),
            "learned_merge_mae_n": float(sel_c3),
            "learned_spread_n": float(sel_spread),
            "learned_law_score_n": float(sel_law_score),
            "learned_c2_n": float(sel_c2),
            # Baseline as "train" equivalent for gap computation
            "train_root_mae_n": float(base_primary),
            "train_leaf_mae_n": float(base_c1),
            "train_merge_mae_n": float(base_c3),
            "train_spread_n": float(base_spread),
            "train_law_score_n": float(base_law_score),
            # Gaps
            "generalization_gap_root_mae_n": float(sel_primary - base_primary),
            "generalization_gap_leaf_mae_n": float(sel_c1 - base_c1),
            "generalization_gap_merge_mae_n": float(sel_c3 - base_c3),
            "generalization_gap_spread_n": float(sel_spread - base_spread),
            "generalization_gap_law_score_n": float(sel_law_score - base_law_score),
            # Objective: prefer the run's configured selection metric, else downstream utility error.
            "heldout_objective_for_report": float(heldout_obj),
            "train_objective_for_report": float(train_obj),
            "generalization_gap_objective_for_report": float(heldout_obj - train_obj),
            "train_loss_final": float("nan"),
        })
    return rows


__all__ = [
    "assess_law_stress_rows",
    "baseline_package_for",
    "load_law_stress_records",
    "load_learnability_records",
]
