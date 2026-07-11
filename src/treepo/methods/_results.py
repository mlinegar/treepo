"""Per-cell ``results.json`` — the grid's one-stop readout artifact (Phase 6).

Every ``fit()`` cell writes one ``results.json`` next to its run manifest so
grid cells plug straight into the W-ledger comparison tooling. Design rules
from ``docs/label_economy_experiment_2026_07_10.md``:

* metrics are reported per split and per dimension, never pooled across
  dimensions;
* the paired normalized L1 (``mean |ŷ−y| / (b−a)``, the W1 ``R_j``) sits next
  to Pearson r, with the scale bounds and their source recorded;
* the sim-side pairing (theta regime accuracy alongside contextual MAE) is a
  first-class block — null when a cell has no sim channel, but always present
  so the pairing rule is visible in the schema;
* cost has three separate components — label cost, one-time compute, marginal
  inference — never blended into one number;
* resummary ops are recorded even when zero: the deployed pipeline never
  re-summarizes, so its C2 stratum is empty BY CONSTRUCTION and the empty cell
  must stay visible rather than read as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.methods._run_manifest import json_default

RESULTS_VERSION = "0.1"
RESULTS_FILENAME = "results.json"

#: Field mapping for W-ledger ingest: how a per-document prediction row in
#: ``prediction_records/*.jsonl`` maps onto the paired-comparison vocabulary.
PAIRED_ROW_FIELDS = {
    "key": "tree_id",
    "prediction": "prediction_scalar",
    "gold": "expert_score",
    "teacher": "teacher_score",
    "split": "split",
}


def write_results_json(
    *,
    spec: Any,
    records: Sequence[Any],
    output_dir: Path,
    status: str,
    summary: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    wall_seconds: float | None = None,
) -> Path | None:
    """Assemble and write the per-cell results artifact; return its path."""

    last = records[-1] if records else None
    payload = {
        "version": RESULTS_VERSION,
        "status": str(status),
        "cell": _cell_block(spec, summary),
        "metrics": _metrics_block(last, summary),
        "local_laws": _local_laws_block(artifacts),
        "cost": _cost_block(
            spec, records, summary, artifacts, wall_seconds=wall_seconds
        ),
        "paired_rows": {
            "files": [str(p) for p in (artifacts.get("prediction_records") or [])],
            "fields": dict(PAIRED_ROW_FIELDS),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RESULTS_FILENAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _cell_block(spec: Any, summary: Mapping[str, Any]) -> dict[str, Any]:
    grid_axes = dict(summary.get("grid_axes") or {})
    axis = dict(getattr(spec, "axis", None) or {})
    return {
        "family": str(summary.get("family") or ""),
        "schedule": str(summary.get("schedule") or ""),
        "seed": grid_axes.get("seed"),
        "axis": axis,
        "supervision": dict(summary.get("supervision") or {}),
        "grid_axes": grid_axes,
        "objective": summary.get("objective"),
        "n_iterations": summary.get("n_iterations"),
        "final_stage_label": summary.get("final_stage_label"),
    }


def _metrics_block(last: Any | None, summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = (getattr(last, "extra", None) or {}).get("prediction_rows") or []
    split_metrics = dict(summary.get("split_metrics") or {})
    by_split: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split") or "unknown")
        by_split.setdefault(split, []).append(row)
        by_split.setdefault("all", []).append(row)

    splits: dict[str, Any] = {}
    split_names = sorted(set(split_metrics) | set(by_split))
    for split in split_names:
        split_rows = by_split.get(split, [])
        recorded = dict(split_metrics.get(split) or {})
        splits[split] = {
            "n": recorded.get("n", len(split_rows)),
            "external": _paired_error_stats(
                split_rows, gold_field="expert_score", recorded=recorded, prefix="external_expert"
            ),
            "internal": _paired_error_stats(
                split_rows, gold_field="teacher_score", recorded=recorded, prefix="internal_f"
            ),
            # Per-dimension metrics carry through unpooled; pooling across
            # dimensions inflates Pearson and is banned by the schema.
            "per_dimension": recorded.get("per_dimension") or {},
            # Standing rule: theta regime accuracy is reported alongside
            # contextual MAE. Fit cells without a sim channel report nulls,
            # keeping the pairing visible in every artifact.
            "sim": {
                "theta_first_regime_accuracy": None,
                "theta_last_regime_accuracy": None,
                "contextual_mae": None,
            },
        }
    return {"pooled_across_dimensions": False, "splits": splits}


def _paired_error_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    gold_field: str,
    recorded: Mapping[str, Any],
    prefix: str,
) -> dict[str, Any]:
    paired: list[tuple[float, float]] = []
    for row in rows:
        pred = _safe_float(row.get("prediction_scalar"))
        gold = _safe_float(row.get(gold_field))
        if pred is not None and gold is not None:
            paired.append((pred, gold))
    out: dict[str, Any] = {
        "n": len(paired),
        "pearson_r": recorded.get(f"{prefix}_pearson"),
        "mae_native": recorded.get(f"{prefix}_mae"),
        "normalized_abs_error": None,
        "scale_bounds": None,
        "scale_bounds_source": None,
    }
    if not paired:
        return out
    if out["mae_native"] is None:
        out["mae_native"] = sum(abs(p - g) for p, g in paired) / len(paired)
    golds = [g for _p, g in paired]
    lo, hi = min(golds), max(golds)
    if hi > lo:
        out["normalized_abs_error"] = sum(abs(p - g) for p, g in paired) / len(paired) / (hi - lo)
        out["scale_bounds"] = [lo, hi]
        out["scale_bounds_source"] = "observed_gold_range"
    return out


def _local_laws_block(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    evidence = artifacts.get("evidence")
    if isinstance(evidence, Mapping):
        laws = evidence.get("local_laws")
        if isinstance(laws, Mapping):
            return dict(laws)
    return {"present": False, "summary": {}, "by_law_kind": {}, "source": "none"}


def _cost_block(
    spec: Any,
    records: Sequence[Any],
    summary: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    wall_seconds: float | None,
) -> dict[str, Any]:
    grid_axes = dict(summary.get("grid_axes") or {})
    doc_gold = dict(grid_axes.get("doc_gold") or {})
    mix = dict(grid_axes.get("local_label_mix") or {})
    node_supervision = _node_supervision(artifacts)

    label_source = str(mix.get("mix") or "none")
    n_leaf_rows = int(node_supervision.get("n_leaf_rows") or 0)
    n_merge_rows = int(node_supervision.get("n_merge_rows") or 0)

    last = records[-1] if records else None
    n_prediction_rows = len(
        (getattr(last, "extra", None) or {}).get("prediction_rows") or []
    )

    resummary_count = _resummary_ops(artifacts)
    return {
        # Three separate components; blending them into one number is banned.
        "label_cost": {
            "gold_doc_labels_consumed": doc_gold.get("selected_count"),
            "node_label_source": label_source,
            "gold_node_labels_consumed": (
                0 if label_source == "llm_distilled" else n_leaf_rows + n_merge_rows
            ),
            "distilled_node_labels_consumed": (
                n_leaf_rows + n_merge_rows if label_source == "llm_distilled" else 0
            ),
            "n_leaf_rows": n_leaf_rows,
            "n_merge_rows": n_merge_rows,
        },
        "one_time_compute": {
            "fit_wall_seconds": wall_seconds,
            "n_train_trees": node_supervision.get("n_trees"),
            "n_iterations": summary.get("n_iterations"),
        },
        "marginal_inference": {
            "n_eval_predictions": n_prediction_rows,
        },
        # Recorded even when zero: an empty C2 stratum is a property of the
        # cell (no resummary op exists), not a pass.
        "resummary_ops": {
            "count": resummary_count,
            "population": "observed" if resummary_count > 0 else "empty_by_construction",
        },
    }


def _node_supervision(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    g_artifact = artifacts.get("g")
    if isinstance(g_artifact, Mapping):
        block = g_artifact.get("node_supervision")
        if isinstance(block, Mapping):
            return dict(block)
    return {}


def _resummary_ops(artifacts: Mapping[str, Any]) -> int:
    for holder_key in ("statistic", "g", "f"):
        holder = artifacts.get(holder_key)
        if isinstance(holder, Mapping) and holder.get("resummary_ops") is not None:
            try:
                return int(holder["resummary_ops"])
            except (TypeError, ValueError):
                return 0
    return 0


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


__all__ = ["PAIRED_ROW_FIELDS", "RESULTS_FILENAME", "RESULTS_VERSION", "write_results_json"]
