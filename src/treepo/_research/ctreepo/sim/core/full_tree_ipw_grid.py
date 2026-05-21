from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    OPSCountSummary,
    build_markov_changepoint_ops_count_data_bundle,
    run_markov_changepoint_ops_count_experiment,
)
from treepo._research.tree.full_tree_ipw import (
    DEFAULT_LAYERED_RATE_GRID,
    DEFAULT_TRADEOFF_RATE_GRID,
    classify_layered_sampling_regime,
)


def _fmt_rate(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _get_float(payload: Mapping[str, Any], key: str) -> float:
    return float(payload.get(key, float("nan")))


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _safe_gap(value: float, reference: float) -> float:
    if not (_finite(value) and _finite(reference)):
        return float("nan")
    return float(value - reference)


def _sample_fraction_from_metrics(metrics: Mapping[str, Any]) -> float:
    population_size = float(metrics.get("population_size", 0.0))
    sampled_nodes = float(metrics.get("sampled_nodes", 0.0))
    if population_size <= 0.0:
        return float("nan")
    return float(sampled_nodes / population_size)


def _grid_semantics_payload() -> Dict[str, Any]:
    return {
        "estimand_name": "realized_full_tree_node_mean_loss",
        "population_kind": "realized_tree_nodes",
        "sampling_design": "bernoulli_realized_node_sampling",
        "propensity_field": "unit_propensity",
        "document_channel": "always_observed_document_top_loss",
        "node_channel": "sampled_realized_tree_nodes",
        "estimator_families": ["naive", "ht", "hajek"],
        "ci_semantics": "point_estimation_only",
    }


def _normalize_rate_axis(
    values: Sequence[float] | None,
    *,
    default: Sequence[float],
    label: str,
) -> List[float]:
    rates = sorted({float(x) for x in (values if values is not None else default)})
    if not rates:
        raise ValueError(f"{label} must be non-empty")
    if any(float(rate) < 0.0 or float(rate) > 1.0 for rate in rates):
        raise ValueError(f"{label} values must lie in [0, 1]")
    return rates


def _resolve_rate_axes(
    *,
    rate_axis: Sequence[float] | None,
    internal_rate_axis: Sequence[float] | None,
    leaf_rate_axis: Sequence[float] | None,
) -> tuple[List[float], List[float]]:
    shared_default = (
        list(rate_axis) if rate_axis is not None else list(DEFAULT_LAYERED_RATE_GRID)
    )
    return (
        _normalize_rate_axis(
            internal_rate_axis,
            default=shared_default,
            label="internal_rate_axis",
        ),
        _normalize_rate_axis(
            leaf_rate_axis,
            default=shared_default,
            label="leaf_rate_axis",
        ),
    )


def markov_full_tree_ipw_cell_filename(
    *,
    doc_sequence_train_fraction: float = 0.0,
    root_only_train_fraction: float,
    internal_rate: float,
    leaf_rate: float,
) -> str:
    prefix = ""
    if abs(float(doc_sequence_train_fraction)) > 1e-12:
        prefix = f"doc_sequence_{_fmt_rate(doc_sequence_train_fraction)}__"
    return (
        f"{prefix}root_only_{_fmt_rate(root_only_train_fraction)}"
        f"__internal_{_fmt_rate(internal_rate)}__leaf_{_fmt_rate(leaf_rate)}.json"
    )


def markov_full_tree_ipw_cell_path(
    *,
    output_dir: Path,
    doc_sequence_train_fraction: float = 0.0,
    root_only_train_fraction: float,
    internal_rate: float,
    leaf_rate: float,
) -> Path:
    return (
        Path(output_dir)
        / "cells"
        / markov_full_tree_ipw_cell_filename(
            doc_sequence_train_fraction=float(doc_sequence_train_fraction),
            root_only_train_fraction=float(root_only_train_fraction),
            internal_rate=float(internal_rate),
            leaf_rate=float(leaf_rate),
        )
    )


def _summary_from_json_path(path: Path) -> OPSCountSummary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return OPSCountSummary(
        config=dict(payload.get("config") or {}),
        training_geometry=dict(payload.get("training_geometry") or {}),
        objective=dict(payload.get("objective") or {}),
        metrics=dict(payload.get("metrics") or {}),
        estimator_diagnostics=dict(payload.get("estimator_diagnostics") or {}),
        local_law_learnability=dict(payload.get("local_law_learnability") or {}),
        g_artifacts=dict(payload.get("g_artifacts") or {}),
    )


def _split_surface(
    learned: Mapping[str, Any],
    *,
    prefix: str,
) -> Dict[str, Any]:
    block = dict(learned.get(f"{prefix}_full_tree_ipw") or {})
    scalar_prefix = "" if prefix == "test" else f"{prefix}_"
    return {
        "root_mae": _get_float(learned, f"{scalar_prefix}root_mae"),
        "doc_sequence_view_root_mae": _get_float(
            learned,
            f"{prefix}_doc_sequence_view_root_mae",
        ),
        "leaf_mae": _get_float(learned, f"{scalar_prefix}leaf_mae"),
        "merge_mae": _get_float(learned, f"{scalar_prefix}merge_mae"),
        "document_top_loss": _get_float(
            learned,
            f"{prefix}_document_top_loss",
        ),
        "document_top_mae": _get_float(
            learned,
            f"{prefix}_document_top_mae",
        ),
        "full_node_exact_mean_loss": _get_float(
            learned,
            f"{prefix}_full_node_exact_mean_loss",
        ),
        "sampled_node_naive_mean_loss": _get_float(
            learned,
            f"{prefix}_sampled_node_naive_mean_loss",
        ),
        "sampled_node_naive_abs_error": _get_float(
            learned,
            f"{prefix}_sampled_node_naive_abs_error",
        ),
        "sampled_node_ht_mean_loss": _get_float(
            learned,
            f"{prefix}_sampled_node_ht_mean_loss",
        ),
        "sampled_node_ht_abs_error": _get_float(
            learned,
            f"{prefix}_sampled_node_ht_abs_error",
        ),
        "sampled_node_hajek_mean_loss": _get_float(
            learned,
            f"{prefix}_sampled_node_hajek_mean_loss",
        ),
        "sampled_node_hajek_abs_error": _get_float(
            learned,
            f"{prefix}_sampled_node_hajek_abs_error",
        ),
        "effective_sample_size": _get_float(
            learned,
            f"{prefix}_full_tree_effective_sample_size",
        ),
        "max_weight": _get_float(
            learned,
            f"{prefix}_full_tree_max_weight",
        ),
        "sampled_nodes": int(learned.get(f"{prefix}_full_tree_sampled_nodes", 0)),
        "population_size": int(learned.get(f"{prefix}_full_tree_population_size", 0)),
        "document_vs_root_node_target_gap_mae": _get_float(
            block,
            "document_vs_root_node_target_gap_mae",
        ),
        "document_vs_root_node_prediction_gap_mae": _get_float(
            block,
            "document_vs_root_node_prediction_gap_mae",
        ),
        "doc_sequence_train_fraction": _get_float(learned, "doc_sequence_train_fraction"),
        "doc_sequence_train_docs_used": int(
            learned.get("doc_sequence_train_docs_used", 0)
        ),
        "depth_breakdown": dict(block.get("depth_breakdown") or {}),
        "type_breakdown": dict(block.get("type_breakdown") or {}),
    }


def extract_markov_full_tree_ipw_cell(summary: OPSCountSummary) -> Dict[str, Any]:
    learned = dict(summary.metrics.get("learned") or {})
    root_only_view_train = dict(summary.metrics.get("learned_root_only_view_train") or {})
    root_only_view_val = dict(summary.metrics.get("learned_root_only_view_val") or {})
    root_only_view_test = dict(summary.metrics.get("learned_root_only_view_test") or {})
    return {
        "train": _split_surface(learned, prefix="train"),
        "val": _split_surface(learned, prefix="val"),
        "test": _split_surface(learned, prefix="test"),
        "root_only_view_train": root_only_view_train,
        "root_only_view_val": root_only_view_val,
        "root_only_view_test": root_only_view_test,
        "epochs_completed": int(learned.get("epochs_completed", 0)),
        "training_selection_best_epoch": int(learned.get("training_selection_best_epoch", 0)),
        "training_selection_metric_name": str(
            learned.get("training_selection_metric_name", "")
        ),
        "training_selection_metric_value": _get_float(
            learned,
            "training_selection_metric_value",
        ),
        "root_only_train_fraction": _get_float(learned, "root_only_train_fraction"),
    }


def _cell_payload_from_summary(
    *,
    summary: OPSCountSummary,
    doc_sequence_train_fraction: float,
    root_only_train_fraction: float,
    internal_rate: float,
    leaf_rate: float,
    summary_path: str,
) -> Dict[str, Any]:
    extracted = extract_markov_full_tree_ipw_cell(summary)
    return {
        "doc_sequence_train_fraction": float(doc_sequence_train_fraction),
        "root_only_train_fraction": float(root_only_train_fraction),
        "p_internal": float(internal_rate),
        "p_leaf": float(leaf_rate),
        "regime": classify_layered_sampling_regime(
            leaf_rate=float(leaf_rate),
            internal_rate=float(internal_rate),
        ),
        "summary_json": str(summary_path),
        "train_metrics": extracted["train"],
        "val_metrics": extracted["val"],
        "test_metrics": extracted["test"],
        "root_only_view_train_metrics": extracted["root_only_view_train"],
        "root_only_view_val_metrics": extracted["root_only_view_val"],
        "root_only_view_test_metrics": extracted["root_only_view_test"],
        "epochs_completed": int(extracted["epochs_completed"]),
        "training_selection_best_epoch": int(
            extracted["training_selection_best_epoch"]
        ),
        "training_selection_metric_name": str(
            extracted["training_selection_metric_name"]
        ),
        "training_selection_metric_value": float(
            extracted["training_selection_metric_value"]
        ),
    }


def load_markov_full_tree_ipw_cell(path: Path) -> Dict[str, Any]:
    summary = _summary_from_json_path(path)
    config = dict(summary.config or {})
    return _cell_payload_from_summary(
        summary=summary,
        doc_sequence_train_fraction=float(config.get("doc_sequence_train_fraction", 0.0)),
        root_only_train_fraction=float(config.get("root_only_train_fraction", 0.0)),
        internal_rate=float(config.get("ipw_internal_sample_rate", 0.0)),
        leaf_rate=float(config.get("ipw_leaf_sample_rate", 0.0)),
        summary_path=str(path),
    )


def _extract_baseline_surface(
    metrics: Mapping[str, Any],
    *,
    key: str,
) -> Dict[str, Any]:
    return {
        "test": dict(metrics.get(key) or {}),
        "val": dict(metrics.get(f"{key}_val") or {}),
        "train": dict(metrics.get(f"{key}_train") or {}),
        "training": dict(metrics.get(f"{key}_training") or {}),
    }


def extract_markov_full_doc_anchor_baselines(summary: OPSCountSummary) -> Dict[str, Any]:
    metrics = dict(summary.metrics or {})
    return {
        "doc_level": _extract_baseline_surface(metrics, key="doc_level"),
        "doc_level_ridge": _extract_baseline_surface(metrics, key="doc_level_ridge"),
        "doc_sequence": _extract_baseline_surface(metrics, key="doc_sequence"),
        "rf_root": _extract_baseline_surface(metrics, key="rf_root"),
    }


def _matrix_from_cells(
    *,
    internal_rate_values: Sequence[float],
    leaf_rate_values: Sequence[float],
    cells: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> List[List[float]]:
    index = {
        (float(cell["p_internal"]), float(cell["p_leaf"])): float(
            cell["test_metrics"][metric_name]
        )
        for cell in cells
    }
    matrix: List[List[float]] = []
    for internal_rate in internal_rate_values:
        row: List[float] = []
        for leaf_rate in leaf_rate_values:
            key = (float(internal_rate), float(leaf_rate))
            if key not in index:
                raise KeyError(
                    "missing cell for "
                    f"p_internal={float(internal_rate):g}, p_leaf={float(leaf_rate):g}"
                )
            row.append(float(index[key]))
        matrix.append(row)
    return matrix


def _find_cell(
    cells: Sequence[Mapping[str, Any]],
    *,
    p_internal: float,
    p_leaf: float,
) -> Mapping[str, Any] | None:
    for cell in cells:
        if abs(float(cell["p_internal"]) - float(p_internal)) <= 1e-12 and abs(
            float(cell["p_leaf"]) - float(p_leaf)
        ) <= 1e-12:
            return cell
    return None


def _tradeoff_record_from_cell(
    cell: Mapping[str, Any],
    *,
    default_reference_root_mae: float,
    doc_only_root_mae: float,
    full_tree_root_mae: float,
    saved_root_only_tree_root_mae: float,
) -> Dict[str, Any]:
    test_metrics = dict(cell.get("test_metrics") or {})
    sample_fraction = _sample_fraction_from_metrics(test_metrics)
    root_mae = _get_float(test_metrics, "root_mae")
    improvement_over_doc_only = _safe_gap(float(doc_only_root_mae), float(root_mae))
    efficiency = (
        float(improvement_over_doc_only / sample_fraction)
        if _finite(improvement_over_doc_only) and _finite(sample_fraction) and sample_fraction > 0.0
        else float("nan")
    )
    return {
        "doc_sequence_train_fraction": float(cell.get("doc_sequence_train_fraction", 0.0)),
        "p_internal": float(cell["p_internal"]),
        "p_leaf": float(cell["p_leaf"]),
        "regime": str(cell["regime"]),
        "sample_fraction": float(sample_fraction),
        "sampled_nodes": int(test_metrics.get("sampled_nodes", 0)),
        "population_size": int(test_metrics.get("population_size", 0)),
        "root_mae": float(root_mae),
        "doc_sequence_view_root_mae": _get_float(
            test_metrics,
            "doc_sequence_view_root_mae",
        ),
        "document_top_mae": _get_float(test_metrics, "document_top_mae"),
        "document_top_loss": _get_float(test_metrics, "document_top_loss"),
        "full_node_exact_mean_loss": _get_float(test_metrics, "full_node_exact_mean_loss"),
        "sampled_node_ht_abs_error": _get_float(test_metrics, "sampled_node_ht_abs_error"),
        "sampled_node_hajek_abs_error": _get_float(test_metrics, "sampled_node_hajek_abs_error"),
        "effective_sample_size": _get_float(test_metrics, "effective_sample_size"),
        "max_weight": _get_float(test_metrics, "max_weight"),
        "root_mae_gap_to_default_reference": _safe_gap(
            float(root_mae),
            float(default_reference_root_mae),
        ),
        "root_mae_gap_to_doc_only": _safe_gap(float(root_mae), float(doc_only_root_mae)),
        "root_mae_gap_to_full_tree": _safe_gap(float(root_mae), float(full_tree_root_mae)),
        "root_mae_gap_to_saved_root_only_tree": _safe_gap(
            float(root_mae),
            float(saved_root_only_tree_root_mae),
        ),
        "improvement_over_doc_only": float(improvement_over_doc_only),
        "improvement_over_doc_only_per_sample_fraction": float(efficiency),
    }


def _pareto_frontier(
    records: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> List[Dict[str, Any]]:
    candidates = [
        dict(record)
        for record in records
        if _finite(float(record.get(x_key, float("nan"))))
        and _finite(float(record.get(y_key, float("nan"))))
    ]
    candidates.sort(
        key=lambda record: (
            float(record[x_key]),
            float(record[y_key]),
            float(record.get("p_internal", 0.0)),
            float(record.get("p_leaf", 0.0)),
        )
    )
    frontier: List[Dict[str, Any]] = []
    best_y = float("inf")
    for record in candidates:
        y_value = float(record[y_key])
        if y_value < best_y - 1e-12:
            frontier.append(record)
            best_y = y_value
    return frontier


def _best_record(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
) -> Dict[str, Any] | None:
    candidates = [
        dict(record)
        for record in records
        if _finite(float(record.get(metric_name, float("nan"))))
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda record: (
            float(record[metric_name]),
            float(record.get("sample_fraction", float("inf"))),
            float(record.get("p_internal", 0.0)),
            float(record.get("p_leaf", 0.0)),
        )
    )
    return candidates[0]


def build_markov_full_tree_ipw_tradeoff_summary(
    payload: Mapping[str, Any],
    *,
    budget_thresholds: Sequence[float] = (
        0.05,
        0.1,
        0.2,
        0.35,
        0.5,
        0.75,
        1.0,
    ),
) -> Dict[str, Any]:
    default_reference = dict(payload.get("default_reference") or {})
    full_doc_anchors = dict(payload.get("full_doc_anchors") or {})
    saved_reference_anchors = dict(payload.get("saved_reference_anchors") or {})
    if not default_reference:
        default_reference_root_mae = _get_float(
            dict(full_doc_anchors.get("doc_sequence", {}).get("test") or {}),
            "root_mae",
        )
        default_reference = {
            "name": "full_doc_doc_sequence",
            "family": "official_neuraloperator_fno",
            "root_mae": float(default_reference_root_mae),
            "source": "inferred_from_full_doc_anchor",
        }
    default_reference_root_mae = float(default_reference.get("root_mae", float("nan")))
    saved_root_only_tree_root_mae = _get_float(
        dict(saved_reference_anchors.get("root_only_tree_neural", {}).get("test") or {}),
        "root_mae",
    )
    planes = list(payload.get("planes") or [])
    if not planes:
        planes = [payload]

    plane_summaries: List[Dict[str, Any]] = []
    for plane in planes:
        cells = list(plane.get("cells") or [])
        doc_only_cell = _find_cell(cells, p_internal=0.0, p_leaf=0.0)
        full_tree_cell = _find_cell(cells, p_internal=1.0, p_leaf=1.0)
        doc_only_root_mae = _get_float(
            dict(doc_only_cell.get("test_metrics") or {}) if doc_only_cell else {},
            "root_mae",
        )
        full_tree_root_mae = _get_float(
            dict(full_tree_cell.get("test_metrics") or {}) if full_tree_cell else {},
            "root_mae",
        )
        tradeoff_records = [
            _tradeoff_record_from_cell(
                cell,
                default_reference_root_mae=float(default_reference_root_mae),
                doc_only_root_mae=float(doc_only_root_mae),
                full_tree_root_mae=float(full_tree_root_mae),
                saved_root_only_tree_root_mae=float(saved_root_only_tree_root_mae),
            )
            for cell in cells
        ]
        intermediate_records = [
            record
            for record in tradeoff_records
            if _finite(float(record["sample_fraction"]))
            and float(record["sample_fraction"]) > 0.0
            and float(record["sample_fraction"]) < 1.0
        ]
        diagonal_records = [
            record
            for record in tradeoff_records
            if abs(float(record["p_internal"]) - float(record["p_leaf"])) <= 1e-12
        ]
        best_by_budget: List[Dict[str, Any]] = []
        for threshold in budget_thresholds:
            candidates = [
                record
                for record in tradeoff_records
                if _finite(float(record["sample_fraction"]))
                and float(record["sample_fraction"]) <= float(threshold) + 1e-12
            ]
            best = _best_record(candidates, metric_name="root_mae")
            if best is not None:
                best_by_budget.append(
                    {
                        "budget_threshold": float(threshold),
                        **best,
                    }
                )
        plane_summaries.append(
            {
                "doc_sequence_train_fraction": float(
                    plane.get("doc_sequence_train_fraction", 0.0)
                ),
                "root_only_train_fraction": float(plane.get("root_only_train_fraction", 0.0)),
                "doc_only_root_mae": float(doc_only_root_mae),
                "full_tree_root_mae": float(full_tree_root_mae),
                "default_reference_root_mae": float(default_reference_root_mae),
                "saved_root_only_tree_root_mae": float(saved_root_only_tree_root_mae),
                "best_intermediate": _best_record(intermediate_records, metric_name="root_mae"),
                "best_diagonal_intermediate": _best_record(
                    [
                        record
                        for record in diagonal_records
                        if _finite(float(record["sample_fraction"]))
                        and 0.0 < float(record["sample_fraction"]) < 1.0
                    ],
                    metric_name="root_mae",
                ),
                "best_internal_heavy_intermediate": _best_record(
                    [record for record in intermediate_records if record["regime"] == "internal_heavy"],
                    metric_name="root_mae",
                ),
                "best_leaf_heavy_intermediate": _best_record(
                    [record for record in intermediate_records if record["regime"] == "leaf_heavy"],
                    metric_name="root_mae",
                ),
                "best_by_budget": best_by_budget,
                "pareto_frontier_root_mae_vs_sample_fraction": _pareto_frontier(
                    tradeoff_records,
                    x_key="sample_fraction",
                    y_key="root_mae",
                ),
                "pareto_frontier_hajek_abs_error_vs_sample_fraction": _pareto_frontier(
                    tradeoff_records,
                    x_key="sample_fraction",
                    y_key="sampled_node_hajek_abs_error",
                ),
                "diagonal_slice": diagonal_records,
            }
        )

    return {
        "simulation": "markov_full_tree_ipw_tradeoff_summary",
        "observed_token_profile": str(payload.get("observed_token_profile", "")),
        "default_reference": default_reference,
        "saved_reference_anchors": saved_reference_anchors,
        "planes": plane_summaries,
        "budget_thresholds": [float(x) for x in budget_thresholds],
        "rate_axis": [float(x) for x in list(payload.get("rate_axis") or [])],
        "doc_sequence_train_fraction_axis": [
            float(x) for x in list(payload.get("doc_sequence_train_fraction_axis") or [])
        ],
    }


def render_markov_full_tree_ipw_tradeoff_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    default_reference = dict(summary.get("default_reference") or {})
    lines.append("# Full-Tree IPW Tradeoff Summary")
    lines.append("")
    lines.append(
        f"- observed-token profile: `{str(summary.get('observed_token_profile', '') or 'custom')}`"
    )
    lines.append(
        "- default reference: "
        f"`{str(default_reference.get('name', 'unknown'))}` "
        f"(root_mae={float(default_reference.get('root_mae', float('nan'))):.6g})"
    )
    for plane in list(summary.get("planes") or []):
        lines.append("")
        lines.append(
            "## "
            f"doc_sequence_train_fraction={float(plane.get('doc_sequence_train_fraction', 0.0)):g}, "
            f"root_only_train_fraction={float(plane.get('root_only_train_fraction', 0.0)):g}"
        )
        lines.append(
            "- anchors: "
            f"doc_only={float(plane.get('doc_only_root_mae', float('nan'))):.6g}, "
            f"full_tree={float(plane.get('full_tree_root_mae', float('nan'))):.6g}"
        )
        best_intermediate = plane.get("best_intermediate")
        if isinstance(best_intermediate, Mapping):
            lines.append(
                "- best intermediate: "
                f"(p_internal={float(best_intermediate.get('p_internal', float('nan'))):g}, "
                f"p_leaf={float(best_intermediate.get('p_leaf', float('nan'))):g}, "
                f"sample_fraction={float(best_intermediate.get('sample_fraction', float('nan'))):.4f}, "
                f"root_mae={float(best_intermediate.get('root_mae', float('nan'))):.6g})"
            )
        lines.append("")
        lines.append("| budget <= sample_fraction | p_internal | p_leaf | regime | root_mae |")
        lines.append("|---:|---:|---:|---|---:|")
        for row in list(plane.get("best_by_budget") or []):
            lines.append(
                "| "
                f"{float(row.get('budget_threshold', float('nan'))):.2f} | "
                f"{float(row.get('p_internal', float('nan'))):g} | "
                f"{float(row.get('p_leaf', float('nan'))):g} | "
                f"{str(row.get('regime', ''))} | "
                f"{float(row.get('root_mae', float('nan'))):.6g} |"
            )
    return "\n".join(lines) + "\n"


def grid_rows_from_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    semantics = dict(payload.get("semantics") or _grid_semantics_payload())
    bundle_metadata = dict(payload.get("bundle_metadata") or {})
    full_doc_anchors = dict(payload.get("full_doc_anchors") or {})
    default_reference = dict(payload.get("default_reference") or {})
    saved_reference_anchors = dict(payload.get("saved_reference_anchors") or {})
    saved_root_only_tree = dict(
        saved_reference_anchors.get("root_only_tree_neural", {}).get("test") or {}
    )
    doc_level_anchor = dict(full_doc_anchors.get("doc_level", {}).get("test") or {})
    doc_level_ridge_anchor = dict(
        full_doc_anchors.get("doc_level_ridge", {}).get("test") or {}
    )
    doc_sequence_anchor = dict(full_doc_anchors.get("doc_sequence", {}).get("test") or {})
    rf_root_anchor = dict(full_doc_anchors.get("rf_root", {}).get("test") or {})
    planes = list(payload.get("planes") or [])
    if not planes:
        planes = [payload]
    for plane in planes:
        doc_sequence_train_fraction = float(
            plane.get("doc_sequence_train_fraction", 0.0)
        )
        root_only_train_fraction = float(plane.get("root_only_train_fraction", 0.0))
        for cell in list(plane.get("cells") or []):
            test_metrics = dict(cell.get("test_metrics") or {})
            root_only_view_test_metrics = dict(cell.get("root_only_view_test_metrics") or {})
            row = {
                "doc_sequence_train_fraction": float(doc_sequence_train_fraction),
                "root_only_train_fraction": float(root_only_train_fraction),
                "p_internal": float(cell["p_internal"]),
                "p_leaf": float(cell["p_leaf"]),
                "regime": str(cell["regime"]),
                "summary_json": str(cell.get("summary_json", "")),
                "estimand_name": str(
                    semantics.get("estimand_name", "realized_full_tree_node_mean_loss")
                ),
                "population_kind": str(
                    semantics.get("population_kind", "realized_tree_nodes")
                ),
                "sampling_design": str(
                    semantics.get(
                        "sampling_design",
                        "bernoulli_realized_node_sampling",
                    )
                ),
                "propensity_field": str(
                    semantics.get("propensity_field", "unit_propensity")
                ),
                "document_channel": str(
                    semantics.get(
                        "document_channel",
                        "always_observed_document_top_loss",
                    )
                ),
                "node_channel": str(
                    semantics.get("node_channel", "sampled_realized_tree_nodes")
                ),
                "estimator_families": list(
                    semantics.get("estimator_families") or ["naive", "ht", "hajek"]
                ),
                "ci_semantics": str(
                    semantics.get("ci_semantics", "point_estimation_only")
                ),
                "train_corpus_signature": str(bundle_metadata.get("train_corpus_signature", "")),
                "val_corpus_signature": str(bundle_metadata.get("val_corpus_signature", "")),
                "test_corpus_signature": str(bundle_metadata.get("test_corpus_signature", "")),
                "epochs_completed": int(cell.get("epochs_completed", 0)),
                "training_selection_best_epoch": int(
                    cell.get("training_selection_best_epoch", 0)
                ),
                "training_selection_metric_name": str(
                    cell.get("training_selection_metric_name", "")
                ),
                "training_selection_metric_value": _get_float(
                    cell,
                    "training_selection_metric_value",
                ),
                "test_root_mae": _get_float(test_metrics, "root_mae"),
                "test_doc_sequence_view_root_mae": _get_float(
                    test_metrics,
                    "doc_sequence_view_root_mae",
                ),
                "test_leaf_mae": _get_float(test_metrics, "leaf_mae"),
                "test_merge_mae": _get_float(test_metrics, "merge_mae"),
                "test_document_top_loss": _get_float(test_metrics, "document_top_loss"),
                "test_document_top_mae": _get_float(test_metrics, "document_top_mae"),
                "test_full_node_exact_mean_loss": _get_float(
                    test_metrics,
                    "full_node_exact_mean_loss",
                ),
                "test_sampled_node_naive_mean_loss": _get_float(
                    test_metrics,
                    "sampled_node_naive_mean_loss",
                ),
                "test_sampled_node_naive_abs_error": _get_float(
                    test_metrics,
                    "sampled_node_naive_abs_error",
                ),
                "test_sampled_node_ht_mean_loss": _get_float(
                    test_metrics,
                    "sampled_node_ht_mean_loss",
                ),
                "test_sampled_node_ht_abs_error": _get_float(
                    test_metrics,
                    "sampled_node_ht_abs_error",
                ),
                "test_sampled_node_hajek_mean_loss": _get_float(
                    test_metrics,
                    "sampled_node_hajek_mean_loss",
                ),
                "test_sampled_node_hajek_abs_error": _get_float(
                    test_metrics,
                    "sampled_node_hajek_abs_error",
                ),
                "test_effective_sample_size": _get_float(
                    test_metrics,
                    "effective_sample_size",
                ),
                "test_max_weight": _get_float(test_metrics, "max_weight"),
                "test_sampled_nodes": int(test_metrics.get("sampled_nodes", 0)),
                "test_sample_fraction": float(_sample_fraction_from_metrics(test_metrics)),
                "test_population_size": int(test_metrics.get("population_size", 0)),
                "test_document_vs_root_node_target_gap_mae": _get_float(
                    test_metrics,
                    "document_vs_root_node_target_gap_mae",
                ),
                "test_document_vs_root_node_prediction_gap_mae": _get_float(
                    test_metrics,
                    "document_vs_root_node_prediction_gap_mae",
                ),
                "test_root_only_view_root_mae": _get_float(
                    root_only_view_test_metrics,
                    "root_mae",
                ),
                "doc_sequence_train_docs_used": int(
                    test_metrics.get("doc_sequence_train_docs_used", 0)
                ),
                "full_doc_doc_level_root_mae": _get_float(doc_level_anchor, "root_mae"),
                "full_doc_doc_level_ridge_root_mae": _get_float(
                    doc_level_ridge_anchor,
                    "root_mae",
                ),
                "full_doc_doc_sequence_root_mae": _get_float(
                    doc_sequence_anchor,
                    "root_mae",
                ),
                "full_doc_rf_root_root_mae": _get_float(rf_root_anchor, "root_mae"),
                "default_reference_root_mae": _get_float(default_reference, "root_mae"),
                "saved_root_only_tree_root_mae": _get_float(
                    saved_root_only_tree,
                    "root_mae",
                ),
            }
            rows.append(row)
    return rows


def write_grid_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _build_plane_payload(
    *,
    internal_rate_values: Sequence[float],
    leaf_rate_values: Sequence[float],
    cells: Sequence[Mapping[str, Any]],
    doc_sequence_train_fraction: float,
    root_only_train_fraction: float,
) -> Dict[str, Any]:
    anchors: Dict[str, Mapping[str, Any]] = {}
    doc_only = _find_cell(cells, p_internal=0.0, p_leaf=0.0)
    if doc_only is not None:
        anchors["doc_only"] = doc_only
    full_tree = _find_cell(cells, p_internal=1.0, p_leaf=1.0)
    if full_tree is not None:
        anchors["full_tree"] = full_tree
    return {
        "doc_sequence_train_fraction": float(doc_sequence_train_fraction),
        "root_only_train_fraction": float(root_only_train_fraction),
        "internal_rate_axis": [float(x) for x in internal_rate_values],
        "leaf_rate_axis": [float(x) for x in leaf_rate_values],
        "cells": list(cells),
        "anchors": anchors,
        "diagonal": [
            cell
            for cell in cells
            if abs(float(cell["p_internal"]) - float(cell["p_leaf"])) <= 1e-12
        ],
        "matrices": {
            "test_root_mae": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="root_mae",
            ),
            "test_document_top_mae": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="document_top_mae",
            ),
            "test_document_top_loss": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="document_top_loss",
            ),
            "test_full_node_exact_mean_loss": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="full_node_exact_mean_loss",
            ),
            "test_sampled_node_naive_abs_error": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="sampled_node_naive_abs_error",
            ),
            "test_sampled_node_ht_abs_error": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="sampled_node_ht_abs_error",
            ),
            "test_sampled_node_hajek_abs_error": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="sampled_node_hajek_abs_error",
            ),
            "test_effective_sample_size": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="effective_sample_size",
            ),
            "test_max_weight": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="max_weight",
            ),
            "test_sampled_nodes": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="sampled_nodes",
            ),
            "test_sample_fraction": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=[
                    {
                        **cell,
                        "test_metrics": {
                            **dict(cell.get("test_metrics") or {}),
                            "sample_fraction": _sample_fraction_from_metrics(
                                dict(cell.get("test_metrics") or {})
                            ),
                        },
                    }
                    for cell in cells
                ],
                metric_name="sample_fraction",
            ),
            "test_document_vs_root_node_target_gap_mae": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=cells,
                metric_name="document_vs_root_node_target_gap_mae",
            ),
            "test_root_only_view_root_mae": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=[
                    {
                        **cell,
                        "test_metrics": dict(cell.get("root_only_view_test_metrics") or {}),
                    }
                    for cell in cells
                ],
                metric_name="root_mae",
            ),
            "test_doc_sequence_view_root_mae": _matrix_from_cells(
                internal_rate_values=internal_rate_values,
                leaf_rate_values=leaf_rate_values,
                cells=[
                    {
                        **cell,
                        "test_metrics": dict(cell.get("test_metrics") or {}),
                    }
                    for cell in cells
                ],
                metric_name="doc_sequence_view_root_mae",
            ),
        },
    }


def run_markov_full_tree_ipw_grid(
    *,
    base_config: OPSCountConfig,
    data_bundle: MarkovOPSDataBundle | None = None,
    rate_axis: Sequence[float] = DEFAULT_LAYERED_RATE_GRID,
    internal_rate_axis: Sequence[float] | None = None,
    leaf_rate_axis: Sequence[float] | None = None,
    root_only_fraction_axis: Sequence[float] = (0.0,),
    doc_sequence_train_fraction_axis: Sequence[float] = (0.0,),
    include_full_doc_anchors: bool = False,
    output_dir: Path | None = None,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    internal_rates, leaf_rates = _resolve_rate_axes(
        rate_axis=rate_axis,
        internal_rate_axis=internal_rate_axis,
        leaf_rate_axis=leaf_rate_axis,
    )
    root_only_fractions = sorted({float(x) for x in root_only_fraction_axis})
    if not root_only_fractions:
        raise ValueError("root_only_fraction_axis must be non-empty")
    if any(float(x) < 0.0 or float(x) > 1.0 for x in root_only_fractions):
        raise ValueError("root_only_fraction_axis values must lie in [0, 1]")
    doc_sequence_train_fractions = sorted(
        {float(x) for x in doc_sequence_train_fraction_axis}
    )
    if not doc_sequence_train_fractions:
        raise ValueError("doc_sequence_train_fraction_axis must be non-empty")
    if any(float(x) < 0.0 or float(x) > 1.0 for x in doc_sequence_train_fractions):
        raise ValueError("doc_sequence_train_fraction_axis values must lie in [0, 1]")

    bundle = (
        data_bundle
        if data_bundle is not None
        else build_markov_changepoint_ops_count_data_bundle(base_config)
    )
    full_doc_anchors: Dict[str, Any] = {}
    if bool(include_full_doc_anchors):
        anchor_artifact_dir = ""
        if output_dir is not None:
            anchor_artifact_dir = str(output_dir / "artifacts" / "full_doc_anchors")
        anchor_cfg = replace(
            base_config,
            include_doc_level_baseline=True,
            include_doc_level_ridge_baseline=True,
            include_doc_sequence_baseline=True,
            include_rf_root_baseline=True,
            artifact_dir=anchor_artifact_dir,
        )
        full_doc_anchors = extract_markov_full_doc_anchor_baselines(
            run_markov_changepoint_ops_count_experiment(anchor_cfg, data_bundle=bundle)
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cells").mkdir(parents=True, exist_ok=True)

    planes: List[Dict[str, Any]] = []
    for doc_sequence_train_fraction in doc_sequence_train_fractions:
        for root_only_train_fraction in root_only_fractions:
            cells: List[Dict[str, Any]] = []
            for internal_rate in internal_rates:
                for leaf_rate in leaf_rates:
                    artifact_dir = ""
                    summary_file: Path | None = None
                    if output_dir is not None:
                        artifact_dir = str(
                            output_dir
                            / "artifacts"
                            / f"doc_sequence_{_fmt_rate(doc_sequence_train_fraction)}"
                            / f"root_only_{_fmt_rate(root_only_train_fraction)}"
                            / f"internal_{_fmt_rate(internal_rate)}__leaf_{_fmt_rate(leaf_rate)}"
                        )
                        summary_file = markov_full_tree_ipw_cell_path(
                            output_dir=output_dir,
                            doc_sequence_train_fraction=float(
                                doc_sequence_train_fraction
                            ),
                            root_only_train_fraction=float(root_only_train_fraction),
                            internal_rate=float(internal_rate),
                            leaf_rate=float(leaf_rate),
                        )
                    if bool(skip_existing) and summary_file is not None and summary_file.exists():
                        cells.append(load_markov_full_tree_ipw_cell(summary_file))
                        continue
                    cfg = replace(
                        base_config,
                        local_law_objective_mode="sampled_ipw",
                        local_law_weight=(
                            float(base_config.local_law_weight)
                            if base_config.local_law_weight is not None
                            else 0.5
                        ),
                        ipw_internal_sample_rate=float(internal_rate),
                        ipw_leaf_sample_rate=float(leaf_rate),
                        root_only_train_fraction=float(root_only_train_fraction),
                        doc_sequence_train_fraction=float(doc_sequence_train_fraction),
                        artifact_dir=artifact_dir,
                    )
                    summary = run_markov_changepoint_ops_count_experiment(
                        cfg,
                        data_bundle=bundle,
                    )
                    summary_path = ""
                    if summary_file is not None:
                        summary_path = str(summary_file)
                        summary_file.write_text(summary.to_json(), encoding="utf-8")
                    cells.append(
                        _cell_payload_from_summary(
                            summary=summary,
                            doc_sequence_train_fraction=float(
                                doc_sequence_train_fraction
                            ),
                            root_only_train_fraction=float(root_only_train_fraction),
                            internal_rate=float(internal_rate),
                            leaf_rate=float(leaf_rate),
                            summary_path=summary_path,
                        )
                    )
            planes.append(
                _build_plane_payload(
                    internal_rate_values=internal_rates,
                    leaf_rate_values=leaf_rates,
                    cells=cells,
                    doc_sequence_train_fraction=float(doc_sequence_train_fraction),
                    root_only_train_fraction=float(root_only_train_fraction),
                )
            )

    payload: Dict[str, Any] = {
        "simulation": "markov_full_tree_ipw_grid",
        "semantics": _grid_semantics_payload(),
        "rate_axis": [float(x) for x in internal_rates]
        if internal_rates == leaf_rates
        else [],
        "internal_rate_axis": [float(x) for x in internal_rates],
        "leaf_rate_axis": [float(x) for x in leaf_rates],
        "root_only_fraction_axis": [float(x) for x in root_only_fractions],
        "doc_sequence_train_fraction_axis": [
            float(x) for x in doc_sequence_train_fractions
        ],
        "base_config": asdict(
            replace(
                base_config,
                local_law_objective_mode="sampled_ipw",
                local_law_weight=(
                    float(base_config.local_law_weight)
                    if base_config.local_law_weight is not None
                    else 0.5
                ),
                ipw_internal_sample_rate=float(base_config.ipw_internal_sample_rate),
                ipw_leaf_sample_rate=float(base_config.ipw_leaf_sample_rate),
                root_only_train_fraction=float(base_config.root_only_train_fraction),
                doc_sequence_train_fraction=float(
                    base_config.doc_sequence_train_fraction
                ),
            )
        ),
        "bundle_metadata": {
            "train_corpus_signature": str(bundle.train_corpus_signature),
            "val_corpus_signature": str(bundle.val_corpus_signature),
            "test_corpus_signature": str(bundle.test_corpus_signature),
            "train_docs": int(len(bundle.train_docs)),
            "val_docs": int(len(bundle.val_docs)),
            "test_docs": int(len(bundle.test_docs)),
        },
        "planes": planes,
        "full_doc_anchors": full_doc_anchors,
    }
    if len(planes) == 1:
        payload.update(planes[0])
    return payload


def load_markov_full_tree_ipw_grid_from_output_dir(
    *,
    output_dir: Path,
    base_config: OPSCountConfig,
    data_bundle: MarkovOPSDataBundle | None = None,
    rate_axis: Sequence[float] = DEFAULT_LAYERED_RATE_GRID,
    internal_rate_axis: Sequence[float] | None = None,
    leaf_rate_axis: Sequence[float] | None = None,
    root_only_fraction_axis: Sequence[float] = (0.0,),
    doc_sequence_train_fraction_axis: Sequence[float] = (0.0,),
    include_full_doc_anchors: bool = False,
) -> Dict[str, Any]:
    internal_rates, leaf_rates = _resolve_rate_axes(
        rate_axis=rate_axis,
        internal_rate_axis=internal_rate_axis,
        leaf_rate_axis=leaf_rate_axis,
    )
    root_only_fractions = sorted({float(x) for x in root_only_fraction_axis})
    if not root_only_fractions:
        raise ValueError("root_only_fraction_axis must be non-empty")
    if any(float(x) < 0.0 or float(x) > 1.0 for x in root_only_fractions):
        raise ValueError("root_only_fraction_axis values must lie in [0, 1]")
    doc_sequence_train_fractions = sorted(
        {float(x) for x in doc_sequence_train_fraction_axis}
    )
    if not doc_sequence_train_fractions:
        raise ValueError("doc_sequence_train_fraction_axis must be non-empty")
    if any(float(x) < 0.0 or float(x) > 1.0 for x in doc_sequence_train_fractions):
        raise ValueError("doc_sequence_train_fraction_axis values must lie in [0, 1]")

    bundle = (
        data_bundle
        if data_bundle is not None
        else build_markov_changepoint_ops_count_data_bundle(base_config)
    )
    full_doc_anchors: Dict[str, Any] = {}
    if bool(include_full_doc_anchors):
        anchor_artifact_dir = str(Path(output_dir) / "artifacts" / "full_doc_anchors")
        anchor_cfg = replace(
            base_config,
            include_doc_level_baseline=True,
            include_doc_level_ridge_baseline=True,
            include_doc_sequence_baseline=True,
            include_rf_root_baseline=True,
            artifact_dir=anchor_artifact_dir,
        )
        full_doc_anchors = extract_markov_full_doc_anchor_baselines(
            run_markov_changepoint_ops_count_experiment(anchor_cfg, data_bundle=bundle)
        )

    planes: List[Dict[str, Any]] = []
    for doc_sequence_train_fraction in doc_sequence_train_fractions:
        for root_only_train_fraction in root_only_fractions:
            cells: List[Dict[str, Any]] = []
            for internal_rate in internal_rates:
                for leaf_rate in leaf_rates:
                    summary_path = markov_full_tree_ipw_cell_path(
                        output_dir=Path(output_dir),
                        doc_sequence_train_fraction=float(
                            doc_sequence_train_fraction
                        ),
                        root_only_train_fraction=float(root_only_train_fraction),
                        internal_rate=float(internal_rate),
                        leaf_rate=float(leaf_rate),
                    )
                    if not summary_path.exists():
                        raise FileNotFoundError(
                            "missing cell summary while aggregating grid: "
                            f"{summary_path}"
                        )
                    cells.append(load_markov_full_tree_ipw_cell(summary_path))
            planes.append(
                _build_plane_payload(
                    internal_rate_values=internal_rates,
                    leaf_rate_values=leaf_rates,
                    cells=cells,
                    doc_sequence_train_fraction=float(doc_sequence_train_fraction),
                    root_only_train_fraction=float(root_only_train_fraction),
                )
            )

    payload: Dict[str, Any] = {
        "simulation": "markov_full_tree_ipw_grid",
        "semantics": _grid_semantics_payload(),
        "rate_axis": [float(x) for x in internal_rates]
        if internal_rates == leaf_rates
        else [],
        "internal_rate_axis": [float(x) for x in internal_rates],
        "leaf_rate_axis": [float(x) for x in leaf_rates],
        "root_only_fraction_axis": [float(x) for x in root_only_fractions],
        "doc_sequence_train_fraction_axis": [
            float(x) for x in doc_sequence_train_fractions
        ],
        "base_config": asdict(
            replace(
                base_config,
                local_law_objective_mode="sampled_ipw",
                local_law_weight=(
                    float(base_config.local_law_weight)
                    if base_config.local_law_weight is not None
                    else 0.5
                ),
                ipw_internal_sample_rate=float(base_config.ipw_internal_sample_rate),
                ipw_leaf_sample_rate=float(base_config.ipw_leaf_sample_rate),
                root_only_train_fraction=float(base_config.root_only_train_fraction),
                doc_sequence_train_fraction=float(
                    base_config.doc_sequence_train_fraction
                ),
            )
        ),
        "bundle_metadata": {
            "train_corpus_signature": str(bundle.train_corpus_signature),
            "val_corpus_signature": str(bundle.val_corpus_signature),
            "test_corpus_signature": str(bundle.test_corpus_signature),
            "train_docs": int(len(bundle.train_docs)),
            "val_docs": int(len(bundle.val_docs)),
            "test_docs": int(len(bundle.test_docs)),
        },
        "planes": planes,
        "full_doc_anchors": full_doc_anchors,
    }
    if len(planes) == 1:
        payload.update(planes[0])
    return payload


__all__ = [
    "build_markov_full_tree_ipw_tradeoff_summary",
    "extract_markov_full_doc_anchor_baselines",
    "extract_markov_full_tree_ipw_cell",
    "grid_rows_from_payload",
    "load_markov_full_tree_ipw_cell",
    "load_markov_full_tree_ipw_grid_from_output_dir",
    "markov_full_tree_ipw_cell_filename",
    "markov_full_tree_ipw_cell_path",
    "render_markov_full_tree_ipw_tradeoff_markdown",
    "run_markov_full_tree_ipw_grid",
    "write_grid_csv",
]
