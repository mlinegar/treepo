#!/usr/bin/env python3
"""Manifest-first report for the observed-token Markov comparison suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from treepo._research.ctreepo.sim.manifest import read_manifest_jsonl
from treepo._research.ctreepo.sim.suite.common import read_suite_meta, resolve_grouped_suite_paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an observed-token Markov comparison report.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(list(argv) if argv is not None else None)


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _suite_summary_paths(output_root: Path) -> tuple[List[Path], Dict[str, object]]:
    paths = resolve_grouped_suite_paths(output_root.resolve())
    if not paths.suite_meta.exists():
        return [], {}
    meta = read_suite_meta(paths.suite_meta)
    manifest_paths: List[Path] = []
    selected_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
    group_manifest_files = dict(meta.get("group_manifest_files", {}) or {})
    for group in selected_groups:
        path = Path(str(group_manifest_files.get(group, "")))
        if path.exists():
            manifest_paths.append(path)
    if not manifest_paths and paths.suite_manifest.exists():
        manifest_paths = [paths.suite_manifest]

    summary_paths: List[Path] = []
    for manifest_path in manifest_paths:
        for run in read_manifest_jsonl(manifest_path):
            out_path = Path(str(run.outputs.get("json_summary", "")))
            if out_path.exists():
                summary_paths.append(out_path)
    deduped = sorted({path.resolve() for path in summary_paths})
    return deduped, meta


def _fallback_summary_paths(output_root: Path) -> List[Path]:
    return sorted((output_root / "markov_changepoint_ops_count").rglob("seed_*.json"))


def _group_payloads(output_root: Path) -> tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    summary_paths, meta = _suite_summary_paths(output_root)
    if not summary_paths:
        summary_paths = _fallback_summary_paths(output_root)
    payloads: Dict[str, Dict[str, object]] = {}
    for path in summary_paths:
        payload = _load_json(path)
        if not payload:
            continue
        config = dict(payload.get("config", {}) or {})
        group_key = str(config.get("comparison_mode", path.parent.name)).strip()
        payloads[group_key] = payload
    return payloads, meta


def _root_mae(payload: Mapping[str, object], key: str) -> float:
    metrics = dict(payload.get("metrics", {}) or {})
    item = dict(metrics.get(key, {}) or {})
    return float(item.get("root_mae", float("nan")))


def _schedule_spread(payload: Mapping[str, object], key: str) -> float:
    metrics = dict(payload.get("metrics", {}) or {})
    item = dict(metrics.get(key, {}) or {})
    return float(item.get("schedule_spread_mean", float("nan")))


def _check(name: str, passed: bool, value: object, description: str) -> Dict[str, object]:
    return {
        "name": str(name),
        "pass": bool(passed),
        "value": value,
        "description": str(description),
    }


def _metric_value(payload: Mapping[str, object], key: str, metric_name: str) -> float:
    metrics = dict(payload.get("metrics", {}) or {})
    item = dict(metrics.get(key, {}) or {})
    return float(item.get(metric_name, float("nan")))


def _format_metric(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    if abs(float(value)) >= 1e-3:
        return f"{float(value):.6f}"
    return f"{float(value):.6g}"


def _markdown_table(
    payload: Mapping[str, object],
    *,
    methods: Sequence[tuple[str, str]],
) -> List[str]:
    lines = [
        "| Method | Root MAE | Leaf MAE | Schedule Spread |",
        "|---|---:|---:|---:|",
    ]
    for key, label in methods:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(label),
                    _format_metric(_metric_value(payload, key, "root_mae")),
                    _format_metric(_metric_value(payload, key, "leaf_mae")),
                    _format_metric(_metric_value(payload, key, "schedule_spread_mean")),
                ]
            )
            + " |"
        )
    return lines


def _bar_panel(
    ax,
    *,
    title: str,
    payload: Mapping[str, object],
    methods: Sequence[tuple[str, str]],
) -> None:
    labels: List[str] = []
    values: List[float] = []
    for key, label in methods:
        value = _root_mae(payload, key)
        if not math.isfinite(value):
            continue
        labels.append(str(label))
        values.append(float(value))
    ypos = list(range(len(labels)))
    ax.barh(ypos, values, color="tab:blue", alpha=0.85)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Root MAE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        [
            "pandoc",
            str(md_path.name),
            "-o",
            str(pdf_path.name),
            "--pdf-engine=pdflatex",
        ],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = args.output_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (output_root / "figures" / "markov_observed_token")
    )
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    payloads, meta = _group_payloads(output_root)
    root_only = payloads.get("root_only")
    local_labels = payloads.get("local_labels")
    if root_only is None:
        raise SystemExit(f"expected at least a root_only summary under {output_root}")
    subset_mode = "paired" if local_labels is not None else "root_only_only"

    root_cfg = dict(root_only.get("config", {}) or {})
    root_metrics = dict(root_only.get("metrics", {}) or {})
    local_cfg = dict(local_labels.get("config", {}) or {}) if local_labels is not None else {}
    local_metrics = dict(local_labels.get("metrics", {}) or {}) if local_labels is not None else {}
    root_doc_sequence_training = dict(root_metrics.get("doc_sequence_training", {}) or {})
    root_doc_transformer_training = dict(root_metrics.get("doc_transformer_training", {}) or {})

    split_signatures = {
        "root_only": {
            "train": str(root_cfg.get("train_corpus_signature", "")),
            "val": str(root_cfg.get("val_corpus_signature", "")),
            "test": str(root_cfg.get("test_corpus_signature", "")),
        }
    }
    if local_labels is not None:
        split_signatures["local_labels"] = {
            "train": str(local_cfg.get("train_corpus_signature", "")),
            "val": str(local_cfg.get("val_corpus_signature", "")),
            "test": str(local_cfg.get("test_corpus_signature", "")),
        }
    signatures_match = (
        split_signatures["root_only"] == split_signatures.get("local_labels", split_signatures["root_only"])
    )
    root_learned = _root_mae(root_only, "learned")
    local_learned = _root_mae(local_labels, "learned") if local_labels is not None else float("nan")
    best_local_simple = (
        min(
            [
                _root_mae(local_labels, "leaf_ridge_tree"),
                _root_mae(local_labels, "leaf_endpoint_table_tree"),
                _root_mae(local_labels, "leaf_dt_tree"),
                _root_mae(local_labels, "leaf_knn_tree"),
                _root_mae(local_labels, "leaf_rf_tree"),
            ]
        )
        if local_labels is not None
        else float("nan")
    )

    checks = {
        "subset_mode": _check(
            "subset_mode",
            True,
            subset_mode,
            "Report subset used for this observed-token suite rendering.",
        ),
        "matching_split_signatures": _check(
            "matching_split_signatures",
            signatures_match,
            split_signatures,
            "Available runs use the same fixed train/val/test bundle.",
        ),
        "token_only_features": _check(
            "token_only_features",
            str(root_cfg.get("feature_mode")) == "token_full"
            and (
                local_labels is None
                or str(local_cfg.get("feature_mode")) == "token_full"
            ),
            {
                "root_only": root_cfg.get("feature_mode"),
                **(
                    {"local_labels": local_cfg.get("feature_mode")}
                    if local_labels is not None
                    else {}
                ),
            },
            "All available runs use observed-token features only; latent regime labels are not exposed.",
        ),
        "root_only_has_no_local_labels": _check(
            "root_only_has_no_local_labels",
            abs(float(root_cfg.get("audit_fraction", float("nan")))) <= 1e-12
            and abs(float(root_cfg.get("leaf_query_rate", float("nan")))) <= 1e-12,
            {
                "audit_fraction": root_cfg.get("audit_fraction"),
                "leaf_query_rate": root_cfg.get("leaf_query_rate"),
            },
            "The root-only run uses document/root supervision only.",
        ),
        "local_labels_enabled": _check(
            "local_labels_enabled",
            abs(float(local_cfg.get("audit_fraction", float("nan"))) - 1.0) <= 1e-12
            and abs(float(local_cfg.get("leaf_query_rate", float("nan"))) - 1.0) <= 1e-12,
            {
                "audit_fraction": local_cfg.get("audit_fraction"),
                "leaf_query_rate": local_cfg.get("leaf_query_rate"),
            },
            "The local-label run enables sampled leaf and internal-node labels on the fixed bundle.",
        ),
        "root_only_learning_is_finite": _check(
            "root_only_learning_is_finite",
            math.isfinite(root_learned),
            root_learned,
            "The tree learner reports a finite root error in the observed-token root-only setting.",
        ),
        "full_doc_sequence_learning_is_finite": _check(
            "full_doc_sequence_learning_is_finite",
            math.isfinite(_root_mae(root_only, "doc_sequence")),
            _root_mae(root_only, "doc_sequence"),
            "The matched full-document neuraloperator FNO baseline learns from the raw observed-token sequence.",
        ),
        "full_doc_transformer_learning_is_finite": _check(
            "full_doc_transformer_learning_is_finite",
            math.isfinite(_root_mae(root_only, "doc_transformer")),
            _root_mae(root_only, "doc_transformer"),
            "The matched full-document transformer learns from the raw observed-token sequence.",
        ),
        "full_doc_neural_inputs_match_exactly": _check(
            "full_doc_neural_inputs_match_exactly",
            dict(root_doc_sequence_training.get("sequence_input_signatures", {}) or {})
            == dict(root_doc_transformer_training.get("sequence_input_signatures", {}) or {})
            and str(root_doc_sequence_training.get("sequence_input_backend", "")) == "shared_token_sequence_arrays"
            and str(root_doc_transformer_training.get("sequence_input_backend", "")) == "shared_token_sequence_arrays",
            {
                "doc_sequence": dict(root_doc_sequence_training.get("sequence_input_signatures", {}) or {}),
                "doc_transformer": dict(root_doc_transformer_training.get("sequence_input_signatures", {}) or {}),
            },
            "The operator and transformer consume the exact same padded full-document token sequences and masks.",
        ),
        "pooled_full_doc_learning_is_finite": _check(
            "pooled_full_doc_learning_is_finite",
            math.isfinite(_root_mae(root_only, "doc_level")),
            _root_mae(root_only, "doc_level"),
            "The pooled full-document MLP baseline also learns from observed-token summary features.",
        ),
        "nondegenerate_root_target": _check(
            "nondegenerate_root_target",
            not bool(root_cfg.get("degenerate_root_target_detected", False))
            and (
                local_labels is None
                or not bool(local_cfg.get("degenerate_root_target_detected", False))
            ),
            {
                "root_only": root_cfg.get("train_target_diagnostics"),
                **(
                    {"local_labels": local_cfg.get("train_target_diagnostics")}
                    if local_labels is not None
                    else {}
                ),
            },
            "Available runs use a nonconstant changepoint-count target distribution on the fixed bundle.",
        ),
    }
    if local_labels is not None:
        checks["local_labels_enabled"] = _check(
            "local_labels_enabled",
            abs(float(local_cfg.get("audit_fraction", float("nan"))) - 1.0) <= 1e-12
            and abs(float(local_cfg.get("leaf_query_rate", float("nan"))) - 1.0) <= 1e-12,
            {
                "audit_fraction": local_cfg.get("audit_fraction"),
                "leaf_query_rate": local_cfg.get("leaf_query_rate"),
            },
            "The local-label run enables sampled leaf and internal-node labels on the fixed bundle.",
        )
        checks["simple_local_controls_preserve_global_structure"] = _check(
            "simple_local_controls_preserve_global_structure",
            all(
                math.isfinite(_schedule_spread(local_labels, key))
                and abs(_schedule_spread(local_labels, key)) <= 1e-6
                for key in (
                    "leaf_ridge_tree",
                    "leaf_endpoint_table_tree",
                    "leaf_dt_tree",
                    "leaf_knn_tree",
                    "leaf_rf_tree",
                )
            ),
            {
                key: _schedule_spread(local_labels, key)
                for key in (
                    "leaf_ridge_tree",
                    "leaf_endpoint_table_tree",
                    "leaf_dt_tree",
                    "leaf_knn_tree",
                    "leaf_rf_tree",
                )
            },
            "The simple local baselines keep the exact global merge law fixed while changing only local information.",
        )

    n_panels = 2 if local_labels is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), constrained_layout=True)
    if not isinstance(axes, (list, tuple)):
        axes = [axes] if n_panels == 1 else list(axes)
    _bar_panel(
        axes[0],
        title="Root-Only | Same Fixed Bundle",
        payload=root_only,
        methods=(
            ("undersupported", "undersupported"),
            ("learned", "tree neural"),
            ("doc_sequence", "full-doc neuraloperator FNO"),
            ("doc_transformer", "full-doc transformer"),
            ("doc_level", "pooled MLP"),
            ("doc_level_ridge_unigram", "ridge unigram"),
            ("doc_level_ridge_bigram", "ridge bigram"),
            ("doc_level_ridge_trigram", "ridge trigram"),
            ("doc_level_ridge", "full-doc ridge"),
            ("rf_root", "full-doc RF"),
        ),
    )
    if local_labels is not None:
        _bar_panel(
            axes[1],
            title="Local Labels | Same Fixed Bundle",
            payload=local_labels,
            methods=(
                ("undersupported", "undersupported"),
                ("learned", "tree neural"),
                ("leaf_ridge_tree", "leaf ridge"),
                ("leaf_endpoint_table_tree", "leaf endpoint table"),
                ("leaf_dt_tree", "leaf DT"),
                ("leaf_knn_tree", "leaf kNN"),
                ("leaf_rf_tree", "leaf RF"),
                ("doc_sequence", "full-doc neuraloperator FNO"),
                ("doc_transformer", "full-doc transformer"),
                ("doc_level", "pooled MLP"),
                ("doc_level_ridge_unigram", "ridge unigram"),
                ("doc_level_ridge_bigram", "ridge bigram"),
                ("doc_level_ridge_trigram", "ridge trigram"),
                ("doc_level_ridge", "full-doc ridge"),
                ("rf_root", "full-doc RF"),
            ),
        )
    fig.suptitle("Observed-Token Markov Comparison", fontsize=12)
    fig_path = pages_dir / "markov_observed_token_root_mae.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    diagnostics = {
        "output_root": str(output_root),
        "suite_name": str(meta.get("suite_name", "markov-observed-token")),
        "suite_role": str(meta.get("suite_role", "diagnostic")),
        "profile": str(meta.get("profile", "")),
        "subset_mode": subset_mode,
        "selected_groups": list(meta.get("selected_groups", []) or ["root_only", "local_labels"]),
        "data_bundle_file": str(meta.get("data_bundle_file", "")),
        "data_signatures": dict(meta.get("data_signatures", {}) or {}),
        "checks": checks,
        "comparisons": {
            "root_only_tree_neural_root_mae": root_learned,
            "root_only_full_doc_sequence_root_mae": _root_mae(root_only, "doc_sequence"),
            "root_only_full_doc_sequence_test_exact_match_rate": float(
                root_doc_sequence_training.get("test_exact_match_rate", float("nan"))
            ),
            "root_only_full_doc_transformer_root_mae": _root_mae(root_only, "doc_transformer"),
            "root_only_full_doc_transformer_test_exact_match_rate": float(
                root_doc_transformer_training.get("test_exact_match_rate", float("nan"))
            ),
            "root_only_pooled_mlp_root_mae": _root_mae(root_only, "doc_level"),
            "root_only_ridge_unigram_root_mae": _root_mae(root_only, "doc_level_ridge_unigram"),
            "root_only_ridge_bigram_root_mae": _root_mae(root_only, "doc_level_ridge_bigram"),
            "root_only_ridge_trigram_root_mae": _root_mae(root_only, "doc_level_ridge_trigram"),
            "root_only_full_doc_ridge_root_mae": _root_mae(root_only, "doc_level_ridge"),
            "root_only_full_doc_rf_root_mae": _root_mae(root_only, "rf_root"),
            "root_only_undersupported_root_mae": _root_mae(root_only, "undersupported"),
            **(
                {
                    "local_labels_tree_neural_root_mae": local_learned,
                    "local_labels_full_doc_sequence_root_mae": _root_mae(local_labels, "doc_sequence"),
                    "local_labels_full_doc_transformer_root_mae": _root_mae(local_labels, "doc_transformer"),
                    "local_labels_best_simple_local_root_mae": best_local_simple,
                    "local_labels_leaf_rf_tree_root_mae": _root_mae(local_labels, "leaf_rf_tree"),
                    "local_labels_leaf_knn_tree_root_mae": _root_mae(local_labels, "leaf_knn_tree"),
                    "local_labels_leaf_endpoint_table_tree_root_mae": _root_mae(
                        local_labels, "leaf_endpoint_table_tree"
                    ),
                }
                if local_labels is not None
                else {}
            ),
        },
        "root_only_metrics": {
            key: dict(root_metrics.get(key, {}) or {})
            for key in (
                "learned",
                "doc_sequence",
                "doc_transformer",
                "doc_level",
                "doc_level_ridge_unigram",
                "doc_level_ridge_bigram",
                "doc_level_ridge_trigram",
                "doc_level_ridge",
                "rf_root",
                "undersupported",
                "exact",
            )
        },
        "local_label_metrics": (
            {
                key: dict(local_metrics.get(key, {}) or {})
                for key in (
                    "learned",
                    "doc_sequence",
                    "doc_transformer",
                    "leaf_ridge_tree",
                    "leaf_endpoint_table_tree",
                    "leaf_dt_tree",
                    "leaf_knn_tree",
                    "leaf_rf_tree",
                    "doc_level",
                    "doc_level_ridge_unigram",
                    "doc_level_ridge_bigram",
                    "doc_level_ridge_trigram",
                    "doc_level_ridge",
                    "rf_root",
                    "undersupported",
                    "exact",
                )
            }
            if local_labels is not None
            else {}
        ),
        "figure": str(fig_path),
        "interpretation_notes": {
            "recoverable_root_only_scope": (
                "The root-only recoverable benchmark is an exact-recovery diagnostic on one fixed saved bundle. "
                "It is meant to test whether full-sequence learners can recover the changepoint-count signal from "
                "tokens alone under matched inputs, not to serve as the main generalization benchmark."
            ),
            "ridge_trigram_vs_bigram": (
                "In the recoverable disjoint-palette setting, changepoint count is already a linear function "
                "of adjacent cross-palette bigram counts. A trigram-only ridge model drops those lower-order "
                "sufficient statistics and moves into a larger, sparser feature space, so it can do worse "
                "than bigram-only ridge on finite data."
            ),
        },
    }

    diag_path = out_dir / "markov_observed_token_latest_diagnostics.json"
    md_path = out_dir / "markov_observed_token_latest.md"
    pdf_path = out_dir / "markov_observed_token_latest.pdf"
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")

    markdown = "\n".join(
        [
            "---",
            "title: Observed-Token Markov Comparison",
            "---",
            "",
            f"**Output root:** `{output_root}`  ",
            f"**Bundle:** `{meta.get('data_bundle_file', '')}`  ",
            f"**Diagnostics:** `{diag_path}`",
            "",
            "## Setup",
            "",
            "This suite fixes one generated Markov train/val/test bundle, then compares learning modes on that exact same corpus:",
            "",
            "- `root_only`: root/document supervision only, with no sampled local labels.",
            *(
                ["- `local_labels`: the same bundle, but with sampled leaf and internal-node labels plus simple local baselines."]
                if local_labels is not None
                else []
            ),
            "",
            "The recoverable `root_only` comparison is an exact-recovery diagnostic on one fixed saved bundle. "
            "It is not the main paper generalization benchmark; its purpose is to check whether full-sequence "
            "neural baselines can recover a changepoint-count signal that is known to be present in the tokens.",
            "",
            "In the disjoint-palette recoverable setting, total changepoint count is already a linear function of adjacent cross-palette bigram counts. "
            "That is why `ridge bigram` can be near-exact while `ridge trigram` can still do worse: trigram-only ridge drops those lower-order sufficient "
            "statistics and fits a much larger, sparser feature space.",
            "",
            "## Key Numbers",
            "",
            f"- `root_only tree neural root_mae`: `{root_learned:.6f}`",
            f"- `root_only full-doc neuraloperator FNO root_mae`: `{_root_mae(root_only, 'doc_sequence'):.6f}`",
            f"- `root_only full-doc transformer root_mae`: `{_root_mae(root_only, 'doc_transformer'):.6f}`",
            f"- `root_only pooled MLP root_mae`: `{_root_mae(root_only, 'doc_level'):.6f}`",
            f"- `root_only ridge unigram root_mae`: `{_root_mae(root_only, 'doc_level_ridge_unigram'):.6g}`",
            f"- `root_only ridge bigram root_mae`: `{_root_mae(root_only, 'doc_level_ridge_bigram'):.6g}`",
            f"- `root_only ridge trigram root_mae`: `{_root_mae(root_only, 'doc_level_ridge_trigram'):.6g}`",
            f"- `root_only full-doc ridge root_mae`: `{_root_mae(root_only, 'doc_level_ridge'):.6g}`",
            f"- `root_only full-doc RF root_mae`: `{_root_mae(root_only, 'rf_root'):.6f}`",
            *(
                [
                    f"- `local_labels tree neural root_mae`: `{local_learned:.6f}`",
                    f"- `local_labels full-doc neuraloperator FNO root_mae`: `{_root_mae(local_labels, 'doc_sequence'):.6f}`",
                    f"- `local_labels full-doc transformer root_mae`: `{_root_mae(local_labels, 'doc_transformer'):.6f}`",
                    f"- `local_labels best simple local root_mae`: `{best_local_simple:.6f}`",
                ]
                if local_labels is not None
                else []
            ),
            "",
            "## Root-Only Table",
            "",
            *_markdown_table(
                root_only,
                methods=(
                    ("undersupported", "undersupported"),
                    ("learned", "tree neural"),
                    ("doc_sequence", "full-doc neuraloperator FNO"),
                    ("doc_transformer", "full-doc transformer"),
                    ("doc_level", "pooled MLP"),
                    ("doc_level_ridge_unigram", "ridge unigram"),
                    ("doc_level_ridge_bigram", "ridge bigram"),
                    ("doc_level_ridge_trigram", "ridge trigram"),
                    ("doc_level_ridge", "full-doc ridge"),
                    ("rf_root", "full-doc RF"),
                    ("exact", "exact"),
                ),
            ),
            *(
                [
                    "",
                    "## Local-Labels Table",
                    "",
                    *_markdown_table(
                        local_labels,
                        methods=(
                            ("undersupported", "undersupported"),
                            ("learned", "tree neural"),
                            ("doc_sequence", "full-doc neuraloperator FNO"),
                            ("doc_transformer", "full-doc transformer"),
                            ("leaf_ridge_tree", "leaf ridge"),
                            ("leaf_endpoint_table_tree", "leaf endpoint table"),
                            ("leaf_dt_tree", "leaf DT"),
                            ("leaf_knn_tree", "leaf kNN"),
                            ("leaf_rf_tree", "leaf RF"),
                            ("doc_level", "pooled MLP"),
                            ("doc_level_ridge_unigram", "ridge unigram"),
                            ("doc_level_ridge_bigram", "ridge bigram"),
                            ("doc_level_ridge_trigram", "ridge trigram"),
                            ("doc_level_ridge", "full-doc ridge"),
                            ("rf_root", "full-doc RF"),
                            ("exact", "exact"),
                        ),
                    ),
                ]
                if local_labels is not None
                else []
            ),
            "",
            "## Checks",
            "",
            *[
                f"- `{name}`: `{'pass' if check['pass'] else 'fail'}`"
                for name, check in checks.items()
            ],
            "",
            "## Figure",
            "",
            f"![](pages/{fig_path.name}){{ width=100% }}",
            "",
        ]
    )
    md_path.write_text(markdown + "\n", encoding="utf-8")

    if bool(args.emit_pdf):
        try:
            emitted_pdf = _run_pandoc(md_path, pdf_path)
        except Exception:
            emitted_pdf = False
    else:
        emitted_pdf = False

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "report_dir": str(out_dir),
                "markdown": str(md_path),
                "diagnostics": str(diag_path),
                "figure": str(fig_path),
                "pdf": str(pdf_path) if emitted_pdf else "",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
