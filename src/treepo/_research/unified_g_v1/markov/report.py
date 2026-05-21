from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from treepo._research.unified_g_v1.core.contracts import MarkovScope
from treepo._research.unified_g_v1.core.manifest import now_iso, write_json
from treepo._research.unified_g_v1.markov.benchmarks import resolve_scope_benchmark
from treepo._research.unified_g_v1.markov.runner import MarkovRunRecord


LEAF_TOKENS = (128, 64, 32, 16, 8)
ROOT_SHARES = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10)


def _safe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return float(out)


def _index_records(
    records: Sequence[MarkovRunRecord],
) -> dict[tuple[str, int, int, str, str], MarkovRunRecord]:
    return {
        (
            record.spec.scope.value,
            int(record.spec.root_share),
            int(record.spec.leaf_tokens),
            record.spec.supervision_policy.value,
            record.spec.profile.value,
        ): record
        for record in records
    }


def _panel_summary(
    *,
    scope: MarkovScope,
    root_share: int,
    indexed: Mapping[tuple[str, int, int, str, str], MarkovRunRecord],
) -> dict[str, Any]:
    primary_by_leaf: dict[str, float] = {}
    secondary_by_leaf: dict[str, float] = {}
    primary_run_keys: dict[str, str] = {}
    secondary_run_keys: dict[str, str] = {}
    for leaf in LEAF_TOKENS:
        primary = indexed.get((scope.value, int(root_share), int(leaf), "root_only", "root_only"))
        secondary = indexed.get(
            (scope.value, int(root_share), int(leaf), "leaf_mass_eq", "standard")
        )
        primary_value = _safe_float((primary.extracted_metrics if primary else {}).get("learned_root_mae"))
        secondary_value = _safe_float((secondary.extracted_metrics if secondary else {}).get("learned_root_mae"))
        if primary_value is not None:
            primary_by_leaf[str(int(leaf))] = float(primary_value)
            primary_run_keys[str(int(leaf))] = primary.spec.run_key
        if secondary_value is not None:
            secondary_by_leaf[str(int(leaf))] = float(secondary_value)
            secondary_run_keys[str(int(leaf))] = secondary.spec.run_key
    canary = indexed.get((scope.value, int(root_share), 128, "root_only", "fno_canary"))
    duplicate = indexed.get(
        (scope.value, int(root_share), 128, "root_only", "duplicate_local_label_one_leaf")
    )
    official_fno_root_mae = _safe_float(
        (canary.extracted_metrics if canary else {}).get("official_fno_root_mae")
    )
    canary_root_mae = _safe_float(
        (canary.extracted_metrics if canary else {}).get("learned_root_mae")
    )
    duplicate_root_mae = _safe_float(
        (duplicate.extracted_metrics if duplicate else {}).get("learned_root_mae")
    )
    best_primary_leaf = None
    best_primary_root_mae = None
    if primary_by_leaf:
        best_primary_leaf = min(primary_by_leaf, key=lambda key: float(primary_by_leaf[key]))
        best_primary_root_mae = float(primary_by_leaf[best_primary_leaf])
    best_secondary_leaf = None
    best_secondary_root_mae = None
    if secondary_by_leaf:
        best_secondary_leaf = min(
            secondary_by_leaf,
            key=lambda key: float(secondary_by_leaf[key]),
        )
        best_secondary_root_mae = float(secondary_by_leaf[best_secondary_leaf])
    return {
        "root_share": int(root_share),
        "primary_series": {
            "profile": "root_only",
            "supervision_policy": "root_only",
            "root_mae_by_leaf_tokens": primary_by_leaf,
            "run_keys_by_leaf_tokens": primary_run_keys,
            "best_leaf_tokens": int(best_primary_leaf) if best_primary_leaf is not None else None,
            "best_root_mae": best_primary_root_mae,
        },
        "secondary_series": {
            "profile": "standard",
            "supervision_policy": "leaf_mass_eq",
            "root_mae_by_leaf_tokens": secondary_by_leaf,
            "run_keys_by_leaf_tokens": secondary_run_keys,
            "best_leaf_tokens": int(best_secondary_leaf) if best_secondary_leaf is not None else None,
            "best_root_mae": best_secondary_root_mae,
        },
        "comparators": {
            "official_fno_root_mae": official_fno_root_mae,
            "one_leaf_canary_root_mae": canary_root_mae,
            "duplicate_local_label_one_leaf_root_mae": duplicate_root_mae,
            "official_fno_run_key": canary.spec.run_key if canary is not None else "",
            "duplicate_local_label_run_key": duplicate.spec.run_key if duplicate is not None else "",
        },
    }


def build_fixed_report_summary(
    records: Sequence[MarkovRunRecord],
    *,
    output_root: str | Path,
    train_doc_count: int,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser()
    indexed = _index_records(records)
    summary = {
        "generated_at": now_iso(),
        "output_root": str(output_root),
        "train_doc_count": int(train_doc_count),
        "root_shares": [int(value) for value in ROOT_SHARES],
        "leaf_tokens": [int(value) for value in LEAF_TOKENS],
        "run_manifest": str(output_root / "run_manifest.json"),
        "figures": {},
        "scopes": {},
    }
    for scope in (MarkovScope.RECOVERABLE_V5_T128, MarkovScope.R12_P079):
        scope_meta = resolve_scope_benchmark(scope)
        panel_summaries = [
            _panel_summary(scope=scope, root_share=int(root_share), indexed=indexed)
            for root_share in ROOT_SHARES
        ]
        summary["scopes"][scope.value] = {
            "scope_key": scope.value,
            "scope_label": scope_meta.scope_label,
            "scope_subtitle": scope_meta.scope_subtitle,
            "train_doc_count": int(train_doc_count),
            "root_shares": [int(value) for value in ROOT_SHARES],
            "leaf_tokens": [int(value) for value in LEAF_TOKENS],
            "panel_summaries": panel_summaries,
            "figures": {
                "root_only": str(
                    output_root / "figures" / scope_meta.figure_filename(
                        train_docs=int(train_doc_count),
                        with_leaf_mass_eq=False,
                    )
                ),
                "with_leaf_mass_eq": str(
                    output_root / "figures" / scope_meta.figure_filename(
                        train_docs=int(train_doc_count),
                        with_leaf_mass_eq=True,
                    )
                ),
            },
        }
        summary["figures"][scope.value] = summary["scopes"][scope.value]["figures"]["root_only"]
        summary["figures"][f"{scope.value}__leaf_mass_eq"] = summary["scopes"][scope.value]["figures"][
            "with_leaf_mass_eq"
        ]
    return summary


def _plot_scope_variant(
    scope_summary: Mapping[str, Any],
    *,
    output_path: Path,
    with_leaf_mass_eq: bool,
) -> None:
    panel_summaries = list(scope_summary.get("panel_summaries") or [])
    fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharey=True, squeeze=False)
    axes_list = list(axes.ravel())
    x = list(range(len(LEAF_TOKENS)))
    tick_labels = [f"{leaf}\n({128 // leaf})" for leaf in LEAF_TOKENS]
    emitted = {
        "primary": False,
        "secondary": False,
        "fno": False,
        "canary": False,
        "duplicate": False,
    }
    for ax, panel in zip(axes_list, panel_summaries):
        primary = dict(((panel.get("primary_series") or {}).get("root_mae_by_leaf_tokens")) or {})
        secondary = dict(((panel.get("secondary_series") or {}).get("root_mae_by_leaf_tokens")) or {})
        primary_y = [_safe_float(primary.get(str(leaf))) for leaf in LEAF_TOKENS]
        primary_points = [
            (idx, value)
            for idx, value in enumerate(primary_y)
            if value is not None
        ]
        if primary_points:
            ax.plot(
                [item[0] for item in primary_points],
                [item[1] for item in primary_points],
                color="#1f77b4",
                marker="o",
                linewidth=2.1,
                label="Unified-G root-only" if not emitted["primary"] else None,
            )
            emitted["primary"] = True
        if with_leaf_mass_eq:
            secondary_y = [_safe_float(secondary.get(str(leaf))) for leaf in LEAF_TOKENS]
            secondary_points = [
                (idx, value)
                for idx, value in enumerate(secondary_y)
                if value is not None
            ]
            if secondary_points:
                ax.plot(
                    [item[0] for item in secondary_points],
                    [item[1] for item in secondary_points],
                    color="#c44e52",
                    marker="s",
                    linewidth=2.0,
                    linestyle="--",
                    label="Unified-G leaf-mass-eq" if not emitted["secondary"] else None,
                )
                emitted["secondary"] = True
        comparators = dict(panel.get("comparators") or {})
        official_fno = _safe_float(comparators.get("official_fno_root_mae"))
        if official_fno is not None:
            ax.axhline(
                official_fno,
                color="black",
                linestyle=":",
                linewidth=1.8,
                label="Official FNO" if not emitted["fno"] else None,
            )
            emitted["fno"] = True
        canary = _safe_float(comparators.get("one_leaf_canary_root_mae"))
        if canary is not None:
            ax.scatter(
                [0],
                [canary],
                facecolors="none",
                edgecolors="black",
                marker="D",
                s=64,
                linewidths=1.1,
                label="One-leaf canary" if not emitted["canary"] else None,
                zorder=5,
            )
            emitted["canary"] = True
        duplicate = _safe_float(comparators.get("duplicate_local_label_one_leaf_root_mae"))
        if duplicate is not None:
            ax.scatter(
                [0],
                [duplicate],
                color="#6c6c6c",
                edgecolors="black",
                marker="^",
                s=68,
                linewidths=0.9,
                label="Duplicate local label" if not emitted["duplicate"] else None,
                zorder=6,
            )
            emitted["duplicate"] = True
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_title(f"Root share {int(panel.get('root_share', 0))}%")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Leaf tokens\n(leaves/doc)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test root MAE")
    for ax in axes_list[len(panel_summaries) :]:
        ax.axis("off")
    fig.suptitle(
        f"{scope_summary.get('scope_label', '')}\n{scope_summary.get('scope_subtitle', '')}",
        fontsize=14,
    )
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(handles)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_fixed_report(
    summary: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser()
    materialized = dict(summary)
    figures = dict(materialized.get("figures") or {})
    scopes = dict(materialized.get("scopes") or {})
    for scope_key, scope_summary in scopes.items():
        scope_figures = dict((scope_summary.get("figures") or {}))
        root_only_path = Path(str(scope_figures.get("root_only", "")))
        mass_eq_path = Path(str(scope_figures.get("with_leaf_mass_eq", "")))
        _plot_scope_variant(scope_summary, output_path=root_only_path, with_leaf_mass_eq=False)
        _plot_scope_variant(scope_summary, output_path=mass_eq_path, with_leaf_mass_eq=True)
        figures[scope_key] = str(root_only_path)
        figures[f"{scope_key}__leaf_mass_eq"] = str(mass_eq_path)
    materialized["figures"] = figures
    write_json(output_root / "summary.json", materialized)
    return materialized
