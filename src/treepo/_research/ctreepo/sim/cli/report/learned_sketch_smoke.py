#!/usr/bin/env python3
"""Manifest-first smoke report for the learned-sketch validation suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from treepo._research.ctreepo.sim.manifest import read_manifest_jsonl
from treepo._research.ctreepo.sim.suite.common import read_suite_meta, resolve_grouped_suite_paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a learned-sketch smoke report.")
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
    return sorted((output_root / "learned_sketch_simulation").rglob("*.json"))


def _load_rows(output_root: Path) -> tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    summary_paths, meta = _suite_summary_paths(output_root)
    if not summary_paths:
        summary_paths = _fallback_summary_paths(output_root)
    rows: List[Dict[str, object]] = []
    runtime_config: Dict[str, object] = {}
    for path in summary_paths:
        payload = _load_json(path)
        if not payload:
            continue
        if not runtime_config and isinstance(payload.get("runtime_config"), dict):
            runtime_config = dict(payload.get("runtime_config", {}) or {})
        for row in (payload.get("rows", []) or []):
            if not isinstance(row, Mapping):
                continue
            rows.append({**dict(row), "summary_path": str(path)})
    return rows, meta, runtime_config


def _is_finite(row: Mapping[str, object], key: str) -> bool:
    try:
        value = float(row[key])
    except Exception:
        return False
    return math.isfinite(value)


def _metric_check(pass_value: bool, *, value: object, description: str) -> Dict[str, object]:
    return {
        "pass": bool(pass_value),
        "value": value,
        "description": str(description),
    }


def _best_row(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    ranked = sorted(
        rows,
        key=lambda row: (
            abs(float(row.get("distance_to_hll_floor_rel_rmse", float("inf")))),
            float(row.get("learned_relative_rmse", float("inf"))),
            float(row.get("learned_schedule_spread_mean", float("inf"))),
        ),
    )
    return dict(ranked[0]) if ranked else {}


def _group_by_state(rows: Sequence[Mapping[str, object]]) -> Dict[int, List[Dict[str, object]]]:
    out: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        try:
            state_dim = int(row.get("state_dim", -1))
        except Exception:
            continue
        out.setdefault(state_dim, []).append(dict(row))
    for state_dim in list(out):
        out[state_dim] = sorted(out[state_dim], key=lambda row: int(row.get("train_size", 0)))
    return out


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
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (output_root / "figures" / "learned_sketch_smoke")
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    rows, meta, runtime_config = _load_rows(output_root)
    if not rows:
        raise SystemExit(f"no learned-sketch rows found under {output_root}")

    best_row = _best_row(rows)
    metric_checks = {
        "has_rows": _metric_check(bool(rows), value=len(rows), description="At least one smoke row was loaded."),
        "finite_learned_relative_rmse": _metric_check(
            all(_is_finite(row, "learned_relative_rmse") for row in rows),
            value=[row.get("learned_relative_rmse") for row in rows],
            description="All loaded rows report a finite learned relative RMSE.",
        ),
        "finite_hll_relative_rmse": _metric_check(
            all(_is_finite(row, "hll_relative_rmse") for row in rows),
            value=[row.get("hll_relative_rmse") for row in rows],
            description="All loaded rows report a finite HLL relative RMSE.",
        ),
        "finite_distance_to_hll_floor_rel_rmse": _metric_check(
            all(_is_finite(row, "distance_to_hll_floor_rel_rmse") for row in rows),
            value=[row.get("distance_to_hll_floor_rel_rmse") for row in rows],
            description="All loaded rows report a finite distance-to-floor metric.",
        ),
        "finite_learned_schedule_spread_mean": _metric_check(
            all(_is_finite(row, "learned_schedule_spread_mean") for row in rows),
            value=[row.get("learned_schedule_spread_mean") for row in rows],
            description="All loaded rows report a finite learned schedule spread.",
        ),
        "finite_train_total_queries_estimate": _metric_check(
            all(_is_finite(row, "train_total_queries_estimate") for row in rows),
            value=[row.get("train_total_queries_estimate") for row in rows],
            description="All loaded rows report a finite oracle query estimate.",
        ),
        "hll_schedule_spread_zero": _metric_check(
            all(abs(float(row.get("hll_schedule_spread_mean", float("inf")))) <= 1e-12 for row in rows),
            value=[row.get("hll_schedule_spread_mean") for row in rows],
            description="The exact HLL baseline remains schedule-invariant in the smoke run.",
        ),
    }

    by_state = _group_by_state(rows)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    ax = axes[0]
    for state_dim, state_rows in sorted(by_state.items()):
        xs = [int(row["train_size"]) for row in state_rows]
        ys = [float(row["learned_relative_rmse"]) for row in state_rows]
        line = ax.plot(xs, ys, marker="o", label=f"learned d={state_dim}")[0]
        color = line.get_color()
        hll_rel = float(state_rows[0]["hll_relative_rmse"])
        hll_theory = float(state_rows[0]["hll_rse_theory"])
        xmin = min(xs)
        xmax = max(xs)
        ax.hlines(hll_rel, xmin=xmin, xmax=xmax, colors=color, linestyles="--", alpha=0.35)
        ax.hlines(hll_theory, xmin=xmin, xmax=xmax, colors=color, linestyles=":", alpha=0.65)
    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Relative RMSE")
    ax.set_title("Relative RMSE vs HLL")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for state_dim, state_rows in sorted(by_state.items()):
        xs = [int(row["train_size"]) for row in state_rows]
        ys = [float(row["distance_to_hll_floor_rel_rmse"]) for row in state_rows]
        ax.plot(xs, ys, marker="o", label=f"d={state_dim}")
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Distance to HLL Theory Floor")
    ax.set_title("Excess Relative RMSE")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ref_state = sorted(by_state)[0]
    geom_rows = by_state[ref_state]
    xs = np.array([int(row["train_size"]) for row in geom_rows], dtype=np.float64)
    mean_internal = np.array([float(row["train_mean_internal_nodes"]) for row in geom_rows], dtype=np.float64)
    mean_audit = np.array([float(row["train_audit_nodes_mean"]) for row in geom_rows], dtype=np.float64)
    coverage = np.array([float(row["train_audit_coverage_mean"]) for row in geom_rows], dtype=np.float64)
    ax.plot(xs, mean_internal, marker="o", label="mean internal nodes/doc")
    ax.plot(xs, mean_audit, marker="s", label="mean audited nodes/doc")
    ax.set_xlabel("Train Documents")
    ax.set_ylabel("Nodes / Document")
    ax.set_title("Audit Geometry")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(xs, coverage, marker="^", linestyle="--", color="tab:green", label="audit coverage")
    ax2.set_ylabel("Audit Coverage")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(
        "Learned-Sketch Smoke Validation | "
        f"device={runtime_config.get('device_used', runtime_config.get('device', 'unknown'))}",
        fontsize=11,
    )
    fig_path = pages_dir / "learned_sketch_smoke_summary.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    diag_path = out_dir / "learned_sketch_smoke_latest_diagnostics.json"
    md_path = out_dir / "learned_sketch_smoke_latest.md"
    pdf_path = out_dir / "learned_sketch_smoke_latest.pdf"

    diagnostics = {
        "output_root": str(output_root),
        "suite_name": str(meta.get("suite_name", "learned-sketch-smoke")),
        "suite_role": str(meta.get("suite_role", "diagnostic")),
        "selected_groups": list(meta.get("selected_groups", []) or ["proxy_baseline"]),
        "row_count": int(len(rows)),
        "runtime_config": runtime_config,
        "metric_checks": metric_checks,
        "best_row": best_row,
        "figure": str(fig_path),
        "mean_distance_to_hll_floor_rel_rmse": float(
            statistics.mean(float(row["distance_to_hll_floor_rel_rmse"]) for row in rows)
        ),
    }
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")

    best_summary_lines = [
        f"- `state_dim`: `{int(best_row.get('state_dim', 0))}`",
        f"- `train_size`: `{int(best_row.get('train_size', 0))}`",
        f"- `learned_relative_rmse`: `{float(best_row.get('learned_relative_rmse', float('nan'))):.4f}`",
        f"- `hll_relative_rmse`: `{float(best_row.get('hll_relative_rmse', float('nan'))):.4f}`",
        f"- `distance_to_hll_floor_rel_rmse`: `{float(best_row.get('distance_to_hll_floor_rel_rmse', float('nan'))):+.4f}`",
        f"- `learned_schedule_spread_mean`: `{float(best_row.get('learned_schedule_spread_mean', float('nan'))):.4f}`",
        f"- `train_total_queries_estimate`: `{int(float(best_row.get('train_total_queries_estimate', 0.0)))}`",
    ]
    check_lines = [
        f"- `{name}`: `{'pass' if check['pass'] else 'fail'}`"
        for name, check in metric_checks.items()
    ]
    markdown = "\n".join(
        [
            "---",
            "title: Learned-Sketch Smoke Validation",
            "---",
            "",
            f"**Output root:** `{output_root}`  ",
            f"**Diagnostics:** `{diag_path}`",
            "",
            "## Scope",
            "",
            "This is the smallest end-to-end mergeable-sketch validation path in the repo. It checks that a tiny local learner runs, emits metrics, and compares cleanly against the exact HLL baseline.",
            "",
            "## Best Row",
            "",
            *best_summary_lines,
            "",
            "## Smoke Checks",
            "",
            *check_lines,
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
        if emitted_pdf:
            print(f"wrote_pdf | {pdf_path}")

    print(f"wrote_markdown | {md_path}")
    print(f"wrote_diagnostics | {diag_path}")
    print(f"wrote_figure | {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
