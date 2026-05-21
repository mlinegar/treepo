#!/usr/bin/env python3
"""Generate an interim PDF report for the LDA tree-recovery production sweep."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from treepo._research.ctreepo.sim.util import safe_float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report current LDA tree-recovery sweep progress.")
    p.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Production sweep root containing exact_cpu/, learned_cpu_shadow/, learned_gpu/, and sweep_spec.txt",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for report artifacts (default: <input-root>/report).",
    )
    return p.parse_args(list(argv) if argv is not None else None)


_safe_float = safe_float


def _safe_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _safe_std(xs: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.std(vals))


def _pct(done: int, total: int) -> float:
    if total <= 0:
        return float("nan")
    return 100.0 * float(done) / float(total)


def _text_page(pdf: PdfPages, *, title: str, lines: Sequence[str], font_size: int = 10) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12)
    ax.text(0.01, 0.98, "\n".join(lines), family="monospace", fontsize=font_size, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(
    pdf: PdfPages,
    *,
    title: str,
    col_labels: Sequence[str],
    cell_text: Sequence[Sequence[str]],
    font_size: int = 9,
    scale_y: float = 1.4,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title(title, pad=12)
    table = ax.table(
        cellText=list(cell_text),
        colLabels=list(col_labels),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)
    try:
        table.auto_set_column_width(col=list(range(len(col_labels))))
    except Exception:
        pass
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax: plt.Axes, mat: np.ndarray, *, title: str, xlabels: Sequence[str], ylabels: Sequence[str], cmap: str, vmin=None, vmax=None) -> None:
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(list(xlabels), rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(list(ylabels))
    plt.colorbar(im, ax=ax, shrink=0.8)


def _parse_spec(spec_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not spec_path.exists():
        return out
    for line in spec_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _count_axis_items(text: str, axis: str) -> int:
    m = re.search(rf"{re.escape(axis)}\(([^)]*)\)", text)
    if not m:
        return 0
    body = m.group(1).strip()
    if not body:
        return 0
    if "," in body:
        return len([x.strip() for x in body.split(",") if x.strip()])
    if axis == "seeds" and re.fullmatch(r"\d+", body):
        return int(body)
    return 1


def _count_axis_items_any(text: str, axes: Sequence[str]) -> int:
    for axis in axes:
        count = _count_axis_items(text, axis)
        if count > 0:
            return count
    return 0


def _expected_counts(spec: Dict[str, str]) -> Dict[str, int]:
    exact_text = spec.get("exact_matrix", "")
    gpu_text = spec.get("learned_gpu_matrix", "")
    cpu_text = spec.get("learned_cpu_shadow_matrix", "")
    exact = (
        _count_axis_items(exact_text, "leaf")
        * _count_axis_items(exact_text, "dtc")
        * _count_axis_items_any(exact_text, ("quadratic_weight", "lambda"))
        * _count_axis_items(exact_text, "seeds")
    )
    learned_gpu = (
        _count_axis_items(gpu_text, "leaf")
        * _count_axis_items(gpu_text, "dtc")
        * _count_axis_items_any(gpu_text, ("quadratic_weight", "lambda"))
        * _count_axis_items(gpu_text, "train")
        * _count_axis_items(gpu_text, "state")
        * _count_axis_items(gpu_text, "seeds")
    )
    learned_cpu = (
        _count_axis_items(cpu_text, "leaf")
        * _count_axis_items(cpu_text, "dtc")
        * _count_axis_items_any(cpu_text, ("quadratic_weight", "lambda"))
        * _count_axis_items(cpu_text, "train")
        * _count_axis_items(cpu_text, "state")
        * _count_axis_items(cpu_text, "seeds")
    )
    learned_bundle = (
        _count_axis_items(gpu_text, "leaf")
        * _count_axis_items_any(gpu_text, ("quadratic_weight", "lambda"))
        * _count_axis_items(gpu_text, "train")
        * _count_axis_items(gpu_text, "state")
    )
    return {
        "exact_cpu": int(exact),
        "learned_gpu": int(learned_gpu),
        "learned_cpu_shadow": int(learned_cpu),
        "learned_gpu_per_bundle": int(learned_bundle),
    }


def _quadratic_weight(cfg: dict) -> float:
    return _safe_float(cfg.get("quadratic_utility_weight"))


def _load_exact_rows(root: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        exact_recovery = payload.get("exact_recovery", {}) or {}
        methods = payload.get("methods", {}) or {}
        leaf_avg = methods.get("leaf_average", {}) or {}
        leaf_u = methods.get("leaf_utility_only", {}) or {}
        rows.append(
            {
                "path": str(path),
                "dtc": _safe_float(cfg.get("doc_topic_concentration")),
                "leaf": int(cfg.get("leaf_tokens", -1)),
                "quad_weight": _quadratic_weight(cfg),
                "seed": int(cfg.get("seed", -1)),
                "exact_root_pi": _safe_float(exact_recovery.get("root_pi_l1_mean")),
                "exact_root_util": _safe_float(exact_recovery.get("root_utility_abs_mean")),
                "leaf_avg_pi": _safe_float(leaf_avg.get("pi_l1_to_full_mean")),
                "leaf_avg_util": _safe_float(leaf_avg.get("utility_abs_to_full_mean")),
                "leaf_u_util": _safe_float(leaf_u.get("utility_abs_to_full_mean")),
            }
        )
    return rows


def _load_learned_rows(root: Path, *, lane: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        learning = payload.get("learning", {}) or {}
        tree_diag = learning.get("tree_svd_sketch", {}) or {}
        methods = payload.get("methods", {}) or {}
        tree = methods.get("tree_svd_sketch", {}) or {}
        full = methods.get("full_doc_operator", {}) or {}
        rows.append(
            {
                "lane": str(lane),
                "path": str(path),
                "dtc": _safe_float(cfg.get("doc_topic_concentration")),
                "leaf": int(cfg.get("leaf_tokens", -1)),
                "train": int(cfg.get("train_docs", -1)),
                "state": int(cfg.get("state_dim", -1)),
                "quad_weight": _quadratic_weight(cfg),
                "seed": int(cfg.get("seed", -1)),
                "vocab": int(cfg.get("vocab_size", -1)),
                "tree_pi": _safe_float(tree.get("pi_l1_to_full_mean")),
                "tree_util": _safe_float(tree.get("utility_abs_to_full_mean")),
                "tree_count": _safe_float(tree.get("count_l1_to_full_mean")),
                "full_pi": _safe_float(full.get("pi_l1_to_full_mean")),
                "full_util": _safe_float(full.get("utility_abs_to_full_mean")),
                "kept_components": int(tree_diag.get("kept_components", -1)),
                "train_rank": int(tree_diag.get("train_rank", -1)),
                "exact_family": bool(tree_diag.get("exact_family_representable", False)),
                "exact_train": bool(tree_diag.get("exact_train_manifold_representable", False)),
            }
        )
    return rows


def _group_mean(rows: Sequence[dict], key_fields: Sequence[str], value_field: str) -> Dict[Tuple[object, ...], float]:
    buckets: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in key_fields)].append(_safe_float(row[value_field]))
    return {k: _safe_mean(v) for k, v in buckets.items()}


def _group_std(rows: Sequence[dict], key_fields: Sequence[str], value_field: str) -> Dict[Tuple[object, ...], float]:
    buckets: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in key_fields)].append(_safe_float(row[value_field]))
    return {k: _safe_std(v) for k, v in buckets.items()}


def _stale_exact_capacity(row: dict) -> bool:
    return bool(int(row.get("state", -1)) == int(row.get("vocab", -2)) and int(row.get("kept_components", -1)) < int(row.get("vocab", -2)))


def _coverage_page(
    pdf: PdfPages,
    *,
    expected: Dict[str, int],
    exact_rows: Sequence[dict],
    learned_cpu_rows: Sequence[dict],
    learned_gpu_rows: Sequence[dict],
    bundle_expected: int,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.5))

    lane_names = ["exact_cpu", "learned_cpu_shadow", "learned_gpu"]
    done = [len(exact_rows), len(learned_cpu_rows), len(learned_gpu_rows)]
    total = [expected["exact_cpu"], expected["learned_cpu_shadow"], expected["learned_gpu"]]
    pct = [_pct(d, t) for d, t in zip(done, total)]
    ax = axs[0, 0]
    ax.bar(range(len(lane_names)), pct, color=["#4C78A8", "#72B7B2", "#59A14F"])
    ax.set_ylim(0.0, 105.0)
    ax.set_xticks(range(len(lane_names)))
    ax.set_xticklabels(lane_names, rotation=20, ha="right")
    ax.set_ylabel("% complete")
    ax.set_title("Lane Completion")
    for i, (d, t, p) in enumerate(zip(done, total, pct)):
        ax.text(i, min(102.0, p + 2.0), f"{d}/{t}", ha="center", va="bottom", fontsize=9)

    dtcs = [0.2, 0.6, 1.5]
    seeds = list(range(8))
    mat = np.zeros((len(dtcs), len(seeds)), dtype=np.float64)
    counts = Counter((row["dtc"], row["seed"]) for row in learned_gpu_rows)
    for i, dtc in enumerate(dtcs):
        for j, seed in enumerate(seeds):
            mat[i, j] = float(counts.get((dtc, seed), 0))
    _heatmap(
        axs[0, 1],
        mat,
        title=f"Learned GPU Bundle Counts (max {bundle_expected})",
        xlabels=[str(x) for x in seeds],
        ylabels=[str(x) for x in dtcs],
        cmap="YlGn",
        vmin=0.0,
        vmax=max(1.0, float(bundle_expected)),
    )
    axs[0, 1].set_xlabel("seed")
    axs[0, 1].set_ylabel("doc_topic_concentration")

    state_counts = Counter(int(row["state"]) for row in learned_gpu_rows)
    states = [8, 16, 32, 64, 128, 256, 512]
    expected_per_state = expected["learned_gpu"] / max(1, len(states))
    vals = [state_counts.get(s, 0) for s in states]
    ax = axs[1, 0]
    ax.bar(range(len(states)), vals, color="#F28E2B")
    ax.axhline(expected_per_state, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels([str(s) for s in states])
    ax.set_ylabel("completed rows")
    ax.set_title("Learned GPU Coverage By State Dimension")

    stale_rows = [row for row in learned_gpu_rows if _stale_exact_capacity(row)]
    stale_groups = Counter((row["leaf"], row["train"], row["quad_weight"]) for row in stale_rows)
    labels = [f"L{leaf}/T{train}/Q{quad_weight:g}" for leaf, train, quad_weight in sorted(stale_groups)]
    vals = [stale_groups[k] for k in sorted(stale_groups)]
    ax = axs[1, 1]
    if labels:
        ax.bar(range(len(labels)), vals, color="#E15759")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("rows")
        ax.set_title("Stale Exact-Capacity Rows\n(old singleton SVD path)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No stale exact-capacity rows detected.", ha="center", va="center")

    fig.suptitle("LDA Tree-Recovery Sweep Coverage", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _exact_family_page(pdf: PdfPages, exact_rows: Sequence[dict]) -> None:
    dtcs = [0.2, 0.6, 1.5]
    quad_weights = [0.0, 1.0, 2.0]
    leaves = [384, 192, 96, 48, 24, 16]
    fig, axs = plt.subplots(2, 3, figsize=(11.0, 8.5), sharex=True)

    mean_leaf_avg_pi = _group_mean(exact_rows, ["dtc", "quad_weight", "leaf"], "leaf_avg_pi")
    mean_leaf_u_util = _group_mean(exact_rows, ["dtc", "quad_weight", "leaf"], "leaf_u_util")

    for col, dtc in enumerate(dtcs):
        ax = axs[0, col]
        for quad_weight, color in zip(quad_weights, ["#4C78A8", "#F28E2B", "#59A14F"]):
            ys = [mean_leaf_avg_pi.get((dtc, quad_weight, leaf), float("nan")) for leaf in leaves]
            ax.plot(range(len(leaves)), ys, marker="o", label=f"quadratic weight={quad_weight:g}", color=color)
        ax.set_title(f"leaf_average π-gap | dtc={dtc}")
        ax.set_xticks(range(len(leaves)))
        ax.set_xticklabels([str(x) for x in leaves], rotation=30)
        ax.set_ylabel("pi_l1_to_full_mean")
        if col == 0:
            ax.legend(loc="upper left", fontsize=8)

        ax2 = axs[1, col]
        for quad_weight, color in zip(quad_weights, ["#4C78A8", "#F28E2B", "#59A14F"]):
            ys = [mean_leaf_u_util.get((dtc, quad_weight, leaf), float("nan")) for leaf in leaves]
            ax2.plot(range(len(leaves)), ys, marker="o", label=f"quadratic weight={quad_weight:g}", color=color)
        ax2.set_title(f"leaf-only utility gap | dtc={dtc}")
        ax2.set_xticks(range(len(leaves)))
        ax2.set_xticklabels([str(x) for x in leaves], rotation=30)
        ax2.set_ylabel("utility_abs_to_full_mean")
        ax2.set_xlabel("leaf_tokens")

    fig.suptitle("Diagnostic Exact Family: Leaves Only Matter As An Algorithmic Partition", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _learned_tree_page(
    pdf: PdfPages,
    *,
    rows: Sequence[dict],
    dtc: float,
    title_suffix: str,
) -> None:
    leaves = [384, 192, 96, 16]
    trains = [128, 512, 2048]
    states = [8, 16, 32, 64, 128, 256, 512]
    quad_weight = 2.0

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True, sharey=True)
    mean_tree_pi = _group_mean(rows, ["dtc", "leaf", "train", "state", "quad_weight"], "tree_pi")
    std_tree_pi = _group_std(rows, ["dtc", "leaf", "train", "state", "quad_weight"], "tree_pi")
    counts = Counter((row["dtc"], row["leaf"], row["train"], row["state"], row["quad_weight"]) for row in rows)
    stale = {
        (row["dtc"], row["leaf"], row["train"], row["state"], row["quad_weight"])
        for row in rows
        if _stale_exact_capacity(row)
    }

    colors = {128: "#4C78A8", 512: "#F28E2B", 2048: "#59A14F"}
    for ax, leaf in zip(axs.reshape(-1), leaves):
        for train in trains:
            xs = np.arange(len(states))
            ys = np.asarray([mean_tree_pi.get((dtc, leaf, train, state, quad_weight), float("nan")) for state in states], dtype=np.float64)
            errs = np.asarray([std_tree_pi.get((dtc, leaf, train, state, quad_weight), float("nan")) for state in states], dtype=np.float64)
            ax.plot(xs, ys, marker="o", color=colors[train], label=f"train={train}")
            finite = np.isfinite(ys) & np.isfinite(errs)
            if np.any(finite):
                ax.fill_between(xs[finite], np.maximum(0.0, ys[finite] - errs[finite]), ys[finite] + errs[finite], color=colors[train], alpha=0.15)
            for idx, state in enumerate(states):
                if (dtc, leaf, train, state, quad_weight) in stale:
                    ax.scatter([idx], [ys[idx]], marker="x", s=80, color="#E15759", zorder=5)
            ns = [counts.get((dtc, leaf, train, state, quad_weight), 0) for state in states]
            if any(n > 0 for n in ns):
                ax.text(0.98, 0.06 + 0.05 * (train == 512) + 0.10 * (train == 128), f"{train}: n={max(ns)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=colors[train])
        ax.set_title(f"leaf={leaf}")
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([str(s) for s in states], rotation=30)
        ax.set_xlabel("state_dim")
        ax.set_ylabel("tree_svd pi_l1_to_full")
    handles, labels = axs.reshape(-1)[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"Diagnostic Learned Tree Sketch Recovery | dtc={dtc} | quadratic weight={quad_weight:g} | {title_suffix}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _full_doc_and_stale_page(pdf: PdfPages, learned_rows: Sequence[dict]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 8.5))

    dedup: Dict[Tuple[float, int, int], dict] = {}
    for row in learned_rows:
        dedup[(row["dtc"], row["train"], row["seed"])] = row
    dedup_rows = list(dedup.values())

    dtcs = sorted({row["dtc"] for row in dedup_rows})
    trains = [128, 512, 2048]
    colors = {0.2: "#4C78A8", 0.6: "#F28E2B", 1.5: "#59A14F"}
    mean_full_pi = _group_mean(dedup_rows, ["dtc", "train"], "full_pi")
    std_full_pi = _group_std(dedup_rows, ["dtc", "train"], "full_pi")
    ax = axs[0]
    for dtc in dtcs:
        ys = np.asarray([mean_full_pi.get((dtc, train), float("nan")) for train in trains], dtype=np.float64)
        errs = np.asarray([std_full_pi.get((dtc, train), float("nan")) for train in trains], dtype=np.float64)
        xs = np.arange(len(trains))
        ax.plot(xs, ys, marker="o", label=f"dtc={dtc}", color=colors.get(dtc, "#888888"))
        finite = np.isfinite(ys) & np.isfinite(errs)
        if np.any(finite):
            ax.fill_between(xs[finite], np.maximum(0.0, ys[finite] - errs[finite]), ys[finite] + errs[finite], alpha=0.15, color=colors.get(dtc, "#888888"))
    ax.set_xticks(range(len(trains)))
    ax.set_xticklabels([str(x) for x in trains])
    ax.set_xlabel("train_docs")
    ax.set_ylabel("full_doc_operator pi_l1_to_full")
    ax.set_title("Full-Document Neural Baseline")
    ax.legend(frameon=False, fontsize=8)

    stale_rows = [row for row in learned_rows if _stale_exact_capacity(row)]
    stale_groups = Counter((row["leaf"], row["train"], row["quad_weight"]) for row in stale_rows)
    labels = [f"L{leaf}/T{train}/Q{quad_weight:g}" for leaf, train, quad_weight in sorted(stale_groups)]
    vals = [stale_groups[k] for k in sorted(stale_groups)]
    ax = axs[1]
    if labels:
        ax.bar(range(len(labels)), vals, color="#E15759")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("row count")
        ax.set_title("Stale Exact-Capacity Learned Rows")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No stale exact-capacity rows detected.", ha="center", va="center")

    fig.suptitle("Issues That Affect How Much More Sweep Volume Helps", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_summary_lines(
    *,
    input_root: Path,
    spec: Dict[str, str],
    expected: Dict[str, int],
    exact_rows: Sequence[dict],
    learned_cpu_rows: Sequence[dict],
    learned_gpu_rows: Sequence[dict],
) -> List[str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        f"Diagnostic LDA Tree-Recovery Report — {input_root.name}",
        f"Generated: {now}",
        "",
        "Note: this sweep varies a quadratic utility weight, not the paper local-law lambda.",
        "",
        "Progress:",
        f"  exact_cpu            {len(exact_rows):4d} / {expected['exact_cpu']:4d}  ({_pct(len(exact_rows), expected['exact_cpu']):5.1f}%)",
        f"  learned_cpu_shadow   {len(learned_cpu_rows):4d} / {expected['learned_cpu_shadow']:4d}  ({_pct(len(learned_cpu_rows), expected['learned_cpu_shadow']):5.1f}%)",
        f"  learned_gpu          {len(learned_gpu_rows):4d} / {expected['learned_gpu']:4d}  ({_pct(len(learned_gpu_rows), expected['learned_gpu']):5.1f}%)",
        "",
    ]

    bundle_expected = expected["learned_gpu_per_bundle"]
    bundle_counts = Counter((row["dtc"], row["seed"]) for row in learned_gpu_rows)
    lines.append("Learned GPU bundle coverage:")
    for dtc in [0.2, 0.6, 1.5]:
        vals = [bundle_counts.get((dtc, seed), 0) for seed in range(8)]
        done_bundles = sum(v == bundle_expected for v in vals)
        started_bundles = sum(v > 0 for v in vals)
        lines.append(f"  dtc={dtc}: started={started_bundles}/8, complete={done_bundles}/8, counts={vals}")
    lines.append("")

    stale_rows = [row for row in learned_gpu_rows if _stale_exact_capacity(row)]
    stale_groups = Counter((row["dtc"], row["leaf"], row["train"], row["quad_weight"]) for row in stale_rows)
    lines.append(f"Stale exact-capacity learned rows: {len(stale_rows)}")
    for (dtc, leaf, train, quad_weight), count in sorted(stale_groups.items()):
        lines.append(f"  dtc={dtc} leaf={leaf} train={train} quadratic_weight={quad_weight:g} -> {count} rows")
    lines.append("")

    dedup: Dict[Tuple[float, int, int], dict] = {}
    for row in learned_gpu_rows:
        dedup[(row["dtc"], row["train"], row["seed"])] = row
    dedup_rows = list(dedup.values())
    for dtc in sorted({row["dtc"] for row in dedup_rows}):
        vals = [row["full_pi"] for row in dedup_rows if row["dtc"] == dtc]
        lines.append(f"Full-doc operator mean pi_l1_to_full at dtc={dtc}: {_safe_mean(vals):.4f}")
    clean_tree_512 = [
        row for row in learned_gpu_rows
        if int(row["state"]) == int(row["vocab"]) and not _stale_exact_capacity(row)
    ]
    for dtc in sorted({row["dtc"] for row in clean_tree_512}):
        vals = [row["tree_pi"] for row in clean_tree_512 if row["dtc"] == dtc]
        lines.append(f"Clean tree state=512 mean pi_l1_to_full at dtc={dtc}: {_safe_mean(vals):.4e}")
    lines.append("")

    lines.extend(
        [
            "Current read:",
            "  1. The exact family is already settled: exact tree recovery is numerically exact across the completed exact sweep.",
            "  2. The tree sketch branch is already informative: dtc=0.2 is fully covered and dtc=0.6 is well underway.",
            "  3. More GPU volume is still useful for robustness to higher doc-topic concentration, especially dtc=1.5 which has not started.",
            "  4. More volume is not the main fix for the full-doc neural operator; that branch is still materially off the exact LDA target.",
            "  5. Some low-train exact-capacity learned rows were produced by the old singleton SVD path and should be rerun before using them as the exact-capacity check.",
        ]
    )
    return lines


def _remaining_usefulness_lines(
    *,
    expected: Dict[str, int],
    learned_gpu_rows: Sequence[dict],
) -> List[str]:
    bundle_expected = expected["learned_gpu_per_bundle"]
    bundle_counts = Counter((row["dtc"], row["seed"]) for row in learned_gpu_rows)
    stale_rows = [row for row in learned_gpu_rows if _stale_exact_capacity(row)]
    dedup: Dict[Tuple[float, int, int], dict] = {}
    for row in learned_gpu_rows:
        dedup[(row["dtc"], row["train"], row["seed"])] = row
    full_doc_vals = defaultdict(list)
    for row in dedup.values():
        full_doc_vals[row["dtc"]].append(row["full_pi"])

    lines = [
        "What remains useful from the overnight sweep?",
        "",
        "Still useful:",
        "  • Filling dtc=0.6 to all 8 seeds, because that turns the current partial robustness check into a clean matched comparison.",
        "  • Running dtc=1.5 at all, because the current report says nothing yet about the high-mixing regime.",
        "  • Rerunning the stale exact-capacity rows from the pre-restart singleton path, because they understate what state_dim=512 should achieve.",
        "",
        "Less useful without code changes:",
        "  • Simply adding more volume for the full-doc operator. The current full-doc branch is still far from the exact LDA target on the completed rows.",
        "",
        "Bundle status:",
    ]
    for dtc in [0.2, 0.6, 1.5]:
        vals = [bundle_counts.get((dtc, seed), 0) for seed in range(8)]
        lines.append(f"  dtc={dtc}: bundle counts {vals} (target per bundle = {bundle_expected})")
    lines.append("")
    lines.append(f"Stale exact-capacity rows currently present: {len(stale_rows)}")
    for dtc in sorted(full_doc_vals):
        lines.append(f"Full-doc operator mean pi_l1_to_full at dtc={dtc}: {_safe_mean(full_doc_vals[dtc]):.4f}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else (input_root / "report")
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _parse_spec(input_root / "sweep_spec.txt")
    expected = _expected_counts(spec)

    exact_rows = _load_exact_rows(input_root / "exact_cpu" / "results")
    learned_cpu_rows = _load_learned_rows(input_root / "learned_cpu_shadow" / "results", lane="learned_cpu_shadow")
    learned_gpu_rows = _load_learned_rows(input_root / "learned_gpu" / "results", lane="learned_gpu")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": str(input_root),
        "counts": {
            "exact_cpu_done": len(exact_rows),
            "learned_cpu_shadow_done": len(learned_cpu_rows),
            "learned_gpu_done": len(learned_gpu_rows),
            **expected,
        },
        "done_counts": {
            "exact_cpu": len(exact_rows),
            "learned_cpu_shadow": len(learned_cpu_rows),
            "learned_gpu": len(learned_gpu_rows),
        },
        "expected_counts": dict(expected),
        "stale_exact_capacity_rows": int(sum(1 for row in learned_gpu_rows if _stale_exact_capacity(row))),
    }
    (output_dir / "lda_tree_recovery_progress_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    pdf_path = output_dir / "lda_tree_recovery_progress_report.pdf"
    with PdfPages(pdf_path) as pdf:
        _text_page(
            pdf,
            title="LDA Tree-Recovery Progress Summary",
            lines=_build_summary_lines(
                input_root=input_root,
                spec=spec,
                expected=expected,
                exact_rows=exact_rows,
                learned_cpu_rows=learned_cpu_rows,
                learned_gpu_rows=learned_gpu_rows,
            ),
            font_size=10,
        )
        _coverage_page(
            pdf,
            expected=expected,
            exact_rows=exact_rows,
            learned_cpu_rows=learned_cpu_rows,
            learned_gpu_rows=learned_gpu_rows,
            bundle_expected=expected["learned_gpu_per_bundle"],
        )
        _exact_family_page(pdf, exact_rows)
        _learned_tree_page(pdf, rows=learned_gpu_rows, dtc=0.2, title_suffix="complete 8-seed slice")
        partial_06 = [row for row in learned_gpu_rows if row["dtc"] == 0.6]
        if partial_06:
            _learned_tree_page(pdf, rows=partial_06, dtc=0.6, title_suffix="partial slice")
        _full_doc_and_stale_page(pdf, learned_gpu_rows)
        _text_page(
            pdf,
            title="What Remaining Runs Still Buy Us",
            lines=_remaining_usefulness_lines(expected=expected, learned_gpu_rows=learned_gpu_rows),
            font_size=10,
        )

    print(f"wrote_pdf | {pdf_path}")
    print(f"wrote_json | {output_dir / 'lda_tree_recovery_progress_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
