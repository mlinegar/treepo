from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from treepo.bench.io import atomic_write_text, dump_json, write_csv_rows


@dataclass(frozen=True)
class CardinalityRow:
    seed: int
    audit_policy: str
    state_dim: int
    train_docs: int
    learned_relative_rmse: float
    hll_relative_rmse: float
    exact_set_relative_rmse: float
    sum_leaf_uniques_relative_rmse: float
    distance_to_hll_floor_rel_rmse: float
    ratio_to_hll_floor_rel_rmse: float
    train_total_queries_estimate: int


def _scan_cardinality_context(output_root: Path) -> Optional[dict]:
    for path in Path(output_root).rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        results = payload.get("results")
        config = payload.get("config")
        if not isinstance(results, list) or not isinstance(config, dict):
            continue
        if len(results) == 0 or not isinstance(results[0], dict):
            continue
        if "learned_metrics" not in results[0]:
            continue
        min_tokens = int(config.get("min_tokens", 0) or 0)
        max_tokens = int(config.get("max_tokens", 0) or 0)
        leaf_size = int(config.get("leaf_size", 0) or 0)
        min_leaves = int(math.ceil(min_tokens / leaf_size)) if min_tokens > 0 and leaf_size > 0 else None
        max_leaves = int(math.ceil(max_tokens / leaf_size)) if max_tokens > 0 and leaf_size > 0 else None
        return {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "leaf_size": leaf_size,
            "min_leaves": min_leaves,
            "max_leaves": max_leaves,
            "state_dims": list(config.get("state_dims", [])),
        }
    return None


def _scan_cardinality_rows(output_root: Path) -> List[CardinalityRow]:
    rows: List[CardinalityRow] = []
    for path in Path(output_root).rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        results = payload.get("results")
        config = payload.get("config")
        if not isinstance(results, list) or not isinstance(config, dict):
            continue
        if len(results) == 0 or not isinstance(results[0], dict):
            continue
        if "learned_metrics" not in results[0]:
            continue
        audit_policy = str(config.get("audit_policy", "unknown"))
        seed = int(config.get("seed", 0))
        for row in results:
            learned = row.get("learned_metrics", {})
            hll = row.get("hll_metrics", {})
            exact = row.get("exact_set_metrics", {})
            wrong = row.get("sum_leaf_uniques_metrics", {})
            rows.append(
                CardinalityRow(
                    seed=seed,
                    audit_policy=audit_policy,
                    state_dim=int(row["state_dim"]),
                    train_docs=int(row["train_size"]),
                    learned_relative_rmse=float(learned.get("relative_rmse", np.nan)),
                    hll_relative_rmse=float(hll.get("relative_rmse", np.nan)),
                    exact_set_relative_rmse=float(exact.get("relative_rmse", np.nan)),
                    sum_leaf_uniques_relative_rmse=float(
                        wrong.get("relative_rmse", np.nan)
                    ),
                    distance_to_hll_floor_rel_rmse=float(
                        row.get("distance_to_hll_floor_rel_rmse", np.nan)
                    ),
                    ratio_to_hll_floor_rel_rmse=float(
                        row.get(
                            "ratio_to_hll_floor_rel_rmse",
                            row.get("ratio_to_floor_rel_rmse", np.nan),
                        )
                    ),
                    train_total_queries_estimate=int(row.get("train_total_queries_estimate", 0)),
                )
            )
    return rows


def _scan_hll_merge_rows(output_root: Path) -> List[dict]:
    rows: List[dict] = []
    for path in Path(output_root).rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        candidate_lists: List[object] = []
        candidate_lists.append(payload.get("rows"))
        candidate_lists.append(payload.get("raw_rows"))
        candidate_lists.append(payload.get("aggregated_rows"))
        config = payload.get("config")
        if not isinstance(config, dict):
            continue
        for summary_rows in candidate_lists:
            if not isinstance(summary_rows, list):
                continue
            if len(summary_rows) == 0 or not isinstance(summary_rows[0], dict):
                continue
            if "precision" not in summary_rows[0]:
                continue
            rows.extend(summary_rows)
            break
    return rows


def _scan_hll_merge_context(output_root: Path) -> Optional[dict]:
    for path in Path(output_root).rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        config = payload.get("config")
        if not isinstance(config, dict):
            continue
        candidate_lists = [payload.get("rows"), payload.get("raw_rows"), payload.get("aggregated_rows")]
        is_hll_payload = False
        for summary_rows in candidate_lists:
            if isinstance(summary_rows, list) and summary_rows and isinstance(summary_rows[0], dict):
                if "precision" in summary_rows[0]:
                    is_hll_payload = True
                    break
        if not is_hll_payload:
            continue
        min_tokens = int(config.get("min_tokens", 0) or 0)
        max_tokens = int(config.get("max_tokens", 0) or 0)
        leaf_size = int(config.get("leaf_size", 0) or 0)
        min_leaves = int(math.ceil(min_tokens / leaf_size)) if min_tokens > 0 and leaf_size > 0 else None
        max_leaves = int(math.ceil(max_tokens / leaf_size)) if max_tokens > 0 and leaf_size > 0 else None
        return {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "leaf_size": leaf_size,
            "min_leaves": min_leaves,
            "max_leaves": max_leaves,
            "n_seeds": len(config.get("seeds", [])) if isinstance(config.get("seeds"), list) else None,
        }
    return None


def _aggregate_cardinality(rows: Sequence[CardinalityRow]) -> List[dict]:
    grouped: Dict[Tuple[str, int, int], List[CardinalityRow]] = {}
    for row in rows:
        grouped.setdefault((row.audit_policy, row.state_dim, row.train_docs), []).append(row)

    out: List[dict] = []
    for (audit_policy, state_dim, train_docs), grows in sorted(grouped.items()):
        def _arr(fn):
            return np.asarray([float(fn(r)) for r in grows], dtype=np.float64)

        learned = _arr(lambda r: r.learned_relative_rmse)
        floor_gap = _arr(lambda r: r.distance_to_hll_floor_rel_rmse)
        ratio = _arr(lambda r: r.ratio_to_hll_floor_rel_rmse)
        queries = _arr(lambda r: r.train_total_queries_estimate)
        out.append(
            {
                "audit_policy": audit_policy,
                "state_dim": int(state_dim),
                "train_docs": int(train_docs),
                "n_seeds": int(len(grows)),
                "learned_relative_rmse_mean": float(np.mean(learned)),
                "learned_relative_rmse_std": float(np.std(learned, ddof=0)),
                "hll_relative_rmse_mean": float(np.mean(_arr(lambda r: r.hll_relative_rmse))),
                "exact_set_relative_rmse_mean": float(
                    np.mean(_arr(lambda r: r.exact_set_relative_rmse))
                ),
                "sum_leaf_uniques_relative_rmse_mean": float(
                    np.mean(_arr(lambda r: r.sum_leaf_uniques_relative_rmse))
                ),
                "distance_to_hll_floor_rel_rmse_mean": float(np.mean(floor_gap)),
                "distance_to_hll_floor_rel_rmse_std": float(np.std(floor_gap, ddof=0)),
                "ratio_to_hll_floor_rel_rmse_mean": float(np.mean(ratio)),
                "train_total_queries_estimate_mean": float(np.mean(queries)),
            }
        )
    return out


def _choose_reference_audit(rows: Sequence[dict]) -> str:
    audits = [str(r["audit_policy"]) for r in rows]
    if "all" in audits:
        return "all"
    return str(sorted(set(audits))[0]) if audits else "all"


def _cardinality_subtitle(context: Optional[dict]) -> Optional[str]:
    if context is None:
        return "state_dim = learned sketch width, not token count."
    min_tokens = context.get("min_tokens")
    max_tokens = context.get("max_tokens")
    leaf_size = context.get("leaf_size")
    min_leaves = context.get("min_leaves")
    max_leaves = context.get("max_leaves")
    if not all(isinstance(x, int) and x > 0 for x in (min_tokens, max_tokens, leaf_size)):
        return "Legend: state_dim = learned sketch width, not token count."
    return (
        f"Docs: {min_tokens}-{max_tokens} tokens; leaves: {leaf_size} tokens "
        f"({min_leaves}-{max_leaves}/doc); "
        "state_dim = learned sketch width."
    )


def _hll_subtitle(context: Optional[dict], *, n_seeds: int) -> Optional[str]:
    if context is None:
        return f"p = HLL precision, m = 2^p registers; mean over {n_seeds} seeds; band = +/-1 std."
    min_tokens = context.get("min_tokens")
    max_tokens = context.get("max_tokens")
    leaf_size = context.get("leaf_size")
    min_leaves = context.get("min_leaves")
    max_leaves = context.get("max_leaves")
    if not all(isinstance(x, int) and x > 0 for x in (min_tokens, max_tokens, leaf_size)):
        return f"p = HLL precision, m = 2^p registers; mean over {n_seeds} seeds; band = +/-1 std."
    return (
        f"Docs: {min_tokens}-{max_tokens} tokens; leaves: {leaf_size} tokens "
        f"({min_leaves}-{max_leaves}/doc); p = HLL precision, m = 2^p registers; "
        f"mean over {n_seeds} seeds; band = +/-1 std."
    )


def _plot_learning_curves(rows: Sequence[dict], output: Path, *, context: Optional[dict] = None) -> None:
    if not rows:
        return
    audit = _choose_reference_audit(rows)
    filt = [r for r in rows if str(r["audit_policy"]) == audit]
    state_dims = sorted({int(r["state_dim"]) for r in filt})
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for state_dim in state_dims:
        srows = sorted(
            [r for r in filt if int(r["state_dim"]) == state_dim],
            key=lambda r: int(r["train_docs"]),
        )
        xs = np.asarray([int(r["train_docs"]) for r in srows], dtype=np.float64)
        ys = np.asarray([float(r["distance_to_hll_floor_rel_rmse_mean"]) for r in srows], dtype=np.float64)
        ystd = np.asarray([float(r["distance_to_hll_floor_rel_rmse_std"]) for r in srows], dtype=np.float64)
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=f"state_dim={state_dim}")
        ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.10)
    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Train docs")
    ax.set_ylabel("Distance to HLL floor (relative RMSE)")
    title = f"Cardinality Recovery vs HLL Floor | audit={audit}"
    subtitle = _cardinality_subtitle(context)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_negative_control(rows: Sequence[dict], output: Path, *, context: Optional[dict] = None) -> None:
    if not rows:
        return
    audit = _choose_reference_audit(rows)
    filt = [r for r in rows if str(r["audit_policy"]) == audit]
    max_state = max(int(r["state_dim"]) for r in filt)
    srows = sorted(
        [r for r in filt if int(r["state_dim"]) == max_state],
        key=lambda r: int(r["train_docs"]),
    )
    xs = np.asarray([int(r["train_docs"]) for r in srows], dtype=np.float64)
    learned = np.asarray([float(r["learned_relative_rmse_mean"]) for r in srows], dtype=np.float64)
    hll = np.asarray([float(r["hll_relative_rmse_mean"]) for r in srows], dtype=np.float64)
    exact = np.asarray([float(r["exact_set_relative_rmse_mean"]) for r in srows], dtype=np.float64)
    wrong = np.asarray(
        [float(r["sum_leaf_uniques_relative_rmse_mean"]) for r in srows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(xs, learned, marker="o", linewidth=1.7, label="TreePO learned")
    ax.plot(xs, hll, linestyle="--", linewidth=1.5, label="Exact HLL")
    ax.plot(xs, exact, linestyle=":", linewidth=1.4, label="Exact set")
    ax.plot(xs, wrong, marker="s", linewidth=1.4, label="Wrong baseline")
    ax.set_xlabel("Train docs")
    ax.set_ylabel("Relative RMSE")
    title = f"Cardinality Recovery Baselines | audit={audit}, state_dim={max_state}"
    subtitle = _cardinality_subtitle(context)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_hll_merge_memory(rows: Sequence[dict], output: Path, *, context: Optional[dict] = None) -> None:
    if not rows:
        return
    train_docs = max(int(r["train_docs"]) for r in rows)
    filt = [r for r in rows if int(r["train_docs"]) == train_docs and str(r["audit_policy"]) == "all"]
    if not filt:
        filt = [r for r in rows if int(r["train_docs"]) == train_docs]
    grouped: Dict[int, List[dict]] = {}
    for row in filt:
        grouped.setdefault(int(row["precision"]), []).append(row)

    def _metric(row: dict, base: str) -> float:
        if f"{base}_mean" in row and row[f"{base}_mean"] not in ("", None):
            return float(row[f"{base}_mean"])
        return float(row[base])

    plot_rows: List[dict] = []
    for precision, grows in sorted(grouped.items()):
        learned_vals = np.asarray([_metric(row, "learned_relative_rmse") for row in grows], dtype=np.float64)
        hll_vals = np.asarray([_metric(row, "hll_relative_rmse") for row in grows], dtype=np.float64)
        plot_rows.append(
            {
                "precision": int(precision),
                "memory_bytes": float(grows[0]["memory_bits"]) / 8.0,
                "learned_relative_rmse": float(np.mean(learned_vals)),
                "learned_relative_rmse_std": float(np.std(learned_vals, ddof=0)),
                "hll_relative_rmse": float(np.mean(hll_vals)),
                "hll_rse_theory": float(np.mean([float(row["hll_rse_theory"]) for row in grows])),
                "n_points": int(len(grows)),
            }
        )

    xs = np.asarray([row["memory_bytes"] for row in plot_rows], dtype=np.float64)
    learned = np.asarray([row["learned_relative_rmse"] for row in plot_rows], dtype=np.float64)
    learned_std = np.asarray([row["learned_relative_rmse_std"] for row in plot_rows], dtype=np.float64)
    hll = np.asarray([row["hll_relative_rmse"] for row in plot_rows], dtype=np.float64)
    floor = np.asarray([row["hll_rse_theory"] for row in plot_rows], dtype=np.float64)
    n_seeds = max(int(row["n_points"]) for row in plot_rows)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(xs, learned, marker="o", linewidth=1.7, label="Learned merge")
    ax.fill_between(xs, learned - learned_std, learned + learned_std, alpha=0.12)
    ax.plot(xs, hll, linestyle="--", linewidth=1.4, label="Exact HLL")
    ax.plot(xs, floor, linestyle=":", linewidth=1.3, label="Theory floor")
    for row in plot_rows:
        ax.annotate(
            f"p={row['precision']}",
            (row["memory_bytes"], row["learned_relative_rmse"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("HLL memory (bytes)")
    ax.set_ylabel("Relative RMSE")
    title = f"HLL Merge Learning | train_docs={train_docs}"
    subtitle = _hll_subtitle(context, n_seeds=n_seeds)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_hll_merge_memory_median(rows: Sequence[dict], output: Path, *, context: Optional[dict] = None) -> None:
    if not rows:
        return
    train_docs = max(int(r["train_docs"]) for r in rows)
    filt = [r for r in rows if int(r["train_docs"]) == train_docs and str(r["audit_policy"]) == "all"]
    if not filt:
        filt = [r for r in rows if int(r["train_docs"]) == train_docs]
    grouped: Dict[int, List[dict]] = {}
    for row in filt:
        grouped.setdefault(int(row["precision"]), []).append(row)

    def _metric(row: dict, base: str) -> float:
        if f"{base}_mean" in row and row[f"{base}_mean"] not in ("", None):
            return float(row[f"{base}_mean"])
        return float(row[base])

    plot_rows: List[dict] = []
    for precision, grows in sorted(grouped.items()):
        learned_vals = np.asarray([_metric(row, "learned_relative_rmse") for row in grows], dtype=np.float64)
        hll_vals = np.asarray([_metric(row, "hll_relative_rmse") for row in grows], dtype=np.float64)
        plot_rows.append(
            {
                "precision": int(precision),
                "memory_bytes": float(grows[0]["memory_bits"]) / 8.0,
                "learned_relative_rmse_median": float(np.median(learned_vals)),
                "learned_relative_rmse_p10": float(np.percentile(learned_vals, 10.0)),
                "learned_relative_rmse_p90": float(np.percentile(learned_vals, 90.0)),
                "hll_relative_rmse_median": float(np.median(hll_vals)),
                "hll_rse_theory": float(np.mean([float(row["hll_rse_theory"]) for row in grows])),
                "n_points": int(len(grows)),
            }
        )

    xs = np.asarray([row["memory_bytes"] for row in plot_rows], dtype=np.float64)
    learned = np.asarray([row["learned_relative_rmse_median"] for row in plot_rows], dtype=np.float64)
    learned_p10 = np.asarray([row["learned_relative_rmse_p10"] for row in plot_rows], dtype=np.float64)
    learned_p90 = np.asarray([row["learned_relative_rmse_p90"] for row in plot_rows], dtype=np.float64)
    hll = np.asarray([row["hll_relative_rmse_median"] for row in plot_rows], dtype=np.float64)
    floor = np.asarray([row["hll_rse_theory"] for row in plot_rows], dtype=np.float64)
    n_seeds = max(int(row["n_points"]) for row in plot_rows)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(xs, learned, marker="o", linewidth=1.7, label="Learned merge (median)")
    ax.fill_between(xs, learned_p10, learned_p90, alpha=0.12)
    ax.plot(xs, hll, linestyle="--", linewidth=1.4, label="Exact HLL")
    ax.plot(xs, floor, linestyle=":", linewidth=1.3, label="Theory floor")
    for row in plot_rows:
        ax.annotate(
            f"p={row['precision']}",
            (row["memory_bytes"], row["learned_relative_rmse_median"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("HLL memory (bytes)")
    ax.set_ylabel("Relative RMSE")
    title = f"HLL Merge Learning | train_docs={train_docs} | median over seeds"
    subtitle = _hll_subtitle(context, n_seeds=n_seeds)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate TreePO cardinality/HLL report artifacts.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.output_root) / "figures" / "cardinality"
    out_dir.mkdir(parents=True, exist_ok=True)

    card_context = _scan_cardinality_context(Path(args.output_root))
    card_rows = _scan_cardinality_rows(Path(args.output_root))
    agg_rows = _aggregate_cardinality(card_rows)
    hll_context = _scan_hll_merge_context(Path(args.output_root))
    hll_rows = _scan_hll_merge_rows(Path(args.output_root))

    raw_card_csv = out_dir / "cardinality_recovery_raw.csv"
    agg_card_csv = out_dir / "cardinality_recovery_agg.csv"
    diag_json = out_dir / "cardinality_latest_diagnostics.json"
    md_path = out_dir / "cardinality_latest.md"

    write_csv_rows(raw_card_csv, [r.__dict__ for r in card_rows])
    write_csv_rows(agg_card_csv, agg_rows)
    _plot_learning_curves(agg_rows, out_dir / "cardinality_learning_curves.png", context=card_context)
    _plot_negative_control(agg_rows, out_dir / "cardinality_negative_control.png", context=card_context)
    if hll_rows:
        write_csv_rows(out_dir / "hll_merge_learning_raw.csv", hll_rows)
        _plot_hll_merge_memory(hll_rows, out_dir / "hll_merge_learning_memory.png", context=hll_context)
        _plot_hll_merge_memory_median(
            hll_rows,
            out_dir / "hll_merge_learning_memory_median.png",
            context=hll_context,
        )

    diagnostics = {
        "n_cardinality_rows": int(len(card_rows)),
        "n_cardinality_agg_rows": int(len(agg_rows)),
        "n_hll_merge_rows": int(len(hll_rows)),
        "reference_audit_policy": _choose_reference_audit(agg_rows) if agg_rows else None,
        "outputs": {
            "cardinality_recovery_raw_csv": str(raw_card_csv),
            "cardinality_recovery_agg_csv": str(agg_card_csv),
            "diagnostics_json": str(diag_json),
            "report_md": str(md_path),
            "hll_merge_learning_memory_png": str(out_dir / "hll_merge_learning_memory.png"),
            "hll_merge_learning_memory_median_png": str(out_dir / "hll_merge_learning_memory_median.png"),
        },
    }
    atomic_write_text(diag_json, dump_json(diagnostics))

    best_line = "no cardinality rows found"
    if agg_rows:
        best = min(agg_rows, key=lambda r: float(r["distance_to_hll_floor_rel_rmse_mean"]))
        best_line = (
            f"best floor gap: audit={best['audit_policy']}, state={best['state_dim']}, "
            f"train_docs={best['train_docs']}, gap={best['distance_to_hll_floor_rel_rmse_mean']:.4f}"
        )
    md = "\n".join(
        [
            "# Cardinality Report",
            "",
            f"- Cardinality rows: `{len(card_rows)}`",
            f"- Aggregated rows: `{len(agg_rows)}`",
            f"- HLL merge-learning rows: `{len(hll_rows)}`",
            f"- {best_line}",
            "",
            "Artifacts:",
            f"- `{raw_card_csv.name}`",
            f"- `{agg_card_csv.name}`",
            "- `cardinality_learning_curves.png`",
            "- `cardinality_negative_control.png`",
            "- `hll_merge_learning_memory.png`",
            "- `hll_merge_learning_memory_median.png`",
            "- `cardinality_latest_diagnostics.json`",
        ]
    )
    atomic_write_text(md_path, md + "\n")
    return 0


__all__ = ["main"]
