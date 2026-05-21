"""Precision-floor recovery utilities for HLL register-state experiments."""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from treepo._research.unified_g_v1.sketch.learned_hll_parity import learned_hll_parity_task
from treepo._research.unified_g_v1.training.fit import fit
from treepo._research.unified_g_v1.training.tree_task import TreeExample


MAIN_METHOD = "learned_g_oracle_state"
DIAGNOSTIC_METHODS = ("learned_g", "learned_joint")


@dataclass(frozen=True)
class HLLPrecisionFloorConfig:
    precisions: tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)
    leaf_counts: tuple[int, ...] = (1, 2, 4, 8, 16)
    n_train: int = 8192
    n_val: int = 1024
    min_tokens: int = 128
    max_tokens: int = 128
    universe_size: int = 10_000
    seed: int = 0
    n_epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    local_law_weight: float = 0.9
    merge_state_weight: float = 100.0
    use_cuda: bool = False
    cuda_device: int | None = None
    include_token_diagnostics: bool = False


def _hll_rse_theory(precision: int) -> float:
    return 1.04 / math.sqrt(float(1 << int(precision)))


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values.astype(np.float64)))


def _relative_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.size == 0:
        return 0.0
    denom = max(1.0, float(np.mean(np.abs(targets.astype(np.float64)))))
    return float(np.sqrt(np.mean((preds - targets) ** 2)) / denom)


def _relative_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.size == 0:
        return 0.0
    denom = max(1.0, float(np.mean(np.abs(targets.astype(np.float64)))))
    return float(np.mean(np.abs(preds - targets)) / denom)


def _predict(model: Any, items: Sequence[TreeExample], *, batch_size: int) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(items), bs):
            batch = list(items[start : start + bs])
            _root_state, prediction, _aux = model.forward_tree(batch)
            chunks.append(prediction.detach().cpu().double().numpy().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def _targets(items: Sequence[TreeExample]) -> tuple[np.ndarray, np.ndarray]:
    hll_reference = np.asarray([float(item.target) for item in items], dtype=np.float64)
    analytic = np.asarray(
        [float(item.extra.get("analytic_root_cardinality", item.target)) for item in items],
        dtype=np.float64,
    )
    return hll_reference, analytic


def _claim_role(method: str) -> str:
    return "main_register_state" if method == MAIN_METHOD else "diagnostic_token_encoder"


def _representation_surface(method: str) -> str:
    if method == MAIN_METHOD:
        return "lossless_hll_register_state"
    return "pooled_learned_token_embedding"


def run_precision_floor_cell(
    *,
    method: str,
    precision: int,
    n_leaves: int,
    config: HLLPrecisionFloorConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one HLL recovery cell and evaluate both reference and floor metrics."""

    method = str(method)
    if method == MAIN_METHOD:
        merge_state_weight = float(config.merge_state_weight)
    else:
        merge_state_weight = 1.0

    cfg = learned_hll_parity_task(
        method=method,
        precision=int(precision),
        n_leaves=int(n_leaves),
        backend="native",
        oracle_kind="hll_reference",
        n_train=int(config.n_train),
        n_val=int(config.n_val),
        seed=int(config.seed),
        universe_size=int(config.universe_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        hidden_dim=int(config.hidden_dim),
        n_epochs=int(config.n_epochs),
        train_batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        local_law_weight=float(config.local_law_weight),
        merge_state_relative_weight=merge_state_weight,
        use_cuda=bool(config.use_cuda),
        cuda_device=config.cuda_device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = fit(trainer_config=cfg, output_dir=output_dir)
    wall_seconds = float(time.perf_counter() - t0)

    val_items = list(cfg.oracle.val_examples())
    preds = _predict(cfg.model, val_items, batch_size=int(config.batch_size))
    hll_targets, analytic_targets = _targets(val_items)

    hll_ref_mae = float(np.mean(np.abs(preds - hll_targets))) if preds.size else 0.0
    hll_ref_rmse = float(np.sqrt(np.mean((preds - hll_targets) ** 2))) if preds.size else 0.0
    learned_analytic_rmse = _relative_rmse(preds, analytic_targets)
    official_hll_rmse = _relative_rmse(hll_targets, analytic_targets)
    hll_rse = _hll_rse_theory(int(precision))
    last_history = dict(result.history[-1]) if result.history else {}

    return {
        "method": method,
        "claim_role": _claim_role(method),
        "representation_surface": _representation_surface(method),
        "precision": int(precision),
        "hll_registers": int(1 << int(precision)),
        "n_leaves": int(n_leaves),
        "seed": int(config.seed),
        "n_train": int(config.n_train),
        "n_val": int(config.n_val),
        "min_tokens": int(config.min_tokens),
        "max_tokens": int(config.max_tokens),
        "universe_size": int(config.universe_size),
        "n_epochs": int(config.n_epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "hll_reference_root_mae": hll_ref_mae,
        "hll_reference_root_rmse": hll_ref_rmse,
        "hll_reference_root_rel_mae": _relative_mae(preds, hll_targets),
        "hll_reference_root_rel_rmse": _relative_rmse(preds, hll_targets),
        "analytic_relative_rmse": learned_analytic_rmse,
        "analytic_relative_mae": _relative_mae(preds, analytic_targets),
        "official_hll_relative_rmse": official_hll_rmse,
        "official_hll_relative_mae": _relative_mae(hll_targets, analytic_targets),
        "hll_rse_theory": hll_rse,
        "distance_to_hll": float(learned_analytic_rmse - official_hll_rmse),
        "distance_to_theory": float(learned_analytic_rmse - hll_rse),
        "official_distance_to_theory": float(official_hll_rmse - hll_rse),
        "target_mean": _safe_mean(analytic_targets),
        "hll_reference_mean": _safe_mean(hll_targets),
        "prediction_mean": _safe_mean(preds),
        "best_epoch": float(result.metrics.get("best_epoch", last_history.get("epoch", 0.0))),
        "best_hll_reference_mae": float(result.metrics.get("best_metric_value", hll_ref_mae)),
        "final_hll_reference_mae": hll_ref_mae,
        "wall_seconds": wall_seconds,
        "output_dir": str(output_dir),
    }


def run_precision_floor_sweep(
    config: HLLPrecisionFloorConfig,
    *,
    output_root: Path,
    jobs: int = 1,
    cuda_devices: Sequence[int] = (),
) -> list[dict[str, Any]]:
    methods = [MAIN_METHOD]
    if config.include_token_diagnostics:
        methods.extend(DIAGNOSTIC_METHODS)
    specs: list[tuple[str, int, int, HLLPrecisionFloorConfig, Path]] = []
    device_cycle = tuple(int(x) for x in cuda_devices)
    cell_idx = 0
    for method in methods:
        for precision in config.precisions:
            for n_leaves in config.leaf_counts:
                cell_config = config
                if bool(config.use_cuda) and device_cycle:
                    cell_config = replace(config, cuda_device=device_cycle[cell_idx % len(device_cycle)])
                cell_dir = output_root / "cells" / f"{method}_p{int(precision)}_L{int(n_leaves)}_seed{int(config.seed)}"
                specs.append((method, int(precision), int(n_leaves), cell_config, cell_dir))
                cell_idx += 1

    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(jobs))
    if max_workers == 1:
        for method, precision, n_leaves, cell_config, cell_dir in specs:
            rows.append(
                run_precision_floor_cell(
                    method=method,
                    precision=precision,
                    n_leaves=n_leaves,
                    config=cell_config,
                    output_dir=cell_dir,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(specs))) as ex:
            futures = [
                ex.submit(
                    run_precision_floor_cell,
                    method=method,
                    precision=precision,
                    n_leaves=n_leaves,
                    config=cell_config,
                    output_dir=cell_dir,
                )
                for method, precision, n_leaves, cell_config, cell_dir in specs
            ]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(
        key=lambda row: (
            str(row.get("method", "")),
            int(row.get("precision", 0)),
            int(row.get("n_leaves", 0)),
            int(row.get("seed", 0)),
        )
    )
    return rows


def write_precision_floor_outputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    config: HLLPrecisionFloorConfig,
) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "hll_precision_floor_recovery.csv"
    json_path = output_root / "hll_precision_floor_recovery.json"
    fieldnames = [
        "method",
        "claim_role",
        "representation_surface",
        "precision",
        "hll_registers",
        "n_leaves",
        "seed",
        "n_train",
        "n_val",
        "min_tokens",
        "max_tokens",
        "universe_size",
        "n_epochs",
        "batch_size",
        "learning_rate",
        "hll_reference_root_mae",
        "hll_reference_root_rmse",
        "hll_reference_root_rel_mae",
        "hll_reference_root_rel_rmse",
        "analytic_relative_rmse",
        "analytic_relative_mae",
        "official_hll_relative_rmse",
        "official_hll_relative_mae",
        "hll_rse_theory",
        "distance_to_hll",
        "distance_to_theory",
        "official_distance_to_theory",
        "target_mean",
        "hll_reference_mean",
        "prediction_mean",
        "best_epoch",
        "best_hll_reference_mae",
        "final_hll_reference_mae",
        "wall_seconds",
        "output_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    json_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "rows": list(rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "json": str(json_path)}


def _best_main_by_precision(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    best: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("method", "")) != MAIN_METHOD:
            continue
        p = int(row["precision"])
        current = best.get(p)
        if current is None or float(row.get("analytic_relative_rmse", math.inf)) < float(
            current.get("analytic_relative_rmse", math.inf)
        ):
            best[p] = row
    return [best[p] for p in sorted(best)]


def plot_precision_floor_recovery(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_stem: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    try:
        from paper.ctreepo.scripts import paperplot
    except Exception:  # pragma: no cover - fallback for unusual import paths
        paperplot = None

    if paperplot is not None:
        paperplot.rcparams()
        figsize = paperplot.FIGSIZE_FULL
        save = paperplot.save
        colors = paperplot.ANCHOR_COLORS
    else:
        figsize = (7.0, 3.0)
        colors = {"hll": "#d95f02", "baseline": "#999999", "theory": "#555555"}

        def save(fig, output_stem, *, formats=("pdf", "png")):
            written = []
            for fmt in formats:
                path = Path(output_stem).with_suffix(f".{fmt}")
                fig.savefig(path, bbox_inches="tight", dpi=300)
                written.append(path)
            return written

    plot_rows = _best_main_by_precision(rows)
    if not plot_rows:
        return []
    xs = np.asarray([int(row["precision"]) for row in plot_rows], dtype=np.int64)
    learned = np.asarray([float(row["analytic_relative_rmse"]) for row in plot_rows])
    official = np.asarray([float(row["official_hll_relative_rmse"]) for row in plot_rows])
    theory = np.asarray([float(row["hll_rse_theory"]) for row in plot_rows])
    parity = np.asarray([float(row["hll_reference_root_rel_rmse"]) for row in plot_rows])

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    ax.plot(xs, official, marker="o", color=colors.get("baseline", "#999999"), label="Official HLL")
    ax.plot(xs, learned, marker="s", color=colors.get("hll", "#d95f02"), label=r"Learned $g$ + fixed $f^*$")
    ax.plot(xs, theory, linestyle=":", color=colors.get("theory", "#555555"), label="HLL RSE theory")
    ax.set_xlabel("HLL precision p")
    ax.set_ylabel("Relative RMSE vs truth")
    ax.set_title("Precision floor")
    ax.set_xticks(xs)

    ax = axes[1]
    ax.plot(xs, parity, marker="s", color=colors.get("hll", "#d95f02"), label=r"Learned $g$")
    ax.axhline(0.0, linestyle=":", color=colors.get("baseline", "#999999"), label="Official parity")
    ax.set_xlabel("HLL precision p")
    ax.set_ylabel("Relative RMSE vs flat HLL")
    ax.set_title("Supplied-oracle parity")
    ax.set_xticks(xs)

    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    written = save(fig, output_stem, formats=("pdf", "png"))
    plt.close(fig)
    return [str(path) for path in written]


def stage_precision_floor_assets(*, output_stem: Path, paper_figures_dir: Path) -> list[str]:
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for suffix in (".pdf", ".png"):
        src = output_stem.with_suffix(suffix)
        if src.exists():
            dst = paper_figures_dir / src.name
            shutil.copy2(src, dst)
            staged.append(str(dst))
    return staged


__all__ = [
    "DIAGNOSTIC_METHODS",
    "HLLPrecisionFloorConfig",
    "MAIN_METHOD",
    "plot_precision_floor_recovery",
    "run_precision_floor_cell",
    "run_precision_floor_sweep",
    "stage_precision_floor_assets",
    "write_precision_floor_outputs",
]
