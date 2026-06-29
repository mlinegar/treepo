#!/usr/bin/env python3
"""Grid learned neural operators over Markov leaf sizes."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class NeuralOperatorMarkovLeafGridConfig:
    operator_kinds: Sequence[str] = field(default_factory=lambda: ("fno", "tfno", "uno", "conv1d"))
    leaf_token_counts: Sequence[int] = field(default_factory=lambda: (32, 64, 128))
    n_train: int = 512
    n_eval: int = 128
    n_states: int = 4
    doc_tokens: int = 256
    transition_prob: float = 0.15
    vocabulary_size: int = 256
    seed: int = 0
    max_iterations: int = 4
    embedding_dim: int = 16
    hidden_channels: int = 12
    n_modes: int = 4
    n_layers: int = 1
    conv_kernel_size: int = 3
    head_hidden_dim: int = 24
    epochs_per_iteration: int = 3
    batch_size: int = 32
    learning_rate: float = 0.01
    device: str = "cpu"
    fixture_device: str | None = None
    normalize_targets: bool = True
    numeric_transition_state_weight: float = 0.05
    numeric_transition_count_scale: float | None = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("neural_operator_markov_leaf_grid.toml"),
    )
    args = ap.parse_args()

    from treepo.methods import run
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo.methods.fixtures import make_markov_changepoint_trees
    from treepo.methods.grid import iter_grid, write_grid_outputs

    cfg = load_dataclass(args.config, NeuralOperatorMarkovLeafGridConfig)
    leaf_sizes = tuple(int(value) for value in cfg.leaf_token_counts)
    operator_kinds = tuple(str(value) for value in cfg.operator_kinds)
    if not leaf_sizes:
        raise ValueError("leaf_token_counts must be non-empty")
    if not operator_kinds:
        raise ValueError("operator_kinds must be non-empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_train, reference_eval = _make_split(cfg, leaf_sizes[0], make_markov_changepoint_trees)
    average_baseline = _average_guess_baseline(reference_train, reference_eval)

    rows: list[dict[str, Any]] = []
    cells = iter_grid(
        {"leaf_token_count": leaf_sizes, "operator_kind": operator_kinds},
        output_root=args.output_dir,
    )
    for cell in cells:
        leaf_token_count = int(cell.values["leaf_token_count"])
        operator_kind = str(cell.values["operator_kind"])
        train, eval_trees = _make_split(cfg, leaf_token_count, make_markov_changepoint_trees)
        result = run(
            "fit",
            {
                "family": "neural_operator",
                "train_data": train,
                "eval_data": eval_trees,
                "backend_config": {
                    "output_dir": str(cell.output_dir / "fit"),
                    "operator_kind": operator_kind,
                    "embedding_dim": cfg.embedding_dim,
                    "hidden_channels": cfg.hidden_channels,
                    "n_modes": cfg.n_modes,
                    "n_layers": cfg.n_layers,
                    "conv_kernel_size": cfg.conv_kernel_size,
                    "head_hidden_dim": cfg.head_hidden_dim,
                    "epochs_per_iteration": cfg.epochs_per_iteration,
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "device": cfg.device,
                    "normalize_targets": cfg.normalize_targets,
                    "numeric_transition_state_weight": cfg.numeric_transition_state_weight,
                    "numeric_transition_count_scale": cfg.numeric_transition_count_scale,
                },
                "axis": {
                    "max_iterations": cfg.max_iterations,
                    "axis_kind": "leaf_token_count",
                    "axis_value": leaf_token_count,
                    "leaf_size_tokens": leaf_token_count,
                },
            },
        )
        metrics = dict(result.metrics or {})
        artifacts = dict(result.artifacts or {})
        f_artifact = dict(artifacts.get("f") or {})
        g_artifact = dict(artifacts.get("g") or {})
        rows.append(
            {
                "leaf_token_count": int(leaf_token_count),
                "leaves_per_doc": int(math.ceil(int(cfg.doc_tokens) / int(leaf_token_count))),
                "operator_kind": operator_kind,
                "status": result.status,
                "internal_f_mae": _metric(metrics, "internal_f_mae"),
                "internal_f_pearson": _metric(metrics, "internal_f_pearson"),
                "mean_prediction": _metric(metrics, "mean_prediction"),
                "mean_teacher": _metric(metrics, "mean_teacher"),
                "n": _metric(metrics, "n"),
                "average_guess_target": float(average_baseline["target"]),
                "average_guess_mae": float(average_baseline["mae"]),
                "f_loss": _metric(f_artifact, "loss"),
                "g_loss": _metric(g_artifact, "loss"),
                "g_trained": g_artifact.get("trained"),
                "normalize_targets": g_artifact.get("normalize_targets"),
                "numeric_transition_state_weight": g_artifact.get("numeric_transition_state_weight"),
                "artifact_kind": _artifact_kind(result),
                "manifest_path": result.manifest_path,
            }
        )

    json_out = args.output_dir / "neural_operator_markov_leaf_grid.json"
    csv_out = args.output_dir / "neural_operator_markov_leaf_grid.csv"
    payload = {
        "config": asdict(cfg),
        "average_guess_baseline": average_baseline,
        "rows": rows,
    }
    write_grid_outputs(json_out=json_out, csv_out=csv_out, payload=payload, rows=rows)
    best = min(
        (row for row in rows if row["internal_f_mae"] is not None),
        key=lambda row: float(row["internal_f_mae"]),
        default=None,
    )
    best_text = "none" if best is None else (
        f"leaf={best['leaf_token_count']} operator={best['operator_kind']} "
        f"mae={best['internal_f_mae']}"
    )
    print(
        f"status=success family=neural_operator task=markov_leaf_grid rows={len(rows)} "
        f"average_guess_mae={average_baseline['mae']} best={best_text} "
        f"json={json_out} csv={csv_out}"
    )
    return 0


def _make_split(
    cfg: NeuralOperatorMarkovLeafGridConfig,
    leaf_token_count: int,
    make_markov_changepoint_trees: Any,
) -> tuple[Any, Any]:
    train = make_markov_changepoint_trees(
        n_trees=cfg.n_train,
        n_states=cfg.n_states,
        doc_tokens=cfg.doc_tokens,
        leaf_token_count=leaf_token_count,
        transition_prob=cfg.transition_prob,
        vocabulary_size=cfg.vocabulary_size,
        seed=cfg.seed,
        split="train",
        generation_device=str(cfg.fixture_device or cfg.device),
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=cfg.n_eval,
        n_states=cfg.n_states,
        doc_tokens=cfg.doc_tokens,
        leaf_token_count=leaf_token_count,
        transition_prob=cfg.transition_prob,
        vocabulary_size=cfg.vocabulary_size,
        seed=cfg.seed + 1,
        split="test",
        generation_device=str(cfg.fixture_device or cfg.device),
    )
    return train, eval_trees


def _average_guess_baseline(train: Sequence[object], eval_trees: Sequence[object]) -> dict[str, float]:
    train_scores = [_score(tree) for tree in train]
    eval_scores = [_score(tree) for tree in eval_trees]
    mean = sum(train_scores) / len(train_scores)
    mae = sum(abs(mean - value) for value in eval_scores) / len(eval_scores)
    return {"target": float(mean), "mae": float(mae)}


def _score(tree: object) -> float:
    return float(getattr(tree, "metadata", {}).get("teacher_score_native"))


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _artifact_kind(result: Any) -> str | None:
    artifact = dict((result.artifacts or {}).get("f") or {})
    value = artifact.get("kind")
    return str(value) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
