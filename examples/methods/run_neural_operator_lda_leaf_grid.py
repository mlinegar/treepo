#!/usr/bin/env python3
"""Grid learned neural operators over LDA leaf sizes."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class NeuralOperatorLDALeafGridConfig:
    operator_kinds: Sequence[str] = field(default_factory=lambda: ("fno", "tfno", "uno", "conv1d"))
    leaf_token_counts: Sequence[int] = field(default_factory=lambda: (32, 64, 128))
    n_train: int = 256
    n_eval: int = 64
    n_topics: int = 8
    doc_tokens: int = 256
    vocabulary_size: int = 160
    doc_topic_concentration: float = 0.7
    topic_word_concentration: float = 0.05
    target_topic: int = 0
    topic_seed: int = 0
    seed: int = 0
    max_iterations: int = 3
    embedding_dim: int = 16
    hidden_channels: int = 12
    n_modes: int = 4
    n_layers: int = 1
    conv_kernel_size: int = 3
    head_hidden_dim: int = 24
    epochs_per_iteration: int = 3
    batch_size: int = 16
    learning_rate: float = 0.01
    device: str = "cpu"
    fixture_device: str | None = None
    sklearn_max_iter: int = 50
    run_sklearn_baseline: bool = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("neural_operator_lda_leaf_grid.toml"),
    )
    args = ap.parse_args()

    from treepo.methods import run
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo.methods.fixtures import make_lda_topic_trees
    from treepo.methods.lda import fit_sklearn_lda_baseline
    from treepo.methods.grid import iter_grid, write_grid_outputs

    cfg = load_dataclass(args.config, NeuralOperatorLDALeafGridConfig)
    leaf_sizes = tuple(int(value) for value in cfg.leaf_token_counts)
    operator_kinds = tuple(str(value) for value in cfg.operator_kinds)
    if not leaf_sizes:
        raise ValueError("leaf_token_counts must be non-empty")
    if not operator_kinds:
        raise ValueError("operator_kinds must be non-empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_key = f"topic_{int(cfg.target_topic)}_proportion"
    reference_leaf = int(leaf_sizes[0])
    baseline_train, baseline_eval = _make_split(cfg, reference_leaf, make_lda_topic_trees)
    sklearn_baseline = (
        fit_sklearn_lda_baseline(
            baseline_train,
            baseline_eval,
            n_topics=cfg.n_topics,
            vocabulary_size=cfg.vocabulary_size,
            doc_topic_prior=cfg.doc_topic_concentration,
            topic_word_prior=cfg.topic_word_concentration,
            max_iter=cfg.sklearn_max_iter,
            random_state=cfg.seed,
            target_topic=cfg.target_topic,
        )
        if bool(cfg.run_sklearn_baseline) and int(cfg.sklearn_max_iter) > 0
        else None
    )
    average_baseline = _average_guess_baseline(baseline_train, baseline_eval, int(cfg.n_topics), int(cfg.target_topic))

    rows: list[dict[str, Any]] = []
    cells = iter_grid(
        {"leaf_token_count": leaf_sizes, "operator_kind": operator_kinds},
        output_root=args.output_dir,
    )
    for cell in cells:
        leaf_token_count = int(cell.values["leaf_token_count"])
        operator_kind = str(cell.values["operator_kind"])
        train, eval_trees = _make_split(cfg, leaf_token_count, make_lda_topic_trees)
        result = run(
            "fit",
            {
                "family": "neural_operator",
                "train_data": train,
                "eval_data": eval_trees,
                "backend_config": {
                    "output_dir": str(cell.output_dir / "fit"),
                    "operator_kind": operator_kind,
                    "target_key": target_key,
                    "target_vector_key": "topic_proportions",
                    "target_dim": cfg.n_topics,
                    "target_min": 0.0,
                    "target_max": 1.0,
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
        rows.append(
            {
                "leaf_token_count": int(leaf_token_count),
                "leaves_per_doc": int(math.ceil(int(cfg.doc_tokens) / int(leaf_token_count))),
                "operator_kind": operator_kind,
                "status": result.status,
                "internal_f_mae": _metric(metrics, "internal_f_mae"),
                "internal_f_pearson": _metric(metrics, "internal_f_pearson"),
                "target_topic_vector_mae": _metric(metrics, f"topic_{int(cfg.target_topic)}_internal_f_mae"),
                "mean_topic_vector_mae": _mean_topic_mae(metrics, int(cfg.n_topics)),
                "mean_prediction": _metric(metrics, "mean_prediction"),
                "mean_teacher": _metric(metrics, "mean_teacher"),
                "n": _metric(metrics, "n"),
                "sklearn_target_mae": None if sklearn_baseline is None else float(sklearn_baseline.target_mae),
                "sklearn_mean_mae": None if sklearn_baseline is None else float(sklearn_baseline.mean_mae),
                "average_guess_target": float(average_baseline["target"]),
                "average_guess_target_mae": float(average_baseline["target_mae"]),
                "average_guess_mean_mae": float(average_baseline["mean_mae"]),
                "artifact_kind": _artifact_kind(result),
                "manifest_path": result.manifest_path,
            }
        )

    json_out = args.output_dir / "neural_operator_lda_leaf_grid.json"
    csv_out = args.output_dir / "neural_operator_lda_leaf_grid.csv"
    payload = {
        "config": asdict(cfg),
        "target_key": target_key,
        "sklearn_baseline": sklearn_baseline.to_dict() if sklearn_baseline is not None else None,
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
        f"status=success family=neural_operator task=lda_leaf_grid rows={len(rows)} "
        f"sklearn_mae={None if sklearn_baseline is None else sklearn_baseline.target_mae} "
        f"average_guess_mae={average_baseline['target_mae']} best={best_text} "
        f"json={json_out} csv={csv_out}"
    )
    return 0


def _average_guess_baseline(train: Sequence[object], eval_trees: Sequence[object], n_topics: int, target_topic: int) -> dict[str, float]:
    train_vectors = [_topic_vector(tree, n_topics) for tree in train]
    eval_vectors = [_topic_vector(tree, n_topics) for tree in eval_trees]
    means = [sum(row[idx] for row in train_vectors) / len(train_vectors) for idx in range(n_topics)]
    errors = [[abs(means[idx] - row[idx]) for idx in range(n_topics)] for row in eval_vectors]
    return {
        "target": float(means[target_topic]),
        "target_mae": float(sum(row[target_topic] for row in errors) / len(errors)),
        "mean_mae": float(sum(sum(row) for row in errors) / (len(errors) * n_topics)),
    }


def _topic_vector(tree: object, n_topics: int) -> list[float]:
    values = getattr(tree, "topic_proportions", None)
    if values is None:
        values = getattr(tree, "metadata", {}).get("topic_proportions")
    out = [float(value) for value in values]
    if len(out) != int(n_topics):
        raise ValueError(f"expected {n_topics} topic proportions, got {len(out)}")
    return out


def _make_split(cfg: NeuralOperatorLDALeafGridConfig, leaf_token_count: int, make_lda_topic_trees: Any) -> tuple[Any, Any]:
    train = make_lda_topic_trees(
        n_trees=cfg.n_train,
        n_topics=cfg.n_topics,
        doc_tokens=cfg.doc_tokens,
        leaf_token_count=leaf_token_count,
        vocabulary_size=cfg.vocabulary_size,
        doc_topic_concentration=cfg.doc_topic_concentration,
        topic_word_concentration=cfg.topic_word_concentration,
        target_topic=cfg.target_topic,
        topic_seed=cfg.topic_seed,
        seed=cfg.seed,
        split="train",
        generation_device=str(cfg.fixture_device or cfg.device),
    )
    eval_trees = make_lda_topic_trees(
        n_trees=cfg.n_eval,
        n_topics=cfg.n_topics,
        doc_tokens=cfg.doc_tokens,
        leaf_token_count=leaf_token_count,
        vocabulary_size=cfg.vocabulary_size,
        doc_topic_concentration=cfg.doc_topic_concentration,
        topic_word_concentration=cfg.topic_word_concentration,
        target_topic=cfg.target_topic,
        topic_seed=cfg.topic_seed,
        seed=cfg.seed + 1,
        split="test",
        generation_device=str(cfg.fixture_device or cfg.device),
    )
    return train, eval_trees


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


def _mean_topic_mae(metrics: dict[str, Any], n_topics: int) -> float | None:
    values = [
        _metric(metrics, f"topic_{idx}_internal_f_mae")
        for idx in range(max(1, int(n_topics)))
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


if __name__ == "__main__":
    raise SystemExit(main())
