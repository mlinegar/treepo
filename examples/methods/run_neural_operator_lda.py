#!/usr/bin/env python3
"""Train neural operators on synthetic Dirichlet LDA topic proportions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class NeuralOperatorLDAConfig:
    operator_kinds: Sequence[str] = field(default_factory=lambda: ("fno", "tfno", "uno"))
    n_train: int = 256
    n_eval: int = 64
    n_topics: int = 8
    doc_tokens: int = 256
    leaf_token_count: int = 32
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
        default=Path(__file__).resolve().with_name("neural_operator_lda.toml"),
    )
    args = ap.parse_args()

    from treepo.methods import run
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo.methods.fixtures import make_lda_topic_trees
    from treepo.methods.lda import fit_sklearn_lda_baseline

    cfg = load_dataclass(args.config, NeuralOperatorLDAConfig)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = make_lda_topic_trees(
        n_trees=cfg.n_train,
        n_topics=cfg.n_topics,
        doc_tokens=cfg.doc_tokens,
        leaf_token_count=cfg.leaf_token_count,
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
        leaf_token_count=cfg.leaf_token_count,
        vocabulary_size=cfg.vocabulary_size,
        doc_topic_concentration=cfg.doc_topic_concentration,
        topic_word_concentration=cfg.topic_word_concentration,
        target_topic=cfg.target_topic,
        topic_seed=cfg.topic_seed,
        seed=cfg.seed + 1,
        split="test",
        generation_device=str(cfg.fixture_device or cfg.device),
    )

    target_key = f"topic_{int(cfg.target_topic)}_proportion"
    sklearn_baseline = (
        fit_sklearn_lda_baseline(
            train,
            eval_trees,
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
    average_baseline = _average_guess_baseline(train, eval_trees, int(cfg.n_topics), int(cfg.target_topic))

    results = {}
    for operator_kind in cfg.operator_kinds:
        kind = str(operator_kind)
        backend = {
            "output_dir": str(args.output_dir / kind / "fit"),
            "operator_kind": kind,
            "target_key": target_key,
            "target_vector_key": "topic_proportions",
            "target_dim": cfg.n_topics,
            "target_min": 0.0,
            "target_max": 1.0,
            "embedding_dim": cfg.embedding_dim,
            "hidden_channels": cfg.hidden_channels,
            "n_modes": cfg.n_modes,
            "n_layers": cfg.n_layers,
            "head_hidden_dim": cfg.head_hidden_dim,
            "epochs_per_iteration": cfg.epochs_per_iteration,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "device": cfg.device,
        }
        result = run(
            "fit",
            {
                "family": "neural_operator",
                "train_data": train,
                "eval_data": eval_trees,
                "backend_config": backend,
                "axis": {"max_iterations": cfg.max_iterations, "axis_value": cfg.target_topic},
            },
        )
        results[kind] = result.to_dict()

    metrics = {kind: payload.get("metrics", {}) for kind, payload in results.items()}
    out = args.output_dir / "neural_operator_lda_result.json"
    out.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "target_key": target_key,
                "metrics": metrics,
                "sklearn_baseline": sklearn_baseline.to_dict() if sklearn_baseline is not None else None,
                "average_guess_baseline": average_baseline,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bits = ", ".join(
        f"{kind}:mae={metrics[kind].get('internal_f_mae')}"
        for kind in sorted(metrics)
    )
    print(
        f"status=success family=neural_operator task=lda target={target_key} "
        f"sklearn_mae={None if sklearn_baseline is None else sklearn_baseline.target_mae} "
        f"average_guess_mae={average_baseline['target_mae']} "
        f"compare={bits} output={out}"
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


if __name__ == "__main__":
    raise SystemExit(main())
