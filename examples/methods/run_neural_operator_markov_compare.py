#!/usr/bin/env python3
"""Compare official dense neuralop backends on a tiny Markov fixture."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class NeuralOperatorCompareConfig:
    operator_kinds: Sequence[str] = field(default_factory=lambda: ("fno", "tfno", "uno"))
    n_train: int = 8
    n_eval: int = 5
    n_states: int = 4
    doc_tokens: int = 32
    leaf_token_count: int = 8
    transition_prob: float = 0.15
    vocabulary_size: int = 64
    seed: int = 0
    max_iterations: int = 3
    embedding_dim: int = 8
    hidden_channels: int = 4
    n_modes: int = 2
    n_layers: int = 1
    conv_kernel_size: int = 3
    head_hidden_dim: int = 8
    epochs_per_iteration: int = 1
    batch_size: int = 4
    learning_rate: float = 0.01
    device: str = "cpu"
    fixture_device: str | None = None
    normalize_targets: bool = True
    numeric_transition_state_weight: float = 0.0
    numeric_transition_count_scale: float | None = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("neural_operator_markov_compare.toml"),
    )
    args = ap.parse_args()

    from treepo.methods import run
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo.methods.fixtures import make_markov_changepoint_trees

    cfg = load_dataclass(args.config, NeuralOperatorCompareConfig)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = make_markov_changepoint_trees(
        n_trees=cfg.n_train,
        n_states=cfg.n_states,
        doc_tokens=cfg.doc_tokens,
        leaf_token_count=cfg.leaf_token_count,
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
        leaf_token_count=cfg.leaf_token_count,
        transition_prob=cfg.transition_prob,
        vocabulary_size=cfg.vocabulary_size,
        seed=cfg.seed + 1,
        split="test",
        generation_device=str(cfg.fixture_device or cfg.device),
    )

    results = {}
    for operator_kind in cfg.operator_kinds:
        kind = str(operator_kind)
        backend = {
            "output_dir": str(args.output_dir / kind / "fit"),
            "operator_kind": kind,
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
        }
        result = run(
            "fit",
            {
                "family": "neural_operator",
                "train_data": train,
                "eval_data": eval_trees,
                "backend_config": backend,
                "axis": {"max_iterations": cfg.max_iterations, "axis_value": 0},
            },
        )
        results[kind] = result.to_dict()

    metrics = {
        kind: payload.get("metrics", {})
        for kind, payload in results.items()
    }
    out = args.output_dir / "neural_operator_markov_compare.json"
    out.write_text(
        json.dumps(
            {"config": asdict(cfg), "metrics": metrics, "results": results},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bits = ", ".join(
        f"{kind}:mae={metrics[kind].get('internal_f_mae')}"
        for kind in sorted(metrics)
    )
    print(f"status=success family=neural_operator compare={bits} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
