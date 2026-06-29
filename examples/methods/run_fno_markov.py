#!/usr/bin/env python3
"""Source-tree example: train FNO through the neural-operator family."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FnoMarkovConfig:
    n_train: int = 8
    n_eval: int = 4
    doc_tokens: int = 32
    leaf_token_count: int = 8
    vocabulary_size: int = 64
    seed: int = 0
    max_iterations: int = 3
    embedding_dim: int = 8
    hidden_channels: int = 4
    n_modes: int = 2
    n_layers: int = 1
    head_hidden_dim: int = 8
    epochs_per_iteration: int = 1
    batch_size: int = 4
    learning_rate: float = 0.01
    device: str = "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("fno_markov.toml"),
    )
    args = ap.parse_args()

    from treepo.methods import run
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo.methods.fixtures import make_markov_changepoint_trees

    cfg = load_dataclass(args.config, FnoMarkovConfig)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = make_markov_changepoint_trees(
        n_trees=cfg.n_train,
        doc_tokens=cfg.doc_tokens,
        leaf_token_count=cfg.leaf_token_count,
        vocabulary_size=cfg.vocabulary_size,
        seed=cfg.seed,
        split="train",
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=cfg.n_eval,
        doc_tokens=cfg.doc_tokens,
        leaf_token_count=cfg.leaf_token_count,
        vocabulary_size=cfg.vocabulary_size,
        seed=cfg.seed + 1,
        split="test",
    )
    backend = {
        "output_dir": str(args.output_dir / "fit"),
        "operator_kind": "fno",
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
            "axis": {"max_iterations": cfg.max_iterations, "axis_value": 0},
        },
    )
    out = args.output_dir / "fno_markov_result.json"
    out.write_text(
        json.dumps({"config": asdict(cfg), "result": result.to_dict()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    mae = result.metrics.get("internal_f_mae")
    print(
        f"status={result.status} family=neural_operator operator=fno "
        f"n={int(result.metrics.get('n', 0))} mae={mae} output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
