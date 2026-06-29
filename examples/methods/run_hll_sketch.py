#!/usr/bin/env python3
"""Run a DataSketches HLL sketch through the unified fit API."""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HllSketchConfig:
    backend: str = "datasketches"
    precision: int = 14
    hash_bits: int = 64
    schedule: str = "balanced"
    n_trees: int = 6
    leaves_per_tree: int = 4
    leaf_token_count: int = 24
    vocabulary_size: int = 200
    seed: int = 0
    max_iterations: int = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().with_name("hll_sketch.toml"))
    ap.add_argument("--precision", type=int, default=None)
    ap.add_argument("--schedule", default=None)
    args = ap.parse_args()

    from treepo.methods import run
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo.methods.fixtures import make_hll_token_trees

    cfg = load_dataclass(args.config, HllSketchConfig, overrides={
        "precision": args.precision, "schedule": args.schedule,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    trees = make_hll_token_trees(
        n_trees=int(payload["n_trees"]),
        leaves_per_tree=int(payload["leaves_per_tree"]),
        leaf_token_count=int(payload["leaf_token_count"]),
        vocabulary_size=int(payload["vocabulary_size"]),
        seed=int(payload["seed"]),
    )
    result = run(
        "fit",
        {
            "family": "classical_sketch",
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {
                "output_dir": str(args.output_dir / "fit"),
                "sketch": "hll",
                "backend": str(payload["backend"]),
                "precision": int(payload["precision"]),
                "hash_bits": int(payload["hash_bits"]),
                "schedule": str(payload["schedule"]),
            },
            "axis": {"max_iterations": int(payload["max_iterations"]), "axis_value": 0},
        },
    )
    out = args.output_dir / "hll_sketch_result.json"
    out.write_text(
        json.dumps({"config": payload, "result": result.to_dict()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"status={result.status} family=classical_sketch sketch=hll "
        f"n={int(result.metrics.get('n', 0))} mae={result.metrics.get('internal_f_mae')} "
        f"output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
