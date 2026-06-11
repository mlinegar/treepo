#!/usr/bin/env python3
"""Example: LDA leaf-local-mixture oracle via `treepo.methods.run("oracle", ...)`.

Eval trees come from `treepo._research.methods.lda_fixtures`, the same
fixture the methods test suite uses for this oracle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/research/methods/lda_oracle.toml")
    args = ap.parse_args()

    from dataclasses import dataclass

    import treepo.methods
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo._research.methods.lda_fixtures import make_leaf_local_mixture_trees

    @dataclass
    class LdaOracleResearchConfig:
        oracle_name: str = "leaf_local_mixture_target"
        n_trees: int = 8
        seed: int = 0
        split: str = "test"

    cfg = load_dataclass(args.config, LdaOracleResearchConfig)
    trees, _world_cfg = make_leaf_local_mixture_trees(seed=cfg.seed, split=cfg.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = treepo.methods.run("oracle", {
        "oracle_name": cfg.oracle_name,
        "eval_data": trees[: cfg.n_trees],
        "n_trees": cfg.n_trees, "seed": cfg.seed, "split": cfg.split,
        "output_dir": str(args.output_dir),
    })
    print(f"status={result.status}")
    print(f"metrics={dict(result.metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
