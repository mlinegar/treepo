#!/usr/bin/env python3
"""Example: LDA leaf-local-mixture oracle via `treepo.cld.run("oracle", ...)`.

The LDA oracle has no v1 auto-fixture; this example builds eval trees
from a small synthetic LDA world via LDATreeRecoveryConfig.
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "treepo" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "treepo.cld/configs/lda_oracle.toml")
    args = ap.parse_args()

    import treepo.cld
    from treepo.cld.canonical_defaults import load_dataclass, LdaOracleConfig
    from treepo._research.ctreepo.sim.core.lda_tree_recovery import (
        LDATreeRecoveryConfig, build_lda_tree_recovery_world,
    )

    cfg = load_dataclass(args.config, LdaOracleConfig)
    world = build_lda_tree_recovery_world(LDATreeRecoveryConfig(
        n_topics=4, vocab_size=64, leaf_tokens=16,
        train_docs=0, test_docs=cfg.n_trees,
        min_tokens=384, max_tokens=384, seed=cfg.seed,
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = treepo.cld.run("oracle", {
        "oracle_name": cfg.oracle_name,
        "eval_data": list(world.test_trees),
        "n_trees": cfg.n_trees, "seed": cfg.seed, "split": cfg.split,
        "output_dir": str(args.output_dir),
    })
    print(f"status={result.status}")
    print(f"metrics={dict(result.metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
