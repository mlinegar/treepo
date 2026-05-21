#!/usr/bin/env python3
"""Example: FNO family fit via `treepo.cld.run("fit", {family="fno", ...})`.

Pattern: load TOML directly into the upstream FNOFamilyConfig — no mirror.
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
                    default=REPO_ROOT / "treepo.cld/configs/fno_smoke.toml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    import treepo.cld
    from treepo.cld.canonical_defaults import load_dataclass
    from treepo.cld.fixtures import make_hll_token_trees
    from treepo._research.ctreepo.fno_family import FNOFamilyConfig

    cfg = load_dataclass(args.config, FNOFamilyConfig,
                         overrides={"epochs_per_iteration": args.epochs})
    trees = list(make_hll_token_trees(
        n_trees=4, leaves_per_tree=4, leaf_token_count=8,
        vocabulary_size=64, seed=0,
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = treepo.cld.run("fit", {
        "family": "fno",
        "train_data": trees, "eval_data": trees,
        "backend_config": {"fno_config": cfg, "output_dir": str(args.output_dir / "fit")},
        "axis": {"max_iterations": 1, "axis_value": 0},
        "initial_artifacts": {"f": "identity", "g": "identity"},
    })
    print(f"status={result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
