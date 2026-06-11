#!/usr/bin/env python3
"""Example: LDA tree-recovery experiment (paper cell 5).

Pattern: TOML loads directly into upstream LDATreeRecoveryConfig; no
dispatcher (LDA recovery is a research script, not a registered method).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/research/methods/lda_recovery_smoke.toml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    from treepo.methods.canonical_defaults import load_dataclass
    from treepo._research.ctreepo.sim.core.lda_tree_recovery import (
        LDATreeRecoveryConfig, run_lda_tree_recovery_experiment,
    )

    cfg = load_dataclass(args.config, LDATreeRecoveryConfig,
                         overrides={"seed": args.seed})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_lda_tree_recovery_experiment(cfg)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict() if hasattr(summary, "to_dict") else summary,
                   indent=2, sort_keys=True, default=str)
    )
    print(f"exact_recovery: {getattr(summary, 'exact_recovery', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
