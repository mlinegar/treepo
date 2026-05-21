#!/usr/bin/env python3
"""Example: Markov change-point oracle via `treepo.cld.run("oracle", ...)`.

The Markov auto-fixture is now registered in `_ORACLE_DOMAIN_FIXTURES`,
so all the example does is load `MarkovChangepointConfig` from TOML,
flatten its fields into the run dict, and dispatch. The oracle builds
eval_data internally.
"""

from __future__ import annotations

import argparse, dataclasses, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "treepo" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "treepo.cld/configs/markov_oracle.toml")
    ap.add_argument("--n-regimes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import treepo.cld
    from treepo.cld.canonical_defaults import load_dataclass
    from treepo._research.tree.markov_changepoint_honesty_simulation import MarkovChangepointConfig

    dgp = load_dataclass(args.config, MarkovChangepointConfig, section="dgp",
                         overrides={"n_regimes": args.n_regimes, "seed": args.seed})

    # Forward DGP fields straight to the oracle dispatcher; auto-fixture handles the rest.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_knobs = {f.name: getattr(dgp, f.name) for f in dataclasses.fields(dgp)
                     if f.name in {"n_regimes", "vocab_size", "min_tokens", "max_tokens",
                                   "min_segments", "max_segments", "min_seg_len", "max_seg_len",
                                   "train_docs", "test_docs", "sinkhorn_iters",
                                   "transition_log_std", "seed"}}
    result = treepo.cld.run("oracle", {
        "oracle_name": "markov_changepoint_count",
        "output_dir": str(args.output_dir),
        **fixture_knobs,
    })
    print(f"status={result.status}")
    print(f"metrics={dict(result.metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
