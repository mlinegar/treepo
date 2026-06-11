#!/usr/bin/env python3
"""Research example: run an HLL classical sketch directly.

This is intentionally not part of the public `treepo.methods` dispatcher.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/research/methods/hll_sketch.toml")
    ap.add_argument("--precision", type=int, default=None)
    ap.add_argument("--schedule", default=None)
    args = ap.parse_args()

    from treepo._research.methods.hll_config import HllSketchConfig
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo.methods.fixtures import make_hll_token_trees
    from treepo.bench.sketches.adapters import make_hll_adapter
    from treepo.bench.sketches.tree_reducer import treepo_reduce

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
    adapter = make_hll_adapter(
        backend=str(payload["backend"]),
        precision=int(payload["precision"]),
        hash_bits=int(payload["hash_bits"]),
    )
    rows = []
    for idx, tree in enumerate(trees):
        root_state = treepo_reduce(
            [list(leaf.tokens) for leaf in tree.leaves],
            adapter,
            schedule=str(payload["schedule"]),
        )
        estimate = float(adapter.query(root_state, None))
        rows.append({
            "tree_id": idx,
            "prediction": estimate,
            "teacher": float(tree.metadata["teacher_score_1_7"]),
        })
    out = args.output_dir / "hll_sketch_predictions.jsonl"
    out.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"status=success rows={len(rows)} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
