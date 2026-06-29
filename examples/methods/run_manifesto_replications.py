#!/usr/bin/env python3
"""Central Manifesto/RILE replication example with qsentence-guided g.

This is intentionally provider-neutral. By default it uses a tiny oracle
predictor so the example is runnable in package tests. Downstream manifesto
replications can swap in a real DSPy program through the same estimator/family
surface.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ManifestoReplicationConfig:
    estimator: str = "dspy"
    model: str = "local-dspy-program"
    max_iterations: int = 2
    use_oracle_predictor: bool = True
    prompt_template: str = ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("manifesto_replications.toml"),
    )
    ap.add_argument("--estimator", choices=("dspy", "prompted_llm", "llm"), default=None)
    args = ap.parse_args()

    from treepo.methods import run
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo.tasks.manifesto import (
        make_manifesto_replication_trees,
        manifesto_oracle_predict_fn,
        manifesto_prompt_template,
        replication_payload,
    )

    cfg = load_dataclass(args.config, ManifestoReplicationConfig, overrides={"estimator": args.estimator})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = make_manifesto_replication_trees(split="train")
    eval_trees = make_manifesto_replication_trees(split="test")
    prompt_template = cfg.prompt_template or manifesto_prompt_template()
    backend = {
        "output_dir": str(args.output_dir / "fit"),
        "model": cfg.model,
        "prompt_template": prompt_template,
        "min_score": -100.0,
        "max_score": 100.0,
    }
    if cfg.use_oracle_predictor:
        backend["predict_fn"] = manifesto_oracle_predict_fn

    result = run(
        "fit",
        {
            "estimator": {"name": cfg.estimator, "target": "g", "model": cfg.model},
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": backend,
            "axis": {"max_iterations": cfg.max_iterations, "axis_value": 0},
        },
    )
    out = args.output_dir / "manifesto_replications_result.json"
    out.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "replications": replication_payload(eval_trees),
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"status={result.status} estimator={cfg.estimator} family={result.summary.get('family')} "
        f"docs={len(eval_trees)} mae={result.metrics.get('internal_f_mae')} output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
