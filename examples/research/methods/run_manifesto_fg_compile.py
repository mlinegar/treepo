#!/usr/bin/env python3
"""Example: DSPy/LLM manifesto f,g alternating compile.

Pattern: two sections in the TOML, each into its own dataclass:
  [family]   → DSPyFamilyConfig (upstream truth — no mirror)
  [lm]       → LmSection (treepo.methods scenario wrapper)
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--scorer", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/research/methods/manifesto_fg_compile.toml")
    ap.add_argument("--budget", default=None)
    ap.add_argument("--vllm-urls", default=None, help="comma-separated endpoint list")
    args = ap.parse_args()

    import treepo.methods
    from treepo.methods.canonical_defaults import build_lm_config_dict, load_dataclass, LmSection
    from treepo._research.ctreepo.dspy_family import DSPyFamilyConfig
    from treepo._research.ctreepo.distillation import load_labeled_trees

    family = load_dataclass(args.config, DSPyFamilyConfig, section="family",
                            overrides={"budget": args.budget})
    lm = load_dataclass(args.config, LmSection, section="lm", overrides={
        "endpoints": ([u.strip() for u in args.vllm_urls.split(",") if u.strip()]
                      if args.vllm_urls else None),
    })

    family.lm_config = build_lm_config_dict(lm, max_tokens=family.max_completion_tokens)
    family.problem_id = "manifesto_benoit"
    family.dimension = "economic"
    family.f_init_path = str(args.scorer)

    all_trees = load_labeled_trees(args.artifact)
    multi = [t for t in all_trees if t.num_chunks > 1]
    train_trees = [t for t in multi if (t.metadata or {}).get("split") == "train"]
    eval_trees = [t for t in multi
                  if family.max_input_chars is None
                  or len(t.document_text or "") < family.max_input_chars]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = treepo.methods.run("fit", {
        "family": "dspy",
        "train_data": train_trees, "eval_data": eval_trees,
        "backend_config": {"dspy_config": family,
                           "output_dir": str(args.output_dir / "fit"),
                           "first_train_side": "g"},
        "axis": {"max_iterations": 2, "axis_value": 0},
        "initial_artifacts": {"f": str(args.scorer), "g": "raw_concat"},
    })
    print(f"status={result.status}")
    print(f"pearson={result.metrics.get('external_expert_pearson')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
