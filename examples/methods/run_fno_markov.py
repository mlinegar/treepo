#!/usr/bin/env python3
"""Source-tree example: train FNO through the neural-operator family."""

from __future__ import annotations

from example_setup import (
    FnoMarkovConfig,
    config_dict,
    load_example_config,
    markov_fit_config,
    markov_preferences,
    markov_split,
    validate_preference_mode,
    write_json,
)


def main() -> int:
    from treepo import fit

    output_dir, cfg = load_example_config(default_config="fno_markov.toml", config_cls=FnoMarkovConfig)
    validate_preference_mode(str(cfg.preference_mode))
    train, eval_trees = markov_split(cfg)
    preferences = markov_preferences(train, mode=str(cfg.preference_mode))
    fit_config = markov_fit_config(
        cfg,
        output_dir=output_dir,
        train=train,
        eval_trees=eval_trees,
        operator_kind="fno",
        leaf_unit_count=int(cfg.leaf_unit_count),
    )
    if preferences is not None:
        fit_config["preference_data"] = preferences
    result = fit(fit_config)

    out = output_dir / "fno_markov_result.json"
    write_json(
        out,
        {
            "config": config_dict(cfg),
            "preferences": {} if preferences is None else preferences.summary(),
            "statistic": dict(result.artifacts.get("statistic") or {}),
            "result": result.to_dict(),
        },
    )
    mae = result.metrics.get("internal_f_mae")
    print(
        f"status={result.status} family=neural_operator operator=fno "
        f"n={int(result.metrics.get('n', 0))} train={len(train)} "
        f"doc_unit={cfg.doc_unit_kind} leaf_unit_count={cfg.leaf_unit_count} "
        f"preferences={cfg.preference_mode} mae={mae} output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
