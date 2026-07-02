#!/usr/bin/env python3
"""Compare official dense neuralop backends on a tiny Markov fixture."""

from __future__ import annotations

from example_setup import (
    NeuralOperatorCompareConfig,
    config_dict,
    load_example_config,
    markov_fit_config,
    markov_split,
    result_metrics,
    result_statistics,
    write_json,
)


def main() -> int:
    from treepo import fit

    output_dir, cfg = load_example_config(
        default_config="neural_operator_markov_compare.toml",
        config_cls=NeuralOperatorCompareConfig,
    )
    train, eval_trees = markov_split(cfg)

    results = {}
    for operator_kind in cfg.operator_kinds:
        kind = str(operator_kind)
        fit_config = markov_fit_config(
            cfg,
            output_dir=output_dir / kind,
            train=train,
            eval_trees=eval_trees,
            operator_kind=kind,
            leaf_unit_count=int(cfg.leaf_unit_count),
        )
        results[kind] = fit(fit_config).to_dict()

    metrics = result_metrics(results)
    out = output_dir / "neural_operator_markov_compare.json"
    write_json(
        out,
        {
            "config": config_dict(cfg),
            "metrics": metrics,
            "statistics": result_statistics(results),
            "results": results,
        },
    )
    bits = ", ".join(
        f"{kind}:mae={metrics[kind].get('internal_f_mae')}"
        for kind in sorted(metrics)
    )
    print(f"status=success family=neural_operator compare={bits} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
