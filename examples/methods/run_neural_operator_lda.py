#!/usr/bin/env python3
"""Train neural operators on synthetic Dirichlet LDA topic proportions."""

from __future__ import annotations

from example_setup import (
    NeuralOperatorLDAConfig,
    config_dict,
    fit_lda_sklearn_baseline,
    lda_average_guess_baseline,
    lda_fit_config,
    lda_split,
    lda_target_key,
    load_example_config,
    result_metrics,
    result_statistics,
    write_json,
)


def main() -> int:
    from treepo import fit

    output_dir, cfg = load_example_config(
        default_config="neural_operator_lda.toml",
        config_cls=NeuralOperatorLDAConfig,
    )
    train, eval_trees = lda_split(cfg)

    target_key = lda_target_key(cfg)
    sklearn_baseline = fit_lda_sklearn_baseline(cfg, train, eval_trees)
    average_baseline = lda_average_guess_baseline(train, eval_trees, int(cfg.n_topics), int(cfg.target_topic))

    results = {}
    for operator_kind in cfg.operator_kinds:
        kind = str(operator_kind)
        fit_config = lda_fit_config(
            cfg,
            output_dir=output_dir / kind,
            train=train,
            eval_trees=eval_trees,
            operator_kind=kind,
            leaf_unit_count=int(cfg.leaf_unit_count),
        )
        results[kind] = fit(fit_config).to_dict()

    metrics = result_metrics(results)
    out = output_dir / "neural_operator_lda_result.json"
    write_json(
        out,
        {
            "config": config_dict(cfg),
            "target_key": target_key,
            "metrics": metrics,
            "statistics": result_statistics(results),
            "sklearn_baseline": sklearn_baseline.to_dict() if sklearn_baseline is not None else None,
            "average_guess_baseline": average_baseline,
            "results": results,
        },
    )
    bits = ", ".join(
        f"{kind}:mae={metrics[kind].get('internal_f_mae')}"
        for kind in sorted(metrics)
    )
    print(
        f"status=success family=neural_operator task=lda target={target_key} "
        f"sklearn_mae={None if sklearn_baseline is None else sklearn_baseline.target_mae} "
        f"average_guess_mae={average_baseline['target_mae']} "
        f"compare={bits} output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
