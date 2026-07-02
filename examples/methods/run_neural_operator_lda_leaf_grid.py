#!/usr/bin/env python3
"""Grid learned neural operators over LDA leaf grouping sizes."""

from __future__ import annotations

from example_setup import (
    NeuralOperatorLDALeafGridConfig,
    config_dict,
    fit_lda_sklearn_baseline,
    grid_values,
    lda_average_guess_baseline,
    lda_fit_config,
    lda_grid_row,
    lda_split,
    lda_target_key,
    load_example_config,
)


def main() -> int:
    from treepo import fit
    from treepo.methods.grid import iter_grid, write_grid_outputs

    output_dir, cfg = load_example_config(
        default_config="neural_operator_lda_leaf_grid.toml",
        config_cls=NeuralOperatorLDALeafGridConfig,
    )
    leaf_counts = tuple(int(value) for value in grid_values(cfg.leaf_unit_counts, name="leaf_unit_counts"))
    operator_kinds = tuple(str(value) for value in grid_values(cfg.operator_kinds, name="operator_kinds"))

    target_key = lda_target_key(cfg)
    reference_leaf = int(leaf_counts[0])
    baseline_train, baseline_eval = lda_split(cfg, leaf_unit_count=reference_leaf)
    sklearn_baseline = fit_lda_sklearn_baseline(cfg, baseline_train, baseline_eval)
    average_baseline = lda_average_guess_baseline(baseline_train, baseline_eval, int(cfg.n_topics), int(cfg.target_topic))

    rows = []
    cells = iter_grid(
        {"leaf_unit_count": leaf_counts, "operator_kind": operator_kinds},
        output_root=output_dir,
    )
    for cell in cells:
        leaf_unit_count = int(cell.values["leaf_unit_count"])
        operator_kind = str(cell.values["operator_kind"])
        train, eval_trees = lda_split(cfg, leaf_unit_count=leaf_unit_count)
        result = fit(
            lda_fit_config(
                cfg,
                output_dir=cell.output_dir,
                train=train,
                eval_trees=eval_trees,
                operator_kind=operator_kind,
                leaf_unit_count=leaf_unit_count,
            )
        )
        rows.append(
            lda_grid_row(
                config=cfg,
                result=result,
                leaf_unit_count=leaf_unit_count,
                operator_kind=operator_kind,
                sklearn_baseline=sklearn_baseline,
                average_baseline=average_baseline,
            )
        )

    json_out = output_dir / "neural_operator_lda_leaf_grid.json"
    csv_out = output_dir / "neural_operator_lda_leaf_grid.csv"
    payload = {
        "config": config_dict(cfg),
        "target_key": target_key,
        "sklearn_baseline": sklearn_baseline.to_dict() if sklearn_baseline is not None else None,
        "average_guess_baseline": average_baseline,
        "rows": rows,
    }
    write_grid_outputs(json_out=json_out, csv_out=csv_out, payload=payload, rows=rows)
    best = min(
        (row for row in rows if row["internal_f_mae"] is not None),
        key=lambda row: float(row["internal_f_mae"]),
        default=None,
    )
    best_text = "none" if best is None else (
        f"leaf_unit_count={best['leaf_unit_count']} operator={best['operator_kind']} "
        f"mae={best['internal_f_mae']}"
    )
    print(
        f"status=success family=neural_operator task=lda_leaf_grid rows={len(rows)} "
        f"sklearn_mae={None if sklearn_baseline is None else sklearn_baseline.target_mae} "
        f"average_guess_mae={average_baseline['target_mae']} best={best_text} "
        f"json={json_out} csv={csv_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
