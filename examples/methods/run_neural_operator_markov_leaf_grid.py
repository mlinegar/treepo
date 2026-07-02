#!/usr/bin/env python3
"""Grid learned neural operators over Markov leaf grouping sizes."""

from __future__ import annotations

from example_setup import (
    NeuralOperatorMarkovLeafGridConfig,
    config_dict,
    grid_values,
    load_example_config,
    markov_average_guess_baseline,
    markov_fit_config,
    markov_grid_row,
    markov_split,
)


def main() -> int:
    from treepo import fit
    from treepo.methods.grid import iter_grid, write_grid_outputs

    output_dir, cfg = load_example_config(
        default_config="neural_operator_markov_leaf_grid.toml",
        config_cls=NeuralOperatorMarkovLeafGridConfig,
    )
    leaf_counts = tuple(int(value) for value in grid_values(cfg.leaf_unit_counts, name="leaf_unit_counts"))
    operator_kinds = tuple(str(value) for value in grid_values(cfg.operator_kinds, name="operator_kinds"))

    reference_train, reference_eval = markov_split(cfg, leaf_unit_count=leaf_counts[0])
    average_baseline = markov_average_guess_baseline(reference_train, reference_eval)

    rows = []
    cells = iter_grid(
        {"leaf_unit_count": leaf_counts, "operator_kind": operator_kinds},
        output_root=output_dir,
    )
    for cell in cells:
        leaf_unit_count = int(cell.values["leaf_unit_count"])
        operator_kind = str(cell.values["operator_kind"])
        train, eval_trees = markov_split(cfg, leaf_unit_count=leaf_unit_count)
        result = fit(
            markov_fit_config(
                cfg,
                output_dir=cell.output_dir,
                train=train,
                eval_trees=eval_trees,
                operator_kind=operator_kind,
                leaf_unit_count=leaf_unit_count,
            )
        )
        rows.append(
            markov_grid_row(
                config=cfg,
                result=result,
                leaf_unit_count=leaf_unit_count,
                operator_kind=operator_kind,
                average_baseline=average_baseline,
            )
        )

    json_out = output_dir / "neural_operator_markov_leaf_grid.json"
    csv_out = output_dir / "neural_operator_markov_leaf_grid.csv"
    payload = {
        "config": config_dict(cfg),
        "average_guess_baseline": average_baseline,
        "rows": rows,
    }
    write_grid_outputs(json_out=json_out, csv_out=csv_out, payload=payload, rows=rows)

    from treepo.methods.tradeoff import TradeoffCurve

    for operator_kind in operator_kinds:
        curve = TradeoffCurve.from_rows(
            (row for row in rows if row["operator_kind"] == operator_kind),
            metric_keys=("internal_f_mae", "average_guess_mae"),
            metadata={"operator_kind": operator_kind, "task": "markov_changepoint"},
        )
        curve.write(
            json_out=output_dir / f"tradeoff_curve_{operator_kind}.json",
            csv_out=output_dir / f"tradeoff_curve_{operator_kind}.csv",
        )

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
        f"status=success family=neural_operator task=markov_leaf_grid rows={len(rows)} "
        f"average_guess_mae={average_baseline['mae']} best={best_text} "
        f"json={json_out} csv={csv_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
