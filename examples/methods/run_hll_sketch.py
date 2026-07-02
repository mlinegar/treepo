#!/usr/bin/env python3
"""Run a DataSketches HLL sketch through the unified fit API."""

from __future__ import annotations

from example_setup import (
    HllSketchConfig,
    config_dict,
    hll_fit_config,
    hll_trees,
    load_example_config,
    write_json,
)


def main() -> int:
    from treepo import fit

    output_dir, cfg = load_example_config(
        default_config="hll_sketch.toml",
        config_cls=HllSketchConfig,
        cli_overrides=(("precision", int), ("schedule", None)),
    )
    trees = hll_trees(cfg)
    result = fit(hll_fit_config(cfg, output_dir=output_dir, trees=trees))

    out = output_dir / "hll_sketch_result.json"
    write_json(
        out,
        {
            "config": config_dict(cfg),
            "statistic": dict(result.artifacts.get("statistic") or {}),
            "result": result.to_dict(),
        },
    )
    print(
        f"status={result.status} family=classical_sketch sketch=hll "
        f"n={int(result.metrics.get('n', 0))} mae={result.metrics.get('internal_f_mae')} "
        f"output={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
