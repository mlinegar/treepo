"""Regression tests for the small shared method-grid helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import treepo.methods
from treepo.methods.grid import iter_grid, write_grid_outputs


def _run_oracle_grid(
    output_root: Path,
    *,
    oracle_names,
    seeds,
    n_trees_values,
    skip_existing: bool = False,
):
    """The shared grid loop used by source examples and tests."""
    rows = []
    cells_skipped = 0
    cells_run = 0
    cells = iter_grid(
        {"oracle_name": oracle_names, "seed": seeds, "n_trees": n_trees_values},
        output_root=output_root,
    )
    for cell in cells:
        oracle_name = str(cell.values["oracle_name"])
        seed = int(cell.values["seed"])
        n_trees = int(cell.values["n_trees"])
        cell_dir = cell.output_dir
        manifest = cell_dir / "treepo_methods_run_manifest.json"

        if skip_existing and manifest.exists():
            payload = json.loads(manifest.read_text())
            rows.append(
                {
                    "oracle": oracle_name,
                    "seed": seed,
                    "n_trees": n_trees,
                    "status": payload["status"],
                    "internal_f_mae": payload["metrics"].get("internal_f_mae"),
                    "manifest_path": str(manifest),
                    "resumed": True,
                }
            )
            cells_skipped += 1
            continue

        result = treepo.methods.run(
            "oracle",
            {
                "oracle_name": oracle_name,
                "seed": seed,
                "n_trees": n_trees,
                "output_dir": str(cell_dir),
            },
        )
        rows.append(
            {
                "oracle": oracle_name,
                "seed": seed,
                "n_trees": n_trees,
                "status": result.status,
                "internal_f_mae": result.metrics.get("internal_f_mae"),
                "manifest_path": result.manifest_path,
                "resumed": False,
            }
        )
        cells_run += 1
    return rows, cells_run, cells_skipped


def test_grid_run_oracle_across_axes(tmp_path: Path) -> None:
    """A 1 x 3 x 2 = 6-cell grid runs end-to-end and aggregates a CSV."""
    rows, ran, skipped = _run_oracle_grid(
        tmp_path,
        oracle_names=["hll_exact"],
        seeds=[0, 1, 2],
        n_trees_values=[4, 8],
    )
    assert len(rows) == 6
    assert ran == 6 and skipped == 0
    assert all(r["status"] == "success" for r in rows)
    # Both oracles compare against precomputed truth → MAE ≈ 0 per cell.
    for r in rows:
        assert r["internal_f_mae"] == pytest.approx(0.0, abs=1e-9), (
            f"cell {r['oracle']}/{r['seed']}/{r['n_trees']} MAE = {r['internal_f_mae']}"
        )
        # Per-cell manifest is real and parseable.
        manifest = json.loads(Path(r["manifest_path"]).read_text())
        assert manifest["status"] == "success"
        assert manifest["metrics"]["n"] == float(r["n_trees"])

    csv_path = tmp_path / "grid_summary.csv"
    write_grid_outputs(json_out=tmp_path / "grid_summary.json", csv_out=csv_path, payload={"rows": rows}, rows=rows)
    assert csv_path.exists()
    # Re-read and confirm round-trip.
    with csv_path.open() as f:
        round_tripped = list(csv.DictReader(f))
    assert len(round_tripped) == 6


def test_grid_run_resume_skips_existing_cells(tmp_path: Path) -> None:
    """Second pass over the same axes skips every cell whose manifest
    already exists. This is the same resume-if-output-exists pattern the
    existing paper scripts use (``if summary.exists(): skip``).
    """
    axes = dict(
        oracle_names=["hll_exact"],
        seeds=[0, 1],
        n_trees_values=[4],
    )
    rows1, ran1, skipped1 = _run_oracle_grid(tmp_path, skip_existing=True, **axes)
    assert ran1 == 2 and skipped1 == 0
    assert all(not r["resumed"] for r in rows1)

    rows2, ran2, skipped2 = _run_oracle_grid(tmp_path, skip_existing=True, **axes)
    assert ran2 == 0 and skipped2 == 2
    assert all(r["resumed"] for r in rows2)
    # Resumed rows must match the first-pass numbers exactly.
    for r1, r2 in zip(rows1, rows2):
        assert r1["status"] == r2["status"]
        assert r1["internal_f_mae"] == r2["internal_f_mae"]


def test_grid_run_fit_method_for_oracle_family(tmp_path: Path) -> None:
    """The ``"fit"`` method handles the oracle family through the same
    grid-loop shape as direct ``run("oracle", ...)`` calls.
    """
    from treepo.methods.fixtures import make_hll_token_trees

    trees = make_hll_token_trees(n_trees=4, seed=21)
    cells = [
        {
            "family": "oracle",
            "eval_data": trees,
            "backend_config": {
                "oracle_name": "hll_exact",
                "output_dir": str(tmp_path / "oracle"),
            },
        },
    ]
    rows = []
    for cell in cells:
        result = treepo.methods.run("fit", cell)
        rows.append(
            {
                "family": cell["family"],
                "status": result.status,
                "internal_f_mae": result.metrics["internal_f_mae"],
            }
        )
    assert len(rows) == 1
    assert all(r["status"] == "success" for r in rows)
    oracle_row = next(r for r in rows if r["family"] == "oracle")
    assert oracle_row["internal_f_mae"] == pytest.approx(0.0, abs=1e-9)
