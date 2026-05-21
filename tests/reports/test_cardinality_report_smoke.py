from __future__ import annotations

import json
from pathlib import Path

from treepo.bench.reports import cardinality


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cardinality_report_smoke(tmp_path: Path) -> None:
    card_json = (
        tmp_path
        / "cardinality"
        / "paper"
        / "audit_all"
        / "seed_0"
        / "summary.json"
    )
    _write_json(
        card_json,
        {
            "config": {"audit_policy": "all", "seed": 0},
            "results": [
                {
                    "state_dim": 32,
                    "train_size": 128,
                    "distance_to_hll_floor_rel_rmse": 0.05,
                    "ratio_to_hll_floor_rel_rmse": 1.2,
                    "train_total_queries_estimate": 256,
                    "learned_metrics": {"relative_rmse": 0.12},
                    "hll_metrics": {"relative_rmse": 0.07},
                    "exact_set_metrics": {"relative_rmse": 0.0},
                    "sum_leaf_uniques_metrics": {"relative_rmse": 0.35},
                }
            ],
        },
    )
    hll_json = tmp_path / "hll_merge" / "summary.json"
    _write_json(
        hll_json,
        {
            "config": {"seed": 0},
            "raw_rows": [
                {
                    "precision": 6,
                    "memory_bits": 384,
                    "train_docs": 128,
                    "audit_policy": "all",
                    "learned_relative_rmse": 0.18,
                    "hll_relative_rmse": 0.14,
                    "hll_rse_theory": 0.13,
                }
            ],
        },
    )

    rc = cardinality.main(
        [
            "--output-root",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "figures" / "cardinality"),
            "--no-emit-pdf",
        ]
    )
    assert rc == 0
    out_dir = tmp_path / "figures" / "cardinality"
    assert (out_dir / "cardinality_latest.md").exists()
    assert (out_dir / "cardinality_latest_diagnostics.json").exists()
    assert (out_dir / "cardinality_learning_curves.png").exists()
    assert (out_dir / "cardinality_negative_control.png").exists()
    assert (out_dir / "hll_merge_learning_memory.png").exists()
    assert (out_dir / "hll_merge_learning_memory_median.png").exists()
