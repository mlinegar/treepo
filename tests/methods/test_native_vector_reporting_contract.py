from __future__ import annotations

from pathlib import Path

import pytest

from treepo import fit
from treepo.forest import OracleTargetSpec
from treepo.methods.families import resolve_family
from treepo.tree import TreeRecord


def _vector_trees(
    target_names: tuple[str, ...],
    *,
    split: str = "test",
) -> list[TreeRecord]:
    trees = []
    for index in range(3):
        target = {
            name: (coordinate + index + 1) / (len(target_names) + 4)
            for coordinate, name in enumerate(target_names)
        }
        trees.append(
            TreeRecord(
                tree_id=f"{split}-{index}",
                text=f"document {index}",
                root_label=None,
                metadata={"split": split, "target_vector": target},
            )
        )
    return trees


def _fit_vector_dspy(
    tmp_path: Path,
    target_names: tuple[str, ...],
):
    trees = _vector_trees(target_names)
    return fit(
        {
            "space_kind": "fixture.native_vector_reporting.v1",
            "family": "dspy",
            "schedule": "fg",
            "oracle_targets": tuple(
                OracleTargetSpec(
                    target_name=name,
                    oracle_id="fixture:oracle",
                )
                for name in target_names
            ),
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {
                "output_dir": str(tmp_path),
                "target_vector_key": "target_vector",
                "target_min": 0.0,
                "target_max": 1.0,
                "dspy_program": lambda *, tree, **_kwargs: dict(
                    tree.metadata["target_vector"]
                ),
                "audit_laws": False,
            },
            "axis": {"max_iterations": 1, "axis_value": len(target_names)},
        }
    )


def test_native_k1_named_vector_projects_only_in_scalar_reporting(
    tmp_path: Path,
) -> None:
    result = _fit_vector_dspy(tmp_path, ("rile_normalized",))
    rows = result.history[-1]["extra"]["prediction_rows"]
    split = result.history[-1]["split_metrics"]["all"]

    assert result.status == "success"
    assert result.metrics["n"] == 3
    assert split["n"] == 3
    assert split["joint_f_l1_n"] == 3
    assert split["joint_f_l1"] == pytest.approx(0.0)
    assert all(
        row["prediction_scalar"]
        == row["prediction_by_target"]["rile_normalized"]
        for row in rows
    )


def test_native_k3_keeps_scalar_reporting_null(tmp_path: Path) -> None:
    result = _fit_vector_dspy(tmp_path, ("left", "other", "right"))
    rows = result.history[-1]["extra"]["prediction_rows"]

    assert result.status == "success"
    assert result.metrics["n"] == 3
    assert all(row["prediction_scalar"] is None for row in rows)


def test_duplicate_json_coordinate_keys_fail_closed() -> None:
    family = resolve_family(
        "dspy",
        {
            "target_names": ("left", "other", "right"),
            "target_oracle_ids": ("fixture:oracle",) * 3,
            "target_dim": 3,
            "target_min": 0.0,
            "target_max": 1.0,
            "dspy_program": lambda **_kwargs: (
                '{"left":0.1,"left":0.7,"other":0.2,"right":0.1}'
            ),
            "audit_laws": False,
        },
    )

    assert family.score_roots_with_f(
        f=None,
        g=None,
        trees=_vector_trees(("left", "other", "right"))[:1],
    ) == [None]
