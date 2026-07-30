from __future__ import annotations

import json
from pathlib import Path

import pytest

from treepo import fit
from treepo.forest import OracleTargetSpec
from treepo.methods.families import resolve_family
from treepo.tree import TreeRecord

TARGETS = (
    OracleTargetSpec(target_name="left", oracle_id="fixture:cmp"),
    OracleTargetSpec(target_name="other", oracle_id="fixture:cmp"),
    OracleTargetSpec(target_name="right", oracle_id="fixture:cmp"),
)


def _trees(split: str) -> list[TreeRecord]:
    return [
        TreeRecord(
            tree_id=f"{split}-a",
            text="left platform",
            metadata={
                "split": split,
                "shares": {"left": 0.7, "other": 0.2, "right": 0.1},
            },
        ),
        TreeRecord(
            tree_id=f"{split}-b",
            text="right platform",
            metadata={
                "split": split,
                "shares": {"left": 0.1, "other": 0.2, "right": 0.7},
            },
        ),
    ]


def _oracle_program(*, tree, **_kwargs):
    return json.dumps(dict(tree.metadata["shares"]), sort_keys=True)


def test_dspy_family_parses_exact_named_json_and_rejects_partial_vectors() -> None:
    config = {
        "target_names": ("left", "other", "right"),
        "target_oracle_ids": ("fixture:cmp",) * 3,
        "target_dim": 3,
        "target_vector_key": "shares",
        "dspy_program": lambda **_kwargs: (
            '{"left":0.2,"other":0.5,"right":0.3}'
        ),
        "audit_laws": False,
    }
    family = resolve_family("dspy", config)
    assert family.score_roots_with_f(
        f=None,
        g=None,
        trees=_trees("test")[:1],
    ) == [{"left": 0.2, "other": 0.5, "right": 0.3}]

    partial = resolve_family(
        "dspy",
        {
            **config,
            "dspy_program": lambda **_kwargs: '{"left":0.2,"right":0.8}',
        },
    )
    assert partial.score_roots_with_f(
        f=None,
        g=None,
        trees=_trees("test")[:1],
    ) == [None]


def test_treepo_fit_runs_one_joint_dspy_vector_and_reports_sum_l1(
    tmp_path: Path,
) -> None:
    result = fit(
        {
            "space_kind": "fixture.shares.v1",
            "family": "dspy",
            "schedule": "fg",
            "oracle_targets": TARGETS,
            "train_data": _trees("train"),
            "eval_data": _trees("test"),
            "backend_config": {
                "output_dir": str(tmp_path),
                "target_vector_key": "shares",
                "node_target_exclusive": True,
                "dspy_program": _oracle_program,
                "audit_laws": False,
            },
            "axis": {"max_iterations": 1, "axis_value": 1},
        }
    )

    assert result.status == "success"
    assert result.metrics["joint_f_l1"] == pytest.approx(0.0)
    assert result.summary["definition"] == "single_shared_g_joint_vector_f_star"

    results = json.loads(
        Path(result.summary["results_path"]).read_text(encoding="utf-8")
    )
    assert results["paired_rows"]["named_vector_fields"]["prediction"] == (
        "prediction_by_target"
    )
    prediction_files = results["paired_rows"]["files"]
    rows = [
        json.loads(line)
        for line in Path(prediction_files[-1]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[0]["target_order"] == ["left", "other", "right"]
    assert set(rows[0]["prediction_by_target"]) == {"left", "other", "right"}


def test_singleton_named_dspy_uses_the_same_json_vector_contract() -> None:
    family = resolve_family(
        "dspy",
        {
            "target_names": ("rile_normalized",),
            "target_oracle_ids": ("fixture:rile",),
            "target_dim": 1,
            "dspy_program": lambda **_kwargs: '{"rile_normalized":0.625}',
            "audit_laws": False,
        },
    )
    assert family.score_roots_with_f(
        f=None,
        g=None,
        trees=_trees("test")[:1],
    ) == [{"rile_normalized": 0.625}]
