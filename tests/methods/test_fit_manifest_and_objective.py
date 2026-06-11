"""Parity surface: manifest sidecar + ObjectiveSpec passthrough.

Verifies that ``fit()`` writes a JSON manifest with the spec snapshot,
final metrics, and (when provided) the ``ObjectiveSpec`` — matching the
sidecar shape that the existing paper scripts under ``scripts/`` use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from treepo._research.ctreepo.contracts import CTreePOLearningSpec, ObjectiveSpec
from treepo.methods import fit
from treepo.methods.fixtures import make_hll_token_trees


def _make_oracle_spec(tmp_path: Path, *, objective=None) -> CTreePOLearningSpec:
    backend_config = {
        "oracle_name": "hll_exact",
        "output_dir": str(tmp_path),
    }
    if objective is not None:
        backend_config["objective"] = objective
    return CTreePOLearningSpec(
        space_kind="manifest_check",
        family="oracle",
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=make_hll_token_trees(n_trees=4, seed=3),
        backend_config=backend_config,
        axis={"max_iterations": 0, "axis_value": 0},
    )


def test_fit_writes_manifest_sidecar(tmp_path: Path) -> None:
    result = fit(_make_oracle_spec(tmp_path))
    assert result.status == "success"
    assert result.manifest_path is not None

    manifest_path = Path(result.manifest_path)
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())

    assert payload["status"] == "success"
    assert payload["spec"]["family"] == "oracle"
    assert payload["spec"]["schedule"] == "fg"
    assert payload["n_iterations"] >= 1
    assert "internal_f_mae" in payload["metrics"]
    assert payload["objective"] is None  # nothing supplied


def test_fit_records_objective_spec_in_manifest(tmp_path: Path) -> None:
    objective = ObjectiveSpec(
        objective_family="local_law",
        local_law_estimator="corrected",
        local_law_weight=0.5,
        root_share=0.5,
    )
    result = fit(_make_oracle_spec(tmp_path, objective=objective))
    assert result.status == "success"

    payload = json.loads(Path(result.manifest_path).read_text())
    obj_payload = payload["objective"]
    assert obj_payload is not None
    assert obj_payload["local_law_weight"] == 0.5
    assert obj_payload["root_share"] == 0.5
    assert obj_payload["objective_family"] == "local_law"

    # And the summary surface in CTreePOFitResult mirrors it.
    summary_obj = result.summary.get("objective")
    assert summary_obj is not None
    assert summary_obj["local_law_weight"] == 0.5


def test_fit_objective_can_be_passed_as_mapping(tmp_path: Path) -> None:
    """For configs loaded from TOML/JSON, ObjectiveSpec arrives as a dict;
    fit() should coerce it.
    """
    objective_mapping = {
        "objective_family": "root_only",
        "local_law_estimator": "none",
        "root_share": 1.0,
        "local_law_weight": 0.0,
    }
    result = fit(_make_oracle_spec(tmp_path, objective=objective_mapping))
    assert result.status == "success"
    payload = json.loads(Path(result.manifest_path).read_text())
    assert payload["objective"]["objective_family"] == "root_only"
    assert payload["objective"]["local_law_weight"] == 0.0


def test_fit_rejects_malformed_objective(tmp_path: Path) -> None:
    """A non-mapping, non-ObjectiveSpec value must surface a TypeError —
    silently dropping a misconfigured objective is the kind of defensive
    fallback the kill list forbids.
    """
    with pytest.raises(TypeError, match="objective"):
        fit(_make_oracle_spec(tmp_path, objective=42))
