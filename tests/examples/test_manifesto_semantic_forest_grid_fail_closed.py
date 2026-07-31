from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "methods"
    / "run_manifesto_semantic_forest_grid.py"
)


def _load_example():
    module_name = "treepo_manifesto_semantic_forest_grid_fail_closed_example"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


grid = _load_example()


def test_fit_success_with_parse_drop_fails_exact_roster_coverage(
    tmp_path: Path,
) -> None:
    def parse_drop_fit(_config):
        return SimpleNamespace(
            status="success",
            metrics={},
            artifacts={"f": {"kind": "test_f"}, "g": {"kind": "test_g"}},
            summary={
                "family": "dspy",
                "model_artifact_contract": {"mode": "independent"},
            },
            manifest_path=None,
            history=[{"extra": {"prediction_rows": []}}],
        )

    cells = grid.build_grid_plan(
        tmp_path,
        seed=31,
        dspy_execution="offline_fixture",
    )
    cell = next(cell for cell in cells if cell.representation_path == "ctree_recursive")
    report = grid.fit_grid_cell(cell, fit_fn=parse_drop_fit)
    coverage = report["prediction_coverage"]

    assert report["status"] == "failed"
    assert coverage["expected_n"] == 4
    assert coverage["observed_complete_vector_n"] == 0
    assert coverage["complete"] is False
    assert len(coverage["missing_tree_ids"]) == 4
    assert coverage["duplicate_tree_ids"] == []
    assert "incomplete" in report["error"]


def test_duplicate_id_cannot_hide_a_missing_eval_document(tmp_path: Path) -> None:
    seen_configs = []

    def duplicate_fit(config):
        seen_configs.append(config)
        evaluation = list(config["eval_data"])
        target_key = config["backend_config"]["target_vector_key"]
        selected = [evaluation[0], evaluation[0], evaluation[2], evaluation[3]]
        rows = []
        for tree in selected:
            target = dict(tree.metadata[target_key])
            rows.append(
                {
                    "tree_id": tree.tree_id,
                    "split": "test",
                    "prediction_by_target": target,
                    "target_by_name": target,
                }
            )
        return SimpleNamespace(
            status="success",
            metrics={"joint_f_l1": 0.0},
            summary={
                "family": "dspy",
                "model_artifact_contract": {"mode": "independent"},
            },
            artifacts={"f": {"kind": "test_f"}, "g": {"kind": "test_g"}},
            manifest_path=None,
            history=[{"extra": {"prediction_rows": rows}}],
        )

    cells = grid.build_family_grid_plan(
        tmp_path,
        "dspy",
        seed=37,
        dspy_execution="offline_fixture",
    )
    cell = next(cell for cell in cells if cell.representation_path == "ctree_recursive")
    report = grid.fit_grid_cell(cell, fit_fn=duplicate_fit)
    coverage = report["prediction_coverage"]
    evaluation_ids = [tree.tree_id for tree in seen_configs[0]["eval_data"]]

    assert coverage["observed_complete_vector_n"] == 4
    assert coverage["duplicate_tree_ids"] == [evaluation_ids[0]]
    assert coverage["missing_tree_ids"] == [evaluation_ids[1]]
    assert coverage["complete"] is False
    assert report["status"] == "failed"
    assert report["live_llm_call"] is False
    assert report["dspy_optimization_executed"] is False
    backend_metadata = seen_configs[0]["backend_config"]["metadata"]
    assert backend_metadata["fixture_backend"] == "deterministic_python_oracle"
    assert backend_metadata["satisfies_live_llm_empirical_requirement"] is False


def test_spoofed_gold_vector_cannot_authenticate_itself(tmp_path: Path) -> None:
    def spoofed_gold_fit(config):
        names = tuple(target.target_name for target in config["oracle_targets"])
        spoofed = {name: 0.0 for name in names}
        rows = [
            {
                "tree_id": tree.tree_id,
                "split": "test",
                "prediction_by_target": dict(spoofed),
                "target_by_name": dict(spoofed),
            }
            for tree in config["eval_data"]
        ]
        return SimpleNamespace(
            status="success",
            metrics={"joint_f_l1": 0.0},
            summary={
                "family": "dspy",
                "model_artifact_contract": {"mode": "independent"},
            },
            manifest_path=None,
            artifacts={"f": {"kind": "test_f"}, "g": {"kind": "test_g"}},
            history=[{"extra": {"prediction_rows": rows}}],
        )

    cell = grid.GridCell(
        target_width=3,
        representation_path="ctree_recursive",
        family="dspy",
        seed=43,
        output_dir=tmp_path / "spoofed_gold",
        dspy_execution="offline_fixture",
    )
    report = grid.fit_grid_cell(cell, fit_fn=spoofed_gold_fit)
    coverage = report["prediction_coverage"]
    expected_ids = sorted(grid._test_document_ids())

    assert report["status"] == "failed"
    assert coverage["complete"] is False
    assert coverage["target_mismatch_tree_ids"] == expected_ids
    assert coverage["observed_complete_vector_n"] == 0
    assert coverage["missing_tree_ids"] == expected_ids
    assert coverage["incomplete_row_tree_ids"] == []
    assert report["common_metrics"]["n"] == 0
    assert report["common_metrics"]["mean_joint_l1"] is None
    assert report["common_metrics"]["raw_rile_mae"] is None
    assert "target_mismatches" in report["error"]
