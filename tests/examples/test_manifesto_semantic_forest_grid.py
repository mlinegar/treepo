from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "methods"
    / "run_manifesto_semantic_forest_grid.py"
)


def _load_example():
    module_name = "treepo_manifesto_semantic_forest_grid_example"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


grid = _load_example()

EXPECTED_TARGET_API_CONTRACT = {
    "target_api": "ordered_named_vector",
    "target_widths": [1, 3, 57],
    "k_changes_only": "ordered_target_catalog_and_output_width",
    "point_metric": "sum_absolute_coordinate_error",
    "k1_scalar_api_branch": False,
    "raw_rile_readout": "reporting_only",
}

REPRESENTATION_PATHS = (
    "full_doc_direct",
    "ctree_base_summary",
    "ctree_recursive",
)
LEARNED_DSPY_BACKEND_CONFIG = {
    "optimizer": "bootstrap_random_search",
    "lm_config": {
        "model": "fixture/test-model",
        "api_base": "http://127.0.0.1:9999/v1",
    },
}
PATH_CONTRACTS = {
    "full_doc_direct": {
        "representation_path": "full_doc_direct",
        "representation": "full_doc",
        "topology_kind": "singleton",
        "leaf_count": 1,
        "merge_application_count": 0,
        "readout_path": "f_direct",
        "leaf_g_application_count": 0,
        "composition_present": False,
    },
    "ctree_base_summary": {
        "representation_path": "ctree_base_summary",
        "representation": "ctree",
        "topology_kind": "singleton",
        "leaf_count": 1,
        "merge_application_count": 0,
        "readout_path": "f_after_g",
        "leaf_g_application_count": 1,
        "composition_present": False,
    },
    "ctree_recursive": {
        "representation_path": "ctree_recursive",
        "representation": "ctree",
        "topology_kind": "recursive",
        "leaf_count": 4,
        "merge_application_count": 3,
        "readout_path": "f_after_reduce_g",
        "leaf_g_application_count": 4,
        "composition_present": True,
    },
}


def _expected_g_contract(
    family: str,
    representation_path: str,
    *,
    requested_max_iterations: int,
    dspy_execution: str = "learned",
) -> dict[str, object]:
    path_contract = dict(PATH_CONTRACTS[representation_path])
    if representation_path == "full_doc_direct":
        g_mode = "identity"
    elif family == "dspy" and dspy_execution == "offline_fixture":
        g_mode = "fixed"
    else:
        g_mode = "learned"
    reuses_model_artifacts = representation_path == "ctree_base_summary"
    effective_max_iterations = (
        0
        if reuses_model_artifacts
        else requested_max_iterations
        if g_mode == "learned"
        else int(requested_max_iterations > 0)
    )
    expected_g_update_count = effective_max_iterations // 2 if g_mode == "learned" else 0
    train_g_called = expected_g_update_count > 0
    roles = ["leaf"] if train_g_called else []
    merge_domain = bool(train_g_called and path_contract["composition_present"])
    if merge_domain:
        roles.append("merge")

    if g_mode == "identity":
        implementation = "identity_shared_g"
        learner = "none_identity"
        learning_evidence = "none_identity"
    elif g_mode == "fixed":
        implementation = "denominator_weighted_analytic_fixture_g"
        learner = "none_analytic_wiring"
        learning_evidence = "analytic_wiring_only_not_learned_shared_g"
    elif family == "dspy":
        implementation = "treepo_dspy_shared_g"
        learner = "treepo_dspy_shared_fg_learner"
        learning_evidence = "realized_dspy_compiler_update_required"
    else:
        implementation = "treepo_fno_shared_g"
        learner = "treepo_fno_shared_g_learner"
        learning_evidence = "realized_fit_update_required"

    if not train_g_called:
        evidence_source = "no_train_g_calls"
    elif family == "dspy":
        evidence_source = "dspy_compiler_mixed_examples"
    else:
        evidence_source = "g_artifact_gradient_path_presence"

    return {
        **path_contract,
        "g_mode": g_mode,
        "g_trainable": g_mode == "learned",
        "expected_g_update_count": expected_g_update_count,
        "learned_g_expected": expected_g_update_count > 0,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
        "model_scope": (
            "direct_control_separate_from_shared_ctree_pair"
            if representation_path == "full_doc_direct"
            else "one_f_g_pair_per_target_width_and_family"
        ),
        "reuses_model_artifacts": reuses_model_artifacts,
        "shared_model_source_path": (
            "ctree_recursive" if reuses_model_artifacts else representation_path
        ),
        "expected_g_training_call_roles": roles,
        "expected_g_training_role_evidence_source": evidence_source,
        "expected_merge_domain_training_observed": merge_domain,
        "expected_shared_g_updated_with_merge_domain": bool(
            expected_g_update_count > 0 and merge_domain
        ),
        "schedule": "fg" if g_mode == "learned" else "f",
        "requested_max_iterations": requested_max_iterations,
        "effective_max_iterations": effective_max_iterations,
        "implementation": implementation,
        "learner": learner,
        "learning_evidence": learning_evidence,
    }


def test_default_plan_is_the_complete_three_by_three_by_two_grid(tmp_path: Path) -> None:
    cells = grid.build_grid_plan(tmp_path, seed=19)

    assert len(cells) == 18
    assert len({cell.cell_id for cell in cells}) == 18
    assert {(cell.target_width, cell.representation_path, cell.family) for cell in cells} == {
        (width, representation_path, family)
        for width in (1, 3, 57)
        for representation_path in REPRESENTATION_PATHS
        for family in ("dspy", "fno")
    }
    assert all(cell.seed == 19 for cell in cells)
    for cell in cells:
        payload = cell.to_dict()
        expected_g = _expected_g_contract(
            cell.family,
            cell.representation_path,
            requested_max_iterations=3,
        )
        assert payload["g_mode"] == expected_g["g_mode"]
        assert payload["schedule"] == expected_g["schedule"]
        assert payload["leaf_count"] == expected_g["leaf_count"]
        assert payload["g_contract"] == expected_g
        for realized_field in (
            "g_training_call_roles",
            "g_training_role_evidence_source",
            "merge_domain_training_observed",
            "shared_g_updated_with_merge_domain",
        ):
            assert realized_field not in payload
            assert realized_field not in payload["g_contract"]
        assert payload["target_api_contract"] == EXPECTED_TARGET_API_CONTRACT
        if cell.family == "dspy":
            assert payload["substrate"] == "llm"
            assert payload["optimizer"] == "dspy_optimizer"
            assert payload["dspy_execution"] == "learned"
            assert payload["analytic_wiring_only"] is False
        else:
            assert payload["substrate"] == "neural_operator_fno"
            assert payload["optimizer"] == "gradient"


def test_plan_separates_expected_learned_contract_from_realized_evidence(
    tmp_path: Path,
) -> None:
    learned = grid._plan_payload(
        tmp_path,
        selected_family="dspy",
        seed=20,
        dspy_execution="learned",
        dspy_config_supplied=False,
    )
    dspy_plan = learned["family_grids"]["dspy"]
    assert dspy_plan["dspy_execution"] == "learned"
    assert dspy_plan["dspy_runtime_config_required"] is True
    assert dspy_plan["dspy_runtime_config_supplied"] is False
    assert dspy_plan["g_contracts"]["ctree_base_summary"]["g_mode"] == "learned"
    assert (
        dspy_plan["g_contracts"]["ctree_recursive"]["implementation"]
        == (dspy_plan["g_contracts"]["ctree_base_summary"]["implementation"])
    )
    for cell in dspy_plan["cells"]:
        assert "executed_g_contract" not in cell
        assert "g_training_call_roles" not in cell
        assert "expected_g_training_call_roles" in cell["g_contract"]

    offline = grid._plan_payload(
        tmp_path,
        selected_family="dspy",
        seed=20,
        dspy_execution="offline_fixture",
    )
    offline_plan = offline["family_grids"]["dspy"]
    assert offline_plan["dspy_runtime_config_required"] is False
    assert offline_plan["g_contracts"]["ctree_recursive"]["g_mode"] == "fixed"
    assert offline_plan["dspy_optimization_executed"] is False
    assert offline_plan["learned_shared_llm_g_evidence"] is False


def test_ctree_paths_share_one_g_implementation_and_learner() -> None:
    for family in ("dspy", "fno"):
        base = grid._g_contract(
            family,
            "ctree_base_summary",
            requested_max_iterations=2,
        )
        recursive = grid._g_contract(
            family,
            "ctree_recursive",
            requested_max_iterations=2,
        )
        assert base["implementation"] == recursive["implementation"]
        assert base["learner"] == recursive["learner"]
        assert base["same_g_across_node_roles"] is True
        assert recursive["same_g_across_node_roles"] is True
        assert base["reduce_g_is_derived"] is True
        assert recursive["reduce_g_is_derived"] is True

    dspy = grid._g_contract(
        "dspy",
        "ctree_recursive",
        requested_max_iterations=2,
    )
    assert dspy["learner"] == "treepo_dspy_shared_fg_learner"
    assert dspy["learning_evidence"] == "realized_dspy_compiler_update_required"
    assert dspy["learned_g_expected"] is True
    assert dspy["expected_g_training_call_roles"] == ["leaf", "merge"]
    assert dspy["expected_g_training_role_evidence_source"] == ("dspy_compiler_mixed_examples")

    offline_base = grid._g_contract(
        "dspy",
        "ctree_base_summary",
        requested_max_iterations=2,
        dspy_execution="offline_fixture",
    )
    offline_recursive = grid._g_contract(
        "dspy",
        "ctree_recursive",
        requested_max_iterations=2,
        dspy_execution="offline_fixture",
    )
    assert offline_base["implementation"] == offline_recursive["implementation"]
    assert offline_base["learner"] == offline_recursive["learner"] == "none_analytic_wiring"
    assert offline_recursive["g_mode"] == "fixed"
    assert offline_recursive["learned_g_expected"] is False


def test_representation_paths_have_identical_root_targets_for_every_k() -> None:
    for target_width in (1, 3, 57):
        by_path = {}
        for representation_path in REPRESENTATION_PATHS:
            train_raw, test_raw = grid.make_cmp_count_records(representation_path)
            records, fragment = grid.prepare_target_records(
                [*train_raw, *test_raw],
                target_width,
            )
            target_key = fragment["backend_config"]["target_vector_key"]
            by_path[representation_path] = {
                record.tree_id: dict(record.metadata[target_key]) for record in records
            }

        assert all(by_path[path] == by_path["full_doc_direct"] for path in REPRESENTATION_PATHS)


def test_k1_uses_common_grid_unit_interval_and_raw_readout() -> None:
    records, fragment = grid.prepare_target_records(
        grid.make_cmp_count_records("ctree_recursive")[0],
        1,
    )

    assert fragment["backend_config"]["target_min"] == 0.0
    assert fragment["backend_config"]["target_max"] == 1.0
    assert fragment["oracle_targets"][0].metadata["bounds"] == [0.0, 1.0]
    for record in records:
        value = record.metadata[grid.RILE_SINGLETON_KEY][grid.RILE_SINGLETON_NAME]
        assert 0.0 <= value <= 1.0
        assert record.metadata["rile_from_components"] == pytest.approx(200.0 * value - 100.0)


def test_grid_supplies_task_owned_rile_instructions_at_every_width() -> None:
    expected_terms = {
        1: ("rile_normalized", "nonheader", "other"),
        3: ("rile", "left", "right", "other"),
        57: ("rile", "cmp", "residual"),
    }
    raw = grid.make_cmp_count_records("ctree_recursive")[0]

    for width in (1, 3, 57):
        _records, fragment = grid.prepare_target_records(raw, width)
        backend = fragment["backend_config"]
        f_instructions = str(backend["f_signature_instructions"]).strip()
        g_instructions = str(backend["g_signature_instructions"]).strip()
        combined = f"{f_instructions}\n{g_instructions}".lower()

        assert f_instructions and g_instructions
        assert "strict json" in f_instructions.lower()
        assert all(role in g_instructions.lower() for role in ("leaf", "unary", "binary"))
        assert all(term in combined for term in expected_terms[width])


@pytest.mark.parametrize("representation_path", ["ctree_base_summary", "ctree_recursive"])
def test_ctree_dspy_program_traverses_leaves_and_uses_weighted_analytic_g(
    representation_path: str,
) -> None:
    raw = grid.make_cmp_count_records(representation_path)[1][0]
    records, fragment = grid.prepare_target_records([raw], 3)
    record = records[0]
    target_key = fragment["backend_config"]["target_vector_key"]
    expected = dict(record.metadata[target_key])
    poisoned_root_metadata = dict(record.metadata)
    poisoned_root_metadata[target_key] = {name: 99.0 for name in expected}
    poisoned = replace(record, metadata=poisoned_root_metadata)

    program = grid.FixtureOracleDSPyProgram(target_key)
    prediction = program(tree=poisoned)

    assert prediction == pytest.approx(expected)
    assert prediction != poisoned_root_metadata[target_key]


def test_learned_dspy_grid_fails_closed_without_optimizer_lm_or_program_config(
    tmp_path: Path,
) -> None:
    cell = grid.GridCell(
        target_width=3,
        representation_path="ctree_recursive",
        family="dspy",
        seed=21,
        output_dir=tmp_path / "learned",
    )

    with pytest.raises(ValueError, match="dspy_backend_config"):
        grid.fit_grid_cell(cell, fit_fn=lambda _config: None)

    with pytest.raises(ValueError, match="optimizer"):
        grid.fit_grid_cell(
            cell,
            dspy_backend_config={
                "lm_config": LEARNED_DSPY_BACKEND_CONFIG["lm_config"],
            },
            fit_fn=lambda _config: None,
        )


def test_offline_fixture_is_explicit_fixed_g_and_never_runs_an_optimizer() -> None:
    contracts = {
        path: grid._g_contract(
            "dspy",
            path,
            requested_max_iterations=2,
            dspy_execution="offline_fixture",
        )
        for path in REPRESENTATION_PATHS
    }

    assert contracts["full_doc_direct"]["g_mode"] == "identity"
    assert contracts["ctree_base_summary"]["g_mode"] == "fixed"
    assert contracts["ctree_recursive"]["g_mode"] == "fixed"
    assert contracts["ctree_recursive"]["expected_g_update_count"] == 0
    assert grid._optimizer("dspy", dspy_execution="offline_fixture") == ("none_offline_fixture")
    provenance = grid._dspy_execution_provenance("offline_fixture")
    assert provenance["dspy_optimization_executed"] is False
    assert provenance["learned_shared_llm_g_evidence"] is False
    assert provenance["analytic_wiring_only"] is True


def test_complete_grid_routes_every_cell_through_fit_and_common_metrics(
    tmp_path: Path,
) -> None:
    requested_max_iterations = 5
    calls = []

    def fake_fit(config):
        calls.append(config)
        names = tuple(target.target_name for target in config["oracle_targets"])
        target_key = config["backend_config"]["target_vector_key"]
        rows = []
        for tree in config["eval_data"]:
            target = dict(tree.metadata[target_key])
            assert tuple(target) == names
            rows.append(
                {
                    "tree_id": tree.tree_id,
                    "split": tree.metadata["split"],
                    "target_order": list(names),
                    "prediction_by_target": dict(target),
                    "target_by_name": dict(target),
                }
            )
        initial = dict(config.get("initial_artifacts") or {})
        if initial.get("f") is not None and initial.get("g") is not None:
            artifacts = {"f": initial["f"], "g": initial["g"]}
        else:
            artifacts = {
                "f": {
                    "kind": "fake_f",
                    "trained": "f",
                    "width": len(names),
                    "family": config["family"],
                },
                "g": initial.get("g")
                or {
                    "kind": "fake_g",
                    "trained": "g",
                    "g_mode": config["g_mode"],
                    "same_g_across_node_roles": True,
                    "reduce_g_is_derived": True,
                    "width": len(names),
                    "family": config["family"],
                },
            }
        return SimpleNamespace(
            status="success",
            metrics={"joint_f_l1": 0.0},
            artifacts=artifacts,
            summary={"family": config["family"]},
            manifest_path=None,
            history=[{"extra": {"prediction_rows": rows}}],
        )

    report = grid.run_complete_grid(
        tmp_path,
        seed=23,
        fit_fn=fake_fit,
        max_iterations=requested_max_iterations,
        dspy_backend_config=LEARNED_DSPY_BACKEND_CONFIG,
    )

    assert len(calls) == 18
    assert report["grid"]["expected_cells"] == 18
    assert report["grid"]["completed_cells"] == 18
    assert report["grid"]["failed_cells"] == 0
    assert report["scientific_claim"] is False
    assert "not_polmeth_evidence" in report["evidence_class"]
    assert report["execution_contracts"]["dspy"]["substrate"] == "llm"
    assert report["execution_contracts"]["dspy"]["learned_shared_g_evidence"] is None
    assert (
        report["execution_contracts"]["dspy"]["expected_learned_shared_g_updates_in_ctree_cells"]
        is True
    )
    assert report["target_api_contract"] == EXPECTED_TARGET_API_CONTRACT
    assert report["comparison_report_schema_identical_across_families"] is True
    assert (
        report["comparison_report_schema"]
        == report["family_grids"]["dspy"]["comparison_report_schema"]
        == report["family_grids"]["fno"]["comparison_report_schema"]
    )
    assert len(report["comparison_rows"]) == 18
    assert len(report["family_telemetry_sidecars"]) == 18
    assert (tmp_path / "semantic_forest_grid.json").is_file()

    for family in ("dspy", "fno"):
        by_path = report["execution_contracts"][family]["g_by_representation_path"]
        for representation_path in REPRESENTATION_PATHS:
            assert by_path[representation_path] == _expected_g_contract(
                family,
                representation_path,
                requested_max_iterations=requested_max_iterations,
            )

    for config in calls:
        width = len(config["oracle_targets"])
        family = config["family"]
        metadata = config["backend_config"]["metadata"]
        representation_path = metadata["representation_path"]
        expected_g = _expected_g_contract(
            family,
            representation_path,
            requested_max_iterations=requested_max_iterations,
        )
        assert width in {1, 3, 57}
        assert family in {"dspy", "fno"}
        assert config["axis"]["axis_kind"] == "target_width"
        assert config["axis"]["axis_value"] == width
        assert config["axis"]["representation"] == expected_g["representation"]
        assert config["axis"]["representation_path"] == representation_path
        assert config["g_mode"] == expected_g["g_mode"]
        assert config["schedule"] == expected_g["schedule"]
        assert config["axis"]["max_iterations"] == expected_g["effective_max_iterations"]
        assert config["axis"]["leaf_count"] == expected_g["leaf_count"]
        for field in (
            "topology_kind",
            "merge_application_count",
            "readout_path",
            "leaf_g_application_count",
            "composition_present",
            "same_g_across_node_roles",
            "reduce_g_is_derived",
        ):
            assert config["axis"][field] == expected_g[field]
        assert config["backend_config"]["target_vector_key"]
        assert metadata["substrate"] == ("llm" if family == "dspy" else "neural_operator_fno")
        assert metadata["optimizer"] == ("dspy_optimizer" if family == "dspy" else "gradient")
        assert metadata["g_mode"] == expected_g["g_mode"]
        assert metadata["schedule"] == expected_g["schedule"]
        assert metadata["requested_max_iterations"] == requested_max_iterations
        assert metadata["effective_max_iterations"] == expected_g["effective_max_iterations"]
        assert metadata["g_contract"] == expected_g
        if expected_g["reuses_model_artifacts"]:
            assert set(config["initial_artifacts"]) == {"f", "g"}
            assert config["axis"]["max_iterations"] == 0
        elif expected_g["g_mode"] == "fixed":
            assert config["initial_artifacts"] == {
                "g": {
                    "kind": "manifesto_fixture_fixed_g",
                    "g_mode": "fixed",
                    "operator": expected_g["implementation"],
                    "trainable": False,
                    "train_g_enabled": False,
                    "same_g_across_node_roles": True,
                    "reduce_g_is_derived": True,
                    "learning_evidence": ("analytic_wiring_only_not_learned_shared_g"),
                }
            }
        else:
            assert "initial_artifacts" not in config
        if family == "dspy":
            assert config["backend_config"]["optimizer"] == "bootstrap_random_search"
            assert (
                config["backend_config"]["lm_config"] == (LEARNED_DSPY_BACKEND_CONFIG["lm_config"])
            )
            assert config["backend_config"]["f_record_source"] == "generated_when_available"
            if expected_g["g_mode"] == "learned":
                assert config["backend_config"]["allow_identity_g_targets"] is True
                assert config["backend_config"]["g_target_source"] == (
                    "explicit_reference_text_fixture"
                )
            else:
                assert "allow_identity_g_targets" not in config["backend_config"]
            assert "dspy_program" not in config["backend_config"]
            assert metadata["dspy_execution"] == "learned"
        else:
            assert config["backend_config"]["device"] == "cpu"
            assert config["backend_config"]["epochs_per_iteration"] == 1

    for cell in report["cells"]:
        expected_g = _expected_g_contract(
            cell["family"],
            cell["representation_path"],
            requested_max_iterations=requested_max_iterations,
        )
        assert cell["common_metrics"]["n"] == 4
        assert cell["common_metrics"]["mean_joint_l1"] == pytest.approx(0.0)
        assert cell["common_metrics"]["raw_rile_mae"] == pytest.approx(0.0)
        assert cell["substrate"] in {"llm", "neural_operator_fno"}
        assert cell["optimizer"] in {"dspy_optimizer", "gradient"}
        assert cell["g_mode"] == expected_g["g_mode"]
        assert cell["schedule"] == expected_g["schedule"]
        assert cell["leaf_count"] == expected_g["leaf_count"]
        assert cell["g_contract"] == expected_g
        assert cell["executed_g_contract"] is None
        for key, expected_value in expected_g.items():
            assert cell["fit_contract"][key] == expected_value
        for key, expected_value in EXPECTED_TARGET_API_CONTRACT.items():
            assert cell["fit_contract"][key] == expected_value
        telemetry = cell["family_telemetry"]
        for key, expected_value in expected_g.items():
            assert telemetry[key] == expected_value
        assert telemetry["target_api_contract"] == EXPECTED_TARGET_API_CONTRACT
        row = cell["comparison_row"]
        assert row["cell_id"] == cell["cell_id"]
        assert row["identity"]["width"] == cell["target_width"]
        assert row["identity"]["representation_path"] == cell["representation_path"]
        for field in (
            "topology_kind",
            "leaf_count",
            "merge_application_count",
            "readout_path",
            "leaf_g_application_count",
            "composition_present",
        ):
            assert row["identity"][field] == expected_g[field]
        assert row["identity"]["g_mode"] == expected_g["g_mode"]
        assert row["identity"]["g_operator"] is None
        assert row["identity"]["g_fit_status"] is None
        assert row["identity"]["g_update_count"] is None
        assert row["identity"]["learned_this_run"] is None
        assert row["identity"]["same_g_across_node_roles"] is True
        assert row["identity"]["reduce_g_is_derived"] is True
        assert row["identity"]["g_training_call_roles"] is None
        assert row["identity"]["g_training_role_evidence_source"] is None
        assert row["identity"]["merge_domain_training_observed"] is None
        assert row["identity"]["shared_g_updated_with_merge_domain"] is None
        assert row["identity"]["schedule"] == expected_g["schedule"]
        assert row["identity"]["requested_max_iterations"] == requested_max_iterations
        assert row["identity"]["effective_max_iterations"] == expected_g["effective_max_iterations"]
        assert row["quality"]["mean_joint_sum_l1"] == pytest.approx(0.0)
        assert row["quality"]["raw_published_rile_mae"] == pytest.approx(0.0)
        assert row["quality"]["exact_prediction_coverage"]["complete"] is True
        assert row["compute_performance"]["fit_wall_seconds"] >= 0.0
        assert row["compute_performance"]["inference_wall_seconds"] is None
        assert row["measurement_status"]["status"] == "partial"
        assert (
            "compute_performance.inference_wall_seconds"
            in row["measurement_status"]["missing_field_reasons"]
        )
        for field in (
            "g_operator",
            "g_fit_status",
            "g_update_count",
            "learned_this_run",
            "g_training_call_roles",
            "g_training_role_evidence_source",
            "merge_domain_training_observed",
            "shared_g_updated_with_merge_domain",
        ):
            assert f"identity.{field}" in row["measurement_status"]["missing_field_reasons"]

    by_key = {
        (cell["family"], cell["target_width"], cell["representation_path"]): cell
        for cell in report["cells"]
    }
    for family in ("dspy", "fno"):
        for width in (1, 3, 57):
            base = by_key[(family, width, "ctree_base_summary")]
            recursive = by_key[(family, width, "ctree_recursive")]
            assert base["model_artifacts"] == recursive["model_artifacts"]
            assert base["reuses_model_artifacts"] is True
            assert base["shared_model_source_cell_id"] == recursive["cell_id"]


def test_failed_cell_leaves_realized_g_outcome_null(tmp_path: Path) -> None:
    cell = grid.GridCell(
        target_width=57,
        representation_path="ctree_recursive",
        family="fno",
        seed=27,
        output_dir=tmp_path / "failed",
    )

    report = grid._failed_cell_report(
        cell,
        error="RuntimeError: fixture failure",
        fno_epochs=1,
        max_iterations=2,
    )

    assert report["status"] == "failed"
    assert report["g_mode"] == "learned"
    assert report["executed_g_contract"] is None
    identity = report["comparison_row"]["identity"]
    assert identity["g_mode"] == "learned"
    assert identity["g_operator"] is None
    assert identity["g_fit_status"] is None
    assert identity["g_update_count"] is None
    assert identity["learned_this_run"] is None
    assert identity["same_g_across_node_roles"] is True
    assert identity["reduce_g_is_derived"] is True
    assert identity["g_training_call_roles"] is None
    assert identity["g_training_role_evidence_source"] is None
    assert identity["merge_domain_training_observed"] is None
    assert identity["shared_g_updated_with_merge_domain"] is None
    for field in (
        "g_operator",
        "g_fit_status",
        "g_update_count",
        "learned_this_run",
        "g_training_call_roles",
        "g_training_role_evidence_source",
        "merge_domain_training_observed",
        "shared_g_updated_with_merge_domain",
    ):
        assert (
            f"identity.{field}"
            in report["comparison_row"]["measurement_status"]["missing_field_reasons"]
        )


def test_common_metrics_use_sum_l1_and_common_raw_rile_scale() -> None:
    rows = [
        {
            "prediction_by_target": {"rile_normalized": 0.75},
            "target_by_name": {"rile_normalized": 0.25},
        }
    ]
    metrics = grid.summarize_common_metrics(rows, target_width=1)

    assert metrics["mean_joint_l1"] == pytest.approx(0.5)
    assert metrics["raw_rile_mae"] == pytest.approx(100.0)


def test_real_fit_executes_all_eighteen_tiny_cells(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("neuralop")

    report = grid.run_complete_grid(
        tmp_path,
        seed=29,
        fno_epochs=1,
        max_iterations=2,
        dspy_execution="offline_fixture",
    )

    assert report["grid"]["completed_cells"] == 18
    assert report["grid"]["failed_cells"] == 0
    assert {
        (
            cell["target_width"],
            cell["representation_path"],
            cell["family"],
        )
        for cell in report["cells"]
    } == {
        (width, representation_path, family)
        for width in (1, 3, 57)
        for representation_path in REPRESENTATION_PATHS
        for family in ("dspy", "fno")
    }
    assert all(cell["common_metrics"]["n"] == 4 for cell in report["cells"])
    for cell in report["cells"]:
        executed = cell["executed_g_contract"]
        assert executed is not None
        identity = cell["comparison_row"]["identity"]
        assert identity["g_mode"] == cell["g_mode"]
        assert identity["g_operator"] == executed["operator"]
        assert identity["g_fit_status"] == executed["fit_status"]
        assert identity["g_update_count"] == executed["g_update_count"]
        assert identity["learned_this_run"] == executed["learned_this_run"]
        assert identity["same_g_across_node_roles"] is True
        assert identity["reduce_g_is_derived"] is True
        for field in (
            "g_training_call_roles",
            "g_training_role_evidence_source",
            "merge_domain_training_observed",
            "shared_g_updated_with_merge_domain",
        ):
            assert identity[field] == executed[field]
        if cell["representation_path"] == "full_doc_direct":
            assert executed["operator"] == "fixed_identity"
            assert executed["fit_status"] == "not_trainable"
            assert executed["g_update_count"] == 0
            assert executed["learned_this_run"] is False
            assert executed["g_training_call_roles"] == []
            assert executed["g_training_role_evidence_source"] == "no_train_g_calls"
            assert executed["merge_domain_training_observed"] is False
            assert executed["shared_g_updated_with_merge_domain"] is False
        elif cell["representation_path"] == "ctree_base_summary" and cell["family"] == "fno":
            assert executed["operator"] == "learned_shared_reused"
            assert executed["fit_status"] == "reused_without_update"
            assert executed["g_update_count"] == 0
            assert executed["learned_this_run"] is False
            assert executed["g_training_call_roles"] == []
            assert executed["g_training_role_evidence_source"] == "no_train_g_calls"
            assert executed["merge_domain_training_observed"] is False
            assert executed["shared_g_updated_with_merge_domain"] is False
        elif cell["family"] == "dspy":
            assert executed["operator"] == "fixed_nonidentity_or_family_owned"
            assert executed["fit_status"] == "not_trainable"
            assert executed["g_update_count"] == 0
            assert executed["learned_this_run"] is False
            assert executed["g_training_call_roles"] == []
            assert executed["g_training_role_evidence_source"] == "no_train_g_calls"
            assert executed["merge_domain_training_observed"] is False
            assert executed["shared_g_updated_with_merge_domain"] is False
        else:
            assert cell["g_mode"] == "learned"
            assert executed["operator"] == "learned_shared"
            assert executed["fit_status"] == "fitted_this_run"
            assert executed["g_update_count"] == 1
            assert executed["learned_this_run"] is True
            merge_supported = cell["representation_path"] == "ctree_recursive"
            assert executed["g_training_call_roles"] == (
                ["leaf", "merge"] if merge_supported else ["leaf"]
            )
            assert executed["g_training_role_evidence_source"] == (
                "g_artifact_gradient_path_presence"
            )
            assert executed["merge_domain_training_observed"] is merge_supported
            assert executed["shared_g_updated_with_merge_domain"] is merge_supported

    by_key = {
        (cell["family"], cell["target_width"], cell["representation_path"]): cell
        for cell in report["cells"]
    }
    for family in ("dspy", "fno"):
        for width in (1, 3, 57):
            base = by_key[(family, width, "ctree_base_summary")]
            recursive = by_key[(family, width, "ctree_recursive")]
            assert base["model_artifacts"] == recursive["model_artifacts"]


def test_executed_g_contract_must_match_configured_mode_and_update_count() -> None:
    configured = grid._g_contract("fno", "ctree_recursive", requested_max_iterations=2)
    realized = {
        "mode": "learned",
        "operator": "learned_shared",
        "fit_status": "fitted_this_run",
        "g_update_count": 1,
        "learned_this_run": True,
        "declared_leaf_count": 4,
        "merge_application_count_per_tree": 3,
        "leaf_g_materialized_application_count_per_tree": 4,
        "topology_kind": "recursive",
        "direct_readout_singleton": False,
        "summarized_singleton": False,
        "composition_present": True,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
        "g_training_call_roles": ["leaf", "merge"],
        "g_training_role_evidence_source": "g_artifact_gradient_path_presence",
        "merge_domain_training_observed": True,
        "shared_g_updated_with_merge_domain": True,
    }
    assert grid._validated_executed_g_contract(realized, configured=configured) == realized

    wrong_mode = {**realized, "mode": "fixed"}
    with pytest.raises(ValueError, match="mode disagrees"):
        grid._validated_executed_g_contract(wrong_mode, configured=configured)

    wrong_count = {**realized, "g_update_count": 0, "learned_this_run": False}
    with pytest.raises(ValueError, match="configured schedule"):
        grid._validated_executed_g_contract(wrong_count, configured=configured)

    wrong_operator = {**realized, "operator": "fixed_identity"}
    with pytest.raises(ValueError, match="operator disagrees"):
        grid._validated_executed_g_contract(wrong_operator, configured=configured)

    wrong_status = {**realized, "fit_status": "not_updated_this_run"}
    with pytest.raises(ValueError, match="fit_status disagrees"):
        grid._validated_executed_g_contract(wrong_status, configured=configured)


def test_learned_dspy_realized_contract_uses_compiler_row_provenance() -> None:
    configured = grid._g_contract(
        "dspy",
        "ctree_recursive",
        requested_max_iterations=2,
    )
    realized = {
        "mode": "learned",
        "operator": "learned_shared",
        "fit_status": "fitted_this_run",
        "g_update_count": 1,
        "learned_this_run": True,
        "declared_leaf_count": 4,
        "merge_application_count_per_tree": 3,
        "leaf_g_materialized_application_count_per_tree": 4,
        "topology_kind": "recursive",
        "direct_readout_singleton": False,
        "summarized_singleton": False,
        "composition_present": True,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
        "g_training_call_roles": ["leaf", "merge"],
        "g_training_role_evidence_source": "dspy_compiler_mixed_examples",
        "merge_domain_training_observed": True,
        "shared_g_updated_with_merge_domain": True,
    }

    assert grid._validated_executed_g_contract(realized, configured=configured) == realized


def test_default_f_g_f_fno_ctree_reports_shared_g_merge_support(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    cell = grid.GridCell(
        target_width=1,
        representation_path="ctree_recursive",
        family="fno",
        seed=31,
        output_dir=tmp_path / "k01" / "ctree" / "fno" / "seed_31",
    )

    report = grid.fit_grid_cell(cell, fno_epochs=1)

    assert report["status"] == "success"
    assert report["g_contract"]["requested_max_iterations"] == 3
    assert report["g_contract"]["effective_max_iterations"] == 3
    executed = report["executed_g_contract"]
    assert executed is not None
    assert executed["mode"] == "learned"
    assert executed["operator"] == "learned_shared"
    assert executed["fit_status"] == "fitted_this_run"
    assert executed["g_update_count"] == 1
    assert executed["learned_this_run"] is True
    identity = report["comparison_row"]["identity"]
    assert identity["g_mode"] == "learned"
    assert identity["g_operator"] == "learned_shared"
    assert identity["g_fit_status"] == "fitted_this_run"
    assert identity["g_update_count"] == 1
    assert identity["learned_this_run"] is True
    assert identity["same_g_across_node_roles"] is True
    assert identity["reduce_g_is_derived"] is True
    assert identity["g_training_call_roles"] == ["leaf", "merge"]
    assert identity["g_training_role_evidence_source"] == ("g_artifact_gradient_path_presence")
    assert identity["merge_domain_training_observed"] is True
    assert identity["shared_g_updated_with_merge_domain"] is True


@pytest.mark.parametrize("target_width", [3, 57])
def test_common_metrics_use_sum_l1_for_wide_named_vectors(target_width: int) -> None:
    names = grid._target_names_for_width(target_width)
    target = {name: 0.0 for name in names}
    prediction = dict(target)
    prediction[names[0]] = 0.1
    prediction[names[1]] = 0.2
    prediction[names[2]] = 0.3
    metrics = grid.summarize_common_metrics(
        [
            {
                "prediction_by_target": prediction,
                "target_by_name": target,
            }
        ],
        target_width=target_width,
    )

    assert metrics["n"] == 1
    assert metrics["mean_joint_l1"] == pytest.approx(0.6)


def test_all_widths_share_the_same_authoritative_raw_rile_readout() -> None:
    train_raw, test_raw = grid.make_cmp_count_records("full_doc_direct")
    raw_records = [*train_raw, *test_raw]
    raw_rile_by_width = {}

    for target_width in (1, 3, 57):
        records, fragment = grid.prepare_target_records(
            raw_records,
            target_width,
        )
        target_key = fragment["backend_config"]["target_vector_key"]
        values = {}
        for record in records:
            target = dict(record.metadata[target_key])
            metrics = grid.summarize_common_metrics(
                [
                    {
                        "prediction_by_target": target,
                        "target_by_name": target,
                    }
                ],
                target_width=target_width,
            )
            values[record.tree_id] = metrics["mean_target_raw_rile"]
        raw_rile_by_width[target_width] = values

    assert raw_rile_by_width[3] == pytest.approx(raw_rile_by_width[1])
    assert raw_rile_by_width[57] == pytest.approx(raw_rile_by_width[1])
