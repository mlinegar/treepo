from __future__ import annotations

import copy

import pytest

from treepo.tasks.manifesto import (
    manifesto_semantic_forest_comparison_report_schema,
    validate_manifesto_semantic_forest_comparison_row,
)


def _row() -> dict:
    missing = {
        "compute_performance.inference_wall_seconds": "not separately timed",
        "compute_performance.docs_per_second": "requires inference timing",
        "compute_performance.peak_accelerator_memory_bytes": "CPU execution",
        "acquisition_resources.oracle_bundle_queries": "no query ledger",
        "acquisition_resources.model_node_rows": "no node-row ledger",
    }
    return {
        "schema_version": manifesto_semantic_forest_comparison_report_schema()["schema_version"],
        "cell_id": "dspy/full_doc/k01",
        "identity": {
            "cell_id": "dspy/full_doc/k01",
            "model_family": "dspy_llm",
            "representation_path": "full_doc_direct",
            "topology_kind": "singleton",
            "leaf_count": 1,
            "merge_application_count": 0,
            "readout_path": "f_direct",
            "leaf_g_application_count": 0,
            "composition_present": False,
            "same_g_across_node_roles": True,
            "reduce_g_is_derived": True,
            "g_training_call_roles": [],
            "g_training_role_evidence_source": "no_train_g_calls",
            "merge_domain_training_observed": False,
            "shared_g_updated_with_merge_domain": False,
            "width": 1,
            "seed": 7,
            "g_mode": "identity",
            "g_operator": "fixed_identity",
            "g_fit_status": "not_trainable",
            "g_update_count": 0,
            "learned_this_run": False,
            "schedule": "f",
            "requested_max_iterations": 2,
            "effective_max_iterations": 1,
            "leaf_scale": "leafs001",
            "eval_split": "test",
            "roster_digest": "roster",
            "config_digest": "config",
        },
        "quality": {
            "raw_published_rile_mae": 2.0,
            "mean_joint_sum_l1": 0.01,
            "per_target_metrics": {
                "rile_normalized": {
                    "n": 4,
                    "mae": 0.01,
                    "conditional_positive_mae": 0.01,
                    "gold_prevalence": 1.0,
                }
            },
            "exact_prediction_coverage": {
                "expected_n": 4,
                "observed_complete_vector_n": 4,
                "expected_tree_ids_digest": "expected",
                "observed_tree_ids_digest": "observed",
                "missing_tree_ids": [],
                "unexpected_tree_ids": [],
                "duplicate_tree_ids": [],
                "incomplete_row_tree_ids": [],
                "target_mismatch_tree_ids": [],
                "complete": True,
            },
        },
        "compute_performance": {
            "fit_wall_seconds": 1.0,
            "inference_wall_seconds": None,
            "docs_per_second": None,
            "peak_accelerator_memory_bytes": None,
        },
        "acquisition_resources": {
            "train_documents": 6,
            "evaluation_documents": 4,
            "oracle_bundle_queries": None,
            "underlying_atomic_annotations": 80,
            "returned_coordinate_values": 4,
            "materialized_node_coordinate_values": 10,
            "model_root_rows": 4,
            "model_node_rows": None,
        },
        "measurement_status": {
            "status": "partial",
            "missing_field_reasons": missing,
            "measurement_started_at": "2026-07-30T00:00:00+00:00",
            "measurement_finished_at": "2026-07-30T00:00:01+00:00",
        },
    }


def test_comparison_schema_is_copied_and_row_validates() -> None:
    first = manifesto_semantic_forest_comparison_report_schema()
    second = manifesto_semantic_forest_comparison_report_schema()
    first["identity"]["required_fields"].append("mutated")

    assert first != second
    assert (
        validate_manifesto_semantic_forest_comparison_row(
            _row(),
            target_names=("rile_normalized",),
        )
        == _row()
    )


def test_comparison_row_rejects_unexplained_null_and_wrong_target_order() -> None:
    unexplained = _row()
    del unexplained["measurement_status"]["missing_field_reasons"][
        "compute_performance.inference_wall_seconds"
    ]
    with pytest.raises(ValueError, match="missing-field reason"):
        validate_manifesto_semantic_forest_comparison_row(
            unexplained,
            target_names=("rile_normalized",),
        )

    wrong_order = _row()
    wrong_order["quality"]["per_target_metrics"] = {
        "other": copy.deepcopy(wrong_order["quality"]["per_target_metrics"]["rile_normalized"])
    }
    with pytest.raises(ValueError, match="exact ordered target catalog"):
        validate_manifesto_semantic_forest_comparison_row(
            wrong_order,
            target_names=("rile_normalized",),
        )


def test_realized_g_fields_are_all_observed_or_all_explained_nulls() -> None:
    absent = _row()
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
        absent["identity"][field] = None
        absent["measurement_status"]["missing_field_reasons"][f"identity.{field}"] = (
            "execution produced no FitResult.summary.g_contract"
        )

    assert (
        validate_manifesto_semantic_forest_comparison_row(
            absent,
            target_names=("rile_normalized",),
        )["identity"]["learned_this_run"]
        is None
    )

    unexplained = copy.deepcopy(absent)
    del unexplained["measurement_status"]["missing_field_reasons"]["identity.learned_this_run"]
    with pytest.raises(ValueError, match="realized g field"):
        validate_manifesto_semantic_forest_comparison_row(
            unexplained,
            target_names=("rile_normalized",),
        )

    partially_observed = copy.deepcopy(absent)
    partially_observed["identity"]["g_update_count"] = 0
    with pytest.raises(ValueError, match="either all observed or all null"):
        validate_manifesto_semantic_forest_comparison_row(
            partially_observed,
            target_names=("rile_normalized",),
        )


def test_realized_g_learning_flag_must_match_mode_and_update_count() -> None:
    inconsistent = _row()
    inconsistent["identity"]["g_update_count"] = 1
    inconsistent["identity"]["learned_this_run"] = True
    with pytest.raises(ValueError, match="identity/fixed g modes"):
        validate_manifesto_semantic_forest_comparison_row(
            inconsistent,
            target_names=("rile_normalized",),
        )

    inconsistent = _row()
    inconsistent["identity"]["learned_this_run"] = True
    with pytest.raises(ValueError, match="agree with identity.g_update_count"):
        validate_manifesto_semantic_forest_comparison_row(
            inconsistent,
            target_names=("rile_normalized",),
        )


def test_realized_g_operator_and_fit_status_must_match_mode_and_updates() -> None:
    wrong_operator = _row()
    wrong_operator["identity"]["g_operator"] = "learned_shared"
    with pytest.raises(ValueError, match="g_operator disagrees"):
        validate_manifesto_semantic_forest_comparison_row(
            wrong_operator,
            target_names=("rile_normalized",),
        )

    wrong_status = _row()
    wrong_status["identity"]["g_fit_status"] = "fitted_this_run"
    with pytest.raises(ValueError, match="g_fit_status disagrees"):
        validate_manifesto_semantic_forest_comparison_row(
            wrong_status,
            target_names=("rile_normalized",),
        )


def test_family_neutral_schema_allows_identity_on_base_summary() -> None:
    row = _row()
    row["identity"].update(
        {
            "representation_path": "ctree_base_summary",
            "topology_kind": "singleton",
            "leaf_count": 1,
            "merge_application_count": 0,
            "readout_path": "f_after_g",
            "leaf_g_application_count": 1,
            "composition_present": False,
        }
    )
    validate_manifesto_semantic_forest_comparison_row(
        row,
        target_names=("rile_normalized",),
    )


def test_family_neutral_schema_allows_learned_mode_on_full_doc_direct() -> None:
    row = _row()
    row["identity"].update(
        {
            "g_mode": "learned",
            "g_operator": "trainable_not_updated_this_run",
            "g_fit_status": "not_updated_this_run",
            "schedule": "fg",
        }
    )
    validate_manifesto_semantic_forest_comparison_row(
        row,
        target_names=("rile_normalized",),
    )


@pytest.mark.parametrize(
    ("target_names", "message"),
    [
        ((), "non-empty ordered catalog"),
        (("rile_normalized", "rile_normalized"), "unique"),
        (("rile_normalized", ""), "non-empty strings"),
        ("rile_normalized", "ordered catalog"),
    ],
)
def test_target_catalog_must_be_nonempty_unique_strings(
    target_names,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_manifesto_semantic_forest_comparison_row(
            _row(),
            target_names=target_names,
        )


@pytest.mark.parametrize("width", [0, -1, 1.0, True])
def test_identity_width_must_be_a_positive_integer(width) -> None:
    row = _row()
    row["identity"]["width"] = width
    with pytest.raises(ValueError, match="width must be a positive integer"):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


def test_identity_width_must_equal_target_catalog_size() -> None:
    row = _row()
    row["identity"]["width"] = 3
    with pytest.raises(ValueError, match=r"width must equal len\(target_names\)"):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


@pytest.mark.parametrize(
    "field",
    [
        "train_documents",
        "evaluation_documents",
        "oracle_bundle_queries",
        "underlying_atomic_annotations",
        "returned_coordinate_values",
        "materialized_node_coordinate_values",
        "model_root_rows",
        "model_node_rows",
    ],
)
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_acquisition_counts_must_be_nonnegative_integers(
    field: str,
    value,
) -> None:
    row = _row()
    row["acquisition_resources"][field] = value
    with pytest.raises(ValueError, match=rf"acquisition_resources\.{field}.*non-negative integer"):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


def test_explained_null_acquisition_count_remains_valid() -> None:
    row = _row()
    row["acquisition_resources"]["train_documents"] = None
    row["measurement_status"]["missing_field_reasons"]["acquisition_resources.train_documents"] = (
        "training roster was not observed"
    )

    assert (
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )["acquisition_resources"]["train_documents"]
        is None
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n", -1, "non-negative integer"),
        ("n", 1.5, "non-negative integer"),
        ("n", True, "non-negative integer"),
        ("mae", -0.1, "non-negative finite"),
        ("mae", float("inf"), "non-negative finite"),
        ("conditional_positive_mae", -0.1, "non-negative finite"),
        ("gold_prevalence", -0.01, r"in \[0, 1\]"),
        ("gold_prevalence", 1.01, r"in \[0, 1\]"),
        ("gold_prevalence", float("inf"), r"in \[0, 1\]"),
    ],
)
def test_per_target_metric_domains_are_enforced(
    field: str,
    value,
    message: str,
) -> None:
    row = _row()
    row["quality"]["per_target_metrics"]["rile_normalized"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


@pytest.mark.parametrize("field", ["raw_published_rile_mae", "mean_joint_sum_l1"])
def test_observed_top_level_errors_must_be_nonnegative(field: str) -> None:
    row = _row()
    row["quality"][field] = -0.01
    with pytest.raises(ValueError, match=rf"quality\.{field}.*non-negative finite"):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


def test_explained_null_per_target_metrics_remain_valid() -> None:
    row = _row()
    metrics = row["quality"]["per_target_metrics"]["rile_normalized"]
    for field in ("n", "mae", "conditional_positive_mae", "gold_prevalence"):
        metrics[field] = None
        row["measurement_status"]["missing_field_reasons"][
            f"quality.per_target_metrics.rile_normalized.{field}"
        ] = "target metric was not observed"

    assert (
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )["quality"]["per_target_metrics"]["rile_normalized"]["n"]
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_n", -1),
        ("expected_n", 1.5),
        ("observed_complete_vector_n", -1),
        ("observed_complete_vector_n", 1.5),
    ],
)
def test_coverage_counts_must_be_nonnegative_integers(field: str, value) -> None:
    row = _row()
    row["quality"]["exact_prediction_coverage"][field] = value
    with pytest.raises(ValueError, match=rf"{field}.*non-negative integer"):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


def test_coverage_observed_count_cannot_exceed_expected() -> None:
    row = _row()
    row["quality"]["exact_prediction_coverage"]["expected_n"] = 3
    with pytest.raises(ValueError, match="cannot exceed expected_n"):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


@pytest.mark.parametrize(
    "field",
    [
        "missing_tree_ids",
        "unexpected_tree_ids",
        "duplicate_tree_ids",
        "incomplete_row_tree_ids",
        "target_mismatch_tree_ids",
    ],
)
def test_complete_coverage_rejects_any_roster_issue(field: str) -> None:
    row = _row()
    row["quality"]["exact_prediction_coverage"][field] = ["doc"]
    with pytest.raises(ValueError, match="complete must agree"):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


def test_clean_equal_coverage_must_be_marked_complete() -> None:
    row = _row()
    row["quality"]["exact_prediction_coverage"]["complete"] = False
    with pytest.raises(ValueError, match="complete must agree"):
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )


def test_zero_expected_roster_is_not_complete() -> None:
    row = _row()
    coverage = row["quality"]["exact_prediction_coverage"]
    coverage["expected_n"] = 0
    coverage["observed_complete_vector_n"] = 0
    coverage["complete"] = False
    row["quality"]["raw_published_rile_mae"] = None
    row["quality"]["mean_joint_sum_l1"] = None
    row["measurement_status"]["missing_field_reasons"]["quality.raw_published_rile_mae"] = (
        "empty evaluation roster"
    )
    row["measurement_status"]["missing_field_reasons"]["quality.mean_joint_sum_l1"] = (
        "empty evaluation roster"
    )
    metrics = row["quality"]["per_target_metrics"]["rile_normalized"]
    metrics["n"] = 0
    for field in ("mae", "conditional_positive_mae", "gold_prevalence"):
        metrics[field] = None
        row["measurement_status"]["missing_field_reasons"][
            f"quality.per_target_metrics.rile_normalized.{field}"
        ] = "empty evaluation roster"

    assert (
        validate_manifesto_semantic_forest_comparison_row(
            row,
            target_names=("rile_normalized",),
        )["quality"]["exact_prediction_coverage"]["complete"]
        is False
    )


@pytest.mark.parametrize(
    ("g_mode", "operator", "fit_status", "update_count", "learned"),
    [
        ("fixed", "fixed_nonidentity_or_family_owned", "not_trainable", 0, False),
        (
            "learned",
            "trainable_not_updated_this_run",
            "not_updated_this_run",
            0,
            False,
        ),
        ("learned", "learned_shared", "fitted_this_run", 1, True),
    ],
)
def test_realized_g_operator_and_status_follow_mode_and_updates(
    g_mode: str,
    operator: str,
    fit_status: str,
    update_count: int,
    learned: bool,
) -> None:
    row = _row()
    row["identity"].update(
        {
            "representation_path": "ctree_base_summary",
            "topology_kind": "singleton",
            "leaf_count": 1,
            "merge_application_count": 0,
            "readout_path": "f_after_g",
            "leaf_g_application_count": 1,
            "composition_present": False,
            "g_mode": g_mode,
            "g_operator": operator,
            "g_fit_status": fit_status,
            "g_update_count": update_count,
            "learned_this_run": learned,
            "g_training_call_roles": ["leaf"] if learned else [],
            "g_training_role_evidence_source": (
                "inferred_from_train_g_and_topology" if learned else "no_train_g_calls"
            ),
            "merge_domain_training_observed": False,
            "shared_g_updated_with_merge_domain": False,
            "schedule": "fg" if g_mode == "learned" else "f",
        }
    )
    validate_manifesto_semantic_forest_comparison_row(
        row,
        target_names=("rile_normalized",),
    )

    wrong_operator = copy.deepcopy(row)
    wrong_operator["identity"]["g_operator"] = "fixed_identity"
    with pytest.raises(ValueError, match="g_operator disagrees"):
        validate_manifesto_semantic_forest_comparison_row(
            wrong_operator,
            target_names=("rile_normalized",),
        )

    wrong_status = copy.deepcopy(row)
    wrong_status["identity"]["g_fit_status"] = (
        "fitted_this_run" if fit_status != "fitted_this_run" else "not_trainable"
    )
    with pytest.raises(ValueError, match="g_fit_status disagrees"):
        validate_manifesto_semantic_forest_comparison_row(
            wrong_status,
            target_names=("rile_normalized",),
        )


def test_same_g_and_derived_reduce_are_required_semantics() -> None:
    for field in ("same_g_across_node_roles", "reduce_g_is_derived"):
        row = _row()
        row["identity"][field] = False
        with pytest.raises(ValueError, match=field):
            validate_manifesto_semantic_forest_comparison_row(
                row,
                target_names=("rile_normalized",),
            )


def test_recursive_topology_does_not_by_itself_claim_merge_domain_support() -> None:
    leaf_only = _row()
    leaf_only["identity"].update(
        {
            "representation_path": "ctree_recursive",
            "topology_kind": "recursive",
            "leaf_count": 4,
            "merge_application_count": 3,
            "readout_path": "f_after_reduce_g",
            "leaf_g_application_count": 4,
            "composition_present": True,
            "g_mode": "learned",
            "g_operator": "learned_shared",
            "g_fit_status": "fitted_this_run",
            "g_update_count": 1,
            "learned_this_run": True,
            "g_training_call_roles": ["leaf"],
            "g_training_role_evidence_source": ("g_artifact_gradient_path_presence"),
            "merge_domain_training_observed": False,
            "shared_g_updated_with_merge_domain": False,
            "schedule": "fg",
        }
    )
    validate_manifesto_semantic_forest_comparison_row(
        leaf_only,
        target_names=("rile_normalized",),
    )

    merge_supported = copy.deepcopy(leaf_only)
    merge_supported["identity"].update(
        {
            "g_training_call_roles": ["leaf", "merge"],
            "merge_domain_training_observed": True,
            "shared_g_updated_with_merge_domain": True,
        }
    )
    validate_manifesto_semantic_forest_comparison_row(
        merge_supported,
        target_names=("rile_normalized",),
    )

    inconsistent = copy.deepcopy(merge_supported)
    inconsistent["identity"]["shared_g_updated_with_merge_domain"] = False
    with pytest.raises(ValueError, match="shared_g_updated_with_merge_domain"):
        validate_manifesto_semantic_forest_comparison_row(
            inconsistent,
            target_names=("rile_normalized",),
        )


def test_dspy_recompression_support_is_not_binary_merge_domain_support() -> None:
    recompression_supported = _row()
    recompression_supported["identity"].update(
        {
            "representation_path": "ctree_recursive",
            "topology_kind": "recursive",
            "leaf_count": 4,
            "merge_application_count": 3,
            "readout_path": "f_after_reduce_g",
            "leaf_g_application_count": 4,
            "composition_present": True,
            "g_mode": "learned",
            "g_operator": "learned_shared",
            "g_fit_status": "fitted_this_run",
            "g_update_count": 1,
            "learned_this_run": True,
            "g_training_call_roles": ["leaf", "recompression"],
            "g_training_role_evidence_source": "dspy_compiler_mixed_examples",
            "merge_domain_training_observed": False,
            "shared_g_updated_with_merge_domain": False,
            "schedule": "fg",
        }
    )
    validate_manifesto_semantic_forest_comparison_row(
        recompression_supported,
        target_names=("rile_normalized",),
    )

    mislabeled_as_merge = copy.deepcopy(recompression_supported)
    mislabeled_as_merge["identity"]["merge_domain_training_observed"] = True
    with pytest.raises(ValueError, match="merge_domain_training_observed"):
        validate_manifesto_semantic_forest_comparison_row(
            mislabeled_as_merge,
            target_names=("rile_normalized",),
        )
