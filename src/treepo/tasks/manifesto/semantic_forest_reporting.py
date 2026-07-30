"""Shared comparison rows for Manifesto/RILE Semantic-Forest grids.

The schema is deliberately family-neutral.  DSPy/LLM and FNO executions
write the same common row and keep substrate-specific telemetry in sidecars
joined by ``cell_id``.  Missing observations remain ``None`` and must be
explained; callers may not zero-fill an unmeasured quantity.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

RILE_SEMANTIC_FOREST_COMPARISON_SCHEMA_VERSION = (
    "treepo.manifesto.semantic_forest.comparison_report.v5"
)

_COMPARISON_REPORT_SCHEMA: dict[str, Any] = {
    "schema_version": RILE_SEMANTIC_FOREST_COMPARISON_SCHEMA_VERSION,
    "row_kind": "one_observed_method_cell",
    "join_key": "cell_id",
    "common_row_is_family_neutral": True,
    "required_sections": [
        "identity",
        "quality",
        "compute_performance",
        "acquisition_resources",
        "measurement_status",
    ],
    "identity": {
        "required_fields": [
            "cell_id",
            "model_family",
            "representation_path",
            "topology_kind",
            "leaf_count",
            "merge_application_count",
            "readout_path",
            "leaf_g_application_count",
            "composition_present",
            "same_g_across_node_roles",
            "reduce_g_is_derived",
            "g_training_call_roles",
            "g_training_role_evidence_source",
            "merge_domain_training_observed",
            "shared_g_updated_with_merge_domain",
            "width",
            "seed",
            "g_mode",
            "g_operator",
            "g_fit_status",
            "g_update_count",
            "learned_this_run",
            "schedule",
            "requested_max_iterations",
            "effective_max_iterations",
            "leaf_scale",
            "eval_split",
            "roster_digest",
            "config_digest",
        ],
        "g_field_semantics": {
            "g_mode": "configured identity, fixed, or learned state-operator mode",
            "g_operator": "realized operator reported by FitResult.summary.g_contract",
            "g_fit_status": "realized fit status reported by FitResult.summary.g_contract",
            "g_update_count": "realized g updates reported by FitResult.summary.g_contract",
            "learned_this_run": ("whether this execution performed at least one realized g update"),
            "same_g_across_node_roles": (
                "required semantic contract: leaf, unary recompression, and binary merge "
                "roles call one shared g"
            ),
            "reduce_g_is_derived": (
                "required semantic contract: reduce_g is a fold of g, not a second learner"
            ),
            "g_training_call_roles": (
                "ordered aggregate call domains supported by realized train_g calls"
            ),
            "g_training_role_evidence_source": (
                "artifact gradient-path evidence when available; otherwise explicitly labeled "
                "inference from train_g calls and training topology"
            ),
            "merge_domain_training_observed": (
                "binary merge-domain training support, not unary recompression or C3 "
                "certification"
            ),
            "shared_g_updated_with_merge_domain": (
                "one shared-g update occurred while merge-domain support was present"
            ),
            "realized_fields_nullable_when": (
                "the execution produced no FitResult.summary.g_contract; every null "
                "then requires a measurement_status.missing_field_reasons entry"
            ),
        },
    },
    "quality": {
        "required_fields": {
            "raw_published_rile_mae": {
                "type": "number_or_null",
                "unit": "raw_RILE_points",
                "aggregation": "document_mean_absolute_error",
            },
            "mean_joint_sum_l1": {
                "type": "number_or_null",
                "unit": "target_vector_mass",
                "aggregation": "document_mean_of_sum_absolute_coordinate_error",
            },
            "per_target_metrics": {
                "type": "mapping",
                "keys": "exact_ordered_target_catalog",
                "required_per_target_fields": [
                    "n",
                    "mae",
                    "conditional_positive_mae",
                    "gold_prevalence",
                ],
            },
            "exact_prediction_coverage": {
                "type": "mapping",
                "required_fields": [
                    "expected_n",
                    "observed_complete_vector_n",
                    "expected_tree_ids_digest",
                    "observed_tree_ids_digest",
                    "missing_tree_ids",
                    "unexpected_tree_ids",
                    "duplicate_tree_ids",
                    "incomplete_row_tree_ids",
                    "target_mismatch_tree_ids",
                    "complete",
                ],
                "complete_criterion": (
                    "exactly one finite complete named prediction plus the "
                    "authoritative target vector for every expected evaluation "
                    "tree_id, and no unexpected tree_id"
                ),
            },
        }
    },
    "compute_performance": {
        "required_fields": {
            "fit_wall_seconds": {
                "type": "number_or_null",
                "unit": "seconds",
                "clock": "monotonic_wall_clock",
            },
            "inference_wall_seconds": {
                "type": "number_or_null",
                "unit": "seconds",
                "clock": "monotonic_wall_clock",
            },
            "docs_per_second": {
                "type": "number_or_null",
                "unit": "complete_documents_per_second",
                "formula": "observed_complete_vector_n / inference_wall_seconds",
            },
            "peak_accelerator_memory_bytes": {
                "type": "integer_or_null",
                "unit": "bytes",
                "nullable_when": (
                    "no accelerator is used or the runtime cannot observe a process-scoped peak"
                ),
            },
        }
    },
    "acquisition_resources": {
        "required_fields": {
            "train_documents": {"type": "integer_or_null", "unit": "documents"},
            "evaluation_documents": {
                "type": "integer_or_null",
                "unit": "documents",
            },
            "oracle_bundle_queries": {
                "type": "integer_or_null",
                "unit": "bundle_queries",
            },
            "underlying_atomic_annotations": {
                "type": "integer_or_null",
                "unit": "CMP_quasi_sentence_annotations",
            },
            "returned_coordinate_values": {
                "type": "integer_or_null",
                "unit": "coordinate_values",
            },
            "materialized_node_coordinate_values": {
                "type": "integer_or_null",
                "unit": "coordinate_values",
            },
            "model_root_rows": {"type": "integer_or_null", "unit": "rows"},
            "model_node_rows": {"type": "integer_or_null", "unit": "rows"},
        },
        "annotation_accounting_note": (
            "K coordinates derived from one CMP annotation bundle are not K "
            "independent human labels"
        ),
    },
    "measurement_status": {
        "required_fields": [
            "status",
            "missing_field_reasons",
            "measurement_started_at",
            "measurement_finished_at",
        ],
        "allowed_status": ["observed", "partial", "failed", "not_run"],
        "no_fabrication_rule": (
            "An unobserved numeric value is null and named in "
            "missing_field_reasons; it is never zero-filled, imputed, or copied "
            "from another family."
        ),
    },
    "family_telemetry_sidecars": {
        "common_row_excludes_family_specific_fields": True,
        "join_key": "cell_id",
        "dspy_llm": {
            "fields": [
                "lm_calls",
                "input_tokens",
                "output_tokens",
                "parse_retries",
                "transport_retries",
                "estimated_cost",
                "cost_currency",
            ]
        },
        "fno": {
            "fields": [
                "parameter_count",
                "trainable_parameter_count",
                "epochs_completed",
                "optimizer_steps",
                "gradient_updates",
            ]
        },
    },
}


def manifesto_semantic_forest_comparison_report_schema() -> dict[str, Any]:
    """Return an independent copy of the shared DSPy/FNO row schema."""

    return copy.deepcopy(_COMPARISON_REPORT_SCHEMA)


def validate_manifesto_semantic_forest_comparison_row(
    row: Mapping[str, Any],
    *,
    target_names: Sequence[str],
) -> dict[str, Any]:
    """Validate and return one JSON-ready family-neutral comparison row."""

    payload = copy.deepcopy(dict(row))
    schema = _COMPARISON_REPORT_SCHEMA
    if payload.get("schema_version") != schema["schema_version"]:
        raise ValueError("comparison row has the wrong schema_version")
    if not str(payload.get("cell_id") or ""):
        raise ValueError("comparison row requires the top-level cell_id join key")
    _require_fields(payload, schema["required_sections"], where="comparison row")
    for section in schema["required_sections"]:
        if not isinstance(payload[section], Mapping):
            raise ValueError(f"comparison row {section!r} must be a mapping")

    _require_fields(
        payload["identity"],
        schema["identity"]["required_fields"],
        where="identity",
    )
    if payload["identity"]["cell_id"] != payload["cell_id"]:
        raise ValueError("top-level and identity cell_id values must agree")
    identity = payload["identity"]
    if isinstance(target_names, (str, bytes)):
        raise ValueError("target_names must be a non-empty ordered catalog of unique names")
    names = tuple(target_names)
    if not names:
        raise ValueError("target_names must be a non-empty ordered catalog")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("target_names must contain only non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("target_names must be unique")
    width = identity["width"]
    if not _positive_integer(width):
        raise ValueError("identity.width must be a positive integer")
    if width != len(names):
        raise ValueError("identity.width must equal len(target_names)")
    g_mode = str(identity["g_mode"])
    if g_mode not in {"identity", "fixed", "learned"}:
        raise ValueError(f"identity.g_mode is invalid: {g_mode!r}")
    expected_schedule = "fg" if g_mode == "learned" else "f"
    if str(identity["schedule"]) != expected_schedule:
        raise ValueError(f"identity.schedule must be {expected_schedule!r} for g_mode={g_mode!r}")
    if identity["same_g_across_node_roles"] is not True:
        raise ValueError("identity.same_g_across_node_roles must be true")
    if identity["reduce_g_is_derived"] is not True:
        raise ValueError("identity.reduce_g_is_derived must be true")
    realized_fields = (
        "g_operator",
        "g_fit_status",
        "g_update_count",
        "learned_this_run",
        "g_training_call_roles",
        "g_training_role_evidence_source",
        "merge_domain_training_observed",
        "shared_g_updated_with_merge_domain",
    )
    realized_values = tuple(identity[field] for field in realized_fields)
    if any(value is None for value in realized_values) and not all(
        value is None for value in realized_values
    ):
        raise ValueError("identity realized g fields must be either all observed or all null")
    for field in ("requested_max_iterations", "effective_max_iterations"):
        value = identity[field]
        if not _finite_number(value) or int(value) != float(value) or int(value) < 0:
            raise ValueError(f"identity.{field} must be a non-negative integer")
    _validate_representation_path_identity(identity)
    quality_fields = schema["quality"]["required_fields"]
    _require_fields(payload["quality"], quality_fields, where="quality")
    coverage = payload["quality"]["exact_prediction_coverage"]
    if not isinstance(coverage, Mapping):
        raise ValueError("quality.exact_prediction_coverage must be a mapping")
    _require_fields(
        coverage,
        quality_fields["exact_prediction_coverage"]["required_fields"],
        where="quality.exact_prediction_coverage",
    )
    expected_n = coverage["expected_n"]
    observed_n = coverage["observed_complete_vector_n"]
    if not _nonnegative_integer(expected_n):
        raise ValueError(
            "quality.exact_prediction_coverage.expected_n must be a non-negative integer"
        )
    if not _nonnegative_integer(observed_n):
        raise ValueError(
            "quality.exact_prediction_coverage.observed_complete_vector_n "
            "must be a non-negative integer"
        )
    if observed_n > expected_n:
        raise ValueError(
            "quality.exact_prediction_coverage.observed_complete_vector_n cannot exceed expected_n"
        )
    roster_issue_fields = (
        "missing_tree_ids",
        "unexpected_tree_ids",
        "duplicate_tree_ids",
        "incomplete_row_tree_ids",
        "target_mismatch_tree_ids",
    )
    for field in roster_issue_fields:
        if not isinstance(coverage[field], list):
            raise ValueError(f"quality.exact_prediction_coverage.{field} must be a list")
    complete = coverage["complete"]
    if not isinstance(complete, bool):
        raise ValueError("quality.exact_prediction_coverage.complete must be boolean")
    should_be_complete = (
        expected_n > 0
        and observed_n == expected_n
        and not any(coverage[field] for field in roster_issue_fields)
    )
    if complete != should_be_complete:
        raise ValueError(
            "quality.exact_prediction_coverage.complete must agree with counts "
            "and roster mismatch/incomplete lists"
        )

    per_target = payload["quality"]["per_target_metrics"]
    if not isinstance(per_target, Mapping) or tuple(per_target) != names:
        raise ValueError("quality.per_target_metrics must use the exact ordered target catalog")
    required_per_target = quality_fields["per_target_metrics"]["required_per_target_fields"]
    for name in names:
        metrics = per_target[name]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"per-target metrics for {name!r} must be a mapping")
        _require_fields(metrics, required_per_target, where=f"per_target[{name!r}]")
        n = metrics["n"]
        if n is not None and not _nonnegative_integer(n):
            raise ValueError(f"quality.per_target_metrics.{name}.n must be a non-negative integer")
        for field in ("mae", "conditional_positive_mae"):
            value = metrics[field]
            if value is not None and (not _finite_number(value) or float(value) < 0.0):
                raise ValueError(
                    f"quality.per_target_metrics.{name}.{field} "
                    "must be a non-negative finite number when observed"
                )
        prevalence = metrics["gold_prevalence"]
        if prevalence is not None and (
            not _finite_number(prevalence) or not 0.0 <= float(prevalence) <= 1.0
        ):
            raise ValueError(
                f"quality.per_target_metrics.{name}.gold_prevalence must be in [0, 1] when observed"
            )

    for field in ("raw_published_rile_mae", "mean_joint_sum_l1"):
        value = payload["quality"][field]
        if value is not None and (not _finite_number(value) or float(value) < 0.0):
            raise ValueError(f"quality.{field} must be a non-negative finite number when observed")

    for section in ("compute_performance", "acquisition_resources"):
        _require_fields(
            payload[section],
            schema[section]["required_fields"],
            where=section,
        )
    for field in schema["acquisition_resources"]["required_fields"]:
        value = payload["acquisition_resources"][field]
        if value is not None and not _nonnegative_integer(value):
            raise ValueError(
                f"acquisition_resources.{field} must be a non-negative integer when observed"
            )
    status = payload["measurement_status"]
    _require_fields(
        status,
        schema["measurement_status"]["required_fields"],
        where="measurement_status",
    )
    if status["status"] not in schema["measurement_status"]["allowed_status"]:
        raise ValueError(f"invalid comparison measurement status {status['status']!r}")
    missing = status["missing_field_reasons"]
    if not isinstance(missing, Mapping):
        raise ValueError("measurement_status.missing_field_reasons must be a mapping")
    if all(value is None for value in realized_values):
        for field in realized_fields:
            path = f"identity.{field}"
            if path not in missing:
                raise ValueError(
                    f"unobserved realized g field {path!r} needs a missing-field reason"
                )
    else:
        if not isinstance(identity["g_operator"], str) or not identity["g_operator"].strip():
            raise ValueError("identity.g_operator must be a non-empty string when observed")
        if not isinstance(identity["g_fit_status"], str) or not identity["g_fit_status"].strip():
            raise ValueError("identity.g_fit_status must be a non-empty string when observed")
        g_update_count = identity["g_update_count"]
        if (
            not _finite_number(g_update_count)
            or int(g_update_count) != float(g_update_count)
            or int(g_update_count) < 0
        ):
            raise ValueError("identity.g_update_count must be a non-negative integer")
        learned_this_run = identity["learned_this_run"]
        if not isinstance(learned_this_run, bool):
            raise ValueError("identity.learned_this_run must be boolean when observed")
        if learned_this_run != (int(g_update_count) > 0):
            raise ValueError("identity.learned_this_run must agree with identity.g_update_count")
        if g_mode != "learned" and (int(g_update_count) != 0 or learned_this_run):
            raise ValueError("identity/fixed g modes cannot report realized learning updates")
        if g_mode == "identity":
            expected_operator, expected_fit_status = "fixed_identity", "not_trainable"
        elif g_mode == "fixed":
            expected_operator = "fixed_nonidentity_or_family_owned"
            expected_fit_status = "not_trainable"
        elif int(g_update_count) > 0:
            expected_operator, expected_fit_status = "learned_shared", "fitted_this_run"
        else:
            expected_operator = "trainable_not_updated_this_run"
            expected_fit_status = "not_updated_this_run"
        if identity["g_operator"] != expected_operator:
            raise ValueError(
                "identity.g_operator disagrees with configured mode and realized updates"
            )
        if identity["g_fit_status"] != expected_fit_status:
            raise ValueError(
                "identity.g_fit_status disagrees with configured mode and realized updates"
            )
        _validate_shared_g_training_support(identity, g_mode=g_mode)

    for path, value in _required_nullable_values(payload):
        if value is None and path not in missing:
            raise ValueError(f"unobserved field {path!r} needs a missing-field reason")
        if value is not None and not _finite_number(value):
            raise ValueError(f"observed field {path!r} must be finite numeric")
    return payload


def _validate_shared_g_training_support(
    identity: Mapping[str, Any],
    *,
    g_mode: str,
) -> None:
    roles = identity["g_training_call_roles"]
    if not isinstance(roles, list):
        raise ValueError("identity.g_training_call_roles must be a list when observed")
    allowed_roles = ("leaf", "recompression", "merge")
    if any(role not in allowed_roles for role in roles):
        raise ValueError("identity.g_training_call_roles contains an invalid role")
    if roles != [role for role in allowed_roles if role in roles]:
        raise ValueError(
            "identity.g_training_call_roles must be unique and ordered as "
            "leaf, recompression, merge"
        )

    source = identity["g_training_role_evidence_source"]
    allowed_sources = {
        "no_train_g_calls",
        "dspy_compiler_mixed_examples",
        "g_artifact_gradient_path_presence",
        "inferred_from_train_g_and_topology",
    }
    if source not in allowed_sources:
        raise ValueError("identity.g_training_role_evidence_source is invalid")
    merge_observed = identity["merge_domain_training_observed"]
    shared_updated = identity["shared_g_updated_with_merge_domain"]
    if not isinstance(merge_observed, bool):
        raise ValueError("identity.merge_domain_training_observed must be boolean")
    if not isinstance(shared_updated, bool):
        raise ValueError("identity.shared_g_updated_with_merge_domain must be boolean")
    if merge_observed != ("merge" in roles):
        raise ValueError("merge_domain_training_observed must agree with g_training_call_roles")
    if source == "no_train_g_calls" and (roles or merge_observed):
        raise ValueError("no_train_g_calls cannot report g training call roles")
    if "merge" in roles and not identity["composition_present"]:
        raise ValueError("merge-domain training support requires compositional topology")
    if g_mode != "learned" and roles:
        raise ValueError("identity/fixed g modes cannot report train_g call roles")
    if identity["learned_this_run"] and not roles:
        raise ValueError("a realized shared-g update requires a reported train_g call role")
    expected_shared_update = bool(identity["learned_this_run"] and merge_observed)
    if shared_updated != expected_shared_update:
        raise ValueError(
            "shared_g_updated_with_merge_domain must agree with the shared-g update "
            "and merge-domain support evidence"
        )


def _validate_representation_path_identity(identity: Mapping[str, Any]) -> None:
    contracts = {
        "full_doc_direct": ("singleton", 1, 0, "f_direct", 0, False),
        "ctree_base_summary": ("singleton", 1, 0, "f_after_g", 1, False),
        "ctree_recursive": ("recursive", 4, 3, "f_after_reduce_g", 4, True),
    }
    representation_path = str(identity["representation_path"])
    try:
        expected = contracts[representation_path]
    except KeyError as exc:
        raise ValueError(
            f"identity.representation_path is invalid: {representation_path!r}"
        ) from exc

    fields = (
        "topology_kind",
        "leaf_count",
        "merge_application_count",
        "readout_path",
        "leaf_g_application_count",
        "composition_present",
    )
    for field, expected_value in zip(fields, expected, strict=True):
        value = identity[field]
        if field in {
            "leaf_count",
            "merge_application_count",
            "leaf_g_application_count",
        } and not _nonnegative_integer(value):
            raise ValueError(f"identity.{field} must be a non-negative integer")
        if field == "composition_present" and not isinstance(value, bool):
            raise ValueError("identity.composition_present must be boolean")
        if value != expected_value:
            raise ValueError(
                f"identity.{field} disagrees with representation_path={representation_path!r}"
            )


def _required_nullable_values(row: Mapping[str, Any]):
    for section in ("compute_performance", "acquisition_resources"):
        for field in _COMPARISON_REPORT_SCHEMA[section]["required_fields"]:
            yield f"{section}.{field}", row[section][field]
    quality = row["quality"]
    for field in ("raw_published_rile_mae", "mean_joint_sum_l1"):
        yield f"quality.{field}", quality[field]
    for name, metrics in quality["per_target_metrics"].items():
        for field in ("n", "mae", "conditional_positive_mae", "gold_prevalence"):
            yield f"quality.per_target_metrics.{name}.{field}", metrics[field]


def _require_fields(value: Mapping[str, Any], fields, *, where: str) -> None:
    missing = [str(field) for field in fields if field not in value]
    if missing:
        raise ValueError(f"{where} is missing required fields {missing!r}")


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _nonnegative_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _positive_integer(value: Any) -> bool:
    return _nonnegative_integer(value) and value > 0


__all__ = [
    "RILE_SEMANTIC_FOREST_COMPARISON_SCHEMA_VERSION",
    "manifesto_semantic_forest_comparison_report_schema",
    "validate_manifesto_semantic_forest_comparison_row",
]
