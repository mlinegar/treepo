"""FitResult assembly and metric payload helpers."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.evidence import build_evidence
from treepo.forest import l1_oracle_metric_schema
from treepo.methods._results import write_results_json
from treepo.methods._run_manifest import joint_target_schema, json_default, write_manifest
from treepo.methods._topology_contract import resolve_topology_contract
from treepo.methods.artifact_alignment import build_model_artifact_contract
from treepo.methods.contracts import (
    G_MODE_FIXED,
    G_MODE_IDENTITY,
    G_MODE_LEARNED,
    FitResult,
    normalize_g_mode,
)
from treepo.methods.preference import PreferenceDataset, export_preference_records

_G_TRAINING_ROLES = ("leaf", "recompression", "merge")
_DSPY_G_TRAINING_FIELDS = (
    "g_training_call_roles",
    "leaf_domain_training_row_count",
    "merge_domain_training_row_count",
    "g_training_role_evidence_source",
)
_DSPY_RECOMPRESSION_TRAINING_FIELD = "recompression_training_row_count"
_GRADIENT_G_TRAINING_FIELDS = (
    "leaf_domain_gradient_path_present",
    "merge_domain_gradient_path_present",
)


def build_result(
    *,
    spec: Any,
    records: Sequence[Any],
    output_dir: Path,
    objective: Any | None,
    preference_dataset: PreferenceDataset,
    grid_axes: Mapping[str, Any] | None = None,
    supervision: Mapping[str, Any] | None = None,
    wall_seconds: float | None = None,
) -> Any:
    last = records[-1] if records else None
    error = (last.extra or {}).get("error") if last is not None else None
    status = "failed" if error else "success"

    metrics = final_metrics(last)
    artifacts: dict[str, Any] = (
        {"f": last.f_artifact, "g": last.g_artifact} if last is not None else {}
    )
    model_artifact_contract = build_model_artifact_contract(
        spec=spec,
        artifacts=artifacts,
        output_dir=output_dir,
    )
    artifacts["model_artifact_contract"] = model_artifact_contract
    write_prediction_records(output_dir, records)
    prediction_records = collect_prediction_records(output_dir)
    if prediction_records:
        artifacts["prediction_records"] = prediction_records
    statistic_artifact = (last.extra or {}).get("statistic") if last is not None else None
    if statistic_artifact:
        artifacts["statistic"] = statistic_artifact
    preference_artifacts = (
        export_preference_records(preference_dataset, output_dir / "preference")
        if len(preference_dataset) > 0
        else {}
    )
    if preference_artifacts:
        artifacts["preference_data"] = preference_artifacts
    history = [dataclasses.asdict(r) for r in records]

    summary: dict[str, Any] = {
        "family": str(spec.family or ""),
        "schedule": str(spec.schedule),
        "n_iterations": len(records),
        "output_dir": str(output_dir),
    }
    g_contract = g_contract_payload(spec, records)
    summary["g_mode"] = g_contract["mode"]
    summary["g_contract"] = g_contract
    summary["configured_max_iterations"] = g_contract["configured_max_iterations"]
    summary["f_update_count"] = g_contract["f_update_count"]
    summary["model_artifact_contract"] = model_artifact_contract
    summary["g_update_count"] = g_contract["g_update_count"]
    summary["oracle_metric"] = l1_oracle_metric_schema()
    joint_schema = joint_target_schema(spec)
    if joint_schema is not None:
        summary.update(joint_schema)
    if grid_axes:
        summary["grid_axes"] = dict(grid_axes)
        artifacts["grid_axes"] = dict(grid_axes)
    if supervision:
        summary["supervision"] = dict(supervision)
    split_metrics = split_metrics_payload(last)
    if split_metrics:
        summary["split_metrics"] = split_metrics
    if last is not None:
        summary["final_stage"] = last.stage_name
        summary["final_stage_label"] = last.stage_label
    if objective is not None:
        summary["objective"] = (
            objective.to_dict() if hasattr(objective, "to_dict") else dataclasses.asdict(objective)
        )
    if preference_artifacts:
        summary["preference_data"] = preference_artifacts["summary"]
    if statistic_artifact:
        summary["statistic"] = dict(statistic_artifact.get("info") or {})
        if statistic_artifact.get("local_law_summary"):
            summary["statistic_local_law"] = dict(statistic_artifact["local_law_summary"])
    artifacts["evidence"] = build_evidence(
        status=status,
        metrics=metrics,
        summary=summary,
        artifacts=artifacts,
    )

    manifest_path = write_manifest(
        spec=spec,
        records=records,
        output_dir=output_dir,
        objective=objective,
        status=status,
        metrics=metrics,
        summary=summary,
        preference_artifacts=preference_artifacts,
    )
    results_path = write_results_json(
        spec=spec,
        records=records,
        output_dir=output_dir,
        status=status,
        summary=summary,
        artifacts=artifacts,
        wall_seconds=wall_seconds,
    )
    if results_path is not None:
        summary["results_path"] = str(results_path)
        artifacts["results_json"] = str(results_path)
    return FitResult(
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        history=history,
        summary=summary,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
    )


def _shared_g_evidence(mode: str, artifact: Any) -> tuple[bool, bool, str]:
    """Return only executable/artifact-backed evidence for the one-g claim."""

    if mode == G_MODE_IDENTITY:
        return True, True, "package_canonical_identity"
    if not isinstance(artifact, Mapping):
        return False, False, "missing_g_artifact_contract"
    nested = artifact.get("shared_g_contract")
    nested = dict(nested) if isinstance(nested, Mapping) else {}
    same_g = bool(
        artifact.get("same_g_across_node_roles") is True
        or (
            nested.get("single_registered_g") is True
            and nested.get("parameter_identity_shared_across_call_domains") is True
        )
    )
    derived = bool(artifact.get("reduce_g_is_derived") is True or same_g)
    if artifact.get("same_g_across_node_roles") is True:
        source = "g_artifact_declaration"
    elif same_g:
        source = "registered_module_parameter_identity"
    else:
        source = "missing_g_artifact_contract"
    return same_g, derived, source


def _latest_g_artifact(records: Sequence[Any], initial: Any) -> Any:
    for record in reversed(tuple(records)):
        artifact = getattr(record, "g_artifact", None)
        if artifact is not None:
            return artifact
    return initial


def g_contract_payload(spec: Any, records: Sequence[Any]) -> dict[str, Any]:
    """Describe the shared state operator and the topology it executed over."""

    mode = normalize_g_mode(getattr(spec, "g_mode", G_MODE_LEARNED))
    f_updates = sum(1 for record in records if getattr(record, "trained", None) == "f")
    g_call_records = [record for record in records if getattr(record, "trained", None) == "g"]
    g_calls = len(g_call_records)
    g_updates = sum(1 for record in g_call_records if _record_g_update_performed(record))
    axis = dict(getattr(spec, "axis", None) or {})
    topology = _topology_payload(spec, records)
    declared_representation = topology.get("representation")
    singleton = bool(topology.get("singleton"))
    composition_present = bool(topology.get("composition_present"))
    training_composition_present = bool(topology.get("training_composition_present"))
    leaf_applications = topology.get("leaf_g_application_count_per_tree")
    merge_applications = topology.get("merge_application_count_per_tree")
    initial_artifacts = dict(getattr(spec, "initial_artifacts", None) or {})
    initial_g_artifact_present = initial_artifacts.get("g") is not None
    executed_g_artifact = _latest_g_artifact(records, initial_artifacts.get("g"))
    (
        same_g_across_node_roles,
        reduce_g_is_derived,
        shared_g_evidence_source,
    ) = _shared_g_evidence(mode, executed_g_artifact)
    configured_max_iterations = int(axis.get("max_iterations", 2))
    if mode == G_MODE_IDENTITY:
        operator = "fixed_identity"
        fit_status = "not_trainable"
        skipped_g = "identity"
        merge_calls = 0
        materialization = "package_canonical_identity"
    elif mode == G_MODE_FIXED:
        operator = "fixed_nonidentity_or_family_owned"
        fit_status = "not_trainable"
        skipped_g = "explicit_fixed_operator"
        merge_calls = 0 if singleton else None
        materialization = "family_owned_required"
    elif g_updates and same_g_across_node_roles:
        operator = "learned_shared"
        fit_status = "fitted_this_run"
        skipped_g = None
        merge_calls = 0 if singleton else None
        materialization = "family_owned_required"
    elif g_updates:
        operator = "learned_unverified_operator"
        fit_status = "updated_without_shared_g_evidence"
        skipped_g = None
        merge_calls = 0 if singleton else None
        materialization = "family_owned_unverified"
    elif initial_g_artifact_present and same_g_across_node_roles:
        operator = "learned_shared_reused"
        fit_status = "reused_without_update"
        skipped_g = "reused_initial_artifact"
        merge_calls = 0 if singleton else None
        materialization = "family_owned_required"
    else:
        operator = "trainable_not_updated_this_run"
        fit_status = "not_updated_this_run"
        skipped_g = "trainable_not_updated_this_run"
        merge_calls = 0 if singleton else None
        materialization = "family_owned_required"
    direct = singleton and mode == G_MODE_IDENTITY
    summarized = mode != G_MODE_IDENTITY
    summarized_singleton = singleton and summarized
    (
        g_training_call_roles,
        merge_domain_training_observed,
        g_training_role_evidence_source,
    ) = _g_training_support(g_call_records, topology=topology)
    return {
        "universal_execution": "f(reduce_g(T))",
        "identity_equation": "g(x)=x" if mode == G_MODE_IDENTITY else None,
        "mode": mode,
        "operator": operator,
        "initial_g_artifact_present": initial_g_artifact_present,
        "fit_status": fit_status,
        "learned_this_run": bool(g_updates),
        "trainable": mode == G_MODE_LEARNED,
        "train_g_enabled": mode == G_MODE_LEARNED,
        "configured_max_iterations": configured_max_iterations,
        "executed_record_count": len(records),
        "executed_iteration_indices": [int(record.iteration) for record in records],
        "f_update_count": f_updates,
        "g_update_count": g_updates,
        "train_g_call_count": g_calls,
        # Compatibility field: actual family merge calls are not centrally
        # instrumented. Singleton/identity paths are known to materialize none.
        "merge_call_count": merge_calls,
        # There is one g. ``reduce_g`` is only the fold that repeatedly calls
        # it; neither a leaf-g learner nor a separate reduced-g learner exists.
        "same_g_across_node_roles": same_g_across_node_roles,
        "reduce_g_is_derived": reduce_g_is_derived,
        "shared_g_evidence_source": shared_g_evidence_source,
        "g_training_call_roles": g_training_call_roles,
        "g_training_role_evidence_source": g_training_role_evidence_source,
        "merge_domain_training_observed": merge_domain_training_observed,
        "shared_g_updated_with_merge_domain": bool(
            g_updates and same_g_across_node_roles and merge_domain_training_observed
        ),
        "skipped_g_interpretation": skipped_g,
        "declared_representation": declared_representation,
        "topology_kind": topology.get("topology_kind"),
        "declared_leaf_count": topology.get("declared_leaf_count"),
        "leaf_g_application_count_per_tree": leaf_applications,
        "merge_application_count_per_tree": merge_applications,
        "leaf_g_application_materialization": materialization,
        "leaf_g_materialized_application_count_per_tree": leaf_applications,
        "singleton": singleton,
        "direct": direct,
        "summarized": summarized,
        "direct_readout_singleton": direct,
        "summarized_singleton": summarized_singleton,
        "composition_present": composition_present,
        "identity_single_leaf_contract": direct,
        "canonical_full_document": direct,
        "training_composition_present": training_composition_present,
        "all_singleton_training": bool(topology.get("all_singleton_training")),
        "train_tree_count": int(topology.get("train_tree_count") or 0),
        "eval_tree_count": int(topology.get("eval_tree_count") or 0),
        "observed_leaf_count_min": topology.get("observed_leaf_count_min"),
        "observed_leaf_count_max": topology.get("observed_leaf_count_max"),
    }


def _g_training_support(
    g_call_records: Sequence[Any],
    *,
    topology: Mapping[str, Any],
) -> tuple[list[str], bool, str]:
    """Return the realized training domains for the one shared ``g``.

    DSPy families report the leaf/recompression/merge row populations used to
    optimize one shared program. Unary recompression is training support for
    that program, but only binary merge rows count as merge-domain/C3 support.
    These rows are not a gradient-path claim or a C3 certificate. Neural
    families may instead report direct gradient-path presence. Compatibility
    families without either schema fall back to conservative inference from
    ``train_g`` calls and topology.
    """

    if not g_call_records:
        return [], False, "no_train_g_calls"

    artifacts = [getattr(record, "g_artifact", None) for record in g_call_records]

    # DSPy provenance is checked first, even if an artifact happens to carry
    # neural-family keys as well. Rows seen by an optimizer are not described
    # as a gradient path.
    if _any_artifact_field(
        artifacts,
        (*_DSPY_G_TRAINING_FIELDS, _DSPY_RECOMPRESSION_TRAINING_FIELD),
    ):
        observed_roles: set[str] = set()
        evidence_source: str | None = None
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise TypeError("DSPy g-training provenance requires mapping artifacts")
            missing = tuple(key for key in _DSPY_G_TRAINING_FIELDS if key not in artifact)
            if missing:
                raise ValueError(
                    f"DSPy g-training provenance must contain every field; missing {missing!r}"
                )
            roles = _validated_g_training_roles(artifact["g_training_call_roles"])
            leaf_rows = _validated_training_row_count(
                artifact["leaf_domain_training_row_count"],
                field_name="leaf_domain_training_row_count",
            )
            recompression_rows = (
                _validated_training_row_count(
                    artifact[_DSPY_RECOMPRESSION_TRAINING_FIELD],
                    field_name=_DSPY_RECOMPRESSION_TRAINING_FIELD,
                )
                if _DSPY_RECOMPRESSION_TRAINING_FIELD in artifact
                else 0
            )
            merge_rows = _validated_training_row_count(
                artifact["merge_domain_training_row_count"],
                field_name="merge_domain_training_row_count",
            )
            expected_roles = [
                role
                for role, count in (
                    ("leaf", leaf_rows),
                    ("recompression", recompression_rows),
                    ("merge", merge_rows),
                )
                if count > 0
            ]
            if roles != expected_roles:
                raise ValueError(
                    "g_training_call_roles must agree exactly with positive "
                    "leaf/recompression/merge training-row counts"
                )
            source = _validated_dspy_evidence_source(artifact["g_training_role_evidence_source"])
            if evidence_source is None:
                evidence_source = source
            elif source != evidence_source:
                raise ValueError(
                    "g_training_role_evidence_source must be consistent across all train_g records"
                )
            observed_roles.update(roles)
        roles = [role for role in _G_TRAINING_ROLES if role in observed_roles]
        return roles, "merge" in observed_roles, str(evidence_source)

    if _any_artifact_field(artifacts, _GRADIENT_G_TRAINING_FIELDS):
        direct_activity: list[tuple[bool, bool]] = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise TypeError("g artifact gradient-path provenance requires mappings")
            missing = tuple(key for key in _GRADIENT_G_TRAINING_FIELDS if key not in artifact)
            if missing:
                raise ValueError(
                    "g artifact gradient-path provenance must contain every field; "
                    f"missing {missing!r}"
                )
            if any(not isinstance(artifact[key], bool) for key in _GRADIENT_G_TRAINING_FIELDS):
                raise TypeError("g artifact gradient-path provenance values must be boolean")
            direct_activity.append(
                (
                    artifact[_GRADIENT_G_TRAINING_FIELDS[0]],
                    artifact[_GRADIENT_G_TRAINING_FIELDS[1]],
                )
            )

        leaf_observed = any(leaf for leaf, _merge in direct_activity)
        merge_observed = any(merge for _leaf, merge in direct_activity)
        roles = [
            role
            for role, observed in (("leaf", leaf_observed), ("merge", merge_observed))
            if observed
        ]
        return roles, merge_observed, "g_artifact_gradient_path_presence"

    # Public compatibility fallback: every non-empty C-Tree has a leaf call
    # domain; a merge domain is inferred only when train_g ran over recursive
    # training topology. This does not claim that a gradient flowed through a
    # particular low-level merge parameter or that C3 holds.
    has_training_trees = int(topology.get("train_tree_count") or 0) > 0
    roles = ["leaf"] if has_training_trees else []
    merge_observed = bool(has_training_trees and topology.get("training_composition_present"))
    if merge_observed:
        roles.append("merge")
    return roles, merge_observed, "inferred_from_train_g_and_topology"


def _any_artifact_field(
    artifacts: Sequence[Any],
    fields: Sequence[str],
) -> bool:
    return any(
        isinstance(artifact, Mapping) and any(field in artifact for field in fields)
        for artifact in artifacts
    )


def _validated_g_training_roles(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("g_training_call_roles must be an ordered list or tuple")
    roles = list(value)
    if any(not isinstance(role, str) for role in roles):
        raise TypeError("g_training_call_roles entries must be strings")
    if any(role not in _G_TRAINING_ROLES for role in roles):
        raise ValueError(
            "g_training_call_roles entries must be 'leaf', 'recompression', or 'merge'"
        )
    canonical = [role for role in _G_TRAINING_ROLES if role in roles]
    if roles != canonical:
        raise ValueError(
            "g_training_call_roles must be unique and ordered as the subset "
            "leaf, recompression, merge"
        )
    return roles


def _validated_training_row_count(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _validated_dspy_evidence_source(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("g_training_role_evidence_source must be a string")
    source = value.strip()
    if not source:
        raise ValueError("g_training_role_evidence_source must be non-empty")
    semantic_source = source.lower().replace("-", "_").replace(" ", "_")
    if "gradient_path" in semantic_source or (
        "c3" in semantic_source and "certif" in semantic_source
    ):
        raise ValueError(
            "DSPy row-role evidence must not be described as a gradient path or C3 certificate"
        )
    return source


def _topology_payload(spec: Any, records: Sequence[Any]) -> dict[str, Any]:
    """Read the fit-boundary topology, with a legacy-record fallback."""

    for record in reversed(tuple(records)):
        extra = dict(getattr(record, "extra", None) or {})
        payload = extra.get("topology_contract")
        if isinstance(payload, Mapping) and payload:
            return dict(payload)
    train_data = _as_sequence(getattr(spec, "train_data", None))
    eval_data = _as_sequence(getattr(spec, "eval_data", None))
    return resolve_topology_contract(
        train_data,
        eval_data,
        axis=dict(getattr(spec, "axis", None) or {}),
        backend_config=dict(getattr(spec, "backend_config", None) or {}),
    ).to_dict()


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _record_g_update_performed(record: Any) -> bool:
    """Read the runtime's realized update signal with legacy compatibility."""

    extra = dict(getattr(record, "extra", None) or {})
    if "g_update_performed" not in extra:
        # Older IterationRecord producers used ``trained='g'`` as the only
        # signal, so retain that interpretation for pre-contract records.
        return getattr(record, "trained", None) == "g"
    value = extra["g_update_performed"]
    if not isinstance(value, bool):
        raise TypeError("IterationRecord.extra['g_update_performed'] must be boolean")
    return value


METRIC_FIELDS: tuple[str, ...] = (
    "joint_f_l1",
    "internal_f_pearson",
    "internal_f_mae",
    "external_expert_pearson",
    "external_expert_mae",
    "f_star_gap",
    "mean_prediction",
    "mean_teacher",
    "mean_expert",
)

SPLIT_ORDER: tuple[str, ...] = ("all", "train", "val", "test")


def final_metrics(record: Any | None) -> Mapping[str, float]:
    """Flatten every available split into a single metric dict."""
    if record is None or not record.split_metrics:
        return {}
    out: dict[str, float] = {}
    for split_name in SPLIT_ORDER:
        sm = record.split_metrics.get(split_name)
        if sm is None:
            continue
        prefix = "" if split_name == "all" else f"{split_name}_"
        for field_name in METRIC_FIELDS:
            value = getattr(sm, field_name, None)
            if value is not None:
                out[f"{prefix}{field_name}"] = float(value)
        for dim_name, dim_metrics in (getattr(sm, "per_dimension", None) or {}).items():
            for field_name, value in dict(dim_metrics or {}).items():
                if value is not None:
                    out[f"{prefix}{dim_name}_{field_name}"] = float(value)
        out[f"{prefix}n"] = float(getattr(sm, "n", 0))
    return out


def split_metrics_payload(record: Any | None) -> Mapping[str, Any]:
    """Structured ``summary['split_metrics']`` carrying every per-split field."""
    if record is None or not record.split_metrics:
        return {}
    out: dict[str, Any] = {}
    for split_name, sm in record.split_metrics.items():
        entry: dict[str, Any] = {"n": int(getattr(sm, "n", 0))}
        for field_name in METRIC_FIELDS:
            value = getattr(sm, field_name, None)
            if value is not None:
                entry[field_name] = float(value)
        entry["joint_f_l1_n"] = int(getattr(sm, "joint_f_l1_n", 0))
        per_dimension = getattr(sm, "per_dimension", None) or {}
        if per_dimension:
            entry["per_dimension"] = {
                str(dim): {
                    str(k): (float(v) if v is not None else None) for k, v in dim_metrics.items()
                }
                for dim, dim_metrics in per_dimension.items()
            }
        out[str(split_name)] = entry
    return out


def collect_prediction_records(output_dir: Path) -> list[str]:
    """Return per-iteration prediction-record JSONL paths."""
    pred_dir = output_dir / "prediction_records"
    if not pred_dir.exists():
        return []
    return sorted(str(p) for p in pred_dir.glob("iter_*_post_eval.jsonl"))


def write_prediction_records(output_dir: Path, records: Sequence[Any]) -> None:
    pred_dir = output_dir / "prediction_records"
    for record in records:
        rows = (getattr(record, "extra", None) or {}).get("prediction_rows") or []
        if not rows:
            continue
        pred_dir.mkdir(parents=True, exist_ok=True)
        path = pred_dir / f"iter_{int(record.iteration):02d}_post_eval.jsonl"
        enriched = []
        for row in rows:
            payload = dict(row)
            payload.setdefault("iteration", int(record.iteration))
            payload.setdefault("stage_name", str(record.stage_name))
            payload.setdefault("family", str(record.family))
            enriched.append(payload)
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True, default=json_default) for row in enriched)
            + "\n",
            encoding="utf-8",
        )


__all__ = [
    "build_result",
    "collect_prediction_records",
    "final_metrics",
    "g_contract_payload",
    "split_metrics_payload",
    "write_prediction_records",
]
