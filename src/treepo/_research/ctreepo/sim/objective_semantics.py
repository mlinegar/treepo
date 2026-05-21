from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


def _float_mapping(values: Optional[Mapping[str, object]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, value in dict(values or {}).items():
        try:
            out[str(name)] = float(value)  # type: ignore[arg-type]
        except Exception:
            continue
    return out


def make_objective_semantics(
    *,
    name: str,
    kind: str,
    optimized_against: str,
    weighting_scheme: str,
    selection_metric_name: str = "",
    interprets_lambda_as: str = "not_applicable",
    component_weights: Optional[Mapping[str, object]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return {
        "name": str(name),
        "kind": str(kind),
        "optimized_against": str(optimized_against),
        "weighting_scheme": str(weighting_scheme),
        "selection_metric_name": str(selection_metric_name or ""),
        "interprets_lambda_as": str(interprets_lambda_as),
        "component_weights": _float_mapping(component_weights),
        "metadata": dict(metadata or {}),
    }


def latent_quadratic_utility_objective_semantics(
    *,
    name: str,
    optimized_against: str,
    quadratic_utility_weight: float,
    linear_component_name: str,
    interaction_component_name: str,
    weighting_scheme: str = "linear_plus_quadratic_interaction",
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return make_objective_semantics(
        name=str(name),
        kind="latent_oracle_target",
        optimized_against=str(optimized_against),
        weighting_scheme=str(weighting_scheme),
        interprets_lambda_as="quadratic_utility_weight",
        component_weights={
            str(linear_component_name): 1.0,
            str(interaction_component_name): float(quadratic_utility_weight),
        },
        metadata={
            "quadratic_utility_weight": float(quadratic_utility_weight),
            **dict(metadata or {}),
        },
    )


def latent_lambda_objective_semantics(
    *,
    name: str,
    optimized_against: str,
    lambda_multiplier: float,
    linear_component_name: str,
    interaction_component_name: str,
    weighting_scheme: str = "linear_plus_lambda_interaction",
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return latent_quadratic_utility_objective_semantics(
        name=str(name),
        optimized_against=str(optimized_against),
        quadratic_utility_weight=float(lambda_multiplier),
        linear_component_name=str(linear_component_name),
        interaction_component_name=str(interaction_component_name),
        weighting_scheme=str(weighting_scheme),
        metadata=metadata,
    )


def mergeable_target_objective_semantics(
    *,
    name: str,
    optimized_against: str,
    target_kind: str,
    component_weights: Optional[Mapping[str, object]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    weights = dict(component_weights or {str(target_kind): 1.0})
    return make_objective_semantics(
        name=str(name),
        kind="mergeable_target",
        optimized_against=str(optimized_against),
        weighting_scheme="direct_target_supervision",
        interprets_lambda_as="not_applicable",
        component_weights=weights,
        metadata={"target_kind": str(target_kind), **dict(metadata or {})},
    )


def mergeable_probability_target_objective_semantics(
    *,
    name: str,
    target_ks: Optional[Sequence[int]] = None,
    target_k: Optional[int] = None,
    optimized_against: str = "probability_of_at_least_k_spikes",
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    ks = [int(k) for k in (list(target_ks or []) or ([int(target_k)] if target_k is not None else []))]
    weights = {f"indicator_ge_{int(k)}_spikes": 1.0 for k in ks}
    return mergeable_target_objective_semantics(
        name=str(name),
        optimized_against=str(optimized_against),
        target_kind="probability_of_at_least_k_spikes",
        component_weights=weights,
        metadata={
            "target_k": int(target_k) if target_k is not None else None,
            "target_ks": ks,
            **dict(metadata or {}),
        },
    )


def mergeable_parameter_vector_objective_semantics(
    *,
    name: str,
    parameter_names: Sequence[str],
    optimized_against: str = "parameter_vector_recovery",
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return mergeable_target_objective_semantics(
        name=str(name),
        optimized_against=str(optimized_against),
        target_kind="parameter_vector",
        component_weights={str(param): 1.0 for param in parameter_names},
        metadata=dict(metadata or {}),
    )


def mergeable_document_objective_semantics(
    *,
    name: str,
    objective_profile: str,
    optimized_against: str = "document_level_ground_truth_objective",
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return mergeable_target_objective_semantics(
        name=str(name),
        optimized_against=str(optimized_against),
        target_kind="document_objective",
        component_weights={str(objective_profile): 1.0},
        metadata={"objective_profile": str(objective_profile), **dict(metadata or {})},
    )


def discrepancy_benchmark_objective_semantics(
    *,
    name: str,
    optimized_against: str,
    benchmark_metric_name: str,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return make_objective_semantics(
        name=str(name),
        kind="discrepancy_benchmark",
        optimized_against=str(optimized_against),
        weighting_scheme="benchmark_metrics_only",
        interprets_lambda_as="not_applicable",
        component_weights={},
        metadata={"benchmark_metric_name": str(benchmark_metric_name), **dict(metadata or {})},
    )


def preference_training_objective_semantics(
    *,
    objective_family: str,
    root_query_rate: float,
    leaf_label_rate: float,
    internal_label_rate: float,
    hybrid_weight: float,
    dpo_beta: float,
    grpo_beta: float,
    ppo_kl_weight: float,
    entropy_weight: float,
    pairwise_prefs_per_doc: float,
    group_pref_groups_per_doc: float,
    ppo_rollouts_per_doc: float,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    family = str(objective_family)
    weights: Dict[str, float]
    optimized_against = "oracle_action_utility"
    if family == "supervised_root":
        weights = {"root_supervision": 1.0}
    elif family == "supervised_state":
        weights = {"state_supervision": 1.0}
    elif family == "dpo":
        weights = {"dpo": 1.0}
    elif family == "grpo":
        weights = {"grpo": 1.0}
    elif family == "ppo":
        weights = {"ppo": 1.0}
    elif family == "hybrid_supervised_plus_dpo":
        weights = {"state_supervision": float(hybrid_weight), "dpo": float(1.0 - hybrid_weight)}
    elif family == "hybrid_supervised_plus_grpo":
        weights = {"state_supervision": float(hybrid_weight), "grpo": float(1.0 - hybrid_weight)}
    elif family == "hybrid_supervised_plus_ppo":
        weights = {"state_supervision": float(hybrid_weight), "ppo": float(1.0 - hybrid_weight)}
    else:
        weights = {}
    return make_objective_semantics(
        name="exact_utility_training_objective",
        kind="preference_training_objective",
        optimized_against=optimized_against,
        weighting_scheme=family,
        interprets_lambda_as="not_applicable",
        component_weights=weights,
        metadata={
            "objective_family": family,
            "root_query_rate": float(root_query_rate),
            "leaf_label_rate": float(leaf_label_rate),
            "internal_label_rate": float(internal_label_rate),
            "hybrid_weight": float(hybrid_weight),
            "dpo_beta": float(dpo_beta),
            "grpo_beta": float(grpo_beta),
            "ppo_kl_weight": float(ppo_kl_weight),
            "entropy_weight": float(entropy_weight),
            "pairwise_prefs_per_doc": float(pairwise_prefs_per_doc),
            "group_pref_groups_per_doc": float(group_pref_groups_per_doc),
            "ppo_rollouts_per_doc": float(ppo_rollouts_per_doc),
            **dict(metadata or {}),
        },
    )


__all__ = [
    "discrepancy_benchmark_objective_semantics",
    "latent_quadratic_utility_objective_semantics",
    "latent_lambda_objective_semantics",
    "make_objective_semantics",
    "mergeable_document_objective_semantics",
    "mergeable_parameter_vector_objective_semantics",
    "mergeable_probability_target_objective_semantics",
    "mergeable_target_objective_semantics",
    "preference_training_objective_semantics",
]
