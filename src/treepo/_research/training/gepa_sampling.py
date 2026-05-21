"""
GEPA sampling helpers for optimization datasets.

This module centralizes GEPA sampling logic so run_pipeline orchestration stays
focused on phase control. It supports:

- Uniform SRSWOR example sampling (legacy behavior)
- Two-stage PPS/Bernoulli sampling with logged propensities
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from treepo._research.stats.sampling import (
    pps_inclusion_probabilities,
    systematic_pps_sample_indices,
)

_SAMPLING_FIELDS: Tuple[str, ...] = (
    "sampling_design",
    "sampling_doc_inclusion_prob",
    "sampling_node_given_doc_prob",
    "sampling_joint_inclusion_prob",
    "sampling_ipw_weight",
    "sampling_hajek_weight",
    "sampling_ht_weight",
    "sampling_population_size",
    "sampling_realized_sample_size",
)


def _clear_sampling_annotations(example: Any) -> None:
    for field in _SAMPLING_FIELDS:
        if hasattr(example, field):
            try:
                delattr(example, field)
            except Exception:
                # Best effort only.
                pass


def _set_sampling_annotations(example: Any, payload: Dict[str, Any]) -> None:
    for key, value in payload.items():
        try:
            setattr(example, key, value)
        except Exception:
            continue


def _clone_example(example: Any) -> Any:
    """Best-effort clone that avoids mutating shared example objects."""
    try:
        return copy.copy(example)
    except Exception:
        try:
            return copy.deepcopy(example)
        except Exception:
            return example


def _extract_doc_id(example: Any, fallback_idx: int) -> str:
    doc_id = getattr(example, "doc_id", None)
    if doc_id is None:
        doc_id = f"doc_{fallback_idx}"
    return str(doc_id)


def _base_metadata(
    *,
    component_id: str,
    split: str,
    population_size: int,
    design: str,
    seed: int,
    target_size: Optional[int],
) -> Dict[str, Any]:
    return {
        "enabled": False,
        "component": str(component_id),
        "split": str(split),
        "population_size": int(population_size),
        "design": str(design),
        "target_size": None if target_size is None else int(target_size),
        "seed": int(seed),
    }


def sample_srswor_examples(
    examples: Sequence[Any],
    *,
    component_id: str,
    split: str,
    seed: int,
    target_size: Optional[int],
    min_required: int = 1,
    min_propensity: float = 1e-6,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Sample examples uniformly without replacement.

    For selected examples, this logs per-example propensity/weight fields used by
    downstream metric weighting.
    """
    n_examples = int(len(examples))
    metadata = _base_metadata(
        component_id=component_id,
        split=split,
        population_size=n_examples,
        design="srswor",
        seed=seed,
        target_size=target_size,
    )
    if n_examples <= 0:
        return list(examples), metadata

    if target_size is None:
        sampled = list(examples)
        metadata["reason"] = "no_target"
        return sampled, metadata

    target = max(int(min_required), int(target_size))
    target = min(target, n_examples)
    if target >= n_examples:
        sampled = list(examples)
        metadata.update(
            {
                "reason": "full_population",
                "sample_size": int(n_examples),
            }
        )
        return sampled, metadata

    rng = random.Random(int(seed))
    selected_indices = sorted(rng.sample(range(n_examples), k=target))
    sampled: List[Any] = []

    inclusion_prob = float(target) / float(n_examples)
    joint_prob = max(float(min_propensity), inclusion_prob)
    ipw = 1.0 / joint_prob
    # For mean aggregation, this recovers the Hajek objective exactly under
    # fixed-size SRS (all weights equal => coefficient 1.0).
    hajek_weight = 1.0
    ht_weight = float(target) / float(n_examples) * ipw

    for idx in selected_indices:
        ex = _clone_example(examples[idx])
        _clear_sampling_annotations(ex)
        _set_sampling_annotations(
            ex,
            {
                "sampling_design": "srswor",
                "sampling_doc_inclusion_prob": inclusion_prob,
                "sampling_node_given_doc_prob": 1.0,
                "sampling_joint_inclusion_prob": inclusion_prob,
                "sampling_ipw_weight": ipw,
                "sampling_hajek_weight": hajek_weight,
                "sampling_ht_weight": ht_weight,
                "sampling_population_size": int(n_examples),
                "sampling_realized_sample_size": int(target),
            },
        )
        sampled.append(ex)

    metadata.update(
        {
            "enabled": True,
            "sample_size": int(target),
            "inclusion_prob": float(inclusion_prob),
            "joint_propensity_min": float(inclusion_prob),
            "joint_propensity_max": float(inclusion_prob),
            "ipw_weight_min": float(ipw),
            "ipw_weight_max": float(ipw),
            "ipw_weight_mean": float(ipw),
            "effective_sample_size": float(target),
        }
    )
    return sampled, metadata


def sample_two_stage_pps_bernoulli(
    examples: Sequence[Any],
    *,
    component_id: str,
    split: str,
    seed: int,
    target_size: Optional[int],
    min_required: int = 1,
    min_propensity: float = 1e-6,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Two-stage sampler:
      1) PPS without replacement over documents
      2) Bernoulli node sampling within selected documents

    This logs exact per-example joint propensities for sampled examples:
      pi_joint = pi_doc * q_doc
    """
    n_examples = int(len(examples))
    metadata = _base_metadata(
        component_id=component_id,
        split=split,
        population_size=n_examples,
        design="two_stage_pps_bernoulli",
        seed=seed,
        target_size=target_size,
    )
    if n_examples <= 0:
        return list(examples), metadata

    if target_size is None:
        sampled = list(examples)
        metadata["reason"] = "no_target"
        return sampled, metadata

    target = max(int(min_required), int(target_size))
    target = min(target, n_examples)
    if target >= n_examples:
        sampled = list(examples)
        metadata.update(
            {
                "reason": "full_population",
                "sample_size": int(n_examples),
            }
        )
        return sampled, metadata

    by_doc: Dict[str, List[int]] = {}
    for idx, example in enumerate(examples):
        doc_id = _extract_doc_id(example, fallback_idx=idx)
        by_doc.setdefault(doc_id, []).append(idx)

    doc_ids = sorted(by_doc.keys())
    n_docs = int(len(doc_ids))
    if n_docs <= 0:
        fallback_examples, fallback_meta = sample_srswor_examples(
            examples,
            component_id=component_id,
            split=split,
            seed=seed,
            target_size=target,
            min_required=min_required,
            min_propensity=min_propensity,
        )
        fallback_meta["design"] = "srswor"
        fallback_meta["fallback_reason"] = "no_docs"
        return fallback_examples, fallback_meta

    k_doc = min(n_docs, target)
    if k_doc <= 0:
        return list(examples), metadata

    doc_sizes = [len(by_doc[doc_id]) for doc_id in doc_ids]
    doc_probs = pps_inclusion_probabilities(doc_sizes, k_doc)
    rng = random.Random(int(seed))
    sampled_doc_positions = systematic_pps_sample_indices(doc_probs, k_doc, rng)
    sampled_doc_positions = sorted(set(int(pos) for pos in sampled_doc_positions))
    if not sampled_doc_positions:
        fallback_examples, fallback_meta = sample_srswor_examples(
            examples,
            component_id=component_id,
            split=split,
            seed=seed,
            target_size=target,
            min_required=min_required,
            min_propensity=min_propensity,
        )
        fallback_meta["design"] = "srswor"
        fallback_meta["fallback_reason"] = "no_docs_selected"
        return fallback_examples, fallback_meta

    c = float(target) / float(max(1, k_doc))

    # Unconditional expected sample size across both stages.
    expected_sample_size = 0.0
    for pos, doc_id in enumerate(doc_ids):
        n_doc_nodes = max(1, int(len(by_doc[doc_id])))
        q_doc = min(1.0, c / float(n_doc_nodes))
        expected_sample_size += float(doc_probs[pos]) * float(n_doc_nodes) * float(q_doc)

    sampled_indices: List[int] = []
    sample_payloads: List[Tuple[int, float, float, float]] = []
    selected_doc_count = 0
    for pos in sampled_doc_positions:
        if pos < 0 or pos >= n_docs:
            continue
        doc_id = doc_ids[pos]
        node_indices = by_doc[doc_id]
        n_doc_nodes = max(1, int(len(node_indices)))
        pi_doc = max(float(min_propensity), float(doc_probs[pos]))
        q_doc = min(1.0, c / float(n_doc_nodes))
        selected_doc_count += 1
        for node_idx in node_indices:
            if rng.random() <= q_doc:
                pi_joint = max(float(min_propensity), pi_doc * q_doc)
                sampled_indices.append(int(node_idx))
                sample_payloads.append((int(node_idx), pi_doc, q_doc, pi_joint))

    if not sampled_indices:
        fallback_examples, fallback_meta = sample_srswor_examples(
            examples,
            component_id=component_id,
            split=split,
            seed=seed,
            target_size=target,
            min_required=min_required,
            min_propensity=min_propensity,
        )
        fallback_meta["design"] = "srswor"
        fallback_meta["fallback_reason"] = "no_nodes_selected"
        return fallback_examples, fallback_meta

    ordered = sorted(zip(sampled_indices, sample_payloads), key=lambda item: item[0])
    sampled: List[Any] = []

    ipw_weights = [1.0 / max(float(min_propensity), payload[3]) for _, payload in ordered]
    sum_w = sum(ipw_weights)
    sum_w_sq = sum(w * w for w in ipw_weights)
    ess = (sum_w * sum_w / sum_w_sq) if sum_w_sq > 0 else 0.0

    realized_sample_size = int(len(ordered))
    for (idx, payload), ipw_w in zip(ordered, ipw_weights):
        _, pi_doc, q_doc, pi_joint = payload
        hajek_weight = (
            float(realized_sample_size) * float(ipw_w) / float(sum_w)
            if sum_w > 0
            else 1.0
        )
        ht_weight = (
            float(realized_sample_size) / float(max(1, n_examples)) * float(ipw_w)
        )
        ex = _clone_example(examples[idx])
        _clear_sampling_annotations(ex)
        _set_sampling_annotations(
            ex,
            {
                "sampling_design": "two_stage_pps_bernoulli",
                "sampling_doc_inclusion_prob": float(pi_doc),
                "sampling_node_given_doc_prob": float(q_doc),
                "sampling_joint_inclusion_prob": float(pi_joint),
                "sampling_ipw_weight": float(ipw_w),
                "sampling_hajek_weight": float(hajek_weight),
                "sampling_ht_weight": float(ht_weight),
                "sampling_population_size": int(n_examples),
                "sampling_realized_sample_size": int(realized_sample_size),
            },
        )
        sampled.append(ex)

    joint_probs = [payload[3] for _, payload in ordered]
    metadata.update(
        {
            "enabled": True,
            "sample_size": int(realized_sample_size),
            "realized_sample_size": int(realized_sample_size),
            "target_expected_sample_size": int(target),
            "expected_sample_size": float(expected_sample_size),
            "doc_population_size": int(n_docs),
            "doc_sample_size": int(selected_doc_count),
            "doc_sample_target": int(k_doc),
            "joint_propensity_min": float(min(joint_probs)),
            "joint_propensity_max": float(max(joint_probs)),
            "joint_propensity_mean": float(sum(joint_probs) / max(1, len(joint_probs))),
            "ipw_weight_min": float(min(ipw_weights)),
            "ipw_weight_max": float(max(ipw_weights)),
            "ipw_weight_mean": float(sum_w / max(1, len(ipw_weights))),
            "effective_sample_size": float(ess),
        }
    )
    return sampled, metadata
