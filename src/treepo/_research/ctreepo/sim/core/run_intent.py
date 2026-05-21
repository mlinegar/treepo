from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Mapping

from treepo._research.ctreepo.opt.serialization import canonical_json
from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    LAW_SET_ALL,
    canonical_law_component_weights,
    canonical_law_set_id,
)
from treepo._research.ctreepo.sim.core.full_doc_config_codec import (
    canonicalize_full_doc_config_mapping,
)
from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig


RUN_INTENT_VERSION = "tree_run_intent_v4"

TOPOLOGY_TREE = "tree"
TOPOLOGY_FULL_DOC = "full_doc"
VALID_TOPOLOGIES = frozenset({TOPOLOGY_TREE, TOPOLOGY_FULL_DOC, ""})

RUN_INTENT_FIELDS: tuple[str, ...] = (
    "run_intent_version",
    "problem_id",
    "method_id",
    "law_set_id",
    "topology",
    "comparison_mode",
    "tree_exact_collapse_mode",
    "fixed_leaf_tokens",
    "tree_root_supervision_kind",
    "tree_document_loss_normalization_mode",
    "tree_supervision_source",
    "tree_local_weighting_mode",
    "tree_c2_mode",
    "local_law_weight",
    "root_share",
    "local_law_component_weights",
    "schedule_consistency_weight",
    "depth_discount_gamma",
    "leaf_supervision_kind",
    "leaf_label_rate",
    "leaf_exact_supervision",
    "internal_supervision_kind",
    "internal_label_rate",
    "max_internal_depth",
    "budget_total_calls",
    "budget_total_calls_per_doc",
    "mass_target_per_doc",
    "full_doc_budget_share",
    "doc_consumption_mode",
    "local_split_mode",
    "local_allocation_policy",
    "package_semantics",
    "aligned_sketch_surface",
    "summary_spec_name",
    "slot_count",
)


def _mapping_from_config_like(config_like: Mapping[str, Any] | OPSCountConfig | None) -> Dict[str, Any]:
    return canonicalize_full_doc_config_mapping(
        config_like,
        allow_private_tree_aliases=True,
    )


def _clean_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping.get(key)
            if value not in {"", None}:
                return value
    return default


def _resolve_local_active(mapping: Mapping[str, Any]) -> bool:
    leaf_rate = _clean_float(mapping.get("leaf_label_rate", 0.0), default=0.0)
    internal_rate = _clean_float(mapping.get("internal_label_rate", 0.0), default=0.0)
    internal_kind = str(mapping.get("internal_supervision_kind", "none") or "none").strip().lower()
    return bool(
        leaf_rate > 1e-12
        or (internal_kind not in {"", "none"} and internal_rate > 1e-12)
    )


def resolve_package_semantics(mapping: Mapping[str, Any]) -> str:
    explicit = str(mapping.get("package_semantics", "") or "").strip()
    if explicit:
        return explicit
    local_active = _resolve_local_active(mapping)
    mass_target = _optional_float(mapping.get("mass_target_per_doc"))
    budget_total_calls_per_doc = _clean_float(
        mapping.get("budget_total_calls_per_doc", 0.0),
        default=0.0,
    )
    if local_active and mass_target is not None and budget_total_calls_per_doc <= max(1e-12, mass_target):
        return "mass_matched"
    if local_active and budget_total_calls_per_doc > 0.0:
        return "superset"
    if local_active:
        return "local_only"
    return "full_doc_only"


def materialize_tree_run_intent(
    config_like: Mapping[str, Any] | OPSCountConfig | None,
    *,
    fixed_leaf_tokens_override: int | None = None,
    baseline_family_override: str | None = None,
    method_id_override: str | None = None,
) -> Dict[str, Any]:
    mapping = _mapping_from_config_like(config_like)
    method_id = str(
        method_id_override
        if method_id_override is not None
        else baseline_family_override
        if baseline_family_override is not None
        else mapping.get("method_id", "")
        or mapping.get("baseline_family", "")
        or ""
    ).strip()
    if not method_id:
        raise ValueError(
            "materialize_tree_run_intent requires a non-empty method_id"
        )
    law_set_id = canonical_law_set_id(
        str(
            mapping.get("law_set_id")
            or mapping.get("law_package")
            or LAW_SET_ALL
        ),
        allow_aliases=True,
    )
    component_weights = canonical_law_component_weights(
        mapping.get("local_law_component_weights")
        or {
            LAW_ID_LEAF_PRESERVATION: _clean_float(
                _first_present(mapping, "c1_relative_weight", "tree_c1_relative_weight", default=1.0),
                default=1.0,
            ),
            LAW_ID_ON_RANGE_IDEMPOTENCE: _clean_float(
                _first_present(mapping, "c2_relative_weight", "tree_c2_relative_weight", default=1.0),
                default=1.0,
            ),
            LAW_ID_MERGE_PRESERVATION: _clean_float(
                _first_present(mapping, "c3_relative_weight", "tree_c3_relative_weight", default=1.0),
                default=1.0,
            ),
        },
        allow_aliases=True,
    )
    fixed_leaf_tokens_raw = (
        mapping.get("fixed_leaf_tokens", 0)
        if fixed_leaf_tokens_override is None
        else fixed_leaf_tokens_override
    )
    try:
        fixed_leaf_tokens = int(fixed_leaf_tokens_raw or 0)
    except Exception:
        fixed_leaf_tokens = 0
    topology_raw = str(mapping.get("topology", "") or "").strip()
    if topology_raw and topology_raw not in VALID_TOPOLOGIES:
        raise ValueError(
            f"topology must be one of {sorted(VALID_TOPOLOGIES)}, got {topology_raw!r}"
        )
    topology = topology_raw

    intent = {
        "run_intent_version": RUN_INTENT_VERSION,
        "problem_id": str(mapping.get("problem_id") or "markov_ops_count"),
        "method_id": method_id,
        "law_set_id": law_set_id,
        "topology": topology,
        "comparison_mode": str(mapping.get("comparison_mode", "legacy") or "legacy"),
        "tree_exact_collapse_mode": str(mapping.get("tree_exact_collapse_mode", "") or ""),
        "fixed_leaf_tokens": int(fixed_leaf_tokens),
        "tree_root_supervision_kind": str(
            mapping.get("tree_root_supervision_kind", "mse") or "mse"
        ),
        "tree_document_loss_normalization_mode": str(
            mapping.get("tree_document_loss_normalization_mode", "auto") or "auto"
        ),
        "tree_supervision_source": str(
            mapping.get("tree_supervision_source", "rate") or "rate"
        ),
        "tree_local_weighting_mode": str(
            mapping.get("tree_local_weighting_mode", "fixed_k_hajek")
            or "fixed_k_hajek"
        ),
        "tree_c2_mode": str(
            mapping.get("tree_c2_mode", "reconstruction") or "reconstruction"
        ),
        "local_law_weight": _optional_float(
            _first_present(mapping, "local_law_weight")
        ),
        "root_share": _optional_float(
            _first_present(mapping, "root_share")
        ),
        "local_law_component_weights": component_weights,
        "schedule_consistency_weight": _clean_float(
            mapping.get("schedule_consistency_weight", 0.0),
            default=0.0,
        ),
        "depth_discount_gamma": _clean_float(
            mapping.get("depth_discount_gamma", 1.0),
            default=1.0,
        ),
        "leaf_supervision_kind": str(
            mapping.get("leaf_supervision_kind", "") or ""
        ),
        "leaf_label_rate": _clean_float(mapping.get("leaf_label_rate", 1.0), default=1.0),
        "leaf_exact_supervision": bool(mapping.get("leaf_exact_supervision", False)),
        "internal_supervision_kind": str(
            mapping.get("internal_supervision_kind", "none") or "none"
        ),
        "internal_label_rate": _clean_float(
            mapping.get("internal_label_rate", 0.0),
            default=0.0,
        ),
        "max_internal_depth": int(_clean_float(mapping.get("max_internal_depth", 0), default=0.0)),
        "budget_total_calls": int(_clean_float(mapping.get("budget_total_calls", 0), default=0.0)),
        "budget_total_calls_per_doc": _clean_float(
            mapping.get("budget_total_calls_per_doc", 0.0),
            default=0.0,
        ),
        "mass_target_per_doc": _optional_float(mapping.get("mass_target_per_doc")),
        "full_doc_budget_share": _clean_float(
            mapping.get("full_doc_budget_share", 1.0),
            default=1.0,
        ),
        "doc_consumption_mode": str(mapping.get("doc_consumption_mode", "") or ""),
        "local_split_mode": str(mapping.get("local_split_mode", "") or ""),
        "local_allocation_policy": str(
            mapping.get("local_allocation_policy", "") or ""
        ),
        "package_semantics": resolve_package_semantics(mapping),
        "aligned_sketch_surface": str(mapping.get("aligned_sketch_surface", "") or ""),
        "summary_spec_name": str(mapping.get("summary_spec_name", "") or ""),
        "slot_count": int(_clean_float(mapping.get("slot_count", 0), default=0.0)),
    }
    _validate_intent_ranges(intent)
    return intent


def _validate_intent_ranges(intent: Dict[str, Any]) -> None:
    gamma = intent["depth_discount_gamma"]
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(
            f"depth_discount_gamma must be in [0, 1], got {gamma}"
        )
    for field in (
        "schedule_consistency_weight",
    ):
        val = intent[field]
        if val < 0.0:
            raise ValueError(f"{field} must be non-negative, got {val}")
    for law_id, val in dict(intent["local_law_component_weights"]).items():
        if float(val) < 0.0:
            raise ValueError(f"local_law_component_weights[{law_id}] must be non-negative, got {val}")
    for field in ("local_law_weight", "root_share"):
        val = intent[field]
        if val is not None and val < 0.0:
            raise ValueError(f"{field} must be non-negative, got {val}")


def intent_diff(
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
    *,
    ignore_fields: tuple[str, ...] = (),
) -> Dict[str, Dict[str, Any]]:
    expected_mapping = dict(expected or {})
    actual_mapping = dict(actual or {})
    keys = sorted((set(expected_mapping) | set(actual_mapping)) - set(ignore_fields))
    diff: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        if expected_mapping.get(key) != actual_mapping.get(key):
            diff[str(key)] = {
                "expected": expected_mapping.get(key),
                "actual": actual_mapping.get(key),
            }
    return diff


def intent_hash(intent: Mapping[str, Any] | None, *, n_chars: int = 16) -> str:
    payload = dict(intent or {})
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[: max(8, int(n_chars))]


def intent_is_complete(intent: Mapping[str, Any] | None) -> bool:
    if not isinstance(intent, Mapping):
        return False
    if str(intent.get("run_intent_version", "")).strip() != RUN_INTENT_VERSION:
        return False
    return all(field_name in intent for field_name in RUN_INTENT_FIELDS)
