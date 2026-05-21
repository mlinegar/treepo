"""Shared TreePO optimizer-facing metadata and weighting helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence

from treepo._research.core.logged_supervision import SamplingMetadata
from treepo._research.core.supervision_metadata import normalize_judgment_law_type

TreePOWeightingMode = Literal["legacy_channel", "discounted_tree"]
TreePORLRole = Literal["dpo_pair", "reward_pair", "scalar_reward", "grpo_prompt"]
TreePOChannel = Literal["root", "c1", "c2", "c3", "unknown"]

_VALID_WEIGHTING_MODES = frozenset({"legacy_channel", "discounted_tree"})
_DEPTH_KEYS = ("depth", "tree_depth", "node_depth", "level", "tree_level")
_NODE_ID_KEYS = ("node_id", "unit_id", "observation_id")
_DOCUMENT_ID_KEYS = ("document_id", "doc_id", "source_doc_id")
_LEGACY_CHANNEL_WEIGHTS: Dict[str, float] = {
    "root": 1.0,
    "c1": 1.0,
    "c2": 1.0,
    "c3": 1.0,
    "unknown": 1.0,
}


def validate_tree_objective_weighting_mode(mode: str) -> str:
    text = str(mode or "legacy_channel").strip().lower()
    if text not in _VALID_WEIGHTING_MODES:
        raise ValueError(
            "tree_objective_weighting_mode must be one of "
            f"{sorted(_VALID_WEIGHTING_MODES)!r}, got {mode!r}"
        )
    return text


def validate_discount_gamma(gamma: float) -> float:
    value = float(gamma)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"discount_gamma must be finite and in [0, 1], got {gamma!r}")
    return value


def resolve_treepo_channel(
    *,
    law_type: Optional[str] = None,
    supervision_channel_name: Optional[str] = None,
    supervision_signal_name: Optional[str] = None,
) -> TreePOChannel:
    law = normalize_judgment_law_type(law_type)
    signal = str(supervision_signal_name or "").strip().lower()
    channel_name = str(supervision_channel_name or "").strip().lower()

    if law == "document_level_target" or signal in {
        "document_level_target",
        "document_target",
        "root",
        "root_target",
    }:
        return "root"
    if law == "sufficiency":
        return "c1"
    if law == "idempotence":
        return "c2"
    if law in {"merge", "substitution"}:
        return "c3"
    if "document" in signal or "document" in channel_name:
        return "root"
    return "unknown"


def resolve_treepo_objective_weight(
    *,
    channel: str,
    depth: int,
    weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> float:
    mode = validate_tree_objective_weighting_mode(weighting_mode)
    gamma = validate_discount_gamma(discount_gamma)
    base_weight = float(_LEGACY_CHANNEL_WEIGHTS.get(str(channel), 1.0))
    if mode == "legacy_channel":
        return base_weight
    return float(base_weight * (gamma ** max(0, int(depth))))


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    return {}


def _first_nonempty_string(values: Iterable[Any]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _parse_nonnegative_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def infer_treepo_depth(*metadata_sources: Mapping[str, Any]) -> int:
    for source in metadata_sources:
        mapping = _as_mapping(source)
        for key in _DEPTH_KEYS:
            parsed = _parse_nonnegative_int(mapping.get(key))
            if parsed is not None:
                return parsed
    return 0


def infer_treepo_document_id(
    *,
    source_doc_id: Optional[str] = None,
    source_example_id: Optional[str] = None,
    metadata_sources: Sequence[Mapping[str, Any]] = (),
) -> str:
    discovered = _first_nonempty_string(
        [source_doc_id]
        + [mapping.get(key) for mapping in metadata_sources for key in _DOCUMENT_ID_KEYS]
        + [source_example_id]
    )
    return discovered or "unknown_document"


def infer_treepo_node_id(
    *,
    fallback_node_id: Optional[str] = None,
    source_observation_ids: Sequence[str] = (),
    metadata_sources: Sequence[Mapping[str, Any]] = (),
) -> str:
    discovered = _first_nonempty_string(
        [mapping.get(key) for mapping in metadata_sources for key in _NODE_ID_KEYS]
        + list(source_observation_ids)
        + [fallback_node_id]
    )
    return discovered or "unknown_node"


@dataclass(frozen=True)
class TreePOOptimizerExportMetadata:
    """Canonical TreePO metadata attached to optimizer-facing exports."""

    document_id: str
    node_id: str
    depth: int
    channel: str
    joint_propensity: float
    joint_propensity_source: str
    objective_weight: float
    ipw_weight: float
    effective_weight: float
    weighting_mode: str = "legacy_channel"
    discount_gamma: float = 1.0
    sample_weight_source: str = "effective_weight"
    rl_role: Optional[str] = None
    local_law_adjustment: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "document_id": self.document_id,
            "node_id": self.node_id,
            "depth": int(self.depth),
            "channel": self.channel,
            "joint_propensity": float(self.joint_propensity),
            "joint_propensity_source": self.joint_propensity_source,
            "objective_weight": float(self.objective_weight),
            "ipw_weight": float(self.ipw_weight),
            "effective_weight": float(self.effective_weight),
            "sample_weight": float(self.effective_weight),
            "sample_weight_source": self.sample_weight_source,
            "weighting_mode": self.weighting_mode,
            "discount_gamma": float(self.discount_gamma),
        }
        if self.rl_role is not None:
            payload["rl_role"] = self.rl_role
        if self.local_law_adjustment is not None:
            payload["local_law_adjustment"] = dict(self.local_law_adjustment)
        return payload


def build_treepo_optimizer_export_metadata(
    *,
    fallback_node_id: Optional[str],
    source_example_id: Optional[str],
    source_doc_id: Optional[str],
    source_observation_ids: Sequence[str],
    sampling: Optional[SamplingMetadata],
    law_type: Optional[str],
    supervision_channel_name: Optional[str],
    supervision_signal_name: Optional[str],
    metadata_sources: Sequence[Mapping[str, Any]] = (),
    weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
    rl_role: Optional[TreePORLRole] = None,
    ipw_weight_override: Optional[float] = None,
    joint_propensity_override: Optional[float] = None,
    joint_propensity_source: Optional[str] = None,
    local_law_adjustment: Optional[Mapping[str, Any]] = None,
) -> TreePOOptimizerExportMetadata:
    mode = validate_tree_objective_weighting_mode(weighting_mode)
    gamma = validate_discount_gamma(discount_gamma)
    sources = tuple(_as_mapping(source) for source in metadata_sources)
    depth = infer_treepo_depth(
        *sources,
        _as_mapping(getattr(sampling, "metadata", {})),
    )
    channel = resolve_treepo_channel(
        law_type=law_type,
        supervision_channel_name=supervision_channel_name,
        supervision_signal_name=supervision_signal_name,
    )
    objective_weight = resolve_treepo_objective_weight(
        channel=channel,
        depth=depth,
        weighting_mode=mode,
        discount_gamma=gamma,
    )

    if ipw_weight_override is not None:
        ipw_weight = max(0.0, float(ipw_weight_override))
        if joint_propensity_override is not None:
            joint_propensity = float(joint_propensity_override)
        elif ipw_weight > 0.0:
            joint_propensity = 1.0 / ipw_weight
        else:
            joint_propensity = 1.0
        propensity_source = joint_propensity_source or "derived_from_ipw_weight_override"
    else:
        effective_sampling = sampling or SamplingMetadata()
        joint_propensity = effective_sampling.effective_joint_propensity()
        ipw_weight = effective_sampling.ipw_weight()
        propensity_source = joint_propensity_source or "sampling_metadata"

    return TreePOOptimizerExportMetadata(
        document_id=infer_treepo_document_id(
            source_doc_id=source_doc_id,
            source_example_id=source_example_id,
            metadata_sources=sources,
        ),
        node_id=infer_treepo_node_id(
            fallback_node_id=fallback_node_id,
            source_observation_ids=source_observation_ids,
            metadata_sources=sources,
        ),
        depth=depth,
        channel=channel,
        joint_propensity=float(joint_propensity),
        joint_propensity_source=str(propensity_source),
        objective_weight=float(objective_weight),
        ipw_weight=float(ipw_weight),
        effective_weight=float(objective_weight * ipw_weight),
        weighting_mode=mode,
        discount_gamma=float(gamma),
        rl_role=rl_role,
        local_law_adjustment=(
            dict(local_law_adjustment) if local_law_adjustment is not None else None
        ),
    )


__all__ = [
    "TreePOChannel",
    "TreePOOptimizerExportMetadata",
    "TreePORLRole",
    "TreePOWeightingMode",
    "build_treepo_optimizer_export_metadata",
    "infer_treepo_depth",
    "resolve_treepo_channel",
    "resolve_treepo_objective_weight",
    "validate_discount_gamma",
    "validate_tree_objective_weighting_mode",
]
