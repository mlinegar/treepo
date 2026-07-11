"""Config dataclasses and coercion for neural-operator families.

This is machinery, not data work: it owns the operator-kind vocabulary, the
``NeuralOperatorFamilyConfig``/``FNOFamilyConfig`` shapes, and the helpers that
coerce arbitrary ``backend_config`` payloads into a validated config. It sits at
the bottom of the FNO module DAG and imports nothing from its siblings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping, TypeVar

from treepo.methods._family_config import dataclass_field_subset

_LOCAL_OPERATOR_KINDS = frozenset({"conv1d"})


@dataclass
class NeuralOperatorFamilyConfig:
    """Config for the generic neural-operator root-score family."""

    operator_kind: str = "fno"
    embedding_dim: int = 32
    hidden_channels: int = 16
    n_modes: int = 8
    n_layers: int = 2
    conv_kernel_size: int = 3
    head_hidden_dim: int = 32
    operator_kwargs: Mapping[str, Any] = field(default_factory=dict)
    learning_rate: float = 1e-3
    epochs_per_iteration: int = 8
    batch_size: int = 8
    seed: int = 0
    device: str = "cpu"
    normalize_targets: bool = True
    target_key: str | None = None
    target_vector_key: str | None = None
    target_dim: int | None = None
    target_min: float | None = None
    target_max: float | None = None
    embedding_salt: str = "treepo_neural_operator"
    numeric_transition_state_weight: float = 0.0
    numeric_transition_count_scale: float | None = None
    # Per-node supervision weights (fit-grid plan Phase 1). Relative weighting
    # of the root term vs supervised leaf/merge node terms; defaults preserve
    # the historical root-only loss exactly. Named levels in
    # treepo.methods._supervision map onto these three knobs.
    root_weight: float = 1.0
    leaf_weight: float = 0.0
    merge_weight: float = 0.0
    #: Node metadata key holding the per-node target. When unset, a node's
    #: ``label`` attribute is read first, then metadata ``score`` /
    #: ``oracle_score``.
    node_target_key: str | None = None
    #: With ``node_target_exclusive`` set, ONLY ``node_target_key`` is read —
    #: nodes without it stay unobserved instead of falling back to ``label`` /
    #: ``score``. The distilled label mix sets this so a cached-teacher cell
    #: never silently trains on gold node labels where the cache has gaps.
    node_target_exclusive: bool = False
    #: Optional pinned ``doc_id::node_id`` unit ids whose leaf labels may be
    #: consumed (the gold_fraction grid axis); leaves outside the set are
    #: treated as unlabeled. ``None`` consumes every labeled leaf.
    supervised_node_units: Any = None
    #: How the tree-level prediction is read out. ``"root_state"`` (default)
    #: applies the readout to the composed root state. ``"leaf_mean"`` is the
    #: additive rollup: the weighted mean of per-leaf readouts — exact for
    #: additive targets like RILE, where the document value IS the
    #: (qsentence-weighted) mean of local values.
    root_readout: str = "root_state"
    #: Leaf-metadata key holding each leaf's rollup weight (e.g.
    #: ``"total_non_header_qsentences"`` for sentence-scale manifesto bundles).
    #: ``None`` = equal weights (exact for single-qsentence leaves). Only
    #: meaningful with ``root_readout="leaf_mean"``.
    rollup_weight_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class FNOFamilyConfig(NeuralOperatorFamilyConfig):
    """Config for the concrete FNO route."""

    operator_kind: str = "fno"
    embedding_salt: str = "treepo_fno"


_ConfigT = TypeVar("_ConfigT", bound=NeuralOperatorFamilyConfig)


def _coerce_config(
    raw: Any,
    backend_config: Mapping[str, Any],
    *,
    config_cls: type[_ConfigT],
) -> _ConfigT:
    if isinstance(raw, config_cls):
        base = raw
    elif isinstance(raw, NeuralOperatorFamilyConfig):
        base = config_cls(**_known_config_keys(_config_payload(raw), config_cls=config_cls))
    elif isinstance(raw, Mapping):
        base = config_cls(**_known_config_keys(raw, config_cls=config_cls))
    elif is_dataclass(raw):
        base = config_cls(**_known_config_keys(getattr(raw, "__dict__", {}), config_cls=config_cls))
    elif raw is None:
        base = config_cls(**_known_config_keys(backend_config, config_cls=config_cls))
    else:
        raise TypeError(
            "backend_config config must be NeuralOperatorFamilyConfig "
            f"or mapping; got {type(raw).__name__}"
        )
    overrides = _known_config_keys(backend_config, config_cls=config_cls)
    if not overrides:
        return base
    data = _config_payload(base)
    data.update(overrides)
    return config_cls(**_known_config_keys(data, config_cls=config_cls))


def _known_config_keys(values: Mapping[str, Any], *, config_cls: type[Any]) -> dict[str, Any]:
    # Field filtering is the shared family-config mechanism; this module only
    # adds the dataclass-instance conversion paths in _coerce_config above.
    return dataclass_field_subset(values, config_cls)


def _config_payload(config: NeuralOperatorFamilyConfig) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def _tensor_payload(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
    except Exception:
        return None


def _normalize_operator_kind(value: Any) -> str:
    return str(value or "fno").strip().lower().replace("-", "_")


def _clamp(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None:
        value = max(float(lower), value)
    if upper is not None:
        value = min(float(upper), value)
    return float(value)


__all__ = [
    "FNOFamilyConfig",
    "NeuralOperatorFamilyConfig",
    "_LOCAL_OPERATOR_KINDS",
    "_clamp",
    "_coerce_config",
    "_config_payload",
    "_known_config_keys",
    "_normalize_operator_kind",
    "_tensor_payload",
]
