"""Config dataclasses and coercion for neural-operator families.

This is machinery, not data work: it owns the operator-kind vocabulary, the
``NeuralOperatorFamilyConfig``/``FNOFamilyConfig`` shapes, and the helpers that
coerce arbitrary ``backend_config`` payloads into a validated config. It sits
near the bottom of the FNO module DAG and depends only on the tensor-agnostic
``_fno_loss`` leaf among its FNO siblings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping, TypeVar

from treepo.common import stable_digest
from treepo.methods._family_config import dataclass_field_subset
from treepo.methods._fno_loss import (
    SUM_L1,
    normalize_training_loss,
    training_loss_definition,
)

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
    #: AdamW weight decay and gradient-norm clip — the TT-ladder training
    #: conventions the parity anchors were recorded under.
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 1.0
    epochs_per_iteration: int = 8
    batch_size: int = 8
    #: Trees per forward/backward chunk inside one optimizer step. Exact
    #: gradient accumulation: the accumulated loss equals the full-batch loss,
    #: so training dynamics are identical to ``batch_size`` — only peak
    #: activation memory shrinks (the embedding-axis FNO at large leaf counts
    #: does not fit a 24GB MIG slice at batch 16 otherwise). ``None`` keeps
    #: single-pass batches. Law-bearing objectives are not supported yet.
    micro_batch_size: int | None = None
    #: Trees per no-grad forward chunk at scoring time (always exact).
    eval_batch_size: int = 32
    #: TT-ladder loss shape: per-tree convex split between the configured root
    #: point loss and the leaf/merge-weighted node point loss. Every tree
    #: contributes equally regardless of node count. ``None`` keeps the pooled
    #: weighted node mean. Mutually exclusive with an ObjectiveSpec.
    per_tree_loss_lambda: float | None = None
    seed: int = 0
    device: str = "cpu"
    normalize_targets: bool = True
    #: Per-example target/readout distance used for root supervision, node-f
    #: supervision, and numeric/exact-state law rows. ``sum_l1`` is the common
    #: K=1/3/57 default: sum_j |prediction_j-target_j|, without dividing by K.
    #: ``coordinate_mean_mse`` is an explicit legacy-compatibility opt-in.
    training_loss: str = SUM_L1
    target_key: str | None = None
    target_vector_key: str | None = None
    target_dim: int | None = None
    #: Canonical coordinate order for one joint vector readout and the
    #: corresponding versioned oracle provenance. All coordinates share g.
    target_names: tuple[str, ...] = ()
    target_oracle_ids: tuple[str, ...] = ()
    #: Optional metadata-backed exact-state witness for law-bearing objectives.
    #: Unlike ``target_vector_key``, this does not change the task/root output
    #: width: the vector supervises the first ``law_state_target_dim`` hidden
    #: trace coordinates while the ordinary root readout remains scalar (or
    #: whatever ``target_key`` requests). Missing node keys are unobserved.
    law_state_target_key: str | None = None
    law_state_target_dim: int | None = None
    #: Optional fixed task readout from the learned state witness.  The
    #: ``"first_coordinate"`` mode returns ``state[..., 0:1]`` directly and
    #: excludes the learned readout head from optimization.  This gives
    #: root-only, local-only, and combined allocations one common scalar
    #: decoder when coordinate zero is the task target itself.  Legacy learned
    #: readout behavior remains the default (``None``).
    law_state_root_readout: str | None = None
    #: Inclusion probabilities for a persistent sampled state-label design.
    #: They are consumed by the canonical sampled-IPW objective; use 1.0 for
    #: dense/exact rows. Leaves are C1 and every canonical merge, including the
    #: root merge, is C3.
    law_state_leaf_propensity: float = 1.0
    law_state_merge_propensity: float = 1.0
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
    #: Optional pinned document ids whose root targets may enter the root
    #: loss. ``None`` means every training tree has an observed root (legacy
    #: behavior); an explicit empty sequence means no root is observed. Trees
    #: outside the mask remain available for leaf/merge supervision.
    root_observed_doc_ids: Any = None
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
    # A task fit fragment may carry the corresponding DSPy prompt contract so
    # the identical fragment can be spread directly into either family. The
    # neural operator records these values as configuration provenance but
    # never interprets them or changes its architecture/objective from them.
    f_signature_instructions: str | None = None
    g_signature_instructions: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.training_loss = normalize_training_loss(self.training_loss)
        if not math.isfinite(float(self.learning_rate)) or float(self.learning_rate) < 0.0:
            raise ValueError(f"learning_rate must be finite and >= 0; got {self.learning_rate!r}")
        target_names = tuple(str(value).strip() for value in (self.target_names or ()))
        oracle_ids = tuple(str(value).strip() for value in (self.target_oracle_ids or ()))
        if any(not value for value in target_names):
            raise ValueError("target_names must contain only non-empty identifiers")
        if len(target_names) != len(set(target_names)):
            raise ValueError("target_names must be unique and preserve declared order")
        if any(not value for value in oracle_ids):
            raise ValueError("target_oracle_ids must contain only non-empty identifiers")
        if oracle_ids and len(oracle_ids) != len(target_names):
            raise ValueError("target_oracle_ids must align one-for-one with target_names")
        if target_names and not oracle_ids:
            raise ValueError("named vector targets require target_oracle_ids provenance")
        if not target_names and oracle_ids:
            raise ValueError("target_oracle_ids requires target_names")
        if target_names:
            if self.target_dim is not None and int(self.target_dim) != len(target_names):
                raise ValueError(
                    f"target_dim={self.target_dim!r} conflicts with "
                    f"len(target_names)={len(target_names)}"
                )
            self.target_dim = len(target_names)
        self.target_names = target_names
        self.target_oracle_ids = oracle_ids

        readout = (
            None
            if self.law_state_root_readout is None
            else str(self.law_state_root_readout).strip().lower()
        )
        if readout not in {None, "first_coordinate"}:
            raise ValueError(
                "law_state_root_readout must be None or 'first_coordinate', "
                f"got {self.law_state_root_readout!r}"
            )
        self.law_state_root_readout = readout
        if readout is not None and len(target_names) > 1:
            raise ValueError(
                "law_state_root_readout='first_coordinate' is incompatible with "
                "a multi-coordinate joint oracle readout"
            )
        if readout is not None and str(self.root_readout) != "root_state":
            raise ValueError(
                "law_state_root_readout requires root_readout='root_state'; "
                "a fixed state-coordinate decoder is not a leaf rollup"
            )
        if self.per_tree_loss_lambda is not None:
            value = float(self.per_tree_loss_lambda)
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    "per_tree_loss_lambda must be in [0, 1] for a convex split, "
                    f"got {self.per_tree_loss_lambda!r}"
                )


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


def _target_schema_payload(config: NeuralOperatorFamilyConfig) -> dict[str, Any] | None:
    names = tuple(config.target_names or ())
    if not names:
        return None
    oracle_ids = tuple(config.target_oracle_ids or ())
    payload: dict[str, Any] = {
        "definition": "single_shared_g_joint_vector_f_star",
        "target_order": list(names),
        "targets": [
            {"target_name": name, "oracle_id": oracle_id}
            for name, oracle_id in zip(names, oracle_ids)
        ],
        "training_loss": str(config.training_loss),
        "training_loss_definition": training_loss_definition(config.training_loss),
        "coordinate_reduction": ("sum_not_mean" if str(config.training_loss) == SUM_L1 else "mean"),
    }
    payload["schema_digest"] = stable_digest(payload)
    return payload


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
    "_target_schema_payload",
    "_tensor_payload",
]
