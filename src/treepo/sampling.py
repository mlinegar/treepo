"""Sampling metadata, node-audit designs, and inverse-propensity weighting.

Defines ``SamplingMetadata`` and ``DocumentSamplingRow`` with propensity
normalization and IPW weight computation used by preference/observation
exports, plus ``sample_node_audit``/``apply_node_audit`` for choosing which
nodes of a tree receive oracle labels under a logged design.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from treepo.common import (
    MIN_PROPENSITY,
    AuditPolicyName,
    audit_sample_count,
    jsonable as _jsonable,
)


DEFAULT_PROPENSITY = 1.0


class ObservationUnitKind(str, Enum):
    DOCUMENT = "document"
    LEAF = "leaf"
    INTERNAL = "internal"
    MERGE = "merge"
    RESUMMARY = "resummary"
    SUBSTITUTION = "substitution"
    PAIR = "pair"


def _normalize_propensity(value: float | None, name: str) -> float:
    if value is None:
        return DEFAULT_PROPENSITY
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1], got {value!r}")
    return parsed


@dataclass(frozen=True)
class SamplingMetadata:
    """Logged design propensities for one observation.

    ``supports_ipw_estimation`` is a downstream logging flag: exporters set
    it ``False`` for units whose design is described but whose weights must
    stay out of IPW estimators (convenience samples, replays); this package
    records and serializes it.
    """

    document_propensity: float = DEFAULT_PROPENSITY
    unit_propensity: float = DEFAULT_PROPENSITY
    label_propensity: float = DEFAULT_PROPENSITY
    joint_propensity: float | None = None
    sampling_scheme: str = ""
    policy_name: str = ""
    unit_kind: ObservationUnitKind | None = None
    supports_ipw_estimation: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_propensity",
            _normalize_propensity(self.document_propensity, "document_propensity"),
        )
        object.__setattr__(
            self,
            "unit_propensity",
            _normalize_propensity(self.unit_propensity, "unit_propensity"),
        )
        object.__setattr__(
            self,
            "label_propensity",
            _normalize_propensity(self.label_propensity, "label_propensity"),
        )
        if self.joint_propensity is not None:
            object.__setattr__(
                self,
                "joint_propensity",
                _normalize_propensity(self.joint_propensity, "joint_propensity"),
            )
        if self.unit_kind is not None and not isinstance(self.unit_kind, ObservationUnitKind):
            object.__setattr__(self, "unit_kind", ObservationUnitKind(str(self.unit_kind)))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def effective_joint_propensity(self, *, min_propensity: float = MIN_PROPENSITY) -> float:
        if self.joint_propensity is not None:
            joint = float(self.joint_propensity)
        else:
            joint = (
                float(self.document_propensity)
                * float(self.unit_propensity)
                * float(self.label_propensity)
            )
        return max(float(min_propensity), float(joint))

    def ipw_weight(
        self,
        *,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: float | None = None,
    ) -> float:
        weight = 1.0 / self.effective_joint_propensity(min_propensity=min_propensity)
        if max_weight is not None:
            weight = min(weight, float(max_weight))
        return float(weight)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DocumentSamplingRow:
    """One document-level design row covering the full population.

    ``prediction``, ``predicted_var``, and ``fold_id`` are downstream logging
    fields (model predictions and cross-validation fold assignment recorded
    next to the design); this package validates and serializes them.
    """

    top_level_unit_id: str
    observed: bool
    inclusion_probability: float
    prediction: float | None = None
    predicted_var: float | None = None
    truth: float | None = None
    split: str = ""
    fold_id: str = "0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.top_level_unit_id):
            raise ValueError("top_level_unit_id is required")
        p = float(self.inclusion_probability)
        if bool(self.observed):
            _normalize_propensity(p, "inclusion_probability")
        elif not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"inclusion_probability must be in [0, 1], got {p!r}")
        object.__setattr__(self, "top_level_unit_id", str(self.top_level_unit_id))
        object.__setattr__(self, "inclusion_probability", p)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def sampling(self) -> SamplingMetadata:
        return SamplingMetadata(
            document_propensity=max(float(self.inclusion_probability), MIN_PROPENSITY),
            unit_propensity=1.0,
            label_propensity=1.0,
            unit_kind=ObservationUnitKind.DOCUMENT,
            sampling_scheme="document",
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class NodeAuditDesign:
    """A realized uniform node-audit design over one node population.

    ``observed`` marks which nodes receive oracle labels; ``propensity`` is
    the shared inclusion probability ``sample_size / population_size`` every
    row logs, whether or not it was drawn.
    """

    observed: tuple[bool, ...]
    propensity: float
    policy: str
    population_size: int
    sample_size: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def sample_node_audit(
    population_size: int,
    *,
    policy: AuditPolicyName = "all",
    fixed_nodes: int = 0,
    fraction: float = 1.0,
    scale: float = 1.0,
    seed: int = 0,
) -> NodeAuditDesign:
    """Draw a uniform without-replacement node audit under a named policy.

    The policy sets the draw count ``q`` (``all``, ``fixed``, ``fraction``,
    ``sqrt``, ``log2`` via ``audit_sample_count``); every node then has
    inclusion probability ``q / population_size``. Theorem-facing audit rows
    require a positive inclusion probability, so a design that selects zero
    nodes raises.
    """

    n = int(population_size)
    if n <= 0:
        raise ValueError("sample_node_audit requires a positive population_size")
    q = audit_sample_count(
        n, policy=policy, fixed_nodes=fixed_nodes, fraction=fraction, scale=scale
    )
    if q <= 0:
        raise ValueError(
            f"audit policy {policy!r} selected zero of {n} nodes; audit rows "
            "require a positive inclusion probability"
        )
    if q >= n:
        observed = [True] * n
    else:
        rng = random.Random(int(seed))
        drawn = set(rng.sample(range(n), q))
        observed = [idx in drawn for idx in range(n)]
    return NodeAuditDesign(
        observed=tuple(observed),
        propensity=float(q / n),
        policy=str(policy),
        population_size=n,
        sample_size=int(q),
        seed=int(seed),
    )


def apply_node_audit(rows: Sequence[Any], design: NodeAuditDesign) -> tuple[Any, ...]:
    """Apply a node-audit design to fully-labeled local-law audit rows.

    Takes rows that carry oracle losses for every node (for example a
    statistic's ``local_law_rows``) and returns rows that realize the design:
    drawn rows keep their oracle loss and log the design propensity;
    undrawn rows keep only the proxy loss with the same logged propensity.
    Row order must match the design's node population.
    """

    from dataclasses import replace

    from treepo.local_law import LocalLawAuditRow

    row_list = list(rows or ())
    if len(row_list) != design.population_size:
        raise ValueError(
            f"design covers {design.population_size} nodes, got {len(row_list)} rows"
        )
    out = []
    for row, observed in zip(row_list, design.observed):
        if not isinstance(row, LocalLawAuditRow):
            row = LocalLawAuditRow(**dict(row))
        if observed and row.oracle_loss is None:
            raise ValueError(
                f"audit design drew node {row.row_id} but the row has no oracle_loss"
            )
        out.append(
            replace(
                row,
                observed=bool(observed),
                propensity=float(design.propensity),
                effective_propensity=None,
                influence_weight=None,
                oracle_loss=row.oracle_loss if observed else None,
                metadata={
                    **dict(row.metadata or {}),
                    "audit_policy": design.policy,
                    "audit_seed": design.seed,
                    "audit_sample_size": design.sample_size,
                },
            )
        )
    return tuple(out)


__all__ = [
    "DEFAULT_PROPENSITY",
    "MIN_PROPENSITY",
    "DocumentSamplingRow",
    "NodeAuditDesign",
    "ObservationUnitKind",
    "SamplingMetadata",
    "apply_node_audit",
    "sample_node_audit",
]
