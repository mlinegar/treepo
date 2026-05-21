from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


MIN_PROPENSITY = 1e-12


class LawKind(str, Enum):
    C1_LEAF = "c1_leaf"
    C2_IDEMPOTENCE = "c2_idempotence"
    C3_MERGE = "c3_merge"

    @classmethod
    def from_value(cls, value: str | "LawKind") -> "LawKind":
        if isinstance(value, LawKind):
            return value
        aliases = {
            "c1": cls.C1_LEAF,
            "l1": cls.C1_LEAF,
            "leaf": cls.C1_LEAF,
            "leaf_preservation": cls.C1_LEAF,
            "c2": cls.C2_IDEMPOTENCE,
            "l3": cls.C2_IDEMPOTENCE,
            "idempotence": cls.C2_IDEMPOTENCE,
            "on_range_idempotence": cls.C2_IDEMPOTENCE,
            "c3": cls.C3_MERGE,
            "l2": cls.C3_MERGE,
            "merge": cls.C3_MERGE,
            "merge_preservation": cls.C3_MERGE,
        }
        normalized = str(value or "").strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


def _finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class LocalLawAuditRow:
    row_id: str
    law_kind: LawKind | str
    proxy_loss: float
    oracle_loss: float | None = None
    observed: bool = False
    propensity: float = 0.0
    effective_propensity: float | None = None
    node_weight: float = 1.0
    depth: int = 0
    influence_weight: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.row_id):
            raise ValueError("local-law row_id is required")
        law_kind = LawKind.from_value(self.law_kind)
        proxy = _finite_float(self.proxy_loss, name="proxy_loss")
        oracle = None if self.oracle_loss is None else _finite_float(self.oracle_loss, name="oracle_loss")
        prop = _finite_float(self.propensity, name="propensity")
        if prop < 0.0 or prop > 1.0:
            raise ValueError(f"propensity must be in [0, 1], got {self.propensity!r}")
        if bool(self.observed):
            if oracle is None:
                raise ValueError("observed local-law rows require oracle_loss")
            if prop <= 0.0:
                raise ValueError("observed local-law rows require positive propensity")
        effective = max(prop, MIN_PROPENSITY) if self.effective_propensity is None else _finite_float(
            self.effective_propensity,
            name="effective_propensity",
        )
        if effective <= 0.0 or effective > 1.0:
            raise ValueError(f"effective_propensity must be in (0, 1], got {effective!r}")
        node_weight = _finite_float(self.node_weight, name="node_weight")
        if node_weight < 0.0:
            raise ValueError("node_weight must be non-negative")
        influence = (
            node_weight / effective
            if self.influence_weight is None
            else _finite_float(self.influence_weight, name="influence_weight")
        )
        object.__setattr__(self, "row_id", str(self.row_id))
        object.__setattr__(self, "law_kind", law_kind)
        object.__setattr__(self, "proxy_loss", proxy)
        object.__setattr__(self, "oracle_loss", oracle)
        object.__setattr__(self, "propensity", prop)
        object.__setattr__(self, "effective_propensity", effective)
        object.__setattr__(self, "node_weight", node_weight)
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "influence_weight", influence)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def corrected_loss(self, *, min_propensity: float = MIN_PROPENSITY) -> float:
        proxy = float(self.proxy_loss)
        if not bool(self.observed):
            return proxy
        oracle = float(self.oracle_loss)
        pi = max(float(min_propensity), float(self.propensity))
        return float(proxy + (oracle - proxy) / pi)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["law_kind"] = self.law_kind.value
        return payload


@dataclass(frozen=True)
class InfluenceWeightedAuditOverlap:
    D_lambda: float
    W_lambda: float
    effective_sample_size: float
    max_weight: float
    n_rows: int
    influence_total: float
    min_propensity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_influence_weighted_overlap(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> InfluenceWeightedAuditOverlap:
    row_list = list(rows)
    min_pi = max(float(min_propensity), MIN_PROPENSITY)
    weighted_ratios: list[float] = []
    lambda_values: list[float] = []
    design_effect = 0.0
    worst = 0.0
    for row in row_list:
        lam = float(row.node_weight)
        pi = max(min_pi, float(row.effective_propensity))
        if lam < 0.0:
            raise ValueError("influence weights must be non-negative")
        if pi <= 0.0 or pi > 1.0:
            raise ValueError("effective propensities must be in (0, 1]")
        ratio = lam / pi
        design_effect += (lam * lam) / pi
        worst = max(worst, ratio)
        weighted_ratios.append(ratio)
        lambda_values.append(lam)
    total_ratio = float(sum(weighted_ratios))
    denom = float(sum(w * w for w in weighted_ratios))
    ess = (total_ratio * total_ratio / denom) if denom > 0.0 else 0.0
    return InfluenceWeightedAuditOverlap(
        D_lambda=float(design_effect),
        W_lambda=float(worst),
        effective_sample_size=float(ess),
        max_weight=float(worst),
        n_rows=len(row_list),
        influence_total=float(sum(lambda_values)),
        min_propensity=float(min_pi),
    )


def corrected_losses_from_rows(rows: Sequence[LocalLawAuditRow]) -> list[float]:
    return [row.corrected_loss() for row in rows]


__all__ = [
    "InfluenceWeightedAuditOverlap",
    "LawKind",
    "LocalLawAuditRow",
    "compute_influence_weighted_overlap",
    "corrected_losses_from_rows",
]
