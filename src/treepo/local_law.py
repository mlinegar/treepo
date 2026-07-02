"""Canonical scalar local-law and audit-row arithmetic.

This module is the public, Lean-aligned home for C1/C2/C3 row validation,
corrected local-law losses, and influence-weighted audit overlap. Training
modules may wrap these helpers in tensors, but theorem-facing rows should use
the dataclasses here so that design propensities and all-row overlap semantics
do not drift across package surfaces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from treepo.common import MIN_PROPENSITY, finite_float


LOCAL_LAW_OBJECTIVE_CORRECTED = "corrected_local_law"
LOCAL_LAW_OBJECTIVE_SAMPLED_IPW = "sampled_ipw"
LOCAL_LAW_OBJECTIVE_MODES = (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
)


class LawKind(str, Enum):
    C1_LEAF = "leaf_preservation"
    C2_IDEMPOTENCE = "on_range_idempotence"
    C3_MERGE = "merge_preservation"

    @classmethod
    def from_value(cls, value: str | "LawKind") -> "LawKind":
        if isinstance(value, LawKind):
            return value
        normalized = str(value or "").strip().lower()
        if normalized in LAW_KIND_ALIASES:
            return LAW_KIND_ALIASES[normalized]
        return cls(normalized)


# Two spellings per law: the paper's C-conditions and the Lean L-numbering
# (L1=leaf, L2=merge, L3=idempotence). The enum values are the canonical
# long names.
LAW_KIND_ALIASES: dict[str, LawKind] = {
    "c1": LawKind.C1_LEAF,
    "l1": LawKind.C1_LEAF,
    "c2": LawKind.C2_IDEMPOTENCE,
    "l3": LawKind.C2_IDEMPOTENCE,
    "c3": LawKind.C3_MERGE,
    "l2": LawKind.C3_MERGE,
}

def normalize_local_law_objective_mode(mode: str) -> str:
    normalized = str(mode or LOCAL_LAW_OBJECTIVE_CORRECTED).strip().lower()
    if normalized == "corrected":
        normalized = LOCAL_LAW_OBJECTIVE_CORRECTED
    if normalized not in LOCAL_LAW_OBJECTIVE_MODES:
        raise ValueError(
            f"unknown local-law objective mode {mode!r}; expected one of "
            f"{LOCAL_LAW_OBJECTIVE_MODES}"
        )
    return normalized


def corrected_local_law_loss(
    *,
    proxy_loss: float,
    oracle_loss: float | None,
    observed: bool,
    propensity: float,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Return ``proxy + R / pi * (oracle - proxy)`` for one scalar row."""

    proxy = finite_float(proxy_loss, name="proxy_loss")
    if not bool(observed):
        return proxy
    if oracle_loss is None:
        raise ValueError("observed corrected local-law rows require oracle_loss")
    oracle = finite_float(oracle_loss, name="oracle_loss")
    pi = finite_float(propensity, name="propensity")
    min_pi = max(finite_float(min_propensity, name="min_propensity"), MIN_PROPENSITY)
    if pi <= 0.0 or pi > 1.0:
        raise ValueError(f"observed local-law propensity must be in (0, 1], got {propensity!r}")
    return float(proxy + (oracle - proxy) / max(min_pi, pi))


@dataclass(frozen=True)
class LocalLawAuditRow:
    """One theorem-facing local-law audit row.

    ``propensity`` is the logged design inclusion probability and must be
    positive for every row, regardless of whether ``observed`` is true.
    ``observed`` is only the realized sampling indicator. ``effective_propensity``
    is the clipped numerical value used in denominators and may differ from the
    logged design value only when clipping metadata is recorded by the caller.
    """

    row_id: str
    law_kind: LawKind | str
    proxy_loss: float
    oracle_loss: float | None = None
    observed: bool = False
    propensity: float = 1.0
    effective_propensity: float | None = None
    node_weight: float = 1.0
    depth: int = 0
    influence_weight: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.row_id):
            raise ValueError("local-law row_id is required")
        law_kind = LawKind.from_value(self.law_kind)
        proxy = finite_float(self.proxy_loss, name="proxy_loss")
        oracle = None if self.oracle_loss is None else finite_float(self.oracle_loss, name="oracle_loss")
        prop = finite_float(self.propensity, name="propensity")
        if prop <= 0.0 or prop > 1.0:
            raise ValueError(f"propensity must be in (0, 1], got {self.propensity!r}")
        if bool(self.observed) and oracle is None:
            raise ValueError("observed local-law rows require oracle_loss")
        effective = (
            max(prop, MIN_PROPENSITY)
            if self.effective_propensity is None
            else finite_float(self.effective_propensity, name="effective_propensity")
        )
        if effective <= 0.0 or effective > 1.0:
            raise ValueError(f"effective_propensity must be in (0, 1], got {effective!r}")
        node_weight = finite_float(self.node_weight, name="node_weight")
        if node_weight < 0.0:
            raise ValueError("node_weight must be non-negative")
        influence = (
            node_weight / effective
            if self.influence_weight is None
            else finite_float(self.influence_weight, name="influence_weight")
        )
        if influence < 0.0:
            raise ValueError("influence_weight must be non-negative")
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
        return corrected_local_law_loss(
            proxy_loss=float(self.proxy_loss),
            oracle_loss=self.oracle_loss,
            observed=bool(self.observed),
            propensity=float(self.propensity),
            min_propensity=float(min_propensity),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["law_kind"] = self.law_kind.value
        return payload


@dataclass(frozen=True)
class LocalLawObjectiveSummary:
    objective: float
    objective_mode: str
    row_count: int
    observed_count: int
    weight_sum: float
    effective_observed_weight_sum: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InfluenceWeightedAuditOverlap:
    """All-row influence overlap plus explicitly named observed diagnostics."""

    D_lambda: float
    W_lambda: float
    effective_sample_size: float
    observed_effective_sample_size: float
    max_weight: float
    n_rows: int
    n_observed: int
    n_total: int
    influence_total: float
    observed_influence_total: float
    min_propensity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _depth_weight(depth: int, *, gamma_depth: float) -> float:
    gamma = finite_float(gamma_depth, name="gamma_depth")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma_depth must be in [0, 1], got {gamma_depth!r}")
    return float(gamma ** int(depth))


def local_law_objective_summary(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    gamma_depth: float = 1.0,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    min_propensity: float = MIN_PROPENSITY,
) -> LocalLawObjectiveSummary:
    row_list = [_coerce_row(row) for row in rows]
    mode = normalize_local_law_objective_mode(objective_mode)
    if not row_list:
        return LocalLawObjectiveSummary(
            objective=0.0,
            objective_mode=mode,
            row_count=0,
            observed_count=0,
            weight_sum=0.0,
            effective_observed_weight_sum=0.0,
        )
    min_pi = max(finite_float(min_propensity, name="min_propensity"), MIN_PROPENSITY)

    if mode == LOCAL_LAW_OBJECTIVE_SAMPLED_IPW:
        weighted_total = 0.0
        eff_obs_weight = 0.0
        observed_count = 0
        for row in row_list:
            if not bool(row.observed):
                continue
            if row.oracle_loss is None:
                raise ValueError("sampled_ipw observed rows require oracle_loss")
            pi = max(min_pi, float(row.effective_propensity))
            w = float(row.node_weight) * _depth_weight(row.depth, gamma_depth=gamma_depth)
            ipw_w = w / pi
            weighted_total += ipw_w * float(row.oracle_loss)
            eff_obs_weight += ipw_w
            observed_count += 1
        objective = weighted_total / eff_obs_weight if eff_obs_weight > 0.0 else 0.0
        return LocalLawObjectiveSummary(
            objective=float(objective),
            objective_mode=mode,
            row_count=len(row_list),
            observed_count=observed_count,
            weight_sum=float(sum(float(r.node_weight) for r in row_list)),
            effective_observed_weight_sum=float(eff_obs_weight),
        )

    weighted_total = 0.0
    weight_sum = 0.0
    eff_obs_weight = 0.0
    observed_count = 0
    for row in row_list:
        w = float(row.node_weight) * _depth_weight(row.depth, gamma_depth=gamma_depth)
        weighted_total += w * row.corrected_loss(min_propensity=min_pi)
        weight_sum += w
        if bool(row.observed):
            observed_count += 1
            eff_obs_weight += w / max(min_pi, float(row.effective_propensity))
    objective = weighted_total / weight_sum if weight_sum > 0.0 else 0.0
    return LocalLawObjectiveSummary(
        objective=float(objective),
        objective_mode=mode,
        row_count=len(row_list),
        observed_count=observed_count,
        weight_sum=float(weight_sum),
        effective_observed_weight_sum=float(eff_obs_weight),
    )


def _local_law_objective_summary_by_law_kind(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    gamma_depth: float = 1.0,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    min_propensity: float = MIN_PROPENSITY,
) -> dict[str, LocalLawObjectiveSummary]:
    by_kind: dict[str, list[LocalLawAuditRow]] = {}
    for row in rows:
        coerced = _coerce_row(row)
        by_kind.setdefault(coerced.law_kind.value, []).append(coerced)
    return {
        kind: local_law_objective_summary(
            kind_rows,
            gamma_depth=gamma_depth,
            objective_mode=objective_mode,
            min_propensity=min_propensity,
        )
        for kind, kind_rows in by_kind.items()
    }


def audit_local_laws(
    rows: Sequence[LocalLawAuditRow | Mapping[str, Any]] | Iterable[LocalLawAuditRow | Mapping[str, Any]],
    *,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    gamma_depth: float = 1.0,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    row_list: list[LocalLawAuditRow] = []
    for item in rows:
        if isinstance(item, LocalLawAuditRow):
            row_list.append(item)
        elif isinstance(item, Mapping):
            row_list.append(LocalLawAuditRow(**dict(item)))
        else:
            raise TypeError(f"audit rows must be LocalLawAuditRow or mappings; got {type(item).__name__}")
    if not row_list:
        raise ValueError("audit_local_laws requires at least one row")
    summary = local_law_objective_summary(row_list, objective_mode=objective_mode, gamma_depth=gamma_depth)
    overlap = compute_influence_weighted_overlap(row_list)
    by_kind_summaries = _local_law_objective_summary_by_law_kind(
        row_list,
        objective_mode=objective_mode,
        gamma_depth=gamma_depth,
    )
    by_kind_overlaps = _compute_influence_weighted_overlap_by_law_kind(row_list)
    payload = {
        "status": "success",
        "local_law_objective": summary.to_dict(),
        "influence_weighted_overlap": overlap.to_dict(),
        "by_law_kind": {
            kind: {
                "local_law_objective": by_kind_summaries[kind].to_dict(),
                "influence_weighted_overlap": by_kind_overlaps[kind].to_dict(),
            }
            for kind in sorted(by_kind_summaries)
        },
        "n_rows": len(row_list),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def compute_influence_weighted_overlap(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> InfluenceWeightedAuditOverlap:
    row_list = [_coerce_row(row) for row in rows]
    min_pi = max(finite_float(min_propensity, name="min_propensity"), MIN_PROPENSITY)
    ratios: list[float] = []
    observed_ratios: list[float] = []
    design_effect = 0.0
    worst = 0.0
    influence_total = 0.0
    observed_influence_total = 0.0
    observed_count = 0
    for row in row_list:
        lam = float(row.node_weight)
        pi = max(min_pi, float(row.effective_propensity))
        if lam < 0.0:
            raise ValueError("node_weight (influence weight) must be non-negative")
        if pi <= 0.0 or pi > 1.0:
            raise ValueError("effective propensities must be in (0, 1]")
        ratio = lam / pi
        ratios.append(ratio)
        design_effect += (lam * lam) / pi
        worst = max(worst, ratio)
        influence_total += lam
        if bool(row.observed):
            observed_ratios.append(ratio)
            observed_influence_total += lam
            observed_count += 1
    return InfluenceWeightedAuditOverlap(
        D_lambda=float(design_effect),
        W_lambda=float(worst),
        effective_sample_size=_kish_ess(ratios),
        observed_effective_sample_size=_kish_ess(observed_ratios),
        max_weight=float(worst),
        n_rows=len(row_list),
        n_observed=int(observed_count),
        n_total=len(row_list),
        influence_total=float(influence_total),
        observed_influence_total=float(observed_influence_total),
        min_propensity=float(min_pi),
    )


def _compute_influence_weighted_overlap_by_law_kind(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> dict[str, InfluenceWeightedAuditOverlap]:
    by_kind: dict[str, list[LocalLawAuditRow]] = {}
    for row in rows:
        coerced = _coerce_row(row)
        by_kind.setdefault(coerced.law_kind.value, []).append(coerced)
    return {
        kind: compute_influence_weighted_overlap(kind_rows, min_propensity=min_propensity)
        for kind, kind_rows in by_kind.items()
    }


def _kish_ess(ratios: Sequence[float]) -> float:
    total = float(sum(ratios))
    denom = float(sum(float(r) * float(r) for r in ratios))
    return float((total * total / denom) if denom > 0.0 else 0.0)


def _coerce_row(row: LocalLawAuditRow | Mapping[str, Any]) -> LocalLawAuditRow:
    if isinstance(row, LocalLawAuditRow):
        return row
    if isinstance(row, Mapping):
        return LocalLawAuditRow(**dict(row))
    raise TypeError(f"local-law row must be LocalLawAuditRow or mapping, got {type(row).__name__}")


__all__ = [
    "InfluenceWeightedAuditOverlap",
    "LAW_KIND_ALIASES",
    "LOCAL_LAW_OBJECTIVE_CORRECTED",
    "LOCAL_LAW_OBJECTIVE_MODES",
    "LOCAL_LAW_OBJECTIVE_SAMPLED_IPW",
    "LawKind",
    "LocalLawAuditRow",
    "LocalLawObjectiveSummary",
    "MIN_PROPENSITY",
    "compute_influence_weighted_overlap",
    "audit_local_laws",
    "corrected_local_law_loss",
    "local_law_objective_summary",
    "normalize_local_law_objective_mode",
]
