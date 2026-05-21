"""Local-law and audit arithmetic — callable from training loops and from
post-hoc analysis.

Ports the explicit formulas that ``treepo_cdx`` introduced (corrected/AIPW
local-law loss, sampled-IPW objective, Kish-style influence-weighted
overlap) into ``treepo.cld`` so the math is *in* this package rather than
in a sibling. Two design choices differ from the cdx port:

1. **Overlap is computed on observed rows only.** cdx's
   ``compute_influence_weighted_overlap`` floors ``effective_propensity``
   to ``MIN_PROPENSITY`` for every row including unobserved ones, which
   lets ``λ/π`` blow up on rows that were never sampled. We restrict the
   sum to ``observed=True`` rows so the overlap reflects the *actual
   sampling population*, not the design population.

2. **Optional in-loop callable.** ``corrected_local_law_loss(...)`` is
   exposed as a plain function so a family's ``train_f`` / ``train_g``
   can call it per-batch. ``local_law_objective_summary`` and
   ``compute_influence_weighted_overlap`` are the post-hoc / evaluator
   surfaces that ``treepo.cld.fit`` itself optionally runs.

The same surface covers both regimes; nothing here imports torch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


MIN_PROPENSITY = 1e-12

_VALID_MODES = ("corrected_local_law", "sampled_ipw")


class LawKind(str, Enum):
    C1_LEAF = "c1_leaf"
    C2_IDEMPOTENCE = "c2_idempotence"
    C3_MERGE = "c3_merge"

    @classmethod
    def from_value(cls, value: "str | LawKind") -> "LawKind":
        if isinstance(value, LawKind):
            return value
        aliases = {
            "c1": cls.C1_LEAF, "leaf": cls.C1_LEAF, "leaf_preservation": cls.C1_LEAF,
            "c2": cls.C2_IDEMPOTENCE, "idempotence": cls.C2_IDEMPOTENCE,
            "on_range_idempotence": cls.C2_IDEMPOTENCE,
            "c3": cls.C3_MERGE, "merge": cls.C3_MERGE, "merge_preservation": cls.C3_MERGE,
        }
        normalized = str(value or "").strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


def _finite(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def normalize_local_law_objective_mode(mode: str) -> str:
    aliases = {
        "corrected": "corrected_local_law",
        "aipw": "corrected_local_law",
        "dr": "corrected_local_law",
        "doubly_robust": "corrected_local_law",
        "ipw": "sampled_ipw",
        "hajek": "sampled_ipw",
        "sampled": "sampled_ipw",
    }
    normalized = str(mode or "corrected_local_law").strip().lower()
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_MODES:
        raise ValueError(
            f"unknown local-law objective mode {mode!r}; expected one of {_VALID_MODES}"
        )
    return normalized


# --------------------------------------------------------------------------- #
# In-loop training signal: per-node corrected-loss formula.
# --------------------------------------------------------------------------- #


def corrected_local_law_loss(
    *,
    proxy_loss: float,
    oracle_loss: float | None,
    observed: bool,
    propensity: float,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """AIPW corrected local-law loss for one node.

    Returns ``proxy_loss`` when the node was not sampled with an oracle
    label; otherwise returns ``proxy + (oracle - proxy) / max(min_pi, pi)``.

    This is the same formula that drives the FNO local-law training in
    ``src/training/supervision/local_law_torch.py``; exposing it here means
    a treepo.cld-managed family can call it without importing the torch
    module (and treepo.cld's own evaluator can use it post-hoc).
    """
    proxy = _finite(proxy_loss, name="proxy_loss")
    if not bool(observed):
        return proxy
    if oracle_loss is None:
        raise ValueError("observed local-law rows require oracle_loss")
    oracle = _finite(oracle_loss, name="oracle_loss")
    pi = _finite(propensity, name="propensity")
    if pi <= 0.0 or pi > 1.0:
        raise ValueError(
            f"observed local-law propensity must be in (0, 1], got {propensity!r}"
        )
    return float(proxy + (oracle - proxy) / max(float(min_propensity), pi, MIN_PROPENSITY))


# --------------------------------------------------------------------------- #
# Rows and the two summary objectives.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LocalLawAuditRow:
    """One node's training/audit row for the local-law objective.

    ``proxy_loss`` is the unobserved-side loss (e.g. cheap closed-form
    score); ``oracle_loss`` is the supervised loss (e.g. oracle target
    error) when the node was sampled (``observed=True``). ``propensity``
    is the design probability of sampling this node; ``node_weight`` is
    the importance weight λ used in the influence-weighted overlap.
    """

    row_id: str
    law_kind: LawKind | str
    proxy_loss: float
    oracle_loss: float | None = None
    observed: bool = False
    propensity: float = 0.0
    node_weight: float = 1.0
    depth: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.row_id):
            raise ValueError("row_id is required")
        law_kind = LawKind.from_value(self.law_kind)
        proxy = _finite(self.proxy_loss, name="proxy_loss")
        oracle = None if self.oracle_loss is None else _finite(self.oracle_loss, name="oracle_loss")
        pi = _finite(self.propensity, name="propensity")
        if pi < 0.0 or pi > 1.0:
            raise ValueError(f"propensity must be in [0, 1], got {self.propensity!r}")
        if bool(self.observed):
            if oracle is None:
                raise ValueError("observed rows require oracle_loss")
            if pi <= 0.0:
                raise ValueError("observed rows require positive propensity")
        node_weight = _finite(self.node_weight, name="node_weight")
        if node_weight < 0.0:
            raise ValueError("node_weight must be non-negative")
        object.__setattr__(self, "row_id", str(self.row_id))
        object.__setattr__(self, "law_kind", law_kind)
        object.__setattr__(self, "proxy_loss", proxy)
        object.__setattr__(self, "oracle_loss", oracle)
        object.__setattr__(self, "propensity", pi)
        object.__setattr__(self, "node_weight", node_weight)
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

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
    """``D_λ = Σ_observed λ²/π``, ``W_λ = max_observed λ/π``, Kish ESS.

    Observed-rows-only (cdx's port included unobserved rows with their
    propensity floored to ``MIN_PROPENSITY``, which lets a single
    unsampled row dominate the overlap. We restrict to observed rows so
    the overlap reflects the actual sampling population.)
    """

    D_lambda: float
    W_lambda: float
    effective_sample_size: float
    n_observed: int
    n_total: int
    influence_total: float
    min_propensity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _depth_weight(depth: int, *, gamma_depth: float) -> float:
    gamma = _finite(gamma_depth, name="gamma_depth")
    if gamma < 0.0:
        raise ValueError("gamma_depth must be non-negative")
    return float(gamma ** int(depth))


def local_law_objective_summary(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    gamma_depth: float = 1.0,
    objective_mode: str = "corrected_local_law",
    min_propensity: float = MIN_PROPENSITY,
) -> LocalLawObjectiveSummary:
    row_list = list(rows)
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
    min_pi = max(float(min_propensity), MIN_PROPENSITY)

    if mode == "sampled_ipw":
        weighted_total = 0.0
        eff_obs_weight = 0.0
        observed_count = 0
        for row in row_list:
            if not bool(row.observed):
                continue
            if row.oracle_loss is None:
                raise ValueError("sampled_ipw observed rows require oracle_loss")
            pi = max(min_pi, float(row.propensity))
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

    # corrected_local_law (AIPW): proxy on unobserved, proxy+(oracle-proxy)/pi on observed.
    weighted_total = 0.0
    weight_sum = 0.0
    eff_obs_weight = 0.0
    observed_count = 0
    for row in row_list:
        w = float(row.node_weight) * _depth_weight(row.depth, gamma_depth=gamma_depth)
        weighted_total += w * corrected_local_law_loss(
            proxy_loss=float(row.proxy_loss),
            oracle_loss=row.oracle_loss,
            observed=bool(row.observed),
            propensity=float(row.propensity),
            min_propensity=min_pi,
        )
        weight_sum += w
        if bool(row.observed):
            observed_count += 1
            eff_obs_weight += w / max(min_pi, float(row.propensity))
    objective = weighted_total / weight_sum if weight_sum > 0.0 else 0.0
    return LocalLawObjectiveSummary(
        objective=float(objective),
        objective_mode=mode,
        row_count=len(row_list),
        observed_count=observed_count,
        weight_sum=float(weight_sum),
        effective_observed_weight_sum=float(eff_obs_weight),
    )


def local_law_objective_summary_by_law_kind(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    gamma_depth: float = 1.0,
    objective_mode: str = "corrected_local_law",
    min_propensity: float = MIN_PROPENSITY,
) -> dict[str, LocalLawObjectiveSummary]:
    """Same arithmetic as :func:`local_law_objective_summary` but
    *decomposed* by ``LawKind`` (``c1_leaf`` / ``c2_idempotence`` /
    ``c3_merge``).

    The user's central f-vs-f* comparison is "how much does f deviate
    from f* at the leaf, idempotence, and merge laws *separately*."
    A single scalar objective hides the answer; a dict by law kind
    surfaces it. Returns one :class:`LocalLawObjectiveSummary` per
    law kind that has at least one row.
    """
    by_kind: dict[str, list[LocalLawAuditRow]] = {}
    for row in rows:
        key = (
            row.law_kind.value if isinstance(row.law_kind, LawKind) else str(row.law_kind)
        )
        by_kind.setdefault(key, []).append(row)
    return {
        kind: local_law_objective_summary(
            kind_rows,
            gamma_depth=gamma_depth,
            objective_mode=objective_mode,
            min_propensity=min_propensity,
        )
        for kind, kind_rows in by_kind.items()
    }


def compute_influence_weighted_overlap_by_law_kind(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> dict[str, InfluenceWeightedAuditOverlap]:
    """D_λ / W_λ / Kish ESS decomposed by ``LawKind``."""
    by_kind: dict[str, list[LocalLawAuditRow]] = {}
    for row in rows:
        key = (
            row.law_kind.value if isinstance(row.law_kind, LawKind) else str(row.law_kind)
        )
        by_kind.setdefault(key, []).append(row)
    return {
        kind: compute_influence_weighted_overlap(kind_rows, min_propensity=min_propensity)
        for kind, kind_rows in by_kind.items()
    }


def compute_influence_weighted_overlap(
    rows: Sequence[LocalLawAuditRow] | Iterable[LocalLawAuditRow],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> InfluenceWeightedAuditOverlap:
    row_list = list(rows)
    min_pi = max(float(min_propensity), MIN_PROPENSITY)
    ratios: list[float] = []
    design_effect = 0.0
    worst = 0.0
    influence_total = 0.0
    observed_count = 0
    for row in row_list:
        if not bool(row.observed):
            continue
        lam = float(row.node_weight)
        pi = max(min_pi, float(row.propensity))
        if lam < 0.0:
            raise ValueError("node_weight (influence weight) must be non-negative")
        if pi <= 0.0 or pi > 1.0:
            raise ValueError("observed rows require propensity in (0, 1]")
        ratio = lam / pi
        ratios.append(ratio)
        design_effect += (lam * lam) / pi
        worst = max(worst, ratio)
        influence_total += lam
        observed_count += 1
    total_ratio = float(sum(ratios))
    denom = float(sum(r * r for r in ratios))
    ess = (total_ratio * total_ratio / denom) if denom > 0.0 else 0.0
    return InfluenceWeightedAuditOverlap(
        D_lambda=float(design_effect),
        W_lambda=float(worst),
        effective_sample_size=float(ess),
        n_observed=int(observed_count),
        n_total=len(row_list),
        influence_total=float(influence_total),
        min_propensity=float(min_pi),
    )


__all__ = [
    "InfluenceWeightedAuditOverlap",
    "LawKind",
    "LocalLawAuditRow",
    "LocalLawObjectiveSummary",
    "MIN_PROPENSITY",
    "compute_influence_weighted_overlap",
    "compute_influence_weighted_overlap_by_law_kind",
    "corrected_local_law_loss",
    "local_law_objective_summary",
    "local_law_objective_summary_by_law_kind",
    "normalize_local_law_objective_mode",
]
