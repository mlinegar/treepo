"""Corrected local-law objective helpers.

This module implements the scalar DSL/AIPW adjustment used by the paper:

    corrected = proxy + R / pi * (oracle - proxy)

The helpers are deliberately small and dependency-free so they can be reused by
DSPy metrics, report code, and simulation diagnostics. Torch-specific gradient
helpers live under ``src.training.supervision``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Sequence


MIN_PROPENSITY = 1e-12
LOCAL_LAW_OBJECTIVE_CORRECTED = "corrected_local_law"
LOCAL_LAW_OBJECTIVE_SAMPLED_IPW = "sampled_ipw"
VALID_LOCAL_LAW_OBJECTIVE_MODES: tuple[str, ...] = (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
)


def _finite_float(value: Any, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def depth_discount(gamma_depth: float, depth: int) -> float:
    """Return the fixed tree-depth discount ``gamma_depth ** depth``."""

    gamma = _finite_float(gamma_depth, name="gamma_depth")
    d = int(depth)
    if d < 0:
        raise ValueError(f"depth must be non-negative, got {depth!r}")
    return float(gamma**d)


def normalize_local_law_objective_mode(mode: str | None) -> str:
    """Return the canonical local-law objective mode name."""

    normalized = str(mode or LOCAL_LAW_OBJECTIVE_CORRECTED).strip().lower()
    if normalized in {"corrected", "dr", "aipw", "doubly_robust"}:
        normalized = LOCAL_LAW_OBJECTIVE_CORRECTED
    if normalized in {"ipw", "sampled", "hajek", "sampled_hajek"}:
        normalized = LOCAL_LAW_OBJECTIVE_SAMPLED_IPW
    if normalized not in VALID_LOCAL_LAW_OBJECTIVE_MODES:
        raise ValueError(
            f"local_law_objective_mode={mode!r} unsupported; expected one of "
            f"{VALID_LOCAL_LAW_OBJECTIVE_MODES}"
        )
    return normalized


def corrected_local_law_loss(
    *,
    proxy_loss: float,
    oracle_loss: Optional[float],
    observed: bool,
    propensity: float,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Return the scalar corrected local-law loss.

    If the oracle residual is not observed, the proxy loss is retained. If it is
    observed, the proxy loss receives the usual IPW residual correction.
    """

    proxy = _finite_float(proxy_loss, name="proxy_loss")
    if not bool(observed):
        return proxy
    if oracle_loss is None:
        raise ValueError("observed local-law rows require oracle_loss")
    oracle = _finite_float(oracle_loss, name="oracle_loss")
    pi = _finite_float(propensity, name="propensity")
    min_pi = max(_finite_float(min_propensity, name="min_propensity"), MIN_PROPENSITY)
    if pi <= 0.0 or pi > 1.0:
        raise ValueError(f"observed local-law propensity must be in (0, 1], got {propensity!r}")
    clipped = max(min_pi, pi)
    return float(proxy + (oracle - proxy) / clipped)


@dataclass(frozen=True)
class LocalLawObservation:
    """One node-level proxy/oracle local-law observation."""

    proxy_loss: float
    oracle_loss: Optional[float] = None
    observed: bool = False
    propensity: float = 1.0
    depth: int = 0
    node_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        proxy = _finite_float(self.proxy_loss, name="proxy_loss")
        object.__setattr__(self, "proxy_loss", proxy)
        if self.oracle_loss is not None:
            oracle = _finite_float(self.oracle_loss, name="oracle_loss")
            object.__setattr__(self, "oracle_loss", oracle)
        propensity = _finite_float(self.propensity, name="propensity")
        if propensity < 0.0 or propensity > 1.0:
            raise ValueError(f"propensity must be in [0, 1], got {self.propensity!r}")
        if bool(self.observed) and propensity <= 0.0:
            raise ValueError("observed local-law rows require strictly positive propensity")
        object.__setattr__(self, "propensity", propensity)
        depth = int(self.depth)
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {self.depth!r}")
        object.__setattr__(self, "depth", depth)
        weight = _finite_float(self.node_weight, name="node_weight")
        if weight < 0.0:
            raise ValueError(f"node_weight must be non-negative, got {self.node_weight!r}")
        object.__setattr__(self, "node_weight", weight)
        if not isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", dict(self.metadata))

    def discounted_weight(self, *, gamma_depth: float = 1.0) -> float:
        return float(self.node_weight * depth_discount(gamma_depth, self.depth))

    def corrected_loss(self, *, min_propensity: float = MIN_PROPENSITY) -> float:
        return corrected_local_law_loss(
            proxy_loss=float(self.proxy_loss),
            oracle_loss=self.oracle_loss,
            observed=bool(self.observed),
            propensity=float(self.propensity),
            min_propensity=float(min_propensity),
        )


@dataclass(frozen=True)
class LocalLawAggregate:
    """Diagnostics for a discounted corrected local-law objective."""

    proxy_total: float
    residual_correction_total: float
    corrected_total: float
    weight_total: float
    population_count: int
    sampled_count: int
    effective_sample_size: float
    max_ipw_weight: float
    local_law_weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def corrected_mean(self) -> float:
        if float(self.weight_total) <= 0.0:
            return float("nan")
        return float(self.corrected_total / self.weight_total)

    @property
    def proxy_mean(self) -> float:
        if float(self.weight_total) <= 0.0:
            return float("nan")
        return float(self.proxy_total / self.weight_total)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "proxy_total": float(self.proxy_total),
            "proxy_mean": float(self.proxy_mean),
            "residual_correction_total": float(self.residual_correction_total),
            "corrected_total": float(self.corrected_total),
            "corrected_mean": float(self.corrected_mean),
            "weight_total": float(self.weight_total),
            "population_count": int(self.population_count),
            "sampled_count": int(self.sampled_count),
            "effective_sample_size": float(self.effective_sample_size),
            "max_ipw_weight": float(self.max_ipw_weight),
            "local_law_weight": float(self.local_law_weight),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def local_law_objective_mean(
    observations: Sequence[LocalLawObservation],
    *,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    gamma_depth: float = 1.0,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Return a discounted local-law objective from already-computed loss rows.

    ``corrected_local_law`` retains the full proxy population and corrects
    sampled oracle residuals. ``sampled_ipw`` uses observed oracle losses only
    with Hájek/IPW normalization. This function is deliberately loss-level so
    it can be shared by text, sketch, FNO, and report code.
    """

    rows: list[LocalLawObservation] = []
    for observation in observations:
        if isinstance(observation, LocalLawObservation):
            rows.append(observation)
        else:
            rows.append(LocalLawObservation(**dict(observation)))  # type: ignore[arg-type]
    if not rows:
        return 0.0

    mode = normalize_local_law_objective_mode(objective_mode)
    min_pi = max(_finite_float(min_propensity, name="min_propensity"), MIN_PROPENSITY)

    if mode == LOCAL_LAW_OBJECTIVE_SAMPLED_IPW:
        total = 0.0
        denom = 0.0
        for row in rows:
            if not bool(row.observed):
                continue
            if row.oracle_loss is None:
                raise ValueError("sampled_ipw rows require oracle_loss when observed")
            if float(row.propensity) <= 0.0:
                raise ValueError("sampled_ipw observed rows require positive propensity")
            weight = row.discounted_weight(gamma_depth=float(gamma_depth))
            ipw_weight = float(weight / max(min_pi, float(row.propensity)))
            total += float(ipw_weight * float(row.oracle_loss))
            denom += float(ipw_weight)
        if denom <= 0.0:
            return 0.0
        return float(total / denom)

    total = 0.0
    denom = 0.0
    for row in rows:
        weight = row.discounted_weight(gamma_depth=float(gamma_depth))
        total += float(weight * row.corrected_loss(min_propensity=min_pi))
        denom += float(weight)
    if denom <= 0.0:
        return 0.0
    return float(total / denom)


def aggregate_local_law_observations(
    observations: Sequence[LocalLawObservation],
    *,
    gamma_depth: float = 1.0,
    local_law_weight: float = 1.0,
    min_propensity: float = MIN_PROPENSITY,
    metadata: Optional[Mapping[str, Any]] = None,
) -> LocalLawAggregate:
    """Aggregate corrected local-law losses over a retained node population."""

    proxy_total = 0.0
    corrected_total = 0.0
    weight_total = 0.0
    sampled_count = 0
    ipw_weights: list[float] = []
    for observation in observations:
        if not isinstance(observation, LocalLawObservation):
            observation = LocalLawObservation(**dict(observation))  # type: ignore[arg-type]
        weight = observation.discounted_weight(gamma_depth=gamma_depth)
        proxy = float(observation.proxy_loss)
        corrected = observation.corrected_loss(min_propensity=min_propensity)
        proxy_total += float(weight * proxy)
        corrected_total += float(weight * corrected)
        weight_total += float(weight)
        if bool(observation.observed):
            sampled_count += 1
            ipw = 1.0 / max(float(min_propensity), float(observation.propensity))
            ipw_weights.append(float(weight * ipw))

    weight_sq = sum(float(w * w) for w in ipw_weights)
    weight_sum = sum(float(w) for w in ipw_weights)
    ess = float((weight_sum * weight_sum) / weight_sq) if weight_sq > 0.0 else 0.0
    max_w = max(ipw_weights) if ipw_weights else 0.0
    return LocalLawAggregate(
        proxy_total=float(proxy_total),
        residual_correction_total=float(corrected_total - proxy_total),
        corrected_total=float(corrected_total),
        weight_total=float(weight_total),
        population_count=int(len(observations)),
        sampled_count=int(sampled_count),
        effective_sample_size=float(ess),
        max_ipw_weight=float(max_w),
        local_law_weight=float(min(1.0, max(0.0, local_law_weight))),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "LOCAL_LAW_OBJECTIVE_CORRECTED",
    "LOCAL_LAW_OBJECTIVE_SAMPLED_IPW",
    "LocalLawAggregate",
    "LocalLawObservation",
    "VALID_LOCAL_LAW_OBJECTIVE_MODES",
    "aggregate_local_law_observations",
    "corrected_local_law_loss",
    "depth_discount",
    "local_law_objective_mean",
    "normalize_local_law_objective_mode",
]
