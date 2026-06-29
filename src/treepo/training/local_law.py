from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

from treepo.local_law import (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_MODES,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
    MIN_PROPENSITY,
    corrected_local_law_loss,
    normalize_local_law_objective_mode,
)
from treepo.common import finite_float


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for torch local-law objectives. "
            "Install with: uv sync --extra torch"
        )


def depth_discount(gamma_depth: float, depth: int) -> float:
    gamma = finite_float(gamma_depth, name="gamma_depth")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma_depth must be in [0, 1], got {gamma_depth!r}")
    d = int(depth)
    if d < 0:
        raise ValueError(f"depth must be non-negative, got {depth!r}")
    return float(gamma**d)


@dataclass(frozen=True)
class LocalLawTrainingRow:
    """One loss-level training row for proxy-only or sampled oracle supervision.

    Training rows intentionally differ from theorem-facing audit rows:
    unobserved proxy-only rows may carry ``propensity=0`` because no division is
    performed for those rows. Observed rows still require a positive logged
    propensity and an oracle loss.
    """

    proxy_loss: float | None = None
    oracle_loss: float | None = None
    observed: bool = False
    propensity: float = 1.0
    depth: int = 0
    node_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    row_id: str = ""
    law_kind: str = ""
    global_axiom: str = ""
    state_kind: str = ""
    law_channel: str = ""
    doc_id: str = ""
    node_id: str = ""
    prediction: float | None = None
    proxy_target: float | None = None
    oracle_target: float | None = None

    def __post_init__(self) -> None:
        prediction = (
            None
            if self.prediction is None
            else finite_float(self.prediction, name="prediction")
        )
        proxy_target = (
            None
            if self.proxy_target is None
            else finite_float(self.proxy_target, name="proxy_target")
        )
        oracle_target = (
            None
            if self.oracle_target is None
            else finite_float(self.oracle_target, name="oracle_target")
        )

        proxy_loss = self.proxy_loss
        if proxy_loss is None and prediction is not None and proxy_target is not None:
            proxy_loss = float((prediction - proxy_target) ** 2)
        if proxy_loss is None:
            raise ValueError("LocalLawTrainingRow requires proxy_loss or prediction+proxy_target")
        proxy = finite_float(proxy_loss, name="proxy_loss")

        oracle_loss = self.oracle_loss
        if oracle_loss is None and prediction is not None and oracle_target is not None:
            oracle_loss = float((prediction - oracle_target) ** 2)
        oracle = (
            None
            if oracle_loss is None
            else finite_float(oracle_loss, name="oracle_loss")
        )

        prop = finite_float(self.propensity, name="propensity")
        if prop < 0.0 or prop > 1.0:
            raise ValueError(f"propensity must be in [0, 1], got {self.propensity!r}")
        if bool(self.observed) and prop <= 0.0:
            raise ValueError("observed local-law rows require strictly positive propensity")
        if bool(self.observed) and oracle is None:
            raise ValueError("observed local-law rows require oracle_loss")

        depth = int(self.depth)
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {self.depth!r}")
        weight = finite_float(self.node_weight, name="node_weight")
        if weight < 0.0:
            raise ValueError("node_weight must be non-negative")

        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "proxy_target", proxy_target)
        object.__setattr__(self, "oracle_target", oracle_target)
        object.__setattr__(self, "proxy_loss", proxy)
        object.__setattr__(self, "oracle_loss", oracle)
        object.__setattr__(self, "propensity", prop)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "node_weight", weight)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "row_id", str(self.row_id or ""))
        object.__setattr__(self, "law_kind", str(self.law_kind or ""))
        object.__setattr__(self, "global_axiom", str(self.global_axiom or ""))
        object.__setattr__(self, "state_kind", str(self.state_kind or ""))
        object.__setattr__(self, "law_channel", str(self.law_channel or ""))
        object.__setattr__(self, "doc_id", str(self.doc_id or ""))
        object.__setattr__(self, "node_id", str(self.node_id or ""))

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalLawTrainingAggregate:
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

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
            payload["metadata"] = dict(self.metadata or {})
        return payload


def _coerce_training_row(row: LocalLawTrainingRow | Mapping[str, Any]) -> LocalLawTrainingRow:
    if isinstance(row, LocalLawTrainingRow):
        return row
    if isinstance(row, Mapping):
        return LocalLawTrainingRow(**dict(row))
    raise TypeError(
        f"local-law training row must be LocalLawTrainingRow or mapping, got {type(row).__name__}"
    )


def local_law_training_objective_mean(
    rows: Sequence[LocalLawTrainingRow] | Iterable[LocalLawTrainingRow],
    *,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    gamma_depth: float = 1.0,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    row_list = [_coerce_training_row(row) for row in rows]
    if not row_list:
        return 0.0

    mode = normalize_local_law_objective_mode(objective_mode)
    min_pi = max(finite_float(min_propensity, name="min_propensity"), MIN_PROPENSITY)
    if mode == LOCAL_LAW_OBJECTIVE_SAMPLED_IPW:
        total = 0.0
        denom = 0.0
        for row in row_list:
            if not bool(row.observed):
                continue
            if row.oracle_loss is None:
                raise ValueError("sampled_ipw rows require oracle_loss when observed")
            weight = row.discounted_weight(gamma_depth=gamma_depth)
            ipw_weight = float(weight / max(min_pi, float(row.propensity)))
            total += float(ipw_weight * float(row.oracle_loss))
            denom += float(ipw_weight)
        return float(total / denom) if denom > 0.0 else 0.0

    total = 0.0
    denom = 0.0
    for row in row_list:
        weight = row.discounted_weight(gamma_depth=gamma_depth)
        total += float(weight * row.corrected_loss(min_propensity=min_pi))
        denom += float(weight)
    return float(total / denom) if denom > 0.0 else 0.0


def aggregate_local_law_training_rows(
    rows: Sequence[LocalLawTrainingRow] | Iterable[LocalLawTrainingRow],
    *,
    gamma_depth: float = 1.0,
    local_law_weight: float = 1.0,
    min_propensity: float = MIN_PROPENSITY,
    metadata: Mapping[str, Any] | None = None,
) -> LocalLawTrainingAggregate:
    row_list = [_coerce_training_row(row) for row in rows]
    proxy_total = 0.0
    corrected_total = 0.0
    weight_total = 0.0
    sampled_count = 0
    ipw_weights: list[float] = []
    min_pi = max(finite_float(min_propensity, name="min_propensity"), MIN_PROPENSITY)
    for row in row_list:
        weight = row.discounted_weight(gamma_depth=gamma_depth)
        proxy = float(row.proxy_loss)
        corrected = row.corrected_loss(min_propensity=min_pi)
        proxy_total += float(weight * proxy)
        corrected_total += float(weight * corrected)
        weight_total += float(weight)
        if bool(row.observed):
            sampled_count += 1
            ipw_weights.append(float(weight / max(min_pi, float(row.propensity))))

    weight_sq = sum(float(w * w) for w in ipw_weights)
    weight_sum = sum(float(w) for w in ipw_weights)
    ess = float((weight_sum * weight_sum) / weight_sq) if weight_sq > 0.0 else 0.0
    max_w = max(ipw_weights) if ipw_weights else 0.0
    llw = finite_float(local_law_weight, name="local_law_weight")
    if llw < 0.0 or llw > 1.0:
        raise ValueError("local_law_weight must be in [0, 1]")
    return LocalLawTrainingAggregate(
        proxy_total=float(proxy_total),
        residual_correction_total=float(corrected_total - proxy_total),
        corrected_total=float(corrected_total),
        weight_total=float(weight_total),
        population_count=len(row_list),
        sampled_count=int(sampled_count),
        effective_sample_size=float(ess),
        max_ipw_weight=float(max_w),
        local_law_weight=float(llw),
        metadata=dict(metadata or {}),
    )


def _validate_observed_propensity_tensor(
    *,
    observed: "torch.Tensor",
    propensity: "torch.Tensor",
) -> None:
    _require_torch()
    obs = observed.reshape(-1).to(dtype=torch.bool)
    pi = propensity.reshape(-1)
    if obs.numel() != pi.numel():
        raise ValueError("observed and propensity tensors must have the same number of rows")
    if not bool(obs.any().detach().cpu()):
        return
    observed_pi = pi[obs]
    invalid = (~torch.isfinite(observed_pi)) | (observed_pi <= 0.0) | (observed_pi > 1.0)
    if bool(invalid.any().detach().cpu()):
        raise ValueError("observed local-law rows require finite propensity in (0, 1]")


def corrected_local_law_loss_tensor(
    *,
    proxy_loss: "torch.Tensor",
    oracle_loss: "torch.Tensor",
    observed: "torch.Tensor",
    propensity: "torch.Tensor",
    min_propensity: float = 1e-12,
) -> "torch.Tensor":
    """Return ``proxy + R / pi * (oracle - proxy)`` elementwise."""

    _require_torch()
    _validate_observed_propensity_tensor(observed=observed, propensity=propensity)
    proxy = proxy_loss
    oracle = oracle_loss.to(device=proxy.device, dtype=proxy.dtype)
    obs = observed.to(device=proxy.device, dtype=proxy.dtype)
    pi = propensity.to(device=proxy.device, dtype=proxy.dtype).clamp(
        min=float(min_propensity),
        max=1.0,
    )
    return proxy + obs * (oracle - proxy) / pi


def corrected_local_law_target_mse(
    *,
    predictions: "torch.Tensor",
    proxy_targets: "torch.Tensor",
    oracle_targets: "torch.Tensor",
    observed: "torch.Tensor",
    propensity: "torch.Tensor",
    weights: Optional["torch.Tensor"] = None,
    min_propensity: float = 1e-12,
) -> "torch.Tensor":
    """Return corrected MSE rows for scalar proxy/oracle targets."""

    _require_torch()
    preds = predictions.reshape(-1)
    proxy = proxy_targets.to(device=preds.device, dtype=preds.dtype).reshape(-1)
    oracle = oracle_targets.to(device=preds.device, dtype=preds.dtype).reshape(-1)
    corrected = corrected_local_law_loss_tensor(
        proxy_loss=(preds - proxy) ** 2,
        oracle_loss=(preds - oracle) ** 2,
        observed=observed.reshape(-1),
        propensity=propensity.reshape(-1),
        min_propensity=float(min_propensity),
    )
    if weights is None:
        return corrected
    return corrected * weights.to(device=preds.device, dtype=preds.dtype).reshape(-1)


def _depth_discount_weights(
    depths: "torch.Tensor",
    *,
    gamma_depth: float,
) -> "torch.Tensor":
    _require_torch()
    gamma = float(gamma_depth)
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma_depth must be in [0, 1], got {gamma_depth!r}")
    depth_values = depths.reshape(-1).to(dtype=torch.float32)
    base = torch.full_like(depth_values, gamma)
    return torch.pow(base, depth_values)


def local_law_objective_from_losses(
    *,
    proxy_loss: "torch.Tensor",
    oracle_loss: "torch.Tensor",
    observed: "torch.Tensor",
    propensity: "torch.Tensor",
    depths: "torch.Tensor",
    node_weights: Optional["torch.Tensor"] = None,
    gamma_depth: float = 1.0,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    min_propensity: float = 1e-12,
) -> "torch.Tensor":
    """Return a scalar corrected/IPW local-law objective from loss rows."""

    _require_torch()
    _validate_observed_propensity_tensor(observed=observed, propensity=propensity)
    proxy = proxy_loss.reshape(-1)
    oracle = oracle_loss.to(device=proxy.device, dtype=proxy.dtype).reshape(-1)
    obs = observed.to(device=proxy.device, dtype=proxy.dtype).reshape(-1)
    pi = propensity.to(device=proxy.device, dtype=proxy.dtype).reshape(-1).clamp(
        min=float(min_propensity),
        max=1.0,
    )
    weights = _depth_discount_weights(
        depths.to(device=proxy.device),
        gamma_depth=float(gamma_depth),
    ).to(device=proxy.device, dtype=proxy.dtype)
    if node_weights is not None:
        weights = weights * node_weights.to(device=proxy.device, dtype=proxy.dtype).reshape(-1)

    mode = normalize_local_law_objective_mode(objective_mode)
    if mode == LOCAL_LAW_OBJECTIVE_SAMPLED_IPW:
        ipw_weights = weights * obs / pi
        if float(ipw_weights.detach().sum().cpu()) <= 0.0:
            return torch.zeros((), device=proxy.device, dtype=proxy.dtype)
        return (ipw_weights * oracle).sum() / ipw_weights.sum().clamp(min=float(min_propensity))

    corrected = corrected_local_law_loss_tensor(
        proxy_loss=proxy,
        oracle_loss=oracle,
        observed=obs,
        propensity=pi,
        min_propensity=float(min_propensity),
    )
    return (weights * corrected).sum() / weights.sum().clamp(min=float(min_propensity))


def local_law_objective_target_mse(
    *,
    predictions: "torch.Tensor",
    proxy_targets: "torch.Tensor",
    oracle_targets: "torch.Tensor",
    observed: "torch.Tensor",
    propensity: "torch.Tensor",
    depths: "torch.Tensor",
    node_weights: Optional["torch.Tensor"] = None,
    gamma_depth: float = 1.0,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    min_propensity: float = 1e-12,
) -> "torch.Tensor":
    """Return a scalar local-law objective for scalar proxy/oracle targets."""

    _require_torch()
    preds = predictions.reshape(-1)
    proxy = proxy_targets.to(device=preds.device, dtype=preds.dtype).reshape(-1)
    oracle = oracle_targets.to(device=preds.device, dtype=preds.dtype).reshape(-1)
    return local_law_objective_from_losses(
        proxy_loss=(preds - proxy) ** 2,
        oracle_loss=(preds - oracle) ** 2,
        observed=observed,
        propensity=propensity,
        depths=depths,
        node_weights=node_weights,
        gamma_depth=float(gamma_depth),
        objective_mode=str(objective_mode),
        min_propensity=float(min_propensity),
    )


def sampled_uniform_node_ipw_mean_loss(
    losses: "torch.Tensor",
    *,
    rate: float,
    node_weights: Optional["torch.Tensor"] = None,
    min_propensity: float = 1e-12,
) -> "torch.Tensor":
    """Sample nodes uniformly and return the master sampled-IPW objective.

    ``losses`` is a realized node-population loss tensor with shape
    ``[batch, nodes]`` or ``[nodes]``. The sampling mask and propensity are
    derived from the actual node width and draw count; no caller-supplied
    propensity constants are needed. ``node_weights`` optionally changes the
    target node-population measure, while the default is the unweighted node
    mean.
    """

    _require_torch()
    values = losses
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim != 2:
        raise ValueError("sampled_uniform_node_ipw_mean_loss expects [batch, nodes] losses")
    batch = int(values.shape[0])
    width = int(values.shape[1])
    if batch <= 0 or width <= 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    if float(rate) <= 0.0:
        return torch.zeros((), dtype=values.dtype, device=values.device)

    if float(rate) >= 1.0:
        observed = torch.ones_like(values, dtype=torch.bool, device=values.device)
        propensity = torch.ones_like(values, dtype=values.dtype, device=values.device)
    else:
        q = max(1, min(width, int(math.ceil(float(rate) * float(width)))))
        scores = torch.rand((batch, width), device=values.device)
        idx = torch.topk(scores, k=q, dim=1).indices
        observed = torch.zeros((batch, width), dtype=torch.bool, device=values.device)
        observed = observed.scatter(1, idx, True)
        propensity = torch.full(
            (batch, width),
            float(q) / float(width),
            dtype=values.dtype,
            device=values.device,
        )

    proxy = torch.zeros_like(values, dtype=values.dtype, device=values.device)
    depths = torch.zeros_like(values, dtype=torch.float32, device=values.device)
    weights = (
        None
        if node_weights is None
        else node_weights.to(device=values.device, dtype=values.dtype).reshape_as(values)
    )
    return local_law_objective_from_losses(
        proxy_loss=proxy,
        oracle_loss=values,
        observed=observed,
        propensity=propensity,
        depths=depths,
        node_weights=weights,
        gamma_depth=1.0,
        objective_mode=LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
        min_propensity=float(min_propensity),
    )


def observed_uniform_node_ipw_mean_loss(
    losses: "torch.Tensor",
    *,
    observed: "torch.Tensor",
    propensity: float | "torch.Tensor",
    node_weights: Optional["torch.Tensor"] = None,
    min_propensity: float = 1e-12,
) -> "torch.Tensor":
    """Return the sampled-IPW objective for an already-realized node mask.

    ``observed`` is a fixed mask over the same node population as ``losses``.
    ``propensity`` is the actual per-node inclusion probability from the
    sampling design; callers must pass it from the design rather than deriving
    it from the realized mask. This supports persistent Bernoulli masks where a
    row can legitimately contain zero observed nodes.
    """

    _require_torch()
    values = losses
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim != 2:
        raise ValueError("observed_uniform_node_ipw_mean_loss expects [batch, nodes] losses")
    batch = int(values.shape[0])
    width = int(values.shape[1])
    if batch <= 0 or width <= 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)

    obs = observed.to(device=values.device, dtype=torch.bool)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if tuple(obs.shape) != tuple(values.shape):
        raise ValueError(
            "observed mask shape must match losses shape; "
            f"got {tuple(obs.shape)} vs {tuple(values.shape)}"
        )
    if not bool(obs.any().detach().cpu()):
        return torch.zeros((), dtype=values.dtype, device=values.device)

    if isinstance(propensity, (float, int)):
        pi = torch.full(
            values.shape,
            float(propensity),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        pi = propensity.to(device=values.device, dtype=values.dtype)
        if pi.ndim == 0:
            pi = torch.full(values.shape, float(pi.detach().cpu()), dtype=values.dtype, device=values.device)
        elif pi.ndim == 1:
            pi = pi.reshape(1, -1)
        pi = pi.expand_as(values)
    proxy = torch.zeros_like(values, dtype=values.dtype, device=values.device)
    depths = torch.zeros_like(values, dtype=torch.float32, device=values.device)
    weights = (
        None
        if node_weights is None
        else node_weights.to(device=values.device, dtype=values.dtype).reshape_as(values)
    )
    return local_law_objective_from_losses(
        proxy_loss=proxy,
        oracle_loss=values,
        observed=obs,
        propensity=pi,
        depths=depths,
        node_weights=weights,
        gamma_depth=1.0,
        objective_mode=LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
        min_propensity=float(min_propensity),
    )


__all__ = [
    "LOCAL_LAW_OBJECTIVE_CORRECTED",
    "LOCAL_LAW_OBJECTIVE_MODES",
    "LOCAL_LAW_OBJECTIVE_SAMPLED_IPW",
    "LocalLawTrainingAggregate",
    "LocalLawTrainingRow",
    "MIN_PROPENSITY",
    "aggregate_local_law_training_rows",
    "corrected_local_law_loss",
    "corrected_local_law_loss_tensor",
    "corrected_local_law_target_mse",
    "depth_discount",
    "local_law_objective_from_losses",
    "local_law_objective_target_mse",
    "local_law_training_objective_mean",
    "normalize_local_law_objective_mode",
    "observed_uniform_node_ipw_mean_loss",
    "sampled_uniform_node_ipw_mean_loss",
]
