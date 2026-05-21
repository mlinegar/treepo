from __future__ import annotations

import math
from typing import Optional


MIN_PROPENSITY = 1e-12
LOCAL_LAW_OBJECTIVE_CORRECTED = "corrected_local_law"
LOCAL_LAW_OBJECTIVE_SAMPLED_IPW = "sampled_ipw"
LOCAL_LAW_OBJECTIVE_MODES = (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
)


def _finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def normalize_local_law_objective_mode(mode: str) -> str:
    normalized = str(mode or LOCAL_LAW_OBJECTIVE_CORRECTED).strip().lower()
    aliases = {
        "corrected": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "aipw": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "adjusted": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "adjusted_local_law": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "dr": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "doubly_robust": LOCAL_LAW_OBJECTIVE_CORRECTED,
        "ipw": LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
        "sampled": LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
        "hajek": LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
        "sampled_hajek": LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LOCAL_LAW_OBJECTIVE_MODES:
        raise ValueError(
            f"unknown local-law objective mode {mode!r}; expected one of "
            f"{LOCAL_LAW_OBJECTIVE_MODES}"
        )
    return normalized


def corrected_local_law_loss(
    *,
    proxy_loss: float,
    oracle_loss: Optional[float],
    observed: bool,
    propensity: float,
    min_propensity: float = 1e-12,
) -> float:
    """Return ``proxy + R / pi * (oracle - proxy)`` for one node loss row."""

    proxy = _finite_float(proxy_loss, name="proxy_loss")
    if not bool(observed):
        return proxy
    if oracle_loss is None:
        raise ValueError("observed corrected local-law rows require oracle_loss")
    oracle = _finite_float(oracle_loss, name="oracle_loss")
    pi = _finite_float(propensity, name="propensity")
    min_pi = max(_finite_float(min_propensity, name="min_propensity"), MIN_PROPENSITY)
    if pi <= 0.0 or pi > 1.0:
        raise ValueError(f"observed local-law propensity must be in (0, 1], got {propensity!r}")
    return float(proxy + (oracle - proxy) / max(min_pi, pi))


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for torch local-law objectives. "
            "Install with: pip install 'treepo[torch]'"
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


def _depth_discount_weights(
    depths: "torch.Tensor",
    *,
    gamma_depth: float,
) -> "torch.Tensor":
    _require_torch()
    gamma = float(gamma_depth)
    if gamma < 0.0:
        raise ValueError(f"gamma_depth must be non-negative, got {gamma_depth!r}")
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
    "corrected_local_law_loss",
    "corrected_local_law_loss_tensor",
    "local_law_objective_from_losses",
    "local_law_objective_target_mse",
    "normalize_local_law_objective_mode",
    "observed_uniform_node_ipw_mean_loss",
    "sampled_uniform_node_ipw_mean_loss",
]
