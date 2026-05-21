"""Torch helpers for corrected local-law losses."""

from __future__ import annotations

from typing import Optional

import torch

from treepo._research.core.local_law_adjustment import (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
    normalize_local_law_objective_mode,
)


def corrected_local_law_loss_tensor(
    *,
    proxy_loss: torch.Tensor,
    oracle_loss: torch.Tensor,
    observed: torch.Tensor,
    propensity: torch.Tensor,
    min_propensity: float = 1e-12,
) -> torch.Tensor:
    """Return ``proxy + R / pi * (oracle - proxy)`` elementwise."""

    _validate_observed_propensity_tensor(observed=observed, propensity=propensity)
    proxy = proxy_loss
    oracle = oracle_loss.to(device=proxy.device, dtype=proxy.dtype)
    obs = observed.to(device=proxy.device, dtype=proxy.dtype)
    pi = propensity.to(device=proxy.device, dtype=proxy.dtype).clamp(
        min=float(min_propensity),
        max=1.0,
    )
    return proxy + obs * (oracle - proxy) / pi


def _validate_observed_propensity_tensor(
    *,
    observed: torch.Tensor,
    propensity: torch.Tensor,
) -> None:
    obs = observed.reshape(-1).to(device=propensity.device, dtype=torch.bool)
    pi = propensity.reshape(-1)
    if obs.numel() != pi.numel():
        raise ValueError("observed and propensity tensors must have the same number of rows")
    if not bool(obs.any().detach().cpu()):
        return
    observed_pi = pi[obs]
    invalid = (~torch.isfinite(observed_pi)) | (observed_pi <= 0.0) | (observed_pi > 1.0)
    if bool(invalid.any().detach().cpu()):
        raise ValueError("observed local-law rows require finite propensity in (0, 1]")


def corrected_local_law_target_mse(
    *,
    predictions: torch.Tensor,
    proxy_targets: torch.Tensor,
    oracle_targets: torch.Tensor,
    observed: torch.Tensor,
    propensity: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    min_propensity: float = 1e-12,
) -> torch.Tensor:
    """Return weighted corrected MSE rows for proxy/oracle scalar targets."""

    preds = predictions.reshape(-1)
    proxy = proxy_targets.to(device=preds.device, dtype=preds.dtype).reshape(-1)
    oracle = oracle_targets.to(device=preds.device, dtype=preds.dtype).reshape(-1)
    proxy_loss = (preds - proxy) ** 2
    oracle_loss = (preds - oracle) ** 2
    corrected = corrected_local_law_loss_tensor(
        proxy_loss=proxy_loss,
        oracle_loss=oracle_loss,
        observed=observed.reshape(-1),
        propensity=propensity.reshape(-1),
        min_propensity=float(min_propensity),
    )
    if weights is None:
        return corrected
    return corrected * weights.to(device=preds.device, dtype=preds.dtype).reshape(-1)


def _depth_discount_weights(
    depths: torch.Tensor,
    *,
    gamma_depth: float,
) -> torch.Tensor:
    gamma = float(gamma_depth)
    if gamma < 0.0:
        raise ValueError(f"gamma_depth must be non-negative, got {gamma_depth!r}")
    depth_values = depths.reshape(-1).to(dtype=torch.float32)
    base = torch.full_like(depth_values, float(gamma))
    return torch.pow(base, depth_values)


def local_law_objective_target_mse(
    *,
    predictions: torch.Tensor,
    proxy_targets: torch.Tensor,
    oracle_targets: torch.Tensor,
    observed: torch.Tensor,
    propensity: torch.Tensor,
    depths: torch.Tensor,
    node_weights: Optional[torch.Tensor] = None,
    gamma_depth: float = 1.0,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    min_propensity: float = 1e-12,
) -> torch.Tensor:
    """Return a scalar local-law objective for scalar proxy/oracle targets."""

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


def local_law_objective_from_losses(
    *,
    proxy_loss: torch.Tensor,
    oracle_loss: torch.Tensor,
    observed: torch.Tensor,
    propensity: torch.Tensor,
    depths: torch.Tensor,
    node_weights: Optional[torch.Tensor] = None,
    gamma_depth: float = 1.0,
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    min_propensity: float = 1e-12,
) -> torch.Tensor:
    """Return a scalar local-law objective from precomputed loss rows."""

    _validate_observed_propensity_tensor(observed=observed, propensity=propensity)
    proxy = proxy_loss.reshape(-1)
    oracle = oracle_loss.to(device=proxy.device, dtype=proxy.dtype).reshape(-1)
    obs = observed.to(device=proxy.device, dtype=proxy.dtype).reshape(-1)
    pi = propensity.to(device=proxy.device, dtype=proxy.dtype).reshape(-1).clamp(
        min=float(min_propensity),
        max=1.0,
    )
    weights = _depth_discount_weights(
        depths.to(device=proxy.device, dtype=proxy.dtype),
        gamma_depth=float(gamma_depth),
    ).to(device=proxy.device, dtype=proxy.dtype)
    if node_weights is not None:
        weights = weights * node_weights.to(device=proxy.device, dtype=proxy.dtype).reshape(-1)

    mode = normalize_local_law_objective_mode(objective_mode)
    if mode == LOCAL_LAW_OBJECTIVE_SAMPLED_IPW:
        ipw_weights = weights * obs / pi
        denom = ipw_weights.sum().clamp(min=float(min_propensity))
        if float(ipw_weights.detach().sum().cpu()) <= 0.0:
            return torch.zeros((), device=proxy.device, dtype=proxy.dtype)
        return (ipw_weights * oracle).sum() / denom

    row_loss = corrected_local_law_loss_tensor(
        proxy_loss=proxy,
        oracle_loss=oracle,
        observed=obs,
        propensity=pi,
        min_propensity=float(min_propensity),
    )
    denom = weights.sum().clamp(min=float(min_propensity))
    return (weights * row_loss).sum() / denom


__all__ = [
    "corrected_local_law_loss_tensor",
    "corrected_local_law_target_mse",
    "local_law_objective_from_losses",
    "local_law_objective_target_mse",
]
