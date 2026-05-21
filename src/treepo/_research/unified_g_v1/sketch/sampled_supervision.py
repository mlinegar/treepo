"""Sampling helpers for local leaf/internal-node supervision."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from treepo.training.local_law import (
    observed_uniform_node_ipw_mean_loss,
    sampled_uniform_node_ipw_mean_loss,
)

PERSISTENT_UNIFORM_NODE_SCORES_KEY = "persistent_uniform_node_scores"
PERSISTENT_UNIFORM_NODE_WIDTH_KEY = "persistent_uniform_node_width"


def sampled_axis_mse(pred: torch.Tensor, target: torch.Tensor, *, rate: float) -> torch.Tensor:
    """MSE over a fixed-size random subset of the node axis.

    ``pred`` and ``target`` are expected to have shape ``[batch, nodes, ...]``.
    A positive rate always samples at least one node when nodes exist, matching
    the legacy R-grid convention.
    """

    if pred.numel() == 0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    if float(rate) <= 0.0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    if float(rate) >= 1.0 or pred.ndim < 2:
        return F.mse_loss(pred, target)

    batch = int(pred.shape[0])
    width = int(pred.shape[1])
    if batch <= 0 or width <= 0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)

    q = max(1, min(width, int(math.ceil(float(rate) * float(width)))))
    scores = torch.rand((batch, width), device=pred.device)
    idx = torch.topk(scores, k=q, dim=1).indices
    mask = torch.zeros((batch, width), dtype=torch.bool, device=pred.device)
    mask = mask.scatter(1, idx, True)

    squared = (pred - target) ** 2
    while mask.ndim < squared.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(dtype=squared.dtype).expand_as(squared)
    return torch.sum(squared * weights) / torch.clamp(torch.sum(weights), min=1.0)


def sampled_batch_mse(pred: torch.Tensor, target: torch.Tensor, *, rate: float) -> torch.Tensor:
    """MSE over a sampled subset of batch rows.

    This is the root-label analogue of :func:`sampled_axis_mse`: each document
    contributes one root label, and a positive rate samples at least one root
    when the batch is non-empty.
    """

    squared = (pred - target) ** 2
    if squared.ndim == 0:
        squared = squared.reshape(1)
    else:
        squared = squared.reshape(int(squared.shape[0]), -1).mean(dim=-1)
    batch = int(squared.shape[0])
    if batch <= 0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    if float(rate) <= 0.0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    if float(rate) >= 1.0:
        return squared.mean()

    q = max(1, min(batch, int(math.ceil(float(rate) * float(batch)))))
    scores = torch.rand((batch,), device=pred.device)
    idx = torch.topk(scores, k=q, dim=0).indices
    mask = torch.zeros((batch,), dtype=torch.bool, device=pred.device)
    mask = mask.scatter(0, idx, True)
    weights = mask.to(dtype=squared.dtype)
    return torch.sum(squared * weights) / torch.clamp(torch.sum(weights), min=1.0)


def uniform_tree_node_width(n_leaves: int) -> int:
    """Number of root+leaf+non-root-internal nodes in the supervision frame."""

    leaves = max(0, int(n_leaves))
    return 1 + leaves + max(0, leaves - 2)


def _scores_from_seed(*, seed: int, index: int, width: int) -> np.ndarray:
    payload = f"{int(seed)}:{int(index)}:{int(width)}".encode("utf-8")
    seed_int = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed_int)
    return rng.random(int(width)).astype(np.float32, copy=False)


def attach_persistent_uniform_node_scores(
    examples: list[Any],
    *,
    seed: int,
) -> list[Any]:
    """Attach fixed per-document node scores used by persistent Bernoulli masks.

    Higher-level experiment code can reuse the same cached examples across
    epochs. A fixed score vector makes R10/R30/... nested Bernoulli annotation
    subsets for each document instead of redrawing supervision every
    optimization step. There is no per-document minimum: small documents can
    legitimately have zero observed nodes at low R.
    """

    for index, item in enumerate(examples):
        extra = getattr(item, "extra", None)
        if not isinstance(extra, dict):
            continue
        width = uniform_tree_node_width(len(getattr(item, "leaves", ())))
        extra[PERSISTENT_UNIFORM_NODE_SCORES_KEY] = _scores_from_seed(
            seed=int(seed),
            index=int(index),
            width=int(width),
        )
        extra[PERSISTENT_UNIFORM_NODE_WIDTH_KEY] = int(width)
        extra["persistent_uniform_node_index"] = int(index)
        extra["persistent_uniform_node_policy"] = "fixed_without_replacement"
    return examples


def _fallback_scores(item: Any, *, width: int) -> np.ndarray:
    extra = getattr(item, "extra", {}) or {}
    flat_tokens = extra.get("flat_tokens") if isinstance(extra, dict) else None
    if flat_tokens is not None:
        payload = np.asarray(list(flat_tokens), dtype=np.int64).tobytes()
    else:
        payload = repr((getattr(item, "leaves", ()), getattr(item, "target", None))).encode(
            "utf-8",
            errors="replace",
        )
    payload += f":{int(width)}".encode("utf-8")
    seed_int = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed_int)
    return rng.random(int(width)).astype(np.float32, copy=False)


def persistent_uniform_node_mask(
    batch: list[Any] | tuple[Any, ...],
    *,
    width: int,
    rate: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the fixed per-document Bernoulli node mask for this batch/rate."""

    batch_size = int(len(batch))
    width = int(width)
    if batch_size <= 0 or width <= 0 or float(rate) <= 0.0:
        return torch.zeros((batch_size, max(0, width)), dtype=torch.bool, device=device)
    if float(rate) >= 1.0:
        return torch.ones((batch_size, width), dtype=torch.bool, device=device)

    rows: list[torch.Tensor] = []
    for item in batch:
        extra = getattr(item, "extra", {}) or {}
        raw = extra.get(PERSISTENT_UNIFORM_NODE_SCORES_KEY) if isinstance(extra, dict) else None
        if raw is None or len(raw) != width:
            raw = _fallback_scores(item, width=width)
            if isinstance(extra, dict):
                extra[PERSISTENT_UNIFORM_NODE_SCORES_KEY] = raw
                extra[PERSISTENT_UNIFORM_NODE_WIDTH_KEY] = int(width)
        rows.append(torch.as_tensor(raw, dtype=torch.float32, device=device))

    scores = torch.stack(rows, dim=0)
    return scores < float(rate)


def _per_node_squared(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    squared = (pred - target) ** 2
    if squared.ndim == 0:
        return squared.reshape(1, 1)
    batch = int(squared.shape[0])
    if squared.ndim == 1:
        return squared.reshape(batch, 1)
    nodes = int(squared.shape[1])
    return squared.reshape(batch, nodes, -1).mean(dim=-1)


def sampled_tree_node_mse(
    *,
    root_pred: torch.Tensor,
    root_target: torch.Tensor,
    leaf_pred: torch.Tensor,
    leaf_target: torch.Tensor,
    internal_pred: torch.Tensor | None = None,
    internal_target: torch.Tensor | None = None,
    rate: float,
    node_mask: torch.Tensor | None = None,
    node_propensity: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """IPW MSE over a uniform random subset of root, leaf, and internal nodes.

    Each tree node contributes one scalar loss to the sampling frame. Vector
    labels are averaged within node first, then nodes are sampled uniformly
    without replacement. The returned loss is the Horvitz-Thompson estimate of
    the full unweighted node-population mean, with inclusion probability
    derived from the actual sampled width and draw count.
    """

    device = root_pred.device
    dtype = root_pred.dtype
    if float(rate) <= 0.0:
        return torch.zeros((), dtype=dtype, device=device)

    pieces = [
        _per_node_squared(root_pred, root_target.to(device=device, dtype=dtype)),
        _per_node_squared(leaf_pred, leaf_target.to(device=device, dtype=leaf_pred.dtype)).to(
            device=device,
            dtype=dtype,
        ),
    ]
    if internal_pred is not None and internal_target is not None and internal_pred.numel() > 0:
        pieces.append(
            _per_node_squared(
                internal_pred,
                internal_target.to(device=internal_pred.device, dtype=internal_pred.dtype),
            ).to(device=device, dtype=dtype)
        )

    values = torch.cat(pieces, dim=1)
    batch = int(values.shape[0])
    width = int(values.shape[1])
    if batch <= 0 or width <= 0:
        return torch.zeros((), dtype=dtype, device=device)
    node_weights = torch.ones_like(values, dtype=dtype, device=device)
    if node_mask is not None:
        return observed_uniform_node_ipw_mean_loss(
            values,
            observed=node_mask,
            propensity=float(rate) if node_propensity is None else node_propensity,
            node_weights=node_weights,
        )
    return sampled_uniform_node_ipw_mean_loss(
        values,
        rate=float(rate),
        node_weights=node_weights,
    )


__all__ = [
    "PERSISTENT_UNIFORM_NODE_SCORES_KEY",
    "PERSISTENT_UNIFORM_NODE_WIDTH_KEY",
    "attach_persistent_uniform_node_scores",
    "persistent_uniform_node_mask",
    "sampled_axis_mse",
    "sampled_batch_mse",
    "sampled_tree_node_mse",
    "uniform_tree_node_width",
]
