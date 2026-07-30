"""Generic metadata-backed exact-state supervision for neural operators.

The task/root readout and the law witness are deliberately separate surfaces.
``target_key`` continues to define the task output.  A node metadata vector
under ``law_state_target_key`` instead supervises the first coordinates of the
hidden trace state.  Consequently a scalar task such as RILE can retain its
original scalar root loss while C1/C3 recover an exact vector witness such as
``(signed CMP count, non-header mass)``.

Rows follow the package balanced trace exactly: leaves first, then every merge
level, with the root merge last. Missing metadata keys are genuinely
unobserved.  Persistent sampled designs carry their logged leaf/merge
propensities into the canonical sampled-IPW objective.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_loss import per_row_training_loss
from treepo.methods._fno_targets import _merge_targets_by_position
from treepo.methods._fno_transition import _pairwise_merge_depths
from treepo.tree import tree_leaves


@dataclass(frozen=True)
class MetadataLawStateTargets:
    """One tree's trace-aligned state targets and sampling design."""

    values: Any
    observed: Any
    propensity: Any
    node_weights: Any


@dataclass(frozen=True)
class LawLossRows:
    """Tensor rows consumed by the canonical local-law objective."""

    losses: Any
    depths: Any
    is_leaf: Any
    observed: Any
    propensity: Any
    node_weights: Any
    objective_mode: str = "corrected_local_law"


def metadata_law_state_targets(
    trees: Sequence[Any] | None,
    config: NeuralOperatorFamilyConfig,
    *,
    torch: Any,
    device: Any,
) -> list[MetadataLawStateTargets] | None:
    """Extract a sparse exact-state target tensor for every training tree."""

    key = str(config.law_state_target_key or "").strip()
    if not key or trees is None:
        return None
    width = int(config.law_state_target_dim or 0)
    if width <= 0:
        raise ValueError("law_state_target_key requires positive law_state_target_dim")
    leaf_pi = _propensity(config.law_state_leaf_propensity, name="law_state_leaf_propensity")
    merge_pi = _propensity(
        config.law_state_merge_propensity, name="law_state_merge_propensity"
    )
    target_config = replace(
        config,
        node_target_key=key,
        node_target_exclusive=True,
    )
    out: list[MetadataLawStateTargets] = []
    any_observed = False
    for tree in trees:
        leaves = tuple(tree_leaves(tree) or ())
        leaf_count = max(1, len(leaves))
        n_nodes = 2 * leaf_count - 1
        values = [[0.0] * width for _ in range(n_nodes)]
        observed = [False] * n_nodes
        propensities = [leaf_pi] * leaf_count + [merge_pi] * (n_nodes - leaf_count)

        for position, leaf in enumerate(leaves):
            value = _metadata_vector(leaf, key=key, width=width)
            if value is not None:
                values[position] = value
                observed[position] = True

        # _merge_targets_by_position uses the same contiguous-span matcher as
        # the executed balanced trace. Unlike ordinary node readout targets we
        # retain its final/root position: root task loss does not substitute
        # for the root call's C3 state obligation.
        for position, value in _merge_targets_by_position(
            tree,
            target_config,
            width=width,
            leaves=leaves,
            leaf_count=leaf_count,
        ):
            values[int(position)] = [float(item) for item in value]
            observed[int(position)] = True

        any_observed = any_observed or any(observed)
        out.append(
            MetadataLawStateTargets(
                values=torch.tensor(values, dtype=torch.float32, device=device),
                observed=torch.tensor(observed, dtype=torch.bool, device=device),
                propensity=torch.tensor(
                    propensities, dtype=torch.float32, device=device
                ),
                node_weights=torch.ones(n_nodes, dtype=torch.float32, device=device),
            )
        )
    return out if any_observed else None


def metadata_law_state_rows(
    traces: Sequence[Any],
    targets: Sequence[MetadataLawStateTargets],
    *,
    training_loss: str,
    torch: Any,
    device: Any,
    dtype: Any,
) -> LawLossRows | None:
    """Compare hidden trace coordinates with sparse exact-state witnesses."""

    if len(traces) != len(targets):
        raise ValueError(
            f"metadata state supervision got {len(traces)} traces for "
            f"{len(targets)} targets"
        )
    loss_chunks = []
    depth_chunks = []
    leaf_chunks = []
    observed_chunks = []
    propensity_chunks = []
    weight_chunks = []
    for pred, target in zip(traces, targets):
        n_nodes = int(pred.shape[0])
        if int(target.values.shape[0]) != n_nodes:
            raise ValueError(
                "metadata state node count mismatch: model trace has "
                f"{n_nodes} nodes, targets have {int(target.values.shape[0])}"
            )
        width = int(target.values.shape[1])
        if width <= 0 or width > int(pred.shape[1]):
            raise ValueError(
                f"law state target width {width} exceeds learned state width "
                f"{int(pred.shape[1])}"
            )
        truth = target.values.to(device=device, dtype=dtype)
        loss_chunks.append(
            per_row_training_loss(
                pred[:, :width],
                truth,
                training_loss=training_loss,
            )
        )
        leaf_count = (n_nodes + 1) // 2
        depth_chunks.append(
            torch.tensor(
                _pairwise_merge_depths(leaf_count)[:n_nodes],
                dtype=torch.long,
                device=device,
            )
        )
        leaf_chunks.append(torch.arange(n_nodes, device=device) < leaf_count)
        observed_chunks.append(target.observed.to(device=device))
        propensity_chunks.append(target.propensity.to(device=device, dtype=dtype))
        weight_chunks.append(target.node_weights.to(device=device, dtype=dtype))
    if not loss_chunks:
        return None
    return LawLossRows(
        losses=torch.cat(loss_chunks),
        depths=torch.cat(depth_chunks),
        is_leaf=torch.cat(leaf_chunks),
        observed=torch.cat(observed_chunks),
        propensity=torch.cat(propensity_chunks),
        node_weights=torch.cat(weight_chunks),
        objective_mode="sampled_ipw",
    )


def metadata_law_state_design_rows(
    targets: Sequence[MetadataLawStateTargets],
    *,
    torch: Any,
    device: Any,
) -> LawLossRows | None:
    """Return trace-design rows without a model forward.

    This is used to compute the one outer batch's global C1/C3 denominators
    before exact activation microbatching. The zero losses are placeholders;
    only depths, channel flags, masks, propensities, and node weights matter.
    """

    loss_chunks = []
    depth_chunks = []
    leaf_chunks = []
    observed_chunks = []
    propensity_chunks = []
    weight_chunks = []
    for target in targets:
        n_nodes = int(target.values.shape[0])
        leaf_count = (n_nodes + 1) // 2
        loss_chunks.append(torch.zeros(n_nodes, dtype=torch.float32, device=device))
        depth_chunks.append(
            torch.tensor(
                _pairwise_merge_depths(leaf_count)[:n_nodes],
                dtype=torch.long,
                device=device,
            )
        )
        leaf_chunks.append(torch.arange(n_nodes, device=device) < leaf_count)
        observed_chunks.append(target.observed.to(device=device))
        propensity_chunks.append(target.propensity.to(device=device))
        weight_chunks.append(target.node_weights.to(device=device))
    if not loss_chunks:
        return None
    return LawLossRows(
        losses=torch.cat(loss_chunks),
        depths=torch.cat(depth_chunks),
        is_leaf=torch.cat(leaf_chunks),
        observed=torch.cat(observed_chunks),
        propensity=torch.cat(propensity_chunks),
        node_weights=torch.cat(weight_chunks),
        objective_mode="sampled_ipw",
    )


def dense_law_loss_rows(rows: Any) -> LawLossRows | None:
    """Upgrade the historical dense ``(loss, depth, is_leaf)`` tuple."""

    if rows is None:
        return None
    losses, depths, is_leaf = rows
    return LawLossRows(
        losses=losses,
        depths=depths,
        is_leaf=is_leaf,
        observed=losses.new_ones(losses.shape[0]).bool(),
        propensity=losses.new_ones(losses.shape[0]),
        node_weights=losses.new_ones(losses.shape[0]),
    )


def _metadata_vector(node: Any, *, key: str, width: int) -> list[float] | None:
    metadata = getattr(node, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    value = metadata.get(key)
    if value is None and isinstance(node, Mapping):
        inner = node.get("metadata")
        if isinstance(inner, Mapping):
            value = inner.get(key)
        if value is None:
            value = node.get(key)
    if value is None or isinstance(value, (str, bytes, bool, Mapping)):
        return None
    try:
        vector = [float(item) for item in value]
    except TypeError:
        return None
    if len(vector) != int(width):
        raise ValueError(
            f"law_state_target_key={key!r} produced {len(vector)} values; "
            f"expected law_state_target_dim={int(width)}"
        )
    return vector


def _propensity(value: Any, *, name: str) -> float:
    out = float(value)
    if out <= 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return out


__all__ = [
    "LawLossRows",
    "MetadataLawStateTargets",
    "dense_law_loss_rows",
    "metadata_law_state_design_rows",
    "metadata_law_state_rows",
    "metadata_law_state_targets",
]
