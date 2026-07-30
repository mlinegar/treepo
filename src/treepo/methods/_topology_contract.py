"""Topology-only validation and provenance for :mod:`treepo.methods`.

The number of leaves determines whether a C-Tree contains composition.  It is
independent of whether the shared state operator ``g`` is identity, fixed, or
learned.  Keeping this normalization in one small internal module prevents the
learning gate and result provenance from drifting apart again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from treepo.tree import tree_leaves

_SINGLETON_REPRESENTATIONS = frozenset({"full_doc", "full_doc_direct", "ctree_base_summary"})
_RECURSIVE_REPRESENTATIONS = frozenset({"ctree_recursive"})


@dataclass(frozen=True)
class TopologyContract:
    """Normalized structural facts for one fit cell."""

    representation: str | None
    topology_kind: str
    declared_leaf_count: int | None
    leaf_g_application_count_per_tree: int | None
    merge_application_count_per_tree: int | None
    singleton: bool
    composition_present: bool
    training_composition_present: bool
    all_singleton_training: bool
    train_tree_count: int
    eval_tree_count: int
    observed_leaf_count_min: int | None
    observed_leaf_count_max: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_topology_contract(
    train_trees: Sequence[Any],
    eval_trees: Sequence[Any],
    *,
    axis: Mapping[str, Any] | None = None,
    backend_config: Mapping[str, Any] | None = None,
) -> TopologyContract:
    """Validate inspectable C-Tree topology without consulting ``g_mode``.

    Named singleton paths require one leaf; the recursive path requires two or
    more; legacy ``ctree`` accepts the full mathematical domain ``L >= 1``.
    Some public family fixtures intentionally carry no stored node topology.
    Those trees are uninspectable, not zero-leaf C-Trees, so declarations and
    representation constraints apply without inventing ``L = 0``.
    """

    axis_payload = dict(axis or {})
    backend_payload = dict(backend_config or {})
    backend_metadata = dict(backend_payload.get("metadata") or {})
    representation = str(
        axis_payload.get("representation") or backend_metadata.get("representation") or ""
    ).strip()
    declared = _optional_leaf_count(axis_payload.get("leaf_count"))
    if declared is not None and declared < 1:
        raise ValueError(f"axis.leaf_count must be >= 1; got {declared}")

    if representation in _SINGLETON_REPRESENTATIONS:
        required_min = required_max = 1
        if declared is not None and declared != 1:
            raise ValueError(
                f"representation={representation!r} requires axis.leaf_count=1; got {declared}"
            )
        declared = 1
    elif representation in _RECURSIVE_REPRESENTATIONS:
        required_min, required_max = 2, None
        if declared is not None and declared < 2:
            raise ValueError(
                f"representation={representation!r} requires axis.leaf_count >= 2; got {declared}"
            )
    else:
        # Legacy ``ctree``, omitted labels, and application-defined labels all
        # share the package-wide non-empty C-Tree domain.
        required_min, required_max = 1, None

    train_observed = _validate_split(
        "train",
        train_trees,
        representation=representation,
        required_min=required_min,
        required_max=required_max,
        declared_leaf_count=declared,
    )
    eval_observed = _validate_split(
        "eval",
        eval_trees,
        representation=representation,
        required_min=required_min,
        required_max=required_max,
        declared_leaf_count=declared,
    )
    all_observed = train_observed + eval_observed
    all_effective = tuple(count if count is not None else declared for count in all_observed)
    train_effective = tuple(count if count is not None else declared for count in train_observed)
    observed_counts = tuple(count for count in all_observed if count is not None)

    fully_resolved = bool(all_effective) and all(count is not None for count in all_effective)
    if fully_resolved:
        resolved_counts = tuple(int(count) for count in all_effective if count is not None)
        singleton = all(count == 1 for count in resolved_counts)
        composition_present = any(count > 1 for count in resolved_counts)
        topology_kind = (
            "singleton"
            if singleton
            else "recursive"
            if all(count >= 2 for count in resolved_counts)
            else "mixed"
        )
    else:
        singleton = representation in _SINGLETON_REPRESENTATIONS or (
            not all_effective and declared == 1
        )
        composition_present = any(count is not None and count > 1 for count in all_effective) or (
            representation in _RECURSIVE_REPRESENTATIONS and bool(all_effective)
        )
        if singleton:
            topology_kind = "singleton"
        elif composition_present:
            topology_kind = "recursive"
        elif any(count is not None for count in all_effective):
            topology_kind = "partially_observed"
        else:
            topology_kind = "unspecified"

    training_composition_present = any(
        count is not None and count > 1 for count in train_effective
    ) or (representation in _RECURSIVE_REPRESENTATIONS and bool(train_effective))
    all_singleton_training = bool(train_effective) and all(count == 1 for count in train_effective)
    uniform_leaf_count = declared
    if uniform_leaf_count is None and fully_resolved:
        distinct = {int(count) for count in all_effective if count is not None}
        if len(distinct) == 1:
            uniform_leaf_count = distinct.pop()

    return TopologyContract(
        representation=representation or None,
        topology_kind=topology_kind,
        declared_leaf_count=declared,
        leaf_g_application_count_per_tree=uniform_leaf_count,
        merge_application_count_per_tree=(
            None if uniform_leaf_count is None else max(0, uniform_leaf_count - 1)
        ),
        singleton=singleton,
        composition_present=composition_present,
        training_composition_present=training_composition_present,
        all_singleton_training=all_singleton_training,
        train_tree_count=len(train_trees),
        eval_tree_count=len(eval_trees),
        observed_leaf_count_min=min(observed_counts) if observed_counts else None,
        observed_leaf_count_max=max(observed_counts) if observed_counts else None,
    )


def _validate_split(
    split_name: str,
    trees: Sequence[Any],
    *,
    representation: str,
    required_min: int,
    required_max: int | None,
    declared_leaf_count: int | None,
) -> tuple[int | None, ...]:
    counts: list[int | None] = []
    for index, tree in enumerate(trees):
        tree_id = getattr(tree, "tree_id", index)
        observed = _known_leaf_count(tree)
        if observed is None:
            counts.append(None)
            continue
        if observed < required_min or (required_max is not None and observed > required_max):
            expected = (
                f"exactly {required_min}"
                if required_max == required_min
                else f"at least {required_min}"
            )
            label = representation or "C-Tree"
            raise ValueError(
                f"representation={label!r} requires {expected} exposed leaves; "
                f"{split_name} tree {tree_id!r} exposes {observed}"
            )
        if declared_leaf_count is not None and observed != declared_leaf_count:
            raise ValueError(
                f"axis.leaf_count={declared_leaf_count} disagrees with "
                f"{split_name} tree {tree_id!r}, which exposes {observed} leaves"
            )
        counts.append(observed)
    return tuple(counts)


def _known_leaf_count(tree: Any) -> int | None:
    """Return an explicit/inspectable count, never infer zero from no nodes."""

    candidates: list[int] = []
    for value in (
        getattr(tree, "leaf_count", None),
        dict(getattr(tree, "metadata", None) or {}).get("leaf_count"),
    ):
        if value is not None and value != "":
            candidates.append(int(value))
    leaves = tuple(tree_leaves(tree) or ())
    if leaves:
        candidates.append(len(leaves))
    if not candidates:
        return None
    if len(set(candidates)) != 1:
        raise ValueError(f"tree leaf-count declarations disagree: {candidates!r}")
    return candidates[0]


def _optional_leaf_count(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


__all__ = ["TopologyContract", "resolve_topology_contract"]
