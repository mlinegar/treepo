"""Leaf extraction and encoding for neural-operator families.

This is the messy data-prep half of the FNO stack: pull leaf text or numeric
tokens off arbitrary tree objects, coerce embeddings to a fixed width, and turn
numeric leaf-token groups into padded feature tensors. The family runtime stays
readable by delegating all of this shaping here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _tree_leaves(tree: Any) -> tuple[Any, ...] | None:
    """Return the leaf sequence of any tree-like value.

    Accepts a ``leaves`` attribute holding a sequence (fixture trees), a
    callable ``leaves()`` method (``TreeRecord``), or a ``get_leaves()``
    method. Leaf order is the composition order the family merges in.
    """

    leaves = getattr(tree, "leaves", None)
    if callable(leaves):
        leaves = leaves()
    if leaves is None and callable(getattr(tree, "get_leaves", None)):
        leaves = tree.get_leaves()
    if leaves is None:
        return None
    out = tuple(leaves)
    return out if out else None


def _leaf_token_groups(tree: Any) -> list[Any] | None:
    leaves = _tree_leaves(tree)
    raw_groups = leaves if leaves else (tree,)
    groups: list[Any] = []
    for leaf in raw_groups:
        tokens = getattr(leaf, "tokens", None)
        if tokens is None and isinstance(leaf, Mapping):
            tokens = leaf.get("tokens")
        if tokens is None:
            return None
        try:
            group = tokens if hasattr(tokens, "dtype") else [int(token) for token in tokens]
        except (TypeError, ValueError):
            return None
        groups.append(group)
    return groups or [[]]


def _encode_numeric_leaf_features(
    groups: Sequence[Sequence[Any] | None],
    *,
    dim: int,
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    import numpy as np

    from treepo.methods._numeric_features import add_numeric_sequence_features

    dim = max(1, int(dim))
    materialized = []
    for group in groups:
        if group is None or len(group) == 0:
            materialized.append([()])
        else:
            materialized.append(list(group))
    lengths = [max(1, len(group)) for group in materialized]
    max_leaves = max(1, max(lengths))
    features = np.zeros((len(materialized) * max_leaves, dim), dtype=np.float32)
    for tree_idx, group in enumerate(materialized):
        for leaf_idx, tokens in enumerate(group[:max_leaves]):
            if len(tokens) == 0:
                continue
            row = tree_idx * max_leaves + leaf_idx
            add_numeric_sequence_features(features[row], tokens, dim=dim, np=np)
    x_flat = torch.as_tensor(features, dtype=torch.float32, device=device)
    denom = torch.clamp(x_flat.sum(dim=-1, keepdim=True), min=1.0)
    x_flat = x_flat / torch.sqrt(denom)
    x = x_flat.reshape(len(materialized), max_leaves, dim)
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    return x, length_tensor


def _tree_sequence_cache_key(trees: Sequence[Any], *, dim: int, device: str) -> tuple[Any, ...]:
    if not trees:
        return ("empty", int(dim), str(device))
    ids = tuple(id(tree) for tree in trees)
    return (len(ids), ids[0], ids[-1], hash(ids), int(dim), str(device))


def _leaf_texts(tree: Any) -> list[str]:
    leaves = _tree_leaves(tree)
    if leaves:
        return [_object_text(leaf) for leaf in leaves]
    return [_object_text(tree)]


def _object_text(value: Any) -> str:
    for attr in ("text", "content", "summary", "tokens"):
        candidate = getattr(value, attr, None)
        if candidate is not None:
            return _text_from_value(candidate)
    if isinstance(value, Mapping):
        for key in ("text", "content", "summary", "tokens"):
            if key in value:
                return _text_from_value(value[key])
    return _text_from_value(value)


def _text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key}:{_text_from_value(val)}" for key, val in sorted(value.items()))
    if isinstance(value, Sequence):
        return " ".join(str(item) for item in value)
    return str(value)


def _coerce_embedding(vector: Sequence[float], dim: int) -> list[float]:
    values = [float(x) for x in vector]
    dim = max(1, int(dim))
    if len(values) < dim:
        values.extend([0.0] * (dim - len(values)))
    return values[:dim]


__all__ = [
    "_coerce_embedding",
    "_encode_numeric_leaf_features",
    "_leaf_texts",
    "_leaf_token_groups",
    "_object_text",
    "_text_from_value",
    "_tree_leaves",
    "_tree_sequence_cache_key",
]
