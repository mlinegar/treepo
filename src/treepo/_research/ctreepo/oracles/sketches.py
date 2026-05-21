"""Sketch-domain oracle registrations.

Three oracles for the classical-sketch / mergeable-ablation track:

- ``type_oracle``: per-type spike counts mod ``n_types``. Lifted from
  :mod:`src.tree.learned_sketch` so the same callable now lives in one place
  and is reachable via ``oracle:type_oracle``.
- ``hll_exact_count``: exact unique-token cardinality (the f* a learned HLL
  family is approximating). Native signature ``Callable[[Iterable[int]], int]``.
- ``hll_max_merge``: the lossy register-wise max merge (HLL's whole point).
  Registered as a named ``g_callable`` so users can opt into it explicitly via
  ``--g-init oracle:hll_max_merge``; it is NOT the universal sketch default.

The ``score_tree`` adapters bridge each oracle to the alternating ladder by
walking a tree-like structure and producing a scalar/vector prediction at the
root.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

from treepo._research.ctreepo.contracts import (
    LEAF_UNIT_STREAM_ITEM,
    LEAF_UNIT_SYNTHETIC_ATOM,
)

from . import OracleSpec, register_oracle


# ---------------------------------------------------------------------------
# type_oracle: per-type spike counts (lifted from treepo._research.tree.learned_sketch)
# ---------------------------------------------------------------------------

# Default spike threshold mirrors the historical inline constant pinned at
# src/tree/learned_sketch.py:60 to preserve existing call-site behavior.
SPIKE_THRESHOLD: float = 0.90


def type_oracle(
    indicators: Sequence[float],
    positions: Sequence[int],
    n_types: int,
    threshold: float = SPIKE_THRESHOLD,
) -> List[float]:
    """Per-type spike counts: position assigned type = ``pos % n_types``.

    Returns a list of ``n_types`` floats (count of spikes per type). This is
    the canonical f* used in the mergeable-ablation / learned-sketch sims.
    """
    counts = [0.0] * int(n_types)
    for val, pos in zip(indicators, positions):
        if float(val) >= float(threshold):
            t = int(pos) % int(n_types)
            counts[t] += 1.0
    return counts


def _score_tree_type_oracle(tree: Any, *, n_types: int, threshold: float = SPIKE_THRESHOLD) -> List[float]:
    """Adapter: walk a synthetic-doc tree and return per-type counts at root.

    Expected tree shape (synthetic-DGP convention used in
    ``learned_sketch_simulation``): ``tree.indicators`` and ``tree.positions``
    are flat sequences over the doc. Falls back to combining leaves if the
    tree exposes ``leaves: Sequence[(indicators, positions)]``.
    """
    indicators = getattr(tree, "indicators", None)
    positions = getattr(tree, "positions", None)
    if indicators is not None and positions is not None:
        return type_oracle(indicators, positions, int(n_types), float(threshold))
    leaves = getattr(tree, "leaves", None)
    if leaves is None:
        raise AttributeError(
            "type_oracle.score_tree expects either tree.indicators/tree.positions "
            "or tree.leaves: Sequence[(indicators, positions)]"
        )
    flat_ind: List[float] = []
    flat_pos: List[int] = []
    for leaf in leaves:
        leaf_ind = getattr(leaf, "indicators", None) or leaf[0]
        leaf_pos = getattr(leaf, "positions", None) or leaf[1]
        flat_ind.extend(float(v) for v in leaf_ind)
        flat_pos.extend(int(p) for p in leaf_pos)
    return type_oracle(flat_ind, flat_pos, int(n_types), float(threshold))


register_oracle(
    OracleSpec(
        name="type_oracle",
        domain="classical_sketch",
        leaf_unit=LEAF_UNIT_SYNTHETIC_ATOM,
        f_callable=type_oracle,
        score_tree=_score_tree_type_oracle,
        metadata={"output_kind": "vector", "spike_threshold_default": SPIKE_THRESHOLD},
    )
)


# ---------------------------------------------------------------------------
# hll_exact_count: ground-truth unique-token cardinality
# ---------------------------------------------------------------------------


def hll_exact_count(token_ids: Iterable[int]) -> int:
    """Exact unique-token count via Python ``set`` — the f* HLL approximates."""
    return len({int(t) for t in token_ids})


def _score_tree_hll_exact(tree: Any) -> int:
    """Adapter: collect tokens across all leaves and return exact unique count.

    Accepts either ``tree.tokens`` (flat) or ``tree.leaves`` whose entries
    expose ``.tokens`` (a la treepo.bench HLL trees).
    """
    tokens = getattr(tree, "tokens", None)
    if tokens is not None:
        return hll_exact_count(tokens)
    leaves = getattr(tree, "leaves", None)
    if leaves is None:
        raise AttributeError(
            "hll_exact_count.score_tree expects tree.tokens or tree.leaves with .tokens"
        )
    flat: List[int] = []
    for leaf in leaves:
        leaf_tokens = getattr(leaf, "tokens", None) or leaf
        flat.extend(int(t) for t in leaf_tokens)
    return hll_exact_count(flat)


register_oracle(
    OracleSpec(
        name="hll_exact",
        domain="classical_sketch",
        leaf_unit=LEAF_UNIT_STREAM_ITEM,
        f_callable=hll_exact_count,
        score_tree=_score_tree_hll_exact,
        metadata={"output_kind": "scalar", "exactness": "exact"},
    )
)


# ---------------------------------------------------------------------------
# hll_max_merge: the lossy native HLL register-wise max merge
# ---------------------------------------------------------------------------


def hll_max_merge(left: Any, right: Any) -> Any:
    """Register-wise max-merge of two HLL sketches.

    Imports lazily to avoid a hard dependency on the standalone ``treepo``
    package at registry import time. The native ``HyperLogLogSketch.merge``
    mutates left in-place and returns left; we copy first to keep merge a
    pure function suitable for tree reducers.
    """
    # Cheap pure-function shim around HyperLogLogSketch.merge.
    if hasattr(left, "copy") and hasattr(left, "merge"):
        return left.copy().merge(right)
    raise TypeError(
        "hll_max_merge expects HyperLogLogSketch-like inputs with .copy() and .merge()"
    )


register_oracle(
    OracleSpec(
        name="hll_max_merge",
        domain="classical_sketch",
        leaf_unit=LEAF_UNIT_STREAM_ITEM,
        # f_callable is a no-op for this g-only oracle — kept to satisfy the
        # required field; callers should use g_callable.
        f_callable=lambda *args, **kwargs: None,
        g_callable=hll_max_merge,
        metadata={
            "output_kind": "sketch_state",
            "lossy_native": True,
            "note": (
                "This is the lossy register-max merge; not a default. The "
                "lossless universal default for sketch g_init=raw_concat is "
                "ConcatSketch (held both children)."
            ),
        },
    )
)


__all__ = ["hll_exact_count", "hll_max_merge", "type_oracle", "SPIKE_THRESHOLD"]
