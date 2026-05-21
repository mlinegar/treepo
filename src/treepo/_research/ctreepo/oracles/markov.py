"""Markov-domain oracle registration.

Lifts ``_changepoint_count`` and ``_oracle_count`` from
:mod:`src.ctreepo.sim.core.markov_changepoint_ops_count` into the unified
oracle registry, exposing them as ``oracle:markov_changepoint_count``.

The native signatures are preserved so existing call sites can become thin
re-exports. The ``score_tree`` adapter converts a synthetic-DGP tree into a
scalar root prediction by counting changepoints in its top-level regime
sequence.
"""

from __future__ import annotations

from typing import Any, Sequence

from treepo._research.ctreepo.contracts import LEAF_UNIT_SYNTHETIC_ATOM

from . import OracleSpec, register_oracle


def markov_changepoint_count(regimes: Sequence[int]) -> int:
    """Count of regime transitions in a flat sequence of regime ids.

    Mirrors the historical ``_changepoint_count`` helper used throughout the
    Markov simulations. Empty / single-element sequences return 0.
    """
    if len(regimes) < 2:
        return 0
    return int(
        sum(1 for a, b in zip(regimes[:-1], regimes[1:]) if int(a) != int(b))
    )


def markov_changepoint_count_for_doc(doc: Any, *, start: int = 0, end: int | None = None) -> int:
    """Apply :func:`markov_changepoint_count` to a slice of ``doc.token_regimes``."""
    regimes = list(doc.token_regimes)
    if end is None:
        end = len(regimes)
    return markov_changepoint_count(regimes[int(start):int(end)])


def _score_tree_markov(tree: Any) -> int:
    """Adapter: count changepoints over the tree's full token-regime range.

    Synthetic Markov trees expose either ``.token_regimes`` directly or a
    ``.doc`` with that attribute. Falls back to flattening leaves when only
    leaf-level regime fragments are available.
    """
    regimes = getattr(tree, "token_regimes", None)
    if regimes is None:
        doc = getattr(tree, "doc", None)
        if doc is not None:
            regimes = getattr(doc, "token_regimes", None)
    if regimes is not None:
        return markov_changepoint_count(regimes)
    leaves = getattr(tree, "leaves", None)
    if leaves is None:
        raise AttributeError(
            "markov_changepoint_count.score_tree expects tree.token_regimes, "
            "tree.doc.token_regimes, or tree.leaves with leaf.token_regimes"
        )
    flat: list[int] = []
    for leaf in leaves:
        leaf_regimes = getattr(leaf, "token_regimes", None)
        if leaf_regimes is None:
            continue
        flat.extend(int(r) for r in leaf_regimes)
    return markov_changepoint_count(flat)


register_oracle(
    OracleSpec(
        name="markov_changepoint_count",
        domain="markov",
        leaf_unit=LEAF_UNIT_SYNTHETIC_ATOM,
        f_callable=markov_changepoint_count,
        score_tree=_score_tree_markov,
        metadata={
            "output_kind": "scalar",
            "exactness": "exact",
            "secondary_callable": "markov_changepoint_count_for_doc",
        },
    )
)


__all__ = ["markov_changepoint_count", "markov_changepoint_count_for_doc"]
