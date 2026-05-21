"""LDA / leaf-local-mixture oracle registration.

Lifts ``_true_doc_target`` from
:mod:`src.ctreepo.sim.core.leaf_local_mixture_utility` into the unified
oracle registry, exposing it as ``oracle:leaf_local_mixture_target``.

The native signature is preserved so the existing module can become a thin
re-export. The ``score_tree`` adapter expects a tree-like object with a
``.doc`` attribute carrying the synthetic ``LeafLocalMixtureDoc`` plus the
DGP parameters (theta, W_base, lambda_multiplier) on the tree's metadata or
on the doc itself.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from treepo._research.ctreepo.contracts import LEAF_UNIT_SYNTHETIC_ATOM

from . import OracleSpec, register_oracle


def leaf_local_mixture_target(
    doc: Any,
    *,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
) -> float:
    """Closed-form target for the leaf-local-mixture DGP.

    Computed as ``sum(_base_leaf_utilities(doc, theta, W_base, lambda_multiplier,
    latent_leaf_tokens=0))``. Lifted verbatim from the historical
    ``_true_doc_target`` helper; the original module re-exports this name.
    """
    # Imported lazily to avoid a circular import: the historical module
    # imports from many places at module load time.
    from treepo._research.ctreepo.sim.core.leaf_local_mixture_utility import _base_leaf_utilities

    return float(
        np.sum(
            _base_leaf_utilities(
                doc,
                theta=theta,
                W_base=W_base,
                lambda_multiplier=lambda_multiplier,
                latent_leaf_tokens=0,
            )
        )
    )


def _score_tree_lda(tree: Any) -> float:
    """Adapter: extract DGP parameters from the tree and compute the target.

    Looks for ``theta``, ``W_base``, ``lambda_multiplier`` either directly
    on the tree, in ``tree.metadata``, or on ``tree.doc``. The doc itself
    is taken from ``tree.doc`` (preferred) or ``tree`` itself if it already
    has the synthetic-doc surface.
    """
    metadata: Mapping[str, Any] = getattr(tree, "metadata", None) or {}
    doc = getattr(tree, "doc", None) or tree

    def _pick(key: str, default: Any = None) -> Any:
        for source in (tree, metadata, doc):
            if isinstance(source, Mapping):
                if key in source:
                    return source[key]
            else:
                value = getattr(source, key, None)
                if value is not None:
                    return value
        return default

    theta = _pick("theta")
    W_base = _pick("W_base")
    lam = _pick("lambda_multiplier", 1.0)
    if theta is None or W_base is None:
        raise AttributeError(
            "leaf_local_mixture_target.score_tree could not find theta/W_base "
            "on the tree, its metadata, or its doc"
        )
    return leaf_local_mixture_target(
        doc,
        theta=np.asarray(theta),
        W_base=np.asarray(W_base),
        lambda_multiplier=float(lam),
    )


register_oracle(
    OracleSpec(
        name="leaf_local_mixture_target",
        domain="lda",
        leaf_unit=LEAF_UNIT_SYNTHETIC_ATOM,
        f_callable=leaf_local_mixture_target,
        score_tree=_score_tree_lda,
        metadata={"output_kind": "scalar", "exactness": "closed_form"},
    )
)


__all__ = ["leaf_local_mixture_target"]
