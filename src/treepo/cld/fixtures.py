"""Real-data fixture builders for exercising ``treepo.cld.run`` / ``fit``.

Thin wrappers around the synthetic-DGP samplers in
``src/ctreepo/sim/core/`` that wrap each generated document as a
tree-like object whose attributes match what the registered oracles'
``score_tree`` adapters expect. ``metadata['teacher_score_1_7']`` carries
the closed-form truth so the alternating-loop evaluator can pair
predictions against it (MAE → 0 when the oracle scores itself, finite
when a sketch or learned family stands in).

Two fixtures cover the v1 paper exercises:

- :func:`make_leaf_local_mixture_trees` for the LDA / leaf-local-mixture
  oracle (``"lda"`` domain).
- :func:`make_hll_token_trees` for the HLL/Count-Min/Theta oracles and
  classical sketches (``"classical_sketch"`` domain).

Adding a fixture for a new oracle domain (e.g. ``"markov"``) means one
new function here plus one line in
``_ORACLE_DOMAIN_FIXTURES`` in :mod:`treepo.cld.methods`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Tuple

import numpy as np

from treepo._research.ctreepo.oracles.lda import leaf_local_mixture_target
from treepo._research.ctreepo.sim.core.leaf_local_mixture_utility import (
    LeafLocalMixtureDoc,
    LeafLocalMixtureUtilityConfig,
    sample_leaf_local_mixture_utility_world,
)


# Fixture caches.
#
# Grid runs in the paper code reuse the SAME synthetic world across many
# cells (e.g. one LDA world, swept over evaluators / methods / families).
# Re-sampling the world per cell would dominate runtime on big grids.
#
# We memoize the public fixture builders on their full argument tuples.
# Both fixtures have hashable inputs (primitives or frozen dataclasses),
# so :func:`functools.lru_cache` is sufficient — no on-disk cache, no
# fcntl locking, no pickle. ``maxsize=32`` is generous for typical
# grids that vary along ~2 fixture axes.
#
# Mutation safety: the trees returned are read-only at every site
# inside treepo.cld and ``src/ctreepo`` (oracle ``score_tree`` adapters,
# ``ClassicalSketchFamilyRuntime.score_roots_with_f``,
# ``run_alternating_family``'s evaluator). If a custom family that
# mutates trees is added, call ``make_*.cache_clear()`` between cells.


# --------------------------------------------------------------------------- #
# LDA / leaf-local-mixture fixture
# --------------------------------------------------------------------------- #


@dataclass
class LDADocTree:
    """Tree-shaped wrapper for an :class:`LeafLocalMixtureDoc`.

    The oracle's ``score_tree`` adapter pulls DGP parameters off
    ``.theta`` / ``.W_base`` / ``.lambda_multiplier`` and the document
    off ``.doc``. ``.metadata`` carries the precomputed scalar target so
    the evaluator can score predictions against truth.
    """

    doc: LeafLocalMixtureDoc
    theta: np.ndarray
    W_base: np.ndarray
    lambda_multiplier: float
    metadata: Mapping[str, Any]


@functools.lru_cache(maxsize=32)
def make_leaf_local_mixture_trees(
    *,
    config: LeafLocalMixtureUtilityConfig | None = None,
    seed: int = 0,
    split: str = "test",
) -> Tuple[List[LDADocTree], LeafLocalMixtureUtilityConfig]:
    """Generate trees for the leaf-local-mixture LDA exercise.

    Default config keeps tunable knobs above ``_validate_config``'s
    floor in ``leaf_local_mixture_utility.py``; do not shrink further
    without re-reading that validator.
    """
    cfg = config or LeafLocalMixtureUtilityConfig(
        n_topics=4,
        vocab_size=64,
        doc_tokens=64,
        atomic_block_tokens=16,
        latent_leaf_tokens=16,
        leaf_fraction=0.25,
        utility_dim=4,
        anchor_words_per_topic=4,  # n_topics * anchors < vocab_size
        train_docs=1,
        val_docs=0,
        test_docs=8,
        relevant_topics=2,
        seed=int(seed),
    )
    world = sample_leaf_local_mixture_utility_world(cfg)
    docs = {"train": world.docs_train, "val": world.docs_val, "test": world.docs_test}[split]

    theta = np.asarray(world.theta_true, dtype=np.float64)
    W_base = np.asarray(world.W_base, dtype=np.float64)
    lam = float(cfg.lambda_multiplier)

    trees: List[LDADocTree] = []
    for doc in docs:
        truth = float(
            leaf_local_mixture_target(doc, theta=theta, W_base=W_base, lambda_multiplier=lam)
        )
        trees.append(
            LDADocTree(
                doc=doc,
                theta=theta,
                W_base=W_base,
                lambda_multiplier=lam,
                metadata={
                    "split": split,
                    "teacher_score_1_7": truth,
                    "teacher_score_native": truth,
                    "expert_score_1_7": truth,
                    "expert_score_native": truth,
                    "expert_target_scale": "raw",
                    "expert_score_for_objective": truth,
                },
            )
        )
    return trees, cfg


# --------------------------------------------------------------------------- #
# HLL / cardinality fixture
# --------------------------------------------------------------------------- #


@dataclass
class _HLLLeaf:
    tokens: Tuple[int, ...]


@dataclass
class HLLTokenTree:
    """Tree-shaped object with per-leaf token sequences and an exact
    distinct-count teacher on metadata."""

    leaves: Tuple[_HLLLeaf, ...]
    tokens: Tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Markov change-point fixture
# --------------------------------------------------------------------------- #


@dataclass
class _MarkovChangepointTree:
    """Tree-shaped wrapper for a ``ChangepointMarkovDoc``.

    Matches the shape ``treepo.cld/tests/reproduction/test_markov_hll_grids.py``
    builds for the ``markov_changepoint_count`` oracle.
    """

    leaves: tuple
    token_regimes: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


def make_markov_changepoint_trees(
    *,
    n_regimes: int = 4,
    vocab_size: int = 96,
    min_tokens: int = 96,
    max_tokens: int = 96,
    min_segments: int = 2,
    max_segments: int = 5,
    min_seg_len: int = 8,
    max_seg_len: int = 32,
    train_docs: int = 120,
    test_docs: int = 60,
    sinkhorn_iters: int = 30,
    transition_log_std: float = 1.25,
    seed: int = 0,
    split: str = "test",
) -> List[_MarkovChangepointTree]:
    """Generate paper-DGP docs and wrap them for ``markov_changepoint_count``.

    Direct in-process port of the ``_markov_docs + _wrap_markov_docs_as_trees``
    pattern from ``test_markov_hll_grids.py``. All knobs default to the
    upstream ``MarkovChangepointConfig`` field defaults.
    """
    from types import SimpleNamespace
    from treepo._research.tree.markov_boundary_honesty_simulation import (  # type: ignore
        _make_transition_matrices,
    )
    from treepo._research.tree.markov_changepoint_honesty_simulation import (  # type: ignore
        MarkovChangepointConfig, generate_changepoint_docs,
    )

    cfg = MarkovChangepointConfig(
        n_regimes=int(n_regimes), vocab_size=int(vocab_size),
        min_tokens=int(min_tokens), max_tokens=int(max_tokens),
        min_segments=int(min_segments), max_segments=int(max_segments),
        min_seg_len=int(min_seg_len), max_seg_len=int(max_seg_len),
        train_docs=int(train_docs), test_docs=int(test_docs),
        sinkhorn_iters=int(sinkhorn_iters),
        transition_log_std=float(transition_log_std),
        seed=int(seed),
    )
    rng = np.random.default_rng(int(seed))
    transitions = _make_transition_matrices(
        n_classes=cfg.n_regimes, vocab_size=cfg.vocab_size,
        log_std=cfg.transition_log_std, sinkhorn_iters=cfg.sinkhorn_iters,
        rng=rng,
    )
    docs = generate_changepoint_docs(cfg, transitions=transitions)
    out: List[_MarkovChangepointTree] = []
    for doc in docs:
        truth = len(doc.true_boundaries)
        out.append(_MarkovChangepointTree(
            leaves=(SimpleNamespace(tokens=[]),),
            token_regimes=doc.token_regimes,
            metadata={
                "split": split,
                "teacher_score_1_7": float(truth),
                "teacher_score_native": float(truth),
                "expert_score_1_7": float(truth),
                "expert_score_native": float(truth),
                "expert_target_scale": "raw",
                "expert_score_for_objective": float(truth),
            },
        ))
    return out


# --------------------------------------------------------------------------- #
# HLL token-tree fixture
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=32)
def make_hll_token_trees(
    *,
    n_trees: int = 8,
    leaves_per_tree: int = 4,
    leaf_token_count: int = 16,
    vocabulary_size: int = 64,
    seed: int = 0,
    split: str = "test",
) -> List[HLLTokenTree]:
    """Generate deterministic token trees with precomputed exact unique counts."""
    if n_trees <= 0 or leaves_per_tree <= 0 or leaf_token_count <= 0:
        raise ValueError("n_trees, leaves_per_tree, leaf_token_count must be positive")
    rng = np.random.default_rng(int(seed))
    trees: List[HLLTokenTree] = []
    for _ in range(int(n_trees)):
        leaves: List[_HLLLeaf] = []
        all_tokens: List[int] = []
        for _leaf_idx in range(int(leaves_per_tree)):
            leaf_tokens = rng.integers(
                low=0, high=int(vocabulary_size), size=int(leaf_token_count)
            ).tolist()
            leaves.append(_HLLLeaf(tokens=tuple(int(t) for t in leaf_tokens)))
            all_tokens.extend(int(t) for t in leaf_tokens)
        exact_unique = int(len(set(all_tokens)))
        trees.append(
            HLLTokenTree(
                leaves=tuple(leaves),
                tokens=tuple(all_tokens),
                metadata={
                    "split": split,
                    "teacher_score_1_7": float(exact_unique),
                    "teacher_score_native": float(exact_unique),
                    "expert_score_1_7": float(exact_unique),
                    "expert_score_native": float(exact_unique),
                    "expert_target_scale": "raw",
                    "expert_score_for_objective": float(exact_unique),
                },
            )
        )
    return trees


__all__ = [
    "HLLTokenTree",
    "LDADocTree",
    "make_hll_token_trees",
    "make_leaf_local_mixture_trees",
]
