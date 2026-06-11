"""Research-only LDA fixture builders for legacy method regressions."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

import numpy as np

from treepo._research.ctreepo.oracles.lda import leaf_local_mixture_target
from treepo._research.ctreepo.sim.core.leaf_local_mixture_utility import (
    LeafLocalMixtureDoc,
    LeafLocalMixtureUtilityConfig,
    sample_leaf_local_mixture_utility_world,
)


@dataclass
class LDADocTree:
    """Tree-shaped wrapper for an LDA leaf-local-mixture document."""

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
    """Generate trees for the research-only leaf-local-mixture LDA exercise."""
    cfg = config or LeafLocalMixtureUtilityConfig(
        n_topics=4,
        vocab_size=64,
        doc_tokens=64,
        atomic_block_tokens=16,
        latent_leaf_tokens=16,
        leaf_fraction=0.25,
        utility_dim=4,
        anchor_words_per_topic=4,
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


__all__ = ["LDADocTree", "make_leaf_local_mixture_trees"]
