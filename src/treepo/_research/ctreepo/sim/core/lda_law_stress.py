"""
LDA-specific law-stress definitions: counterexample families and re-exports.

DGP-agnostic classification lives in ``law_stress_common``.  This module
provides LDA-specific exact counterexample families that demonstrate
designed failures of individual local laws (C1/C2/C3).

Family semantics (parallel to Markov's exact families):
- ``oracle``:            g = true topic mixture → C1≈0, C2≈0, C3≈0
- ``scrambled_topics``:  g permutes topic indices → C1 fails, C2 passes
- ``uniform_prior``:     g = uniform (1/K,...,1/K) → C1 fails, C2 passes, C3 passes
- ``adversarial_merge``: g trained on half the leaves → C1 ok, C3 fails on cross-boundary merges
"""
from __future__ import annotations

import numpy as np

from treepo._research.ctreepo.sim.core.law_stress_common import (  # noqa: F401
    DEFAULT_LAW_GAIN_THRESHOLD,
    DEFAULT_ROOT_RATIO_LIMIT,
    DEFAULT_SPREAD_GAIN_THRESHOLD,
    LawStressAssessment,
    classify_law_stress,
    infer_law_stress_failure_reason,
    law_bundle_score,
)

VALID_LDA_LAW_PACKAGES = (
    "root_only",
    "c1_only",
    "c3_only",
    "c1c3",
    "c2_only",
    "all_laws",
)

VALID_LDA_EXACT_FAMILIES = (
    "oracle",
    "scrambled_topics",
    "uniform_prior",
    "adversarial_merge",
)


def build_exact_family_calibrator(
    family: str,
    *,
    n_topics: int,
    seed: int = 0,
) -> dict:
    """Return a calibrator dict for the given exact counterexample family.

    Returns a dict with keys ``kind``, ``w``, ``b``, ``family`` that can be
    passed to ``_apply_calibrator`` in the LDA simulation.
    """
    if family not in VALID_LDA_EXACT_FAMILIES:
        raise ValueError(f"Unknown LDA exact family {family!r}; valid: {VALID_LDA_EXACT_FAMILIES}")

    K = int(n_topics)

    if family == "oracle":
        # Identity calibrator — when applied to true mixtures, C1/C2/C3 ≈ 0.
        # This is the "everything works" reference.
        return {
            "kind": "affine",
            "w": np.eye(K, dtype=np.float64),
            "b": np.zeros(K, dtype=np.float64),
            "family": "oracle",
        }

    if family == "scrambled_topics":
        # Random permutation matrix — breaks C1 (wrong mixture) but C2 passes
        # (applying the same permutation twice is the same as applying it once
        # on already-permuted data — the permutation is idempotent in the
        # sense that L1(P·x, x) is constant under reapplication of P).
        rng = np.random.default_rng(seed)
        perm = rng.permutation(K)
        # Avoid identity permutation
        while np.all(perm == np.arange(K)):
            perm = rng.permutation(K)
        P = np.zeros((K, K), dtype=np.float64)
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        return {
            "kind": "affine",
            "w": P,
            "b": np.zeros(K, dtype=np.float64),
            "family": "scrambled_topics",
        }

    if family == "uniform_prior":
        # Always returns uniform (1/K,...,1/K) regardless of input.
        # C1 fails (wrong mixture), C2 passes (uniform is a fixed point),
        # C3 passes (merging uniforms gives uniform, which matches).
        return {
            "kind": "affine",
            "w": np.zeros((K, K), dtype=np.float64),
            "b": np.full(K, 1.0 / K, dtype=np.float64),
            "family": "uniform_prior",
        }

    if family == "adversarial_merge":
        # Calibrator that zeros out half the topics.  This is designed so
        # leaf-level C1 might look OK for topics that are present, but merges
        # across sections with different active topics will fail (C3 breaks).
        # Specifically: project onto first K//2 topics and renormalize.
        mask = np.zeros((K, K), dtype=np.float64)
        half = max(1, K // 2)
        for i in range(half):
            mask[i, i] = 1.0
        return {
            "kind": "affine",
            "w": mask,
            "b": np.full(K, 1e-6, dtype=np.float64),  # small epsilon to avoid zero rows
            "family": "adversarial_merge",
        }

    raise ValueError(f"Unhandled LDA exact family: {family!r}")


__all__ = [
    "DEFAULT_LAW_GAIN_THRESHOLD",
    "DEFAULT_ROOT_RATIO_LIMIT",
    "DEFAULT_SPREAD_GAIN_THRESHOLD",
    "LawStressAssessment",
    "VALID_LDA_EXACT_FAMILIES",
    "VALID_LDA_LAW_PACKAGES",
    "build_exact_family_calibrator",
    "classify_law_stress",
    "infer_law_stress_failure_reason",
    "law_bundle_score",
]
