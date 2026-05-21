"""Generalized optimization primitives for C-TreePO-style learning.

This subpackage is the "glue layer" between:

- a **compressor** ``g`` (often mergeable / tree-reducible),
- an **oracle/proxy** ``f*`` / ``f_hat`` (utility, scorer, reward model),
- a **policy** ``pi_theta`` (generator of candidates, trained via DPO/GRPO/PPO, etc.).

The core idea is that *preferences can be induced from any oracle-measurable loss*:
given two candidates ``a,b`` for an input ``x`` and an oracle target ``y* = f*(x)``,
prefer the candidate with smaller loss (or larger utility). This is exactly the
bridge between "benchmark sweeps" (utility/distance minimization) and preference
learning objectives like DPO.

The implementation here is intentionally backend-agnostic: it provides protocols
and record types plus small, dependency-light helpers. Adapters to the repo's
existing preference-learning stack live behind lazy imports.
"""

from __future__ import annotations

from .collect import collect_pairwise_preferences, collect_proxy_training_data
from .protocols import CandidateGenerator, Compressor, MergeableCompressor, ProxyOracle, Policy
from .preferences import (
    PreferenceOutcome,
    derive_preference_from_losses,
    derive_preference_from_scores,
    derive_preference_from_utilities,
)
from .records import PairwisePreference, SamplingMetadata
from .sklearn_proxy import SklearnProxyOracle
from .torch_proxy import TorchMSEProxyOracle
from .training_adapter import to_training_preference_dataset

__all__ = [
    "CandidateGenerator",
    "Compressor",
    "MergeableCompressor",
    "ProxyOracle",
    "Policy",
    "collect_pairwise_preferences",
    "collect_proxy_training_data",
    "PairwisePreference",
    "SamplingMetadata",
    "PreferenceOutcome",
    "derive_preference_from_losses",
    "derive_preference_from_scores",
    "derive_preference_from_utilities",
    "SklearnProxyOracle",
    "TorchMSEProxyOracle",
    "to_training_preference_dataset",
]
