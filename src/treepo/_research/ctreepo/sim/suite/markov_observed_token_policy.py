"""Re-export from core for backward compatibility.

The canonical location is now src.ctreepo.sim.core.markov_observed_token_policy.
"""
from treepo._research.ctreepo.sim.core.markov_observed_token_policy import (  # noqa: F401
    MarkovObservedTokenPolicy,
    resolve_markov_observed_token_policy,
)

__all__ = [
    "MarkovObservedTokenPolicy",
    "resolve_markov_observed_token_policy",
]
