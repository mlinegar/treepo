from __future__ import annotations

from typing import Optional, Protocol, Sequence, TypeVar, runtime_checkable

X = TypeVar("X")  # inputs (documents, contexts, states, ...)
Z = TypeVar("Z")  # compressed representation (summary, sketch, embedding, ...)
Y = TypeVar("Y")  # oracle label/utility target
A = TypeVar("A")  # action/candidate output


@runtime_checkable
class Compressor(Protocol[X, Z]):
    """Compressor ``g``: maps an input into a smaller representation.

    In the current opt layer this is proxy-only by default. Flat compressors
    here do not imply Lean local-law structure unless a higher-level wrapper
    adds explicit encode/merge/decode semantics.
    """

    def compress(self, x: X) -> Z: ...


@runtime_checkable
class MergeableCompressor(Protocol[X, Z]):
    """Mergeable compressor: leaf encoding + merge operator.

    This interface alone is still insufficient for theorem-backed claims because
    it lacks decode / re-summary semantics.
    """

    def leaf(self, x: X) -> Z: ...
    def merge(self, left: Z, right: Z) -> Z: ...


@runtime_checkable
class ProxyOracle(Protocol[Z, Y]):
    """Proxy oracle ``f_hat``: predicts oracle targets from compressed reps."""

    def fit(
        self,
        inputs: Sequence[Z],
        targets: Sequence[Y],
        *,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "ProxyOracle[Z, Y]": ...

    def predict(self, inputs: Sequence[Z]) -> Sequence[Y]: ...


@runtime_checkable
class CandidateGenerator(Protocol[X, A]):
    """Exploration/generation mechanism producing candidates for comparison."""

    def generate(self, x: X, *, n: int, seed: Optional[int] = None) -> Sequence[A]: ...


@runtime_checkable
class Policy(CandidateGenerator[X, A], Protocol[X, A]):
    """Policy ``pi_theta``: generator with (optional) log-probability access."""

    def logprob(self, x: X, a: A) -> float: ...
