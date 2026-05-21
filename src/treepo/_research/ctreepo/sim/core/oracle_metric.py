"""Theory-aligned oracle metric space for contrastive fiber preservation.

Mirrors the Lean formalization in FiberPreservingObjective.lean:

    def contrastiveFiberLoss (fstar : Strings → Y) (feature : Strings → Feature)
        (margin : ℝ) (x x' : Strings) : ℝ :=
      if dist (fstar x) (fstar x') = 0
      then dist (feature x) (feature x')
      else max 0 (margin - dist (feature x) (feature x'))

The key abstraction: Y is a BoundedMetricSpace. The oracle f* maps inputs to Y.
Pair classification uses continuous dist_Y, not discrete equivalence class keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F


@runtime_checkable
class OracleMetricSpace(Protocol):
    """Mirrors Lean: fstar : Strings → Y, [BoundedMetricSpace Y].

    Each DGP implements this protocol to define its oracle output space
    and the metric on it.  The protocol is DGP-agnostic: Markov, LDA,
    or any future DGP just provides oracle_vector and distance.
    """

    @property
    def oracle_dim(self) -> int:
        """Dimension of Y (the oracle output space).

        This is a *hint* for choosing phi_dim, not a constraint.
        The Lean places no restriction on Feature dimension vs Y dimension.
        """
        ...

    def oracle_vector(
        self,
        *,
        count: float,
        first: int | None = None,
        last: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Compute f*(x) as a real vector in R^oracle_dim."""
        ...

    def distance(self, y1: np.ndarray, y2: np.ndarray) -> float:
        """Compute dist_Y(y1, y2).  Must satisfy metric axioms."""
        ...

    def task_readout(self, y: np.ndarray) -> float:
        """Extract the supervised prediction target from an oracle vector."""
        ...


# ---------------------------------------------------------------------------
# Contrastive pair data (replaces TheoremFeaturePairSets for the metric path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContrastivePairData:
    """Pair classification with continuous oracle distances preserved.

    Unlike TheoremFeaturePairSets which only stores index pairs,
    this also carries the oracle distance for each pair — enabling
    optional distance-weighted push-apart losses.
    """

    same_pairs: tuple[tuple[int, int], ...]
    different_pairs: tuple[tuple[int, int], ...]
    same_distances: tuple[float, ...]
    different_distances: tuple[float, ...]


def build_contrastive_pairs(
    oracle_vectors: Sequence[np.ndarray],
    *,
    metric: OracleMetricSpace,
    same_threshold: float = 0.0,
    diff_threshold: float = 0.0,
) -> ContrastivePairData:
    """Build contrastive pair sets using continuous oracle distances.

    Mirrors the Lean's ``contrastiveFiberLoss`` branch condition:
    - same fiber:      dist_Y(y1, y2) <= same_threshold
    - different fiber:  dist_Y(y1, y2) >= diff_threshold
    - ambiguous:       same_threshold < dist_Y < diff_threshold  (excluded)

    With defaults (same_threshold=0, diff_threshold=0) the classification is:
    - same fiber:  dist_Y = 0  (exact match — the Lean's ``dist = 0``)
    - different:   dist_Y > 0  (any nonzero distance)
    """
    vectors = [np.asarray(v, dtype=np.float64) for v in oracle_vectors]
    n = len(vectors)

    same_pairs: list[tuple[int, int]] = []
    different_pairs: list[tuple[int, int]] = []
    same_distances: list[float] = []
    different_distances: list[float] = []

    effective_diff = max(float(diff_threshold), float(same_threshold))

    for i in range(n):
        for j in range(i + 1, n):
            d = float(metric.distance(vectors[i], vectors[j]))
            if d <= float(same_threshold):
                same_pairs.append((i, j))
                same_distances.append(d)
            elif d >= effective_diff:
                different_pairs.append((i, j))
                different_distances.append(d)
            # else: ambiguous pair — excluded from training

    return ContrastivePairData(
        same_pairs=tuple(same_pairs),
        different_pairs=tuple(different_pairs),
        same_distances=tuple(same_distances),
        different_distances=tuple(different_distances),
    )


# ---------------------------------------------------------------------------
# Contrastive fiber loss  (mirrors FiberPreservingObjective.lean)
# ---------------------------------------------------------------------------


def contrastive_fiber_loss(
    embeddings: torch.Tensor,
    pair_data: ContrastivePairData,
    *,
    margin: float = 0.5,
    weighted_push: bool = False,
) -> torch.Tensor:
    """Contrastive fiber loss mirroring the Lean's ``contrastiveFiberLoss``.

    Same pairs:  loss = 1 - cosine_sim(phi(x), phi(x'))
                 (pull phi embeddings together on same fibers)

    Different pairs:  loss = max(0, cosine_sim(phi(x), phi(x')) - margin_cos)
                      where margin_cos = 1 - margin  (converted from distance margin)
                 (push phi embeddings apart on different fibers)

    When ``weighted_push=True``, the different-pair loss is scaled by the
    oracle distance d_Y(y1, y2), so pairs that are farther apart in oracle
    space push harder.

    Cosine similarity on L2-normalized embeddings is a valid pseudometric
    (decision: keep current behavior, which is scale-invariant and well-tested).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2 [N, phi_dim]")
    if not pair_data.same_pairs and not pair_data.different_pairs:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    normalized = F.normalize(embeddings, dim=-1)
    same_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    diff_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    active_terms = 0
    if pair_data.same_pairs:
        same_index = torch.as_tensor(
            list(pair_data.same_pairs),
            device=embeddings.device,
            dtype=torch.long,
        )
        same_sim = torch.sum(
            normalized.index_select(0, same_index[:, 0])
            * normalized.index_select(0, same_index[:, 1]),
            dim=-1,
        )
        same_loss = (1.0 - same_sim).mean()
        active_terms += 1
    if pair_data.different_pairs:
        diff_index = torch.as_tensor(
            list(pair_data.different_pairs),
            device=embeddings.device,
            dtype=torch.long,
        )
        diff_sim = torch.sum(
            normalized.index_select(0, diff_index[:, 0])
            * normalized.index_select(0, diff_index[:, 1]),
            dim=-1,
        )
        diff_terms = F.relu(diff_sim - float(margin))
        if bool(weighted_push) and pair_data.different_distances:
            diff_weights = torch.as_tensor(
                list(pair_data.different_distances),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            diff_terms = diff_terms * diff_weights
        diff_loss = diff_terms.mean()
        active_terms += 1
    if active_terms <= 0:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    return (same_loss + diff_loss) / float(active_terms)


# ---------------------------------------------------------------------------
# Oracle metric registry (parallel to theorem_feature_route adapter registry)
# ---------------------------------------------------------------------------

_ORACLE_METRIC_REGISTRY: dict[str, type[OracleMetricSpace]] = {}


def register_oracle_metric(
    name: str,
    cls: type[OracleMetricSpace],
    *,
    overwrite: bool = False,
) -> None:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("oracle metric name must be non-empty")
    if normalized in _ORACLE_METRIC_REGISTRY and not overwrite:
        raise ValueError(f"oracle metric {normalized!r} is already registered")
    _ORACLE_METRIC_REGISTRY[normalized] = cls


def resolve_oracle_metric(name: str) -> OracleMetricSpace:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("oracle metric name must be non-empty")
    cls = _ORACLE_METRIC_REGISTRY.get(normalized)
    if cls is None:
        available = sorted(_ORACLE_METRIC_REGISTRY.keys())
        raise ValueError(
            f"unknown oracle metric {normalized!r}; available: {available}"
        )
    return cls()
