"""Tree scaffold for LLM-side RILE pipelines (f / g / head).

Given a manifesto document:

1. Chunk its text into leaf spans at a fixed token budget.
2. For each leaf and each internal node of a balanced binary merge tree,
   compute the character span and look up the corresponding Manifesto
   per-span RILE from `src.tasks.manifesto.rile_codes.RILECorpusIndex`.

The resulting `RILETreeScaffold` is the data backbone for:

* `ManifestoRileTreeOracle` — yields `TreeExample`s that carry per-leaf
  text + per-node RILE targets so the DSPy-GEPA / TRL-GRPO trainers can
  enforce strict C1 (per-leaf), C2 (merge commutativity), C3 (per-merge)
  in addition to root supervision.
* `dspy_rile_tree_metric` — turns per-node RILE predictions from a
  tree-structured DSPy program into a compound GEPA reward + feedback.

This module intentionally avoids any DSPy / TRL imports so it can be
unit-tested standalone without a running vLLM endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


# ---------------------------------------------------------------------------
# Data containers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RILETreeLeaf:
    index: int
    text: str
    start_char: int
    end_char: int
    rile_target: Optional[float]  # None → no coded span coverage


@dataclass(frozen=True)
class RILETreeInternal:
    index: int
    left: int  # index into the combined nodes list (leaves + internals)
    right: int
    start_char: int
    end_char: int
    rile_target: Optional[float]


@dataclass(frozen=True)
class RILETreeScaffold:
    """All the per-node metadata an LLM RILE pipeline needs for strict laws."""

    doc_id: str
    root_rile: float  # the document-level target (from ManifestoSample.rile)
    leaves: tuple[RILETreeLeaf, ...]
    internals: tuple[RILETreeInternal, ...]

    @property
    def root_index(self) -> int:
        """Index (within `leaves + internals`) of the root node.

        The root is always the last internal when there are any merges. For
        degenerate single-leaf docs there are no internals and the root is
        the single leaf.
        """
        if self.internals:
            return len(self.leaves) + len(self.internals) - 1
        return 0

    @property
    def n_nodes(self) -> int:
        return len(self.leaves) + len(self.internals)


# ---------------------------------------------------------------------------
# Scaffold builder.
# ---------------------------------------------------------------------------


def _balanced_internals(
    leaf_positions: Sequence[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Return `[(left_idx, right_idx, start_char, end_char), ...]` for a
    balanced binary merge tree over the given leaf char ranges.

    Indices are into the combined `leaves + internals` list (leaves come
    first). Odd levels carry the unpaired leaf up to the next level.
    """
    current: list[tuple[int, int, int]] = [
        (idx, start, end) for idx, (start, end) in enumerate(leaf_positions)
    ]
    internals: list[tuple[int, int, int, int]] = []
    next_idx = len(leaf_positions)
    while len(current) > 1:
        nxt: list[tuple[int, int, int]] = []
        i = 0
        while i < len(current):
            if i + 1 >= len(current):
                # Unpaired leaf/subtree carries to the next level unchanged.
                nxt.append(current[i])
                i += 1
                continue
            left = current[i]
            right = current[i + 1]
            start_char = min(left[1], right[1])
            end_char = max(left[2], right[2])
            internals.append((left[0], right[0], start_char, end_char))
            nxt.append((next_idx, start_char, end_char))
            next_idx += 1
            i += 2
        current = nxt
    return internals


def build_rile_tree_scaffold(
    *,
    doc_id: str,
    text: str,
    root_rile: float,
    chunk_texts_with_spans: Sequence[tuple[str, int, int]],
    span_rile_fn: Optional[Callable[[int, int], Optional[float]]],
) -> RILETreeScaffold:
    """Assemble a scaffold from pre-chunked leaf spans + optional span RILE.

    `chunk_texts_with_spans` is the output of the same chunker the
    embedding-FNO path uses (`_token_leaf_chunks` in
    `realdoc/embedding_fno_training.py`). When `span_rile_fn` is None,
    `rile_target` is None on every node — the soft/root-only path.
    """
    del text  # reserved for future callers that want the full joined text

    def _fetch(start: int, end: int) -> Optional[float]:
        if span_rile_fn is None:
            return None
        return span_rile_fn(int(start), int(end))

    leaves = tuple(
        RILETreeLeaf(
            index=idx,
            text=str(chunk_text),
            start_char=int(start),
            end_char=int(end),
            rile_target=_fetch(int(start), int(end)),
        )
        for idx, (chunk_text, start, end) in enumerate(chunk_texts_with_spans)
    )
    leaf_positions = tuple((leaf.start_char, leaf.end_char) for leaf in leaves)
    internal_raw = _balanced_internals(leaf_positions)
    internals = tuple(
        RILETreeInternal(
            index=len(leaves) + idx,
            left=int(left),
            right=int(right),
            start_char=int(start),
            end_char=int(end),
            rile_target=_fetch(int(start), int(end)),
        )
        for idx, (left, right, start, end) in enumerate(internal_raw)
    )
    return RILETreeScaffold(
        doc_id=str(doc_id),
        root_rile=float(root_rile),
        leaves=leaves,
        internals=internals,
    )


# ---------------------------------------------------------------------------
# Per-node RILE metric for LLM pipelines (DSPy-GEPA / TRL-GRPO).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerNodeRilePrediction:
    """A single LLM prediction at one node of the tree."""

    node_index: int
    predicted_rile: float


@dataclass
class RileTreeReward:
    score: float  # combined reward in [0, 1]
    feedback: str  # human-readable multi-line summary for GEPA reflection
    root_mae: float
    leaf_mae: float  # averaged over leaves with a strict target
    merge_mae: float  # averaged over internal nodes with a strict target
    commutativity_mae: float  # mean |f(merge(a,b)) - f(merge(b,a))|
    n_leaves_strict: int
    n_merges_strict: int
    n_commutativity_samples: int


def _mae_to_reward(mae: float, scale: float = 200.0) -> float:
    if mae != mae:  # NaN
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(mae) / float(scale)))


def _collect_mae(
    *,
    preds_by_index: dict[int, float],
    node_targets: Sequence[tuple[int, Optional[float]]],
) -> tuple[float, int]:
    """Mean absolute error over nodes that have both a prediction AND a target."""
    errors: list[float] = []
    for index, target in node_targets:
        if target is None:
            continue
        pred = preds_by_index.get(int(index))
        if pred is None:
            continue
        errors.append(abs(float(pred) - float(target)))
    if not errors:
        return float("nan"), 0
    return float(sum(errors) / len(errors)), len(errors)


DEFAULT_LOCAL_LAW_WEIGHT: float = 0.3
"""Canonical λ balancing root vs. local-law objectives.

The composite reward is `(1 - λ) · r_root + λ · Σᵢ ρᵢ · r_i / Σρ` with ρᵢ the
relative weights across active local laws. Default matches the canonical
Markov config used in `scripts/run_markov_publication_bundle.py` and the
Lean-aligned simulation tests (local_law_weight=0.3).
"""


def rile_tree_reward(
    *,
    scaffold: RILETreeScaffold,
    root_prediction: float,
    leaf_predictions: Sequence[PerNodeRilePrediction] = (),
    merge_predictions: Sequence[PerNodeRilePrediction] = (),
    commutativity_pairs: Sequence[tuple[float, float]] = (),
    local_law_weight: float = DEFAULT_LOCAL_LAW_WEIGHT,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
) -> RileTreeReward:
    """Compound reward: `(1 - λ) · r_root + λ · Σ ρᵢ · r_i / Σ ρ` over active local laws.

    * **Root** (always active): `r_root = 1 - |root_pred − root_target| / 200`.
    * **C1** (per-leaf, strict-only): `1 - mean|leaf_pred − leaf_target| / 200`
      averaged over leaves that have a coded RILE target.
    * **C3** (per-merge, strict-only): same as C1 but over internal nodes.
    * **C2** (merge commutativity): `1 - mean|merge(a,b) − merge(b,a)| / 200`
      over provided commutativity pairs. RILE is permutation-invariant so a
      well-calibrated merger must be commutative at the prediction level.

    `local_law_weight` (λ ∈ [0, 1]) balances root vs. local laws. The three
    `*_relative_weight` knobs balance the local laws against each other; by
    default they're equal (ρ_C1 = ρ_C2 = ρ_C3 = 1). Components with no data
    drop out of the ρ sum so the λ portion always represents the share the
    caller intended. When no local law has data, the reward reduces to
    `r_root`.
    """
    preds_leaf = {int(p.node_index): float(p.predicted_rile) for p in leaf_predictions}
    preds_merge = {int(p.node_index): float(p.predicted_rile) for p in merge_predictions}

    root_mae = abs(float(root_prediction) - float(scaffold.root_rile))
    root_reward = _mae_to_reward(root_mae)

    leaf_targets = [(leaf.index, leaf.rile_target) for leaf in scaffold.leaves]
    leaf_mae, n_leaves_strict = _collect_mae(
        preds_by_index=preds_leaf, node_targets=leaf_targets
    )
    leaf_reward = _mae_to_reward(leaf_mae) if n_leaves_strict else None

    merge_targets = [(m.index, m.rile_target) for m in scaffold.internals]
    merge_mae, n_merges_strict = _collect_mae(
        preds_by_index=preds_merge, node_targets=merge_targets
    )
    merge_reward = _mae_to_reward(merge_mae) if n_merges_strict else None

    commutativity_errors = [
        abs(float(ab) - float(ba)) for ab, ba in commutativity_pairs
    ]
    commutativity_mae = (
        float(sum(commutativity_errors) / len(commutativity_errors))
        if commutativity_errors
        else float("nan")
    )
    commutativity_reward = (
        _mae_to_reward(commutativity_mae) if commutativity_errors else None
    )

    # Normalize ρ over the ACTIVE local laws only — missing components drop
    # out so the λ portion always represents the share the caller intended.
    active: list[tuple[str, float, float]] = []
    if leaf_reward is not None:
        active.append(("c1", float(c1_relative_weight), float(leaf_reward)))
    if commutativity_reward is not None:
        active.append(("c2", float(c2_relative_weight), float(commutativity_reward)))
    if merge_reward is not None:
        active.append(("c3", float(c3_relative_weight), float(merge_reward)))
    rho_total = sum(rho for _name, rho, _r in active)

    lam = float(local_law_weight)
    lam = max(0.0, min(1.0, lam))
    if not active or rho_total <= 0.0:
        # No local law had data (or all ρ were zeroed) → collapse to root.
        score = float(root_reward)
    else:
        local_component = sum(
            (rho / rho_total) * reward for _name, rho, reward in active
        )
        score = float((1.0 - lam) * root_reward + lam * local_component)

    def _fmt_opt(value: float) -> str:
        if value != value:
            return "n/a"
        return f"{value:.2f}"

    feedback_lines = [
        f"RILE root: pred={float(root_prediction):.2f} target={float(scaffold.root_rile):.2f} err={root_mae:.2f}",
    ]
    if n_leaves_strict:
        feedback_lines.append(
            f"C1 per-leaf MAE over {n_leaves_strict} coded leaves: {_fmt_opt(leaf_mae)}"
        )
    else:
        feedback_lines.append(
            "C1: no coded leaf targets for this doc (skipped)"
        )
    if commutativity_errors:
        feedback_lines.append(
            f"C2 merge commutativity MAE over {len(commutativity_errors)} pairs: {_fmt_opt(commutativity_mae)}"
        )
    else:
        feedback_lines.append("C2: no commutativity probe provided (skipped)")
    if n_merges_strict:
        feedback_lines.append(
            f"C3 per-merge MAE over {n_merges_strict} coded merges: {_fmt_opt(merge_mae)}"
        )
    else:
        feedback_lines.append(
            "C3: no coded merge targets for this doc (skipped)"
        )
    feedback_lines.append(
        "Goal: keep all four aligned — leaves, merges, commutativity swap, and root must all track the coded RILE signal."
    )
    return RileTreeReward(
        score=float(score),
        feedback="\n".join(feedback_lines),
        root_mae=float(root_mae),
        leaf_mae=float(leaf_mae) if n_leaves_strict else float("nan"),
        merge_mae=float(merge_mae) if n_merges_strict else float("nan"),
        commutativity_mae=float(commutativity_mae),
        n_leaves_strict=int(n_leaves_strict),
        n_merges_strict=int(n_merges_strict),
        n_commutativity_samples=int(len(commutativity_errors)),
    )
