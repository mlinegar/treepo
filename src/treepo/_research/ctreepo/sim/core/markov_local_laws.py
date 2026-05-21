"""Shared Markov count-sketch local-law helpers.

The theorem-domain state is the Lean ``MarkovCountSketch`` witness encoded as
``[count / target_scale, first_one_hot, last_one_hot]``.  These helpers keep
the Python training lanes aligned with the theorem-facing local-law vocabulary:
L1 leaf preservation, L2 merge preservation, and L3 on-range idempotence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from treepo._research.core.local_law_adjustment import LocalLawObservation
from treepo._research.core.ops_checks import ApproxLocalLawsBundle, EvidenceStatus, LawKind


MARKOV_COUNT_SKETCH_LAW_SET_ID = "markov_count_sketch"


@dataclass(frozen=True)
class MarkovLocalLawRows:
    """Per-law local-law observation rows keyed by ``LawKind``."""

    rows_by_law: Mapping[LawKind, tuple[LocalLawObservation, ...]]
    supervision_mode: str
    law_set_id: str = MARKOV_COUNT_SKETCH_LAW_SET_ID

    def to_counts(self) -> dict[str, int]:
        return {law.law_id: int(len(rows)) for law, rows in self.rows_by_law.items()}

    def to_observed_counts(self) -> dict[str, int]:
        return {
            law.law_id: int(sum(1 for row in rows if bool(row.observed)))
            for law, rows in self.rows_by_law.items()
        }

    def to_propensity_means(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for law, rows in self.rows_by_law.items():
            values = [float(row.propensity) for row in rows]
            out[law.law_id] = float(np.mean(values)) if values else float("nan")
        return out

    def to_metadata(self) -> dict[str, Any]:
        return {
            "law_set_id": self.law_set_id,
            "supervision_mode": str(self.supervision_mode),
            "row_counts": self.to_counts(),
            "observed_counts": self.to_observed_counts(),
            "propensity_means": self.to_propensity_means(),
        }


def _resolve_n_regimes(
    block_by_token: Sequence[int],
    n_regimes: int | None,
) -> int:
    blocks = [int(x) for x in block_by_token]
    if not blocks:
        raise ValueError("block_by_token must be non-empty")
    out = int(max(blocks)) + 1 if n_regimes is None else int(n_regimes)
    if out <= 0:
        raise ValueError("n_regimes must be positive")
    if any(block < 0 or block >= out for block in blocks):
        raise ValueError("block_by_token contains regime ids outside [0, n_regimes)")
    return out


def markov_exact_sketch_from_tokens_np(
    tokens: np.ndarray | Sequence[Sequence[int]],
    *,
    block_by_token: Sequence[int],
    pad_id: int,
    target_scale: float,
    n_regimes: int | None = None,
) -> np.ndarray:
    """Encode token rows as exact Markov count sketches."""

    token_array = np.asarray(tokens, dtype=np.int64)
    if token_array.ndim != 2:
        raise ValueError(f"tokens must be 2D; got shape {tuple(token_array.shape)}")
    n_blocks = _resolve_n_regimes(block_by_token, n_regimes)
    block_map = np.asarray([int(x) for x in block_by_token], dtype=np.int64)
    rows: list[np.ndarray] = []
    for row in token_array:
        toks = [int(tok) for tok in row if int(tok) != int(pad_id)]
        if not toks:
            count_norm = 0.0
            first = 0
            last = 0
        else:
            blocks = block_map[np.asarray(toks, dtype=np.int64)]
            count = int(np.sum(blocks[1:] != blocks[:-1])) if blocks.size > 1 else 0
            count_norm = float(count) / float(target_scale)
            first = int(blocks[0])
            last = int(blocks[-1])
        first_oh = np.zeros((n_blocks,), dtype=np.float32)
        last_oh = np.zeros((n_blocks,), dtype=np.float32)
        first_oh[first] = 1.0
        last_oh[last] = 1.0
        rows.append(
            np.concatenate(
                [np.asarray([count_norm], dtype=np.float32), first_oh, last_oh],
                axis=0,
            )
        )
    return np.asarray(rows, dtype=np.float32)


def markov_canonical_project_np(
    states: np.ndarray,
    *,
    target_scale: float,
    n_regimes: int,
) -> np.ndarray:
    """Project sketch-shaped states onto canonical count/endpoint slots."""

    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim != 2 or int(arr.shape[1]) != 1 + 2 * int(n_regimes):
        raise ValueError(f"states must be (B, {1 + 2 * int(n_regimes)}); got {tuple(arr.shape)}")
    n = int(n_regimes)
    out = np.zeros_like(arr, dtype=np.float32)
    out[:, 0] = np.round(arr[:, 0] * float(target_scale)) / float(target_scale)
    first_idx = np.argmax(arr[:, 1 : 1 + n], axis=1)
    last_idx = np.argmax(arr[:, 1 + n : 1 + 2 * n], axis=1)
    out[np.arange(arr.shape[0]), 1 + first_idx] = 1.0
    out[np.arange(arr.shape[0]), 1 + n + last_idx] = 1.0
    return out


def markov_exact_merge_np(
    left_states: np.ndarray,
    right_states: np.ndarray,
    *,
    target_scale: float,
    n_regimes: int,
    canonicalize_inputs: bool = True,
) -> np.ndarray:
    """Merge exact/canonical Markov sketch states."""

    left = np.asarray(left_states, dtype=np.float32)
    right = np.asarray(right_states, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"left/right shape mismatch: {left.shape} vs {right.shape}")
    n = int(n_regimes)
    expected_dim = 1 + 2 * n
    if left.ndim != 2 or int(left.shape[1]) != expected_dim:
        raise ValueError(f"states must be (B, {expected_dim}); got {tuple(left.shape)}")
    if canonicalize_inputs:
        left = markov_canonical_project_np(left, target_scale=target_scale, n_regimes=n)
        right = markov_canonical_project_np(right, target_scale=target_scale, n_regimes=n)
    left_count = left[:, 0]
    right_count = right[:, 0]
    left_first = left[:, 1 : 1 + n]
    left_last = left[:, 1 + n : 1 + 2 * n]
    right_first = right[:, 1 : 1 + n]
    right_last = right[:, 1 + n : 1 + 2 * n]
    same_boundary = np.sum(left_last * right_first, axis=1)
    join = 1.0 - same_boundary
    count = left_count + right_count + join / float(target_scale)
    return np.concatenate([count[:, None], left_first, right_last], axis=1).astype(np.float32)


def markov_contextual_decode_np(
    states: np.ndarray,
    *,
    context_left_raw: Sequence[Sequence[int]],
    context_right_raw: Sequence[Sequence[int]],
    block_by_token: Sequence[int],
    target_scale: float,
    n_regimes: int,
) -> np.ndarray:
    """Decode fixed two-sided contextual responses from Markov sketch states."""

    arr = np.asarray(states, dtype=np.float32)
    n = int(n_regimes)
    if arr.ndim != 2 or int(arr.shape[1]) != 1 + 2 * n:
        raise ValueError(f"states must be (B, {1 + 2 * n}); got {tuple(arr.shape)}")
    block_map = [int(x) for x in block_by_token]
    preds = np.zeros((int(arr.shape[0]), int(len(context_left_raw))), dtype=np.float32)
    count_norm = arr[:, 0]
    first = arr[:, 1 : 1 + n]
    last = arr[:, 1 + n : 1 + 2 * n]
    for ctx_idx, (left, right) in enumerate(zip(context_left_raw, context_right_raw, strict=True)):
        left_tokens = [int(tok) for tok in left]
        right_tokens = [int(tok) for tok in right]
        left_count = _exact_count_for_tokens(left_tokens, block_by_token=block_map)
        right_count = _exact_count_for_tokens(right_tokens, block_by_token=block_map)
        left_has = bool(left_tokens)
        right_has = bool(right_tokens)
        left_last = block_map[int(left_tokens[-1])] if left_has else 0
        right_first = block_map[int(right_tokens[0])] if right_has else 0
        left_boundary = (1.0 - first[:, left_last]) / float(target_scale) if left_has else 0.0
        right_boundary = (1.0 - last[:, right_first]) / float(target_scale) if right_has else 0.0
        preds[:, ctx_idx] = (
            count_norm
            + float(left_count + right_count) / float(target_scale)
            + left_boundary
            + right_boundary
        )
    return preds


def _exact_count_for_tokens(
    tokens: Sequence[int],
    *,
    block_by_token: Sequence[int],
) -> int:
    if len(tokens) <= 1:
        return 0
    blocks = [int(block_by_token[int(tok)]) for tok in tokens]
    return int(sum(1 for left, right in zip(blocks[:-1], blocks[1:]) if left != right))


def markov_approx_local_laws_bundle(
    *,
    leaf_pred: np.ndarray,
    leaf_target: np.ndarray,
    merge_pred: np.ndarray | None = None,
    merge_target: np.ndarray | None = None,
    idempotence_pred: np.ndarray | None = None,
    idempotence_target: np.ndarray | None = None,
) -> ApproxLocalLawsBundle:
    """Compute an approximate local-law bundle from sketch residuals."""

    def _mae(pred: np.ndarray | None, target: np.ndarray | None) -> float:
        if pred is None or target is None:
            return float("nan")
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        if p.shape != t.shape:
            raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
        return float(np.mean(np.abs(p - t))) if p.size else 0.0

    return ApproxLocalLawsBundle(
        eps_leaf=_mae(leaf_pred, leaf_target),
        eps_merge=_mae(merge_pred, merge_target),
        eps_idemp=_mae(idempotence_pred, idempotence_target),
        evidence_status=EvidenceStatus.APPROX_AUDITED,
        notes="Markov count-sketch residuals in canonical theorem-domain coordinates.",
    )


def markov_local_law_observation_rows(
    *,
    leaf_losses: Sequence[float],
    merge_losses: Sequence[float],
    idempotence_losses: Sequence[float],
    supervision_mode: str = "dense_exact",
    leaf_rate: float = 1.0,
    merge_rate: float = 1.0,
    idempotence_rate: float = 1.0,
    seed: int = 0,
    law_set_id: str = MARKOV_COUNT_SKETCH_LAW_SET_ID,
) -> MarkovLocalLawRows:
    """Build local-law observation rows with dense or sparse/IPW observation masks."""

    mode = str(supervision_mode)
    if mode == "dual":
        mode = "dense_exact"
    if mode not in {"dense_exact", "sparse_ipw"}:
        raise ValueError("supervision_mode must be dense_exact, sparse_ipw, or dual")
    rng = np.random.default_rng(int(seed))

    def _rows(
        losses: Sequence[float],
        *,
        law: LawKind,
        rate: float,
    ) -> tuple[LocalLawObservation, ...]:
        r = float(rate)
        if r < 0.0 or r > 1.0:
            raise ValueError(f"{law.law_id} rate must be in [0, 1], got {rate!r}")
        out: list[LocalLawObservation] = []
        for idx, loss in enumerate(losses):
            observed = True if mode == "dense_exact" else bool(rng.random() < r)
            propensity = 1.0 if mode == "dense_exact" else r
            out.append(
                LocalLawObservation(
                    proxy_loss=float(loss),
                    oracle_loss=float(loss) if observed else None,
                    observed=observed,
                    propensity=propensity,
                    depth=0,
                    node_weight=1.0,
                    metadata={
                        "law_id": law.law_id,
                        "law_kind": law.value,
                        "lean_name": law.lean_name,
                        "paper_condition": law.paper_condition,
                        "row_index": int(idx),
                    },
                )
            )
        return tuple(out)

    return MarkovLocalLawRows(
        rows_by_law={
            LawKind.L1_LEAF: _rows(
                leaf_losses,
                law=LawKind.L1_LEAF,
                rate=leaf_rate,
            ),
            LawKind.L2_MERGE: _rows(
                merge_losses,
                law=LawKind.L2_MERGE,
                rate=merge_rate,
            ),
            LawKind.L3_IDEMPOTENCE: _rows(
                idempotence_losses,
                law=LawKind.L3_IDEMPOTENCE,
                rate=idempotence_rate,
            ),
        },
        supervision_mode=str(supervision_mode),
        law_set_id=str(law_set_id),
    )


__all__ = [
    "MARKOV_COUNT_SKETCH_LAW_SET_ID",
    "MarkovLocalLawRows",
    "markov_approx_local_laws_bundle",
    "markov_canonical_project_np",
    "markov_contextual_decode_np",
    "markov_exact_merge_np",
    "markov_exact_sketch_from_tokens_np",
    "markov_local_law_observation_rows",
]
