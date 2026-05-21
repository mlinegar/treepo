"""
Learn a merge operator over HyperLogLog (HLL) register state.

Goal: make the "mergeable sketches" story explicit and theory-linked.

We treat HLL as the classical mergeable sketch with a known asymptotic relative
standard error (RSE) floor:

  RSE_theory(p) ~= 1.04 / sqrt(m),  with m = 2^p registers.

This module simulates learning a projected/fiber-preserving merge family over
HLL register-shaped states using local, tree-node supervision:

  encode_leaf(x) = g_theta(x)
  merge(a, b) = g_theta(a + b)

The legacy direct state merger remains available as an ablation:

  M_theta(S_left, S_right)  ≈  max(S_left, S_right)   (elementwise)
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

    class _Module:  # pragma: no cover
        pass

    class _NNNamespace:  # pragma: no cover
        Module = _Module

    nn = _NNNamespace()  # type: ignore[assignment]

from treepo.common import AuditPolicyName, ScheduleName, VALID_SCHEDULES, audit_sample_count
from treepo.hll import (
    HLLConfig,
    HyperLogLogSketch,
    hll_relative_standard_error,
)
from treepo.training.local_law import (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    local_law_objective_target_mse,
    normalize_local_law_objective_mode,
)
from treepo.bench.weighting import (
    DEFAULT_WEIGHTING_MODES,
    WeightingMode,
    parse_weighting_modes,
    validate_legacy_weighting_mode,
    weighted_mean_ci95,
)

def _set_global_seed(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_rel_error(pred: float, truth: float) -> float:
    denom = max(1.0, float(truth))
    return (float(pred) - float(truth)) / denom


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError(
            "PyTorch is required for HLL merge-learning simulations. "
            "Install with: pip install 'treepo[torch]'"
        )


def _parse_csv_floats(xs: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(x) for x in xs)


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    if len(values) == 0:
        return float("nan")
    if len(values) != len(weights):
        raise ValueError("values and weights must align")
    qq = max(0.0, min(1.0, float(q)))
    vals = np.asarray(values, dtype=np.float64)
    ws = np.asarray(weights, dtype=np.float64)
    ws = np.maximum(ws, 0.0)
    wsum = float(np.sum(ws))
    if wsum <= 0.0:
        return float(np.percentile(vals, 100.0 * qq))
    order = np.argsort(vals)
    vals_s = vals[order]
    ws_s = ws[order]
    csum = np.cumsum(ws_s)
    idx = int(np.searchsorted(csum, qq * wsum, side="left"))
    idx = max(0, min(idx, len(vals_s) - 1))
    return float(vals_s[idx])


def _build_merge_metric_weighting_views(
    *,
    rel_sq: Sequence[float],
    abs_rel: Sequence[float],
    spreads: Sequence[float],
    token_weights: Sequence[float],
    leaf_weights: Sequence[float],
    weighting_modes: Sequence[WeightingMode],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in weighting_modes:
        if mode == WeightingMode.DOC:
            ws = [1.0 for _ in rel_sq]
        elif mode == WeightingMode.LEAF:
            ws = [float(x) for x in leaf_weights]
        elif mode == WeightingMode.TOKEN:
            ws = [float(x) for x in token_weights]
        else:  # pragma: no cover
            raise ValueError(f"unsupported weighting mode: {mode!r}")

        rel_sq_stats = weighted_mean_ci95(rel_sq, ws)
        abs_rel_stats = weighted_mean_ci95(abs_rel, ws)
        spread_mean_stats = weighted_mean_ci95(spreads, ws)
        rel_mean = max(0.0, float(rel_sq_stats["mean"]))
        rel_se_sq = max(0.0, float(rel_sq_stats["se"]))
        if rel_mean > 0.0:
            rel_se = 0.5 * rel_se_sq / math.sqrt(rel_mean)
        else:
            rel_se = 0.0
        out[mode.value] = {
            "relative_rmse": {
                "mean_hat": float(math.sqrt(rel_mean)),
                "se": float(rel_se),
                "ci95_low": float(math.sqrt(max(0.0, float(rel_sq_stats["ci95_low"])))),
                "ci95_high": float(math.sqrt(max(0.0, float(rel_sq_stats["ci95_high"])))),
                "weight_sum": float(rel_sq_stats["weight_sum"]),
                "effective_n": float(rel_sq_stats["effective_n"]),
            },
            "mean_abs_rel_error": {
                "mean_hat": float(abs_rel_stats["mean"]),
                "se": float(abs_rel_stats["se"]),
                "ci95_low": float(abs_rel_stats["ci95_low"]),
                "ci95_high": float(abs_rel_stats["ci95_high"]),
                "weight_sum": float(abs_rel_stats["weight_sum"]),
                "effective_n": float(abs_rel_stats["effective_n"]),
            },
            "schedule_spread_mean": {
                "mean_hat": float(spread_mean_stats["mean"]),
                "se": float(spread_mean_stats["se"]),
                "ci95_low": float(spread_mean_stats["ci95_low"]),
                "ci95_high": float(spread_mean_stats["ci95_high"]),
                "weight_sum": float(spread_mean_stats["weight_sum"]),
                "effective_n": float(spread_mean_stats["effective_n"]),
            },
            "schedule_spread_p95": {
                "mean_hat": float(_weighted_quantile(spreads, ws, 0.95)),
                "se": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "weight_sum": float(np.sum(np.asarray(ws, dtype=np.float64))),
                "effective_n": float(spread_mean_stats["effective_n"]),
            },
        }
    return out


def _max_rho(hash_bits: int, precision: int) -> int:
    # remaining bits = hash_bits - precision; rho in [1, remaining_bits+1]
    return int(hash_bits) - int(precision) + 1


def _hll_memory_bits(precision: int, hash_bits: int = 64) -> int:
    sk = HyperLogLogSketch(HLLConfig(precision=int(precision), hash_bits=int(hash_bits)))
    return int(sk.memory_bits)


def _merge_schedule_registers_max(
    leaf_registers: Sequence[np.ndarray],
    *,
    schedule: ScheduleName,
) -> np.ndarray:
    """Merge leaf HLL registers with the correct (max) merge under a schedule."""
    if len(leaf_registers) == 0:
        raise ValueError("leaf_registers must be non-empty")
    cur = [np.array(x, copy=True) for x in leaf_registers]
    if schedule == "balanced":
        while len(cur) > 1:
            nxt: List[np.ndarray] = []
            i = 0
            while i < len(cur):
                if i + 1 >= len(cur):
                    nxt.append(cur[i])
                    i += 1
                    continue
                nxt.append(np.maximum(cur[i], cur[i + 1]))
                i += 2
            cur = nxt
        return cur[0]
    if schedule in ("left_to_right", "right_to_left"):
        if schedule == "right_to_left":
            cur = list(reversed(cur))
        acc = np.array(cur[0], copy=True)
        for reg in cur[1:]:
            np.maximum(acc, reg, out=acc)
        return acc
    raise ValueError(f"unsupported schedule: {schedule!r}")


def _hll_estimate_from_registers(
    registers: np.ndarray,
    *,
    precision: int,
    hash_bits: int = 64,
) -> float:
    """Compute HLL estimate given registers (uint8). Mirrors HyperLogLogSketch.estimate()."""
    cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
    sk = HyperLogLogSketch(cfg)
    sk.registers[:] = registers.astype(np.uint8, copy=False)
    return float(sk.estimate())


def hll_estimate_differentiable(
    registers: "torch.Tensor",
    *,
    hash_bits: int = 64,
) -> "torch.Tensor":
    """Classical HLL estimator on float register tensors.

    The returned scalar remains differentiable through the register values.
    Evaluation still rounds/clips through ``HyperLogLogSketch`` for package
    parity; training uses this path for readout-level local-law losses.
    """

    _require_torch()
    m = int(registers.shape[-1])
    alpha = 0.673 if m == 16 else 0.697 if m == 32 else 0.709 if m == 64 else 0.7213 / (1.0 + 1.079 / float(m))
    clamped = torch.clamp(registers.double(), min=0.0, max=64.0)
    z = torch.exp2(-clamped).sum(dim=-1)
    raw = (float(alpha) * float(m) * float(m)) / torch.clamp(z, min=1e-9)
    n_zeros = torch.clamp(1.0 - clamped, min=0.0).sum(dim=-1)
    linear = float(m) * torch.log(float(m) / torch.clamp(n_zeros, min=1e-3))
    use_linear = ((raw <= 2.5 * float(m)) & (n_zeros > 0.5)).detach()
    small_range = torch.where(use_linear, linear, raw)
    hash_space = float(2.0 ** int(hash_bits))
    clipped = torch.clamp(raw / hash_space, max=1.0 - 1e-12)
    large_range = -hash_space * torch.log1p(-clipped)
    use_large = (raw > hash_space / 30.0).detach()
    return torch.where(use_large, large_range, small_range).to(dtype=registers.dtype)


@dataclass(frozen=True)
class TokenStreamDoc:
    token_ids: Tuple[int, ...]
    leaf_token_lists: Tuple[Tuple[int, ...], ...]
    true_cardinality: int


def _build_zipf_probability_bank(
    universe_size: int,
    alphas: Sequence[float],
) -> Dict[float, np.ndarray]:
    if universe_size <= 0:
        raise ValueError("universe_size must be positive")
    if len(alphas) == 0:
        raise ValueError("alphas must be non-empty")
    if any(float(a) <= 0.0 for a in alphas):
        raise ValueError("zipf alphas must be > 0")

    ranks = np.arange(1, int(universe_size) + 1, dtype=np.float64)
    bank: Dict[float, np.ndarray] = {}
    for a in alphas:
        weights = np.power(ranks, -float(a))
        probs = weights / float(weights.sum())
        bank[float(a)] = probs.astype(np.float64, copy=False)
    return bank


def generate_token_stream_docs(
    n_docs: int,
    *,
    universe_size: int,
    min_tokens: int,
    max_tokens: int,
    leaf_size: int,
    zipf_alphas: Sequence[float],
    seed: int,
) -> Tuple[TokenStreamDoc, ...]:
    if n_docs <= 0:
        return tuple()
    if max_tokens <= 0 or min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("require 0 < min_tokens <= max_tokens")
    if universe_size <= 1:
        raise ValueError("universe_size must be >= 2")
    if leaf_size <= 0:
        raise ValueError("leaf_size must be positive")
    if len(zipf_alphas) == 0:
        raise ValueError("zipf_alphas must be non-empty")

    rng = np.random.default_rng(int(seed))
    alphas = tuple(float(a) for a in zipf_alphas)
    bank = _build_zipf_probability_bank(int(universe_size), alphas)
    alpha_keys = tuple(bank.keys())
    docs: List[TokenStreamDoc] = []

    for _ in range(int(n_docs)):
        alpha = float(alpha_keys[int(rng.integers(0, len(alpha_keys)))])
        probs = bank[alpha]
        n_tok = int(rng.integers(int(min_tokens), int(max_tokens) + 1))
        token_ids = rng.choice(
            int(universe_size),
            size=int(n_tok),
            replace=True,
            p=probs,
        ).astype(np.int64, copy=False)
        leaf_tokens: List[Tuple[int, ...]] = []
        for i in range(0, int(token_ids.shape[0]), int(leaf_size)):
            leaf = token_ids[i : i + int(leaf_size)]
            leaf_tokens.append(tuple(int(x) for x in leaf.tolist()))
        true_card = int(np.unique(token_ids).shape[0])
        docs.append(
            TokenStreamDoc(
                token_ids=tuple(int(x) for x in token_ids.tolist()),
                leaf_token_lists=tuple(leaf_tokens),
                true_cardinality=true_card,
            )
        )
    return tuple(docs)


def leaf_hll_registers(
    doc: TokenStreamDoc,
    *,
    precision: int,
    hash_bits: int = 64,
) -> Tuple[np.ndarray, ...]:
    cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
    regs: List[np.ndarray] = []
    for leaf in doc.leaf_token_lists:
        sk = HyperLogLogSketch.from_tokens(cfg, leaf)
        regs.append(np.array(sk.registers, copy=True))
    return tuple(regs)


def precompute_leaf_hll_registers(
    docs: Sequence[TokenStreamDoc],
    *,
    precision: int,
    hash_bits: int = 64,
) -> Tuple[Tuple[np.ndarray, ...], ...]:
    """Precompute per-document leaf registers to avoid re-hashing each epoch."""
    return tuple(
        leaf_hll_registers(doc, precision=int(precision), hash_bits=int(hash_bits))
        for doc in docs
    )


class LearnedHLLMerger(nn.Module):
    """Elementwise MLP that merges two HLL register vectors."""

    def __init__(self, *, precision: int, hash_bits: int = 64, hidden_dim: int = 16):
        super().__init__()
        _require_torch()
        self.precision = int(precision)
        self.hash_bits = int(hash_bits)
        self.hidden_dim = int(hidden_dim)
        self.max_rho = float(_max_rho(self.hash_bits, self.precision))

        self.net = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        x = torch.stack([left, right], dim=-1)  # (..., 2)
        out = self.net(x).squeeze(-1)
        return torch.clamp(out, 0.0, self.max_rho)


class InducedProjectionHLLMerger(nn.Module):
    """Shared ``g_theta`` family with merge induced as ``g_theta(a + b)``."""

    def __init__(self, *, precision: int, hash_bits: int = 64, hidden_dim: int = 16):
        super().__init__()
        _require_torch()
        self.precision = int(precision)
        self.hash_bits = int(hash_bits)
        self.hidden_dim = int(hidden_dim)
        self.max_rho = float(_max_rho(self.hash_bits, self.precision))
        self.net = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        x = state.to(dtype=torch.float32)
        normalized = (x / max(1.0, self.max_rho)).unsqueeze(-1)
        delta = self.net(normalized).squeeze(-1)
        projected = x + 0.25 * self.max_rho * delta
        return torch.clamp(projected, 0.0, self.max_rho)

    def encode_leaf(self, state: torch.Tensor) -> torch.Tensor:
        return self.project(state)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.project(left + right)


class ExactMaxMerger(nn.Module):
    """Reference merger: elementwise max (merge-safe)."""

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.maximum(left, right)


class MeanMerger(nn.Module):
    """A deliberately wrong, non-associative merge (for negative controls)."""

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 0.5 * (left + right)


def _encode_leaf_state(merger: nn.Module, state: torch.Tensor) -> torch.Tensor:
    encode = getattr(merger, "encode_leaf", None)
    if callable(encode):
        return encode(state)
    return state


def merge_leaf_states(
    merger: nn.Module,
    leaf_states: Sequence[torch.Tensor],
    *,
    schedule: ScheduleName,
) -> torch.Tensor:
    """Merge leaf states under a schedule (no supervision, inference-only)."""
    _require_torch()
    root, _, _ = _merge_states_torch(
        merger,
        leaf_states,
        leaf_states,
        schedule=str(schedule),
        audit_indices=None,
        collect_losses=False,
        idem_weight=0.0,
        comm_weight=0.0,
    )
    return root


def _merge_states_torch(
    merger: nn.Module,
    leaf_states: Sequence[torch.Tensor],
    leaf_oracle: Sequence[torch.Tensor],
    *,
    schedule: ScheduleName,
    audit_indices: Optional[set[int]],
    collect_losses: bool,
    idem_weight: float,
    comm_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if len(leaf_states) == 0:
        raise ValueError("leaf_states must be non-empty")
    if len(leaf_states) != len(leaf_oracle):
        raise ValueError("leaf_states and leaf_oracle must align")

    pred = [_encode_leaf_state(merger, st) for st in leaf_states]
    oracle = list(leaf_oracle)

    c3_loss = torch.zeros((), device=pred[0].device, dtype=torch.float32)
    c3_count = 0
    merge_idx = 0

    def _maybe_supervise(
        merged_pred: torch.Tensor,
        left_pred: torch.Tensor,
        right_pred: torch.Tensor,
        merged_oracle: torch.Tensor,
    ) -> None:
        nonlocal c3_loss, c3_count
        if not collect_losses:
            return
        if audit_indices is not None and merge_idx not in audit_indices:
            return

        loss = F.mse_loss(merged_pred, merged_oracle, reduction="mean")
        if comm_weight > 0.0:
            merged_rev = merger.merge(right_pred, left_pred)
            loss = loss + float(comm_weight) * F.mse_loss(merged_pred, merged_rev, reduction="mean")
        if idem_weight > 0.0:
            idem_l = merger.merge(left_pred, left_pred)
            idem_r = merger.merge(right_pred, right_pred)
            loss = loss + float(idem_weight) * (
                F.mse_loss(idem_l, left_pred, reduction="mean")
                + F.mse_loss(idem_r, right_pred, reduction="mean")
            )
        c3_loss = c3_loss + loss
        c3_count += 1

    if schedule == "balanced":
        while len(pred) > 1:
            nxt_pred: List[torch.Tensor] = []
            nxt_oracle: List[torch.Tensor] = []
            i = 0
            while i < len(pred):
                if i + 1 >= len(pred):
                    nxt_pred.append(pred[i])
                    nxt_oracle.append(oracle[i])
                    i += 1
                    continue
                merged_pred = merger.merge(pred[i], pred[i + 1])
                merged_oracle = torch.maximum(oracle[i], oracle[i + 1])
                _maybe_supervise(merged_pred, pred[i], pred[i + 1], merged_oracle)
                merge_idx += 1
                nxt_pred.append(merged_pred)
                nxt_oracle.append(merged_oracle)
                i += 2
            pred = nxt_pred
            oracle = nxt_oracle
        return pred[0], c3_loss, c3_count

    if schedule in ("left_to_right", "right_to_left"):
        if schedule == "right_to_left":
            pred = list(reversed(pred))
            oracle = list(reversed(oracle))
        acc_pred = pred[0]
        acc_oracle = oracle[0]
        for st, st_or in zip(pred[1:], oracle[1:]):
            merged_pred = merger.merge(acc_pred, st)
            merged_oracle = torch.maximum(acc_oracle, st_or)
            _maybe_supervise(merged_pred, acc_pred, st, merged_oracle)
            merge_idx += 1
            acc_pred = merged_pred
            acc_oracle = merged_oracle
        return acc_pred, c3_loss, c3_count

    raise ValueError(f"unsupported schedule: {schedule!r}")


@dataclass(frozen=True)
class MergeEvalMetrics:
    relative_rmse: float
    mean_abs_rel_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float


@dataclass(frozen=True)
class HLLBaselineMetrics:
    precision: int
    registers: int
    memory_bits: int
    memory_bytes: float
    rse_theory: float
    metrics: MergeEvalMetrics
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None


@dataclass(frozen=True)
class HLLLocalLawTrainingDiagnostics:
    corrected_local_law_loss_mean: float = float("nan")
    proxy_loss_mean: float = float("nan")
    oracle_ipw_loss_mean: float = float("nan")
    ipw_correction_mean: float = float("nan")
    observed_rows_mean: float = float("nan")
    observed_rows_per_doc: float = float("nan")
    sampled_internal_rows_mean: float = float("nan")
    merge_mse_mean: float = float("nan")


@dataclass(frozen=True)
class LearnedMergerMetrics:
    metrics: MergeEvalMetrics
    merge_mse_mean: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
    corrected_local_law_loss_mean: float = float("nan")
    proxy_loss_mean: float = float("nan")
    oracle_ipw_loss_mean: float = float("nan")
    ipw_correction_mean: float = float("nan")
    observed_rows_mean: float = float("nan")
    observed_rows_per_doc: float = float("nan")
    sampled_internal_rows_mean: float = float("nan")


@dataclass(frozen=True)
class HLLMergeLearningRun:
    seed: int
    data_seed: int
    precision: int
    registers: int
    memory_bits: int
    train_docs: int
    n_test: int
    audit_policy: AuditPolicyName
    audit_fixed_nodes: int
    audit_fraction: float
    audit_scale: float
    n_epochs: int
    hidden_dim: int
    lr: float
    weight_decay: float
    idem_weight: float
    comm_weight: float
    train_mean_internal_nodes: float
    train_audit_nodes_mean: float
    train_audit_coverage_mean: float
    train_total_queries_estimate: int
    hll_baseline: HLLBaselineMetrics
    learned: LearnedMergerMetrics
    distance_to_hll_floor_rel_rmse: float
    ratio_to_hll_floor_rel_rmse: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = None
    model_kind: str = "induced_projection"
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED
    proxy_mode: str = "frozen_rollout"
    lean_adjusted_loss: str = "proxy + R / pi * (oracle - proxy)"
    lean_merge_adapter: str = "merge(a,b)=g_theta(a+b); encode_leaf(x)=g_theta(x)"
    lean_projection_target: str = "f*(x+y)=f*(g*(g*(x)+g*(y)))"


@dataclass(frozen=True)
class HLLMergeLearningConfig:
    model_kind: str = "induced_projection"
    objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED
    proxy_mode: str = "frozen_rollout"
    local_law_leaf_discount_gamma: float = 1.0
    universe_size: int = 65_536
    min_tokens: int = 4096
    max_tokens: int = 16384
    leaf_size: int = 512
    zipf_alphas: Tuple[float, ...] = (0.8, 1.0, 1.2)
    precisions: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)
    train_docs_grid: Tuple[int, ...] = (25, 50, 100, 200, 500, 1000)
    audit_policies: Tuple[AuditPolicyName, ...] = ("all", "sqrt", "log2", "fraction")
    audit_fixed_nodes: int = 0
    audit_fraction: float = 0.25
    audit_scale: float = 1.0
    n_test: int = 256
    hidden_dim: int = 16
    n_epochs: int = 6
    batch_docs: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    idem_weight: float = 0.10
    comm_weight: float = 0.10
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    torch_threads: int = 0
    weighting_modes: Tuple[str, ...] = ("doc", "leaf", "token")
    legacy_weighting_mode: str = "doc"
    seed: int = 0
    data_seed: int = 0


@dataclass(frozen=True)
class HLLMergeLearningSummary:
    config: Dict[str, object]
    results: Tuple[HLLMergeLearningRun, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "config": self.config,
                "rows": experiment_rows(self.results),
            },
            indent=2,
            sort_keys=True,
        )


def _summarize_audit_geometry(
    leaf_regs_by_doc: Sequence[Tuple[np.ndarray, ...]],
    *,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
) -> Dict[str, float | int]:
    if len(leaf_regs_by_doc) == 0:
        return {
            "mean_internal_nodes": 0.0,
            "audit_nodes_mean": 0.0,
            "audit_coverage_mean": 0.0,
            "total_queries_estimate": 0,
        }
    internal: List[float] = []
    audits: List[float] = []
    covers: List[float] = []
    audit_nodes_total = 0
    for leaf_regs in leaf_regs_by_doc:
        n_leaves = int(len(leaf_regs))
        n_internal = int(max(0, n_leaves - 1))
        q = audit_sample_count(
            n_internal,
            policy=str(audit_policy),
            fixed_nodes=int(audit_fixed_nodes),
            fraction=float(audit_fraction),
            scale=float(audit_scale),
        )
        internal.append(float(n_internal))
        audits.append(float(q))
        covers.append(float(q) / float(n_internal) if n_internal > 0 else 1.0)
        audit_nodes_total += int(q)
    n = float(len(leaf_regs_by_doc))
    return {
        "mean_internal_nodes": float(sum(internal) / n),
        "audit_nodes_mean": float(sum(audits) / n),
        "audit_coverage_mean": float(sum(covers) / n),
        "total_queries_estimate": int(audit_nodes_total),
    }


def _evaluate_merger_doc_arrays(
    docs: Sequence[TokenStreamDoc],
    *,
    merger: nn.Module,
    precision: int,
    hash_bits: int = 64,
    leaf_regs_by_doc: Optional[Sequence[Tuple[np.ndarray, ...]]] = None,
    device: torch.device,
) -> Tuple[MergeEvalMetrics, List[float], List[float], List[float], List[float], List[float]]:
    if len(docs) == 0:
        empty = MergeEvalMetrics(
            relative_rmse=0.0,
            mean_abs_rel_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
        )
        return (empty, [], [], [], [], [])

    rel_sq: List[float] = []
    abs_rel: List[float] = []
    spreads: List[float] = []
    token_weights: List[float] = []
    leaf_weights: List[float] = []

    merger.eval()
    with torch.no_grad():
        if leaf_regs_by_doc is None:
            leaf_regs_by_doc = precompute_leaf_hll_registers(
                docs, precision=int(precision), hash_bits=int(hash_bits)
            )
        if len(leaf_regs_by_doc) != len(docs):
            raise ValueError("leaf_regs_by_doc must have same length as docs")

        for doc, leaf_regs in zip(docs, leaf_regs_by_doc):
            leaf = [torch.tensor(x, dtype=torch.float32, device=device) for x in leaf_regs]
            ests: Dict[str, float] = {}
            for sched in VALID_SCHEDULES:
                root, _, _ = _merge_states_torch(
                    merger,
                    leaf,
                    leaf,
                    schedule=sched,
                    audit_indices=None,
                    collect_losses=False,
                    idem_weight=0.0,
                    comm_weight=0.0,
                )
                root_np = root.detach().cpu().numpy()
                root_uint = np.rint(np.clip(root_np, 0.0, float(_max_rho(hash_bits, precision)))).astype(
                    np.uint8
                )
                ests[sched] = _hll_estimate_from_registers(
                    root_uint,
                    precision=int(precision),
                    hash_bits=int(hash_bits),
                )
            pred = float(ests["balanced"])
            truth = float(doc.true_cardinality)
            rel = _safe_rel_error(pred, truth)
            rel_sq.append(rel * rel)
            abs_rel.append(abs(rel))
            spreads.append(max(ests.values()) - min(ests.values()))
            token_weights.append(float(max(1, len(doc.token_ids))))
            leaf_weights.append(float(max(1, len(leaf_regs))))

    rel_rmse = float(math.sqrt(float(np.mean(np.array(rel_sq, dtype=np.float64)))))
    metrics = MergeEvalMetrics(
        relative_rmse=rel_rmse,
        mean_abs_rel_error=float(np.mean(np.array(abs_rel, dtype=np.float64))),
        schedule_spread_mean=float(np.mean(np.array(spreads, dtype=np.float64))),
        schedule_spread_p95=float(np.percentile(np.array(spreads, dtype=np.float64), 95.0)),
    )
    return (metrics, rel_sq, abs_rel, spreads, token_weights, leaf_weights)


def evaluate_merger_on_docs(
    docs: Sequence[TokenStreamDoc],
    *,
    merger: nn.Module,
    precision: int,
    hash_bits: int = 64,
    leaf_regs_by_doc: Optional[Sequence[Tuple[np.ndarray, ...]]] = None,
    device: torch.device,
) -> MergeEvalMetrics:
    metrics, _, _, _, _, _ = _evaluate_merger_doc_arrays(
        docs,
        merger=merger,
        precision=int(precision),
        hash_bits=int(hash_bits),
        leaf_regs_by_doc=leaf_regs_by_doc,
        device=device,
    )
    return metrics


def evaluate_merger_on_docs_with_weighting(
    docs: Sequence[TokenStreamDoc],
    *,
    merger: nn.Module,
    precision: int,
    hash_bits: int = 64,
    leaf_regs_by_doc: Optional[Sequence[Tuple[np.ndarray, ...]]] = None,
    device: torch.device,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> Tuple[MergeEvalMetrics, str, Dict[str, Dict[str, Dict[str, float]]]]:
    modes = parse_weighting_modes(
        weighting_modes if weighting_modes is not None else DEFAULT_WEIGHTING_MODES
    )
    legacy = validate_legacy_weighting_mode(legacy_weighting_mode, weighting_modes=modes)
    metrics, rel_sq, abs_rel, spreads, token_weights, leaf_weights = _evaluate_merger_doc_arrays(
        docs,
        merger=merger,
        precision=int(precision),
        hash_bits=int(hash_bits),
        leaf_regs_by_doc=leaf_regs_by_doc,
        device=device,
    )
    views = _build_merge_metric_weighting_views(
        rel_sq=rel_sq,
        abs_rel=abs_rel,
        spreads=spreads,
        token_weights=token_weights,
        leaf_weights=leaf_weights,
        weighting_modes=modes,
    )
    return (metrics, legacy.value, views)


def evaluate_hll_baseline(
    docs: Sequence[TokenStreamDoc],
    *,
    precision: int,
    hash_bits: int = 64,
    leaf_regs_by_doc: Optional[Sequence[Tuple[np.ndarray, ...]]] = None,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> HLLBaselineMetrics:
    cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
    proto = HyperLogLogSketch(cfg)
    rse = float(hll_relative_standard_error(int(precision)))
    if leaf_regs_by_doc is None:
        leaf_regs_by_doc = precompute_leaf_hll_registers(
            docs, precision=int(precision), hash_bits=int(hash_bits)
        )
    if len(leaf_regs_by_doc) != len(docs):
        raise ValueError("leaf_regs_by_doc must have same length as docs")
    metrics, legacy_mode, views = evaluate_merger_on_docs_with_weighting(
        docs,
        merger=ExactMaxMerger(),
        precision=int(precision),
        hash_bits=int(hash_bits),
        leaf_regs_by_doc=leaf_regs_by_doc,
        device=torch.device("cpu"),
        weighting_modes=weighting_modes,
        legacy_weighting_mode=legacy_weighting_mode,
    )
    return HLLBaselineMetrics(
        precision=int(precision),
        registers=int(proto.m),
        memory_bits=int(proto.memory_bits),
        memory_bytes=float(proto.memory_bits) / 8.0,
        rse_theory=rse,
        metrics=metrics,
        legacy_weighting_mode=str(legacy_mode),
        weighting_views=views,
    )


def _normalize_model_kind(model_kind: str) -> str:
    normalized = str(model_kind or "induced_projection").strip().lower()
    aliases = {
        "induced": "induced_projection",
        "projection": "induced_projection",
        "lean": "induced_projection",
        "direct": "direct_state_mlp",
        "direct_mlp": "direct_state_mlp",
        "legacy": "direct_state_mlp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"induced_projection", "direct_state_mlp"}:
        raise ValueError(
            f"unknown HLL merge model_kind={model_kind!r}; expected "
            "induced_projection or direct_state_mlp"
        )
    return normalized


def _freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _rollout_hll_node_rows(
    merger: nn.Module,
    leaf_states: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return learned/exact node rows for a balanced HLL tree.

    Rows are leaves first, then internal nodes in balanced reduction order.
    Depths are distance from the root so ``gamma < 1`` discounts leaf rows.
    Odd carried nodes keep their actual shallower tree depth.
    """

    if len(leaf_states) == 0:
        raise ValueError("leaf_states must be non-empty")
    pred = [_encode_leaf_state(merger, st) for st in leaf_states]
    exact = list(leaf_states)
    pred_rows = list(pred)
    exact_rows = list(exact)
    row_children: List[Tuple[Optional[int], Optional[int]]] = [
        (None, None) for _ in leaf_states
    ]
    row_indices = list(range(len(leaf_states)))
    internal_count = 0

    while len(pred) > 1:
        nxt_pred: List[torch.Tensor] = []
        nxt_exact: List[torch.Tensor] = []
        nxt_row_indices: List[int] = []
        i = 0
        while i < len(pred):
            if i + 1 >= len(pred):
                nxt_pred.append(pred[i])
                nxt_exact.append(exact[i])
                nxt_row_indices.append(row_indices[i])
                i += 1
                continue
            merged_pred = merger.merge(pred[i], pred[i + 1])
            merged_exact = torch.maximum(exact[i], exact[i + 1])
            row_idx = len(pred_rows)
            pred_rows.append(merged_pred)
            exact_rows.append(merged_exact)
            row_children.append((row_indices[i], row_indices[i + 1]))
            internal_count += 1
            nxt_pred.append(merged_pred)
            nxt_exact.append(merged_exact)
            nxt_row_indices.append(row_idx)
            i += 2
        pred = nxt_pred
        exact = nxt_exact
        row_indices = nxt_row_indices

    depths_by_row = [0 for _ in pred_rows]
    stack: List[Tuple[int, int]] = [(row_indices[0], 0)]
    while stack:
        row_idx, depth = stack.pop()
        depths_by_row[row_idx] = int(depth)
        left_idx, right_idx = row_children[row_idx]
        if left_idx is not None:
            stack.append((left_idx, int(depth) + 1))
        if right_idx is not None:
            stack.append((right_idx, int(depth) + 1))

    return (
        torch.stack(pred_rows, dim=0),
        torch.stack(exact_rows, dim=0),
        torch.tensor(depths_by_row, dtype=torch.long, device=pred_rows[0].device),
        int(internal_count),
    )


def _sample_internal_audit_indices(
    *,
    n_internal: int,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    rng: random.Random,
) -> Tuple[Optional[set[int]], int]:
    n = int(max(0, n_internal))
    q = audit_sample_count(
        n,
        policy=str(audit_policy),
        fixed_nodes=int(audit_fixed_nodes),
        fraction=float(audit_fraction),
        scale=float(audit_scale),
    )
    if q <= 0:
        return set(), 0
    if q >= n:
        return None, n
    return set(rng.sample(range(n), k=int(q))), int(q)


def _node_row_observation_tensors(
    *,
    n_leaves: int,
    n_internal: int,
    audit_indices: Optional[set[int]],
    n_audit: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    observed: List[bool] = [True for _ in range(int(n_leaves))]
    propensity: List[float] = [1.0 for _ in range(int(n_leaves))]
    if n_internal > 0:
        internal_pi = 1.0 if audit_indices is None else float(n_audit) / float(n_internal)
        for idx in range(int(n_internal)):
            selected = audit_indices is None or idx in audit_indices
            observed.append(bool(selected))
            propensity.append(float(internal_pi if selected else 0.0))
    return (
        torch.tensor(observed, dtype=torch.bool, device=device),
        torch.tensor(propensity, dtype=torch.float32, device=device),
    )


def _local_law_component_metrics(
    *,
    predictions: torch.Tensor,
    proxy_targets: torch.Tensor,
    oracle_targets: torch.Tensor,
    observed: torch.Tensor,
    propensity: torch.Tensor,
    depths: torch.Tensor,
    gamma_depth: float,
) -> Dict[str, float]:
    pred = predictions.reshape(-1)
    proxy = proxy_targets.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    oracle = oracle_targets.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    obs = observed.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    pi = propensity.to(device=pred.device, dtype=pred.dtype).reshape(-1).clamp(min=1e-12, max=1.0)
    gamma = float(gamma_depth)
    depth_values = depths.to(device=pred.device, dtype=torch.float32).reshape(-1)
    weights = torch.pow(torch.full_like(depth_values, gamma), depth_values).to(dtype=pred.dtype)
    denom = weights.sum().clamp(min=1e-12)
    proxy_loss = (pred - proxy) ** 2
    oracle_loss = (pred - oracle) ** 2
    oracle_ipw_rows = obs * oracle_loss / pi
    correction_rows = obs * (oracle_loss - proxy_loss) / pi
    proxy_mean = (weights * proxy_loss).sum() / denom
    oracle_ipw_mean = (weights * oracle_ipw_rows).sum() / denom
    correction_mean = (weights * correction_rows).sum() / denom
    corrected_mean = proxy_mean + correction_mean
    return {
        "corrected_local_law_loss": float(corrected_mean.detach().cpu()),
        "proxy_loss": float(proxy_mean.detach().cpu()),
        "oracle_ipw_loss": float(oracle_ipw_mean.detach().cpu()),
        "ipw_correction": float(correction_mean.detach().cpu()),
        "observed_rows": float(observed.detach().to(dtype=torch.float32).sum().cpu()),
    }


def _safe_float_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def train_learned_merger(
    model: nn.Module,
    train_docs: Sequence[TokenStreamDoc],
    *,
    train_leaf_regs_by_doc: Sequence[Tuple[np.ndarray, ...]],
    precision: int,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    n_epochs: int,
    batch_docs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    idem_weight: float,
    comm_weight: float,
    device: torch.device,
    seed: int,
) -> float:
    _require_torch()
    if len(train_docs) == 0:
        raise ValueError("train_docs must be non-empty")
    if batch_docs <= 0:
        raise ValueError("batch_docs must be positive")
    if len(train_leaf_regs_by_doc) != len(train_docs):
        raise ValueError("train_leaf_regs_by_doc must align with train_docs")

    model.to(device)
    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    rng = random.Random(int(seed))

    merge_mse_terms: List[float] = []
    idxs = list(range(len(train_docs)))
    for _ in range(int(n_epochs)):
        rng.shuffle(idxs)
        for b0 in range(0, len(idxs), int(batch_docs)):
            batch_idx = idxs[b0 : b0 + int(batch_docs)]
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device, dtype=torch.float32)
            n_losses = 0
            for i in batch_idx:
                doc = train_docs[i]
                leaf_regs = train_leaf_regs_by_doc[i]
                leaf = [
                    torch.tensor(x, dtype=torch.float32, device=device) for x in leaf_regs
                ]
                n_internal = int(max(0, len(leaf) - 1))
                n_audit = audit_sample_count(
                    n_internal,
                    policy=str(audit_policy),
                    fixed_nodes=int(audit_fixed_nodes),
                    fraction=float(audit_fraction),
                    scale=float(audit_scale),
                )
                if n_audit <= 0:
                    audit_indices: Optional[set[int]] = set()
                elif n_audit >= n_internal:
                    audit_indices = None
                else:
                    audit_indices = set(rng.sample(range(n_internal), k=int(n_audit)))

                root, c3_loss, c3_count = _merge_states_torch(
                    model,
                    leaf,
                    leaf,
                    schedule="balanced",
                    audit_indices=audit_indices,
                    collect_losses=True,
                    idem_weight=float(idem_weight),
                    comm_weight=float(comm_weight),
                )
                if int(c3_count) > 0:
                    doc_loss = c3_loss / float(c3_count)
                    loss = loss + doc_loss
                    n_losses += 1

                    # Track raw merge MSE (without regularizers) for reporting.
                    with torch.no_grad():
                        oracle_root = torch.max(torch.stack(leaf, dim=0), dim=0).values
                        mse = F.mse_loss(root, oracle_root, reduction="mean")
                        merge_mse_terms.append(float(mse.detach().cpu()))
            if n_losses <= 0:
                continue
            loss = loss / float(n_losses)
            loss.backward()
            if float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()

    if len(merge_mse_terms) == 0:
        return float("nan")
    return float(np.mean(np.array(merge_mse_terms, dtype=np.float64)))


def train_induced_projection_merger(
    model: InducedProjectionHLLMerger,
    train_docs: Sequence[TokenStreamDoc],
    *,
    train_leaf_regs_by_doc: Sequence[Tuple[np.ndarray, ...]],
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    n_epochs: int,
    batch_docs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    objective_mode: str,
    proxy_mode: str,
    local_law_leaf_discount_gamma: float,
    device: torch.device,
    seed: int,
) -> HLLLocalLawTrainingDiagnostics:
    _require_torch()
    if len(train_docs) == 0:
        raise ValueError("train_docs must be non-empty")
    if batch_docs <= 0:
        raise ValueError("batch_docs must be positive")
    if len(train_leaf_regs_by_doc) != len(train_docs):
        raise ValueError("train_leaf_regs_by_doc must align with train_docs")

    objective = normalize_local_law_objective_mode(objective_mode)
    proxy = str(proxy_mode or "frozen_rollout").strip().lower()
    if proxy != "frozen_rollout":
        raise ValueError(f"unsupported proxy_mode={proxy_mode!r}; expected frozen_rollout")

    model.to(device)
    model.train()
    proxy_model = _freeze_module(copy.deepcopy(model).to(device))
    opt = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    rng = random.Random(int(seed))

    corrected_terms: List[float] = []
    proxy_terms: List[float] = []
    oracle_ipw_terms: List[float] = []
    correction_terms: List[float] = []
    observed_terms: List[float] = []
    observed_per_doc_terms: List[float] = []
    sampled_internal_terms: List[float] = []
    merge_mse_terms: List[float] = []

    idxs = list(range(len(train_docs)))
    for _ in range(int(n_epochs)):
        rng.shuffle(idxs)
        for b0 in range(0, len(idxs), int(batch_docs)):
            batch_idx = idxs[b0 : b0 + int(batch_docs)]
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device, dtype=torch.float32)
            n_losses = 0
            for i in batch_idx:
                leaf_regs = train_leaf_regs_by_doc[i]
                leaf = [
                    torch.tensor(x, dtype=torch.float32, device=device) for x in leaf_regs
                ]
                pred_rows, exact_rows, depths, n_internal = _rollout_hll_node_rows(model, leaf)
                audit_indices, n_audit = _sample_internal_audit_indices(
                    n_internal=int(n_internal),
                    audit_policy=str(audit_policy),
                    audit_fixed_nodes=int(audit_fixed_nodes),
                    audit_fraction=float(audit_fraction),
                    audit_scale=float(audit_scale),
                    rng=rng,
                )
                observed, propensity = _node_row_observation_tensors(
                    n_leaves=len(leaf),
                    n_internal=int(n_internal),
                    audit_indices=audit_indices,
                    n_audit=int(n_audit),
                    device=device,
                )
                with torch.no_grad():
                    proxy_rows, _, _, _ = _rollout_hll_node_rows(proxy_model, leaf)
                    proxy_targets = hll_estimate_differentiable(proxy_rows).detach()
                    oracle_targets = hll_estimate_differentiable(exact_rows).detach()
                predictions = hll_estimate_differentiable(pred_rows)
                doc_loss = local_law_objective_target_mse(
                    predictions=predictions,
                    proxy_targets=proxy_targets,
                    oracle_targets=oracle_targets,
                    observed=observed,
                    propensity=propensity,
                    depths=depths,
                    gamma_depth=float(local_law_leaf_discount_gamma),
                    objective_mode=objective,
                )
                loss = loss + doc_loss
                n_losses += 1

                with torch.no_grad():
                    components = _local_law_component_metrics(
                        predictions=predictions.detach(),
                        proxy_targets=proxy_targets,
                        oracle_targets=oracle_targets,
                        observed=observed,
                        propensity=propensity,
                        depths=depths,
                        gamma_depth=float(local_law_leaf_discount_gamma),
                    )
                    corrected_terms.append(float(components["corrected_local_law_loss"]))
                    proxy_terms.append(float(components["proxy_loss"]))
                    oracle_ipw_terms.append(float(components["oracle_ipw_loss"]))
                    correction_terms.append(float(components["ipw_correction"]))
                    observed_terms.append(float(components["observed_rows"]))
                    observed_per_doc_terms.append(float(components["observed_rows"]))
                    sampled_internal_terms.append(float(n_audit))
                    merge_mse = F.mse_loss(pred_rows[-1], exact_rows[-1], reduction="mean")
                    merge_mse_terms.append(float(merge_mse.detach().cpu()))
            if n_losses <= 0:
                continue
            loss = loss / float(n_losses)
            loss.backward()
            if float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()

    return HLLLocalLawTrainingDiagnostics(
        corrected_local_law_loss_mean=_safe_float_mean(corrected_terms),
        proxy_loss_mean=_safe_float_mean(proxy_terms),
        oracle_ipw_loss_mean=_safe_float_mean(oracle_ipw_terms),
        ipw_correction_mean=_safe_float_mean(correction_terms),
        observed_rows_mean=_safe_float_mean(observed_terms),
        observed_rows_per_doc=_safe_float_mean(observed_per_doc_terms),
        sampled_internal_rows_mean=_safe_float_mean(sampled_internal_terms),
        merge_mse_mean=_safe_float_mean(merge_mse_terms),
    )


def run_hll_merge_learning_experiment(config: HLLMergeLearningConfig) -> Tuple[HLLMergeLearningRun, ...]:
    _require_torch()
    _set_global_seed(int(config.seed))
    model_kind = _normalize_model_kind(config.model_kind)
    objective_mode = normalize_local_law_objective_mode(config.objective_mode)
    proxy_mode = str(config.proxy_mode or "frozen_rollout").strip().lower()
    if float(config.local_law_leaf_discount_gamma) < 0.0:
        raise ValueError(
            "local_law_leaf_discount_gamma must be non-negative, got "
            f"{config.local_law_leaf_discount_gamma!r}"
        )
    modes = parse_weighting_modes(config.weighting_modes)
    legacy_mode = validate_legacy_weighting_mode(
        config.legacy_weighting_mode,
        weighting_modes=modes,
    )
    mode_names = tuple(m.value for m in modes)

    if config.use_cuda and torch.cuda.is_available():
        if config.cuda_device is not None:
            cuda_idx = int(config.cuda_device)
            n_cuda = int(torch.cuda.device_count())
            if cuda_idx < 0 or cuda_idx >= n_cuda:
                raise ValueError(
                    f"cuda_device={cuda_idx} out of range; available devices: 0..{n_cuda - 1}"
                )
            torch.cuda.set_device(cuda_idx)
            device = torch.device(f"cuda:{cuda_idx}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    max_train = max(int(x) for x in config.train_docs_grid) if config.train_docs_grid else 0
    total_docs = int(max_train + config.n_test)
    data_seed = int(config.data_seed)
    docs = generate_token_stream_docs(
        total_docs,
        universe_size=int(config.universe_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        leaf_size=int(config.leaf_size),
        zipf_alphas=_parse_csv_floats(config.zipf_alphas),
        seed=data_seed,
    )
    train_pool = docs[:max_train]
    test_docs = docs[max_train:]

    results: List[HLLMergeLearningRun] = []
    baseline_cache: Dict[int, HLLBaselineMetrics] = {}

    for p in (int(x) for x in config.precisions):
        # Precompute leaf registers for this precision once; reuse for all train/policy slices.
        leaf_regs_all = precompute_leaf_hll_registers(
            docs,
            precision=int(p),
            hash_bits=64,
        )
        train_leaf_regs_all = leaf_regs_all[:max_train]
        test_leaf_regs = leaf_regs_all[max_train:]

        if p not in baseline_cache:
            baseline_cache[p] = evaluate_hll_baseline(
                test_docs,
                precision=p,
                hash_bits=64,
                leaf_regs_by_doc=test_leaf_regs,
                weighting_modes=mode_names,
                legacy_weighting_mode=legacy_mode.value,
            )
        baseline = baseline_cache[p]

        for audit_policy in config.audit_policies:
            for train_docs in (int(x) for x in config.train_docs_grid):
                if train_docs <= 0:
                    continue
                if model_kind == "induced_projection":
                    model = InducedProjectionHLLMerger(
                        precision=int(p),
                        hash_bits=64,
                        hidden_dim=int(config.hidden_dim),
                    )
                else:
                    model = LearnedHLLMerger(
                        precision=int(p),
                        hash_bits=64,
                        hidden_dim=int(config.hidden_dim),
                    )
                train_subset = train_pool[:train_docs]
                train_leaf_regs = train_leaf_regs_all[:train_docs]
                geom = _summarize_audit_geometry(
                    train_leaf_regs,
                    audit_policy=str(audit_policy),
                    audit_fixed_nodes=int(config.audit_fixed_nodes),
                    audit_fraction=float(config.audit_fraction),
                    audit_scale=float(config.audit_scale),
                )

                if model_kind == "induced_projection":
                    diagnostics = train_induced_projection_merger(
                        model,
                        train_subset,
                        train_leaf_regs_by_doc=train_leaf_regs,
                        audit_policy=str(audit_policy),
                        audit_fixed_nodes=int(config.audit_fixed_nodes),
                        audit_fraction=float(config.audit_fraction),
                        audit_scale=float(config.audit_scale),
                        n_epochs=int(config.n_epochs),
                        batch_docs=int(config.batch_docs),
                        lr=float(config.lr),
                        weight_decay=float(config.weight_decay),
                        grad_clip_norm=float(config.grad_clip_norm),
                        objective_mode=str(objective_mode),
                        proxy_mode=str(proxy_mode),
                        local_law_leaf_discount_gamma=float(config.local_law_leaf_discount_gamma),
                        device=device,
                        seed=int(config.seed + 7919 + p + train_docs),
                    )
                else:
                    merge_mse_mean = train_learned_merger(
                        model,
                        train_subset,
                        train_leaf_regs_by_doc=train_leaf_regs,
                        precision=int(p),
                        audit_policy=str(audit_policy),
                        audit_fixed_nodes=int(config.audit_fixed_nodes),
                        audit_fraction=float(config.audit_fraction),
                        audit_scale=float(config.audit_scale),
                        n_epochs=int(config.n_epochs),
                        batch_docs=int(config.batch_docs),
                        lr=float(config.lr),
                        weight_decay=float(config.weight_decay),
                        grad_clip_norm=float(config.grad_clip_norm),
                        idem_weight=float(config.idem_weight),
                        comm_weight=float(config.comm_weight),
                        device=device,
                        seed=int(config.seed + 7919 + p + train_docs),
                    )
                    diagnostics = HLLLocalLawTrainingDiagnostics(
                        merge_mse_mean=float(merge_mse_mean),
                    )

                learned_metrics, learned_legacy_mode, learned_views = evaluate_merger_on_docs_with_weighting(
                    test_docs,
                    merger=model,
                    precision=int(p),
                    hash_bits=64,
                    leaf_regs_by_doc=test_leaf_regs,
                    device=device,
                    weighting_modes=mode_names,
                    legacy_weighting_mode=legacy_mode.value,
                )
                dist = float(learned_metrics.relative_rmse - baseline.rse_theory)
                ratio = float(learned_metrics.relative_rmse / max(1e-12, baseline.rse_theory))

                results.append(
                    HLLMergeLearningRun(
                        seed=int(config.seed),
                        data_seed=int(data_seed),
                        precision=int(p),
                        registers=int(baseline.registers),
                        memory_bits=int(baseline.memory_bits),
                        train_docs=int(train_docs),
                        n_test=int(config.n_test),
                        audit_policy=str(audit_policy),
                        audit_fixed_nodes=int(config.audit_fixed_nodes),
                        audit_fraction=float(config.audit_fraction),
                        audit_scale=float(config.audit_scale),
                        n_epochs=int(config.n_epochs),
                        hidden_dim=int(config.hidden_dim),
                        lr=float(config.lr),
                        weight_decay=float(config.weight_decay),
                        idem_weight=float(config.idem_weight),
                        comm_weight=float(config.comm_weight),
                        train_mean_internal_nodes=float(geom["mean_internal_nodes"]),
                        train_audit_nodes_mean=float(geom["audit_nodes_mean"]),
                        train_audit_coverage_mean=float(geom["audit_coverage_mean"]),
                        train_total_queries_estimate=int(geom["total_queries_estimate"]),
                        hll_baseline=baseline,
                        learned=LearnedMergerMetrics(
                            metrics=learned_metrics,
                            merge_mse_mean=float(diagnostics.merge_mse_mean),
                            legacy_weighting_mode=str(learned_legacy_mode),
                            weighting_views=learned_views,
                            corrected_local_law_loss_mean=float(
                                diagnostics.corrected_local_law_loss_mean
                            ),
                            proxy_loss_mean=float(diagnostics.proxy_loss_mean),
                            oracle_ipw_loss_mean=float(diagnostics.oracle_ipw_loss_mean),
                            ipw_correction_mean=float(diagnostics.ipw_correction_mean),
                            observed_rows_mean=float(diagnostics.observed_rows_mean),
                            observed_rows_per_doc=float(diagnostics.observed_rows_per_doc),
                            sampled_internal_rows_mean=float(
                                diagnostics.sampled_internal_rows_mean
                            ),
                        ),
                        distance_to_hll_floor_rel_rmse=dist,
                        ratio_to_hll_floor_rel_rmse=ratio,
                        legacy_weighting_mode=str(legacy_mode.value),
                        weighting_views={
                            "hll_baseline": baseline.weighting_views or {},
                            "learned": learned_views,
                        },
                        model_kind=str(model_kind),
                        objective_mode=(
                            str(objective_mode)
                            if model_kind == "induced_projection"
                            else "direct_state_mse"
                        ),
                        proxy_mode=(
                            str(proxy_mode)
                            if model_kind == "induced_projection"
                            else "none"
                        ),
                    )
                )

    return tuple(results)


def experiment_rows(results: Sequence[HLLMergeLearningRun]) -> List[dict]:
    rows: List[dict] = []
    for r in results:
        hb = r.hll_baseline
        lm = r.learned.metrics
        bm = hb.metrics
        row = {
            "seed": int(r.seed),
            "data_seed": int(r.data_seed),
            "model_kind": str(r.model_kind),
            "objective_mode": str(r.objective_mode),
            "proxy_mode": str(r.proxy_mode),
            "lean_adjusted_loss": str(r.lean_adjusted_loss),
            "lean_merge_adapter": str(r.lean_merge_adapter),
            "lean_projection_target": str(r.lean_projection_target),
            "precision": int(r.precision),
            "registers": int(r.registers),
            "memory_bits": int(r.memory_bits),
            "memory_bytes": float(r.memory_bits) / 8.0,
            "train_docs": int(r.train_docs),
            "n_test": int(r.n_test),
            "audit_policy": str(r.audit_policy),
            "audit_fixed_nodes": int(r.audit_fixed_nodes),
            "audit_fraction": float(r.audit_fraction),
            "audit_scale": float(r.audit_scale),
            "n_epochs": int(r.n_epochs),
            "hidden_dim": int(r.hidden_dim),
            "lr": float(r.lr),
            "weight_decay": float(r.weight_decay),
            "idem_weight": float(r.idem_weight),
            "comm_weight": float(r.comm_weight),
            "train_mean_internal_nodes": float(r.train_mean_internal_nodes),
            "train_audit_nodes_mean": float(r.train_audit_nodes_mean),
            "train_audit_coverage_mean": float(r.train_audit_coverage_mean),
            "train_total_queries_estimate": int(r.train_total_queries_estimate),
            "hll_rse_theory": float(hb.rse_theory),
            "hll_relative_rmse": float(bm.relative_rmse),
            "hll_schedule_spread_mean": float(bm.schedule_spread_mean),
            "learned_relative_rmse": float(lm.relative_rmse),
            "learned_mean_abs_rel_error": float(lm.mean_abs_rel_error),
            "learned_schedule_spread_mean": float(lm.schedule_spread_mean),
            "learned_schedule_spread_p95": float(lm.schedule_spread_p95),
            "merge_mse_mean": float(r.learned.merge_mse_mean),
            "corrected_local_law_loss_mean": float(
                r.learned.corrected_local_law_loss_mean
            ),
            "proxy_loss_mean": float(r.learned.proxy_loss_mean),
            "oracle_ipw_loss_mean": float(r.learned.oracle_ipw_loss_mean),
            "ipw_correction_mean": float(r.learned.ipw_correction_mean),
            "observed_rows_mean": float(r.learned.observed_rows_mean),
            "observed_rows_per_doc": float(r.learned.observed_rows_per_doc),
            "sampled_internal_rows_mean": float(r.learned.sampled_internal_rows_mean),
            "distance_to_hll_floor_rel_rmse": float(r.distance_to_hll_floor_rel_rmse),
            "ratio_to_hll_floor_rel_rmse": float(r.ratio_to_hll_floor_rel_rmse),
            "collapse_indicator": float(1.0 if float(lm.relative_rmse) >= 0.95 else 0.0),
            "legacy_weighting_mode": str(r.legacy_weighting_mode),
            "weighting_views": r.weighting_views,
        }
        views = r.weighting_views or {}
        for family_name, prefix in (("hll_baseline", "hll"), ("learned", "learned")):
            family_views = views.get(family_name, {})
            if not isinstance(family_views, dict):
                continue
            for mode_name, metric_map in family_views.items():
                if not isinstance(metric_map, dict):
                    continue
                for metric_name in (
                    "relative_rmse",
                    "mean_abs_rel_error",
                    "schedule_spread_mean",
                    "schedule_spread_p95",
                ):
                    stats = metric_map.get(metric_name, {})
                    if not isinstance(stats, dict):
                        continue
                    key_base = f"{prefix}_{metric_name}_{mode_name}"
                    row[key_base] = stats.get("mean_hat")
                    row[f"{key_base}_se"] = stats.get("se")
                    row[f"{key_base}_ci95_low"] = stats.get("ci95_low")
                    row[f"{key_base}_ci95_high"] = stats.get("ci95_high")
        rows.append(row)
    return rows


def experiment_summary_json(config: HLLMergeLearningConfig, results: Sequence[HLLMergeLearningRun]) -> str:
    return HLLMergeLearningSummary(
        config=asdict(config),
        results=tuple(results),
    ).to_json()


__all__ = [
    "VALID_SCHEDULES",
    "ExactMaxMerger",
    "HLLMergeLearningConfig",
    "HLLMergeLearningRun",
    "HLLMergeLearningSummary",
    "HLLLocalLawTrainingDiagnostics",
    "InducedProjectionHLLMerger",
    "LearnedHLLMerger",
    "MeanMerger",
    "MergeEvalMetrics",
    "TokenStreamDoc",
    "evaluate_hll_baseline",
    "evaluate_merger_on_docs",
    "evaluate_merger_on_docs_with_weighting",
    "experiment_rows",
    "experiment_summary_json",
    "generate_token_stream_docs",
    "hll_estimate_differentiable",
    "leaf_hll_registers",
    "merge_leaf_states",
    "run_hll_merge_learning_experiment",
    "train_induced_projection_merger",
    "_merge_schedule_registers_max",
]
