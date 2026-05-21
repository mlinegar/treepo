"""
Learned mergeable-sketch simulation with an HLL baseline.

This module provides a numeric simulation comparing a learned tree sketch
against a classical mergeable sketch.

By default it runs a proxy-only latent-state baseline. When explicitly
requested, it also emits decoded approximate local-law summaries, but those
results should still be interpreted as empirical rather than Lean-certified.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from treepo.common import (
    AuditPolicyName,
    ScheduleName,
    VALID_AUDIT_POLICIES,
    VALID_SCHEDULES,
    audit_sample_count,
)
from treepo.hll import (
    HLLConfig,
    HyperLogLogSketch,
    hll_relative_standard_error,
    match_hll_precision_for_bits,
    reduce_hll_sketches,
)

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
SimulationModeName = str
VALID_SIMULATION_MODES: Tuple[SimulationModeName, ...] = (
    "latent_proxy_baseline",
    "law_backed_learned_sketch",
)

PROXY_ONLY_EVIDENCE = "proxy_only"
APPROX_AUDITED_EVIDENCE = "approx_audited"
DEFAULT_REGULARIZER_WEIGHT = 0.25
DEFAULT_SUMMARY_SHARE = 0.5
DEFAULT_LAW_STRENGTH = 1.0 - DEFAULT_SUMMARY_SHARE
DEFAULT_LAW_COMPONENT_SHARE = 1.0 / 3.0


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError(
            "PyTorch is required for TreePO cardinality recovery experiments. "
            "Install with: pip install 'treepo[torch]'"
        )


def _torch_no_grad(fn):
    if torch is None:
        return fn
    return torch.no_grad()(fn)


def _set_global_seed(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_rel_error(pred: float, truth: float) -> float:
    denom = max(1.0, float(truth))
    return (float(pred) - float(truth)) / denom


@dataclass(frozen=True)
class SimulationConfig:
    universe_size: int = 2048
    min_tokens: int = 128
    max_tokens: int = 512
    leaf_size: int = 32
    zipf_alphas: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    state_dims: Tuple[int, ...] = (16, 32, 64)
    train_docs_grid: Tuple[int, ...] = (128, 256, 512)
    train_sizes: Optional[Tuple[int, ...]] = None
    n_val: int = 256
    n_test: int = 512
    hidden_dim: int = 128
    n_epochs: int = 14
    batch_size: int = 24
    lr: float = 3e-4
    weight_decay: float = 1e-5
    c3_weight: float = 0.20
    leaf_weight: float = 0.05
    grad_clip_norm: float = 1.0
    audit_policy: AuditPolicyName = "all"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 1.0
    audit_scale: float = 1.0
    audit_include_root_query: bool = True
    idemp_weight: float = 0.05
    simulation_mode: SimulationModeName = "latent_proxy_baseline"
    regularizer_weight: float = DEFAULT_REGULARIZER_WEIGHT
    summary_regularizer_share: float = DEFAULT_SUMMARY_SHARE
    law_leaf_share: float = DEFAULT_LAW_COMPONENT_SHARE
    law_merge_share: float = DEFAULT_LAW_COMPONENT_SHARE
    law_idemp_share: float = DEFAULT_LAW_COMPONENT_SHARE
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    seed: int = 0
    data_seed: int = 0

    def resolved_train_docs_grid(self) -> Tuple[int, ...]:
        if self.train_sizes is not None:
            return tuple(int(x) for x in self.train_sizes)
        return tuple(int(x) for x in self.train_docs_grid)


@dataclass(frozen=True)
class CardinalityDocument:
    token_ids: Tuple[int, ...]
    leaf_vectors: Tuple[torch.Tensor, ...]
    leaf_cardinalities: Tuple[float, ...]
    true_cardinality: float


@dataclass(frozen=True)
class ModelEvalMetrics:
    mae: float
    rmse: float
    relative_rmse: float
    mean_rel_error: float
    mean_abs_rel_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float
    latent_merge_state_mse: float
    eps_leaf: float
    eps_merge: float
    eps_idemp: float
    evidence_status: str
    simulation_mode: str


@dataclass(frozen=True)
class HLLMetrics:
    precision: int
    registers: int
    register_bits: int
    memory_bits: int
    memory_bytes: float
    mae: float
    rmse: float
    relative_rmse: float
    mean_rel_error: float
    mean_abs_rel_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float


@dataclass(frozen=True)
class CardinalityBaselineMetrics:
    name: str
    mae: float
    rmse: float
    relative_rmse: float
    mean_rel_error: float
    mean_abs_rel_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float


@dataclass(frozen=True)
class RegularizedObjectiveMetrics:
    global_error: float
    summary_budget_penalty: float
    law_penalty: float
    combined_regularizer: float
    total: float
    regularizer_weight: float
    summary_share: float
    law_strength: float
    leaf_share: float
    merge_share: float
    idemp_share: float
    law_scale: float
    uses_proxy_law_penalty: bool


@dataclass(frozen=True)
class LearningRunSummary:
    state_dim: int
    learned_memory_bits: int
    train_size: int
    val_loss_final: float
    train_loss_final: float
    learned_metrics: ModelEvalMetrics
    hll_metrics: HLLMetrics
    exact_set_metrics: CardinalityBaselineMetrics
    sum_leaf_uniques_metrics: CardinalityBaselineMetrics
    hll_rse_theory: float
    distance_to_hll_floor_rel_rmse: float
    ratio_to_hll_floor_rel_rmse: float
    distance_to_hll_empirical_rel_rmse: float
    train_mean_tokens: float
    train_mean_leaves: float
    train_mean_internal_nodes: float
    train_audit_nodes_mean: float
    train_audit_coverage_mean: float
    train_root_queries_total: int
    train_audit_nodes_total: int
    train_total_queries_estimate: int
    rmse_gap_vs_hll: float
    abs_rel_error_gap_vs_hll: float
    # Distance-to-floor metrics (absolute RMSE domain).
    theoretical_floor_rmse: float
    excess_rmse: float
    ratio_to_floor_rmse: float
    ratio_to_floor_rel_rmse: float
    hll_empirical_excess_rmse: float
    hll_empirical_excess_rel_rmse: float
    test_cardinality_rms: float
    test_cardinality_mean: float
    regularized_objective: RegularizedObjectiveMetrics


@dataclass(frozen=True)
class ExperimentSummary:
    config: Dict[str, object]
    results: Tuple[LearningRunSummary, ...]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "results": [_serialize_learning_run(x) for x in self.results],
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _serialize_learning_run(run: LearningRunSummary) -> Dict[str, object]:
    return {
        "state_dim": run.state_dim,
        "learned_memory_bits": run.learned_memory_bits,
        "train_size": run.train_size,
        "val_loss_final": run.val_loss_final,
        "train_loss_final": run.train_loss_final,
        "exact_set_metrics": asdict(run.exact_set_metrics),
        "sum_leaf_uniques_metrics": asdict(run.sum_leaf_uniques_metrics),
        "hll_rse_theory": run.hll_rse_theory,
        "distance_to_hll_floor_rel_rmse": run.distance_to_hll_floor_rel_rmse,
        "ratio_to_hll_floor_rel_rmse": run.ratio_to_hll_floor_rel_rmse,
        "distance_to_hll_empirical_rel_rmse": run.distance_to_hll_empirical_rel_rmse,
        "train_mean_tokens": run.train_mean_tokens,
        "train_mean_leaves": run.train_mean_leaves,
        "train_mean_internal_nodes": run.train_mean_internal_nodes,
        "train_audit_nodes_mean": run.train_audit_nodes_mean,
        "train_audit_coverage_mean": run.train_audit_coverage_mean,
        "train_root_queries_total": run.train_root_queries_total,
        "train_audit_nodes_total": run.train_audit_nodes_total,
        "train_total_queries_estimate": run.train_total_queries_estimate,
        "rmse_gap_vs_hll": run.rmse_gap_vs_hll,
        "abs_rel_error_gap_vs_hll": run.abs_rel_error_gap_vs_hll,
        "theoretical_floor_rmse": run.theoretical_floor_rmse,
        "excess_rmse": run.excess_rmse,
        "ratio_to_floor_rmse": run.ratio_to_floor_rmse,
        "ratio_to_floor_rel_rmse": run.ratio_to_floor_rel_rmse,
        "hll_empirical_excess_rmse": run.hll_empirical_excess_rmse,
        "hll_empirical_excess_rel_rmse": run.hll_empirical_excess_rel_rmse,
        "test_cardinality_rms": run.test_cardinality_rms,
        "test_cardinality_mean": run.test_cardinality_mean,
        "regularized_objective": asdict(run.regularized_objective),
        "learned_metrics": asdict(run.learned_metrics),
        "hll_metrics": asdict(run.hll_metrics),
    }


def audit_sample_count(
    internal_nodes: int,
    *,
    policy: AuditPolicyName,
    fixed_nodes: int = 0,
    fraction: float = 1.0,
    scale: float = 1.0,
) -> int:
    n = int(max(0, internal_nodes))
    if n <= 0:
        return 0

    pol = str(policy)
    if pol == "all":
        q = n
    elif pol == "fixed":
        q = int(max(0, fixed_nodes))
    elif pol == "fraction":
        q = int(math.ceil(float(fraction) * float(n)))
    elif pol == "sqrt":
        q = int(math.ceil(float(scale) * math.sqrt(float(n))))
    elif pol == "log2":
        q = int(math.ceil(float(scale) * math.log2(float(n) + 1.0)))
    else:
        raise ValueError(
            f"unsupported audit policy: {policy!r}; expected one of {VALID_AUDIT_POLICIES}"
        )
    return int(max(0, min(n, q)))


def _summarize_audit_geometry(
    docs: Sequence[CardinalityDocument],
    *,
    policy: AuditPolicyName,
    fixed_nodes: int,
    fraction: float,
    scale: float,
    include_root_query: bool,
) -> Dict[str, float | int]:
    if len(docs) == 0:
        return {
            "mean_tokens": 0.0,
            "mean_leaves": 0.0,
            "mean_internal_nodes": 0.0,
            "audit_nodes_mean": 0.0,
            "audit_coverage_mean": 0.0,
            "root_queries_total": 0,
            "audit_nodes_total": 0,
            "total_queries_estimate": 0,
        }

    n_docs = int(len(docs))
    toks: List[float] = []
    leaves: List[float] = []
    internals: List[float] = []
    audits: List[float] = []
    covers: List[float] = []
    audit_nodes_total = 0

    for doc in docs:
        n_tok = int(len(doc.token_ids))
        n_leaves = int(len(doc.leaf_vectors))
        n_internal = int(max(0, n_leaves - 1))
        q = audit_sample_count(
            n_internal,
            policy=policy,
            fixed_nodes=int(fixed_nodes),
            fraction=float(fraction),
            scale=float(scale),
        )
        toks.append(float(n_tok))
        leaves.append(float(n_leaves))
        internals.append(float(n_internal))
        audits.append(float(q))
        covers.append(float(q) / float(n_internal) if n_internal > 0 else 1.0)
        audit_nodes_total += int(q)

    root_queries_total = int(n_docs if include_root_query else 0)
    return {
        "mean_tokens": float(np.mean(np.array(toks, dtype=np.float64))),
        "mean_leaves": float(np.mean(np.array(leaves, dtype=np.float64))),
        "mean_internal_nodes": float(np.mean(np.array(internals, dtype=np.float64))),
        "audit_nodes_mean": float(np.mean(np.array(audits, dtype=np.float64))),
        "audit_coverage_mean": float(np.mean(np.array(covers, dtype=np.float64))),
        "root_queries_total": int(root_queries_total),
        "audit_nodes_total": int(audit_nodes_total),
        "total_queries_estimate": int(root_queries_total + audit_nodes_total),
    }


def _normalized_law_shares(
    *,
    leaf_share: float,
    merge_share: float,
    idemp_share: float,
) -> Tuple[float, float, float]:
    raw = np.asarray(
        [max(0.0, float(leaf_share)), max(0.0, float(merge_share)), max(0.0, float(idemp_share))],
        dtype=np.float64,
    )
    total = float(np.sum(raw))
    if total <= 0.0:
        return (
            DEFAULT_LAW_COMPONENT_SHARE,
            DEFAULT_LAW_COMPONENT_SHARE,
            DEFAULT_LAW_COMPONENT_SHARE,
        )
    norm = raw / total
    return float(norm[0]), float(norm[1]), float(norm[2])


def compute_regularized_objective_metrics(
    *,
    learned_metrics: ModelEvalMetrics,
    learned_memory_bits: int,
    max_learned_memory_bits: int,
    test_cardinality_mean: float,
    regularizer_weight: float,
    summary_regularizer_share: float,
    law_leaf_share: float,
    law_merge_share: float,
    law_idemp_share: float,
) -> RegularizedObjectiveMetrics:
    reg_w = _clamp01(float(regularizer_weight))
    summary_share = _clamp01(float(summary_regularizer_share))
    leaf_share, merge_share, idemp_share = _normalized_law_shares(
        leaf_share=float(law_leaf_share),
        merge_share=float(law_merge_share),
        idemp_share=float(law_idemp_share),
    )

    global_error = float(max(0.0, float(learned_metrics.relative_rmse)))
    summary_budget_penalty = float(
        max(0.0, float(learned_memory_bits)) / max(1.0, float(max_learned_memory_bits))
    )
    law_scale = float(max(1.0, float(test_cardinality_mean)))

    if str(learned_metrics.evidence_status) == APPROX_AUDITED_EVIDENCE:
        law_penalty = float(
            leaf_share * max(0.0, float(learned_metrics.eps_leaf)) / law_scale
            + merge_share * max(0.0, float(learned_metrics.eps_merge)) / law_scale
            + idemp_share * max(0.0, float(learned_metrics.eps_idemp)) / law_scale
        )
        uses_proxy_law_penalty = False
    else:
        law_penalty = float(max(0.0, float(learned_metrics.latent_merge_state_mse)))
        uses_proxy_law_penalty = True

    combined_regularizer = float(
        summary_share * summary_budget_penalty + (1.0 - summary_share) * law_penalty
    )
    total = float((1.0 - reg_w) * global_error + reg_w * combined_regularizer)
    law_strength = float(1.0 - summary_share)
    return RegularizedObjectiveMetrics(
        global_error=global_error,
        summary_budget_penalty=summary_budget_penalty,
        law_penalty=law_penalty,
        combined_regularizer=combined_regularizer,
        total=total,
        regularizer_weight=reg_w,
        summary_share=summary_share,
        law_strength=law_strength,
        leaf_share=leaf_share,
        merge_share=merge_share,
        idemp_share=idemp_share,
        law_scale=law_scale,
        uses_proxy_law_penalty=bool(uses_proxy_law_penalty),
    )

def compute_theoretical_floor_rmse(
    hll_rse_theory: float,
    test_cardinalities: Sequence[float],
) -> float:
    """Absolute RMSE floor from HLL asymptotic theory.

    HLL RSE = 1.04/sqrt(m) is the *relative* standard error: for a
    single stream with true cardinality n, RMSE ≈ RSE × n.  Over a test
    set with varying cardinalities {n_i}, the population RMSE floor is:

        floor_rmse = RSE × sqrt(mean_i(n_i²))

    For cardinalities in the linear-counting regime (n < 2.5m), HLL
    uses a bias-corrected estimator, so empirical HLL RMSE may slightly
    exceed this asymptotic formula.  This is expected and visible as a
    small positive ``hll_empirical_excess_rmse``.
    """
    if len(test_cardinalities) == 0:
        return 0.0
    cards_sq = np.array(
        [float(n) ** 2 for n in test_cardinalities], dtype=np.float64
    )
    return float(hll_rse_theory) * math.sqrt(float(np.mean(cards_sq)))

def _build_zipf_probability_bank(
    universe_size: int,
    alphas: Sequence[float],
) -> Dict[float, np.ndarray]:
    bank: Dict[float, np.ndarray] = {}
    ranks = np.arange(1, int(universe_size) + 1, dtype=np.float64)
    for a in alphas:
        weights = np.power(ranks, -float(a))
        probs = weights / weights.sum()
        bank[float(a)] = probs.astype(np.float64)
    return bank


def _tokens_to_leaf_multihots(
    token_ids: np.ndarray,
    *,
    leaf_size: int,
    universe_size: int,
) -> Tuple[torch.Tensor, ...]:
    _require_torch()
    out: List[torch.Tensor] = []
    n = int(token_ids.shape[0])
    for start in range(0, n, int(leaf_size)):
        chunk = token_ids[start : start + int(leaf_size)]
        vec = np.zeros(int(universe_size), dtype=np.float32)
        vec[np.unique(chunk)] = 1.0
        out.append(torch.from_numpy(vec))
    return tuple(out)


def generate_cardinality_documents(
    n_docs: int,
    *,
    universe_size: int,
    min_tokens: int,
    max_tokens: int,
    leaf_size: int,
    zipf_alphas: Sequence[float],
    seed: int,
) -> Tuple[CardinalityDocument, ...]:
    if n_docs <= 0:
        return tuple()
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("require 0 < min_tokens <= max_tokens")
    if leaf_size <= 0:
        raise ValueError("leaf_size must be positive")
    if len(zipf_alphas) == 0:
        raise ValueError("zipf_alphas must be non-empty")

    rng = np.random.default_rng(int(seed))
    bank = _build_zipf_probability_bank(int(universe_size), tuple(float(a) for a in zipf_alphas))
    alphas = tuple(bank.keys())
    docs: List[CardinalityDocument] = []

    for _ in range(int(n_docs)):
        alpha = float(alphas[int(rng.integers(0, len(alphas)))])
        probs = bank[alpha]
        n_tok = int(rng.integers(int(min_tokens), int(max_tokens) + 1))
        token_ids = rng.choice(int(universe_size), size=n_tok, replace=True, p=probs).astype(np.int64)
        leaf_vectors = _tokens_to_leaf_multihots(
            token_ids,
            leaf_size=int(leaf_size),
            universe_size=int(universe_size),
        )
        leaf_cards = tuple(float(v.sum()) for v in leaf_vectors)
        true_card = float(np.unique(token_ids).shape[0])
        docs.append(
            CardinalityDocument(
                token_ids=tuple(int(x) for x in token_ids.tolist()),
                leaf_vectors=leaf_vectors,
                leaf_cardinalities=leaf_cards,
                true_cardinality=true_card,
            )
        )
    return tuple(docs)


class LearnedMergeableSketch(nn.Module):
    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int, target_scale: float):
        super().__init__()
        _require_torch()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.target_scale = float(target_scale)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.merger = nn.Sequential(
            nn.Linear(2 * self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.count_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

    def predict_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        logit = self.readout(state)
        return torch.sigmoid(logit).squeeze(-1)

    def decode_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict_norm_from_state(state) * self.target_scale

    def encode_count(self, count: torch.Tensor) -> torch.Tensor:
        if count.ndim == 0:
            count = count.reshape(1, 1)
        elif count.ndim == 1:
            count = count.unsqueeze(-1)
        return self.count_encoder(count)

    def _merge_states(
        self,
        states: Sequence[torch.Tensor],
        unions: Sequence[torch.Tensor],
        *,
        schedule: ScheduleName,
        c3_collect: bool,
        c3_audit_indices: Optional[set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if len(states) != len(unions):
            raise ValueError("states and unions must align")
        if len(states) == 0:
            raise ValueError("need at least one state")
        if len(states) == 1:
            zero = torch.zeros((), device=states[0].device, dtype=states[0].dtype)
            return states[0], zero, 0

        if schedule == "balanced":
            cur_states = list(states)
            cur_unions = list(unions)
            c3_loss = torch.zeros((), device=states[0].device, dtype=states[0].dtype)
            c3_count = 0
            merge_idx = 0
            while len(cur_states) > 1:
                nxt_states: List[torch.Tensor] = []
                nxt_unions: List[torch.Tensor] = []
                i = 0
                while i < len(cur_states):
                    if i + 1 >= len(cur_states):
                        nxt_states.append(cur_states[i])
                        nxt_unions.append(cur_unions[i])
                        i += 1
                        continue
                    merged = self.merger(torch.cat([cur_states[i], cur_states[i + 1]], dim=-1))
                    union = torch.clamp(cur_unions[i] + cur_unions[i + 1], min=0.0, max=1.0)
                    if c3_collect and (
                        c3_audit_indices is None or merge_idx in c3_audit_indices
                    ):
                        joint = self.encoder(union)
                        c3_loss = c3_loss + F.mse_loss(merged, joint, reduction="mean")
                        c3_count += 1
                    merge_idx += 1
                    nxt_states.append(merged)
                    nxt_unions.append(union)
                    i += 2
                cur_states = nxt_states
                cur_unions = nxt_unions
            return cur_states[0], c3_loss, c3_count

        if schedule in ("left_to_right", "right_to_left"):
            if schedule == "left_to_right":
                it_states = list(states)
                it_unions = list(unions)
            else:
                it_states = list(reversed(states))
                it_unions = list(reversed(unions))
            acc_state = it_states[0]
            acc_union = it_unions[0]
            c3_loss = torch.zeros((), device=acc_state.device, dtype=acc_state.dtype)
            c3_count = 0
            merge_idx = 0
            for st, un in zip(it_states[1:], it_unions[1:]):
                merged = self.merger(torch.cat([acc_state, st], dim=-1))
                acc_union = torch.clamp(acc_union + un, min=0.0, max=1.0)
                if c3_collect and (c3_audit_indices is None or merge_idx in c3_audit_indices):
                    joint = self.encoder(acc_union)
                    c3_loss = c3_loss + F.mse_loss(merged, joint, reduction="mean")
                    c3_count += 1
                acc_state = merged
                merge_idx += 1
            return acc_state, c3_loss, c3_count

        raise ValueError(f"unsupported schedule: {schedule!r}")

    def forward_doc(
        self,
        leaf_vectors: Sequence[torch.Tensor],
        leaf_cardinalities: Sequence[float],
        *,
        schedule: ScheduleName,
        collect_c3: bool = True,
        collect_leaf: bool = True,
        collect_idemp: bool = True,
        c3_audit_indices: Optional[set[int]] = None,
    ) -> Dict[str, torch.Tensor | float]:
        if len(leaf_vectors) == 0:
            raise ValueError("leaf_vectors must be non-empty")
        if len(leaf_vectors) != len(leaf_cardinalities):
            raise ValueError("leaf vectors and cardinalities must have same length")
        states = [self.encoder(v) for v in leaf_vectors]
        unions = list(leaf_vectors)
        root_state, c3_loss, c3_count = self._merge_states(
            states,
            unions,
            schedule=schedule,
            c3_collect=collect_c3,
            c3_audit_indices=c3_audit_indices,
        )
        pred_norm = self.predict_norm_from_state(root_state)
        out: Dict[str, torch.Tensor | float] = {
            "pred_norm": pred_norm,
            "pred_count": pred_norm * self.target_scale,
        }

        if collect_c3:
            out["c3_loss"] = c3_loss / max(1, c3_count)
            out["c3_count"] = float(c3_count)
        else:
            out["c3_loss"] = torch.zeros((), device=root_state.device, dtype=root_state.dtype)
            out["c3_count"] = 0.0

        if collect_leaf:
            leaf_loss = torch.zeros((), device=root_state.device, dtype=root_state.dtype)
            for state, true_leaf in zip(states, leaf_cardinalities):
                pred_leaf_norm = self.predict_norm_from_state(state)
                true_leaf_norm = torch.tensor(
                    float(true_leaf) / self.target_scale,
                    device=root_state.device,
                    dtype=pred_leaf_norm.dtype,
                )
                leaf_loss = leaf_loss + F.mse_loss(pred_leaf_norm, true_leaf_norm, reduction="mean")
            out["leaf_loss"] = leaf_loss / float(len(states))
        else:
            out["leaf_loss"] = torch.zeros((), device=root_state.device, dtype=root_state.dtype)
        if collect_idemp:
            idemp_loss = torch.zeros((), device=root_state.device, dtype=root_state.dtype)
            all_states = list(states) + [root_state]
            for st in all_states:
                pred_count = self.decode_count_from_state(st)
                re_state = self.encode_count(pred_count)
                re_count = self.decode_count_from_state(re_state).reshape_as(pred_count)
                idemp_loss = idemp_loss + F.mse_loss(re_count, pred_count, reduction="mean")
            out["idemp_loss"] = idemp_loss / float(len(all_states))
        else:
            out["idemp_loss"] = torch.zeros((), device=root_state.device, dtype=root_state.dtype)
        return out


@dataclass(frozen=True)
class TrainDiagnostics:
    train_loss_final: float
    val_loss_final: float


def _to_device_leaf_tensors(
    doc: CardinalityDocument,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[torch.Tensor], List[float]]:
    _require_torch()
    leaf_vecs: List[torch.Tensor] = []
    for v in doc.leaf_vectors:
        if v.device == device and v.dtype == dtype:
            leaf_vecs.append(v)
        else:
            leaf_vecs.append(v.to(device=device, dtype=dtype, non_blocking=True))
    leaf_cards = [float(x) for x in doc.leaf_cardinalities]
    return leaf_vecs, leaf_cards


def train_learned_model(
    model: LearnedMergeableSketch,
    train_docs: Sequence[CardinalityDocument],
    val_docs: Sequence[CardinalityDocument],
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    c3_weight: float,
    leaf_weight: float,
    idemp_weight: float,
    grad_clip_norm: float,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    audit_include_root_query: bool,
    device: torch.device,
    seed: int,
) -> TrainDiagnostics:
    _require_torch()
    if len(train_docs) == 0:
        raise ValueError("train_docs must be non-empty")

    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = random.Random(int(seed))

    train_loss_final = float("nan")
    val_loss_final = float("nan")

    idxs = list(range(len(train_docs)))
    for _ in range(int(n_epochs)):
        rng.shuffle(idxs)
        model.train()
        batch_losses: List[float] = []
        for b0 in range(0, len(idxs), int(batch_size)):
            batch_idx = idxs[b0 : b0 + int(batch_size)]
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device, dtype=torch.float32)
            for i in batch_idx:
                doc = train_docs[i]
                leaf_vecs, leaf_cards = _to_device_leaf_tensors(doc, device=device)
                n_internal = int(max(0, len(leaf_vecs) - 1))
                n_audit = audit_sample_count(
                    n_internal,
                    policy=audit_policy,
                    fixed_nodes=int(audit_fixed_nodes),
                    fraction=float(audit_fraction),
                    scale=float(audit_scale),
                )
                if n_audit <= 0:
                    c3_audit_indices: Optional[set[int]] = set()
                elif n_audit >= n_internal:
                    c3_audit_indices = None
                else:
                    c3_audit_indices = set(rng.sample(range(n_internal), k=n_audit))
                out = model.forward_doc(
                    leaf_vecs,
                    leaf_cards,
                schedule="balanced",
                collect_c3=True,
                collect_leaf=True,
                collect_idemp=True,
                c3_audit_indices=c3_audit_indices,
            )
                pred_norm = out["pred_norm"]
                true_norm = torch.tensor(
                    float(doc.true_cardinality) / model.target_scale,
                    device=device,
                    dtype=pred_norm.dtype,
                )
                if audit_include_root_query:
                    task_loss = F.mse_loss(pred_norm, true_norm, reduction="mean")
                else:
                    task_loss = torch.zeros((), device=device, dtype=pred_norm.dtype)
                doc_loss = (
                    task_loss
                    + float(c3_weight) * out["c3_loss"]
                    + float(leaf_weight) * out["leaf_loss"]
                    + float(idemp_weight) * out["idemp_loss"]
                )
                batch_loss = batch_loss + doc_loss
            batch_loss = batch_loss / float(len(batch_idx))
            batch_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()
            batch_losses.append(float(batch_loss.detach().cpu()))
        train_loss_final = float(np.mean(batch_losses))
        val_loss_final = evaluate_model_loss(
            model,
            val_docs,
            device=device,
            c3_weight=c3_weight,
            leaf_weight=leaf_weight,
            idemp_weight=idemp_weight,
            include_root_query=audit_include_root_query,
        )
    return TrainDiagnostics(
        train_loss_final=float(train_loss_final),
        val_loss_final=float(val_loss_final),
    )


@_torch_no_grad
def evaluate_model_loss(
    model: LearnedMergeableSketch,
    docs: Sequence[CardinalityDocument],
    *,
    device: torch.device,
    c3_weight: float,
    leaf_weight: float,
    idemp_weight: float,
    include_root_query: bool,
) -> float:
    _require_torch()
    if len(docs) == 0:
        return 0.0
    model.eval()
    losses: List[float] = []
    for doc in docs:
        leaf_vecs, leaf_cards = _to_device_leaf_tensors(doc, device=device)
        out = model.forward_doc(
            leaf_vecs,
            leaf_cards,
            schedule="balanced",
            collect_c3=True,
            collect_leaf=True,
            collect_idemp=True,
        )
        pred_norm = out["pred_norm"]
        true_norm = torch.tensor(
            float(doc.true_cardinality) / model.target_scale,
            device=device,
            dtype=pred_norm.dtype,
        )
        if include_root_query:
            task_loss = F.mse_loss(pred_norm, true_norm, reduction="mean")
        else:
            task_loss = torch.zeros((), device=device, dtype=pred_norm.dtype)
        total = (
            task_loss
            + float(c3_weight) * out["c3_loss"]
            + float(leaf_weight) * out["leaf_loss"]
            + float(idemp_weight) * out["idemp_loss"]
        )
        losses.append(float(total.detach().cpu()))
    return float(np.mean(losses))


@_torch_no_grad
def evaluate_learned_model(
    model: LearnedMergeableSketch,
    docs: Sequence[CardinalityDocument],
    *,
    device: torch.device,
    simulation_mode: SimulationModeName = "latent_proxy_baseline",
) -> ModelEvalMetrics:
    _require_torch()
    if len(docs) == 0:
        return ModelEvalMetrics(
            mae=0.0,
            rmse=0.0,
            relative_rmse=0.0,
            mean_rel_error=0.0,
            mean_abs_rel_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            latent_merge_state_mse=0.0,
            eps_leaf=0.0,
            eps_merge=0.0,
            eps_idemp=0.0,
            evidence_status=PROXY_ONLY_EVIDENCE,
            simulation_mode=str(simulation_mode),
        )
    model.eval()

    abs_errs: List[float] = []
    sq_errs: List[float] = []
    rel_errs: List[float] = []
    rel_sq_errs: List[float] = []
    abs_rel_errs: List[float] = []
    spreads: List[float] = []
    c3_vals: List[float] = []
    leaf_eps_vals: List[float] = []
    merge_eps_vals: List[float] = []
    idemp_vals: List[float] = []

    for doc in docs:
        leaf_vecs, leaf_cards = _to_device_leaf_tensors(doc, device=device)
        preds: Dict[str, float] = {}
        c3_for_doc = 0.0
        for sched in VALID_SCHEDULES:
            out = model.forward_doc(
                leaf_vecs,
                leaf_cards,
                schedule=sched,
                collect_c3=True,
                collect_leaf=False,
                collect_idemp=True,
            )
            pred_count = float(out["pred_count"].detach().cpu())
            preds[sched] = pred_count
            if sched == "balanced":
                c3_for_doc = float(out["c3_loss"].detach().cpu())
                idemp_vals.append(float(out["idemp_loss"].detach().cpu()))

        pred = preds["balanced"]
        truth = float(doc.true_cardinality)
        err = pred - truth
        abs_errs.append(abs(err))
        sq_errs.append(err * err)
        rel = _safe_rel_error(pred, truth)
        rel_errs.append(rel)
        rel_sq_errs.append(rel * rel)
        abs_rel_errs.append(abs(rel))
        spread = max(preds.values()) - min(preds.values())
        spreads.append(spread)
        c3_vals.append(c3_for_doc)
        with torch.no_grad():
            leaf_states = [model.encoder(v.to(device=device, dtype=torch.float32)) for v in leaf_vecs]
            for st, true_leaf in zip(leaf_states, leaf_cards):
                pred_leaf = float(model.decode_count_from_state(st).detach().cpu())
                leaf_eps_vals.append(abs(pred_leaf - float(true_leaf)))
            root_state = model._merge_states(
                leaf_states,
                [v.to(device=device, dtype=torch.float32) for v in leaf_vecs],
                schedule="balanced",
                c3_collect=False,
            )[0]
            merge_eps_vals.append(abs(float(model.decode_count_from_state(root_state).detach().cpu()) - truth))

    if str(simulation_mode) == "law_backed_learned_sketch":
        evidence_status = APPROX_AUDITED_EVIDENCE
        eps_leaf = float(np.mean(np.array(leaf_eps_vals, dtype=np.float64))) if leaf_eps_vals else 0.0
        eps_merge = float(np.mean(np.array(merge_eps_vals, dtype=np.float64))) if merge_eps_vals else 0.0
        eps_idemp = float(np.mean(np.array(idemp_vals, dtype=np.float64))) if idemp_vals else 0.0
    else:
        evidence_status = PROXY_ONLY_EVIDENCE
        eps_leaf = float("nan")
        eps_merge = float("nan")
        eps_idemp = float("nan")

    return ModelEvalMetrics(
        mae=float(np.mean(abs_errs)),
        rmse=float(math.sqrt(np.mean(sq_errs))),
        relative_rmse=float(math.sqrt(np.mean(rel_sq_errs))),
        mean_rel_error=float(np.mean(rel_errs)),
        mean_abs_rel_error=float(np.mean(abs_rel_errs)),
        schedule_spread_mean=float(np.mean(spreads)),
        schedule_spread_p95=float(np.percentile(np.array(spreads), 95.0)),
        latent_merge_state_mse=float(np.mean(c3_vals)),
        eps_leaf=eps_leaf,
        eps_merge=eps_merge,
        eps_idemp=eps_idemp,
        evidence_status=evidence_status,
        simulation_mode=str(simulation_mode),
    )


def _hll_from_leaves(config: HLLConfig, leaf_token_lists: Sequence[Sequence[int]], schedule: ScheduleName) -> HyperLogLogSketch:
    if len(leaf_token_lists) == 0:
        return HyperLogLogSketch(config)
    leaf_sketches = [HyperLogLogSketch.from_tokens(config, toks) for toks in leaf_token_lists]
    return reduce_hll_sketches(leaf_sketches, schedule=schedule)


def _cardinality_metrics_from_preds(
    *,
    name: str,
    preds_by_schedule: Sequence[Dict[str, float]],
    truths: Sequence[float],
) -> CardinalityBaselineMetrics:
    if len(preds_by_schedule) == 0:
        return CardinalityBaselineMetrics(
            name=str(name),
            mae=0.0,
            rmse=0.0,
            relative_rmse=0.0,
            mean_rel_error=0.0,
            mean_abs_rel_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
        )

    err_abs: List[float] = []
    err_sq: List[float] = []
    err_rel: List[float] = []
    err_abs_rel: List[float] = []
    spreads: List[float] = []
    for pred_map, truth in zip(preds_by_schedule, truths):
        pred = float(pred_map["balanced"])
        diff = pred - float(truth)
        rel = _safe_rel_error(pred, float(truth))
        err_abs.append(abs(diff))
        err_sq.append(diff * diff)
        err_rel.append(rel)
        err_abs_rel.append(abs(rel))
        spreads.append(max(pred_map.values()) - min(pred_map.values()))
    return CardinalityBaselineMetrics(
        name=str(name),
        mae=float(np.mean(err_abs)),
        rmse=float(math.sqrt(np.mean(err_sq))),
        relative_rmse=float(math.sqrt(np.mean(np.square(np.asarray(err_rel, dtype=np.float64))))),
        mean_rel_error=float(np.mean(err_rel)),
        mean_abs_rel_error=float(np.mean(err_abs_rel)),
        schedule_spread_mean=float(np.mean(spreads)),
        schedule_spread_p95=float(np.percentile(np.asarray(spreads, dtype=np.float64), 95.0)),
    )


def evaluate_exact_set_baseline(
    docs: Sequence[CardinalityDocument],
) -> CardinalityBaselineMetrics:
    preds = []
    truths = []
    for doc in docs:
        truth = float(doc.true_cardinality)
        truths.append(truth)
        preds.append({sched: truth for sched in VALID_SCHEDULES})
    return _cardinality_metrics_from_preds(name="exact_set", preds_by_schedule=preds, truths=truths)


def evaluate_sum_leaf_uniques_baseline(
    docs: Sequence[CardinalityDocument],
    *,
    leaf_size: int,
) -> CardinalityBaselineMetrics:
    preds = []
    truths = []
    for doc in docs:
        token_ids = tuple(int(x) for x in doc.token_ids)
        leaf_tokens = [
            token_ids[i : i + int(leaf_size)] for i in range(0, len(token_ids), int(leaf_size))
        ]
        pred = float(sum(len(set(chunk)) for chunk in leaf_tokens))
        truths.append(float(doc.true_cardinality))
        preds.append({sched: pred for sched in VALID_SCHEDULES})
    return _cardinality_metrics_from_preds(
        name="sum_leaf_uniques",
        preds_by_schedule=preds,
        truths=truths,
    )


def evaluate_hll_baseline(
    docs: Sequence[CardinalityDocument],
    *,
    precision: int,
    leaf_size: int,
    hash_bits: int = 64,
) -> HLLMetrics:
    if len(docs) == 0:
        cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
        empty = HyperLogLogSketch(cfg)
        return HLLMetrics(
            precision=int(precision),
            registers=int(empty.m),
            register_bits=int(empty.register_bits),
            memory_bits=int(empty.memory_bits),
            memory_bytes=float(empty.memory_bits) / 8.0,
            mae=0.0,
            rmse=0.0,
            relative_rmse=0.0,
            mean_rel_error=0.0,
            mean_abs_rel_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
        )

    cfg = HLLConfig(precision=int(precision), hash_bits=int(hash_bits))
    err_abs: List[float] = []
    err_sq: List[float] = []
    err_rel: List[float] = []
    err_rel_sq: List[float] = []
    err_abs_rel: List[float] = []
    spreads: List[float] = []

    for doc in docs:
        token_ids = tuple(int(x) for x in doc.token_ids)
        leaf_tokens = [
            token_ids[i : i + int(leaf_size)] for i in range(0, len(token_ids), int(leaf_size))
        ]
        ests: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            sk = _hll_from_leaves(cfg, leaf_tokens, schedule=sched)
            ests[sched] = float(sk.estimate())

        pred = ests["balanced"]
        truth = float(doc.true_cardinality)
        diff = pred - truth
        err_abs.append(abs(diff))
        err_sq.append(diff * diff)
        rel = _safe_rel_error(pred, truth)
        err_rel.append(rel)
        err_rel_sq.append(rel * rel)
        err_abs_rel.append(abs(rel))
        spreads.append(max(ests.values()) - min(ests.values()))

    proto = HyperLogLogSketch(cfg)
    return HLLMetrics(
        precision=int(precision),
        registers=int(proto.m),
        register_bits=int(proto.register_bits),
        memory_bits=int(proto.memory_bits),
        memory_bytes=float(proto.memory_bits) / 8.0,
        mae=float(np.mean(err_abs)),
        rmse=float(math.sqrt(np.mean(err_sq))),
        relative_rmse=float(math.sqrt(np.mean(err_rel_sq))),
        mean_rel_error=float(np.mean(err_rel)),
        mean_abs_rel_error=float(np.mean(err_abs_rel)),
        schedule_spread_mean=float(np.mean(spreads)),
        schedule_spread_p95=float(np.percentile(np.array(spreads), 95.0)),
    )


def run_learning_vs_hll_experiment(config: SimulationConfig) -> ExperimentSummary:
    _require_torch()
    train_docs_grid = config.resolved_train_docs_grid()
    if len(config.state_dims) == 0:
        raise ValueError("state_dims must be non-empty")
    if len(train_docs_grid) == 0:
        raise ValueError("train_docs_grid must be non-empty")
    if config.max_tokens <= config.min_tokens:
        raise ValueError("max_tokens must exceed min_tokens")
    if config.leaf_size <= 0:
        raise ValueError("leaf_size must be positive")
    if str(config.audit_policy) not in VALID_AUDIT_POLICIES:
        raise ValueError(
            f"audit_policy={config.audit_policy!r} unsupported; "
            f"expected one of {VALID_AUDIT_POLICIES}"
        )
    if float(config.audit_fraction) <= 0.0:
        raise ValueError("audit_fraction must be positive")
    if float(config.audit_scale) <= 0.0:
        raise ValueError("audit_scale must be positive")
    if int(config.audit_fixed_nodes) < 0:
        raise ValueError("audit_fixed_nodes must be non-negative")
    if not (0.0 <= float(config.regularizer_weight) <= 1.0):
        raise ValueError("regularizer_weight must be in [0, 1]")
    if not (0.0 <= float(config.summary_regularizer_share) <= 1.0):
        raise ValueError("summary_regularizer_share must be in [0, 1]")

    _set_global_seed(int(config.seed))
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

    max_train = max(int(x) for x in train_docs_grid)
    n_total = int(max_train + config.n_val + config.n_test)
    data_seed = int(config.data_seed)
    docs = generate_cardinality_documents(
        n_docs=n_total,
        universe_size=int(config.universe_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        leaf_size=int(config.leaf_size),
        zipf_alphas=config.zipf_alphas,
        seed=data_seed,
    )
    train_pool = docs[:max_train]
    val_docs = docs[max_train : max_train + int(config.n_val)]
    test_docs = docs[max_train + int(config.n_val) :]

    results: List[LearningRunSummary] = []
    hll_cache: Dict[int, HLLMetrics] = {}
    exact_metrics = evaluate_exact_set_baseline(test_docs)
    wrong_metrics = evaluate_sum_leaf_uniques_baseline(
        test_docs,
        leaf_size=int(config.leaf_size),
    )
    max_learned_memory_bits = int(max(int(x) for x in config.state_dims) * 32)

    for state_dim in (int(x) for x in config.state_dims):
        learned_bits = int(state_dim * 32)
        p = match_hll_precision_for_bits(
            learned_bits,
            hash_bits=64,
            p_min=4,
            p_max=16,
        )
        if p not in hll_cache:
            hll_cache[p] = evaluate_hll_baseline(
                test_docs,
                precision=p,
                leaf_size=int(config.leaf_size),
                hash_bits=64,
            )
        hll_metrics = hll_cache[p]
        hll_rse_theory = float(hll_relative_standard_error(p))

        # Test-set cardinality stats (constant across train_sizes).
        _test_cards = np.array(
            [float(d.true_cardinality) for d in test_docs], dtype=np.float64
        )
        _test_card_mean = float(np.mean(_test_cards))
        _test_card_rms = float(math.sqrt(np.mean(_test_cards ** 2)))
        _floor_rmse = float(hll_rse_theory * _test_card_rms)
        _hll_emp_excess_rmse = float(hll_metrics.rmse - _floor_rmse)
        _hll_emp_excess_rel = float(hll_metrics.relative_rmse - hll_rse_theory)

        for train_size in (int(x) for x in train_docs_grid):
            model = LearnedMergeableSketch(
                input_dim=int(config.universe_size),
                state_dim=state_dim,
                hidden_dim=int(config.hidden_dim),
                target_scale=float(config.max_tokens),
            )
            train_docs = train_pool[:train_size]
            diag = train_learned_model(
                model,
                train_docs,
                val_docs,
                n_epochs=int(config.n_epochs),
                batch_size=int(config.batch_size),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay),
                c3_weight=float(config.c3_weight),
                leaf_weight=float(config.leaf_weight),
                idemp_weight=float(config.idemp_weight),
                grad_clip_norm=float(config.grad_clip_norm),
                audit_policy=str(config.audit_policy),
                audit_fixed_nodes=int(config.audit_fixed_nodes),
                audit_fraction=float(config.audit_fraction),
                audit_scale=float(config.audit_scale),
                audit_include_root_query=bool(config.audit_include_root_query),
                device=device,
                seed=int(config.seed + 7919 + state_dim + train_size),
            )
            learned_metrics = evaluate_learned_model(
                model,
                test_docs,
                device=device,
                simulation_mode=str(config.simulation_mode),
            )
            geom = _summarize_audit_geometry(
                train_docs,
                policy=str(config.audit_policy),
                fixed_nodes=int(config.audit_fixed_nodes),
                fraction=float(config.audit_fraction),
                scale=float(config.audit_scale),
                include_root_query=bool(config.audit_include_root_query),
            )
            dist_floor = float(learned_metrics.relative_rmse - hll_rse_theory)
            dist_emp = float(learned_metrics.relative_rmse - hll_metrics.relative_rmse)
            regularized_objective = compute_regularized_objective_metrics(
                learned_metrics=learned_metrics,
                learned_memory_bits=learned_bits,
                max_learned_memory_bits=max_learned_memory_bits,
                test_cardinality_mean=_test_card_mean,
                regularizer_weight=float(config.regularizer_weight),
                summary_regularizer_share=float(config.summary_regularizer_share),
                law_leaf_share=float(config.law_leaf_share),
                law_merge_share=float(config.law_merge_share),
                law_idemp_share=float(config.law_idemp_share),
            )
            results.append(
                LearningRunSummary(
                    state_dim=state_dim,
                    learned_memory_bits=learned_bits,
                    train_size=train_size,
                    val_loss_final=float(diag.val_loss_final),
                    train_loss_final=float(diag.train_loss_final),
                    learned_metrics=learned_metrics,
                    hll_metrics=hll_metrics,
                    exact_set_metrics=exact_metrics,
                    sum_leaf_uniques_metrics=wrong_metrics,
                    hll_rse_theory=hll_rse_theory,
                    distance_to_hll_floor_rel_rmse=dist_floor,
                    ratio_to_hll_floor_rel_rmse=float(
                        learned_metrics.relative_rmse / max(1e-12, hll_rse_theory)
                    ),
                    distance_to_hll_empirical_rel_rmse=dist_emp,
                    train_mean_tokens=float(geom["mean_tokens"]),
                    train_mean_leaves=float(geom["mean_leaves"]),
                    train_mean_internal_nodes=float(geom["mean_internal_nodes"]),
                    train_audit_nodes_mean=float(geom["audit_nodes_mean"]),
                    train_audit_coverage_mean=float(geom["audit_coverage_mean"]),
                    train_root_queries_total=int(geom["root_queries_total"]),
                    train_audit_nodes_total=int(geom["audit_nodes_total"]),
                    train_total_queries_estimate=int(geom["total_queries_estimate"]),
                    rmse_gap_vs_hll=float(learned_metrics.rmse - hll_metrics.rmse),
                    abs_rel_error_gap_vs_hll=float(
                        learned_metrics.mean_abs_rel_error - hll_metrics.mean_abs_rel_error
                    ),
                    theoretical_floor_rmse=_floor_rmse,
                    excess_rmse=float(learned_metrics.rmse - _floor_rmse),
                    ratio_to_floor_rmse=float(
                        learned_metrics.rmse / max(1e-12, _floor_rmse)
                    ),
                    ratio_to_floor_rel_rmse=float(
                        learned_metrics.relative_rmse / max(1e-12, hll_rse_theory)
                    ),
                    hll_empirical_excess_rmse=_hll_emp_excess_rmse,
                    hll_empirical_excess_rel_rmse=_hll_emp_excess_rel,
                    test_cardinality_rms=_test_card_rms,
                    test_cardinality_mean=_test_card_mean,
                    regularized_objective=regularized_objective,
                )
            )

    cfg_dict = asdict(config)
    cfg_dict["resolved_train_docs_grid"] = [int(x) for x in train_docs_grid]
    cfg_dict["device_used"] = str(device)
    if device.type == "cuda":
        cfg_dict["cuda_current_device"] = int(torch.cuda.current_device())
        cfg_dict["cuda_device_name"] = str(
            torch.cuda.get_device_name(torch.cuda.current_device())
        )
    return ExperimentSummary(config=cfg_dict, results=tuple(results))


def experiment_rows(results: Sequence[LearningRunSummary]) -> List[dict]:
    rows: List[dict] = []
    for r in results:
        learned = r.learned_metrics
        hll = r.hll_metrics
        exact = r.exact_set_metrics
        wrong = r.sum_leaf_uniques_metrics
        reg = r.regularized_objective
        rows.append(
            {
                "state_dim": int(r.state_dim),
                "learned_memory_bits": int(r.learned_memory_bits),
                "train_docs": int(r.train_size),
                "val_loss_final": float(r.val_loss_final),
                "train_loss_final": float(r.train_loss_final),
                "hll_precision": int(hll.precision),
                "hll_registers": int(hll.registers),
                "hll_memory_bits": int(hll.memory_bits),
                "hll_rse_theory": float(r.hll_rse_theory),
                "learned_mae": float(learned.mae),
                "learned_rmse": float(learned.rmse),
                "learned_relative_rmse": float(learned.relative_rmse),
                "learned_mean_abs_rel_error": float(learned.mean_abs_rel_error),
                "learned_schedule_spread_mean": float(learned.schedule_spread_mean),
                "learned_schedule_spread_p95": float(learned.schedule_spread_p95),
                "hll_mae": float(hll.mae),
                "hll_rmse": float(hll.rmse),
                "hll_relative_rmse": float(hll.relative_rmse),
                "hll_mean_abs_rel_error": float(hll.mean_abs_rel_error),
                "hll_schedule_spread_mean": float(hll.schedule_spread_mean),
                "hll_schedule_spread_p95": float(hll.schedule_spread_p95),
                "exact_set_relative_rmse": float(exact.relative_rmse),
                "exact_set_mean_abs_rel_error": float(exact.mean_abs_rel_error),
                "sum_leaf_uniques_relative_rmse": float(wrong.relative_rmse),
                "sum_leaf_uniques_mean_abs_rel_error": float(wrong.mean_abs_rel_error),
                "distance_to_hll_floor_rel_rmse": float(r.distance_to_hll_floor_rel_rmse),
                "ratio_to_hll_floor_rel_rmse": float(r.ratio_to_hll_floor_rel_rmse),
                "distance_to_hll_empirical_rel_rmse": float(r.distance_to_hll_empirical_rel_rmse),
                "train_mean_tokens": float(r.train_mean_tokens),
                "train_mean_leaves": float(r.train_mean_leaves),
                "train_mean_internal_nodes": float(r.train_mean_internal_nodes),
                "train_audit_nodes_mean": float(r.train_audit_nodes_mean),
                "train_audit_coverage_mean": float(r.train_audit_coverage_mean),
                "train_total_queries_estimate": int(r.train_total_queries_estimate),
                "rmse_gap_vs_hll": float(r.rmse_gap_vs_hll),
                "abs_rel_error_gap_vs_hll": float(r.abs_rel_error_gap_vs_hll),
                "ratio_to_floor_rmse": float(r.ratio_to_floor_rmse),
                "ratio_to_floor_rel_rmse": float(r.ratio_to_floor_rel_rmse),
                "test_cardinality_rms": float(r.test_cardinality_rms),
                "test_cardinality_mean": float(r.test_cardinality_mean),
                "evidence_status": str(learned.evidence_status),
                "simulation_mode": str(learned.simulation_mode),
                "regularized_objective_total": float(reg.total),
                "regularized_objective_global_error": float(reg.global_error),
                "regularized_objective_summary_budget_penalty": float(
                    reg.summary_budget_penalty
                ),
                "regularized_objective_law_penalty": float(reg.law_penalty),
                "regularized_objective_combined_regularizer": float(
                    reg.combined_regularizer
                ),
                "regularized_objective_lambda": float(reg.regularizer_weight),
                "regularized_objective_summary_share": float(reg.summary_share),
                "regularized_objective_law_strength": float(reg.law_strength),
                "regularized_objective_leaf_share": float(reg.leaf_share),
                "regularized_objective_merge_share": float(reg.merge_share),
                "regularized_objective_idemp_share": float(reg.idemp_share),
                "regularized_objective_law_scale": float(reg.law_scale),
                "regularized_objective_uses_proxy_law_penalty": bool(
                    reg.uses_proxy_law_penalty
                ),
            }
        )
    return rows


CardinalityRecoveryConfig = SimulationConfig
CardinalityRecoveryRun = LearningRunSummary
CardinalityRecoverySummary = ExperimentSummary
run_cardinality_recovery_experiment = run_learning_vs_hll_experiment


__all__ = [
    "APPROX_AUDITED_EVIDENCE",
    "DEFAULT_LAW_COMPONENT_SHARE",
    "DEFAULT_LAW_STRENGTH",
    "DEFAULT_REGULARIZER_WEIGHT",
    "DEFAULT_SUMMARY_SHARE",
    "PROXY_ONLY_EVIDENCE",
    "VALID_AUDIT_POLICIES",
    "CardinalityDocument",
    "CardinalityBaselineMetrics",
    "CardinalityRecoveryConfig",
    "CardinalityRecoveryRun",
    "CardinalityRecoverySummary",
    "ExperimentSummary",
    "HLLConfig",
    "HLLMetrics",
    "LearningRunSummary",
    "LearnedMergeableSketch",
    "ModelEvalMetrics",
    "SimulationConfig",
    "VALID_SCHEDULES",
    "audit_sample_count",
    "compute_theoretical_floor_rmse",
    "evaluate_exact_set_baseline",
    "evaluate_hll_baseline",
    "evaluate_learned_model",
    "evaluate_sum_leaf_uniques_baseline",
    "experiment_rows",
    "generate_cardinality_documents",
    "hll_relative_standard_error",
    "match_hll_precision_for_bits",
    "run_cardinality_recovery_experiment",
    "run_learning_vs_hll_experiment",
    "train_learned_model",
]
