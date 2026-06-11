from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import random
from typing import Dict, Generic, Iterable, List, Literal, Optional, Protocol, Sequence, Tuple, TypeVar

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for exact utility transport simulations. "
        "Install with: uv sync --extra torch"
    ) from e


ObjectiveFamily = Literal[
    "supervised_state",
    "supervised_root",
    "dpo",
    "grpo",
    "ppo",
    "hybrid_supervised_plus_dpo",
    "hybrid_supervised_plus_grpo",
    "hybrid_supervised_plus_ppo",
]
UtilityLane = Literal["markov", "nonseparable", "boundary_topic"]
OracleProfile = str
StructuralArm = Literal[
    "oracle_exact",
    "tree_exact_supported",
    "tree_neural_supported",
    "tree_undersupported",
    "flat_equal_info",
    "flat_span_equal_info",
    "one_leaf_control",
    "right_rule_wrong_chunker",
]


ObsT = TypeVar("ObsT")
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class BudgetVector:
    train_docs: int
    test_docs: int
    doc_scale_tokens: float
    fixed_leaf_tokens: int
    leaves_per_doc: float
    leaf_label_coverage: float
    internal_label_coverage: float
    root_query_rate: float
    local_oracle_coverage: float
    pairwise_prefs_per_doc: float
    group_pref_groups_per_doc: float
    group_size: int
    ppo_rollouts_per_doc: float
    total_oracle_calls_estimate: float


@dataclass(frozen=True, kw_only=True)
class ExactUtilityRunConfig:
    lane: UtilityLane
    oracle_profile: str
    objective_family: ObjectiveFamily
    structural_arm: str
    train_docs: int = 256
    test_docs: int = 128
    fixed_leaf_tokens: int = 16
    seed: int = 0
    hidden_dim: int = 64
    n_epochs: int = 25
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-5
    pairwise_prefs_per_doc: int = 4
    group_pref_groups_per_doc: int = 2
    group_size: int = 4
    ppo_rollouts_per_doc: int = 4
    leaf_label_rate: float = 0.0
    internal_label_rate: float = 0.0
    root_query_rate: float = 1.0
    hybrid_weight: float = 0.5
    dpo_beta: float = 1.0
    grpo_beta: float = 1.0
    ppo_kl_weight: float = 0.02
    entropy_weight: float = 0.01
    ppo_advantage_center: bool = True
    ppo_advantage_normalize: bool = True
    ppo_reward_baseline: Literal["mean_reward", "none"] = "mean_reward"
    ppo_clip_epsilon: float = 0.2
    use_cuda: bool = False
    cuda_device: Optional[int] = None
    torch_threads: int = 0


@dataclass(frozen=True)
class ExactStateRecovery:
    exact_state_accuracy: float
    utility_regret: float
    state_l1: float
    count_mae: float = float("nan")
    first_accuracy: float = float("nan")
    last_accuracy: float = float("nan")
    theta_l1: float = float("nan")
    boundary_l1: float = float("nan")
    downstream_regret: float = float("nan")


@dataclass(frozen=True)
class ExactUtilityMetrics:
    root_recovery: ExactStateRecovery
    merge_recovery: Optional[ExactStateRecovery]
    dpo_gap_to_oracle: float
    grpo_gap_to_oracle: float
    ppo_reward_gap_to_oracle: float
    root_mae: float = float("nan")
    merge_mae: float = float("nan")
    schedule_spread: float = float("nan")
    gap_to_exact_ceiling: float = float("nan")


@dataclass(frozen=True)
class ExactUtilitySummary:
    lane: UtilityLane
    oracle_profile: str
    objective_family: ObjectiveFamily
    structural_arm: str
    config: Dict[str, object]
    budget: Dict[str, object]
    metrics: Dict[str, object]
    metadata: Dict[str, object] = field(default_factory=dict)
    objective: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "lane": self.lane,
            "oracle_profile": self.oracle_profile,
            "objective_family": self.objective_family,
            "structural_arm": self.structural_arm,
            "config": dict(self.config),
            "budget": dict(self.budget),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "objective": dict(self.objective),
        }


@dataclass(frozen=True)
class NodeObservation(Generic[ObsT]):
    leaf_obs: ObsT
    true_action_idx: int


@dataclass(frozen=True)
class UtilityDoc(Generic[ObsT]):
    leaf_observations: Tuple[ObsT, ...]
    leaf_action_indices: Tuple[int, ...]
    merge_action_indices_balanced: Tuple[int, ...]
    root_action_idx: int
    n_tokens: int
    metadata: Dict[str, object] = field(default_factory=dict)


class ExactUtilityDGP(Protocol, Generic[ObsT]):
    lane: UtilityLane
    oracle_profile: str

    @property
    def action_labels(self) -> Sequence[str]:
        ...

    @property
    def observation_dim(self) -> int:
        ...

    def sample_docs(self, *, n_docs: int, fixed_leaf_tokens: int, seed: int) -> Tuple[UtilityDoc[np.ndarray], ...]:
        ...

    def utility(self, action_idx: int, true_action_idx: int) -> float:
        ...

    def state_distance(self, action_idx: int, true_action_idx: int) -> float:
        ...

    def pair_generator(self, true_action_idx: int, *, rng: random.Random) -> Tuple[int, int]:
        ...

    def group_generator(self, true_action_idx: int, *, rng: random.Random, k: int) -> List[int]:
        ...

    def action_metrics(self, action_idx: int, true_action_idx: int) -> Dict[str, float]:
        ...

    def tree_relevance_tag(self) -> str:
        ...


def lean_theorem_refs(lane: UtilityLane, oracle_profile: str) -> List[str]:
    refs: List[str] = [
        "MainTheorems.oracle_indexed_objective_transport",
        "MainTheorems.supervised_state_objective_transport",
        "MainTheorems.normalized_state_utility_zero_regret_iff_zero_error",
        "MainTheorems.exact_mergeable_state_utility_on_tree",
    ]
    if lane == "markov":
        refs.extend(
            [
                "MainTheorems.markov_state_utility_exact_on_tree",
                "MainTheorems.markov_count_only_exact_on_tree",
                "MainTheorems.markov_count_endpoints_exact_on_tree",
            ]
        )
    elif lane == "nonseparable":
        refs.extend(
            [
                "MainTheorems.complementarity_state_utility_exact_on_tree",
                "MainTheorems.complementarity_threshold_exact_on_tree",
            ]
        )
    elif lane == "boundary_topic":
        refs.extend(
            [
                "MainTheorems.topic_state_utility_exact_on_tree",
                "MainTheorems.topic_mass_only_exact_on_tree",
                "MainTheorems.topic_plus_boundary_exact_on_tree",
            ]
        )
    return refs


def set_global_seed(seed: int, *, torch_threads: int = 0) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if int(torch_threads) > 0:
        try:
            torch.set_num_threads(int(torch_threads))
        except RuntimeError:
            pass
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(int(torch_threads))
            except RuntimeError:
                pass


def resolve_device(config: ExactUtilityRunConfig) -> torch.device:
    if bool(config.use_cuda) and torch.cuda.is_available():
        if config.cuda_device is not None:
            idx = int(config.cuda_device)
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}")
        return torch.device("cuda")
    return torch.device("cpu")


class NeuralTreePolicy(nn.Module):
    def __init__(self, *, obs_dim: int, hidden_dim: int, n_actions: int) -> None:
        super().__init__()
        self.leaf_encoder = nn.Sequential(
            nn.Linear(int(obs_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.merge_net = nn.Sequential(
            nn.Linear(2 * int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.head = nn.Linear(int(hidden_dim), int(n_actions))

    def encode_leaf(self, obs: torch.Tensor) -> torch.Tensor:
        return self.leaf_encoder(obs)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.merge_net(torch.cat([left, right], dim=-1))

    def logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(state)

    def decode_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.logits_from_state(state), dim=-1)


class FlatPolicy(nn.Module):
    def __init__(self, *, obs_dim: int, hidden_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(obs_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(n_actions)),
        )

    def logits_from_leaf_batch(self, leaf_stack: torch.Tensor) -> torch.Tensor:
        pooled_sum = leaf_stack.sum(dim=0)
        pooled_mean = leaf_stack.mean(dim=0)
        return self.net(torch.cat([pooled_sum, pooled_mean], dim=-1))


class FlatSpanPolicy(nn.Module):
    def __init__(self, *, obs_dim: int, hidden_dim: int, n_actions: int) -> None:
        super().__init__()
        self.leaf_encoder = nn.Sequential(
            nn.Linear(int(obs_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.span_head = nn.Sequential(
            nn.Linear(2 * int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(n_actions)),
        )

    def encode_leaf_batch(self, leaf_stack: torch.Tensor) -> torch.Tensor:
        return self.leaf_encoder(leaf_stack)

    def logits_from_encoded_span(self, encoded_leaf_stack: torch.Tensor, span_leaf_indices: Sequence[int]) -> torch.Tensor:
        idx = torch.tensor([int(i) for i in span_leaf_indices], device=encoded_leaf_stack.device, dtype=torch.long)
        span_states = torch.index_select(encoded_leaf_stack, dim=0, index=idx)
        pooled_mean = span_states.mean(dim=0)
        pooled_max = torch.max(span_states, dim=0).values
        return self.span_head(torch.cat([pooled_mean, pooled_max], dim=-1))


def _balanced_merge_states(
    model: NeuralTreePolicy,
    leaf_states: Sequence[torch.Tensor],
    *,
    collect_internal: bool,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    cur = list(leaf_states)
    internal: List[torch.Tensor] = []
    while len(cur) > 1:
        nxt: List[torch.Tensor] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                i += 1
                continue
            merged = model.merge(cur[i], cur[i + 1])
            if collect_internal:
                internal.append(merged)
            nxt.append(merged)
            i += 2
        cur = nxt
    return cur[0], internal


def _state_supervision_loss(
    logits: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    target = torch.tensor([int(target_idx)], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits.unsqueeze(0), target)


def _dpo_loss(
    logits: torch.Tensor,
    winner_idx: int,
    loser_idx: int,
    *,
    beta: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    margin = float(beta) * (log_probs[int(winner_idx)] - log_probs[int(loser_idx)])
    return F.softplus(-margin)


def _grpo_loss(
    logits: torch.Tensor,
    candidates: Sequence[int],
    true_idx: int,
    dgp: ExactUtilityDGP[np.ndarray],
    *,
    beta: float,
) -> torch.Tensor:
    cand_idx = [int(c) for c in candidates]
    cand_tensor = torch.tensor(cand_idx, device=logits.device, dtype=torch.long)
    cand_logits = torch.index_select(logits, dim=0, index=cand_tensor)
    rewards = np.asarray([dgp.utility(c, int(true_idx)) for c in cand_idx], dtype=np.float32)
    weights = torch.tensor(rewards, device=logits.device, dtype=logits.dtype)
    weights = torch.softmax(float(beta) * weights, dim=0)
    return -(weights * F.log_softmax(cand_logits, dim=0)).sum()


def _ppo_style_loss(
    logits: torch.Tensor,
    true_idx: int,
    dgp: ExactUtilityDGP[np.ndarray],
    *,
    n_rollouts: int,
    kl_weight: float,
    entropy_weight: float,
    advantage_center: bool,
    advantage_normalize: bool,
    reward_baseline: Literal["mean_reward", "none"],
    clip_epsilon: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    if int(n_rollouts) <= 0:
        return 0.0 * logits.sum()
    dist = torch.distributions.Categorical(probs=probs)
    sampled = dist.sample((int(n_rollouts),))
    sampled_rewards = torch.tensor(
        [dgp.utility(int(idx), int(true_idx)) for idx in sampled.detach().cpu().tolist()],
        device=logits.device,
        dtype=logits.dtype,
    )
    advantages = sampled_rewards
    if reward_baseline == "mean_reward":
        advantages = advantages - sampled_rewards.mean()
    if advantage_center:
        advantages = advantages - advantages.mean()
    if advantage_normalize:
        advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)
    sampled_log_probs = log_probs[sampled]
    if float(clip_epsilon) > 0.0:
        clipped_advantages = torch.clamp(advantages.detach(), min=-float(clip_epsilon), max=float(clip_epsilon))
    else:
        clipped_advantages = advantages.detach()
    policy_term = -(clipped_advantages * sampled_log_probs).mean()
    uniform = torch.full_like(probs, 1.0 / float(probs.numel()))
    kl = torch.sum(probs * (log_probs - torch.log(uniform + 1e-12)))
    entropy = -(probs * log_probs).sum()
    return policy_term + float(kl_weight) * kl - float(entropy_weight) * entropy


def _sample_fractional_indices(total: int, rate: float, rng: random.Random) -> List[int]:
    total_i = int(total)
    rate_f = float(rate)
    if total_i <= 0 or rate_f <= 0.0:
        return []
    if rate_f >= 1.0:
        return list(range(total_i))
    q = max(1, int(math.ceil(rate_f * float(total_i))))
    ordering = list(range(total_i))
    rng.shuffle(ordering)
    return list(sorted(ordering[: min(total_i, q)]))


def _use_root_label(rate: float, rng: random.Random) -> bool:
    rate_f = float(rate)
    if rate_f <= 0.0:
        return False
    if rate_f >= 1.0:
        return True
    return bool(rng.random() < rate_f)


def _objective_uses_root_supervision(objective: str) -> bool:
    return objective in {"supervised_state", "supervised_root"} or objective.startswith("hybrid_supervised_plus_")


def _objective_uses_local_supervision(objective: str) -> bool:
    return objective == "supervised_state" or objective.startswith("hybrid_supervised_plus_")


def _compute_budget(
    docs: Sequence[UtilityDoc[np.ndarray]],
    config: ExactUtilityRunConfig,
) -> BudgetVector:
    if len(docs) == 0:
        return BudgetVector(
            train_docs=int(config.train_docs),
            test_docs=int(config.test_docs),
            doc_scale_tokens=0.0,
            fixed_leaf_tokens=int(config.fixed_leaf_tokens),
            leaves_per_doc=0.0,
            leaf_label_coverage=float(config.leaf_label_rate),
            internal_label_coverage=float(config.internal_label_rate),
            root_query_rate=float(config.root_query_rate),
            local_oracle_coverage=0.0,
            pairwise_prefs_per_doc=float(config.pairwise_prefs_per_doc),
            group_pref_groups_per_doc=float(config.group_pref_groups_per_doc),
            group_size=int(config.group_size),
            ppo_rollouts_per_doc=float(config.ppo_rollouts_per_doc),
            total_oracle_calls_estimate=0.0,
        )
    mean_tokens = float(np.mean(np.asarray([float(d.n_tokens) for d in docs], dtype=np.float64)))
    mean_leaves = float(np.mean(np.asarray([float(len(d.leaf_observations)) for d in docs], dtype=np.float64)))
    mean_internal = float(np.mean(np.asarray([float(max(0, len(d.leaf_observations) - 1)) for d in docs], dtype=np.float64)))
    local_cov_num = float(config.leaf_label_rate) * mean_leaves + float(config.internal_label_rate) * mean_internal
    local_cov_den = max(1.0, mean_leaves + mean_internal)
    total_calls = (
        float(config.root_query_rate) * float(len(docs))
        + float(config.leaf_label_rate) * mean_leaves * float(len(docs))
        + float(config.internal_label_rate) * mean_internal * float(len(docs))
        + float(config.pairwise_prefs_per_doc) * float(len(docs))
        + float(config.group_pref_groups_per_doc) * float(config.group_size) * float(len(docs))
        + float(config.ppo_rollouts_per_doc) * float(len(docs))
    )
    return BudgetVector(
        train_docs=int(config.train_docs),
        test_docs=int(config.test_docs),
        doc_scale_tokens=float(mean_tokens),
        fixed_leaf_tokens=int(config.fixed_leaf_tokens),
        leaves_per_doc=float(mean_leaves),
        leaf_label_coverage=float(config.leaf_label_rate),
        internal_label_coverage=float(config.internal_label_rate),
        root_query_rate=float(config.root_query_rate),
        local_oracle_coverage=float(local_cov_num / local_cov_den),
        pairwise_prefs_per_doc=float(config.pairwise_prefs_per_doc),
        group_pref_groups_per_doc=float(config.group_pref_groups_per_doc),
        group_size=int(config.group_size),
        ppo_rollouts_per_doc=float(config.ppo_rollouts_per_doc),
        total_oracle_calls_estimate=float(total_calls),
    )


def realized_budget_dict(
    docs: Sequence[UtilityDoc[np.ndarray]],
    *,
    config: ExactUtilityRunConfig,
    structural_arm: str,
) -> Dict[str, object]:
    effective = config
    if str(structural_arm) == "flat_equal_info":
        effective = replace(config, leaf_label_rate=0.0, internal_label_rate=0.0)
    return asdict(_compute_budget(docs, effective))


def _action_probs_from_tree(
    model: NeuralTreePolicy,
    doc: UtilityDoc[np.ndarray],
    *,
    device: torch.device,
    one_leaf_override: bool = False,
    obs_transform=None,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    if one_leaf_override:
        obs = np.mean(np.stack([np.asarray(x, dtype=np.float32) for x in doc.leaf_observations], axis=0), axis=0)
        leaf_obs = [torch.tensor(obs_transform(obs) if obs_transform else obs, device=device, dtype=torch.float32)]
    else:
        leaf_obs = [
            torch.tensor(obs_transform(x) if obs_transform else x, device=device, dtype=torch.float32)
            for x in doc.leaf_observations
        ]
    leaf_states = [model.encode_leaf(obs) for obs in leaf_obs]
    root_state, internal_states = _balanced_merge_states(model, leaf_states, collect_internal=True)
    return model.logits_from_state(root_state), root_state, internal_states, leaf_states


def _balanced_span_leaf_indices(n_leaves: int) -> List[Tuple[int, ...]]:
    cur: List[Tuple[int, ...]] = [tuple([i]) for i in range(int(n_leaves))]
    out: List[Tuple[int, ...]] = []
    while len(cur) > 1:
        nxt: List[Tuple[int, ...]] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                i += 1
                continue
            merged = tuple(list(cur[i]) + list(cur[i + 1]))
            out.append(merged)
            nxt.append(merged)
            i += 2
        cur = nxt
    return out


def _leaf_tensor_stack(
    doc: UtilityDoc[np.ndarray],
    *,
    device: torch.device,
    obs_transform=None,
    one_leaf_override: bool = False,
) -> torch.Tensor:
    if one_leaf_override:
        obs = np.mean(np.stack([np.asarray(x, dtype=np.float32) for x in doc.leaf_observations], axis=0), axis=0)
        arr = [obs_transform(obs) if obs_transform else obs]
    else:
        arr = [obs_transform(x) if obs_transform else x for x in doc.leaf_observations]
    return torch.tensor(np.stack(arr, axis=0), device=device, dtype=torch.float32)


def train_neural_tree_policy(
    dgp: ExactUtilityDGP[np.ndarray],
    train_docs: Sequence[UtilityDoc[np.ndarray]],
    *,
    config: ExactUtilityRunConfig,
    obs_transform=None,
    one_leaf_override: bool = False,
) -> NeuralTreePolicy:
    device = resolve_device(config)
    model = NeuralTreePolicy(
        obs_dim=int(dgp.observation_dim),
        hidden_dim=int(config.hidden_dim),
        n_actions=len(dgp.action_labels),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    rng = random.Random(int(config.seed))
    objective = str(config.objective_family)

    for _ in range(int(max(1, config.n_epochs))):
        idxs = list(range(len(train_docs)))
        rng.shuffle(idxs)
        for b0 in range(0, len(idxs), int(max(1, config.batch_size))):
            batch = idxs[b0 : b0 + int(max(1, config.batch_size))]
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device, dtype=torch.float32)
            batch_terms = 0
            for i in batch:
                doc = train_docs[i]
                root_logits, _root_state, internal_states, leaf_states = _action_probs_from_tree(
                    model,
                    doc,
                    device=device,
                    one_leaf_override=one_leaf_override,
                    obs_transform=obs_transform,
                )
                sup_losses: List[torch.Tensor] = []
                obj_losses: List[torch.Tensor] = []

                if _objective_uses_root_supervision(objective) and _use_root_label(float(config.root_query_rate), rng):
                    sup_losses.append(_state_supervision_loss(root_logits, doc.root_action_idx))

                if _objective_uses_local_supervision(objective):
                    if (not one_leaf_override) and float(config.leaf_label_rate) > 0.0 and len(leaf_states) == len(doc.leaf_action_indices):
                        for idx in _sample_fractional_indices(len(leaf_states), float(config.leaf_label_rate), rng):
                            sup_losses.append(_state_supervision_loss(model.logits_from_state(leaf_states[idx]), int(doc.leaf_action_indices[idx])))
                    if float(config.internal_label_rate) > 0.0 and len(internal_states) > 0 and len(doc.merge_action_indices_balanced) > 0:
                        for idx in _sample_fractional_indices(len(internal_states), float(config.internal_label_rate), rng):
                            target_idx = int(doc.merge_action_indices_balanced[idx])
                            sup_losses.append(_state_supervision_loss(model.logits_from_state(internal_states[idx]), target_idx))

                if objective == "dpo" or objective.endswith("dpo"):
                    for _ in range(int(max(0, config.pairwise_prefs_per_doc))):
                        w, l = dgp.pair_generator(doc.root_action_idx, rng=rng)
                        obj_losses.append(_dpo_loss(root_logits, w, l, beta=float(config.dpo_beta)))
                elif objective == "grpo" or objective.endswith("grpo"):
                    for _ in range(int(max(0, config.group_pref_groups_per_doc))):
                        cand = dgp.group_generator(doc.root_action_idx, rng=rng, k=int(config.group_size))
                        obj_losses.append(
                            _grpo_loss(
                                root_logits,
                                cand,
                                doc.root_action_idx,
                                dgp,
                                beta=float(config.grpo_beta),
                            )
                        )
                elif objective == "ppo" or objective.endswith("ppo"):
                    obj = _ppo_style_loss(
                        root_logits,
                        doc.root_action_idx,
                        dgp,
                        n_rollouts=int(max(0, config.ppo_rollouts_per_doc)),
                        kl_weight=float(config.ppo_kl_weight),
                        entropy_weight=float(config.entropy_weight),
                        advantage_center=bool(config.ppo_advantage_center),
                        advantage_normalize=bool(config.ppo_advantage_normalize),
                        reward_baseline=str(config.ppo_reward_baseline),
                        clip_epsilon=float(config.ppo_clip_epsilon),
                    )
                    if obj.requires_grad:
                        obj_losses.append(obj)
                elif objective not in {"supervised_state", "supervised_root"} and not objective.startswith("hybrid_supervised_plus_"):  # pragma: no cover
                    raise ValueError(f"unsupported objective_family: {objective!r}")

                doc_loss: Optional[torch.Tensor] = None
                if objective in {"supervised_state", "supervised_root"}:
                    if sup_losses:
                        doc_loss = torch.stack(sup_losses).mean()
                elif objective in {"dpo", "grpo", "ppo"}:
                    if obj_losses:
                        doc_loss = torch.stack(obj_losses).mean()
                elif objective.startswith("hybrid_supervised_plus_"):
                    pieces: List[torch.Tensor] = []
                    if sup_losses:
                        pieces.append(float(config.hybrid_weight) * torch.stack(sup_losses).mean())
                    if obj_losses:
                        pieces.append((1.0 - float(config.hybrid_weight)) * torch.stack(obj_losses).mean())
                    if pieces:
                        doc_loss = torch.stack(pieces).sum()
                else:  # pragma: no cover
                    raise ValueError(f"unsupported objective_family: {objective!r}")

                if doc_loss is not None and doc_loss.requires_grad:
                    batch_loss = batch_loss + doc_loss
                    batch_terms += 1

            if batch_terms <= 0:
                continue
            batch_loss = batch_loss / float(batch_terms)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu()


def predict_action_tree(
    model: NeuralTreePolicy,
    doc: UtilityDoc[np.ndarray],
    *,
    obs_transform=None,
    one_leaf_override: bool = False,
) -> Tuple[int, List[int]]:
    device = next(model.parameters()).device
    with torch.no_grad():
        root_logits, _root_state, internal_states, _leaf_states = _action_probs_from_tree(
            model,
            doc,
            device=device,
            one_leaf_override=one_leaf_override,
            obs_transform=obs_transform,
        )
        root_idx = int(torch.argmax(root_logits).item())
        internal_idx = [int(torch.argmax(model.logits_from_state(st)).item()) for st in internal_states]
    return root_idx, internal_idx


def predict_action_flat(
    model: FlatPolicy,
    doc: UtilityDoc[np.ndarray],
    *,
    obs_transform=None,
    one_leaf_override: bool = False,
) -> int:
    device = next(model.parameters()).device
    leaf_stack = _leaf_tensor_stack(doc, device=device, obs_transform=obs_transform, one_leaf_override=one_leaf_override)
    with torch.no_grad():
        logits = model.logits_from_leaf_batch(leaf_stack)
    return int(torch.argmax(logits).item())


def predict_action_flat_span(
    model: FlatSpanPolicy,
    doc: UtilityDoc[np.ndarray],
    *,
    obs_transform=None,
    one_leaf_override: bool = False,
) -> Tuple[int, List[int]]:
    device = next(model.parameters()).device
    leaf_stack = _leaf_tensor_stack(doc, device=device, obs_transform=obs_transform, one_leaf_override=one_leaf_override)
    with torch.no_grad():
        encoded = model.encode_leaf_batch(leaf_stack)
        full_span = tuple(range(int(leaf_stack.shape[0])))
        root_logits = model.logits_from_encoded_span(encoded, full_span)
        root_idx = int(torch.argmax(root_logits).item())
        if one_leaf_override:
            return root_idx, []
        span_idxs = _balanced_span_leaf_indices(len(doc.leaf_observations))
        merge_idxs = [int(torch.argmax(model.logits_from_encoded_span(encoded, span)).item()) for span in span_idxs]
    return root_idx, merge_idxs


def train_flat_policy(
    dgp: ExactUtilityDGP[np.ndarray],
    train_docs: Sequence[UtilityDoc[np.ndarray]],
    *,
    config: ExactUtilityRunConfig,
    obs_transform=None,
    one_leaf_override: bool = False,
) -> FlatPolicy:
    device = resolve_device(config)
    model = FlatPolicy(obs_dim=int(dgp.observation_dim), hidden_dim=int(config.hidden_dim), n_actions=len(dgp.action_labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    rng = random.Random(int(config.seed))
    objective = str(config.objective_family)
    for _ in range(int(max(1, config.n_epochs))):
        idxs = list(range(len(train_docs)))
        rng.shuffle(idxs)
        for b0 in range(0, len(idxs), int(max(1, config.batch_size))):
            batch = idxs[b0 : b0 + int(max(1, config.batch_size))]
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device, dtype=torch.float32)
            batch_terms = 0
            for i in batch:
                doc = train_docs[i]
                if one_leaf_override:
                    obs = np.mean(np.stack([np.asarray(x, dtype=np.float32) for x in doc.leaf_observations], axis=0), axis=0)
                    arr = [obs_transform(obs) if obs_transform else obs]
                else:
                    arr = [obs_transform(x) if obs_transform else x for x in doc.leaf_observations]
                leaf_stack = torch.tensor(np.stack(arr, axis=0), device=device, dtype=torch.float32)
                logits = model.logits_from_leaf_batch(leaf_stack)
                sup_losses: List[torch.Tensor] = []
                obj_losses: List[torch.Tensor] = []

                if _objective_uses_root_supervision(objective) and _use_root_label(float(config.root_query_rate), rng):
                    sup_losses.append(_state_supervision_loss(logits, doc.root_action_idx))

                if objective == "dpo" or objective.endswith("dpo"):
                    for _ in range(int(max(0, config.pairwise_prefs_per_doc))):
                        w, l = dgp.pair_generator(doc.root_action_idx, rng=rng)
                        obj_losses.append(_dpo_loss(logits, w, l, beta=float(config.dpo_beta)))
                elif objective == "grpo" or objective.endswith("grpo"):
                    for _ in range(int(max(0, config.group_pref_groups_per_doc))):
                        cand = dgp.group_generator(doc.root_action_idx, rng=rng, k=int(config.group_size))
                        obj_losses.append(_grpo_loss(logits, cand, doc.root_action_idx, dgp, beta=float(config.grpo_beta)))
                elif objective == "ppo" or objective.endswith("ppo"):
                    obj = _ppo_style_loss(
                        logits,
                        doc.root_action_idx,
                        dgp,
                        n_rollouts=int(max(0, config.ppo_rollouts_per_doc)),
                        kl_weight=float(config.ppo_kl_weight),
                        entropy_weight=float(config.entropy_weight),
                        advantage_center=bool(config.ppo_advantage_center),
                        advantage_normalize=bool(config.ppo_advantage_normalize),
                        reward_baseline=str(config.ppo_reward_baseline),
                        clip_epsilon=float(config.ppo_clip_epsilon),
                    )
                    if obj.requires_grad:
                        obj_losses.append(obj)
                elif objective not in {"supervised_state", "supervised_root"} and not objective.startswith("hybrid_supervised_plus_"):  # pragma: no cover
                    raise ValueError(f"unsupported objective_family: {objective!r}")

                doc_loss: Optional[torch.Tensor] = None
                if objective in {"supervised_state", "supervised_root"}:
                    if sup_losses:
                        doc_loss = torch.stack(sup_losses).mean()
                elif objective in {"dpo", "grpo", "ppo"}:
                    if obj_losses:
                        doc_loss = torch.stack(obj_losses).mean()
                elif objective.startswith("hybrid_supervised_plus_"):
                    pieces: List[torch.Tensor] = []
                    if sup_losses:
                        pieces.append(float(config.hybrid_weight) * torch.stack(sup_losses).mean())
                    if obj_losses:
                        pieces.append((1.0 - float(config.hybrid_weight)) * torch.stack(obj_losses).mean())
                    if pieces:
                        doc_loss = torch.stack(pieces).sum()
                else:  # pragma: no cover
                    raise ValueError(f"unsupported objective_family: {objective!r}")

                if doc_loss is not None and doc_loss.requires_grad:
                    loss = loss + doc_loss
                    batch_terms += 1
            if batch_terms <= 0:
                continue
            loss = loss / float(batch_terms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu()


def train_flat_span_policy(
    dgp: ExactUtilityDGP[np.ndarray],
    train_docs: Sequence[UtilityDoc[np.ndarray]],
    *,
    config: ExactUtilityRunConfig,
    obs_transform=None,
    one_leaf_override: bool = False,
) -> FlatSpanPolicy:
    device = resolve_device(config)
    model = FlatSpanPolicy(obs_dim=int(dgp.observation_dim), hidden_dim=int(config.hidden_dim), n_actions=len(dgp.action_labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    rng = random.Random(int(config.seed))
    objective = str(config.objective_family)
    for _ in range(int(max(1, config.n_epochs))):
        idxs = list(range(len(train_docs)))
        rng.shuffle(idxs)
        for b0 in range(0, len(idxs), int(max(1, config.batch_size))):
            batch = idxs[b0 : b0 + int(max(1, config.batch_size))]
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device, dtype=torch.float32)
            batch_terms = 0
            for i in batch:
                doc = train_docs[i]
                leaf_stack = _leaf_tensor_stack(doc, device=device, obs_transform=obs_transform, one_leaf_override=one_leaf_override)
                encoded = model.encode_leaf_batch(leaf_stack)
                root_logits = model.logits_from_encoded_span(encoded, tuple(range(int(leaf_stack.shape[0]))))
                span_leaf_indices = [] if one_leaf_override else _balanced_span_leaf_indices(len(doc.leaf_observations))
                sup_losses: List[torch.Tensor] = []
                obj_losses: List[torch.Tensor] = []

                if _objective_uses_root_supervision(objective) and _use_root_label(float(config.root_query_rate), rng):
                    sup_losses.append(_state_supervision_loss(root_logits, doc.root_action_idx))

                if _objective_uses_local_supervision(objective):
                    if (not one_leaf_override) and float(config.leaf_label_rate) > 0.0 and len(doc.leaf_action_indices) > 0:
                        for idx in _sample_fractional_indices(len(doc.leaf_action_indices), float(config.leaf_label_rate), rng):
                            sup_losses.append(_state_supervision_loss(model.logits_from_encoded_span(encoded, (idx,)), int(doc.leaf_action_indices[idx])))
                    if float(config.internal_label_rate) > 0.0 and span_leaf_indices and len(doc.merge_action_indices_balanced) > 0:
                        for idx in _sample_fractional_indices(len(span_leaf_indices), float(config.internal_label_rate), rng):
                            sup_losses.append(
                                _state_supervision_loss(
                                    model.logits_from_encoded_span(encoded, span_leaf_indices[idx]),
                                    int(doc.merge_action_indices_balanced[idx]),
                                )
                            )

                if objective == "dpo" or objective.endswith("dpo"):
                    for _ in range(int(max(0, config.pairwise_prefs_per_doc))):
                        w, l = dgp.pair_generator(doc.root_action_idx, rng=rng)
                        obj_losses.append(_dpo_loss(root_logits, w, l, beta=float(config.dpo_beta)))
                elif objective == "grpo" or objective.endswith("grpo"):
                    for _ in range(int(max(0, config.group_pref_groups_per_doc))):
                        cand = dgp.group_generator(doc.root_action_idx, rng=rng, k=int(config.group_size))
                        obj_losses.append(_grpo_loss(root_logits, cand, doc.root_action_idx, dgp, beta=float(config.grpo_beta)))
                elif objective == "ppo" or objective.endswith("ppo"):
                    obj = _ppo_style_loss(
                        root_logits,
                        doc.root_action_idx,
                        dgp,
                        n_rollouts=int(max(0, config.ppo_rollouts_per_doc)),
                        kl_weight=float(config.ppo_kl_weight),
                        entropy_weight=float(config.entropy_weight),
                        advantage_center=bool(config.ppo_advantage_center),
                        advantage_normalize=bool(config.ppo_advantage_normalize),
                        reward_baseline=str(config.ppo_reward_baseline),
                        clip_epsilon=float(config.ppo_clip_epsilon),
                    )
                    if obj.requires_grad:
                        obj_losses.append(obj)
                elif objective not in {"supervised_state", "supervised_root"} and not objective.startswith("hybrid_supervised_plus_"):  # pragma: no cover
                    raise ValueError(f"unsupported objective_family: {objective!r}")

                doc_loss: Optional[torch.Tensor] = None
                if objective in {"supervised_state", "supervised_root"}:
                    if sup_losses:
                        doc_loss = torch.stack(sup_losses).mean()
                elif objective in {"dpo", "grpo", "ppo"}:
                    if obj_losses:
                        doc_loss = torch.stack(obj_losses).mean()
                elif objective.startswith("hybrid_supervised_plus_"):
                    pieces: List[torch.Tensor] = []
                    if sup_losses:
                        pieces.append(float(config.hybrid_weight) * torch.stack(sup_losses).mean())
                    if obj_losses:
                        pieces.append((1.0 - float(config.hybrid_weight)) * torch.stack(obj_losses).mean())
                    if pieces:
                        doc_loss = torch.stack(pieces).sum()
                else:  # pragma: no cover
                    raise ValueError(f"unsupported objective_family: {objective!r}")

                if doc_loss is not None and doc_loss.requires_grad:
                    loss = loss + doc_loss
                    batch_terms += 1
            if batch_terms <= 0:
                continue
            loss = loss / float(batch_terms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu()


def summarize_predictions(
    dgp: ExactUtilityDGP[np.ndarray],
    docs: Sequence[UtilityDoc[np.ndarray]],
    *,
    root_predictions: Sequence[int],
    merge_predictions: Optional[Sequence[Sequence[int]]] = None,
) -> ExactUtilityMetrics:
    root_rows = [dgp.action_metrics(pred, doc.root_action_idx) for pred, doc in zip(root_predictions, docs)]
    exact_acc = float(np.mean(np.asarray([r.get("exact_state_accuracy", 0.0) for r in root_rows], dtype=np.float64)))
    utility = float(np.mean(np.asarray([dgp.utility(pred, doc.root_action_idx) for pred, doc in zip(root_predictions, docs)], dtype=np.float64)))
    utility_regret = float(max(0.0, 1.0 - utility))
    state_l1 = float(np.mean(np.asarray([dgp.state_distance(pred, doc.root_action_idx) for pred, doc in zip(root_predictions, docs)], dtype=np.float64)))
    root_recovery = ExactStateRecovery(
        exact_state_accuracy=exact_acc,
        utility_regret=utility_regret,
        state_l1=state_l1,
        count_mae=float(np.mean(np.asarray([r.get("count_mae", math.nan) for r in root_rows], dtype=np.float64))),
        first_accuracy=float(np.mean(np.asarray([r.get("first_accuracy", math.nan) for r in root_rows], dtype=np.float64))),
        last_accuracy=float(np.mean(np.asarray([r.get("last_accuracy", math.nan) for r in root_rows], dtype=np.float64))),
        theta_l1=float(np.mean(np.asarray([r.get("theta_l1", math.nan) for r in root_rows], dtype=np.float64))),
        boundary_l1=float(np.mean(np.asarray([r.get("boundary_l1", math.nan) for r in root_rows], dtype=np.float64))),
        downstream_regret=float(np.mean(np.asarray([r.get("downstream_regret", math.nan) for r in root_rows], dtype=np.float64))),
    )

    merge_recovery: Optional[ExactStateRecovery] = None
    merge_mae = float("nan")
    if merge_predictions is not None:
        merge_rows: List[Dict[str, float]] = []
        merge_dists: List[float] = []
        for doc, pred_seq in zip(docs, merge_predictions):
            for pred_idx, truth_idx in zip(pred_seq, doc.merge_action_indices_balanced):
                merge_rows.append(dgp.action_metrics(int(pred_idx), int(truth_idx)))
                merge_dists.append(float(dgp.state_distance(int(pred_idx), int(truth_idx))))
        if merge_rows:
            merge_recovery = ExactStateRecovery(
                exact_state_accuracy=float(np.mean(np.asarray([r.get("exact_state_accuracy", 0.0) for r in merge_rows], dtype=np.float64))),
                utility_regret=float(np.mean(np.asarray([1.0 - r.get("utility", 0.0) for r in merge_rows], dtype=np.float64))),
                state_l1=float(np.mean(np.asarray(merge_dists, dtype=np.float64))),
                count_mae=float(np.mean(np.asarray([r.get("count_mae", math.nan) for r in merge_rows], dtype=np.float64))),
                first_accuracy=float(np.mean(np.asarray([r.get("first_accuracy", math.nan) for r in merge_rows], dtype=np.float64))),
                last_accuracy=float(np.mean(np.asarray([r.get("last_accuracy", math.nan) for r in merge_rows], dtype=np.float64))),
                theta_l1=float(np.mean(np.asarray([r.get("theta_l1", math.nan) for r in merge_rows], dtype=np.float64))),
                boundary_l1=float(np.mean(np.asarray([r.get("boundary_l1", math.nan) for r in merge_rows], dtype=np.float64))),
                downstream_regret=float(np.mean(np.asarray([r.get("downstream_regret", math.nan) for r in merge_rows], dtype=np.float64))),
            )
            merge_mae = float(np.mean(np.asarray(merge_dists, dtype=np.float64)))

    return ExactUtilityMetrics(
        root_recovery=root_recovery,
        merge_recovery=merge_recovery,
        dpo_gap_to_oracle=float(root_recovery.utility_regret),
        grpo_gap_to_oracle=float(root_recovery.utility_regret),
        ppo_reward_gap_to_oracle=float(root_recovery.utility_regret),
        root_mae=float(root_recovery.state_l1),
        merge_mae=float(merge_mae),
        schedule_spread=0.0,
        gap_to_exact_ceiling=float(root_recovery.utility_regret),
    )


def row_from_summary(summary: ExactUtilitySummary, *, source_path: str) -> Dict[str, object]:
    row = {
        "lane": summary.lane,
        "oracle_profile": summary.oracle_profile,
        "objective_family": summary.objective_family,
        "structural_arm": summary.structural_arm,
        "source_path": source_path,
    }
    row.update(summary.config)
    row.update({f"budget_{k}": v for k, v in summary.budget.items()})
    row.update({f"metric_{k}": v for k, v in summary.metrics.items()})
    return row


__all__ = [
    "BudgetVector",
    "ExactStateRecovery",
    "ExactUtilityDGP",
    "ExactUtilityMetrics",
    "ExactUtilityRunConfig",
    "ExactUtilitySummary",
    "FlatPolicy",
    "FlatSpanPolicy",
    "NeuralTreePolicy",
    "ObjectiveFamily",
    "OracleProfile",
    "StructuralArm",
    "UtilityDoc",
    "UtilityLane",
    "predict_action_flat",
    "predict_action_flat_span",
    "predict_action_tree",
    "realized_budget_dict",
    "resolve_device",
    "row_from_summary",
    "set_global_seed",
    "summarize_predictions",
    "train_flat_policy",
    "train_flat_span_policy",
    "train_neural_tree_policy",
]
