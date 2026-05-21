"""Exact-state learned-g lanes for additive mergeable sketches.

The generic scalar-sketch projection model intentionally learns a latent state
from scalar supervision. This module covers the cleaner control case: the leaf
state and readout f* are supplied, and only the binary merge g is trainable.
For additive summaries, the exact merge law is in the model class at
initialization, so these tasks should have zero error just like the HLL
register-state lane.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from treepo.training.local_law import (
    observed_uniform_node_ipw_mean_loss,
    sampled_uniform_node_ipw_mean_loss,
)

from treepo._research.unified_g_v1.dimension_guards import promote_dim
from treepo._research.unified_g_v1.sketch.classical_parity import (
    BackendName,
    ClassicalHLLParityConfig,
    ScheduleName,
    generate_documents,
)
from treepo._research.unified_g_v1.sketch.sampled_supervision import (
    attach_persistent_uniform_node_scores,
    persistent_uniform_node_mask,
    sampled_axis_mse,
    sampled_batch_mse,
    sampled_tree_node_mse,
)
from treepo._research.unified_g_v1.training.tree_task import TrainerConfig, TreeExample

ExactNumericStateKind = Literal[
    "exact_distinct_union_state_space",
    "exact_frequency_state_space",
    "count_min_state_space",
    "exact_total_weight_state_space",
]
AdditiveStateKind = ExactNumericStateKind


@dataclass(frozen=True)
class ExactNumericStateSpec:
    """Deterministic vector-state interface for exact mergeable sketch lanes."""

    target_kind: str
    state_dim: int
    g_input_dim: int
    state_space_kind: str
    merge_kind: str
    readout_kind: str


@dataclass(frozen=True)
class LearnedAdditiveStateConfig:
    target_kind: ExactNumericStateKind = "count_min_state_space"
    precision: int = 8
    n_leaves: int | None = 4
    leaf_size: int | None = None
    schedule: ScheduleName = "balanced"
    backend: BackendName = "native"
    n_train: int = 64
    n_val: int = 16
    seed: int = 0
    universe_size: int = 1_000
    min_tokens: int = 64
    max_tokens: int = 256
    zipf_alphas: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    focus_token: int = 0
    cms_num_hashes: int = 5
    cms_num_buckets: int = 256

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["zipf_alphas"] = list(self.zipf_alphas)
        return out


def _prime_at_least(n: int) -> int:
    """Small deterministic prime search for Count-Min hash arithmetic."""

    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        limit = int(math.sqrt(float(x)))
        for div in range(3, limit + 1, 2):
            if x % div == 0:
                return False
        return True

    candidate = max(3, int(n) | 1)
    while not is_prime(candidate):
        candidate += 2
    return candidate


def _cms_hash_params(num_hashes: int, universe_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    prime = _prime_at_least(max(10_007, int(universe_size) * 4 + 17))
    idx = np.arange(int(num_hashes), dtype=np.int64)
    # Fixed odd multipliers and offsets; not cryptographic, just a stable
    # independent-enough Count-Min family for the synthetic universe.
    a = ((2 * idx + 1) * 1_103_515_245 + 12_345) % prime
    b = ((idx + 17) * 2_654_435_761 + 97) % prime
    a = np.where(a == 0, 1, a)
    return a.astype(np.int64), b.astype(np.int64), int(prime)


def _cms_buckets(
    token: int,
    *,
    num_hashes: int,
    num_buckets: int,
    universe_size: int,
) -> np.ndarray:
    a, b, prime = _cms_hash_params(int(num_hashes), int(universe_size))
    tok = int(token) % int(universe_size)
    return ((a * tok + b) % prime % int(num_buckets)).astype(np.int64)


def _state_dim(cfg: LearnedAdditiveStateConfig) -> int:
    if cfg.target_kind == "exact_distinct_union_state_space":
        return int(cfg.universe_size)
    if cfg.target_kind == "count_min_state_space":
        return int(cfg.cms_num_hashes) * int(cfg.cms_num_buckets)
    return 1


def _state_spec(cfg: LearnedAdditiveStateConfig) -> ExactNumericStateSpec:
    kind = str(cfg.target_kind)
    state_dim = _state_dim(cfg)
    if kind == "exact_distinct_union_state_space":
        merge_kind = "max_union"
        readout_kind = "distinct_union_cardinality"
    elif kind == "count_min_state_space":
        merge_kind = "additive"
        readout_kind = "count_min_focus_frequency"
    elif kind == "exact_frequency_state_space":
        merge_kind = "additive"
        readout_kind = "exact_focus_frequency"
    elif kind == "exact_total_weight_state_space":
        merge_kind = "additive"
        readout_kind = "exact_total_weight"
    else:
        raise ValueError(f"unsupported exact numeric state target: {cfg.target_kind!r}")
    return ExactNumericStateSpec(
        target_kind=kind,
        state_dim=int(state_dim),
        g_input_dim=int(2 * state_dim),
        state_space_kind="fixed_numeric_vector",
        merge_kind=merge_kind,
        readout_kind=readout_kind,
    )


def exact_numeric_state_spec(cfg: LearnedAdditiveStateConfig) -> ExactNumericStateSpec:
    return _state_spec(cfg)


def _leaf_state(tokens: Sequence[int], cfg: LearnedAdditiveStateConfig) -> np.ndarray:
    kind = str(cfg.target_kind)
    if kind == "exact_distinct_union_state_space":
        state = np.zeros(int(cfg.universe_size), dtype=np.float32)
        for token in tokens:
            state[int(token) % int(cfg.universe_size)] = 1.0
        return state
    if kind == "exact_total_weight_state_space":
        return np.asarray([float(len(tokens))], dtype=np.float32)
    if kind == "exact_frequency_state_space":
        focus = int(cfg.focus_token)
        return np.asarray(
            [float(sum(1 for token in tokens if int(token) == focus))],
            dtype=np.float32,
        )
    if kind == "count_min_state_space":
        table = np.zeros(
            (int(cfg.cms_num_hashes), int(cfg.cms_num_buckets)),
            dtype=np.float32,
        )
        for token in tokens:
            buckets = _cms_buckets(
                int(token),
                num_hashes=int(cfg.cms_num_hashes),
                num_buckets=int(cfg.cms_num_buckets),
                universe_size=int(cfg.universe_size),
            )
            table[np.arange(int(cfg.cms_num_hashes)), buckets] += 1.0
        return table.reshape(-1)
    raise ValueError(f"unsupported additive state target: {cfg.target_kind!r}")


def exact_numeric_leaf_state(
    tokens: Sequence[int],
    cfg: LearnedAdditiveStateConfig,
) -> np.ndarray:
    return _leaf_state(tokens, cfg)


def _merge_state_np(
    left: np.ndarray,
    right: np.ndarray,
    cfg: LearnedAdditiveStateConfig,
) -> np.ndarray:
    spec = _state_spec(cfg)
    lhs = np.asarray(left, dtype=np.float32)
    rhs = np.asarray(right, dtype=np.float32)
    if spec.merge_kind == "max_union":
        return np.maximum(lhs, rhs).astype(np.float32, copy=False)
    if spec.merge_kind == "additive":
        return (lhs + rhs).astype(np.float32, copy=False)
    raise ValueError(f"unsupported exact numeric merge kind: {spec.merge_kind!r}")


def exact_numeric_merge_state(
    left: np.ndarray,
    right: np.ndarray,
    cfg: LearnedAdditiveStateConfig,
) -> np.ndarray:
    return _merge_state_np(left, right, cfg)


def _readout_np(state: np.ndarray, cfg: LearnedAdditiveStateConfig) -> float:
    kind = str(cfg.target_kind)
    if kind == "exact_distinct_union_state_space":
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        return float(np.clip(values, 0.0, 1.0).sum())
    if kind in {"exact_frequency_state_space", "exact_total_weight_state_space"}:
        return float(np.asarray(state, dtype=np.float32).reshape(-1)[0])
    if kind == "count_min_state_space":
        table = np.asarray(state, dtype=np.float32).reshape(
            int(cfg.cms_num_hashes),
            int(cfg.cms_num_buckets),
        )
        buckets = _cms_buckets(
            int(cfg.focus_token),
            num_hashes=int(cfg.cms_num_hashes),
            num_buckets=int(cfg.cms_num_buckets),
            universe_size=int(cfg.universe_size),
        )
        return float(np.min(table[np.arange(int(cfg.cms_num_hashes)), buckets]))
    raise ValueError(f"unsupported additive state target: {cfg.target_kind!r}")


def exact_numeric_readout(state: np.ndarray, cfg: LearnedAdditiveStateConfig) -> float:
    return _readout_np(state, cfg)


def _leaf_count_batches(
    items: Sequence[TreeExample],
    batch_size: int,
) -> list[list[TreeExample]]:
    size = max(1, int(batch_size))
    buckets: dict[int, list[TreeExample]] = {}
    order: list[int] = []
    for item in items:
        key = int(len(item.leaves))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)
    out: list[list[TreeExample]] = []
    for key in order:
        bucket = buckets[key]
        for start in range(0, len(bucket), size):
            out.append(list(bucket[start : start + size]))
    return out


class LearnedAdditiveStateOracle:
    """Train/validation oracle with cached supplied leaf and merge states."""

    def __init__(self, *, config: LearnedAdditiveStateConfig) -> None:
        self.config = config
        self._cached: list[TreeExample] | None = None

    def _to_tree_example(
        self,
        leaves: tuple[tuple[int, ...], ...],
        _analytic_truth: float,
        flat_tokens: list[int],
    ) -> TreeExample:
        leaf_states = tuple(_leaf_state(leaf, self.config) for leaf in leaves)
        leaf_values = [_readout_np(state, self.config) for state in leaf_states]
        cumulative_states: list[np.ndarray] = []
        cumulative_values: list[float] = []
        if leaf_states:
            state = np.asarray(leaf_states[0], dtype=np.float32).copy()
            for leaf_state in leaf_states[1:]:
                state = _merge_state_np(state, np.asarray(leaf_state, dtype=np.float32), self.config)
                cumulative_states.append(state.copy())
                cumulative_values.append(_readout_np(state, self.config))
            root_state = state
        else:
            root_state = np.zeros(_state_dim(self.config), dtype=np.float32)
        root_target = _readout_np(root_state, self.config)
        return TreeExample(
            leaves=leaves,
            target=root_target,
            extra={
                "flat_tokens": list(flat_tokens),
                "leaf_values": leaf_values,
                "cumulative_values": cumulative_values,
                "leaf_states": leaf_states,
                "cumulative_states": tuple(cumulative_states),
                "target_kind": str(self.config.target_kind),
            },
        )

    def _all_examples(self) -> list[TreeExample]:
        if self._cached is None:
            data_cfg = ClassicalHLLParityConfig(
                precision=int(self.config.precision),
                n_leaves=(
                    int(self.config.n_leaves)
                    if self.config.n_leaves is not None
                    else None
                ),
                leaf_size=(
                    int(self.config.leaf_size)
                    if self.config.n_leaves is None
                    else None
                ),
                schedule=self.config.schedule,
                backend=self.config.backend,
                n_val=int(self.config.n_train + self.config.n_val),
                seed=int(self.config.seed),
                universe_size=int(self.config.universe_size),
                min_tokens=int(self.config.min_tokens),
                max_tokens=int(self.config.max_tokens),
                zipf_alphas=tuple(float(a) for a in self.config.zipf_alphas),
                oracle_kind="analytic",
            )
            self._cached = [self._to_tree_example(*item) for item in generate_documents(data_cfg)]
            self._cached = attach_persistent_uniform_node_scores(
                self._cached,
                seed=int(self.config.seed),
            )
        return self._cached

    def train_examples(self) -> Sequence[TreeExample]:
        return self._all_examples()[: int(self.config.n_train)]

    def val_examples(self) -> Sequence[TreeExample]:
        start = int(self.config.n_train)
        return self._all_examples()[start : start + int(self.config.n_val)]

    def metadata(self) -> Mapping[str, Any]:
        spec = _state_spec(self.config)
        return {
            "oracle": "learned_additive_state",
            "space_kind": "numeric_sequence",
            "exact_numeric_state_spec": asdict(spec),
            **self.config.as_dict(),
        }


class OracleStateAdditiveMergeModel(nn.Module):
    """Learn exact numeric-state merges while leaf state/readout may be fixed."""

    def __init__(
        self,
        *,
        target_kind: ExactNumericStateKind,
        state_dim: int,
        merge_kind: str = "additive",
        readout_kind: str = "exact_total_weight",
        state_space_kind: str = "fixed_numeric_vector",
        cms_num_hashes: int = 5,
        cms_num_buckets: int = 256,
        focus_buckets: Sequence[int] | None = None,
        schedule: ScheduleName = "balanced",
        variant: Literal["f", "g"] = "g",
        readout_arch: Literal["structured", "mlp"] = "structured",
        readout_hidden_dim: int = 128,
        use_learned_readout: bool | None = None,
        init_from: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.target_kind = str(target_kind)
        self.state_dim = int(state_dim)
        self.merge_kind = str(merge_kind)
        self.readout_kind = str(readout_kind)
        self.state_space_kind = str(state_space_kind)
        self.cms_num_hashes = int(cms_num_hashes)
        self.cms_num_buckets = int(cms_num_buckets)
        self.schedule = str(schedule)
        if variant not in {"f", "g"}:
            raise ValueError(f"additive oracle-state variant must be 'f' or 'g', got {variant!r}")
        self.variant = str(variant)
        self.learn_readout = (
            self.variant == "f"
            if use_learned_readout is None
            else bool(use_learned_readout)
        )
        if readout_arch not in {"structured", "mlp"}:
            raise ValueError(f"readout_arch must be 'structured' or 'mlp', got {readout_arch!r}")
        self.readout_arch = str(readout_arch)
        self.left_delta = nn.Parameter(torch.zeros(self.state_dim, dtype=torch.float32))
        self.right_delta = nn.Parameter(torch.zeros(self.state_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.state_dim, dtype=torch.float32))
        self.readout_scale = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.readout_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.log_softmin_sharpness = nn.Parameter(torch.tensor(40.0, dtype=torch.float32))
        self.f_head = nn.Sequential(
            nn.Linear(self.state_dim, int(readout_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(readout_hidden_dim), 1),
        )
        if init_from is not None:
            self._load_from_checkpoint(Path(init_from))
        if self.readout_arch == "structured":
            final = self.f_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        if focus_buckets is None:
            focus_buckets = []
        self.register_buffer(
            "focus_buckets",
            torch.as_tensor(list(focus_buckets), dtype=torch.long),
            persistent=False,
        )
        if self.variant == "f":
            for parameter in (self.left_delta, self.right_delta, self.bias):
                parameter.requires_grad = False
        else:
            for parameter in (self.readout_scale, self.readout_bias, self.log_softmin_sharpness):
                parameter.requires_grad = False
            for parameter in self.f_head.parameters():
                parameter.requires_grad = False

    def _load_from_checkpoint(self, ckpt_path: Path) -> None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        own = self.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in own and tuple(value.shape) == tuple(own[key].shape)
        }
        if not compatible:
            raise ValueError(f"additive oracle-state: no compatible weights in {ckpt_path}")
        own.update(compatible)
        self.load_state_dict(own, strict=False)

    def _parameter_anchor(self) -> torch.Tensor:
        return (self.left_delta.sum() + self.right_delta.sum() + self.bias.sum()) * 0.0

    def _oracle_merge_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.merge_kind == "max_union":
            return torch.maximum(left, right)
        if self.merge_kind == "additive":
            return left + right
        raise ValueError(f"unsupported exact numeric merge kind: {self.merge_kind!r}")

    def _encode_item_leaves(self, item: TreeExample, device: torch.device) -> torch.Tensor:
        cached = item.extra.get("leaf_states") if hasattr(item, "extra") else None
        if cached is None:
            raise ValueError("additive oracle-state model requires cached leaf_states")
        return torch.as_tensor(
            np.asarray(cached, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

    def _is_rectangular_batch(self, batch: Sequence[TreeExample]) -> bool:
        if not batch:
            return False
        width = len(batch[0].leaves)
        return all(len(item.leaves) == width for item in batch)

    def _encode_batch_leaf_states(
        self,
        batch: Sequence[TreeExample],
        device: torch.device,
    ) -> torch.Tensor:
        if not self._is_rectangular_batch(batch):
            raise ValueError("additive oracle-state batch must have equal leaf counts")
        return torch.as_tensor(
            np.asarray([item.extra["leaf_states"] for item in batch], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

    def _cached_batch_cumulative_targets(
        self,
        batch: Sequence[TreeExample],
        device: torch.device,
    ) -> torch.Tensor | None:
        rows: list[Any] = []
        for item in batch:
            cached = item.extra.get("cumulative_states") if hasattr(item, "extra") else None
            if cached is None:
                return None
            rows.append(cached)
        return torch.as_tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32, device=device)

    def _predict_scalar(self, state: torch.Tensor) -> torch.Tensor:
        if self.learn_readout and self.readout_arch == "mlp":
            return self.f_head(state.reshape(-1, self.state_dim)).reshape(-1)
        if self.target_kind == "exact_distinct_union_state_space":
            value = state.reshape(-1, self.state_dim).clamp(0.0, 1.0).sum(dim=-1)
            if self.learn_readout:
                residual = self.f_head(state.reshape(-1, self.state_dim)).reshape(-1).to(value.dtype)
                return value + residual
            return value + self._parameter_anchor()
        if self.target_kind in {"exact_frequency_state_space", "exact_total_weight_state_space"}:
            value = state[..., 0].reshape(-1)
            if self.learn_readout:
                residual = self.f_head(state.reshape(-1, self.state_dim)).reshape(-1).to(value.dtype)
                return value + residual
            return value + self._parameter_anchor()
        if self.target_kind == "count_min_state_space":
            table = state.reshape(-1, self.cms_num_hashes, self.cms_num_buckets)
            hashes = torch.arange(self.cms_num_hashes, device=state.device)
            vals = table[:, hashes, self.focus_buckets.to(state.device)]
            decoded = torch.min(vals, dim=-1).values
            if self.learn_readout:
                residual = self.f_head(state.reshape(-1, self.state_dim)).reshape(-1).to(vals.dtype)
                return decoded + residual
            return decoded + self._parameter_anchor()
        raise ValueError(f"unsupported additive state target: {self.target_kind!r}")

    def _merge_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.merge_kind == "max_union":
            return self._oracle_merge_pair(left, right) + self._parameter_anchor()
        return (
            left
            + right
            + left * self.left_delta.to(left.dtype)
            + right * self.right_delta.to(right.dtype)
            + self.bias.to(left.dtype)
        )

    def _merge_states_batch(self, states: torch.Tensor, schedule: str) -> torch.Tensor:
        if states.shape[1] <= 1:
            return states[:, 0, :]
        sched = str(schedule)
        if sched == "right_to_left":
            state = states[:, -1, :]
            for idx in range(states.shape[1] - 2, -1, -1):
                state = self._merge_pair(states[:, idx, :], state)
            return state
        if sched == "balanced":
            current = states
            while current.shape[1] > 1:
                pair_count = current.shape[1] // 2
                left = current[:, : 2 * pair_count : 2, :]
                right = current[:, 1 : 2 * pair_count : 2, :]
                merged = self._merge_pair(
                    left.reshape(-1, self.state_dim),
                    right.reshape(-1, self.state_dim),
                ).reshape(current.shape[0], pair_count, self.state_dim)
                if current.shape[1] % 2:
                    current = torch.cat([merged, current[:, -1:, :]], dim=1)
                else:
                    current = merged
            return current[:, 0, :]
        state = states[:, 0, :]
        for idx in range(1, states.shape[1]):
            state = self._merge_pair(state, states[:, idx, :])
        return state

    def forward_tree(
        self,
        batch: Sequence[TreeExample],
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        device = next(self.parameters()).device
        if self._is_rectangular_batch(batch):
            leaf_states = self._encode_batch_leaf_states(batch, device)
            batch_size, n_leaves, _state_dim = leaf_states.shape
            leaf_scalars = self._predict_scalar(
                leaf_states.reshape(batch_size * n_leaves, self.state_dim)
            ).reshape(batch_size, n_leaves)
            state = leaf_states[:, 0, :]
            oracle_state = leaf_states[:, 0, :]
            cumulative_targets = self._cached_batch_cumulative_targets(batch, device)
            merge_scalars: list[torch.Tensor] = []
            merge_states: list[torch.Tensor] = []
            merge_state_targets: list[torch.Tensor] = []
            for idx in range(1, n_leaves):
                state = self._merge_pair(state, leaf_states[:, idx, :])
                oracle_state = self._oracle_merge_pair(oracle_state, leaf_states[:, idx, :])
                target_state = (
                    cumulative_targets[:, idx - 1, :]
                    if cumulative_targets is not None
                    else oracle_state
                )
                merge_scalars.append(self._predict_scalar(state).unsqueeze(1))
                merge_states.append(state.unsqueeze(1))
                merge_state_targets.append(target_state.unsqueeze(1))
            if merge_scalars:
                merge_scalars_tensor = torch.cat(merge_scalars, dim=1)
                merge_states_tensor = torch.cat(merge_states, dim=1)
                merge_state_targets_tensor = torch.cat(merge_state_targets, dim=1)
            else:
                merge_scalars_tensor = torch.zeros(batch_size, 0, device=device)
                merge_states_tensor = torch.zeros(batch_size, 0, self.state_dim, device=device)
                merge_state_targets_tensor = torch.zeros(batch_size, 0, self.state_dim, device=device)
            root_state = self._merge_states_batch(leaf_states, self.schedule)
            # One-leaf exact-state cells do not exercise the merge operator.
            # Keep a zero-valued dependency on the merge parameters so generic
            # training loops can still call backward() in fixed-readout g stages.
            root_scalar = self._predict_scalar(root_state) + self._parameter_anchor()
            return root_state, root_scalar, {
                "leaf_scalars": leaf_scalars,
                "merge_scalars": merge_scalars_tensor,
                "merge_states": merge_states_tensor,
                "merge_state_targets": merge_state_targets_tensor,
            }

        roots: list[torch.Tensor] = []
        leaf_scalars_list: list[torch.Tensor] = []
        merge_scalars_list: list[torch.Tensor] = []
        merge_states_list: list[torch.Tensor] = []
        merge_state_targets_list: list[torch.Tensor] = []
        for item in batch:
            leaf_states = self._encode_item_leaves(item, device)
            leaf_scalars = self._predict_scalar(leaf_states)
            state = leaf_states[0]
            oracle_state = leaf_states[0]
            merge_scalars: list[torch.Tensor] = []
            merge_states: list[torch.Tensor] = []
            merge_state_targets: list[torch.Tensor] = []
            cumulative_targets = item.extra.get("cumulative_states") if hasattr(item, "extra") else None
            cumulative_tensor = (
                torch.as_tensor(np.asarray(cumulative_targets, dtype=np.float32), device=device)
                if cumulative_targets is not None
                else None
            )
            for idx in range(1, leaf_states.shape[0]):
                state = self._merge_pair(state, leaf_states[idx])
                oracle_state = self._oracle_merge_pair(oracle_state, leaf_states[idx])
                target_state = cumulative_tensor[idx - 1] if cumulative_tensor is not None else oracle_state
                merge_scalars.append(self._predict_scalar(state.unsqueeze(0)).squeeze(0))
                merge_states.append(state)
                merge_state_targets.append(target_state)
            roots.append(self._merge_states_batch(leaf_states.unsqueeze(0), self.schedule))
            leaf_scalars_list.append(leaf_scalars.unsqueeze(0))
            if merge_scalars:
                merge_scalars_list.append(torch.stack(merge_scalars).unsqueeze(0))
                merge_states_list.append(torch.stack(merge_states).unsqueeze(0))
                merge_state_targets_list.append(torch.stack(merge_state_targets).unsqueeze(0))
            else:
                merge_scalars_list.append(torch.zeros(1, 0, device=device))
                merge_states_list.append(torch.zeros(1, 0, self.state_dim, device=device))
                merge_state_targets_list.append(torch.zeros(1, 0, self.state_dim, device=device))
        root_state = torch.cat(roots, dim=0)
        root_scalar = self._predict_scalar(root_state) + self._parameter_anchor()
        return root_state, root_scalar, {
            "leaf_scalars": torch.cat(leaf_scalars_list, dim=0),
            "merge_scalars": torch.cat(merge_scalars_list, dim=0),
            "merge_states": torch.cat(merge_states_list, dim=0),
            "merge_state_targets": torch.cat(merge_state_targets_list, dim=0),
        }


@dataclass
class LearnedAdditiveStateObjective:
    local_law_weight: float = 0.3
    c1_relative_weight: float = 1.0
    c2_relative_weight: float = 1.0
    c3_relative_weight: float = 1.0
    merge_state_relative_weight: float = 1.0
    root_query_rate: float = 1.0
    leaf_query_rate: float = 1.0
    internal_query_rate: float = 1.0
    supervision_sampling_policy: str = "separate_axes"

    def _stack_1d(self, items: Sequence[TreeExample], key: str) -> torch.Tensor:
        return torch.tensor([list(item.extra[key]) for item in items], dtype=torch.float32)

    def _stack_targets(self, items: Sequence[TreeExample]) -> torch.Tensor:
        return torch.tensor([float(item.target) for item in items], dtype=torch.float32)

    def compute_loss(
        self,
        *,
        root_state: torch.Tensor,
        prediction: torch.Tensor,
        batch: Sequence[TreeExample],
        forward_aux: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, int, Mapping[str, Any]]:
        del root_state
        device = prediction.device
        targets = self._stack_targets(batch).to(device=device, dtype=prediction.dtype)
        root_loss = sampled_batch_mse(
            prediction,
            targets,
            rate=float(self.root_query_rate),
        )
        if forward_aux is None:
            return root_loss, int(len(batch)), {
                "root_loss": float(root_loss.detach()),
                "root_query_rate": float(self.root_query_rate),
            }

        leaf_targets = self._stack_1d(batch, "leaf_values").to(
            device=device,
            dtype=forward_aux["leaf_scalars"].dtype,
        )
        c1_loss = sampled_axis_mse(
            forward_aux["leaf_scalars"],
            leaf_targets,
            rate=float(self.leaf_query_rate),
        )
        merge_scalars = forward_aux["merge_scalars"]
        merge_state_loss = torch.zeros((), dtype=root_loss.dtype, device=device)
        if merge_scalars.numel() > 0:
            merge_targets = self._stack_1d(batch, "cumulative_values").to(
                device=device,
                dtype=merge_scalars.dtype,
            )
            c3_scalar_loss = sampled_axis_mse(
                merge_scalars,
                merge_targets,
                rate=float(self.internal_query_rate),
            )
            if "merge_states" in forward_aux and "merge_state_targets" in forward_aux:
                merge_state_rows = (
                    forward_aux["merge_states"] - forward_aux["merge_state_targets"].detach()
                ) ** 2
                merge_state_rows = merge_state_rows.reshape(
                    int(merge_state_rows.shape[0]),
                    int(merge_state_rows.shape[1]),
                    -1,
                ).mean(dim=-1)
                merge_state_loss = sampled_uniform_node_ipw_mean_loss(
                    merge_state_rows,
                    rate=float(self.internal_query_rate),
                ).to(device)
            c3_loss = c3_scalar_loss + float(self.merge_state_relative_weight) * merge_state_loss
        else:
            c3_loss = torch.zeros((), dtype=root_loss.dtype, device=device)

        if str(self.supervision_sampling_policy) == "uniform_all_nodes":
            node_rate = (
                float(self.root_query_rate)
                if abs(float(self.root_query_rate) - float(self.leaf_query_rate)) <= 1e-12
                and abs(float(self.root_query_rate) - float(self.internal_query_rate)) <= 1e-12
                else None
            )
            if node_rate is None:
                raise ValueError(
                    "uniform_all_nodes requires equal root, leaf, and internal query rates"
                )
            # ``cumulative_values`` includes the final root merge. The uniform
            # node pool adds root explicitly, so keep only non-root internals.
            internal_pred = merge_scalars[:, :-1] if merge_scalars.ndim >= 2 else merge_scalars[:0]
            internal_target = (
                merge_targets[:, :-1]
                if merge_scalars.numel() > 0 and merge_targets.ndim >= 2
                else None
            )
            node_width = 1 + int(leaf_targets.shape[1]) + int(internal_pred.shape[1])
            node_mask = persistent_uniform_node_mask(
                list(batch),
                width=int(node_width),
                rate=float(node_rate),
                device=device,
            )
            scalar_node_loss = sampled_tree_node_mse(
                root_pred=prediction,
                root_target=targets,
                leaf_pred=forward_aux["leaf_scalars"],
                leaf_target=leaf_targets,
                internal_pred=internal_pred,
                internal_target=internal_target,
                rate=float(node_rate),
                node_mask=node_mask,
                node_propensity=float(node_rate),
            )
            persistent_merge_state_loss = torch.zeros((), dtype=root_loss.dtype, device=device)
            if (
                "merge_states" in forward_aux
                and "merge_state_targets" in forward_aux
                and internal_pred.numel() > 0
            ):
                merge_state_rows = (
                    forward_aux["merge_states"][:, :-1, :]
                    - forward_aux["merge_state_targets"][:, :-1, :].detach()
                ) ** 2
                merge_state_rows = merge_state_rows.reshape(
                    int(merge_state_rows.shape[0]),
                    int(merge_state_rows.shape[1]),
                    -1,
                ).mean(dim=-1)
                internal_mask = node_mask[:, 1 + int(leaf_targets.shape[1]) :]
                persistent_merge_state_loss = observed_uniform_node_ipw_mean_loss(
                    merge_state_rows,
                    observed=internal_mask,
                    propensity=float(node_rate),
                ).to(device)
            total = scalar_node_loss + float(self.merge_state_relative_weight) * persistent_merge_state_loss
            return total, int(len(batch)), {
                "root_loss": float(root_loss.detach()),
                "c1_loss": float(c1_loss.detach()),
                "c2_loss": 0.0,
                "c3_loss": float(c3_loss.detach()),
                "merge_state_loss": float(persistent_merge_state_loss.detach()),
                "local_block": float(total.detach()),
                "uniform_node_loss": float(scalar_node_loss.detach()),
                "local_law_weight": 1.0,
                "root_query_rate": float(self.root_query_rate),
                "leaf_query_rate": float(self.leaf_query_rate),
                "internal_query_rate": float(self.internal_query_rate),
                "node_query_rate": float(node_rate),
                "supervision_sampling_policy": str(self.supervision_sampling_policy),
            }

        rho = {
            "c1": max(0.0, float(self.c1_relative_weight)),
            "c2": max(0.0, float(self.c2_relative_weight)),
            "c3": max(0.0, float(self.c3_relative_weight)),
        }
        rho_total = rho["c1"] + rho["c2"] + rho["c3"]
        lam = max(0.0, min(1.0, float(self.local_law_weight)))
        has_root_supervision = float(self.root_query_rate) > 0.0
        has_local_supervision = (
            float(self.leaf_query_rate) > 0.0 or float(self.internal_query_rate) > 0.0
        )
        if not has_root_supervision and has_local_supervision:
            lam = 1.0
        elif has_root_supervision and not has_local_supervision:
            lam = 0.0
        if rho_total <= 0.0 or not has_local_supervision:
            total = root_loss
            local = torch.zeros((), dtype=root_loss.dtype, device=device)
        else:
            c2_loss = torch.zeros((), dtype=root_loss.dtype, device=device)
            local = (rho["c1"] * c1_loss + rho["c2"] * c2_loss + rho["c3"] * c3_loss) / rho_total
            total = (1.0 - lam) * root_loss + lam * local
        return total, int(len(batch)), {
            "root_loss": float(root_loss.detach()),
            "c1_loss": float(c1_loss.detach()),
            "c2_loss": 0.0,
            "c3_loss": float(c3_loss.detach()),
            "merge_state_loss": float(merge_state_loss.detach()),
            "local_block": float(local.detach()),
            "local_law_weight": float(lam),
            "root_query_rate": float(self.root_query_rate),
            "leaf_query_rate": float(self.leaf_query_rate),
            "internal_query_rate": float(self.internal_query_rate),
        }

    def evaluate(
        self,
        *,
        model: nn.Module,
        items: Sequence[TreeExample],
        batch_size: int,
    ) -> Mapping[str, Any]:
        model.eval()
        preds_chunks: list[torch.Tensor] = []
        targets_chunks: list[torch.Tensor] = []
        c1_chunks: list[torch.Tensor] = []
        c3_chunks: list[torch.Tensor] = []
        merge_state_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in _leaf_count_batches(list(items), int(batch_size)):
                _root_state, prediction, forward_aux = model.forward_tree(batch)
                preds_chunks.append(prediction.detach().cpu())
                targets_chunks.append(self._stack_targets(batch))
                leaf_targets = self._stack_1d(batch, "leaf_values")
                c1_chunks.append(
                    (forward_aux["leaf_scalars"].detach().cpu() - leaf_targets)
                    .abs()
                    .mean(dim=-1)
                )
                if forward_aux["merge_scalars"].numel() > 0:
                    merge_targets = self._stack_1d(batch, "cumulative_values")
                    c3_chunks.append(
                        (forward_aux["merge_scalars"].detach().cpu() - merge_targets)
                        .abs()
                        .mean(dim=-1)
                    )
                if (
                    "merge_states" in forward_aux
                    and "merge_state_targets" in forward_aux
                    and forward_aux["merge_states"].numel() > 0
                ):
                    merge_state_chunks.append(
                        (
                            forward_aux["merge_states"].detach().cpu()
                            - forward_aux["merge_state_targets"].detach().cpu()
                        )
                        .abs()
                        .mean(dim=(-1, -2))
                    )
        preds = torch.cat(preds_chunks) if preds_chunks else torch.zeros(0)
        tgts = torch.cat(targets_chunks) if targets_chunks else torch.zeros(0)
        mae = float((preds - tgts).abs().mean()) if preds.numel() else 0.0
        rmse = float(torch.sqrt(((preds - tgts) ** 2).mean())) if preds.numel() else 0.0
        scale = float(tgts.abs().clamp_min(1.0).mean()) if tgts.numel() else 1.0
        out: dict[str, Any] = {
            "count": int(len(items)),
            "mae_raw": mae,
            "mae_normalized": mae / max(1.0, scale),
            "root_mae": mae,
            "root_rmse": rmse,
            "root_rel_mae": mae / max(1.0, scale),
        }
        if c1_chunks:
            out["c1_mae"] = float(torch.cat(c1_chunks).mean())
            out["val_leaf_loss"] = out["c1_mae"]
        if c3_chunks:
            out["c3_mae"] = float(torch.cat(c3_chunks).mean())
            out["val_c3_loss"] = out["c3_mae"]
        if merge_state_chunks:
            out["merge_state_mae"] = float(torch.cat(merge_state_chunks).mean())
            out["val_merge_state_loss"] = out["merge_state_mae"]
        return out


def learned_additive_state_task(
    *,
    target_kind: ExactNumericStateKind = "count_min_state_space",
    precision: int = 8,
    n_leaves: int | None = 4,
    leaf_size: int | None = None,
    schedule: ScheduleName = "balanced",
    backend: BackendName = "native",
    n_train: int = 64,
    n_val: int = 16,
    seed: int = 0,
    universe_size: int = 1_000,
    min_tokens: int = 64,
    max_tokens: int = 256,
    zipf_alphas: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
    focus_token: int = 0,
    cms_num_hashes: int = 5,
    cms_num_buckets: int = 256,
    variant: Literal["f", "g"] = "g",
    readout_arch: Literal["structured", "mlp"] = "structured",
    readout_hidden_dim: int | None = None,
    use_learned_readout: bool | None = None,
    init_from: str | Path | None = None,
    n_epochs: int = 20,
    train_batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
    merge_state_relative_weight: float = 1.0,
    root_query_rate: float = 1.0,
    leaf_query_rate: float = 1.0,
    internal_query_rate: float = 1.0,
    supervision_sampling_policy: str = "separate_axes",
    best_metric_key: str = "mae_raw",
    use_cuda: bool = False,
    cuda_device: int | None = None,
    eval_every_n_epochs: int = 1,
    evaluate_train_on_eval: bool = True,
) -> TrainerConfig:
    cfg = LearnedAdditiveStateConfig(
        target_kind=target_kind,
        precision=int(precision),
        n_leaves=int(n_leaves) if n_leaves is not None else None,
        leaf_size=int(leaf_size) if n_leaves is None and leaf_size is not None else None,
        schedule=schedule,
        backend=backend,
        n_train=int(n_train),
        n_val=int(n_val),
        seed=int(seed),
        universe_size=int(universe_size),
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
        zipf_alphas=tuple(float(a) for a in zipf_alphas),
        focus_token=int(focus_token),
        cms_num_hashes=int(cms_num_hashes),
        cms_num_buckets=int(cms_num_buckets),
    )
    spec = _state_spec(cfg)
    state_dim = int(spec.state_dim)
    resolved_readout_hidden_dim = promote_dim(
        name="readout_hidden_dim",
        requested=readout_hidden_dim,
        default=max(128, 2 * int(state_dim)),
        minimum=2 * int(state_dim),
        context="learned_additive_state",
        reason="oracle-state readout width must cover 2*state_dim",
    )
    focus_buckets = (
        _cms_buckets(
            int(focus_token),
            num_hashes=int(cms_num_hashes),
            num_buckets=int(cms_num_buckets),
            universe_size=int(universe_size),
        ).tolist()
        if str(target_kind) == "count_min_state_space"
        else []
    )
    oracle = LearnedAdditiveStateOracle(config=cfg)
    model = OracleStateAdditiveMergeModel(
        target_kind=target_kind,
        state_dim=int(state_dim),
        merge_kind=str(spec.merge_kind),
        readout_kind=str(spec.readout_kind),
        state_space_kind=str(spec.state_space_kind),
        cms_num_hashes=int(cms_num_hashes),
        cms_num_buckets=int(cms_num_buckets),
        focus_buckets=focus_buckets,
        schedule=schedule,
        variant=variant,
        readout_arch=readout_arch,
        readout_hidden_dim=int(resolved_readout_hidden_dim),
        use_learned_readout=use_learned_readout,
        init_from=init_from,
    )
    objective = LearnedAdditiveStateObjective(
        local_law_weight=float(local_law_weight),
        c1_relative_weight=float(c1_relative_weight),
        c2_relative_weight=float(c2_relative_weight),
        c3_relative_weight=float(c3_relative_weight),
        merge_state_relative_weight=float(merge_state_relative_weight),
        root_query_rate=float(root_query_rate),
        leaf_query_rate=float(leaf_query_rate),
        internal_query_rate=float(internal_query_rate),
        supervision_sampling_policy=str(supervision_sampling_policy),
    )
    return TrainerConfig(
        oracle=oracle,
        model=model,
        objective=objective,
        n_epochs=int(n_epochs),
        train_batch_size=int(train_batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        seed=int(seed),
        best_metric_key=str(best_metric_key),
        use_cuda=bool(use_cuda),
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        extra={
            "method": "learned_f_oracle_state" if str(variant) == "f" else "learned_g_oracle_state",
            "variant": str(variant),
            "target_kind": str(target_kind),
            "fixed_oracle_state": True,
            "fixed_oracle_readout": not bool(model.learn_readout),
            "learn_readout": bool(model.learn_readout),
            "readout_arch": str(readout_arch),
            "init_from": str(init_from) if init_from is not None else None,
            "projection_kind": f"{str(target_kind).replace('_state_space', '')}_oracle_state",
            "state_space_kind": str(spec.state_space_kind),
            "merge_kind": str(spec.merge_kind),
            "readout_kind": str(spec.readout_kind),
            "exact_numeric_state_interface": True,
            "state_dim": int(state_dim),
            "summary_dim": int(state_dim),
            "embedding_dim": 0,
            "hidden_dim": int(resolved_readout_hidden_dim),
            "readout_hidden_dim": int(resolved_readout_hidden_dim),
            "leaf_width_floor": 0,
            "leaf_feature_dim": int(state_dim),
            "g_input_dim": int(2 * state_dim),
            "batch_key": "leaf_count" if n_leaves is None else "",
            "eval_every_n_epochs": int(eval_every_n_epochs),
            "evaluate_train_on_eval": bool(evaluate_train_on_eval),
            "leaf_query_rate": float(leaf_query_rate),
            "internal_query_rate": float(internal_query_rate),
        },
    )


__all__ = [
    "AdditiveStateKind",
    "ExactNumericStateKind",
    "ExactNumericStateSpec",
    "LearnedAdditiveStateConfig",
    "LearnedAdditiveStateObjective",
    "LearnedAdditiveStateOracle",
    "OracleStateAdditiveMergeModel",
    "exact_numeric_leaf_state",
    "exact_numeric_merge_state",
    "exact_numeric_readout",
    "exact_numeric_state_spec",
    "learned_additive_state_task",
]
