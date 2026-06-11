from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np


ScheduleName = Literal["balanced", "left_to_right", "right_to_left"]
VALID_SCHEDULES: Tuple[ScheduleName, ...] = ("balanced", "left_to_right", "right_to_left")

AuditPolicyName = Literal["all", "fixed", "fraction", "sqrt", "log2"]
VALID_AUDIT_POLICIES: Tuple[AuditPolicyName, ...] = ("all", "fixed", "fraction", "sqrt", "log2")

OutputMode = Literal["regression", "simplex"]
Span = Tuple[int, int]  # [start,end) token indices


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
        raise ValueError(f"unsupported audit policy: {policy!r}; expected one of {VALID_AUDIT_POLICIES}")
    return int(max(0, min(n, q)))


def leaf_sample_count(leaves: int, *, rate: float) -> int:
    n = int(max(0, leaves))
    if n <= 0:
        return 0
    r = float(rate)
    if r <= 0.0:
        return 0
    if r >= 1.0:
        return n
    q = int(math.ceil(r * float(n)))
    return int(max(1, min(n, q)))


def balanced_internal_spans(leaf_spans: Sequence[Span]) -> List[Span]:
    """Internal-node spans (token index) in the same order as a balanced merge reduction."""

    cur = [tuple(map(int, sp)) for sp in leaf_spans]
    out: List[Span] = []
    while len(cur) > 1:
        nxt: List[Span] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                i += 1
                continue
            left = cur[i]
            right = cur[i + 1]
            merged = (int(left[0]), int(right[1]))
            out.append(merged)
            nxt.append(merged)
            i += 2
        cur = nxt
    return out


def _merge_schedule(
    model: "TorchMergeableSketch",
    leaf_states: Sequence["torch.Tensor"],
    *,
    schedule: ScheduleName,
    collect_merge_states: bool,
) -> Tuple["torch.Tensor", List["torch.Tensor"]]:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for learned-g experiments. "
            "Install with: uv sync --extra torch"
        ) from e

    if len(leaf_states) == 0:
        raise ValueError("need at least one leaf state")
    if len(leaf_states) == 1:
        return leaf_states[0], []

    merges: List["torch.Tensor"] = []
    sched = str(schedule)
    if sched == "balanced":
        cur = list(leaf_states)
        while len(cur) > 1:
            nxt: List["torch.Tensor"] = []
            i = 0
            while i < len(cur):
                if i + 1 >= len(cur):
                    nxt.append(cur[i])
                    i += 1
                    continue
                merged = model.merge(cur[i], cur[i + 1])
                nxt.append(merged)
                if collect_merge_states:
                    merges.append(merged)
                i += 2
            cur = nxt
        return cur[0], merges

    if sched == "left_to_right":
        state = leaf_states[0]
        for i in range(1, len(leaf_states)):
            state = model.merge(state, leaf_states[i])
            if collect_merge_states:
                merges.append(state)
        return state, merges

    if sched == "right_to_left":
        state = leaf_states[-1]
        for i in range(len(leaf_states) - 2, -1, -1):
            state = model.merge(leaf_states[i], state)
            if collect_merge_states:
                merges.append(state)
        return state, merges

    raise ValueError(f"unsupported schedule: {schedule!r}; expected one of {VALID_SCHEDULES}")


def _safe_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def _p95(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), 95.0)) if vals else float("nan")


def _median(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def schedule_spread_l1(root_preds: Sequence[np.ndarray]) -> float:
    preds = [np.asarray(p, dtype=np.float64).reshape(-1) for p in root_preds]
    if len(preds) <= 1:
        return 0.0
    best = 0.0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            best = max(best, float(np.sum(np.abs(preds[i] - preds[j]))))
    return float(best)


@dataclass(frozen=True)
class LearnedGTrainingConfig:
    # Model.
    state_dim: int = 32
    hidden_dim: int = 128
    output_mode: OutputMode = "regression"
    include_endpoints: bool = False
    endpoint_categories: int = 0
    include_mass: bool = False

    # Optimization.
    n_epochs: int = 10
    batch_docs: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    # Supervision / budgets.
    leaf_query_rate: float = 1.0
    audit_policy: AuditPolicyName = "fraction"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 0.2
    audit_scale: float = 1.0
    include_root_query: bool = True

    # Objective weights.
    root_weight: float = 1.0
    leaf_weight: float = 0.05
    c3_weight: float = 0.20
    schedule_consistency_weight: float = 0.0
    idempotence_weight: float = 0.0

    # Metrics.
    violation_tau: float = 0.0

    seed: int = 0
    torch_threads: int = 1


@dataclass(frozen=True)
class LearnedGDoc:
    leaf_features: np.ndarray  # [L, D]
    leaf_oracle: np.ndarray  # [L, O]
    internal_oracle_balanced: np.ndarray  # [L-1, O] (balanced merge order)
    root_oracle: np.ndarray  # [O]
    leaf_first_onehot: Optional[np.ndarray] = None  # [L, C]
    leaf_last_onehot: Optional[np.ndarray] = None  # [L, C]
    leaf_mass: Optional[np.ndarray] = None  # [L]


@dataclass(frozen=True)
class LearnedGMetrics:
    root_mae: float
    root_median_abs_error: float
    root_p95_abs_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float
    leaf_mae: float
    leaf_violation_rate: float
    merge_mae: float
    merge_violation_rate: float
    n_docs: int


@dataclass(frozen=True)
class LearnedGTrainingGeometry:
    n_docs: int
    mean_leaves: float
    mean_internal_nodes: float
    mean_leaf_labels: float
    mean_internal_labels: float
    mean_queries_per_doc: float
    root_queries_total: int
    leaf_labels_total: int
    internal_labels_total: int
    total_labels_total: int


class TorchMergeableSketch:  # pragma: no cover - exercised by torch-dependent tests
    """
    CPU-only torch model for a mergeable sketch:
      g_leaf: x_leaf -> state
      merge: (state_left, state_right) -> state_parent
      readout: state -> oracle prediction

    The state can optionally carry:
      - endpoints (first/last one-hot categories) propagated exactly,
      - mass (scalar) propagated additively.
    """

    def __init__(
        self,
        *,
        leaf_feature_dim: int,
        output_dim: int,
        config: LearnedGTrainingConfig,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for learned-g experiments. "
                "Install with: uv sync --extra torch"
            ) from e

        self.torch = torch
        self.nn = nn
        self.output_mode: OutputMode = str(config.output_mode)  # type: ignore[assignment]
        if self.output_mode not in {"regression", "simplex"}:
            raise ValueError("output_mode must be 'regression' or 'simplex'")

        self.include_endpoints = bool(config.include_endpoints)
        self.endpoint_categories = int(config.endpoint_categories) if self.include_endpoints else 0
        if self.include_endpoints and self.endpoint_categories <= 0:
            raise ValueError("endpoint_categories must be > 0 when include_endpoints=True")

        self.include_mass = bool(config.include_mass)
        self.state_dim = int(max(1, int(config.state_dim)))
        self.hidden_dim = int(max(4, int(config.hidden_dim)))
        self.leaf_feature_dim = int(max(1, int(leaf_feature_dim)))
        self.output_dim = int(max(1, int(output_dim)))

        core_in = self.leaf_feature_dim
        if self.include_mass:
            core_in += 1

        self.model = nn.Module()  # dummy container
        self.model.leaf_encoder = nn.Sequential(
            nn.Linear(int(core_in), int(self.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_dim), int(self.state_dim)),
        )

        merge_in = 2 * int(self.state_dim)
        if self.include_mass:
            merge_in += 2  # mass_left, mass_right
        if self.include_endpoints:
            merge_in += 2 * int(self.endpoint_categories)  # left_last, right_first
        self.model.merger = nn.Sequential(
            nn.Linear(int(merge_in), int(self.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_dim), int(self.state_dim)),
        )
        self.model.readout = nn.Linear(int(self.state_dim), int(self.output_dim))

        if float(config.idempotence_weight) > 0.0:
            self.model.idem = nn.Sequential(
                nn.Linear(self._state_total_dim(), int(self.hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(self.hidden_dim), self._state_total_dim()),
            )
        else:
            self.model.idem = None

    def parameters(self) -> Iterable["torch.nn.parameter.Parameter"]:
        return self.model.parameters()

    def _state_total_dim(self) -> int:
        d = int(self.state_dim)
        if self.include_mass:
            d += 1
        if self.include_endpoints:
            d += 2 * int(self.endpoint_categories)
        return int(d)

    def _split_state(
        self, state: "torch.Tensor"
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        # Returns: h, mass, first, last (where absent are None)
        d = int(self.state_dim)
        if state.shape[-1] != self._state_total_dim():
            raise ValueError("unexpected state dimension")
        h = state[..., :d]
        offset = d
        mass = None
        if self.include_mass:
            mass = state[..., offset : offset + 1]
            offset += 1
        first = None
        last = None
        if self.include_endpoints:
            c = int(self.endpoint_categories)
            first = state[..., offset : offset + c]
            last = state[..., offset + c : offset + 2 * c]
        return h, mass, first, last

    def encode_leaf(
        self,
        x: "torch.Tensor",
        *,
        mass: Optional["torch.Tensor"] = None,
        first_onehot: Optional["torch.Tensor"] = None,
        last_onehot: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        if x.ndim != 1:
            raise ValueError("encode_leaf expects a 1D feature vector")
        parts: List["torch.Tensor"] = [x]
        if self.include_mass:
            if mass is None:
                raise ValueError("mass required when include_mass=True")
            parts.append(mass.reshape(1))
        core = self.torch.cat(parts, dim=0)
        h = self.model.leaf_encoder(core)

        out_parts: List["torch.Tensor"] = [h]
        if self.include_mass:
            out_parts.append(mass.reshape(1))  # type: ignore[union-attr]
        if self.include_endpoints:
            if first_onehot is None or last_onehot is None:
                raise ValueError("endpoints required when include_endpoints=True")
            out_parts.append(first_onehot)
            out_parts.append(last_onehot)
        return self.torch.cat(out_parts, dim=0) if len(out_parts) > 1 else h

    def merge(self, left: "torch.Tensor", right: "torch.Tensor") -> "torch.Tensor":
        h_l, mass_l, _first_l, last_l = self._split_state(left)
        h_r, mass_r, first_r, _last_r = self._split_state(right)

        feats: List["torch.Tensor"] = [h_l, h_r]
        if self.include_mass:
            if mass_l is None or mass_r is None:
                raise RuntimeError("internal error: missing mass components")
            feats.append(mass_l.reshape(1))
            feats.append(mass_r.reshape(1))
        if self.include_endpoints:
            if last_l is None or first_r is None:
                raise RuntimeError("internal error: missing endpoint components")
            feats.append(last_l)
            feats.append(first_r)

        h_parent = self.model.merger(self.torch.cat(feats, dim=0))

        if self.include_mass:
            mass_parent = (mass_l + mass_r).reshape(1)  # type: ignore[operator]
        else:
            mass_parent = None
        if self.include_endpoints:
            # propagate endpoints exactly
            _h, _m, first_l, _last_l2 = self._split_state(left)
            _h2, _m2, _first_r2, last_r = self._split_state(right)
            if first_l is None or last_r is None:
                raise RuntimeError("internal error: missing propagated endpoints")
            out_parts: List["torch.Tensor"] = [h_parent]
            if self.include_mass:
                out_parts.append(mass_parent.reshape(1))  # type: ignore[union-attr]
            out_parts.append(first_l)
            out_parts.append(last_r)
            return self.torch.cat(out_parts, dim=0)

        if self.include_mass:
            return self.torch.cat([h_parent, mass_parent.reshape(1)], dim=0)  # type: ignore[union-attr]
        return h_parent

    def predict(self, state: "torch.Tensor") -> "torch.Tensor":
        h, _mass, _first, _last = self._split_state(state)
        logits = self.model.readout(h)
        if self.output_mode == "simplex":
            return self.torch.softmax(logits, dim=-1)
        return logits

    def idempotence_loss(self, state: "torch.Tensor") -> "torch.Tensor":
        if self.model.idem is None:
            return self.torch.zeros((), dtype=state.dtype, device=state.device)
        state2 = self.model.idem(state)
        pred1 = self.predict(state)
        pred2 = self.predict(state2)
        return self.torch.mean((pred1 - pred2) ** 2)


def _loss_tensor(pred: "torch.Tensor", target: "torch.Tensor", *, mode: OutputMode) -> "torch.Tensor":
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for learned-g experiments. "
            "Install with: uv sync --extra torch"
        ) from e

    if str(mode) == "regression":
        return torch.mean((pred - target) ** 2)
    if str(mode) == "simplex":
        eps = 1e-8
        p = torch.clamp(pred, min=eps)
        return torch.mean(-torch.sum(target * torch.log(p), dim=-1))
    raise ValueError(f"unknown output_mode: {mode!r}")


def _l1_np(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sum(np.abs(diff)))


def train_torch_mergeable_sketch(
    docs_train: Sequence[LearnedGDoc],
    *,
    leaf_feature_dim: int,
    output_dim: int,
    config: LearnedGTrainingConfig,
) -> Tuple[TorchMergeableSketch, LearnedGTrainingGeometry, float]:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for learned-g experiments. "
            "Install with: uv sync --extra torch"
        ) from e

    if len(docs_train) == 0:
        raise ValueError("need at least one training doc")

    torch.manual_seed(int(config.seed))
    if int(config.torch_threads) > 0:
        try:
            torch.set_num_threads(int(config.torch_threads))
            torch.set_num_interop_threads(int(config.torch_threads))
        except Exception:
            pass

    model = TorchMergeableSketch(leaf_feature_dim=int(leaf_feature_dim), output_dim=int(output_dim), config=config)
    opt = torch.optim.Adam(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))

    rng = np.random.default_rng(int(config.seed))

    # Query accounting (realized indices)
    leaf_labels_total = 0
    internal_labels_total = 0
    root_queries_total = 0
    leaves_per_doc: List[float] = []
    internal_per_doc: List[float] = []
    leaf_labels_per_doc: List[float] = []
    internal_labels_per_doc: List[float] = []

    last_loss = float("nan")
    for _ep in range(int(max(1, config.n_epochs))):
        order = rng.permutation(len(docs_train))
        for start in range(0, len(order), int(max(1, config.batch_docs))):
            batch_ids = order[start : start + int(max(1, config.batch_docs))]
            batch_loss = torch.zeros((), dtype=torch.float32)
            for doc_id in batch_ids.tolist():
                doc = docs_train[int(doc_id)]
                x = np.asarray(doc.leaf_features, dtype=np.float32)
                y_leaf = np.asarray(doc.leaf_oracle, dtype=np.float32)
                y_int = np.asarray(doc.internal_oracle_balanced, dtype=np.float32)
                y_root = np.asarray(doc.root_oracle, dtype=np.float32)

                L = int(x.shape[0])
                n_int = int(y_int.shape[0])
                leaves_per_doc.append(float(L))
                internal_per_doc.append(float(n_int))

                # realized supervision
                q_leaf = leaf_sample_count(L, rate=float(config.leaf_query_rate))
                q_int = audit_sample_count(
                    n_int,
                    policy=config.audit_policy,
                    fixed_nodes=int(config.audit_fixed_nodes),
                    fraction=float(config.audit_fraction),
                    scale=float(config.audit_scale),
                )
                leaf_labels_per_doc.append(float(q_leaf))
                internal_labels_per_doc.append(float(q_int))
                leaf_labels_total += int(q_leaf)
                internal_labels_total += int(q_int)

                if bool(config.include_root_query):
                    root_queries_total += 1

                leaf_idx = (
                    np.arange(L, dtype=np.int64)
                    if q_leaf >= L
                    else rng.choice(np.arange(L, dtype=np.int64), size=int(q_leaf), replace=False)
                    if q_leaf > 0
                    else np.asarray([], dtype=np.int64)
                )
                int_idx = (
                    np.arange(n_int, dtype=np.int64)
                    if q_int >= n_int
                    else rng.choice(np.arange(n_int, dtype=np.int64), size=int(q_int), replace=False)
                    if q_int > 0
                    else np.asarray([], dtype=np.int64)
                )

                # leaf forward
                first = None
                last = None
                if model.include_endpoints:
                    if doc.leaf_first_onehot is None or doc.leaf_last_onehot is None:
                        raise ValueError("doc missing endpoint features but model expects endpoints")
                    first = np.asarray(doc.leaf_first_onehot, dtype=np.float32)
                    last = np.asarray(doc.leaf_last_onehot, dtype=np.float32)
                mass = None
                if model.include_mass:
                    if doc.leaf_mass is None:
                        raise ValueError("doc missing mass features but model expects mass")
                    mass = np.asarray(doc.leaf_mass, dtype=np.float32).reshape(-1)

                leaf_states: List[torch.Tensor] = []
                leaf_preds: List[torch.Tensor] = []
                for i in range(L):
                    xi = torch.tensor(x[i], dtype=torch.float32)
                    mi = torch.tensor(float(mass[i]), dtype=torch.float32) if mass is not None else None
                    fi = torch.tensor(first[i], dtype=torch.float32) if first is not None else None
                    li = torch.tensor(last[i], dtype=torch.float32) if last is not None else None
                    st = model.encode_leaf(xi, mass=mi, first_onehot=fi, last_onehot=li)
                    leaf_states.append(st)
                    leaf_preds.append(model.predict(st))

                root_state, merge_states = _merge_schedule(
                    model,
                    leaf_states,
                    schedule="balanced",
                    collect_merge_states=True,
                )
                root_pred = model.predict(root_state)

                # Loss terms
                loss_doc = torch.zeros((), dtype=torch.float32)
                if bool(config.include_root_query) and float(config.root_weight) > 0.0:
                    loss_doc = loss_doc + float(config.root_weight) * _loss_tensor(
                        root_pred, torch.tensor(y_root, dtype=torch.float32), mode=config.output_mode
                    )
                if leaf_idx.size and float(config.leaf_weight) > 0.0:
                    pred_stack = torch.stack([leaf_preds[int(i)] for i in leaf_idx.tolist()], dim=0)
                    tgt = torch.tensor(y_leaf[leaf_idx], dtype=torch.float32)
                    loss_doc = loss_doc + float(config.leaf_weight) * _loss_tensor(pred_stack, tgt, mode=config.output_mode)
                if int_idx.size and float(config.c3_weight) > 0.0:
                    pred_stack = torch.stack([model.predict(merge_states[int(i)]) for i in int_idx.tolist()], dim=0)
                    tgt = torch.tensor(y_int[int_idx], dtype=torch.float32)
                    loss_doc = loss_doc + float(config.c3_weight) * _loss_tensor(pred_stack, tgt, mode=config.output_mode)

                if float(config.schedule_consistency_weight) > 0.0 and L > 1:
                    sched_preds = []
                    for sched in VALID_SCHEDULES:
                        root_s, _m = _merge_schedule(model, leaf_states, schedule=sched, collect_merge_states=False)
                        sched_preds.append(model.predict(root_s))
                    stack = torch.stack(sched_preds, dim=0)
                    loss_doc = loss_doc + float(config.schedule_consistency_weight) * torch.mean(
                        (stack - torch.mean(stack, dim=0, keepdim=True)) ** 2
                    )

                if float(config.idempotence_weight) > 0.0:
                    loss_doc = loss_doc + float(config.idempotence_weight) * model.idempotence_loss(root_state)

                batch_loss = batch_loss + loss_doc

            batch_loss = batch_loss / float(len(batch_ids))
            opt.zero_grad(set_to_none=True)
            if bool(getattr(batch_loss, "requires_grad", False)):
                batch_loss.backward()
                if float(config.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()), float(config.grad_clip_norm))
                opt.step()
            last_loss = float(batch_loss.detach().cpu().item())

    geom = LearnedGTrainingGeometry(
        n_docs=int(len(docs_train)),
        mean_leaves=_safe_mean(leaves_per_doc),
        mean_internal_nodes=_safe_mean(internal_per_doc),
        mean_leaf_labels=_safe_mean(leaf_labels_per_doc),
        mean_internal_labels=_safe_mean(internal_labels_per_doc),
        mean_queries_per_doc=_safe_mean(
            [a + b + (1.0 if bool(config.include_root_query) else 0.0) for a, b in zip(leaf_labels_per_doc, internal_labels_per_doc)]
        ),
        root_queries_total=int(root_queries_total),
        leaf_labels_total=int(leaf_labels_total),
        internal_labels_total=int(internal_labels_total),
        total_labels_total=int(root_queries_total + leaf_labels_total + internal_labels_total),
    )
    return model, geom, float(last_loss)


def eval_torch_mergeable_sketch(
    model: TorchMergeableSketch,
    docs: Sequence[LearnedGDoc],
    *,
    output_mode: OutputMode,
    violation_tau: float,
) -> LearnedGMetrics:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for learned-g experiments. "
            "Install with: uv sync --extra torch"
        ) from e

    tau = float(violation_tau)
    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []

    with torch.no_grad():
        for doc in docs:
            x = np.asarray(doc.leaf_features, dtype=np.float32)
            y_leaf = np.asarray(doc.leaf_oracle, dtype=np.float32)
            y_int = np.asarray(doc.internal_oracle_balanced, dtype=np.float32)
            y_root = np.asarray(doc.root_oracle, dtype=np.float32)
            L = int(x.shape[0])
            if L <= 0:
                continue

            first = None
            last = None
            if model.include_endpoints:
                if doc.leaf_first_onehot is None or doc.leaf_last_onehot is None:
                    raise ValueError("doc missing endpoint features but model expects endpoints")
                first = np.asarray(doc.leaf_first_onehot, dtype=np.float32)
                last = np.asarray(doc.leaf_last_onehot, dtype=np.float32)
            mass = None
            if model.include_mass:
                if doc.leaf_mass is None:
                    raise ValueError("doc missing mass features but model expects mass")
                mass = np.asarray(doc.leaf_mass, dtype=np.float32).reshape(-1)

            leaf_states: List[torch.Tensor] = []
            leaf_preds_np: List[np.ndarray] = []
            for i in range(L):
                xi = torch.tensor(x[i], dtype=torch.float32)
                mi = torch.tensor(float(mass[i]), dtype=torch.float32) if mass is not None else None
                fi = torch.tensor(first[i], dtype=torch.float32) if first is not None else None
                li = torch.tensor(last[i], dtype=torch.float32) if last is not None else None
                st = model.encode_leaf(xi, mass=mi, first_onehot=fi, last_onehot=li)
                leaf_states.append(st)
                leaf_preds_np.append(model.predict(st).detach().cpu().numpy().astype(np.float64))

            # balanced forward for leaf/internal metrics
            root_state, merge_states = _merge_schedule(model, leaf_states, schedule="balanced", collect_merge_states=True)
            root_pred = model.predict(root_state).detach().cpu().numpy().astype(np.float64)
            root_abs.append(_l1_np(root_pred, y_root))

            for i in range(L):
                leaf_abs.append(_l1_np(leaf_preds_np[i], y_leaf[i]))

            for i, st in enumerate(merge_states):
                pred_i = model.predict(st).detach().cpu().numpy().astype(np.float64)
                merge_abs.append(_l1_np(pred_i, y_int[i]))

            # schedule spread
            root_preds = []
            for sched in VALID_SCHEDULES:
                r_state, _m = _merge_schedule(model, leaf_states, schedule=sched, collect_merge_states=False)
                root_preds.append(model.predict(r_state).detach().cpu().numpy().astype(np.float64))
            spreads.append(schedule_spread_l1(root_preds))

    root_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    leaf_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_arr = np.asarray(merge_abs, dtype=np.float64)

    return LearnedGMetrics(
        root_mae=float(np.mean(root_arr)) if root_arr.size else 0.0,
        root_median_abs_error=_median(root_arr.tolist()),
        root_p95_abs_error=_p95(root_arr.tolist()),
        schedule_spread_mean=float(np.mean(spreads_arr)) if spreads_arr.size else 0.0,
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)) if spreads_arr.size else 0.0,
        leaf_mae=float(np.mean(leaf_arr)) if leaf_arr.size else 0.0,
        leaf_violation_rate=float(np.mean((leaf_arr > tau).astype(np.float64))) if leaf_arr.size else 0.0,
        merge_mae=float(np.mean(merge_arr)) if merge_arr.size else 0.0,
        merge_violation_rate=float(np.mean((merge_arr > tau).astype(np.float64))) if merge_arr.size else 0.0,
        n_docs=int(len(root_abs)),
    )


def build_docs_from_oracle(
    *,
    leaf_features: Sequence[np.ndarray],
    leaf_spans: Sequence[Sequence[Span]],
    oracle_fn: Callable[[int, Span], np.ndarray],
    output_dim: int,
    endpoint_categories: int = 0,
    endpoints_fn: Optional[Callable[[int, Span], Tuple[int, int]]] = None,
    leaf_mass: Optional[Sequence[np.ndarray]] = None,
) -> List[LearnedGDoc]:
    """
    Utility to precompute (leaf, internal, root) oracle labels for a batch of docs.

    Args:
      leaf_features: list of per-doc [L,D] arrays.
      leaf_spans: list of per-doc token spans aligned with rows of leaf_features.
      oracle_fn: callable (doc_index, span) -> oracle vector [O].
      endpoints_fn: callable (doc_index, span) -> (first_cat,last_cat) ints, if endpoint one-hots desired.
      leaf_mass: optional list of per-doc [L] token masses.
    """

    out: List[LearnedGDoc] = []
    for doc_id, (x, spans) in enumerate(zip(leaf_features, leaf_spans)):
        x_mat = np.asarray(x, dtype=np.float32)
        spans_list = [tuple(map(int, sp)) for sp in spans]
        L = int(x_mat.shape[0])
        if L != len(spans_list):
            raise ValueError("leaf_features and leaf_spans must align per doc")

        leaf_y = np.zeros((L, int(output_dim)), dtype=np.float32)
        for i, sp in enumerate(spans_list):
            y = np.asarray(oracle_fn(int(doc_id), sp), dtype=np.float32).reshape(-1)
            if y.size != int(output_dim):
                raise ValueError("oracle_fn returned wrong output_dim")
            leaf_y[i] = y

        internal_spans = balanced_internal_spans(spans_list)
        internal_y = np.zeros((len(internal_spans), int(output_dim)), dtype=np.float32)
        for i, sp in enumerate(internal_spans):
            y = np.asarray(oracle_fn(int(doc_id), sp), dtype=np.float32).reshape(-1)
            if y.size != int(output_dim):
                raise ValueError("oracle_fn returned wrong output_dim")
            internal_y[i] = y

        root_span = (int(spans_list[0][0]), int(spans_list[-1][1])) if spans_list else (0, 0)
        root_y = np.asarray(oracle_fn(int(doc_id), root_span), dtype=np.float32).reshape(-1)
        if root_y.size != int(output_dim):
            raise ValueError("oracle_fn returned wrong output_dim")

        first_oh = None
        last_oh = None
        if endpoint_categories > 0:
            if endpoints_fn is None:
                raise ValueError("endpoints_fn required when endpoint_categories>0")
            C = int(endpoint_categories)
            first_oh = np.zeros((L, C), dtype=np.float32)
            last_oh = np.zeros((L, C), dtype=np.float32)
            for i, sp in enumerate(spans_list):
                a, b = endpoints_fn(int(doc_id), sp)
                if not (0 <= int(a) < C and 0 <= int(b) < C):
                    raise ValueError("endpoint category out of range")
                first_oh[i, int(a)] = 1.0
                last_oh[i, int(b)] = 1.0

        mass_arr = None
        if leaf_mass is not None:
            mass_arr = np.asarray(leaf_mass[int(doc_id)], dtype=np.float32).reshape(-1)
            if mass_arr.shape[0] != L:
                raise ValueError("leaf_mass must align with leaves")

        out.append(
            LearnedGDoc(
                leaf_features=x_mat,
                leaf_oracle=leaf_y,
                internal_oracle_balanced=internal_y,
                root_oracle=root_y,
                leaf_first_onehot=first_oh,
                leaf_last_onehot=last_oh,
                leaf_mass=mass_arr,
            )
        )
    return out
