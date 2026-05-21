"""
Stage-1 tree-relevant LDA family: exact and compressed recovery of mergeable utility sketches.

The ordinary bag-of-words LDA document model is unchanged:

    pi_d ~ Dir(alpha)
    z_{d,t} | pi_d ~ Cat(pi_d)
    x_{d,t} | z_{d,t}=k ~ Cat(phi_k)

What changes is the target. The primary object is a **document-level scalar objective**

    Y_d = r^T U c_d

for a fixed readout vector `r`, utility matrix `U`, and full-document histogram `c_d`.

To make exact subset supervision possible, the family introduces the mergeable intermediate sketch

    u(A) = U c_A

where `c_A` is the bag-of-words histogram on span `A` and `U` is a fixed utility matrix.

So the tree is learning or compressing a mergeable representation of a target that still
belongs to the full document.

This is the clean positive-control ladder for tree methods:

1. `full_doc_exact_utility`: compute the full-document sketch and scalar objective directly.
2. `tree_exact_utility`: compute `U c_leaf` on leaves and merge by addition.
3. `count_svd_ceiling`: compress counts additively, then read out the document objective.
4. `utility_pca_practical`: compress utility sketches directly and merge by addition.
5. `full_doc_mlp_diag`: appendix-only learned document operator on full-document counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is optional for appendix diagnostics.
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None

from treepo._research.ctreepo.sim.core import lda_tree_recovery as _base
from treepo._research.ctreepo.sim.core.segment_lda_ops_weight_recovery import (
    _splitmix64,
    build_leaf_spans,
    sample_topic_distributions,
)
from treepo._research.ctreepo.sim.core.training_selection import (
    TrainingSelectionMetadata,
    clone_module_state,
    improved_metric,
    restore_module_state,
)
from treepo._research.ctreepo.sim.objective_semantics import mergeable_target_objective_semantics


VALID_EMISSION_MODES: Tuple[str, ...] = _base.VALID_EMISSION_MODES
VALID_SCHEDULES: Tuple[str, ...] = _base.VALID_SCHEDULES
VALID_UTILITY_DESIGNS: Tuple[str, ...] = ("topic_anchored_sparse",)
VALID_DEVICE_MODES: Tuple[str, ...] = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class LDATreeUtilityVectorConfig:
    n_topics: int = 8
    vocab_size: int = 512
    doc_tokens: int = 384
    doc_topic_concentration: float = 0.6

    topic_concentration: float = 0.2
    emission_mode: str = "anchored"
    anchor_words_per_topic: int = 20
    anchor_multiplier: float = 25.0

    utility_dim: int = 16
    utility_design: str = "topic_anchored_sparse"
    leaf_fraction: float = 1.0 / 24.0

    train_docs: int = 512
    test_docs: int = 256

    state_dim: int = 64

    run_full_doc_mlp_diag: bool = True
    full_hidden_dim: int = 128
    full_n_layers: int = 2
    n_epochs: int = 40
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-5
    device: str = "auto"
    cuda_device: Optional[int] = None
    torch_threads: int = 0

    seed: int = 0


@dataclass(frozen=True)
class LDAUtilityVectorWorld:
    signature: Dict[str, object]
    topic_meta: Dict[str, object]
    topics_phi: Tuple[np.ndarray, ...]
    utility_matrix: np.ndarray
    utility_topic_meta: Dict[str, object]
    readout_vector: np.ndarray
    docs_train: Tuple[_base.LDATreeRecoveryDoc, ...]
    docs_test: Tuple[_base.LDATreeRecoveryDoc, ...]


@dataclass(frozen=True)
class LDAUtilityVectorSummary:
    family: str
    target_kind: str
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    utility_topic_meta: Dict[str, object]
    world_stats: Dict[str, object]
    exact_recovery: Dict[str, object]
    methods: Dict[str, object]
    is_stale_generation: bool = False
    objective: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "family": self.family,
                "target_kind": self.target_kind,
                "config": self.config,
                "topic_meta": self.topic_meta,
                "utility_topic_meta": self.utility_topic_meta,
                "world_stats": self.world_stats,
                "exact_recovery": self.exact_recovery,
                "methods": self.methods,
                "is_stale_generation": bool(self.is_stale_generation),
                "objective": self.objective,
            },
            indent=2,
            sort_keys=True,
        )


@dataclass(frozen=True)
class _ProjectionSketch:
    components: np.ndarray  # [state_dim, dim]

    @property
    def state_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def dim(self) -> int:
        return int(self.components.shape[1])

    def encode(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape(self.dim)
        return self.components @ arr

    def decode(self, s: np.ndarray) -> np.ndarray:
        state = np.asarray(s, dtype=np.float64).reshape(self.state_dim)
        return self.components.T @ state


def leaf_tokens_from_fraction(doc_tokens: int, leaf_fraction: float) -> int:
    n_doc = int(doc_tokens)
    frac = float(leaf_fraction)
    if n_doc <= 0:
        raise ValueError("doc_tokens must be positive")
    if not (0.0 < frac <= 1.0):
        raise ValueError("leaf_fraction must be in (0, 1]")
    leaf = int(round(float(n_doc) * frac))
    return max(1, min(n_doc, leaf))


def leaf_fraction_label(leaf_fraction: float) -> str:
    pct = 100.0 * float(leaf_fraction)
    if abs(pct - round(pct)) <= 1e-9:
        return f"{int(round(pct))}%"
    return f"{pct:.2f}%"


def _leaf_metadata(config: LDATreeUtilityVectorConfig) -> Dict[str, object]:
    leaf_tokens = leaf_tokens_from_fraction(int(config.doc_tokens), float(config.leaf_fraction))
    return {
        "doc_tokens": int(config.doc_tokens),
        "leaf_fraction": float(config.leaf_fraction),
        "leaf_percent_of_doc": float(100.0 * float(config.leaf_fraction)),
        "leaf_fraction_label": leaf_fraction_label(float(config.leaf_fraction)),
        "leaf_tokens": int(leaf_tokens),
    }


def _validate_config(config: LDATreeUtilityVectorConfig) -> None:
    base_cfg = _base.LDATreeRecoveryConfig(
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        min_tokens=int(config.doc_tokens),
        max_tokens=int(config.doc_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        topic_concentration=float(config.topic_concentration),
        emission_mode=str(config.emission_mode),
        anchor_words_per_topic=int(config.anchor_words_per_topic),
        anchor_multiplier=float(config.anchor_multiplier),
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=False,
        lambda_multiplier=0.0,
        leaf_tokens=leaf_tokens_from_fraction(int(config.doc_tokens), float(config.leaf_fraction)),
        train_docs=int(config.train_docs),
        test_docs=int(config.test_docs),
        inference_prior_mass=0.25,
        inference_max_iter=20,
        inference_tol=1e-9,
        seed=int(config.seed),
    )
    _base._validate_config(base_cfg)
    if int(config.utility_dim) <= 0:
        raise ValueError("utility_dim must be positive")
    if str(config.utility_design).strip().lower() not in VALID_UTILITY_DESIGNS:
        raise ValueError(f"utility_design must be one of {VALID_UTILITY_DESIGNS}")
    if int(config.train_docs) <= 0:
        raise ValueError("train_docs must be positive")
    if int(config.test_docs) <= 0:
        raise ValueError("test_docs must be positive")
    if int(config.state_dim) <= 0:
        raise ValueError("state_dim must be positive")
    if bool(config.run_full_doc_mlp_diag):
        if int(config.full_hidden_dim) <= 0:
            raise ValueError("full_hidden_dim must be positive")
        if int(config.full_n_layers) <= 0:
            raise ValueError("full_n_layers must be positive")
        if int(config.n_epochs) <= 0:
            raise ValueError("n_epochs must be positive")
        if int(config.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if float(config.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(config.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
    if str(config.device).strip().lower() not in VALID_DEVICE_MODES:
        raise ValueError(f"device must be one of {VALID_DEVICE_MODES}")


def _world_signature(config: LDATreeUtilityVectorConfig) -> Dict[str, object]:
    return {
        "family": "lda_tree_utility_vector",
        "n_topics": int(config.n_topics),
        "vocab_size": int(config.vocab_size),
        "doc_tokens": int(config.doc_tokens),
        "doc_topic_concentration": float(config.doc_topic_concentration),
        "topic_concentration": float(config.topic_concentration),
        "emission_mode": str(config.emission_mode),
        "anchor_words_per_topic": int(config.anchor_words_per_topic),
        "anchor_multiplier": float(config.anchor_multiplier),
        "utility_dim": int(config.utility_dim),
        "utility_design": str(config.utility_design),
        "seed": int(config.seed),
    }


def lda_tree_utility_vector_world_cache_signature(
    config: LDATreeUtilityVectorConfig,
    *,
    train_docs_capacity: int,
    test_docs_capacity: int,
) -> Dict[str, object]:
    return {
        **_world_signature(config),
        "train_docs_capacity": int(train_docs_capacity),
        "test_docs_capacity": int(test_docs_capacity),
    }


def _anchor_words_for_topic(
    topic_id: int,
    *,
    topics_phi: Sequence[np.ndarray],
    topic_meta: Dict[str, object],
    take: int,
) -> List[int]:
    anchors = topic_meta.get("anchors")
    if isinstance(anchors, list) and 0 <= int(topic_id) < len(anchors):
        items = anchors[int(topic_id)]
        if isinstance(items, list) and items:
            return [int(x) for x in items[: max(1, int(take))]]
    probs = np.asarray(topics_phi[int(topic_id)], dtype=np.float64)
    top_idx = np.argsort(probs)[::-1][: max(1, int(take))]
    return [int(x) for x in top_idx.tolist()]


def sample_topic_anchored_utility_matrix(
    *,
    topics_phi: Sequence[np.ndarray],
    topic_meta: Dict[str, object],
    utility_dim: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    phi = np.stack([np.asarray(x, dtype=np.float64) for x in topics_phi], axis=0)
    n_topics = int(phi.shape[0])
    vocab_size = int(phi.shape[1])
    m = int(utility_dim)
    rows = np.zeros((m, vocab_size), dtype=np.float64)
    row_meta: List[Dict[str, object]] = []
    anchor_take = max(1, min(6, int(topic_meta.get("anchor_words_per_topic", 6))))

    for row_idx in range(m):
        pos_topics = [int(row_idx % n_topics)]
        if n_topics >= 2 and row_idx % 3 == 1:
            pos_topics.append(int((row_idx + 1) % n_topics))
        pos_topics = sorted(set(pos_topics))
        neg_topic = int((row_idx + 2) % n_topics) if n_topics >= 3 and row_idx % 4 == 0 else None

        w = np.zeros((vocab_size,), dtype=np.float64)
        pos_words: List[int] = []
        neg_words: List[int] = []
        for topic_id in pos_topics:
            words = _anchor_words_for_topic(
                topic_id,
                topics_phi=topics_phi,
                topic_meta=topic_meta,
                take=anchor_take,
            )
            pos_words.extend(words)
        pos_words = sorted(set(pos_words))
        if pos_words:
            w[pos_words] += 1.0 / float(len(pos_words))
        if neg_topic is not None:
            neg_words = _anchor_words_for_topic(
                neg_topic,
                topics_phi=topics_phi,
                topic_meta=topic_meta,
                take=anchor_take,
            )
            neg_words = sorted(set(neg_words))
            if neg_words:
                w[neg_words] -= 0.75 / float(len(neg_words))
        jitter = rng.normal(loc=0.0, scale=1e-3, size=(vocab_size,)).astype(np.float64, copy=False)
        w = w + jitter
        norm = float(np.linalg.norm(w))
        if not math.isfinite(norm) or norm <= 0.0:
            w = np.zeros((vocab_size,), dtype=np.float64)
            w[int(row_idx % vocab_size)] = 1.0
        else:
            w = w / norm
        rows[row_idx, :] = w
        row_meta.append(
            {
                "row_index": int(row_idx),
                "positive_topics": list(int(x) for x in pos_topics),
                "negative_topic": None if neg_topic is None else int(neg_topic),
                "positive_words": list(int(x) for x in pos_words),
                "negative_words": list(int(x) for x in neg_words),
                "topic_coefficients": [float(x) for x in (phi @ w).tolist()],
            }
        )

    return rows, {
        "utility_design": "topic_anchored_sparse",
        "utility_dim": int(m),
        "rows": row_meta,
    }


def utility_vector_from_counts(counts: np.ndarray, utility_matrix: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    u = np.asarray(utility_matrix, dtype=np.float64)
    return u @ c


def _balanced_node_vectors(leaf_vectors: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    nodes: List[np.ndarray] = [np.asarray(x, dtype=np.float64).copy() for x in leaf_vectors]
    cur = [np.asarray(x, dtype=np.float64).copy() for x in leaf_vectors]
    while len(cur) > 1:
        nxt: List[np.ndarray] = []
        i = 0
        while i < len(cur):
            if i + 1 < len(cur):
                merged = cur[i] + cur[i + 1]
                nodes.append(merged)
                nxt.append(merged)
                i += 2
            else:
                nxt.append(cur[i])
                i += 1
        cur = nxt
    return tuple(nodes)


def _right_singular_system(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected a rank-2 matrix")
    gram = arr.T @ arr
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.clip(np.asarray(evals[order], dtype=np.float64), 0.0, None)
    s = np.sqrt(evals, dtype=np.float64)
    vt = np.asarray(evecs[:, order].T, dtype=np.float64)
    return s, vt


def fit_projection_sketch(
    examples: np.ndarray,
    *,
    state_dim: int,
    input_dim: int,
) -> Tuple[_ProjectionSketch, Dict[str, object]]:
    x = np.asarray(examples, dtype=np.float64)
    if x.ndim != 2 or int(x.shape[1]) != int(input_dim):
        raise ValueError("examples must have shape [n_examples, input_dim]")
    if int(x.shape[0]) <= 0:
        raise ValueError("need at least one example to fit a sketch")
    m = int(state_dim)
    d = int(input_dim)

    if m >= d:
        comps = np.zeros((m, d), dtype=np.float64)
        comps[:d, :] = np.eye(d, dtype=np.float64)
        return _ProjectionSketch(comps), {
            "fit_mode": "identity_ceiling",
            "state_dim": int(m),
            "train_rank": int(np.linalg.matrix_rank(x)),
            "kept_components": int(d),
            "explained_variance_ratio_train": 1.0,
            "exact_family_representable": True,
            "exact_train_manifold_representable": True,
        }

    s, vt = _right_singular_system(x)
    train_rank = int(np.sum(s > 1e-10))
    kept = min(int(m), int(vt.shape[0]))
    comps = np.zeros((m, d), dtype=np.float64)
    if kept > 0:
        comps[:kept, :] = vt[:kept, :]
    total_var = float(np.sum(s ** 2))
    kept_var = float(np.sum(s[:kept] ** 2)) if kept > 0 else 0.0
    return _ProjectionSketch(comps), {
        "fit_mode": "pca_projection",
        "state_dim": int(m),
        "train_rank": int(train_rank),
        "kept_components": int(kept),
        "explained_variance_ratio_train": (kept_var / total_var) if total_var > 0.0 else 1.0,
        "exact_family_representable": bool(int(m) >= int(d)),
        "exact_train_manifold_representable": bool(int(m) >= int(train_rank)),
    }


def _set_global_seed(seed: int, *, torch_threads: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is None:  # pragma: no cover
        return
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


def _resolve_device(config: LDATreeUtilityVectorConfig):
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    mode = str(config.device).strip().lower()
    if mode == "auto":
        mode = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if config.cuda_device is not None:
            idx = int(config.cuda_device)
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}")
        return torch.device("cuda")
    return torch.device("cpu")


def _full_doc_features_from_counts(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    total = float(np.sum(c))
    if total > 0.0:
        freqs = c / total
    else:
        freqs = np.zeros_like(c, dtype=np.float64)
    return np.concatenate([freqs, np.asarray([math.log1p(total)], dtype=np.float64)], axis=0)


class _FullDocUtilityMLP(nn.Module):  # pragma: no cover - exercised by smoke runs, not unit tests.
    def __init__(self, *, input_dim: int, hidden_dim: int, n_layers: int, output_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # pragma: no cover
        return self.net(x)


def _train_full_doc_utility_mlp(
    train_counts: np.ndarray,
    train_targets: np.ndarray,
    val_counts: np.ndarray,
    val_targets: np.ndarray,
    *,
    config: LDATreeUtilityVectorConfig,
) -> Tuple[Optional[_FullDocUtilityMLP], Dict[str, object]]:
    if torch is None:  # pragma: no cover
        return None, {"fit_mode": "skipped_no_torch", "available": False}
    _set_global_seed(int(config.seed), torch_threads=int(config.torch_threads))
    device = _resolve_device(config)
    model = _FullDocUtilityMLP(
        input_dim=int(train_counts.shape[1]),
        hidden_dim=int(config.full_hidden_dim),
        n_layers=int(config.full_n_layers),
        output_dim=int(train_targets.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    train_ds = TensorDataset(
        torch.tensor(train_counts, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=True,
        drop_last=False,
    )
    train_loss_last = float("nan")
    val_loss_last = float("nan")
    best_state = None
    best_selection = TrainingSelectionMetadata(
        mode="best_val_loss" if int(val_counts.shape[0]) > 0 else "final_epoch_no_validation",
        split="val" if int(val_counts.shape[0]) > 0 else "config",
        metric_name="val_loss_final" if int(val_counts.shape[0]) > 0 else "train_loss_final",
        metric_value=float("nan"),
        best_epoch=0,
    )
    for _ in range(int(config.n_epochs)):
        epoch_idx = int(_)
        model.train()
        losses: List[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss_last = _base._safe_stat(losses, kind="mean")
        if int(val_counts.shape[0]) > 0:
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(val_counts, dtype=torch.float32, device=device)
                yv = torch.tensor(val_targets, dtype=torch.float32, device=device)
                val_loss_last = float(F.mse_loss(model(xv), yv).detach().cpu())
            if improved_metric(val_loss_last, best_selection.metric_value):
                best_selection = TrainingSelectionMetadata(
                    mode="best_val_loss",
                    split="val",
                    metric_name="val_loss_final",
                    metric_value=float(val_loss_last),
                    best_epoch=int(epoch_idx),
                )
                best_state = clone_module_state(model)
    if int(val_counts.shape[0]) > 0:
        restore_module_state(model, best_state)
    else:
        best_selection = TrainingSelectionMetadata(
            mode="final_epoch_no_validation",
            split="config",
            metric_name="train_loss_final",
            metric_value=float(train_loss_last),
            best_epoch=max(0, int(config.n_epochs) - 1),
        )
    model.eval()
    return model, {
        "fit_mode": "sgd_mlp",
        "train_loss_final": float(train_loss_last),
        "val_loss_final": float(val_loss_last),
        **best_selection.to_dict(),
        "device": str(device),
        "full_hidden_dim": int(config.full_hidden_dim),
        "full_n_layers": int(config.full_n_layers),
        "n_epochs": int(config.n_epochs),
        "batch_size": int(config.batch_size),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "available": True,
    }


def sample_lda_tree_utility_vector_world(
    config: LDATreeUtilityVectorConfig,
    *,
    train_docs_capacity: Optional[int] = None,
    test_docs_capacity: Optional[int] = None,
) -> LDAUtilityVectorWorld:
    _validate_config(config)
    train_cap = int(config.train_docs if train_docs_capacity is None else train_docs_capacity)
    test_cap = int(config.test_docs if test_docs_capacity is None else test_docs_capacity)
    if train_cap <= 0 or test_cap <= 0:
        raise ValueError("train_docs_capacity and test_docs_capacity must be positive")

    topics_phi, topic_meta = sample_topic_distributions(
        vocab_size=int(config.vocab_size),
        n_topics=int(config.n_topics),
        topic_concentration=float(config.topic_concentration),
        emission_mode=str(config.emission_mode),
        anchor_words_per_topic=int(config.anchor_words_per_topic),
        anchor_multiplier=float(config.anchor_multiplier),
        seed=int(_splitmix64(int(config.seed) + 101) & 0xFFFFFFFF),
    )
    utility_matrix, utility_topic_meta = sample_topic_anchored_utility_matrix(
        topics_phi=topics_phi,
        topic_meta=topic_meta,
        utility_dim=int(config.utility_dim),
        seed=int(_splitmix64(int(config.seed) + 303) & 0xFFFFFFFF),
    )
    readout_rng = np.random.default_rng(int(_splitmix64(int(config.seed) + 404) & 0xFFFFFFFF))
    readout = readout_rng.normal(loc=0.0, scale=1.0, size=(int(config.utility_dim),)).astype(np.float64, copy=False)
    norm = float(np.linalg.norm(readout))
    if norm > 0.0:
        readout = readout / norm

    docs_train, _ = _base._generate_bag_of_words_docs(
        train_cap,
        topics_phi=topics_phi,
        min_tokens=int(config.doc_tokens),
        max_tokens=int(config.doc_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        seed=int(_splitmix64(int(config.seed) + 7) & 0xFFFFFFFF),
    )
    docs_test, _ = _base._generate_bag_of_words_docs(
        test_cap,
        topics_phi=topics_phi,
        min_tokens=int(config.doc_tokens),
        max_tokens=int(config.doc_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        seed=int(_splitmix64(int(config.seed) + 11) & 0xFFFFFFFF),
    )

    utility_topic_meta = dict(utility_topic_meta)
    utility_topic_meta["readout_vector"] = [float(x) for x in readout.tolist()]
    return LDAUtilityVectorWorld(
        signature=_world_signature(config),
        topic_meta=dict(topic_meta),
        topics_phi=tuple(np.asarray(t, dtype=np.float64).copy() for t in topics_phi),
        utility_matrix=np.asarray(utility_matrix, dtype=np.float64).copy(),
        utility_topic_meta=utility_topic_meta,
        readout_vector=np.asarray(readout, dtype=np.float64).copy(),
        docs_train=tuple(docs_train),
        docs_test=tuple(docs_test),
    )


def _summarize_docs(
    docs: Sequence[_base.LDATreeRecoveryDoc],
    *,
    leaf_tokens: int,
) -> Dict[str, float]:
    return _base._summarize_docs(docs, leaf_tokens=int(leaf_tokens))


def _reduce_vectors(vectors: Sequence[np.ndarray], *, schedule: str, dim: int) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((int(dim),), dtype=np.float64)
    return _base._reduce_counts(vectors, schedule=schedule, vocab_size=int(dim))


def _method_metric_dict(
    *,
    supervision_kind: str,
    n_docs: int,
    utility_l1_to_full: Sequence[float],
    utility_l2_to_full: Sequence[float],
    scalar_abs_to_full: Sequence[float],
    schedule_utility_spread: Sequence[float],
    schedule_scalar_spread: Sequence[float],
    count_l1_to_full: Optional[Sequence[float]] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    out = {
        "supervision_kind": str(supervision_kind),
        "n_docs": int(n_docs),
        "utility_l1_to_full_mean": _base._safe_stat(utility_l1_to_full, kind="mean"),
        "utility_l1_to_full_median": _base._safe_stat(utility_l1_to_full, kind="median"),
        "utility_l1_to_full_p95": _base._safe_stat(utility_l1_to_full, kind="p95"),
        "utility_l2_to_full_mean": _base._safe_stat(utility_l2_to_full, kind="mean"),
        "utility_l2_to_full_median": _base._safe_stat(utility_l2_to_full, kind="median"),
        "utility_l2_to_full_p95": _base._safe_stat(utility_l2_to_full, kind="p95"),
        "scalar_abs_to_full_mean": _base._safe_stat(scalar_abs_to_full, kind="mean"),
        "scalar_abs_to_full_median": _base._safe_stat(scalar_abs_to_full, kind="median"),
        "scalar_abs_to_full_p95": _base._safe_stat(scalar_abs_to_full, kind="p95"),
        "schedule_utility_l1_spread_mean": _base._safe_stat(schedule_utility_spread, kind="mean"),
        "schedule_utility_l1_spread_p95": _base._safe_stat(schedule_utility_spread, kind="p95"),
        "schedule_scalar_abs_spread_mean": _base._safe_stat(schedule_scalar_spread, kind="mean"),
        "schedule_scalar_abs_spread_p95": _base._safe_stat(schedule_scalar_spread, kind="p95"),
    }
    if count_l1_to_full is not None:
        out["count_l1_to_full_mean"] = _base._safe_stat(count_l1_to_full, kind="mean")
        out["count_l1_to_full_median"] = _base._safe_stat(count_l1_to_full, kind="median")
        out["count_l1_to_full_p95"] = _base._safe_stat(count_l1_to_full, kind="p95")
    if diagnostics:
        out.update({str(k): v for k, v in diagnostics.items()})
    return out


def run_lda_tree_utility_vector_experiment_from_world(
    config: LDATreeUtilityVectorConfig,
    world: LDAUtilityVectorWorld,
) -> LDAUtilityVectorSummary:
    _validate_config(config)
    if dict(world.signature) != _world_signature(config):
        raise ValueError("config is incompatible with the provided fixed world")
    if int(config.train_docs) > len(world.docs_train):
        raise ValueError("config.train_docs exceeds fixed world train_docs capacity")
    if int(config.test_docs) > len(world.docs_test):
        raise ValueError("config.test_docs exceeds fixed world test_docs capacity")

    leaf_meta = _leaf_metadata(config)
    leaf_tokens = int(leaf_meta["leaf_tokens"])
    utility_matrix = np.asarray(world.utility_matrix, dtype=np.float64)
    readout = np.asarray(world.readout_vector, dtype=np.float64)
    docs_train = tuple(world.docs_train[: int(config.train_docs)])
    docs_test = tuple(world.docs_test[: int(config.test_docs)])

    train_full_counts: List[np.ndarray] = []
    train_full_utility: List[np.ndarray] = []
    train_node_counts: List[np.ndarray] = []
    train_node_utility: List[np.ndarray] = []

    for doc in docs_train:
        counts_full = _base._counts_from_tokens(doc.tokens, vocab_size=int(config.vocab_size))
        train_full_counts.append(counts_full)
        train_full_utility.append(utility_vector_from_counts(counts_full, utility_matrix))
        leaves = build_leaf_spans(len(doc.tokens), leaf_tokens=leaf_tokens)
        leaf_counts = [
            _base._counts_from_tokens(doc.tokens[int(start) : int(end)], vocab_size=int(config.vocab_size))
            for start, end in leaves
        ]
        train_node_counts.extend(_balanced_node_vectors(leaf_counts))
        train_node_utility.extend(
            _balanced_node_vectors([utility_vector_from_counts(c, utility_matrix) for c in leaf_counts])
        )

    count_sketch, count_diag = fit_projection_sketch(
        np.stack(train_node_counts, axis=0),
        state_dim=int(config.state_dim),
        input_dim=int(config.vocab_size),
    )
    utility_sketch, utility_diag = fit_projection_sketch(
        np.stack(train_node_utility, axis=0),
        state_dim=int(config.state_dim),
        input_dim=int(config.utility_dim),
    )

    x_train = np.stack([_full_doc_features_from_counts(c) for c in train_full_counts], axis=0)
    y_train = np.stack(train_full_utility, axis=0)
    n_val = min(max(1, int(len(x_train) // 5)), max(1, int(len(x_train) - 1)))
    x_fit = x_train[:-n_val] if int(len(x_train)) > 1 else x_train
    y_fit = y_train[:-n_val] if int(len(y_train)) > 1 else y_train
    x_val = x_train[-n_val:] if int(len(x_train)) > 1 else np.zeros((0, x_train.shape[1]), dtype=np.float64)
    y_val = y_train[-n_val:] if int(len(y_train)) > 1 else np.zeros((0, y_train.shape[1]), dtype=np.float64)
    if bool(config.run_full_doc_mlp_diag):
        mlp_model, mlp_diag = _train_full_doc_utility_mlp(
            x_fit.astype(np.float64, copy=False),
            y_fit.astype(np.float64, copy=False),
            x_val.astype(np.float64, copy=False),
            y_val.astype(np.float64, copy=False),
            config=config,
        )
    else:
        mlp_model, mlp_diag = None, {
            "fit_mode": "disabled",
            "available": False,
        }

    exact_utility_l1: List[float] = []
    exact_utility_l2: List[float] = []
    exact_scalar_abs: List[float] = []
    exact_sched_util_spread: List[float] = []
    exact_sched_scalar_spread: List[float] = []

    tree_utility_l1: List[float] = []
    tree_utility_l2: List[float] = []
    tree_scalar_abs: List[float] = []
    tree_sched_util_spread: List[float] = []
    tree_sched_scalar_spread: List[float] = []

    count_utility_l1: List[float] = []
    count_utility_l2: List[float] = []
    count_scalar_abs: List[float] = []
    count_sched_util_spread: List[float] = []
    count_sched_scalar_spread: List[float] = []
    count_l1_to_full: List[float] = []

    util_utility_l1: List[float] = []
    util_utility_l2: List[float] = []
    util_scalar_abs: List[float] = []
    util_sched_util_spread: List[float] = []
    util_sched_scalar_spread: List[float] = []

    mlp_utility_l1: List[float] = []
    mlp_utility_l2: List[float] = []
    mlp_scalar_abs: List[float] = []

    mlp_device = None
    if mlp_model is not None and torch is not None:
        mlp_device = next(mlp_model.parameters()).device

    for doc in docs_test:
        counts_full = _base._counts_from_tokens(doc.tokens, vocab_size=int(config.vocab_size))
        utility_full = utility_vector_from_counts(counts_full, utility_matrix)
        scalar_full = float(readout @ utility_full)

        leaves = build_leaf_spans(len(doc.tokens), leaf_tokens=leaf_tokens)
        leaf_counts = [
            _base._counts_from_tokens(doc.tokens[int(start) : int(end)], vocab_size=int(config.vocab_size))
            for start, end in leaves
        ]
        leaf_utility = [utility_vector_from_counts(c, utility_matrix) for c in leaf_counts]

        util_by_schedule: Dict[str, np.ndarray] = {}
        scalar_by_schedule: Dict[str, float] = {}
        for schedule in VALID_SCHEDULES:
            root_u = _reduce_vectors(leaf_utility, schedule=schedule, dim=int(config.utility_dim))
            util_by_schedule[str(schedule)] = root_u
            scalar_by_schedule[str(schedule)] = float(readout @ root_u)
        tree_u = util_by_schedule["balanced"]
        tree_s = scalar_by_schedule["balanced"]
        exact_utility_l1.append(float(np.sum(np.abs(tree_u - utility_full))))
        exact_utility_l2.append(float(np.linalg.norm(tree_u - utility_full)))
        exact_scalar_abs.append(abs(float(tree_s) - float(scalar_full)))
        exact_sched_util_spread.append(_base._pairwise_spread(list(util_by_schedule.values()), metric="l1"))
        exact_sched_scalar_spread.append(_base._pairwise_spread(list(scalar_by_schedule.values()), metric="abs"))
        tree_utility_l1.append(float(np.sum(np.abs(tree_u - utility_full))))
        tree_utility_l2.append(float(np.linalg.norm(tree_u - utility_full)))
        tree_scalar_abs.append(abs(float(tree_s) - float(scalar_full)))
        tree_sched_util_spread.append(_base._pairwise_spread(list(util_by_schedule.values()), metric="l1"))
        tree_sched_scalar_spread.append(_base._pairwise_spread(list(scalar_by_schedule.values()), metric="abs"))

        count_util_by_schedule: Dict[str, np.ndarray] = {}
        count_scalar_by_schedule: Dict[str, float] = {}
        count_pred_by_schedule: Dict[str, np.ndarray] = {}
        count_states = [count_sketch.encode(c) for c in leaf_counts]
        for schedule in VALID_SCHEDULES:
            root_state = _reduce_vectors(count_states, schedule=schedule, dim=int(config.state_dim))
            counts_hat = count_sketch.decode(root_state)
            utility_hat = utility_vector_from_counts(counts_hat, utility_matrix)
            count_pred_by_schedule[str(schedule)] = counts_hat
            count_util_by_schedule[str(schedule)] = utility_hat
            count_scalar_by_schedule[str(schedule)] = float(readout @ utility_hat)
        counts_hat = count_pred_by_schedule["balanced"]
        utility_hat = count_util_by_schedule["balanced"]
        scalar_hat = count_scalar_by_schedule["balanced"]
        count_l1_to_full.append(float(np.sum(np.abs(counts_hat - counts_full))))
        count_utility_l1.append(float(np.sum(np.abs(utility_hat - utility_full))))
        count_utility_l2.append(float(np.linalg.norm(utility_hat - utility_full)))
        count_scalar_abs.append(abs(float(scalar_hat) - float(scalar_full)))
        count_sched_util_spread.append(_base._pairwise_spread(list(count_util_by_schedule.values()), metric="l1"))
        count_sched_scalar_spread.append(_base._pairwise_spread(list(count_scalar_by_schedule.values()), metric="abs"))

        util_util_by_schedule: Dict[str, np.ndarray] = {}
        util_scalar_by_schedule: Dict[str, float] = {}
        util_states = [utility_sketch.encode(u) for u in leaf_utility]
        for schedule in VALID_SCHEDULES:
            root_state = _reduce_vectors(util_states, schedule=schedule, dim=int(config.state_dim))
            utility_hat2 = utility_sketch.decode(root_state)
            util_util_by_schedule[str(schedule)] = utility_hat2
            util_scalar_by_schedule[str(schedule)] = float(readout @ utility_hat2)
        utility_hat2 = util_util_by_schedule["balanced"]
        scalar_hat2 = util_scalar_by_schedule["balanced"]
        util_utility_l1.append(float(np.sum(np.abs(utility_hat2 - utility_full))))
        util_utility_l2.append(float(np.linalg.norm(utility_hat2 - utility_full)))
        util_scalar_abs.append(abs(float(scalar_hat2) - float(scalar_full)))
        util_sched_util_spread.append(_base._pairwise_spread(list(util_util_by_schedule.values()), metric="l1"))
        util_sched_scalar_spread.append(_base._pairwise_spread(list(util_scalar_by_schedule.values()), metric="abs"))

        if mlp_model is not None and torch is not None and mlp_device is not None:
            with torch.no_grad():
                x = torch.tensor(
                    _full_doc_features_from_counts(counts_full),
                    dtype=torch.float32,
                    device=mlp_device,
                ).unsqueeze(0)
                pred = mlp_model(x).squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
            mlp_utility_l1.append(float(np.sum(np.abs(pred - utility_full))))
            mlp_utility_l2.append(float(np.linalg.norm(pred - utility_full)))
            mlp_scalar_abs.append(abs(float(readout @ pred) - float(scalar_full)))

    exact_recovery = {
        "n_docs": int(len(docs_test)),
        "root_utility_l1_mean": _base._safe_stat(exact_utility_l1, kind="mean"),
        "root_utility_l2_mean": _base._safe_stat(exact_utility_l2, kind="mean"),
        "root_scalar_abs_mean": _base._safe_stat(exact_scalar_abs, kind="mean"),
        "schedule_utility_l1_spread_mean": _base._safe_stat(exact_sched_util_spread, kind="mean"),
        "schedule_scalar_abs_spread_mean": _base._safe_stat(exact_sched_scalar_spread, kind="mean"),
    }

    methods = {
        "full_doc_exact_utility": _method_metric_dict(
            supervision_kind="utility_vector_labels",
            n_docs=len(docs_test),
            utility_l1_to_full=[0.0] * len(docs_test),
            utility_l2_to_full=[0.0] * len(docs_test),
            scalar_abs_to_full=[0.0] * len(docs_test),
            schedule_utility_spread=[0.0] * len(docs_test),
            schedule_scalar_spread=[0.0] * len(docs_test),
        ),
        "tree_exact_utility": _method_metric_dict(
            supervision_kind="utility_vector_labels",
            n_docs=len(docs_test),
            utility_l1_to_full=tree_utility_l1,
            utility_l2_to_full=tree_utility_l2,
            scalar_abs_to_full=tree_scalar_abs,
            schedule_utility_spread=tree_sched_util_spread,
            schedule_scalar_spread=tree_sched_scalar_spread,
        ),
        "count_svd_ceiling": _method_metric_dict(
            supervision_kind="count_ceiling",
            n_docs=len(docs_test),
            utility_l1_to_full=count_utility_l1,
            utility_l2_to_full=count_utility_l2,
            scalar_abs_to_full=count_scalar_abs,
            schedule_utility_spread=count_sched_util_spread,
            schedule_scalar_spread=count_sched_scalar_spread,
            count_l1_to_full=count_l1_to_full,
            diagnostics=count_diag,
        ),
        "utility_pca_practical": _method_metric_dict(
            supervision_kind="utility_vector_labels",
            n_docs=len(docs_test),
            utility_l1_to_full=util_utility_l1,
            utility_l2_to_full=util_utility_l2,
            scalar_abs_to_full=util_scalar_abs,
            schedule_utility_spread=util_sched_util_spread,
            schedule_scalar_spread=util_sched_scalar_spread,
            diagnostics=utility_diag,
        ),
    }
    if mlp_model is not None:
        methods["full_doc_mlp_diag"] = _method_metric_dict(
            supervision_kind="utility_vector_labels",
            n_docs=len(docs_test),
            utility_l1_to_full=mlp_utility_l1,
            utility_l2_to_full=mlp_utility_l2,
            scalar_abs_to_full=mlp_scalar_abs,
            schedule_utility_spread=[0.0] * len(mlp_utility_l1),
            schedule_scalar_spread=[0.0] * len(mlp_utility_l1),
            diagnostics=mlp_diag,
        )
    else:
        methods["full_doc_mlp_diag"] = {
            "supervision_kind": "utility_vector_labels",
            "available": False,
            **mlp_diag,
        }

    world_stats = {
        **leaf_meta,
        "train_docs_fit": int(config.train_docs),
        "test_docs_evaluated": int(config.test_docs),
        **{f"test_{k}": v for k, v in _summarize_docs(docs_test, leaf_tokens=leaf_tokens).items()},
    }
    config_payload = asdict(config)
    config_payload.update(leaf_meta)
    return LDAUtilityVectorSummary(
        family="lda_tree_utility_vector",
        target_kind="utility_vector",
        config=config_payload,
        topic_meta=dict(world.topic_meta),
        utility_topic_meta=dict(world.utility_topic_meta),
        world_stats=world_stats,
        exact_recovery=exact_recovery,
        methods=methods,
        is_stale_generation=False,
        objective=mergeable_target_objective_semantics(
            name="lda_utility_vector_target",
            optimized_against="utility_vector_labels",
            target_kind="utility_vector",
            metadata={"family": "lda_tree_utility_vector"},
        ),
    )


def run_lda_tree_utility_vector_experiment(
    config: LDATreeUtilityVectorConfig,
) -> LDAUtilityVectorSummary:
    world = sample_lda_tree_utility_vector_world(config)
    return run_lda_tree_utility_vector_experiment_from_world(config, world)


__all__ = [
    "LDAUtilityVectorSummary",
    "LDATreeUtilityVectorConfig",
    "LDAUtilityVectorWorld",
    "VALID_DEVICE_MODES",
    "VALID_EMISSION_MODES",
    "VALID_SCHEDULES",
    "VALID_UTILITY_DESIGNS",
    "fit_projection_sketch",
    "leaf_fraction_label",
    "leaf_tokens_from_fraction",
    "lda_tree_utility_vector_world_cache_signature",
    "run_lda_tree_utility_vector_experiment",
    "run_lda_tree_utility_vector_experiment_from_world",
    "sample_lda_tree_utility_vector_world",
    "sample_topic_anchored_utility_matrix",
    "utility_vector_from_counts",
]
