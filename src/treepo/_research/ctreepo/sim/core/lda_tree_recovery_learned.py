"""
Learned ordinary-LDA recovery with additive count sketches.

This module extends the exact bag-of-words LDA recovery family with two learned
approximations that are still anchored to the exact sufficient statistic:

1. `full_doc_operator`: a neural operator on the full document histogram that
   learns the exact known-topic LDA posterior map `c_d -> pi_d`.
2. `tree_svd_sketch`: a learned additive sketch of leaf histograms,
   `s_leaf = A c_leaf`, merged by addition and decoded with `c_hat = B s_root`.

The second family is theory-friendly by construction:

    s_root = sum_leaf A c_leaf = A c_doc
    c_hat = B s_root = B A c_doc

So the only approximation axis is the rank-limited linear map `B A`. When
`state_dim >= vocab_size`, exact recovery of arbitrary histograms is in-model.
More generally, when `state_dim` is at least the rank of the sampled training
node histogram matrix, the sampled training nodes are reconstructed exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for learned LDA tree recovery. "
        "Install with: pip install torch>=2.0.0"
    ) from e

from treepo._research.ctreepo.sim.core import lda_tree_recovery as _base
from treepo._research.ctreepo.sim.core.segment_lda_ops_weight_recovery import build_leaf_spans
from treepo._research.ctreepo.sim.core.training_selection import (
    TrainingSelectionMetadata,
    clone_module_state,
    improved_metric,
    restore_module_state,
)
from treepo._research.ctreepo.sim.objective_semantics import latent_quadratic_utility_objective_semantics


VALID_DEVICE_MODES: Tuple[str, ...] = ("auto", "cpu", "cuda")
VALID_SCHEDULES: Tuple[str, ...] = _base.VALID_SCHEDULES


@dataclass(frozen=True)
class LDATreeRecoveryLearnedConfig:
    # LDA DGP.
    n_topics: int = 8
    vocab_size: int = 512
    min_tokens: int = 384
    max_tokens: int = 384
    doc_topic_concentration: float = 0.6

    # Topic-word distributions.
    topic_concentration: float = 0.2
    emission_mode: str = "anchored"
    anchor_words_per_topic: int = 20
    anchor_multiplier: float = 25.0

    # Document-level utility on inferred topic mixtures.
    # `lambda_multiplier` is a legacy internal name for the quadratic utility
    # weight, not the paper-facing local-law lambda.
    relevant_topics: int = 2
    theta_scale: float = 1.0
    zero_diagonal: bool = False
    lambda_multiplier: float = 1.0

    # Tree geometry.
    leaf_tokens: int = 16

    # Fixed-world sizes.
    train_docs: int = 512
    test_docs: int = 256

    # Known-topic document-mixture inference with fixed topics.
    inference_prior_mass: float = 0.25
    inference_max_iter: int = 200
    inference_tol: float = 1e-9

    # Learned full-document operator.
    full_hidden_dim: int = 128
    full_n_layers: int = 2

    # Learned additive sketch.
    state_dim: int = 64
    supervise_all_balanced_nodes: bool = True

    # Optimization.
    n_epochs: int = 40
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-5

    # Runtime.
    device: str = "auto"
    cuda_device: Optional[int] = None
    torch_threads: int = 0
    seed: int = 0


@dataclass(frozen=True)
class _PreparedDoc:
    counts_full: np.ndarray
    pi_true: np.ndarray
    pi_full: np.ndarray
    utility_true: float
    utility_full: float
    loglik_full: float
    leaf_counts: Tuple[np.ndarray, ...]
    balanced_node_counts: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class LearnedMethodMetrics:
    n_docs: int
    count_l1_to_full_mean: float
    count_l1_to_full_median: float
    count_l1_to_full_p95: float
    node_count_l1_mean: float
    pi_l1_to_true_mean: float
    pi_l1_to_true_median: float
    pi_l1_to_true_p95: float
    pi_l1_to_full_mean: float
    pi_l1_to_full_median: float
    pi_l1_to_full_p95: float
    utility_abs_to_true_mean: float
    utility_abs_to_true_median: float
    utility_abs_to_true_p95: float
    utility_abs_to_full_mean: float
    utility_abs_to_full_median: float
    utility_abs_to_full_p95: float
    log_likelihood_mean: float
    log_likelihood_abs_to_full_mean: float
    log_likelihood_abs_to_full_median: float
    log_likelihood_abs_to_full_p95: float
    schedule_count_l1_spread_mean: float
    schedule_pi_l1_spread_mean: float
    schedule_utility_spread_mean: float
    schedule_loglik_spread_mean: float


@dataclass(frozen=True)
class LDATreeRecoveryLearnedSummary:
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    utility_truth: Dict[str, object]
    exact_reference: Dict[str, object]
    learning: Dict[str, object]
    methods: Dict[str, object]
    objective: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "config": self.config,
                "topic_meta": self.topic_meta,
                "utility_truth": self.utility_truth,
                "exact_reference": self.exact_reference,
                "learning": self.learning,
                "methods": self.methods,
                "objective": self.objective,
            },
            indent=2,
            sort_keys=True,
        )


def _validate_config(config: LDATreeRecoveryLearnedConfig) -> None:
    if int(config.train_docs) <= 0:
        raise ValueError("train_docs must be positive for the learned recovery family")
    if int(config.test_docs) <= 0:
        raise ValueError("test_docs must be positive")
    if int(config.state_dim) <= 0:
        raise ValueError("state_dim must be positive")
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
    if (
        str(config.emission_mode).strip().lower() == "anchored"
        and int(config.n_topics) * int(config.anchor_words_per_topic) >= int(config.vocab_size)
    ):
        raise ValueError("anchored emission_mode requires n_topics * anchor_words_per_topic < vocab_size")
    base_cfg = _base_config(config)
    _base._validate_config(base_cfg)


def _base_config(config: LDATreeRecoveryLearnedConfig) -> _base.LDATreeRecoveryConfig:
    return _base.LDATreeRecoveryConfig(
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        topic_concentration=float(config.topic_concentration),
        emission_mode=str(config.emission_mode),
        anchor_words_per_topic=int(config.anchor_words_per_topic),
        anchor_multiplier=float(config.anchor_multiplier),
        relevant_topics=int(config.relevant_topics),
        theta_scale=float(config.theta_scale),
        zero_diagonal=bool(config.zero_diagonal),
        lambda_multiplier=float(config.lambda_multiplier),
        leaf_tokens=int(config.leaf_tokens),
        train_docs=int(config.train_docs),
        test_docs=int(config.test_docs),
        inference_prior_mass=float(config.inference_prior_mass),
        inference_max_iter=int(config.inference_max_iter),
        inference_tol=float(config.inference_tol),
        seed=int(config.seed),
    )


def _set_global_seed(seed: int, *, torch_threads: int) -> None:
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


def _resolve_device(config: LDATreeRecoveryLearnedConfig) -> torch.device:
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


def _safe_stat(xs: Sequence[float], *, kind: str) -> float:
    arr = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if kind == "mean":
        return float(np.mean(arr))
    if kind == "median":
        return float(np.median(arr))
    if kind == "p95":
        return float(np.percentile(arr, 95.0))
    raise ValueError(f"unsupported stat kind: {kind!r}")


def _balanced_node_counts(leaf_counts: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    nodes: List[np.ndarray] = [np.asarray(x, dtype=np.float64).copy() for x in leaf_counts]
    cur = [np.asarray(x, dtype=np.float64).copy() for x in leaf_counts]
    while len(cur) > 1:
        nxt: List[np.ndarray] = []
        i = 0
        while i < len(cur):
            if i + 1 < len(cur):
                merged = cur[i] + cur[i + 1]
                nxt.append(merged)
                nodes.append(merged)
                i += 2
            else:
                nxt.append(cur[i])
                i += 1
        cur = nxt
    return tuple(nodes)


def _prepare_docs(
    docs: Sequence[_base.LDATreeRecoveryDoc],
    *,
    config: LDATreeRecoveryLearnedConfig,
    topics_phi: Sequence[np.ndarray],
    theta_true: np.ndarray,
    W_base: np.ndarray,
) -> Tuple[_PreparedDoc, ...]:
    out: List[_PreparedDoc] = []
    for doc in docs:
        counts_full = _base._counts_from_tokens(doc.tokens, vocab_size=int(config.vocab_size))
        pi_true = np.asarray(doc.topic_weights, dtype=np.float64)
        pi_full = _base._infer_topic_mixture_from_counts(
            counts_full,
            topics_phi=topics_phi,
            prior_mass=float(config.inference_prior_mass),
            max_iter=int(config.inference_max_iter),
            tol=float(config.inference_tol),
        )
        utility_true = _base._utility_from_pi(
            pi_true,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        utility_full = _base._utility_from_pi(
            pi_full,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        loglik_full = _base._doc_log_likelihood(counts_full, pi=pi_full, topics_phi=topics_phi)

        leaf_spans = build_leaf_spans(len(doc.tokens), leaf_tokens=int(config.leaf_tokens))
        leaf_counts = tuple(
            _base._counts_from_tokens(doc.tokens[int(start) : int(end)], vocab_size=int(config.vocab_size))
            for start, end in leaf_spans
        )
        out.append(
            _PreparedDoc(
                counts_full=np.asarray(counts_full, dtype=np.float64).copy(),
                pi_true=np.asarray(pi_true, dtype=np.float64).copy(),
                pi_full=np.asarray(pi_full, dtype=np.float64).copy(),
                utility_true=float(utility_true),
                utility_full=float(utility_full),
                loglik_full=float(loglik_full),
                leaf_counts=tuple(np.asarray(x, dtype=np.float64).copy() for x in leaf_counts),
                balanced_node_counts=_balanced_node_counts(leaf_counts),
            )
        )
    return tuple(out)


def _split_train_val(
    docs: Sequence[_PreparedDoc],
) -> Tuple[Tuple[_PreparedDoc, ...], Tuple[_PreparedDoc, ...]]:
    n = int(len(docs))
    if n <= 4:
        return (tuple(docs), tuple())
    n_val = min(max(1, n // 5), n - 1)
    return (tuple(docs[:-n_val]), tuple(docs[-n_val:]))


def _full_doc_features_from_counts(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    total = float(np.sum(c))
    if total > 0.0:
        freqs = c / total
    else:
        freqs = np.zeros_like(c, dtype=np.float64)
    return np.concatenate([freqs, np.asarray([math.log1p(total)], dtype=np.float64)], axis=0)


def _project_counts_to_histogram(raw_counts: np.ndarray, *, total_tokens: int, vocab_size: int) -> np.ndarray:
    arr = np.asarray(raw_counts, dtype=np.float64).reshape(int(vocab_size))
    clipped = np.clip(arr, 0.0, None)
    mass = float(np.sum(clipped))
    if mass <= 0.0:
        return np.full((int(vocab_size),), float(total_tokens) / float(vocab_size), dtype=np.float64)
    return (float(total_tokens) / mass) * clipped


class FullDocTopicOperator(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class LinearCountSketch(nn.Module):
    def __init__(self, *, vocab_size: int, state_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(int(vocab_size), int(state_dim), bias=False)
        self.decoder = nn.Linear(int(state_dim), int(vocab_size), bias=False)

    def encode(self, counts: torch.Tensor) -> torch.Tensor:
        return self.encoder(counts)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(counts))


@dataclass(frozen=True)
class _SVDCountSketch:
    components: np.ndarray  # shape [m, V]

    @property
    def state_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.components.shape[1])

    def encode(self, counts: np.ndarray) -> np.ndarray:
        x = np.asarray(counts, dtype=np.float64).reshape(self.vocab_size)
        return self.components @ x

    def decode(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).reshape(self.state_dim)
        return self.components.T @ s


def _right_singular_system(node_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(node_counts, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("node_counts must be a rank-2 array")
    gram = x.T @ x
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.clip(np.asarray(evals[order], dtype=np.float64), 0.0, None)
    s = np.sqrt(evals, dtype=np.float64)
    vt = np.asarray(evecs[:, order].T, dtype=np.float64)
    return s, vt


def _svd_fit_from_right_singular_system(
    singular_values: np.ndarray,
    right_vectors: np.ndarray,
    *,
    state_dim: int,
    vocab_size: int,
) -> Tuple[_SVDCountSketch, Dict[str, object]]:
    s = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    vt = np.asarray(right_vectors, dtype=np.float64)
    if vt.ndim != 2 or int(vt.shape[1]) != int(vocab_size):
        raise ValueError("right_vectors must have shape [rank, vocab_size]")
    train_rank = int(np.sum(s > 1e-10))
    kept = min(int(state_dim), int(vt.shape[0]))
    components = np.zeros((int(state_dim), int(vocab_size)), dtype=np.float64)
    if kept > 0:
        components[:kept, :] = vt[:kept, :]
    total_var = float(np.sum(s ** 2))
    kept_var = float(np.sum((s[:kept] ** 2))) if kept > 0 else 0.0
    fit = _SVDCountSketch(components=components)
    diag = {
        "fit_mode": "svd",
        "state_dim": int(state_dim),
        "train_rank": int(train_rank),
        "kept_components": int(kept),
        "explained_variance_ratio_train": (kept_var / total_var) if total_var > 0.0 else 1.0,
        "exact_family_representable": bool(int(state_dim) >= int(vocab_size)),
        "exact_train_manifold_representable": bool(int(state_dim) >= int(train_rank)),
    }
    return fit, diag


def _fit_svd_count_sketch(
    node_counts: np.ndarray,
    *,
    state_dim: int,
    vocab_size: int,
) -> Tuple[_SVDCountSketch, Dict[str, object]]:
    x = np.asarray(node_counts, dtype=np.float64)
    if x.ndim != 2 or int(x.shape[1]) != int(vocab_size):
        raise ValueError("node_counts must have shape [n_examples, vocab_size]")
    if int(x.shape[0]) == 0:
        raise ValueError("need at least one node count example to fit the SVD sketch")
    s, vt = _right_singular_system(x)
    return _svd_fit_from_right_singular_system(
        s,
        vt,
        state_dim=int(state_dim),
        vocab_size=int(vocab_size),
    )


def _train_full_doc_operator(
    train_docs: Sequence[_PreparedDoc],
    val_docs: Sequence[_PreparedDoc],
    *,
    config: LDATreeRecoveryLearnedConfig,
    device: torch.device,
) -> Tuple[FullDocTopicOperator, Dict[str, object]]:
    x_train = np.stack([_full_doc_features_from_counts(doc.counts_full) for doc in train_docs], axis=0)
    y_train = np.stack([np.asarray(doc.pi_full, dtype=np.float64) for doc in train_docs], axis=0)

    x_val = np.stack([_full_doc_features_from_counts(doc.counts_full) for doc in val_docs], axis=0) if val_docs else np.zeros((0, x_train.shape[1]), dtype=np.float64)
    y_val = np.stack([np.asarray(doc.pi_full, dtype=np.float64) for doc in val_docs], axis=0) if val_docs else np.zeros((0, y_train.shape[1]), dtype=np.float64)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=int(config.batch_size), shuffle=True, drop_last=False)

    model = FullDocTopicOperator(
        input_dim=int(x_train.shape[1]),
        hidden_dim=int(config.full_hidden_dim),
        n_layers=int(config.full_n_layers),
        output_dim=int(config.n_topics),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    train_loss_last = float("nan")
    val_loss_last = float("nan")
    best_state = None
    best_selection = TrainingSelectionMetadata(
        mode="best_val_loss" if val_docs else "final_epoch_no_validation",
        split="val" if val_docs else "config",
        metric_name="val_loss_final" if val_docs else "train_loss_final",
        metric_value=float("nan"),
        best_epoch=0,
    )
    for _epoch in range(int(config.n_epochs)):
        epoch_idx = int(_epoch)
        model.train()
        losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss_last = _safe_stat(losses, kind="mean")
        if val_docs:
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(x_val, dtype=torch.float32, device=device)
                yv = torch.tensor(y_val, dtype=torch.float32, device=device)
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

    if val_docs:
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
        "train_docs_fit": int(len(train_docs)),
        "val_docs_fit": int(len(val_docs)),
        "train_loss_final": float(train_loss_last),
        "val_loss_final": float(val_loss_last),
        **best_selection.to_dict(),
        "full_hidden_dim": int(config.full_hidden_dim),
        "full_n_layers": int(config.full_n_layers),
        "n_epochs": int(config.n_epochs),
        "batch_size": int(config.batch_size),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "device": str(device),
    }


def _eval_full_doc_operator(
    model: FullDocTopicOperator,
    docs: Sequence[_PreparedDoc],
    *,
    config: LDATreeRecoveryLearnedConfig,
    topics_phi: Sequence[np.ndarray],
    theta_true: np.ndarray,
    W_base: np.ndarray,
    device: torch.device,
) -> LearnedMethodMetrics:
    count_to_full = [float("nan")] * len(docs)
    node_count = [float("nan")] * len(docs)
    pi_to_true: List[float] = []
    pi_to_full: List[float] = []
    util_to_true: List[float] = []
    util_to_full: List[float] = []
    logliks: List[float] = []
    loglik_to_full: List[float] = []
    schedule_count_spread = [float("nan")] * len(docs)
    schedule_pi_spread = [float("nan")] * len(docs)
    schedule_util_spread = [float("nan")] * len(docs)
    schedule_loglik_spread = [float("nan")] * len(docs)

    with torch.no_grad():
        for doc in docs:
            feat = torch.tensor(
                _full_doc_features_from_counts(doc.counts_full),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            pi_pred = model(feat).squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
            utility_pred = _base._utility_from_pi(
                pi_pred,
                theta=theta_true,
                W_base=W_base,
                lambda_multiplier=float(config.lambda_multiplier),
            )
            utility_true = _base._utility_from_pi(
                doc.pi_true,
                theta=theta_true,
                W_base=W_base,
                lambda_multiplier=float(config.lambda_multiplier),
            )
            utility_full = _base._utility_from_pi(
                doc.pi_full,
                theta=theta_true,
                W_base=W_base,
                lambda_multiplier=float(config.lambda_multiplier),
            )
            loglik_pred = _base._doc_log_likelihood(doc.counts_full, pi=pi_pred, topics_phi=topics_phi)

            pi_to_true.append(float(np.sum(np.abs(pi_pred - doc.pi_true))))
            pi_to_full.append(float(np.sum(np.abs(pi_pred - doc.pi_full))))
            util_to_true.append(abs(float(utility_pred) - float(utility_true)))
            util_to_full.append(abs(float(utility_pred) - float(utility_full)))
            logliks.append(float(loglik_pred))
            loglik_to_full.append(abs(float(loglik_pred) - float(doc.loglik_full)))

    return LearnedMethodMetrics(
        n_docs=int(len(docs)),
        count_l1_to_full_mean=_safe_stat(count_to_full, kind="mean"),
        count_l1_to_full_median=_safe_stat(count_to_full, kind="median"),
        count_l1_to_full_p95=_safe_stat(count_to_full, kind="p95"),
        node_count_l1_mean=_safe_stat(node_count, kind="mean"),
        pi_l1_to_true_mean=_safe_stat(pi_to_true, kind="mean"),
        pi_l1_to_true_median=_safe_stat(pi_to_true, kind="median"),
        pi_l1_to_true_p95=_safe_stat(pi_to_true, kind="p95"),
        pi_l1_to_full_mean=_safe_stat(pi_to_full, kind="mean"),
        pi_l1_to_full_median=_safe_stat(pi_to_full, kind="median"),
        pi_l1_to_full_p95=_safe_stat(pi_to_full, kind="p95"),
        utility_abs_to_true_mean=_safe_stat(util_to_true, kind="mean"),
        utility_abs_to_true_median=_safe_stat(util_to_true, kind="median"),
        utility_abs_to_true_p95=_safe_stat(util_to_true, kind="p95"),
        utility_abs_to_full_mean=_safe_stat(util_to_full, kind="mean"),
        utility_abs_to_full_median=_safe_stat(util_to_full, kind="median"),
        utility_abs_to_full_p95=_safe_stat(util_to_full, kind="p95"),
        log_likelihood_mean=_safe_stat(logliks, kind="mean"),
        log_likelihood_abs_to_full_mean=_safe_stat(loglik_to_full, kind="mean"),
        log_likelihood_abs_to_full_median=_safe_stat(loglik_to_full, kind="median"),
        log_likelihood_abs_to_full_p95=_safe_stat(loglik_to_full, kind="p95"),
        schedule_count_l1_spread_mean=_safe_stat(schedule_count_spread, kind="mean"),
        schedule_pi_l1_spread_mean=_safe_stat(schedule_pi_spread, kind="mean"),
        schedule_utility_spread_mean=_safe_stat(schedule_util_spread, kind="mean"),
        schedule_loglik_spread_mean=_safe_stat(schedule_loglik_spread, kind="mean"),
    )


def _reduce_states_svd(
    leaf_states: Sequence[np.ndarray],
    *,
    schedule: str,
    state_dim: int,
) -> np.ndarray:
    if len(leaf_states) == 0:
        return np.zeros((int(state_dim),), dtype=np.float64)
    sch = str(schedule).strip().lower()
    if sch == "left_to_right":
        acc = np.zeros((int(state_dim),), dtype=np.float64)
        for s in leaf_states:
            acc = acc + np.asarray(s, dtype=np.float64)
        return acc
    if sch == "right_to_left":
        acc = np.zeros((int(state_dim),), dtype=np.float64)
        for s in reversed(list(leaf_states)):
            acc = acc + np.asarray(s, dtype=np.float64)
        return acc
    if sch == "balanced":
        cur = [np.asarray(s, dtype=np.float64) for s in leaf_states]
        while len(cur) > 1:
            nxt: List[np.ndarray] = []
            i = 0
            while i < len(cur):
                if i + 1 < len(cur):
                    nxt.append(cur[i] + cur[i + 1])
                    i += 2
                else:
                    nxt.append(cur[i])
                    i += 1
            cur = nxt
        return cur[0]
    raise ValueError(f"unsupported schedule: {schedule!r}; expected one of {VALID_SCHEDULES}")


def _decode_balanced_nodes_svd(
    fit: _SVDCountSketch,
    leaf_counts: Sequence[np.ndarray],
) -> Tuple[np.ndarray, ...]:
    decoded: List[np.ndarray] = []
    cur = [fit.encode(c) for c in leaf_counts]
    decoded.extend(fit.decode(s) for s in cur)
    while len(cur) > 1:
        nxt: List[np.ndarray] = []
        i = 0
        while i < len(cur):
            if i + 1 < len(cur):
                merged = cur[i] + cur[i + 1]
                nxt.append(merged)
                decoded.append(fit.decode(merged))
                i += 2
            else:
                nxt.append(cur[i])
                i += 1
        cur = nxt
    return tuple(np.asarray(x, dtype=np.float64).copy() for x in decoded)


def _eval_tree_svd_sketch(
    fit: _SVDCountSketch,
    docs: Sequence[_PreparedDoc],
    *,
    config: LDATreeRecoveryLearnedConfig,
    topics_phi: Sequence[np.ndarray],
    theta_true: np.ndarray,
    W_base: np.ndarray,
) -> LearnedMethodMetrics:
    count_to_full: List[float] = []
    node_count: List[float] = []
    pi_to_true: List[float] = []
    pi_to_full: List[float] = []
    util_to_true: List[float] = []
    util_to_full: List[float] = []
    logliks: List[float] = []
    loglik_to_full: List[float] = []
    schedule_count_spread: List[float] = []
    schedule_pi_spread: List[float] = []
    schedule_util_spread: List[float] = []
    schedule_loglik_spread: List[float] = []

    for doc in docs:
        utility_true = _base._utility_from_pi(
            doc.pi_true,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        utility_full = _base._utility_from_pi(
            doc.pi_full,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        leaf_states = [fit.encode(c) for c in doc.leaf_counts]
        counts_by_schedule: Dict[str, np.ndarray] = {}
        pi_by_schedule: Dict[str, np.ndarray] = {}
        utility_by_schedule: Dict[str, float] = {}
        loglik_by_schedule: Dict[str, float] = {}

        total_tokens = int(np.sum(doc.counts_full))
        for schedule in VALID_SCHEDULES:
            root_state = _reduce_states_svd(leaf_states, schedule=schedule, state_dim=fit.state_dim)
            counts_raw = fit.decode(root_state)
            counts_pred = _project_counts_to_histogram(
                counts_raw,
                total_tokens=total_tokens,
                vocab_size=int(config.vocab_size),
            )
            pi_pred = _base._infer_topic_mixture_from_counts(
                counts_pred,
                topics_phi=topics_phi,
                prior_mass=float(config.inference_prior_mass),
                max_iter=int(config.inference_max_iter),
                tol=float(config.inference_tol),
            )
            utility_pred = _base._utility_from_pi(
                pi_pred,
                theta=theta_true,
                W_base=W_base,
                lambda_multiplier=float(config.lambda_multiplier),
            )
            loglik_pred = _base._doc_log_likelihood(doc.counts_full, pi=pi_pred, topics_phi=topics_phi)
            counts_by_schedule[str(schedule)] = counts_pred
            pi_by_schedule[str(schedule)] = pi_pred
            utility_by_schedule[str(schedule)] = float(utility_pred)
            loglik_by_schedule[str(schedule)] = float(loglik_pred)

        counts_bal = counts_by_schedule["balanced"]
        pi_bal = pi_by_schedule["balanced"]
        utility_bal = utility_by_schedule["balanced"]
        loglik_bal = loglik_by_schedule["balanced"]

        count_to_full.append(float(np.sum(np.abs(counts_bal - doc.counts_full))))
        pi_to_true.append(float(np.sum(np.abs(pi_bal - doc.pi_true))))
        pi_to_full.append(float(np.sum(np.abs(pi_bal - doc.pi_full))))
        util_to_true.append(abs(float(utility_bal) - float(utility_true)))
        util_to_full.append(abs(float(utility_bal) - float(utility_full)))
        logliks.append(float(loglik_bal))
        loglik_to_full.append(abs(float(loglik_bal) - float(doc.loglik_full)))

        schedule_count_spread.append(_base._pairwise_spread(list(counts_by_schedule.values()), metric="l1"))
        schedule_pi_spread.append(_base._pairwise_spread(list(pi_by_schedule.values()), metric="l1"))
        schedule_util_spread.append(_base._pairwise_spread(list(utility_by_schedule.values()), metric="abs"))
        schedule_loglik_spread.append(_base._pairwise_spread(list(loglik_by_schedule.values()), metric="abs"))

        decoded_nodes = _decode_balanced_nodes_svd(fit, doc.leaf_counts)
        if len(decoded_nodes) != len(doc.balanced_node_counts):
            raise RuntimeError("decoded node count list does not match balanced_node_counts")
        node_l1 = [
            float(np.sum(np.abs(
                _project_counts_to_histogram(pred, total_tokens=int(np.sum(truth)), vocab_size=int(config.vocab_size))
                - truth
            )))
            for pred, truth in zip(decoded_nodes, doc.balanced_node_counts)
        ]
        node_count.append(_safe_stat(node_l1, kind="mean"))

    return LearnedMethodMetrics(
        n_docs=int(len(docs)),
        count_l1_to_full_mean=_safe_stat(count_to_full, kind="mean"),
        count_l1_to_full_median=_safe_stat(count_to_full, kind="median"),
        count_l1_to_full_p95=_safe_stat(count_to_full, kind="p95"),
        node_count_l1_mean=_safe_stat(node_count, kind="mean"),
        pi_l1_to_true_mean=_safe_stat(pi_to_true, kind="mean"),
        pi_l1_to_true_median=_safe_stat(pi_to_true, kind="median"),
        pi_l1_to_true_p95=_safe_stat(pi_to_true, kind="p95"),
        pi_l1_to_full_mean=_safe_stat(pi_to_full, kind="mean"),
        pi_l1_to_full_median=_safe_stat(pi_to_full, kind="median"),
        pi_l1_to_full_p95=_safe_stat(pi_to_full, kind="p95"),
        utility_abs_to_true_mean=_safe_stat(util_to_true, kind="mean"),
        utility_abs_to_true_median=_safe_stat(util_to_true, kind="median"),
        utility_abs_to_true_p95=_safe_stat(util_to_true, kind="p95"),
        utility_abs_to_full_mean=_safe_stat(util_to_full, kind="mean"),
        utility_abs_to_full_median=_safe_stat(util_to_full, kind="median"),
        utility_abs_to_full_p95=_safe_stat(util_to_full, kind="p95"),
        log_likelihood_mean=_safe_stat(logliks, kind="mean"),
        log_likelihood_abs_to_full_mean=_safe_stat(loglik_to_full, kind="mean"),
        log_likelihood_abs_to_full_median=_safe_stat(loglik_to_full, kind="median"),
        log_likelihood_abs_to_full_p95=_safe_stat(loglik_to_full, kind="p95"),
        schedule_count_l1_spread_mean=_safe_stat(schedule_count_spread, kind="mean"),
        schedule_pi_l1_spread_mean=_safe_stat(schedule_pi_spread, kind="mean"),
        schedule_utility_spread_mean=_safe_stat(schedule_util_spread, kind="mean"),
        schedule_loglik_spread_mean=_safe_stat(schedule_loglik_spread, kind="mean"),
    )


def _evaluate_svd_reconstruction(
    fit: _SVDCountSketch,
    node_counts: np.ndarray,
    *,
    vocab_size: int,
) -> Dict[str, float]:
    if node_counts.size == 0:
        return {
            "count_l1_mean": float("nan"),
            "count_rmse_mean": float("nan"),
        }
    l1s: List[float] = []
    rmses: List[float] = []
    for row in np.asarray(node_counts, dtype=np.float64):
        pred = fit.decode(fit.encode(row))
        clipped = _project_counts_to_histogram(pred, total_tokens=int(np.sum(row)), vocab_size=int(vocab_size))
        diff = clipped - row
        l1s.append(float(np.sum(np.abs(diff))))
        rmses.append(float(np.sqrt(np.mean(diff ** 2))))
    return {
        "count_l1_mean": _safe_stat(l1s, kind="mean"),
        "count_rmse_mean": _safe_stat(rmses, kind="mean"),
    }


def run_lda_tree_recovery_learned_experiment(
    config: LDATreeRecoveryLearnedConfig,
) -> LDATreeRecoveryLearnedSummary:
    _validate_config(config)
    _set_global_seed(int(config.seed), torch_threads=int(config.torch_threads))
    device = _resolve_device(config)

    base_cfg = _base_config(config)
    world = _base.sample_lda_tree_recovery_world(base_cfg)
    exact_summary = _base.run_lda_tree_recovery_experiment_from_world(base_cfg, world)

    topics_phi = tuple(np.asarray(t, dtype=np.float64) for t in world.topics_phi)
    theta_true = np.asarray(world.theta_true, dtype=np.float64)
    W_base = np.asarray(world.W_base, dtype=np.float64)

    docs_train = _prepare_docs(
        world.docs_train[: int(config.train_docs)],
        config=config,
        topics_phi=topics_phi,
        theta_true=theta_true,
        W_base=W_base,
    )
    docs_test = _prepare_docs(
        world.docs_test[: int(config.test_docs)],
        config=config,
        topics_phi=topics_phi,
        theta_true=theta_true,
        W_base=W_base,
    )
    train_fit_docs, val_fit_docs = _split_train_val(docs_train)

    full_doc_model, full_doc_diag = _train_full_doc_operator(
        train_fit_docs,
        val_fit_docs,
        config=config,
        device=device,
    )
    full_doc_metrics = _eval_full_doc_operator(
        full_doc_model,
        docs_test,
        config=config,
        topics_phi=topics_phi,
        theta_true=theta_true,
        W_base=W_base,
        device=device,
    )

    train_node_counts = np.stack(
        [
            np.asarray(node, dtype=np.float64)
            for doc in train_fit_docs
            for node in (
                doc.balanced_node_counts
                if bool(config.supervise_all_balanced_nodes)
                else (doc.counts_full,)
            )
        ],
        axis=0,
    )
    val_node_counts = np.stack(
        [
            np.asarray(node, dtype=np.float64)
            for doc in val_fit_docs
            for node in (
                doc.balanced_node_counts
                if bool(config.supervise_all_balanced_nodes)
                else (doc.counts_full,)
            )
        ],
        axis=0,
    ) if val_fit_docs else np.zeros((0, int(config.vocab_size)), dtype=np.float64)

    tree_fit, tree_diag = _fit_svd_count_sketch(
        train_node_counts,
        state_dim=int(config.state_dim),
        vocab_size=int(config.vocab_size),
    )
    tree_diag = {
        **tree_diag,
        "train_docs_fit": int(len(train_fit_docs)),
        "val_docs_fit": int(len(val_fit_docs)),
        "train_node_examples": int(train_node_counts.shape[0]),
        "val_node_examples": int(val_node_counts.shape[0]),
        "supervise_all_balanced_nodes": bool(config.supervise_all_balanced_nodes),
        "train_reconstruction": _evaluate_svd_reconstruction(
            tree_fit,
            train_node_counts,
            vocab_size=int(config.vocab_size),
        ),
        "val_reconstruction": _evaluate_svd_reconstruction(
            tree_fit,
            val_node_counts,
            vocab_size=int(config.vocab_size),
        ),
    }
    tree_metrics = _eval_tree_svd_sketch(
        tree_fit,
        docs_test,
        config=config,
        topics_phi=topics_phi,
        theta_true=theta_true,
        W_base=W_base,
    )

    learning = {
        "train_docs_requested": int(config.train_docs),
        "test_docs_requested": int(config.test_docs),
        "full_doc_operator": full_doc_diag,
        "tree_svd_sketch": tree_diag,
    }

    methods = {
        "full_doc_operator": asdict(full_doc_metrics),
        "tree_svd_sketch": asdict(tree_metrics),
    }

    public_config = {**asdict(config), "quadratic_utility_weight": float(config.lambda_multiplier)}
    public_config.pop("lambda_multiplier", None)
    return LDATreeRecoveryLearnedSummary(
        config=public_config,
        topic_meta=dict(world.topic_meta),
        utility_truth=dict(exact_summary.utility_truth),
        exact_reference={
            "exact_recovery": dict(exact_summary.exact_recovery),
            "methods": dict(exact_summary.methods),
            "world_stats": dict(exact_summary.world_stats),
        },
        learning=learning,
        methods=methods,
        objective=latent_quadratic_utility_objective_semantics(
            name="lda_document_utility_target",
            optimized_against="learned_approximation_to_document_utility",
            quadratic_utility_weight=float(config.lambda_multiplier),
            linear_component_name="topic_mixture_linear_term",
            interaction_component_name="topic_mixture_quadratic_term",
            weighting_scheme="linear_plus_quadratic_utility",
            metadata={"problem_id": "lda_tree_recovery_learned"},
        ),
    )


__all__ = [
    "LDATreeRecoveryLearnedConfig",
    "LDATreeRecoveryLearnedSummary",
    "LearnedMethodMetrics",
    "VALID_DEVICE_MODES",
    "VALID_SCHEDULES",
    "_PreparedDoc",
    "_base_config",
    "_balanced_node_counts",
    "_eval_full_doc_operator",
    "_eval_tree_svd_sketch",
    "_fit_svd_count_sketch",
    "_full_doc_features_from_counts",
    "_prepare_docs",
    "_project_counts_to_histogram",
    "_right_singular_system",
    "_split_train_val",
    "_svd_fit_from_right_singular_system",
    "_train_full_doc_operator",
    "_validate_config",
    "run_lda_tree_recovery_learned_experiment",
]
