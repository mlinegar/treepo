"""
Hashed-counts classification honesty simulation.

We generate class-conditional token streams, hash tokens into count vectors,
learn an encoder/merger/readout, and evaluate OPS-style honesty:
- C1 (Sufficiency): leaf posterior matches Bayes-optimal posterior given hashed counts.
- C2 (Idempotence): re-merging a state with itself preserves the posterior.
- C3 (Merge consistency): merged posterior matches Bayes-optimal posterior
  for the concatenated counts.
"""

from __future__ import annotations

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
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for hashed-classification honesty simulations. "
        "Install with: pip install torch>=2.0.0"
    ) from e

from treepo._research.tree.learned_sketch_simulation import audit_sample_count


@dataclass(frozen=True)
class HashedDoc:
    label: int
    leaf_counts: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class HashedClassificationConfig:
    n_classes: int = 5
    vocab_size: int = 10000
    hash_size: int = 2048
    dirichlet_alpha: float = 0.3
    min_tokens: int = 128
    max_tokens: int = 512
    min_leaf_tokens: int = 16
    max_leaf_tokens: int = 64
    train_docs: int = 200
    test_docs: int = 80
    state_dim: int = 64
    hidden_dim: int = 128
    merger_hidden_dim: int = 64
    n_epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    leaf_weight: float = 1.0
    c2_weight: float = 0.1
    c3_weight: float = 1.0
    c3_state_weight: float = 0.5
    audit_policy: str = "all"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 1.0
    audit_scale: float = 1.0
    use_log1p: bool = True
    normalize_counts: bool = True
    discrepancy_threshold: float = 0.1
    seed: int = 0
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    torch_threads: int = 0


@dataclass(frozen=True)
class HonestyStats:
    mean_discrepancy: float
    violation_rate: float
    n: int


@dataclass(frozen=True)
class HashedClassificationSummary:
    config: Dict[str, object]
    train_loss_final: float
    leaf_accuracy: float
    root_accuracy: float
    c1: HonestyStats
    c2: HonestyStats
    c3: HonestyStats

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "train_loss_final": self.train_loss_final,
            "leaf_accuracy": self.leaf_accuracy,
            "root_accuracy": self.root_accuracy,
            "c1": asdict(self.c1),
            "c2": asdict(self.c2),
            "c3": asdict(self.c3),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


class Encoder(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(state_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Merger(nn.Module):
    def __init__(self, *, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(state_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(state_dim)),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class Readout(nn.Module):
    def __init__(self, *, state_dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(state_dim), int(n_classes))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _hash_tokens(tokens: np.ndarray, hash_size: int) -> np.ndarray:
    # Deterministic multiplicative hash (Knuth) mod hash_size.
    return (tokens.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(hash_size)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - float(np.max(x))
    exp = np.exp(x)
    return exp / float(np.sum(exp))


def _counts_to_features(
    counts: np.ndarray,
    *,
    use_log1p: bool,
    normalize: bool,
) -> np.ndarray:
    x = counts.astype(np.float32)
    if use_log1p:
        x = np.log1p(x)
    if normalize:
        denom = float(np.sum(x))
        if denom > 0:
            x = x / denom
    return x


def _soft_ce(logits: torch.Tensor, target_probs: np.ndarray) -> torch.Tensor:
    target = torch.tensor(target_probs, dtype=logits.dtype, device=logits.device)
    logp = torch.log_softmax(logits, dim=-1)
    return -(target * logp).sum(dim=-1).mean()


def _l1_discrepancy(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(np.abs(p - q)))


def _stats(xs: List[float], threshold: float) -> HonestyStats:
    if len(xs) == 0:
        return HonestyStats(mean_discrepancy=0.0, violation_rate=0.0, n=0)
    arr = np.array(xs, dtype=np.float64)
    mean = float(np.mean(arr))
    viol = float(np.mean(arr > float(threshold)))
    return HonestyStats(mean_discrepancy=mean, violation_rate=viol, n=int(len(xs)))


def _make_class_distributions(
    *,
    n_classes: int,
    vocab_size: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if alpha <= 0.0:
        raise ValueError("dirichlet_alpha must be > 0")
    return rng.dirichlet(alpha * np.ones(int(vocab_size)), size=int(n_classes)).astype(
        np.float64
    )


def _hash_class_probs(
    class_probs: np.ndarray,
    *,
    hash_size: int,
) -> np.ndarray:
    n_classes, vocab_size = class_probs.shape
    tokens = np.arange(int(vocab_size), dtype=np.uint64)
    bins = _hash_tokens(tokens, int(hash_size)).astype(np.int64)
    out = np.zeros((int(n_classes), int(hash_size)), dtype=np.float64)
    for c in range(int(n_classes)):
        np.add.at(out[c], bins, class_probs[c])
    out = np.maximum(out, 1e-12)
    return out


def _bayes_posterior(
    counts: np.ndarray,
    *,
    log_probs: np.ndarray,
    log_prior: np.ndarray,
) -> np.ndarray:
    logits = log_prior + log_probs @ counts.astype(np.float64)
    return _softmax_np(logits)


def generate_docs(config: HashedClassificationConfig) -> Tuple[HashedDoc, ...]:
    rng = np.random.default_rng(int(config.seed))
    class_probs = _make_class_distributions(
        n_classes=int(config.n_classes),
        vocab_size=int(config.vocab_size),
        alpha=float(config.dirichlet_alpha),
        rng=rng,
    )
    n_docs = int(config.train_docs) + int(config.test_docs)
    docs: List[HashedDoc] = []
    for _ in range(n_docs):
        label = int(rng.integers(0, int(config.n_classes)))
        n_tokens = int(rng.integers(int(config.min_tokens), int(config.max_tokens) + 1))
        tokens = rng.choice(
            int(config.vocab_size), size=n_tokens, replace=True, p=class_probs[label]
        ).astype(np.int64)
        bins = _hash_tokens(tokens, int(config.hash_size)).astype(np.int64)

        leaves: List[np.ndarray] = []
        idx = 0
        while idx < n_tokens:
            leaf_len = int(
                rng.integers(int(config.min_leaf_tokens), int(config.max_leaf_tokens) + 1)
            )
            leaf_bins = bins[idx : idx + leaf_len]
            counts = np.bincount(leaf_bins, minlength=int(config.hash_size)).astype(
                np.int32
            )
            leaves.append(counts)
            idx += leaf_len

        docs.append(HashedDoc(label=label, leaf_counts=tuple(leaves)))

    return tuple(docs)


def train_model(
    encoder: Encoder,
    merger: Merger,
    readout: Readout,
    train_docs: Sequence[HashedDoc],
    *,
    config: HashedClassificationConfig,
    log_probs: np.ndarray,
    log_prior: np.ndarray,
    device: torch.device,
) -> float:
    encoder.to(device)
    merger.to(device)
    readout.to(device)
    encoder.train()
    merger.train()
    readout.train()

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(merger.parameters()) + list(readout.parameters()),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    rng = random.Random(int(config.seed))

    train_loss_final = float("nan")
    for _ in range(int(config.n_epochs)):
        idxs = list(range(len(train_docs)))
        rng.shuffle(idxs)
        epoch_losses: List[float] = []
        for i in idxs:
            doc = train_docs[i]
            opt.zero_grad(set_to_none=True)
            leaf_losses: List[torch.Tensor] = []
            idem_losses: List[torch.Tensor] = []
            c3_losses: List[torch.Tensor] = []
            c3_state_losses: List[torch.Tensor] = []

            leaf_states: List[torch.Tensor] = []
            leaf_counts: List[np.ndarray] = []

            for counts in doc.leaf_counts:
                feats = _counts_to_features(
                    counts,
                    use_log1p=bool(config.use_log1p),
                    normalize=bool(config.normalize_counts),
                )
                x = torch.tensor(feats, dtype=torch.float32, device=device)
                state = encoder(x)
                logits = readout(state)
                target = _bayes_posterior(counts, log_probs=log_probs, log_prior=log_prior)
                leaf_losses.append(_soft_ce(logits.unsqueeze(0), target))

                idem_state = merger(state, state)
                idem_logits = readout(idem_state)
                pred = torch.softmax(logits, dim=-1)
                idem_pred = torch.softmax(idem_logits, dim=-1)
                idem_losses.append(F.mse_loss(idem_pred, pred.detach(), reduction="mean"))

                leaf_states.append(state)
                leaf_counts.append(counts)

            if len(leaf_states) >= 2:
                n_internal = int(len(leaf_states) - 1)
                n_audit = audit_sample_count(
                    n_internal,
                    policy=str(config.audit_policy),
                    fixed_nodes=int(config.audit_fixed_nodes),
                    fraction=float(config.audit_fraction),
                    scale=float(config.audit_scale),
                )
                if n_audit <= 0:
                    audit_indices: Optional[set[int]] = set()
                elif n_audit >= n_internal:
                    audit_indices = None
                else:
                    audit_indices = set(rng.sample(range(n_internal), k=int(n_audit)))

                pred_level = list(leaf_states)
                count_level = list(leaf_counts)
                merge_idx = 0
                while len(pred_level) > 1:
                    nxt_states: List[torch.Tensor] = []
                    nxt_counts: List[np.ndarray] = []
                    i0 = 0
                    while i0 < len(pred_level):
                        if i0 + 1 >= len(pred_level):
                            nxt_states.append(pred_level[i0])
                            nxt_counts.append(count_level[i0])
                            i0 += 1
                            continue
                        merged_state = merger(pred_level[i0], pred_level[i0 + 1])
                        merged_counts = count_level[i0] + count_level[i0 + 1]
                        if audit_indices is None or merge_idx in audit_indices:
                            target = _bayes_posterior(
                                merged_counts, log_probs=log_probs, log_prior=log_prior
                            )
                            merged_logits = readout(merged_state)
                            c3_losses.append(_soft_ce(merged_logits.unsqueeze(0), target))
                            if float(config.c3_state_weight) > 0.0:
                                feats = _counts_to_features(
                                    merged_counts,
                                    use_log1p=bool(config.use_log1p),
                                    normalize=bool(config.normalize_counts),
                                )
                                target_state = encoder(
                                    torch.tensor(
                                        feats, dtype=torch.float32, device=device
                                    )
                                )
                                c3_state_losses.append(
                                    F.mse_loss(merged_state, target_state, reduction="mean")
                                )
                        merge_idx += 1
                        nxt_states.append(merged_state)
                        nxt_counts.append(merged_counts)
                        i0 += 2
                    pred_level = nxt_states
                    count_level = nxt_counts

            leaf_loss = torch.stack(leaf_losses).mean() if leaf_losses else 0.0
            idem_loss = torch.stack(idem_losses).mean() if idem_losses else 0.0
            c3_loss = torch.stack(c3_losses).mean() if c3_losses else 0.0
            c3_state_loss = (
                torch.stack(c3_state_losses).mean() if c3_state_losses else 0.0
            )

            total = (
                float(config.leaf_weight) * leaf_loss
                + float(config.c2_weight) * idem_loss
                + float(config.c3_weight) * c3_loss
                + float(config.c3_state_weight) * c3_state_loss
            )
            total.backward()
            if float(config.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters())
                    + list(merger.parameters())
                    + list(readout.parameters()),
                    float(config.grad_clip_norm),
                )
            opt.step()
            epoch_losses.append(float(total.detach().cpu()))

        train_loss_final = float(np.mean(np.array(epoch_losses, dtype=np.float64)))

    return train_loss_final


@torch.no_grad()
def evaluate_model(
    encoder: Encoder,
    merger: Merger,
    readout: Readout,
    docs: Sequence[HashedDoc],
    *,
    config: HashedClassificationConfig,
    log_probs: np.ndarray,
    log_prior: np.ndarray,
    device: torch.device,
) -> Tuple[HonestyStats, HonestyStats, HonestyStats, float, float]:
    encoder.to(device)
    merger.to(device)
    readout.to(device)
    encoder.eval()
    merger.eval()
    readout.eval()

    c1_disc: List[float] = []
    c2_disc: List[float] = []
    c3_disc: List[float] = []

    leaf_correct = 0
    leaf_total = 0
    root_correct = 0
    root_total = 0

    for doc in docs:
        leaf_states: List[torch.Tensor] = []
        leaf_counts: List[np.ndarray] = []

        for counts in doc.leaf_counts:
            feats = _counts_to_features(
                counts,
                use_log1p=bool(config.use_log1p),
                normalize=bool(config.normalize_counts),
            )
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            state = encoder(x)
            logits = readout(state)
            pred = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            target = _bayes_posterior(counts, log_probs=log_probs, log_prior=log_prior)
            c1_disc.append(_l1_discrepancy(pred, target))

            idem_state = merger(state, state)
            idem_logits = readout(idem_state)
            idem_pred = torch.softmax(idem_logits, dim=-1).detach().cpu().numpy()
            c2_disc.append(_l1_discrepancy(idem_pred, pred))

            leaf_correct += int(int(np.argmax(pred)) == int(doc.label))
            leaf_total += 1

            leaf_states.append(state)
            leaf_counts.append(counts)

        # Merge for root
        if len(leaf_states) == 0:
            continue
        pred_level = list(leaf_states)
        count_level = list(leaf_counts)
        while len(pred_level) > 1:
            nxt_states: List[torch.Tensor] = []
            nxt_counts: List[np.ndarray] = []
            i0 = 0
            while i0 < len(pred_level):
                if i0 + 1 >= len(pred_level):
                    nxt_states.append(pred_level[i0])
                    nxt_counts.append(count_level[i0])
                    i0 += 1
                    continue
                merged_state = merger(pred_level[i0], pred_level[i0 + 1])
                merged_counts = count_level[i0] + count_level[i0 + 1]
                target = _bayes_posterior(
                    merged_counts, log_probs=log_probs, log_prior=log_prior
                )
                merged_logits = readout(merged_state)
                merged_pred = torch.softmax(merged_logits, dim=-1).detach().cpu().numpy()
                c3_disc.append(_l1_discrepancy(merged_pred, target))

                nxt_states.append(merged_state)
                nxt_counts.append(merged_counts)
                i0 += 2
            pred_level = nxt_states
            count_level = nxt_counts

        root_logits = readout(pred_level[0])
        root_pred = torch.softmax(root_logits, dim=-1).detach().cpu().numpy()
        root_correct += int(int(np.argmax(root_pred)) == int(doc.label))
        root_total += 1

    leaf_acc = float(leaf_correct) / float(leaf_total) if leaf_total > 0 else 0.0
    root_acc = float(root_correct) / float(root_total) if root_total > 0 else 0.0

    thresh = float(config.discrepancy_threshold)
    return _stats(c1_disc, thresh), _stats(c2_disc, thresh), _stats(
        c3_disc, thresh
    ), leaf_acc, root_acc


def run_hashed_classification_experiment(
    config: HashedClassificationConfig,
) -> HashedClassificationSummary:
    _set_global_seed(int(config.seed))

    if int(config.torch_threads) > 0:
        try:
            torch.set_num_threads(int(config.torch_threads))
        except RuntimeError:
            pass
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(int(config.torch_threads))
            except RuntimeError:
                pass

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

    rng = np.random.default_rng(int(config.seed))
    class_probs = _make_class_distributions(
        n_classes=int(config.n_classes),
        vocab_size=int(config.vocab_size),
        alpha=float(config.dirichlet_alpha),
        rng=rng,
    )
    hashed_probs = _hash_class_probs(class_probs, hash_size=int(config.hash_size))
    log_probs = np.log(hashed_probs)
    log_prior = np.log(np.full(int(config.n_classes), 1.0 / float(config.n_classes)))

    docs = generate_docs(config)
    train_docs = docs[: int(config.train_docs)]
    test_docs = docs[int(config.train_docs) :]

    encoder = Encoder(
        input_dim=int(config.hash_size),
        hidden_dim=int(config.hidden_dim),
        state_dim=int(config.state_dim),
    )
    merger = Merger(state_dim=int(config.state_dim), hidden_dim=int(config.merger_hidden_dim))
    readout = Readout(state_dim=int(config.state_dim), n_classes=int(config.n_classes))

    train_loss_final = train_model(
        encoder,
        merger,
        readout,
        train_docs,
        config=config,
        log_probs=log_probs,
        log_prior=log_prior,
        device=device,
    )

    c1, c2, c3, leaf_acc, root_acc = evaluate_model(
        encoder,
        merger,
        readout,
        test_docs,
        config=config,
        log_probs=log_probs,
        log_prior=log_prior,
        device=device,
    )

    cfg_dict = asdict(config)
    cfg_dict["device_used"] = str(device)
    if device.type == "cuda":
        cfg_dict["cuda_current_device"] = int(torch.cuda.current_device())
        cfg_dict["cuda_device_name"] = str(
            torch.cuda.get_device_name(torch.cuda.current_device())
        )

    return HashedClassificationSummary(
        config=cfg_dict,
        train_loss_final=float(train_loss_final),
        leaf_accuracy=float(leaf_acc),
        root_accuracy=float(root_acc),
        c1=c1,
        c2=c2,
        c3=c3,
    )


__all__ = [
    "HashedClassificationConfig",
    "HashedClassificationSummary",
    "HashedDoc",
    "run_hashed_classification_experiment",
]
