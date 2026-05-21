"""
Key-value ledger honesty simulation.

This simulation targets OPS-style honesty checks in a deterministic setting:
- C1 (Sufficiency): leaf summary preserves the oracle state.
- C2 (Idempotence): re-summarizing a summary is stable.
- C3 (Merge consistency): merging summaries matches summarizing concatenation.

We model a ledger as a sequence of (key, value) updates. The oracle state
is the last value per key (or missing if unseen). Leaves are variable-length
windows; merges use right-over-left override per key.
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
        "PyTorch is required for ledger honesty simulations. "
        "Install with: pip install torch>=2.0.0"
    ) from e

from treepo._research.tree.learned_sketch_simulation import audit_sample_count


Update = Tuple[int, int]


@dataclass(frozen=True)
class LedgerDocument:
    updates: Tuple[Update, ...]
    leaves: Tuple[Tuple[Update, ...], ...]


@dataclass(frozen=True)
class LedgerHonestyConfig:
    num_keys: int = 16
    num_values: int = 8
    key_zipf_alpha: float = 1.1
    min_updates: int = 64
    max_updates: int = 256
    min_leaf_updates: int = 8
    max_leaf_updates: int = 32
    train_docs: int = 120
    test_docs: int = 40
    emb_dim: int = 32
    hidden_dim: int = 64
    merger_hidden_dim: int = 32
    n_epochs: int = 6
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    c1_weight: float = 1.0
    c2_weight: float = 0.5
    c3_weight: float = 1.0
    audit_policy: str = "all"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 1.0
    audit_scale: float = 1.0
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
class LedgerHonestySummary:
    config: Dict[str, object]
    train_loss_final: float
    c1: HonestyStats
    c2: HonestyStats
    c3: HonestyStats

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "train_loss_final": self.train_loss_final,
            "c1": asdict(self.c1),
            "c2": asdict(self.c2),
            "c3": asdict(self.c3),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _zipf_probs(n: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, int(n) + 1, dtype=np.float64)
    weights = np.power(ranks, -float(alpha))
    probs = weights / float(weights.sum())
    return probs.astype(np.float64, copy=False)


def generate_ledger_documents(config: LedgerHonestyConfig, *, seed: int) -> Tuple[LedgerDocument, ...]:
    if config.min_updates <= 0 or config.max_updates < config.min_updates:
        raise ValueError("require 0 < min_updates <= max_updates")
    if config.min_leaf_updates <= 0 or config.max_leaf_updates < config.min_leaf_updates:
        raise ValueError("require 0 < min_leaf_updates <= max_leaf_updates")
    if config.num_keys <= 1:
        raise ValueError("num_keys must be >= 2")
    if config.num_values <= 1:
        raise ValueError("num_values must be >= 2")

    rng = np.random.default_rng(int(seed))
    key_probs = _zipf_probs(int(config.num_keys), float(config.key_zipf_alpha))
    docs: List[LedgerDocument] = []
    n_docs = int(config.train_docs) + int(config.test_docs)

    for _ in range(n_docs):
        n_updates = int(rng.integers(int(config.min_updates), int(config.max_updates) + 1))
        keys = rng.choice(int(config.num_keys), size=n_updates, replace=True, p=key_probs)
        values = rng.integers(0, int(config.num_values), size=n_updates)
        updates = [(int(k), int(v)) for k, v in zip(keys, values)]

        leaves: List[Tuple[Update, ...]] = []
        idx = 0
        while idx < n_updates:
            leaf_len = int(
                rng.integers(int(config.min_leaf_updates), int(config.max_leaf_updates) + 1)
            )
            leaf = updates[idx : idx + leaf_len]
            leaves.append(tuple(leaf))
            idx += leaf_len

        docs.append(LedgerDocument(updates=tuple(updates), leaves=tuple(leaves)))

    return tuple(docs)


def oracle_summary(
    updates: Sequence[Update],
    *,
    num_keys: int,
    missing_value: int,
) -> np.ndarray:
    summary = np.full(int(num_keys), int(missing_value), dtype=np.int64)
    for key, value in updates:
        summary[int(key)] = int(value)
    return summary


def merge_oracle_summaries(
    left: np.ndarray,
    right: np.ndarray,
    *,
    missing_value: int,
) -> np.ndarray:
    merged = np.array(left, copy=True)
    mask = right != int(missing_value)
    merged[mask] = right[mask]
    return merged


def summary_to_updates(summary: Sequence[int]) -> Tuple[Update, ...]:
    return tuple((int(k), int(v)) for k, v in enumerate(summary))


class LedgerEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_keys: int,
        value_vocab: int,
        emb_dim: int,
        hidden_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.num_keys = int(num_keys)
        self.value_vocab = int(value_vocab)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)

        self.key_emb = nn.Embedding(self.num_keys, self.emb_dim)
        self.val_emb = nn.Embedding(self.value_vocab, self.emb_dim)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.emb_dim)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Linear(self.hidden_dim, self.num_keys * self.value_vocab)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = keys.shape
        pos_ids = torch.arange(seq_len, device=keys.device).unsqueeze(0).expand(batch, seq_len)
        pos_ids = torch.clamp(pos_ids, max=self.max_seq_len - 1)
        emb = self.key_emb(keys) + self.val_emb(values) + self.pos_emb(pos_ids)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = h.squeeze(0)
        logits = self.head(h)
        return logits.view(batch, self.num_keys, self.value_vocab)


class LedgerMerger(nn.Module):
    def __init__(self, *, value_vocab: int, hidden_dim: int) -> None:
        super().__init__()
        self.value_vocab = int(value_vocab)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * self.value_vocab, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.value_vocab),
        )

    def forward(self, left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
        if left_logits.shape != right_logits.shape:
            raise ValueError("left_logits and right_logits must have same shape")
        x = torch.cat([left_logits, right_logits], dim=-1)
        out = self.net(x)
        return out


def _updates_to_tensors(
    updates: Sequence[Update],
    *,
    num_keys: int,
    missing_value: int,
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = int(min(len(updates), max_seq_len))
    keys = torch.zeros((1, max_seq_len), dtype=torch.long, device=device)
    values = torch.full(
        (1, max_seq_len), int(missing_value), dtype=torch.long, device=device
    )
    if seq_len > 0:
        for i, (k, v) in enumerate(updates[:seq_len]):
            keys[0, i] = int(k)
            values[0, i] = int(v)
    lengths = torch.tensor([max(1, seq_len)], dtype=torch.long, device=device)
    return keys, values, lengths


def _encode_updates(
    encoder: LedgerEncoder,
    updates: Sequence[Update],
    *,
    num_keys: int,
    missing_value: int,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    keys, values, lengths = _updates_to_tensors(
        updates,
        num_keys=num_keys,
        missing_value=missing_value,
        max_seq_len=max_seq_len,
        device=device,
    )
    logits = encoder(keys, values, lengths)
    return logits.squeeze(0)


def _summary_ce(logits: torch.Tensor, target: np.ndarray) -> torch.Tensor:
    target_t = torch.tensor(target, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target_t, reduction="mean")


def _summary_argmax(logits: torch.Tensor) -> np.ndarray:
    return logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)


def _discrepancy(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("discrepancy inputs must have same shape")
    return float(np.mean(a != b))


def train_ledger_model(
    encoder: LedgerEncoder,
    merger: LedgerMerger,
    train_docs: Sequence[LedgerDocument],
    *,
    config: LedgerHonestyConfig,
    device: torch.device,
) -> float:
    encoder.to(device)
    merger.to(device)
    encoder.train()
    merger.train()

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(merger.parameters()),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    missing_value = int(config.num_values)
    max_seq_len = int(max(config.max_updates, config.num_keys))
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
            merge_losses: List[torch.Tensor] = []

            leaf_preds: List[torch.Tensor] = []
            leaf_oracles: List[np.ndarray] = []

            for leaf in doc.leaves:
                oracle = oracle_summary(
                    leaf,
                    num_keys=int(config.num_keys),
                    missing_value=missing_value,
                )
                logits = _encode_updates(
                    encoder,
                    leaf,
                    num_keys=int(config.num_keys),
                    missing_value=missing_value,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                leaf_losses.append(_summary_ce(logits, oracle))

                idem_updates = summary_to_updates(oracle)
                idem_logits = _encode_updates(
                    encoder,
                    idem_updates,
                    num_keys=int(config.num_keys),
                    missing_value=missing_value,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                idem_losses.append(_summary_ce(idem_logits, oracle))

                leaf_preds.append(logits)
                leaf_oracles.append(oracle)

            if len(leaf_preds) >= 2:
                n_internal = int(len(leaf_preds) - 1)
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

                pred_level = list(leaf_preds)
                oracle_level = list(leaf_oracles)
                merge_idx = 0
                while len(pred_level) > 1:
                    nxt_preds: List[torch.Tensor] = []
                    nxt_oracles: List[np.ndarray] = []
                    i0 = 0
                    while i0 < len(pred_level):
                        if i0 + 1 >= len(pred_level):
                            nxt_preds.append(pred_level[i0])
                            nxt_oracles.append(oracle_level[i0])
                            i0 += 1
                            continue
                        merged_logits = merger(pred_level[i0], pred_level[i0 + 1])
                        merged_oracle = merge_oracle_summaries(
                            oracle_level[i0],
                            oracle_level[i0 + 1],
                            missing_value=missing_value,
                        )
                        if audit_indices is None or merge_idx in audit_indices:
                            merge_losses.append(_summary_ce(merged_logits, merged_oracle))
                        merge_idx += 1
                        nxt_preds.append(merged_logits)
                        nxt_oracles.append(merged_oracle)
                        i0 += 2
                    pred_level = nxt_preds
                    oracle_level = nxt_oracles

            leaf_loss = torch.stack(leaf_losses).mean() if leaf_losses else 0.0
            idem_loss = torch.stack(idem_losses).mean() if idem_losses else 0.0
            merge_loss = torch.stack(merge_losses).mean() if merge_losses else 0.0
            total = (
                float(config.c1_weight) * leaf_loss
                + float(config.c2_weight) * idem_loss
                + float(config.c3_weight) * merge_loss
            )
            total.backward()
            if float(config.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(merger.parameters()),
                    float(config.grad_clip_norm),
                )
            opt.step()
            epoch_losses.append(float(total.detach().cpu()))

        train_loss_final = float(np.mean(np.array(epoch_losses, dtype=np.float64)))

    return train_loss_final


@torch.no_grad()
def evaluate_honesty(
    encoder: LedgerEncoder,
    merger: LedgerMerger,
    docs: Sequence[LedgerDocument],
    *,
    config: LedgerHonestyConfig,
    device: torch.device,
    discrepancy_threshold: float = 0.1,
) -> Tuple[HonestyStats, HonestyStats, HonestyStats]:
    encoder.to(device)
    merger.to(device)
    encoder.eval()
    merger.eval()

    missing_value = int(config.num_values)
    max_seq_len = int(max(config.max_updates, config.num_keys))

    c1_disc: List[float] = []
    c2_disc: List[float] = []
    c3_disc: List[float] = []

    for doc in docs:
        leaf_preds: List[np.ndarray] = []
        leaf_logits: List[torch.Tensor] = []
        leaf_updates: List[Tuple[Update, ...]] = []

        for leaf in doc.leaves:
            oracle = oracle_summary(
                leaf,
                num_keys=int(config.num_keys),
                missing_value=missing_value,
            )
            logits = _encode_updates(
                encoder,
                leaf,
                num_keys=int(config.num_keys),
                missing_value=missing_value,
                max_seq_len=max_seq_len,
                device=device,
            )
            pred = _summary_argmax(logits)
            c1_disc.append(_discrepancy(pred, oracle))

            idem_updates = summary_to_updates(pred)
            idem_logits = _encode_updates(
                encoder,
                idem_updates,
                num_keys=int(config.num_keys),
                missing_value=missing_value,
                max_seq_len=max_seq_len,
                device=device,
            )
            idem_pred = _summary_argmax(idem_logits)
            c2_disc.append(_discrepancy(idem_pred, pred))

            leaf_preds.append(pred)
            leaf_logits.append(logits)
            leaf_updates.append(tuple(leaf))

        if len(leaf_preds) >= 2:
            pred_level = list(leaf_preds)
            logit_level = list(leaf_logits)
            update_level = list(leaf_updates)
            while len(pred_level) > 1:
                nxt_preds: List[np.ndarray] = []
                nxt_logits: List[torch.Tensor] = []
                nxt_updates: List[Tuple[Update, ...]] = []
                i0 = 0
                while i0 < len(pred_level):
                    if i0 + 1 >= len(pred_level):
                        nxt_preds.append(pred_level[i0])
                        nxt_logits.append(logit_level[i0])
                        nxt_updates.append(update_level[i0])
                        i0 += 1
                        continue
                    merged_logits = merger(logit_level[i0], logit_level[i0 + 1])
                    merged_pred = _summary_argmax(merged_logits)

                    concat_updates = update_level[i0] + update_level[i0 + 1]
                    concat_logits = _encode_updates(
                        encoder,
                        concat_updates,
                        num_keys=int(config.num_keys),
                        missing_value=missing_value,
                        max_seq_len=max_seq_len,
                        device=device,
                    )
                    concat_pred = _summary_argmax(concat_logits)
                    c3_disc.append(_discrepancy(merged_pred, concat_pred))

                    nxt_preds.append(merged_pred)
                    nxt_logits.append(merged_logits)
                    nxt_updates.append(concat_updates)
                    i0 += 2
                pred_level = nxt_preds
                logit_level = nxt_logits
                update_level = nxt_updates

    def _stats(xs: List[float]) -> HonestyStats:
        if len(xs) == 0:
            return HonestyStats(mean_discrepancy=0.0, violation_rate=0.0, n=0)
        arr = np.array(xs, dtype=np.float64)
        mean = float(np.mean(arr))
        viol = float(np.mean(arr > float(discrepancy_threshold)))
        return HonestyStats(mean_discrepancy=mean, violation_rate=viol, n=int(len(xs)))

    return _stats(c1_disc), _stats(c2_disc), _stats(c3_disc)


def run_ledger_honesty_experiment(config: LedgerHonestyConfig) -> LedgerHonestySummary:
    _set_global_seed(int(config.seed))

    if int(config.torch_threads) > 0:
        torch.set_num_threads(int(config.torch_threads))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(int(config.torch_threads))

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

    docs = generate_ledger_documents(config, seed=int(config.seed))
    train_docs = docs[: int(config.train_docs)]
    test_docs = docs[int(config.train_docs) :]

    value_vocab = int(config.num_values) + 1
    max_seq_len = int(max(config.max_updates, config.num_keys))
    encoder = LedgerEncoder(
        num_keys=int(config.num_keys),
        value_vocab=value_vocab,
        emb_dim=int(config.emb_dim),
        hidden_dim=int(config.hidden_dim),
        max_seq_len=max_seq_len,
    )
    merger = LedgerMerger(value_vocab=value_vocab, hidden_dim=int(config.merger_hidden_dim))

    train_loss_final = train_ledger_model(
        encoder,
        merger,
        train_docs,
        config=config,
        device=device,
    )

    c1, c2, c3 = evaluate_honesty(
        encoder,
        merger,
        test_docs,
        config=config,
        device=device,
    )

    cfg_dict = asdict(config)
    cfg_dict["device_used"] = str(device)
    if device.type == "cuda":
        cfg_dict["cuda_current_device"] = int(torch.cuda.current_device())
        cfg_dict["cuda_device_name"] = str(
            torch.cuda.get_device_name(torch.cuda.current_device())
        )

    return LedgerHonestySummary(
        config=cfg_dict,
        train_loss_final=float(train_loss_final),
        c1=c1,
        c2=c2,
        c3=c3,
    )


__all__ = [
    "LedgerDocument",
    "LedgerHonestyConfig",
    "LedgerHonestySummary",
    "LedgerEncoder",
    "LedgerMerger",
    "generate_ledger_documents",
    "run_ledger_honesty_experiment",
]
