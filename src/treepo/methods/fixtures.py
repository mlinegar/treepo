"""Lightweight fixture builders for exercising ``treepo.methods.run`` / ``fit``.

The package fixtures are small deterministic tree-shaped datasets for HLL,
synthetic LDA, and Markov method checks.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np


# Fixture caches.
#
# We memoize the public fixture builders on their full argument tuples.
# The fixture has hashable inputs (primitives), so :func:`functools.lru_cache`
# is sufficient — no on-disk cache, no fcntl locking, no pickle.
# ``maxsize=32`` is generous for typical grids that vary along ~2 fixture axes.
#
# Mutation safety: the trees returned are treated as read-only by the built-in
# oracle and evaluator. If a custom family mutates trees, call
# ``make_*.cache_clear()`` between cells.


# --------------------------------------------------------------------------- #
# HLL / cardinality fixture
# --------------------------------------------------------------------------- #


@dataclass
class _HLLLeaf:
    tokens: Tuple[int, ...]


@dataclass
class HLLTokenTree:
    """Tree-shaped object with per-leaf token sequences and an exact
    distinct-count teacher on metadata."""

    leaves: Tuple[_HLLLeaf, ...]
    tokens: Tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# HLL token-tree fixture
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=32)
def make_hll_token_trees(
    *,
    n_trees: int = 8,
    leaves_per_tree: int = 4,
    leaf_token_count: int = 16,
    vocabulary_size: int = 64,
    seed: int = 0,
    split: str = "test",
) -> List[HLLTokenTree]:
    """Generate deterministic token trees with precomputed exact unique counts."""
    if n_trees <= 0 or leaves_per_tree <= 0 or leaf_token_count <= 0:
        raise ValueError("n_trees, leaves_per_tree, leaf_token_count must be positive")
    rng = np.random.default_rng(int(seed))
    trees: List[HLLTokenTree] = []
    for _ in range(int(n_trees)):
        leaves: List[_HLLLeaf] = []
        all_tokens: List[int] = []
        for _leaf_idx in range(int(leaves_per_tree)):
            leaf_tokens = rng.integers(
                low=0, high=int(vocabulary_size), size=int(leaf_token_count)
            ).tolist()
            leaves.append(_HLLLeaf(tokens=tuple(int(t) for t in leaf_tokens)))
            all_tokens.extend(int(t) for t in leaf_tokens)
        exact_unique = int(len(set(all_tokens)))
        trees.append(
            HLLTokenTree(
                leaves=tuple(leaves),
                tokens=tuple(all_tokens),
                metadata={
                    "split": split,
                    "teacher_score_1_7": float(exact_unique),
                    "teacher_score_native": float(exact_unique),
                    "expert_score_1_7": float(exact_unique),
                    "expert_score_native": float(exact_unique),
                    "expert_target_scale": "raw",
                    "expert_score_for_objective": float(exact_unique),
                },
            )
        )
    return trees


# --------------------------------------------------------------------------- #
# LDA topic-mixture fixture
# --------------------------------------------------------------------------- #


@dataclass
class _LDALeaf:
    tokens: Tuple[int, ...]
    topics: Tuple[int, ...]


@dataclass
class LDATopicTree:
    """Tree-shaped LDA document with exact latent topic proportions."""

    leaves: Tuple[_LDALeaf, ...]
    tokens: Tuple[int, ...]
    topics: Tuple[int, ...]
    topic_proportions: Tuple[float, ...]
    dirichlet_topic_proportions: Tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@functools.lru_cache(maxsize=32)
def make_lda_topic_trees(
    *,
    n_trees: int = 8,
    n_topics: int = 3,
    doc_tokens: int = 96,
    leaf_token_count: int = 16,
    vocabulary_size: int = 60,
    doc_topic_concentration: float = 0.7,
    topic_word_concentration: float = 0.3,
    target_topic: int = 0,
    topic_seed: int = 0,
    seed: int = 0,
    split: str = "test",
    generation_device: str = "cpu",
) -> List[LDATopicTree]:
    """Generate deterministic LDA documents with exact topic-coordinate targets.

    Each topic has a full-vocabulary word distribution drawn from a Dirichlet.
    Token topics are sampled from each document's Dirichlet topic mixture, and
    the realized latent topic proportions are stored as the exact target.
    """

    if n_trees <= 0 or n_topics <= 1 or doc_tokens <= 0 or leaf_token_count <= 0:
        raise ValueError(
            "n_trees, doc_tokens, leaf_token_count must be positive and n_topics > 1"
        )
    if vocabulary_size < n_topics:
        raise ValueError("vocabulary_size must be at least n_topics")
    if not 0 <= int(target_topic) < int(n_topics):
        raise ValueError("target_topic must be in [0, n_topics)")
    if doc_topic_concentration <= 0.0 or topic_word_concentration <= 0.0:
        raise ValueError("Dirichlet concentration parameters must be positive")

    if str(generation_device).startswith("cuda"):
        return _make_lda_topic_trees_torch(
            n_trees=int(n_trees),
            n_topics=int(n_topics),
            doc_tokens=int(doc_tokens),
            leaf_token_count=int(leaf_token_count),
            vocabulary_size=int(vocabulary_size),
            doc_topic_concentration=float(doc_topic_concentration),
            topic_word_concentration=float(topic_word_concentration),
            target_topic=int(target_topic),
            topic_seed=int(topic_seed),
            seed=int(seed),
            split=str(split),
            generation_device=str(generation_device),
        )

    topic_rng = np.random.default_rng(int(topic_seed))
    rng = np.random.default_rng(int(seed))
    topic_word_probs = tuple(
        topic_rng.dirichlet(np.full(int(vocabulary_size), float(topic_word_concentration))).astype(float)
        for _ in range(int(n_topics))
    )
    topic_word_distributions = tuple(
        tuple(float(x) for x in probs.tolist()) for probs in topic_word_probs
    )
    trees: List[LDATopicTree] = []
    for tree_idx in range(int(n_trees)):
        theta = rng.dirichlet(np.full(int(n_topics), float(doc_topic_concentration))).astype(float)
        topics = rng.choice(int(n_topics), size=int(doc_tokens), p=theta).astype(int)
        tokens = np.empty(int(doc_tokens), dtype=np.int64)
        for topic_idx, probs in enumerate(topic_word_probs):
            mask = topics == int(topic_idx)
            count = int(mask.sum())
            if count:
                tokens[mask] = rng.choice(int(vocabulary_size), size=count, p=probs)
        counts = np.bincount(topics, minlength=int(n_topics)).astype(float)
        proportions = tuple(float(x) for x in (counts / float(doc_tokens)).tolist())
        theta_tuple = tuple(float(x) for x in theta.tolist())
        trees.append(_lda_topic_tree_from_arrays(
            tree_idx=tree_idx,
            tokens=tokens,
            topics=topics,
            proportions=proportions,
            theta=theta_tuple,
            topic_word_distributions=topic_word_distributions,
            split=split,
            seed=int(seed),
            topic_seed=int(topic_seed),
            target_topic=int(target_topic),
            n_topics=int(n_topics),
            doc_tokens=int(doc_tokens),
            leaf_token_count=int(leaf_token_count),
            vocabulary_size=int(vocabulary_size),
            doc_topic_concentration=float(doc_topic_concentration),
            topic_word_concentration=float(topic_word_concentration),
        ))
    return trees




def _make_lda_topic_trees_torch(
    *,
    n_trees: int,
    n_topics: int,
    doc_tokens: int,
    leaf_token_count: int,
    vocabulary_size: int,
    doc_topic_concentration: float,
    topic_word_concentration: float,
    target_topic: int,
    topic_seed: int,
    seed: int,
    split: str,
    generation_device: str,
) -> List[LDATopicTree]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("generation_device='cuda' requires torch") from exc

    device = torch.device(str(generation_device))
    if device.type != "cuda":
        raise ValueError("torch LDA fixture generation is only used for CUDA devices")
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise RuntimeError("generation_device='cuda' requested but CUDA is unavailable")

    rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(topic_seed))
        torch.cuda.manual_seed_all(int(topic_seed))
        topic_prior = torch.full(
            (int(vocabulary_size),),
            float(topic_word_concentration),
            dtype=torch.float32,
            device=device,
        )
        topic_word_probs_t = torch.distributions.Dirichlet(topic_prior).sample((int(n_topics),))

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        doc_prior = torch.full(
            (int(n_topics),),
            float(doc_topic_concentration),
            dtype=torch.float32,
            device=device,
        )
        theta_t = torch.distributions.Dirichlet(doc_prior).sample((int(n_trees),))
        topics_t = torch.multinomial(theta_t, int(doc_tokens), replacement=True)
        tokens_t = torch.empty_like(topics_t)
        for topic_idx in range(int(n_topics)):
            mask = topics_t == int(topic_idx)
            count = int(mask.sum().item())
            if count:
                sampled = torch.multinomial(topic_word_probs_t[int(topic_idx)], count, replacement=True)
                tokens_t[mask] = sampled

    counts_t = torch.stack(
        [(topics_t == int(topic_idx)).sum(dim=1) for topic_idx in range(int(n_topics))],
        dim=1,
    ).to(dtype=torch.float32)
    proportions_t = counts_t / float(doc_tokens)
    tokens_np = tokens_t.to(device="cpu", dtype=torch.int64).numpy()
    topics_np = topics_t.to(device="cpu", dtype=torch.int64).numpy()
    theta_np = theta_t.to(device="cpu", dtype=torch.float64).numpy()
    proportions_np = proportions_t.to(device="cpu", dtype=torch.float64).numpy()
    topic_word_probs_np = topic_word_probs_t.to(device="cpu", dtype=torch.float64).numpy()
    topic_word_distributions = tuple(
        tuple(float(x) for x in row.tolist()) for row in topic_word_probs_np
    )
    return [
        _lda_topic_tree_from_arrays(
            tree_idx=tree_idx,
            tokens=tokens_np[tree_idx],
            topics=topics_np[tree_idx],
            proportions=tuple(float(x) for x in proportions_np[tree_idx].tolist()),
            theta=tuple(float(x) for x in theta_np[tree_idx].tolist()),
            topic_word_distributions=topic_word_distributions,
            split=split,
            seed=int(seed),
            topic_seed=int(topic_seed),
            target_topic=int(target_topic),
            n_topics=int(n_topics),
            doc_tokens=int(doc_tokens),
            leaf_token_count=int(leaf_token_count),
            vocabulary_size=int(vocabulary_size),
            doc_topic_concentration=float(doc_topic_concentration),
            topic_word_concentration=float(topic_word_concentration),
        )
        for tree_idx in range(int(n_trees))
    ]


def _lda_topic_tree_from_arrays(
    *,
    tree_idx: int,
    tokens: Any,
    topics: Any,
    proportions: Tuple[float, ...],
    theta: Tuple[float, ...],
    topic_word_distributions: Tuple[Tuple[float, ...], ...],
    split: str,
    seed: int,
    topic_seed: int,
    target_topic: int,
    n_topics: int,
    doc_tokens: int,
    leaf_token_count: int,
    vocabulary_size: int,
    doc_topic_concentration: float,
    topic_word_concentration: float,
) -> LDATopicTree:
    token_values = tuple(int(x) for x in tokens)
    topic_values = tuple(int(x) for x in topics)
    leaves: List[_LDALeaf] = []
    for start in range(0, int(doc_tokens), int(leaf_token_count)):
        end = min(int(doc_tokens), start + int(leaf_token_count))
        leaves.append(
            _LDALeaf(
                tokens=tuple(token_values[start:end]),
                topics=tuple(topic_values[start:end]),
            )
        )
    target = float(proportions[int(target_topic)])
    topic_fields = {
        f"topic_{idx}_proportion": float(value)
        for idx, value in enumerate(proportions)
    }
    return LDATopicTree(
        leaves=tuple(leaves),
        tokens=token_values,
        topics=topic_values,
        topic_proportions=proportions,
        dirichlet_topic_proportions=theta,
        metadata={
            "tree_id": f"lda_{int(seed)}_{int(tree_idx)}",
            "split": split,
            "teacher_score_1_7": target,
            "teacher_score_native": target,
            "expert_score_1_7": target,
            "expert_score_native": target,
            "expert_target_scale": "probability",
            "expert_score_for_objective": target,
            "target_topic": int(target_topic),
            "topic_seed": int(topic_seed),
            "n_topics": int(n_topics),
            "doc_tokens": int(doc_tokens),
            "leaf_token_count": int(leaf_token_count),
            "vocabulary_size": int(vocabulary_size),
            "doc_topic_concentration": float(doc_topic_concentration),
            "topic_word_concentration": float(topic_word_concentration),
            "topic_proportions": proportions,
            "dirichlet_topic_proportions": theta,
            "topic_word_distributions": topic_word_distributions,
            **topic_fields,
        },
    )

# --------------------------------------------------------------------------- #
# Markov changepoint fixture
# --------------------------------------------------------------------------- #


@dataclass
class _MarkovLeaf:
    tokens: Sequence[int]
    regimes: Sequence[int]


@dataclass
class MarkovChangepointTree:
    """Tree-shaped Markov sequence with a changepoint-count teacher."""

    leaves: Tuple[_MarkovLeaf, ...]
    tokens: Sequence[int]
    regimes: Sequence[int]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@functools.lru_cache(maxsize=32)
def make_markov_changepoint_trees(
    *,
    n_trees: int = 8,
    n_states: int = 4,
    doc_tokens: int = 128,
    leaf_token_count: int = 16,
    transition_prob: float = 0.15,
    vocabulary_size: int = 256,
    seed: int = 0,
    split: str = "test",
    generation_device: str = "cpu",
) -> List[MarkovChangepointTree]:
    """Generate deterministic Markov trees with exact changepoint targets."""

    if n_trees <= 0 or n_states <= 1 or doc_tokens <= 0 or leaf_token_count <= 0:
        raise ValueError("n_trees, doc_tokens, leaf_token_count must be positive and n_states > 1")
    if not 0.0 <= float(transition_prob) <= 1.0:
        raise ValueError("transition_prob must be in [0, 1]")

    if str(generation_device).startswith("cuda"):
        return _make_markov_changepoint_trees_torch(
            n_trees=int(n_trees),
            n_states=int(n_states),
            doc_tokens=int(doc_tokens),
            leaf_token_count=int(leaf_token_count),
            transition_prob=float(transition_prob),
            vocabulary_size=int(vocabulary_size),
            seed=int(seed),
            split=str(split),
            generation_device=str(generation_device),
        )

    rng = np.random.default_rng(int(seed))
    bucket = max(1, int(vocabulary_size) // int(n_states))
    trees: List[MarkovChangepointTree] = []
    for tree_idx in range(int(n_trees)):
        regimes: List[int] = [int(rng.integers(0, int(n_states)))]
        for _ in range(1, int(doc_tokens)):
            cur = int(regimes[-1])
            if float(rng.random()) < float(transition_prob):
                step = int(rng.integers(1, int(n_states)))
                regimes.append(int((cur + step) % int(n_states)))
            else:
                regimes.append(cur)

        tokens: List[int] = []
        for state in regimes:
            low = int(state) * bucket
            high = min(int(vocabulary_size), low + bucket)
            if high <= low:
                tokens.append(int(state))
            else:
                tokens.append(int(rng.integers(low, high)))

        leaves: List[_MarkovLeaf] = []
        for start in range(0, int(doc_tokens), int(leaf_token_count)):
            end = min(int(doc_tokens), start + int(leaf_token_count))
            leaves.append(
                _MarkovLeaf(
                    tokens=tuple(int(x) for x in tokens[start:end]),
                    regimes=tuple(int(x) for x in regimes[start:end]),
                )
            )
        changepoints = sum(
            1 for left, right in zip(regimes, regimes[1:]) if int(left) != int(right)
        )
        trees.append(
            MarkovChangepointTree(
                leaves=tuple(leaves),
                tokens=tuple(tokens),
                regimes=tuple(regimes),
                metadata={
                    "tree_id": f"markov_{int(seed)}_{tree_idx}",
                    "split": split,
                    "teacher_score_1_7": float(changepoints),
                    "teacher_score_native": float(changepoints),
                    "expert_score_1_7": float(changepoints),
                    "expert_score_native": float(changepoints),
                    "expert_target_scale": "raw",
                    "expert_score_for_objective": float(changepoints),
                    "n_states": int(n_states),
                    "doc_tokens": int(doc_tokens),
                    "leaf_token_count": int(leaf_token_count),
                    "vocabulary_size": int(vocabulary_size),
                    "transition_prob": float(transition_prob),
                },
            )
        )
    return trees


def _make_markov_changepoint_trees_torch(
    *,
    n_trees: int,
    n_states: int,
    doc_tokens: int,
    leaf_token_count: int,
    transition_prob: float,
    vocabulary_size: int,
    seed: int,
    split: str,
    generation_device: str,
) -> List[MarkovChangepointTree]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("generation_device='cuda' requires torch") from exc

    device = torch.device(str(generation_device))
    if device.type != "cuda":
        raise ValueError("torch Markov fixture generation is only used for CUDA devices")
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise RuntimeError("generation_device='cuda' requested but CUDA is unavailable")

    bucket = max(1, int(vocabulary_size) // int(n_states))
    rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        initial = torch.randint(0, int(n_states), (int(n_trees), 1), device=device)
        if int(doc_tokens) > 1:
            changes = torch.rand((int(n_trees), int(doc_tokens) - 1), device=device) < float(transition_prob)
            steps = torch.randint(1, int(n_states), (int(n_trees), int(doc_tokens) - 1), device=device)
            increments = torch.where(changes, steps, torch.zeros_like(steps))
            regimes_t = torch.cat([initial, (initial + torch.cumsum(increments, dim=1)) % int(n_states)], dim=1)
        else:
            regimes_t = initial
        offsets = torch.randint(0, bucket, regimes_t.shape, device=device)
        tokens_t = regimes_t * int(bucket) + offsets
        if int(vocabulary_size) < int(n_states) * int(bucket):
            tokens_t = torch.clamp(tokens_t, max=int(vocabulary_size) - 1)
        changepoints_t = (regimes_t[:, 1:] != regimes_t[:, :-1]).sum(dim=1) if int(doc_tokens) > 1 else torch.zeros((int(n_trees),), device=device, dtype=torch.long)

    regimes_np = regimes_t.to(device="cpu", dtype=torch.int64).numpy()
    tokens_np = tokens_t.to(device="cpu", dtype=torch.int64).numpy()
    changepoints_np = changepoints_t.to(device="cpu", dtype=torch.int64).numpy()
    trees: List[MarkovChangepointTree] = []
    for tree_idx in range(int(n_trees)):
        regimes = regimes_np[tree_idx]
        tokens = tokens_np[tree_idx]
        leaves: List[_MarkovLeaf] = []
        for start in range(0, int(doc_tokens), int(leaf_token_count)):
            end = min(int(doc_tokens), start + int(leaf_token_count))
            leaves.append(
                _MarkovLeaf(
                    tokens=tokens[start:end],
                    regimes=regimes[start:end],
                )
            )
        changepoints = int(changepoints_np[tree_idx])
        trees.append(
            MarkovChangepointTree(
                leaves=tuple(leaves),
                tokens=tokens,
                regimes=regimes,
                metadata={
                    "tree_id": f"markov_{int(seed)}_{tree_idx}",
                    "split": split,
                    "teacher_score_1_7": float(changepoints),
                    "teacher_score_native": float(changepoints),
                    "expert_score_1_7": float(changepoints),
                    "expert_score_native": float(changepoints),
                    "expert_target_scale": "raw",
                    "expert_score_for_objective": float(changepoints),
                    "n_states": int(n_states),
                    "doc_tokens": int(doc_tokens),
                    "leaf_token_count": int(leaf_token_count),
                    "vocabulary_size": int(vocabulary_size),
                    "transition_prob": float(transition_prob),
                },
            )
        )
    return trees


__all__ = [
    "HLLTokenTree",
    "LDATopicTree",
    "MarkovChangepointTree",
    "make_hll_token_trees",
    "make_lda_topic_trees",
    "make_markov_changepoint_trees",
]
