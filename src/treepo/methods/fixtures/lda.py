"""Synthetic LDA topic-mixture fixture."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Tuple

import numpy as np

from treepo.methods.fixtures.common import (
    exact_score_metadata,
    int_tuple,
    leaf_slices,
    require_cuda_torch,
)


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
    leaf_unit_count: int = 16,
    doc_unit_kind: str = "token",
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

    if n_trees <= 0 or n_topics <= 1 or doc_tokens <= 0 or leaf_unit_count <= 0:
        raise ValueError(
            "n_trees, doc_tokens, leaf_unit_count must be positive and n_topics > 1"
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
            leaf_unit_count=int(leaf_unit_count),
            doc_unit_kind=str(doc_unit_kind),
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
        topic_rng.dirichlet(
            np.full(int(vocabulary_size), float(topic_word_concentration))
        ).astype(float)
        for _ in range(int(n_topics))
    )
    topic_word_distributions = tuple(
        tuple(float(x) for x in probs.tolist()) for probs in topic_word_probs
    )
    trees: List[LDATopicTree] = []
    for tree_idx in range(int(n_trees)):
        theta = rng.dirichlet(
            np.full(int(n_topics), float(doc_topic_concentration))
        ).astype(float)
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
        trees.append(
            _lda_topic_tree_from_arrays(
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
                leaf_unit_count=int(leaf_unit_count),
                doc_unit_kind=str(doc_unit_kind),
                vocabulary_size=int(vocabulary_size),
                doc_topic_concentration=float(doc_topic_concentration),
                topic_word_concentration=float(topic_word_concentration),
            )
        )
    return trees


def _make_lda_topic_trees_torch(
    *,
    n_trees: int,
    n_topics: int,
    doc_tokens: int,
    leaf_unit_count: int,
    doc_unit_kind: str,
    vocabulary_size: int,
    doc_topic_concentration: float,
    topic_word_concentration: float,
    target_topic: int,
    topic_seed: int,
    seed: int,
    split: str,
    generation_device: str,
) -> List[LDATopicTree]:
    torch, device, rng_devices = require_cuda_torch(
        generation_device, fixture_name="LDA"
    )

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(topic_seed))
        torch.cuda.manual_seed_all(int(topic_seed))
        topic_prior = torch.full(
            (int(vocabulary_size),),
            float(topic_word_concentration),
            dtype=torch.float32,
            device=device,
        )
        topic_word_probs_t = torch.distributions.Dirichlet(topic_prior).sample(
            (int(n_topics),)
        )

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
                sampled = torch.multinomial(
                    topic_word_probs_t[int(topic_idx)], count, replacement=True
                )
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
            leaf_unit_count=int(leaf_unit_count),
            doc_unit_kind=str(doc_unit_kind),
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
    leaf_unit_count: int,
    doc_unit_kind: str,
    vocabulary_size: int,
    doc_topic_concentration: float,
    topic_word_concentration: float,
) -> LDATopicTree:
    token_values = int_tuple(tokens)
    topic_values = int_tuple(topics)
    leaves: List[_LDALeaf] = []
    for start, end in leaf_slices(doc_tokens, leaf_unit_count):
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
            **exact_score_metadata(target, target_scale="probability"),
            "target_topic": int(target_topic),
            "topic_seed": int(topic_seed),
            "n_topics": int(n_topics),
            "doc_tokens": int(doc_tokens),
            "doc_unit_kind": str(doc_unit_kind),
            "leaf_unit_count": int(leaf_unit_count),
            "vocabulary_size": int(vocabulary_size),
            "doc_topic_concentration": float(doc_topic_concentration),
            "topic_word_concentration": float(topic_word_concentration),
            "topic_proportions": proportions,
            "dirichlet_topic_proportions": theta,
            "topic_word_distributions": topic_word_distributions,
            **topic_fields,
        },
    )


def lda_tree_records(
    trees: List[LDATopicTree],
    *,
    vector_labels: bool = False,
) -> List[Any]:
    """Convert LDA fixture trees into canonical ``TreeRecord`` artifacts.

    Each record is a flat star: token leaves in position order under a
    labeled root. Leaves carry realized gold on the same scale the family
    reads out: the target-topic proportion by default, or the full topic
    vector with ``vector_labels=True`` (matching ``target_dim=n_topics``
    training). The full per-leaf topic mix is always in metadata.
    ``tree_id`` matches ``metadata["tree_id"]``, which is also the prefix of
    statistic law-row ids.
    """

    from treepo.tree import TreeRecord

    records: List[Any] = []
    for tree in trees or ():
        metadata = dict(tree.metadata or {})
        target_topic = int(metadata.get("target_topic", 0))
        n_topics = int(metadata.get("n_topics", 0)) or (
            max(tree.topics) + 1 if tree.topics else 1
        )
        root_label: Any = (
            [float(x) for x in tree.topic_proportions]
            if vector_labels
            else metadata.get("teacher_score_native")
        )
        nodes: List[dict] = []
        for idx, leaf in enumerate(tree.leaves):
            topics = [int(t) for t in leaf.topics]
            counts = [0] * n_topics
            for topic in topics:
                counts[topic] += 1
            total = max(1, len(topics))
            proportions = [count / total for count in counts]
            nodes.append(
                {
                    "node_id": f"leaf_{idx}",
                    "unit_type": "leaf",
                    "text": " ".join(str(int(token)) for token in leaf.tokens),
                    "parent_id": "root",
                    "level": 0,
                    "position": idx,
                    "label": proportions if vector_labels else proportions[target_topic],
                    "metadata": {
                        "topics": topics,
                        "leaf_topic_proportions": proportions,
                    },
                }
            )
        nodes.append(
            {
                "node_id": "root",
                "unit_type": "root",
                "level": 1,
                "label": root_label,
                "metadata": {
                    "topic_proportions": [float(x) for x in tree.topic_proportions],
                },
            }
        )
        records.append(
            TreeRecord(
                tree_id=str(metadata.get("tree_id") or "lda"),
                nodes=nodes,
                root_label=root_label,
                metadata=metadata,
            )
        )
    return records


__all__ = ["LDATopicTree", "lda_tree_records", "make_lda_topic_trees"]
