"""
Learn a mergeable sketch `g` for the Segment-LDA OPS functional.

This is the "PO-side" analogue of the ridge baseline in
`segment_lda_ops_weight_recovery.py`, but with an explicit learned state and
learned merge operator trained from local oracle labels at leaves + sampled
internal nodes (C1/C3-style supervision).

CPU-only; requires `treepo[torch]`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Dict, List, Tuple

import numpy as np

from treepo.bench.learned.mergeable_sketch import (
    LearnedGTrainingConfig,
    build_docs_from_oracle,
    eval_torch_mergeable_sketch,
    train_torch_mergeable_sketch,
)
from treepo._research.bench.lda.segment_lda_ops_weight_recovery import (
    SegmentLDADoc,
    _prefix_counts,
    _oracle_from_prefix,
    build_leaf_spans,
    generate_segment_lda_docs,
    sample_sparse_oracle_weights,
    sample_topic_distributions,
)


@dataclass(frozen=True)
class LearnedSegmentLDAOpsGConfig:
    # Generator.
    n_topics: int = 8
    vocab_size: int = 256
    min_tokens: int = 192
    max_tokens: int = 192
    min_segments: int = 2
    max_segments: int = 6
    min_seg_len: int = 32
    max_seg_len: int = 128
    leaf_tokens: int = 16
    align_segments_to_leaves: bool = True
    doc_topic_concentration: float = 0.6
    topic_process: str = "segments"
    boundary_profile: str = "uniform"
    boundary_profile_strength: float = 0.0
    boundary_profile_seed: int = -1
    segment_length_power: float = 0.0

    # Topic-word distributions.
    topic_concentration: float = 0.2
    emission_mode: str = "anchored"  # "anchored" or "disjoint"
    anchor_words_per_topic: int = 16
    anchor_multiplier: float = 25.0

    # Oracle weights.
    relevant_topics: int = 2
    theta_scale: float = 1.0
    zero_diagonal: bool = True
    lambda_multiplier: float = 1.0

    # Train/eval sizes.
    train_docs: int = 256
    test_docs: int = 256

    # Learned sketch / local-law training.
    state_dim: int = 32
    hidden_dim: int = 128
    n_epochs: int = 10
    batch_docs: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    leaf_query_rate: float = 1.0
    audit_policy: str = "fraction"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 0.2
    audit_scale: float = 1.0
    include_root_query: bool = True

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
class LearnedSegmentLDAOpsGSummary:
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    weight_truth: Dict[str, object]
    training_geometry: Dict[str, object]
    metrics: Dict[str, object]

    def to_json(self) -> str:
        return json.dumps(
            {
                "config": self.config,
                "topic_meta": self.topic_meta,
                "weight_truth": self.weight_truth,
                "training_geometry": self.training_geometry,
                "metrics": self.metrics,
            },
            indent=2,
            sort_keys=True,
        )


def _leaf_topic_onehot_features(doc: SegmentLDADoc, *, n_topics: int, leaf_tokens: int) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Per-leaf features: flattened token-topic one-hots for a fixed-width leaf window."""

    z = np.asarray(list(doc.topics), dtype=np.int64)
    spans = build_leaf_spans(len(z), leaf_tokens=int(leaf_tokens))
    k = int(n_topics)
    lt = int(max(1, leaf_tokens))

    feats: List[np.ndarray] = []
    masses: List[float] = []
    for (s, e) in spans:
        seg = z[int(s) : int(e)]
        x = np.zeros((lt, k), dtype=np.float32)
        for j, tid in enumerate(seg.tolist()[:lt]):
            t = int(tid)
            if 0 <= t < k:
                x[int(j), int(t)] = 1.0
        feats.append(x.reshape(-1))
        masses.append(float(len(seg)))
    if not feats:
        feats = [np.zeros((lt * k,), dtype=np.float32)]
        spans = [(0, 0)]
        masses = [0.0]
    return np.stack(feats, axis=0), spans, np.asarray(masses, dtype=np.float32)


def run_learned_segment_lda_ops_g_experiment(config: LearnedSegmentLDAOpsGConfig) -> LearnedSegmentLDAOpsGSummary:
    # Sample topic distributions + oracle weights.
    topics_phi, topic_meta = sample_topic_distributions(
        vocab_size=int(config.vocab_size),
        n_topics=int(config.n_topics),
        topic_concentration=float(config.topic_concentration),
        emission_mode=str(config.emission_mode),
        anchor_words_per_topic=int(config.anchor_words_per_topic),
        anchor_multiplier=float(config.anchor_multiplier),
        seed=int(config.seed) + 101,
    )
    R, theta_true, W_base = sample_sparse_oracle_weights(
        n_topics=int(config.n_topics),
        relevant_topics=int(config.relevant_topics),
        theta_scale=float(config.theta_scale),
        zero_diagonal=bool(config.zero_diagonal),
        seed=int(config.seed) + 202,
    )
    lambda_multiplier = float(config.lambda_multiplier)
    w_big_true = lambda_multiplier * np.asarray(W_base, dtype=np.float64).reshape(-1)

    # Docs.
    boundary_profile_seed = int(config.boundary_profile_seed)
    if boundary_profile_seed < 0:
        boundary_profile_seed = int(config.seed) + 19
    docs_train, _train_stats = generate_segment_lda_docs(
        int(config.train_docs),
        topics=topics_phi,
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_len=int(config.min_seg_len),
        max_seg_len=int(config.max_seg_len),
        leaf_tokens=int(config.leaf_tokens),
        align_segments_to_leaves=bool(config.align_segments_to_leaves),
        doc_topic_concentration=float(config.doc_topic_concentration),
        topic_process=str(config.topic_process),
        boundary_profile=str(config.boundary_profile),
        boundary_profile_strength=float(config.boundary_profile_strength),
        boundary_profile_seed=int(boundary_profile_seed),
        segment_length_power=float(config.segment_length_power),
        seed=int(config.seed) + 7,
    )
    docs_test, _test_stats = generate_segment_lda_docs(
        int(config.test_docs),
        topics=topics_phi,
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_len=int(config.min_seg_len),
        max_seg_len=int(config.max_seg_len),
        leaf_tokens=int(config.leaf_tokens),
        align_segments_to_leaves=bool(config.align_segments_to_leaves),
        doc_topic_concentration=float(config.doc_topic_concentration),
        topic_process=str(config.topic_process),
        boundary_profile=str(config.boundary_profile),
        boundary_profile_strength=float(config.boundary_profile_strength),
        boundary_profile_seed=int(boundary_profile_seed),
        segment_length_power=float(config.segment_length_power),
        seed=int(config.seed) + 11,
    )

    # Precompute oracle prefix arrays for each doc for fast span queries.
    prefixes_train = [_prefix_counts(d.topics, n_topics=int(config.n_topics)) for d in docs_train]
    prefixes_test = [_prefix_counts(d.topics, n_topics=int(config.n_topics)) for d in docs_test]

    def _oracle(doc_index: int, span: Tuple[int, int]) -> np.ndarray:
        topic_prefix, bigram_prefix = prefixes_train[int(doc_index)] if doc_index < len(prefixes_train) else prefixes_test[int(doc_index) - len(prefixes_train)]
        doc = docs_train[int(doc_index)] if doc_index < len(docs_train) else docs_test[int(doc_index) - len(docs_train)]
        val = _oracle_from_prefix(theta_true, w_big_true, topic_prefix, bigram_prefix, doc.topics, span)
        return np.asarray([float(val)], dtype=np.float32)

    def _endpoints(doc_index: int, span: Tuple[int, int]) -> Tuple[int, int]:
        doc = docs_train[int(doc_index)] if doc_index < len(docs_train) else docs_test[int(doc_index) - len(docs_train)]
        s, e = span
        if int(e) <= int(s):
            return (0, 0)
        z = doc.topics
        return (int(z[int(s)]), int(z[int(e) - 1]))

    # Leaf features + spans
    feats_train: List[np.ndarray] = []
    spans_train: List[List[Tuple[int, int]]] = []
    masses_train: List[np.ndarray] = []
    for d in docs_train:
        x, spans, m = _leaf_topic_onehot_features(d, n_topics=int(config.n_topics), leaf_tokens=int(config.leaf_tokens))
        feats_train.append(x)
        spans_train.append(spans)
        masses_train.append(m)

    feats_test: List[np.ndarray] = []
    spans_test: List[List[Tuple[int, int]]] = []
    masses_test: List[np.ndarray] = []
    for d in docs_test:
        x, spans, m = _leaf_topic_onehot_features(d, n_topics=int(config.n_topics), leaf_tokens=int(config.leaf_tokens))
        feats_test.append(x)
        spans_test.append(spans)
        masses_test.append(m)

    # Build supervised docs (train then test, with a doc_index offset for oracle_fn dispatch).
    docs_sup_train = build_docs_from_oracle(
        leaf_features=feats_train,
        leaf_spans=spans_train,
        oracle_fn=_oracle,
        output_dim=1,
        endpoint_categories=int(config.n_topics),
        endpoints_fn=_endpoints,
        leaf_mass=masses_train,
    )

    # For test docs, we re-index oracle_fn by offsetting doc_index into the concatenated list.
    offset = len(docs_train)

    def _oracle_test(doc_index: int, span: Tuple[int, int]) -> np.ndarray:
        return _oracle(int(doc_index) + offset, span)

    def _endpoints_test(doc_index: int, span: Tuple[int, int]) -> Tuple[int, int]:
        return _endpoints(int(doc_index) + offset, span)

    docs_sup_test = build_docs_from_oracle(
        leaf_features=feats_test,
        leaf_spans=spans_test,
        oracle_fn=_oracle_test,
        output_dim=1,
        endpoint_categories=int(config.n_topics),
        endpoints_fn=_endpoints_test,
        leaf_mass=masses_test,
    )

    train_cfg = LearnedGTrainingConfig(
        state_dim=int(config.state_dim),
        hidden_dim=int(config.hidden_dim),
        output_mode="regression",
        include_endpoints=True,
        endpoint_categories=int(config.n_topics),
        include_mass=True,
        n_epochs=int(config.n_epochs),
        batch_docs=int(config.batch_docs),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
        grad_clip_norm=float(config.grad_clip_norm),
        leaf_query_rate=float(config.leaf_query_rate),
        audit_policy=str(config.audit_policy),  # type: ignore[arg-type]
        audit_fixed_nodes=int(config.audit_fixed_nodes),
        audit_fraction=float(config.audit_fraction),
        audit_scale=float(config.audit_scale),
        include_root_query=bool(config.include_root_query),
        root_weight=float(config.root_weight),
        leaf_weight=float(config.leaf_weight),
        c3_weight=float(config.c3_weight),
        schedule_consistency_weight=float(config.schedule_consistency_weight),
        idempotence_weight=float(config.idempotence_weight),
        violation_tau=float(config.violation_tau),
        seed=int(config.seed),
        torch_threads=int(config.torch_threads),
    )

    model, geom, train_loss_final = train_torch_mergeable_sketch(
        docs_sup_train,
        leaf_feature_dim=int(feats_train[0].shape[1]) if feats_train else int(config.leaf_tokens) * int(config.n_topics),
        output_dim=1,
        config=train_cfg,
    )
    metrics = eval_torch_mergeable_sketch(
        model,
        docs_sup_test,
        output_mode=train_cfg.output_mode,
        violation_tau=float(config.violation_tau),
    )

    weight_truth = {
        "relevant_topics": [int(x) for x in R],
        "theta_true": np.asarray(theta_true, dtype=np.float64).tolist(),
        "W_base": np.asarray(W_base, dtype=np.float64).tolist(),
        "lambda_multiplier": float(lambda_multiplier),
    }

    training_geometry = {**asdict(geom), "train_loss_final": float(train_loss_final)}
    metrics_dict = asdict(metrics)

    return LearnedSegmentLDAOpsGSummary(
        config=asdict(config),
        topic_meta=dict(topic_meta),
        weight_truth=weight_truth,
        training_geometry=training_geometry,
        metrics=metrics_dict,
    )
