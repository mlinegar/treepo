"""
Learn a mergeable sketch `g` for topic-mixture recovery in Segmented-LDA.

This learns:
  - g_leaf: word-count leaf -> state
  - merge: (state_left,state_right) -> state_parent
  - readout: state -> θ(span) ∈ Δ^{K-1}

Training supervision is local (leaf + sampled internal-node oracle labels), and
evaluation reports C1/C3-style L1 discrepancies plus schedule spread.

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
from treepo.bench.lda.segmented_lda_ctreepo import (
    SegmentedBook,
    SegmentedLDACtreePOConfig,
    _generate_segmented_corpus,
    _leaf_spans,
    _sample_topic_word_matrix,
    _span_topic_theta,
    _span_word_counts,
)


@dataclass(frozen=True)
class LearnedSegmentedLDATopicThetaGConfig:
    # Core LDA params.
    n_topics: int = 5
    vocab_size: int = 256
    alpha_topic: float = 0.20
    beta_word: float = 0.10

    # Segmentation DGP.
    topic_process: str = "segments"  # segments|bag_of_words
    n_books_train: int = 256
    n_books_test: int = 256
    min_segments: int = 8
    max_segments: int = 20
    min_seg_tokens: int = 24
    max_seg_tokens: int = 64
    segment_concentration: float = 80.0
    segment_background: float = 2.0

    # Leaf partition.
    fixed_leaf_tokens: int = 32

    # Learned sketch / local-law training.
    state_dim: int = 64
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
class LearnedSegmentedLDATopicThetaGSummary:
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    training_geometry: Dict[str, object]
    metrics: Dict[str, object]

    def to_json(self) -> str:
        return json.dumps(
            {
                "config": self.config,
                "topic_meta": self.topic_meta,
                "training_geometry": self.training_geometry,
                "metrics": self.metrics,
            },
            indent=2,
            sort_keys=True,
        )


def _to_generator_config(config: LearnedSegmentedLDATopicThetaGConfig) -> SegmentedLDACtreePOConfig:
    # Reuse the existing generator implementation by mapping fields.
    return SegmentedLDACtreePOConfig(
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        alpha_topic=float(config.alpha_topic),
        beta_word=float(config.beta_word),
        topic_process=str(config.topic_process),
        n_books_train=int(config.n_books_train),
        n_books_test=int(config.n_books_test),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_tokens=int(config.min_seg_tokens),
        max_seg_tokens=int(config.max_seg_tokens),
        segment_concentration=float(config.segment_concentration),
        segment_background=float(config.segment_background),
        fixed_leaf_tokens=int(config.fixed_leaf_tokens),
        seed=int(config.seed),
    )


def _leaf_wordcount_features(book: SegmentedBook, *, vocab_size: int, leaf_tokens: int) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    spans = _leaf_spans(len(book.token_words), leaf_tokens=int(leaf_tokens))
    feats: List[np.ndarray] = []
    masses: List[float] = []
    for (s, e) in spans:
        wc = _span_word_counts(book.token_words, start=int(s), end=int(e), vocab_size=int(vocab_size))
        total = float(np.sum(wc))
        freq = (wc / total) if total > 0.0 else wc
        feats.append(np.asarray(freq, dtype=np.float32))
        masses.append(float(max(0, int(e) - int(s))))
    if not feats:
        feats = [np.zeros((int(vocab_size),), dtype=np.float32)]
        spans = [(0, 0)]
        masses = [0.0]
    return np.stack(feats, axis=0), spans, np.asarray(masses, dtype=np.float32)


def run_learned_segmented_lda_theta_g_experiment(
    config: LearnedSegmentedLDATopicThetaGConfig,
) -> LearnedSegmentedLDATopicThetaGSummary:
    gen_cfg = _to_generator_config(config)
    rng_topic = np.random.default_rng(int(config.seed) + 13)
    rng_train = np.random.default_rng(int(config.seed) + 7)
    rng_test = np.random.default_rng(int(config.seed) + 11)

    topic_word_true = _sample_topic_word_matrix(gen_cfg, rng=rng_topic)
    corpus_train = _generate_segmented_corpus(
        gen_cfg, topic_word_true=np.asarray(topic_word_true, dtype=np.float64), n_books=int(config.n_books_train), rng=rng_train
    )
    corpus_test = _generate_segmented_corpus(
        gen_cfg, topic_word_true=np.asarray(topic_word_true, dtype=np.float64), n_books=int(config.n_books_test), rng=rng_test
    )

    books_train = list(corpus_train.books)
    books_test = list(corpus_test.books)

    # Per-doc leaf features + spans + masses.
    feats_train: List[np.ndarray] = []
    spans_train: List[List[Tuple[int, int]]] = []
    masses_train: List[np.ndarray] = []
    for b in books_train:
        x, spans, m = _leaf_wordcount_features(b, vocab_size=int(config.vocab_size), leaf_tokens=int(config.fixed_leaf_tokens))
        feats_train.append(x)
        spans_train.append(spans)
        masses_train.append(m)

    feats_test: List[np.ndarray] = []
    spans_test: List[List[Tuple[int, int]]] = []
    masses_test: List[np.ndarray] = []
    for b in books_test:
        x, spans, m = _leaf_wordcount_features(b, vocab_size=int(config.vocab_size), leaf_tokens=int(config.fixed_leaf_tokens))
        feats_test.append(x)
        spans_test.append(spans)
        masses_test.append(m)

    def _oracle_train(doc_index: int, span: Tuple[int, int]) -> np.ndarray:
        book = books_train[int(doc_index)]
        theta = _span_topic_theta(book.token_topics, start=int(span[0]), end=int(span[1]), n_topics=int(config.n_topics))
        return np.asarray(theta, dtype=np.float32)

    docs_sup_train = build_docs_from_oracle(
        leaf_features=feats_train,
        leaf_spans=spans_train,
        oracle_fn=_oracle_train,
        output_dim=int(config.n_topics),
        leaf_mass=masses_train,
    )

    def _oracle_test(doc_index: int, span: Tuple[int, int]) -> np.ndarray:
        book = books_test[int(doc_index)]
        theta = _span_topic_theta(book.token_topics, start=int(span[0]), end=int(span[1]), n_topics=int(config.n_topics))
        return np.asarray(theta, dtype=np.float32)

    docs_sup_test = build_docs_from_oracle(
        leaf_features=feats_test,
        leaf_spans=spans_test,
        oracle_fn=_oracle_test,
        output_dim=int(config.n_topics),
        leaf_mass=masses_test,
    )

    train_cfg = LearnedGTrainingConfig(
        state_dim=int(config.state_dim),
        hidden_dim=int(config.hidden_dim),
        output_mode="simplex",
        include_endpoints=False,
        endpoint_categories=0,
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
        leaf_feature_dim=int(feats_train[0].shape[1]) if feats_train else int(config.vocab_size),
        output_dim=int(config.n_topics),
        config=train_cfg,
    )
    metrics = eval_torch_mergeable_sketch(
        model,
        docs_sup_test,
        output_mode=train_cfg.output_mode,
        violation_tau=float(config.violation_tau),
    )

    topic_meta = {
        "topic_word_true_row_sums": np.sum(np.asarray(topic_word_true, dtype=np.float64), axis=1).tolist(),
    }

    training_geometry = {**asdict(geom), "train_loss_final": float(train_loss_final)}
    return LearnedSegmentedLDATopicThetaGSummary(
        config=asdict(config),
        topic_meta=topic_meta,
        training_geometry=training_geometry,
        metrics=asdict(metrics),
    )

