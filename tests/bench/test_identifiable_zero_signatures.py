from __future__ import annotations

import math

import numpy as np
import pytest

from treepo.bench.lda.segmented_lda_ctreepo import (
    SegmentedLDACtreePOConfig,
    _counts_to_freq_rows,
    _fit_leaf_theta_mlp,
    _fit_leaf_theta_rf,
    _normalize_simplex_rows,
    _predict_leaf_theta_model,
    run_segmented_lda_ctreepo_simulation,
)
from treepo.bench.lda.segment_lda_ops_weight_recovery import estimate_topic_distributions


def test_ctree_test_set_signature_stable_across_train_docs() -> None:
    seed = 123
    cfg_small = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=64,
        n_books_train=32,
        n_books_test=64,
        min_segments=4,
        max_segments=4,
        min_seg_tokens=12,
        max_seg_tokens=12,
        fixed_leaf_tokens=8,
        topic_phi_estimator="true",
        topic_phi_docs=0,
        topic_phi_permute=False,
        calibration_leaf_query_rate=0.10,
        eval_leaf_query_rate=0.0,
        eval_internal_query_rate=0.0,
        selection_audit_trials=0,
        seed=seed,
    )
    cfg_large = SegmentedLDACtreePOConfig(**{**cfg_small.__dict__, "n_books_train": 64})

    s_small = run_segmented_lda_ctreepo_simulation(cfg_small)
    s_large = run_segmented_lda_ctreepo_simulation(cfg_large)

    sig_small = str((s_small.topic_meta or {}).get("corpus_signature_test") or "")
    sig_large = str((s_large.topic_meta or {}).get("corpus_signature_test") or "")
    assert sig_small, "missing corpus_signature_test"
    assert sig_large, "missing corpus_signature_test"
    assert sig_small == sig_large


def test_ctree_bag_of_words_test_set_signature_stable_across_train_docs() -> None:
    seed = 123
    cfg_small = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=64,
        topic_process="bag_of_words",
        n_books_train=32,
        n_books_test=64,
        min_segments=4,
        max_segments=4,
        min_seg_tokens=12,
        max_seg_tokens=12,
        fixed_leaf_tokens=8,
        topic_phi_estimator="true",
        topic_phi_docs=0,
        topic_phi_permute=False,
        calibration_leaf_query_rate=0.10,
        eval_leaf_query_rate=0.0,
        eval_internal_query_rate=0.0,
        selection_audit_trials=0,
        seed=seed,
    )
    cfg_large = SegmentedLDACtreePOConfig(**{**cfg_small.__dict__, "n_books_train": 64})

    s_small = run_segmented_lda_ctreepo_simulation(cfg_small)
    s_large = run_segmented_lda_ctreepo_simulation(cfg_large)

    sig_small = str((s_small.topic_meta or {}).get("corpus_signature_test") or "")
    sig_large = str((s_large.topic_meta or {}).get("corpus_signature_test") or "")
    assert sig_small, "missing corpus_signature_test"
    assert sig_large, "missing corpus_signature_test"
    assert sig_small == sig_large


def test_leaf_theta_mlp_predictor_output_simplex() -> None:
    rng = np.random.default_rng(0)
    n = 128
    v = 32
    k = 4

    counts = rng.poisson(lam=3.0, size=(n, v)).astype(np.float64)
    x = _counts_to_freq_rows(counts)
    y = _normalize_simplex_rows(rng.random(size=(n, k)).astype(np.float64))

    # MLP (torch) always available in this repo's default environments; if not, skip.
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not available")

    mlp, _meta = _fit_leaf_theta_mlp(
        x,
        y,
        seed=0,
        hidden_dim=16,
        epochs=2,
        batch_size=32,
        lr=1e-2,
        weight_decay=0.0,
    )
    pred = _predict_leaf_theta_model(mlp, counts[:17])
    assert pred.shape == (17, k)
    assert np.all(np.isfinite(pred))
    assert np.min(pred) >= -1e-12
    sums = np.sum(pred, axis=1)
    assert np.all(np.isfinite(sums))
    assert np.max(np.abs(sums - 1.0)) <= 1e-6


def test_leaf_theta_rf_predictor_output_simplex() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n = 128
    v = 32
    k = 4

    counts = rng.poisson(lam=3.0, size=(n, v)).astype(np.float64)
    x = _counts_to_freq_rows(counts)
    y = _normalize_simplex_rows(rng.random(size=(n, k)).astype(np.float64))

    rf, _rf_meta = _fit_leaf_theta_rf(x, y, seed=0, n_estimators=10, max_depth=4, min_samples_leaf=2)
    pred_rf = _predict_leaf_theta_model(rf, counts[:19])
    assert pred_rf.shape == (19, k)
    assert np.all(np.isfinite(pred_rf))
    assert np.min(pred_rf) >= -1e-12
    sums_rf = np.sum(pred_rf, axis=1)
    assert np.max(np.abs(sums_rf - 1.0)) <= 1e-6

    assert math.isfinite(float(np.mean(pred_rf)))


def test_topic_phi_estimator_sklearn_lda_runs_small() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    k = 4
    v = 48
    d_docs = 80
    doc_len = 64

    topics_true = [rng.dirichlet(np.full((v,), 0.20, dtype=np.float64)).astype(np.float64) for _ in range(k)]
    docs_tokens = []
    alpha = np.full((k,), 0.20, dtype=np.float64)
    for _ in range(d_docs):
        theta = rng.dirichlet(alpha).astype(np.float64)
        z = rng.choice(np.arange(k), size=doc_len, p=theta).astype(np.int64)
        w = [int(rng.choice(np.arange(v), p=np.asarray(topics_true[int(t)], dtype=np.float64))) for t in z.tolist()]
        docs_tokens.append(w)

    topics_est, meta, perm = estimate_topic_distributions(
        topics_true,
        estimator="sklearn_lda",
        n_docs=d_docs,
        doc_topic_concentration=0.20,
        tlda_delta=0.10,
        tlda_rate_constant=1.0,
        sigmaK_floor=1e-6,
        permute=False,
        seed=0,
        topic_word_concentration=0.20,
        docs_tokens=docs_tokens,
    )
    assert len(topics_est) == k
    assert tuple(int(np.asarray(t).size) for t in topics_est) == (v,) * k
    assert str(meta.get("topic_phi_estimator")) == "sklearn_lda"
    assert len(perm) == k
