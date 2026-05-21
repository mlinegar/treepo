import numpy as np
import pytest

from treepo.bench.lda.segment_lda_ops_weight_recovery import (
    SegmentLDADoc,
    SegmentLDAOpsWeightRecoveryConfig,
    _eval_sketch_family,
    _fit_ridge,
    _oracle_from_prefix,
    _prefix_counts,
    _span_features_from_prefix,
    generate_segment_lda_docs,
    sample_topic_distributions,
    run_segment_lda_ops_weight_recovery_experiment,
)


def test_leaf_only_queries_cannot_identify_boundary_bigram_weight_but_root_query_can():
    # Two leaves of length 2 with a single cross-leaf boundary bigram (0,1).
    # Oracle: f⋆ = w[0,1] * count(0->1); all other weights are 0.
    k = 2
    theta_true = np.zeros((k,), dtype=np.float64)
    W_base = np.zeros((k, k), dtype=np.float64)
    W_base[0, 1] = 1.0
    w_big_true = W_base.reshape(-1).astype(np.float64, copy=False)

    topics = (0, 0, 1, 1)
    topic_prefix, bigram_prefix = _prefix_counts(topics, n_topics=k)

    leaf1 = (0, 2)
    leaf2 = (2, 4)
    root = (0, 4)

    x1, _f1, _l1 = _span_features_from_prefix(topic_prefix, bigram_prefix, topics, leaf1, n_topics=k)
    x2, _f2, _l2 = _span_features_from_prefix(topic_prefix, bigram_prefix, topics, leaf2, n_topics=k)
    xr, _fr, _lr = _span_features_from_prefix(topic_prefix, bigram_prefix, topics, root, n_topics=k)

    y1 = _oracle_from_prefix(theta_true, w_big_true, topic_prefix, bigram_prefix, topics, leaf1)
    y2 = _oracle_from_prefix(theta_true, w_big_true, topic_prefix, bigram_prefix, topics, leaf2)
    yr = _oracle_from_prefix(theta_true, w_big_true, topic_prefix, bigram_prefix, topics, root)
    assert y1 == pytest.approx(0.0, abs=1e-12)
    assert y2 == pytest.approx(0.0, abs=1e-12)
    assert yr == pytest.approx(1.0, abs=1e-12)

    # Fit from leaf-only queries: boundary weight is unobserved -> ridge returns 0.
    beta_leaf = _fit_ridge(np.vstack([x1, x2]), np.asarray([y1, y2]), ridge_lambda=1e-12)
    pred_root_leaf = float(np.dot(beta_leaf, xr))
    assert pred_root_leaf == pytest.approx(0.0, abs=1e-10)

    # Add one root query: boundary becomes identifiable.
    beta_root = _fit_ridge(
        np.vstack([x1, x2, xr]),
        np.asarray([y1, y2, yr], dtype=np.float64),
        ridge_lambda=1e-12,
    )
    boundary_idx = k + (0 * k + 1)  # offset by theta, then bigram index
    assert float(beta_root[boundary_idx]) == pytest.approx(1.0, abs=1e-6)
    pred_root = float(np.dot(beta_root, xr))
    assert pred_root == pytest.approx(float(yr), abs=1e-6)

    # Multiplier recovery via norm: since ||W_base||=1 and lambda=1, ||w_big_hat|| ≈ 1.
    w_big_hat = beta_root[k:]
    assert float(np.linalg.norm(w_big_hat)) == pytest.approx(1.0, abs=1e-6)


def test_exact_undersupported_and_flip_families_match_expected_root_distortion():
    k = 2
    theta_true = np.zeros((k,), dtype=np.float64)
    W_base = np.zeros((k, k), dtype=np.float64)
    W_base[0, 1] = 1.0
    w_big_true = W_base.reshape(-1).astype(np.float64, copy=False)

    doc = SegmentLDADoc(tokens=(0, 0, 0, 0), topics=(0, 0, 1, 1))
    leaf_tokens = 2

    exact = _eval_sketch_family(
        [doc], theta_true=theta_true, w_big_true=w_big_true, leaf_tokens=leaf_tokens, tau=0.0, kind="exact"
    )
    assert exact.root_mae == pytest.approx(0.0, abs=1e-12)

    undersupported = _eval_sketch_family(
        [doc],
        theta_true=theta_true,
        w_big_true=w_big_true,
        leaf_tokens=leaf_tokens,
        tau=0.0,
        kind="undersupported",
    )
    assert undersupported.root_mae == pytest.approx(1.0, abs=1e-12)
    assert undersupported.merge_violation_rate > 0.0

    flip_r1 = _eval_sketch_family(
        [doc],
        theta_true=theta_true,
        w_big_true=w_big_true,
        leaf_tokens=leaf_tokens,
        tau=0.0,
        kind="flip",
        rounds=1,
    )
    flip_r2 = _eval_sketch_family(
        [doc],
        theta_true=theta_true,
        w_big_true=w_big_true,
        leaf_tokens=leaf_tokens,
        tau=0.0,
        kind="flip",
        rounds=2,
    )
    assert flip_r1.root_mae == pytest.approx(0.0, abs=1e-12)
    assert flip_r2.root_mae == pytest.approx(1.0, abs=1e-12)


def test_segment_lda_doc_generator_supports_bag_of_words_and_segments_modes():
    topics_phi, _meta = sample_topic_distributions(
        vocab_size=64,
        n_topics=4,
        topic_concentration=0.2,
        emission_mode="disjoint",
        anchor_words_per_topic=5,
        anchor_multiplier=5.0,
        seed=0,
    )

    docs_bow, stats_bow = generate_segment_lda_docs(
        10,
        topics=topics_phi,
        min_tokens=64,
        max_tokens=64,
        min_segments=2,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=32,
        leaf_tokens=16,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.6,
        topic_process="bag_of_words",
        boundary_profile="middle",
        boundary_profile_strength=10.0,
        boundary_profile_seed=123,
        seed=1,
    )
    assert stats_bow["mean_segments"] == pytest.approx(0.0, abs=1e-12)
    assert all(len(d.tokens) == len(d.topics) for d in docs_bow)

    docs_seg, stats_seg = generate_segment_lda_docs(
        10,
        topics=topics_phi,
        min_tokens=64,
        max_tokens=64,
        min_segments=2,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=32,
        leaf_tokens=16,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.6,
        topic_process="segments",
        boundary_profile="middle",
        boundary_profile_strength=10.0,
        boundary_profile_seed=123,
        seed=1,
    )
    assert stats_seg["mean_segments"] >= 1.0
    assert all(len(d.tokens) == len(d.topics) for d in docs_seg)


def test_topic_permutation_does_not_break_weight_recovery_metrics_when_aligned():
    # With disjoint topic supports and eps=0, the noisy topic estimator can only permute topics.
    # The ridge model should still recover weights (up to permutation), and the simulation should
    # report aligned metrics near-perfect.
    cfg = SegmentLDAOpsWeightRecoveryConfig(
        n_topics=4,
        vocab_size=64,
        min_tokens=64,
        max_tokens=64,
        min_segments=2,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=32,
        leaf_tokens=16,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.6,
        topic_process="segments",
        boundary_profile="uniform",
        boundary_profile_strength=0.0,
        boundary_profile_seed=123,
        segment_length_power=1.0,
        topic_concentration=0.2,
        emission_mode="disjoint",
        anchor_words_per_topic=5,
        anchor_multiplier=5.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=True,
        lambda_multiplier=1.0,
        oracle_noise_std=0.0,
        audit_policy="all",
        audit_strategy="random",
        ridge_lambda=1e-8,
        topic_source="infer",
        feature_inference="hard",
        topic_phi_estimator="noisy_theory",
        topic_phi_docs=100,
        tlda_delta=0.10,
        tlda_rate_constant=0.0,  # eps=0 => exact topics up to permutation
        tlda_sigmaK_floor=1e-6,
        topic_phi_permute=True,
        run_all_feature_modes=True,
        violation_tau=0.0,
        train_docs=30,
        test_docs=30,
        seed=7,
    )
    summary = run_segment_lda_ops_weight_recovery_experiment(cfg)

    # Topic estimation error should be ~0 after best alignment.
    assert float(summary.topic_meta.get("topic_phi_l2_error_max", 1.0)) < 1e-8

    ridge = summary.metrics.get("ridge", {})
    assert isinstance(ridge, dict)
    assert float(ridge.get("theta_cosine", 0.0)) > 0.98
    assert float(ridge.get("bigram_cosine", 0.0)) > 0.90


def test_online_tensor_lda_estimator_runs_end_to_end():
    cfg = SegmentLDAOpsWeightRecoveryConfig(
        n_topics=3,
        vocab_size=48,
        min_tokens=192,
        max_tokens=192,
        min_segments=2,
        max_segments=4,
        min_seg_len=32,
        max_seg_len=96,
        leaf_tokens=16,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.6,
        topic_process="bag_of_words",
        boundary_profile="uniform",
        boundary_profile_strength=0.0,
        boundary_profile_seed=123,
        segment_length_power=1.0,
        topic_concentration=0.2,
        emission_mode="anchored",
        anchor_words_per_topic=4,
        anchor_multiplier=10.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=True,
        lambda_multiplier=1.0,
        oracle_noise_std=0.0,
        audit_policy="fraction",
        audit_fraction=0.2,
        audit_strategy="random",
        ridge_lambda=1e-3,
        topic_source="infer",
        feature_inference="hard",
        topic_phi_estimator="online_tensor_lda",
        topic_phi_docs=100,
        tlda_delta=0.1,
        tlda_rate_constant=1.0,
        tlda_sigmaK_floor=1e-6,
        topic_phi_permute=True,
        online_tensor_lda_burn_in_docs=50,
        online_tensor_lda_batch_docs=25,
        online_tensor_lda_passes=1,
        online_tensor_lda_lr=0.1,
        online_tensor_lda_grad_clip_norm=2.0,
        run_all_feature_modes=False,
        violation_tau=0.0,
        train_docs=60,
        test_docs=40,
        seed=11,
    )
    summary = run_segment_lda_ops_weight_recovery_experiment(cfg)
    assert summary.topic_meta.get("topic_phi_estimator") == "online_tensor_lda"
    assert np.isfinite(float(summary.topic_meta.get("topic_phi_l2_error_mean", float("nan"))))


def test_neural_hybrid_topic_estimator_runs_end_to_end():
    cfg = SegmentLDAOpsWeightRecoveryConfig(
        n_topics=3,
        vocab_size=48,
        min_tokens=128,
        max_tokens=128,
        min_segments=2,
        max_segments=4,
        min_seg_len=24,
        max_seg_len=64,
        leaf_tokens=16,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.6,
        topic_process="segments",
        boundary_profile="uniform",
        boundary_profile_strength=0.0,
        boundary_profile_seed=123,
        segment_length_power=1.0,
        topic_concentration=0.2,
        emission_mode="anchored",
        anchor_words_per_topic=4,
        anchor_multiplier=10.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=True,
        lambda_multiplier=1.0,
        oracle_noise_std=0.0,
        audit_policy="fraction",
        audit_fraction=0.2,
        audit_strategy="random",
        ridge_lambda=1e-3,
        topic_source="infer",
        feature_inference="hard",
        topic_phi_estimator="neural_hybrid",
        topic_phi_docs=80,
        tlda_delta=0.1,
        tlda_rate_constant=1.0,
        tlda_sigmaK_floor=1e-6,
        topic_phi_permute=True,
        neural_topic_base_estimator="noisy_theory",
        neural_topic_seed_fraction=0.5,
        neural_topic_hidden_dim=16,
        neural_topic_steps=20,
        neural_topic_lr=1e-2,
        neural_topic_mix_samples=32,
        run_all_feature_modes=False,
        violation_tau=0.0,
        train_docs=40,
        test_docs=30,
        seed=17,
    )
    summary = run_segment_lda_ops_weight_recovery_experiment(cfg)
    assert summary.topic_meta.get("topic_phi_estimator") == "neural_hybrid"
    assert summary.topic_meta.get("topic_phi_neural_base_estimator") == "noisy_theory"
    assert np.isfinite(float(summary.topic_meta.get("topic_phi_l2_error_mean", float("nan"))))


def test_embedding_spectral_topic_estimator_runs_end_to_end():
    cfg = SegmentLDAOpsWeightRecoveryConfig(
        n_topics=3,
        vocab_size=48,
        min_tokens=160,
        max_tokens=160,
        min_segments=2,
        max_segments=4,
        min_seg_len=24,
        max_seg_len=80,
        leaf_tokens=16,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.6,
        topic_process="segments",
        boundary_profile="uniform",
        boundary_profile_strength=0.0,
        boundary_profile_seed=123,
        segment_length_power=1.0,
        topic_concentration=0.2,
        emission_mode="anchored",
        anchor_words_per_topic=4,
        anchor_multiplier=10.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=True,
        lambda_multiplier=1.0,
        oracle_noise_std=0.0,
        audit_policy="fraction",
        audit_fraction=0.2,
        audit_strategy="random",
        ridge_lambda=1e-3,
        topic_source="infer",
        feature_inference="hard",
        topic_phi_estimator="embedding_spectral",
        topic_phi_docs=64,
        embedding_topic_svd_dim_extra=3,
        embedding_topic_kmeans_inits=4,
        embedding_topic_kmeans_max_iter=40,
        embedding_topic_assignment_temperature=0.35,
        embedding_topic_ppmi_shift=1.0,
        run_all_feature_modes=False,
        violation_tau=0.0,
        train_docs=40,
        test_docs=30,
        seed=29,
    )
    summary = run_segment_lda_ops_weight_recovery_experiment(cfg)
    assert summary.topic_meta.get("topic_phi_estimator") == "embedding_spectral"
    assert np.isfinite(float(summary.topic_meta.get("topic_phi_l2_error_mean", float("nan"))))


def test_neural_embedding_hybrid_topic_estimator_runs_end_to_end():
    cfg = SegmentLDAOpsWeightRecoveryConfig(
        n_topics=3,
        vocab_size=48,
        min_tokens=160,
        max_tokens=160,
        min_segments=2,
        max_segments=4,
        min_seg_len=24,
        max_seg_len=80,
        leaf_tokens=16,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.6,
        topic_process="segments",
        boundary_profile="uniform",
        boundary_profile_strength=0.0,
        boundary_profile_seed=123,
        segment_length_power=1.0,
        topic_concentration=0.2,
        emission_mode="anchored",
        anchor_words_per_topic=4,
        anchor_multiplier=10.0,
        relevant_topics=2,
        theta_scale=1.0,
        zero_diagonal=True,
        lambda_multiplier=1.0,
        oracle_noise_std=0.0,
        audit_policy="fraction",
        audit_fraction=0.2,
        audit_strategy="random",
        ridge_lambda=1e-3,
        topic_source="infer",
        feature_inference="hard",
        topic_phi_estimator="neural_embedding_hybrid",
        topic_phi_docs=64,
        embedding_topic_svd_dim_extra=3,
        embedding_topic_kmeans_inits=4,
        embedding_topic_kmeans_max_iter=40,
        embedding_topic_assignment_temperature=0.35,
        embedding_topic_ppmi_shift=1.0,
        neural_topic_seed_fraction=0.5,
        neural_topic_hidden_dim=16,
        neural_topic_steps=20,
        neural_topic_lr=1e-2,
        neural_topic_mix_samples=32,
        run_all_feature_modes=False,
        violation_tau=0.0,
        train_docs=40,
        test_docs=30,
        seed=31,
    )
    summary = run_segment_lda_ops_weight_recovery_experiment(cfg)
    assert summary.topic_meta.get("topic_phi_estimator") == "neural_embedding_hybrid"
    assert summary.topic_meta.get("topic_phi_neural_base_estimator") == "embedding_spectral"
    assert np.isfinite(float(summary.topic_meta.get("topic_phi_l2_error_mean", float("nan"))))
