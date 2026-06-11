import numpy as np

from treepo._research.bench.lda.segmented_lda_ctreepo import (
    SegmentedLDACtreePOConfig,
    run_segmented_lda_ctreepo_simulation,
)


def test_true_phi_estimator_has_zero_bound():
    cfg = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=80,
        n_books_train=32,
        n_books_test=32,
        min_segments=4,
        max_segments=6,
        min_seg_tokens=12,
        max_seg_tokens=20,
        fixed_leaf_tokens=16,
        topic_phi_estimator="true",
        seed=0,
    )
    out = run_segmented_lda_ctreepo_simulation(cfg)
    assert out.topic_meta["topic_phi_estimator"] == "true"
    assert float(out.topic_meta["topic_phi_eps_bound"]) == 0.0


def test_noisy_theory_bound_scales_like_inverse_sqrt_train_docs():
    cfg_small = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=80,
        n_books_train=25,
        n_books_test=16,
        min_segments=4,
        max_segments=5,
        min_seg_tokens=12,
        max_seg_tokens=16,
        fixed_leaf_tokens=16,
        topic_phi_estimator="noisy_theory",
        tlda_rate_constant=2.0,
        seed=1,
    )
    cfg_large = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=80,
        n_books_train=100,
        n_books_test=16,
        min_segments=4,
        max_segments=5,
        min_seg_tokens=12,
        max_seg_tokens=16,
        fixed_leaf_tokens=16,
        topic_phi_estimator="noisy_theory",
        tlda_rate_constant=2.0,
        seed=1,
    )
    out_small = run_segmented_lda_ctreepo_simulation(cfg_small)
    out_large = run_segmented_lda_ctreepo_simulation(cfg_large)
    eps_small = float(out_small.topic_meta["topic_phi_eps_bound"])
    eps_large = float(out_large.topic_meta["topic_phi_eps_bound"])
    assert eps_small > eps_large
    assert abs((eps_small / eps_large) - (np.sqrt(100.0) / np.sqrt(25.0))) < 1e-6


def test_end_to_end_triangle_upper_bound_holds():
    cfg = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=120,
        n_books_train=64,
        n_books_test=64,
        min_segments=6,
        max_segments=10,
        min_seg_tokens=16,
        max_seg_tokens=24,
        fixed_leaf_tokens=16,
        calibration_leaf_query_rate=0.25,
        eval_leaf_query_rate=0.10,
        eval_internal_query_rate=0.25,
        eval_internal_query_design="risk",
        seed=3,
    )
    out = run_segmented_lda_ctreepo_simulation(cfg)
    assert out.decomposition.upper_bound_mean + 1e-10 >= out.decomposition.total_root_l1_mean
    assert out.decomposition.slack_mean >= -1e-10


def test_guidance_does_not_increase_root_error():
    cfg = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=120,
        n_books_train=64,
        n_books_test=64,
        min_segments=6,
        max_segments=10,
        min_seg_tokens=16,
        max_seg_tokens=24,
        fixed_leaf_tokens=16,
        calibration_leaf_query_rate=0.20,
        eval_leaf_query_rate=0.20,
        eval_internal_query_rate=1.00,
        eval_internal_query_design="risk",
        seed=7,
    )
    out = run_segmented_lda_ctreepo_simulation(cfg)
    unguided = out.metrics["estimated_calibrated"].root_l1_mean
    guided = out.metrics["estimated_calibrated_budgeted"].root_l1_mean
    assert guided <= unguided + 1e-10


def test_spectral_numpy_mode_runs_and_reports_finite_error_proxy():
    cfg = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=120,
        n_books_train=96,
        n_books_test=48,
        min_segments=6,
        max_segments=10,
        min_seg_tokens=12,
        max_seg_tokens=24,
        fixed_leaf_tokens=16,
        topic_phi_estimator="spectral_numpy",
        spectral_svd_dim_extra=2,
        spectral_max_leaves=2000,
        spectral_kmeans_inits=3,
        spectral_kmeans_max_iter=30,
        calibration_leaf_query_rate=0.15,
        eval_internal_query_rate=0.20,
        eval_internal_query_design="risk",
        seed=9,
    )
    out = run_segmented_lda_ctreepo_simulation(cfg)
    l2_mean = float(out.topic_meta.get("topic_phi_l2_error_mean", float("nan")))
    assert np.isfinite(l2_mean)
    assert l2_mean >= 0.0
    assert np.isfinite(out.metrics["estimated_uncalibrated"].root_l1_mean)


def test_neural_hybrid_mode_runs_and_reports_finite_error_proxy():
    cfg = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=100,
        n_books_train=64,
        n_books_test=32,
        min_segments=5,
        max_segments=8,
        min_seg_tokens=12,
        max_seg_tokens=24,
        fixed_leaf_tokens=16,
        topic_phi_estimator="neural_hybrid",
        topic_phi_docs=64,
        neural_topic_base_estimator="noisy_theory",
        neural_topic_seed_fraction=0.5,
        neural_topic_hidden_dim=16,
        neural_topic_steps=20,
        neural_topic_lr=1e-2,
        neural_topic_mix_samples=32,
        calibration_leaf_query_rate=0.10,
        eval_internal_query_rate=0.15,
        eval_internal_query_design="risk",
        seed=13,
    )
    out = run_segmented_lda_ctreepo_simulation(cfg)
    assert out.topic_meta.get("topic_phi_estimator") == "neural_hybrid"
    assert out.topic_meta.get("topic_phi_neural_base_estimator") == "noisy_theory"
    l2_mean = float(out.topic_meta.get("topic_phi_l2_error_mean", float("nan")))
    assert np.isfinite(l2_mean)
    assert l2_mean >= 0.0


def test_embedding_spectral_mode_runs_and_reports_finite_error_proxy():
    cfg = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=100,
        n_books_train=64,
        n_books_test=32,
        min_segments=5,
        max_segments=8,
        min_seg_tokens=12,
        max_seg_tokens=24,
        fixed_leaf_tokens=16,
        topic_phi_estimator="embedding_spectral",
        topic_phi_docs=64,
        embedding_topic_svd_dim_extra=3,
        embedding_topic_kmeans_inits=4,
        embedding_topic_kmeans_max_iter=40,
        embedding_topic_assignment_temperature=0.35,
        embedding_topic_ppmi_shift=1.0,
        calibration_leaf_query_rate=0.10,
        eval_internal_query_rate=0.15,
        eval_internal_query_design="risk",
        seed=41,
    )
    out = run_segmented_lda_ctreepo_simulation(cfg)
    assert out.topic_meta.get("topic_phi_estimator") == "embedding_spectral"
    l2_mean = float(out.topic_meta.get("topic_phi_l2_error_mean", float("nan")))
    assert np.isfinite(l2_mean)
    assert l2_mean >= 0.0


def test_neural_embedding_hybrid_mode_runs_and_reports_finite_error_proxy():
    cfg = SegmentedLDACtreePOConfig(
        n_topics=4,
        vocab_size=100,
        n_books_train=64,
        n_books_test=32,
        min_segments=5,
        max_segments=8,
        min_seg_tokens=12,
        max_seg_tokens=24,
        fixed_leaf_tokens=16,
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
        calibration_leaf_query_rate=0.10,
        eval_internal_query_rate=0.15,
        eval_internal_query_design="risk",
        seed=43,
    )
    out = run_segmented_lda_ctreepo_simulation(cfg)
    assert out.topic_meta.get("topic_phi_estimator") == "neural_embedding_hybrid"
    assert out.topic_meta.get("topic_phi_neural_base_estimator") == "embedding_spectral"
    l2_mean = float(out.topic_meta.get("topic_phi_l2_error_mean", float("nan")))
    assert np.isfinite(l2_mean)
    assert l2_mean >= 0.0
