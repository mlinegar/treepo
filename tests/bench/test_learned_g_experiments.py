from __future__ import annotations

import json
import math

import pytest


def _torch_or_skip() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not available")


def test_learned_segment_lda_ops_g_runs_small_cpu() -> None:
    _torch_or_skip()
    from treepo._research.bench.lda.learned_segment_lda_ops_g import (
        LearnedSegmentLDAOpsGConfig,
        run_learned_segment_lda_ops_g_experiment,
    )

    cfg = LearnedSegmentLDAOpsGConfig(
        n_topics=4,
        vocab_size=64,
        anchor_words_per_topic=8,
        min_tokens=64,
        max_tokens=64,
        min_segments=2,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=32,
        leaf_tokens=8,
        train_docs=24,
        test_docs=24,
        state_dim=16,
        hidden_dim=48,
        n_epochs=2,
        batch_docs=8,
        leaf_query_rate=1.0,
        audit_policy="fraction",
        audit_fraction=0.25,
        schedule_consistency_weight=0.1,
        seed=0,
        torch_threads=1,
    )
    summary = run_learned_segment_lda_ops_g_experiment(cfg)
    payload = json.loads(summary.to_json())

    metrics = payload["metrics"]
    assert math.isfinite(float(metrics["root_mae"]))
    assert math.isfinite(float(metrics["leaf_mae"]))
    assert math.isfinite(float(metrics["merge_mae"]))
    assert math.isfinite(float(metrics["schedule_spread_mean"]))
    assert int(metrics["n_docs"]) > 0


def test_learned_segmented_lda_theta_g_runs_small_cpu() -> None:
    _torch_or_skip()
    from treepo._research.bench.lda.learned_segmented_lda_theta_g import (
        LearnedSegmentedLDATopicThetaGConfig,
        run_learned_segmented_lda_theta_g_experiment,
    )

    cfg = LearnedSegmentedLDATopicThetaGConfig(
        n_topics=4,
        vocab_size=64,
        n_books_train=24,
        n_books_test=24,
        min_segments=2,
        max_segments=6,
        min_seg_tokens=16,
        max_seg_tokens=32,
        fixed_leaf_tokens=16,
        state_dim=24,
        hidden_dim=64,
        n_epochs=2,
        batch_docs=8,
        leaf_query_rate=1.0,
        audit_policy="fraction",
        audit_fraction=0.25,
        schedule_consistency_weight=0.1,
        seed=0,
        torch_threads=1,
    )
    summary = run_learned_segmented_lda_theta_g_experiment(cfg)
    payload = json.loads(summary.to_json())

    metrics = payload["metrics"]
    assert math.isfinite(float(metrics["root_mae"]))
    assert math.isfinite(float(metrics["leaf_mae"]))
    assert math.isfinite(float(metrics["merge_mae"]))
    assert math.isfinite(float(metrics["schedule_spread_mean"]))
    assert int(metrics["n_docs"]) > 0
