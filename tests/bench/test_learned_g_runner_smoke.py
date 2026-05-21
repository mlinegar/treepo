from __future__ import annotations

import json
from pathlib import Path

import pytest


def _torch_or_skip() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not available")


def test_runner_single_learned_ops_writes_json_csv(tmp_path: Path) -> None:
    _torch_or_skip()
    from treepo.bench.runner import EXPERIMENT_LEARNED_OPS_G, run_single

    cfg = {
        "n_topics": 4,
        "vocab_size": 64,
        "anchor_words_per_topic": 8,
        "min_tokens": 64,
        "max_tokens": 64,
        "min_segments": 2,
        "max_segments": 4,
        "min_seg_len": 16,
        "max_seg_len": 32,
        "leaf_tokens": 8,
        "train_docs": 12,
        "test_docs": 12,
        "state_dim": 12,
        "hidden_dim": 32,
        "n_epochs": 1,
        "batch_docs": 6,
        "leaf_query_rate": 1.0,
        "audit_policy": "fraction",
        "audit_fraction": 0.25,
        "seed": 0,
        "torch_threads": 1,
    }
    json_out = tmp_path / "ops.json"
    csv_out = tmp_path / "ops.csv"
    run_single(experiment=EXPERIMENT_LEARNED_OPS_G, config=cfg, json_out=json_out, csv_out=csv_out)

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert "runtime_meta" in payload
    assert Path(csv_out).exists()


def test_runner_single_learned_theta_writes_json_csv(tmp_path: Path) -> None:
    _torch_or_skip()
    from treepo.bench.runner import EXPERIMENT_LEARNED_SEGMENTED_THETA_G, run_single

    cfg = {
        "n_topics": 4,
        "vocab_size": 64,
        "n_books_train": 12,
        "n_books_test": 12,
        "min_segments": 2,
        "max_segments": 4,
        "min_seg_tokens": 16,
        "max_seg_tokens": 32,
        "fixed_leaf_tokens": 16,
        "state_dim": 16,
        "hidden_dim": 48,
        "n_epochs": 1,
        "batch_docs": 6,
        "leaf_query_rate": 1.0,
        "audit_policy": "fraction",
        "audit_fraction": 0.25,
        "seed": 0,
        "torch_threads": 1,
    }
    json_out = tmp_path / "theta.json"
    csv_out = tmp_path / "theta.csv"
    run_single(experiment=EXPERIMENT_LEARNED_SEGMENTED_THETA_G, config=cfg, json_out=json_out, csv_out=csv_out)

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert "runtime_meta" in payload
    assert Path(csv_out).exists()
