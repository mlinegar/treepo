from __future__ import annotations

import json
from pathlib import Path

import pytest


def _torch_or_skip() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not available")


def test_runner_single_cardinality_recovery_writes_json_csv(tmp_path: Path) -> None:
    _torch_or_skip()
    from treepo.bench.runner import EXPERIMENT_CARDINALITY_RECOVERY, run_single

    cfg = {
        "state_dims": [8],
        "train_docs_grid": [8],
        "train_sizes": None,
        "n_val": 8,
        "n_test": 8,
        "hidden_dim": 32,
        "n_epochs": 1,
        "batch_size": 4,
        "use_cuda": False,
        "seed": 0,
    }
    json_out = tmp_path / "cardinality.json"
    csv_out = tmp_path / "cardinality.csv"
    run_single(
        experiment=EXPERIMENT_CARDINALITY_RECOVERY,
        config=cfg,
        json_out=json_out,
        csv_out=csv_out,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert "runtime_meta" in payload
    assert len(payload["results"]) == 1
    assert csv_out.exists()


def test_runner_single_hll_merge_learning_writes_json_csv(tmp_path: Path) -> None:
    _torch_or_skip()
    from treepo.bench.runner import EXPERIMENT_HLL_MERGE_LEARNING, run_single

    cfg = {
        "model_kind": "induced_projection",
        "precisions": [6],
        "train_docs_grid": [8],
        "audit_policies": ["all"],
        "min_tokens": 64,
        "max_tokens": 64,
        "leaf_size": 32,
        "n_test": 8,
        "n_epochs": 1,
        "batch_docs": 4,
        "hidden_dim": 8,
        "use_cuda": False,
        "seed": 0,
    }
    json_out = tmp_path / "hll.json"
    csv_out = tmp_path / "hll.csv"
    run_single(
        experiment=EXPERIMENT_HLL_MERGE_LEARNING,
        config=cfg,
        json_out=json_out,
        csv_out=csv_out,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert "runtime_meta" in payload
    assert len(payload["rows"]) == 1
    assert "collapse_indicator" in payload["rows"][0]
    assert payload["rows"][0]["model_kind"] == "induced_projection"
    assert payload["rows"][0]["objective_mode"] == "corrected_local_law"
    assert csv_out.exists()


def test_runner_single_hll_merge_learning_direct_legacy_writes_json_csv(tmp_path: Path) -> None:
    _torch_or_skip()
    from treepo.bench.runner import EXPERIMENT_HLL_MERGE_LEARNING, run_single

    cfg = {
        "model_kind": "direct_state_mlp",
        "precisions": [6],
        "train_docs_grid": [8],
        "audit_policies": ["all"],
        "min_tokens": 64,
        "max_tokens": 64,
        "leaf_size": 32,
        "n_test": 8,
        "n_epochs": 1,
        "batch_docs": 4,
        "hidden_dim": 8,
        "use_cuda": False,
        "seed": 0,
    }
    json_out = tmp_path / "hll_direct.json"
    csv_out = tmp_path / "hll_direct.csv"
    run_single(
        experiment=EXPERIMENT_HLL_MERGE_LEARNING,
        config=cfg,
        json_out=json_out,
        csv_out=csv_out,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["model_kind"] == "direct_state_mlp"
    assert payload["rows"][0]["objective_mode"] == "direct_state_mse"
    assert csv_out.exists()
