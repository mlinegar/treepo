from __future__ import annotations

import json
from pathlib import Path

import pytest

from treepo.bench.runner import VALID_EXPERIMENTS, run_single
from treepo.methods.oracles import score_oracle


def test_markov_is_public_task_benchmark() -> None:
    assert "markov" in VALID_EXPERIMENTS


def test_markov_oracle_fixture_is_exact(tmp_path: Path) -> None:
    result = score_oracle(
        {
            "oracle_name": "markov_changepoint_count",
            "n_trees": 6,
            "n_states": 3,
            "doc_tokens": 64,
            "doc_unit_kind": "token",
            "leaf_unit_count": 8,
            "transition_prob": 0.2,
            "seed": 11,
            "output_dir": str(tmp_path / "methods"),
        },
    )
    payload = result.to_dict()
    assert payload["status"] == "success"
    assert payload["metrics"]["n"] == 6
    assert payload["metrics"]["internal_f_mae"] == 0.0
    assert payload["metrics"]["external_expert_mae"] == 0.0


def test_treepo_bench_markov_writes_json_csv(tmp_path: Path) -> None:
    run_single(
        experiment="markov",
        config={
            "method": "oracle",
            "scorer": "markov_changepoint_count",
            "seed": 3,
            "split": "test",
            "n_trees": 5,
            "task_config": {
                "n_states": 3,
                "doc_tokens": 64,
                "doc_unit_kind": "token",
                "leaf_unit_count": 8,
                "transition_prob": 0.25,
                "vocabulary_size": 96,
            },
        },
        json_out=tmp_path / "markov.json",
        csv_out=tmp_path / "markov.csv",
    )
    payload = json.loads((tmp_path / "markov.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["experiment"] == "markov"
    assert row["method"] == "oracle"
    assert row["scorer"] == "markov_changepoint_count"
    assert row["oracle_name"] == "markov_changepoint_count"
    assert row["n"] == 5
    assert row["internal_f_mae"] == 0.0
    assert row["external_expert_mae"] == 0.0
    assert payload["config"]["task_config"]["doc_tokens"] == 64
    assert (tmp_path / "markov.csv").exists()


def test_markov_rejects_unknown_task_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown task_config keys"):
        run_single(
            experiment="markov",
            config={"task_config": {"not_a_markov_knob": True}},
            json_out=tmp_path / "markov.json",
            csv_out=tmp_path / "markov.csv",
        )
