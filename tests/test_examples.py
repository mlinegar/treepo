from __future__ import annotations

from pathlib import Path

import yaml

from treepo.bench.runner import run_single, validate_config_dict


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "research"


def _load(name: str) -> dict:
    payload = yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_runnable_example_configs_validate() -> None:
    validate_config_dict("cardinality-recovery", _load("bench/cardinality_recovery.yaml"))
    validate_config_dict("hll-merge-learning", _load("bench/hll_merge_learning.yaml"))
    validate_config_dict("classical-sketches", _load("bench/classical_sketches.yaml"))
    validate_config_dict("longbench-runtime", _load("runtime/runtime_all_methods.yaml"))


def test_runtime_role_examples_are_clear_and_parseable() -> None:
    full_context = _load("runtime/runtime_llm_full_context.yaml")
    retrieval = _load("runtime/runtime_embedding_retrieval.yaml")
    summary = _load("runtime/runtime_summary_tree.yaml")
    fno = _load("runtime/runtime_fno_state_model.yaml")
    all_methods = _load("runtime/runtime_all_methods.yaml")

    assert full_context["methods"] == ["full_context"]
    assert "scorer" in full_context
    assert retrieval["methods"] == ["retrieval"]
    assert "embedder" in retrieval
    assert summary["methods"] == ["summary_tree"]
    assert "summarizer" in summary
    assert fno["methods"] == ["neural_operator"]
    assert fno["state_model"]["kind"] == "native_fno"
    assert all_methods["methods"] == [
        "full_context",
        "retrieval",
        "summary_tree",
        "state_tree",
        "neural_operator",
    ]


def test_longbench_runtime_example_runs_all_methods(tmp_path: Path) -> None:
    run_single(
        experiment="longbench-runtime",
        config=_load("runtime/runtime_all_methods.yaml"),
        json_out=tmp_path / "runtime.json",
        csv_out=tmp_path / "runtime.csv",
    )
    assert (tmp_path / "runtime.json").exists()
    assert (tmp_path / "runtime.csv").exists()
