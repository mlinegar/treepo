from __future__ import annotations

from pathlib import Path

import yaml

from treepo.bench.runner import run_single, validate_config_dict


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str) -> dict:
    payload = yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_runnable_example_configs_validate() -> None:
    validate_config_dict("classical-sketches", _load("bench/classical_sketches.yaml"))
    validate_config_dict("markov", _load("bench/markov.yaml"))


def test_bench_example_runs_markov(tmp_path: Path) -> None:
    run_single(
        experiment="markov",
        config=_load("bench/markov.yaml"),
        json_out=tmp_path / "markov.json",
        csv_out=tmp_path / "markov.csv",
    )
    assert (tmp_path / "markov.json").exists()
    assert (tmp_path / "markov.csv").exists()
