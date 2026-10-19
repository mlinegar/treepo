from __future__ import annotations

import subprocess
import sys
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


def test_quickstart_audit_recovers_merge_violation_rate() -> None:
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    rows = {line.split()[0]: line.split() for line in proc.stdout.splitlines()[1:]}
    assert set(rows) == {"leaf_preservation", "merge_preservation", "on_range_idempotence"}
    true_rate, proxy_rate, audit_rate = (float(x) for x in rows["merge_preservation"][1:4])
    assert abs(audit_rate - true_rate) < 0.1
    assert proxy_rate - true_rate > 0.1
