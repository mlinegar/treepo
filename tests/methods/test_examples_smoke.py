"""Smoke-run the CPU-only ``examples/research/methods`` walkthroughs.

The examples are the user-facing ``fit()``/``run()`` walkthrough documents;
they previously drifted (stale imports, wrong fixture types, configs that
violate validators) because nothing in CI executed them. Each test here runs
one example exactly as a reader would: ``python examples/research/methods/
run_X.py --output-dir <tmp>`` with its default TOML.

The two examples needing external services are excluded:
``run_manifesto_fg_compile.py`` (live vLLM) and ``run_markov_probe.py``
(GPU probe subprocess).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples" / "research" / "methods"

CPU_EXAMPLES = [
    "run_hll_sketch.py",
    "run_markov_oracle.py",
    "run_lda_oracle.py",
    "run_lda_recovery.py",
]


def _run_example(name: str, tmp_path: Path, *extra: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name), "--output-dir", str(tmp_path), *extra],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"{name} exited {proc.returncode}\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    return proc.stdout


@pytest.mark.parametrize("name", CPU_EXAMPLES)
def test_cpu_example_runs_with_default_config(name: str, tmp_path: Path) -> None:
    stdout = _run_example(name, tmp_path)
    if "status=" in stdout:
        assert "status=success" in stdout, stdout


def test_fno_family_example_runs_with_default_config(tmp_path: Path) -> None:
    pytest.importorskip("neuralop")
    stdout = _run_example("run_fno_family.py", tmp_path, "--epochs", "1")
    assert "status=success" in stdout, stdout
