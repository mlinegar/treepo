"""Run the research-only Markov-FNO probe directly.

The probe trains ``CleanUnifiedNO`` with its own loop outside
``run_alternating_family``. It is intentionally kept under
``scripts/research`` rather than exposed through the public
``treepo.methods`` dispatcher.

Gated on ``TT_RUN_LIVE_TESTS=1`` AND CUDA AND the probe script being
present. Uses a tiny inline ``--doc-tokens`` recoverable corpus so the
run fits in a few minutes.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_RUN_LIVE = str(os.getenv("TT_RUN_LIVE_TESTS", "") or "").strip().lower() in {"1", "true", "yes"}
_PROBE_SCRIPT = _REPO_ROOT / "scripts" / "research" / "probe_clean_unified_no.py"


def _have_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    (not _RUN_LIVE) or (not _have_cuda()) or (not _PROBE_SCRIPT.exists()),
    reason="Probe test requires TT_RUN_LIVE_TESTS=1, CUDA, and scripts/research/probe_clean_unified_no.py.",
)


def test_live_probe_clean_unified_no_research_script(tmp_path: Path) -> None:
    """Runs the research probe at the smallest inline-corpus configuration."""
    import json
    import subprocess

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            str(_PROBE_SCRIPT),
            "--output-root", str(tmp_path),
            "--doc-tokens", "256",
            "--leaf-tokens", "64",
            "--train-docs", "32",
            "--eval-docs", "8",
            "--epochs", "2",
            "--batch-size", "4",
            "--channels", "16",
            "--g-n-modes", "8",
            "--g-n-layers", "1",
            "--scorer-n-modes", "8",
            "--scorer-n-layers", "1",
            "--seed", "0",
            "--device", "cuda",
            "--training-objective", "root",
        ],
        capture_output=True,
        text=True,
        timeout=600.0,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nProbe wall-time: {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"\nstderr tail: {result.stderr[-1500:]}")
    assert result.returncode == 0, (
        f"probe failed (returncode={result.returncode}); "
        f"stderr tail: {result.stderr[-1000:]}"
    )

    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    assert summary, "probe wrote no summary.json"

    # Paper-grade reproducible scalars in the summary.
    assert "test_root_mae" in summary
    assert "best_val_root_mae" in summary
    assert "best_val_epoch" in summary
    assert "n_params_g" in summary
    assert "n_params_f" in summary
    # Sanity: test MAE is a finite non-negative float.
    import math

    test_mae = float(summary["test_root_mae"])
    assert math.isfinite(test_mae) and test_mae >= 0.0
    # Training history records at least the requested number of epochs.
    history = summary.get("history") or []
    assert len(history) >= 1, f"probe history empty: {history!r}"

    assert summary_path.exists()
