"""Reuse the paper's standalone Markov-FNO probe verbatim through
``treepo.cld.run("probe", ...)``.

The probe (``scripts/probe_clean_unified_no.py``) trains
``CleanUnifiedNO`` with its own loop outside ``run_alternating_family``.
Rather than rewriting it, the ``"probe"`` method subprocess-dispatches
the paper script with user CLI args and returns the parsed
``summary.json`` it writes. Same code, different invocation surface.

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


_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_RUN_LIVE = str(os.getenv("TT_RUN_LIVE_TESTS", "") or "").strip().lower() in {"1", "true", "yes"}
_PROBE_SCRIPT = _REPO_ROOT / "scripts" / "probe_clean_unified_no.py"


def _have_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    (not _RUN_LIVE) or (not _have_cuda()) or (not _PROBE_SCRIPT.exists()),
    reason="Probe test requires TT_RUN_LIVE_TESTS=1, CUDA, and scripts/probe_clean_unified_no.py.",
)


def test_live_probe_clean_unified_no_through_treepo_cld(tmp_path: Path) -> None:
    """Runs the paper's standalone Markov-FNO probe at the smallest
    inline-corpus configuration, dispatched through ``treepo.cld.run(
    "probe", ...)`` so the probe is reachable from the centralized
    surface without losing any of its existing behavior.
    """
    import treepo.cld

    t0 = time.perf_counter()
    result = treepo.cld.run(
        "probe",
        {
            "output_root": str(tmp_path),
            "doc_tokens": 256,            # inline recoverable corpus
            "leaf_tokens": 64,            # 4 leaves/doc
            "train_docs": 32,
            "eval_docs": 8,
            "epochs": 2,
            "batch_size": 4,
            "channels": 16,
            "g_n_modes": 8,
            "g_n_layers": 1,
            "scorer_n_modes": 8,
            "scorer_n_layers": 1,
            "seed": 0,
            "device": "cuda",
            "training_objective": "root",
            "timeout": 600.0,
        },
    )
    elapsed = time.perf_counter() - t0
    print(f"\nProbe wall-time: {elapsed:.1f}s")

    if result["status"] != "success":
        print(f"\nstderr tail: {result['stderr_tail'][-1500:]}")
    assert result["status"] == "success", (
        f"probe failed (returncode={result['returncode']}); "
        f"stderr tail: {result['stderr_tail'][-1000:]}"
    )

    summary = result["summary"]
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

    # Sanity that the summary.json lives where treepo.cld says it does.
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
