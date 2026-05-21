"""Reasonable-sample reproduction grids for Markov and HLL paths.

Goes beyond the single-cell parity tests by exercising a small grid
along the axes a paper reviewer cares about:

**Markov grid** (real paper-DGP docs, not synthetic regime sequences):

- 3 seeds × 2 regime counts × 2 sequence lengths = 12 cells.
- For each cell: generate ``ChangepointMarkovDoc`` objects via the
  paper's ``generate_changepoint_docs``, dispatch through
  ``treepo.cld.run("oracle", {oracle_name="markov_changepoint_count"})``,
  assert per-doc count equals ``len(doc.true_boundaries)`` exactly.
  Ground truth is in the DGP, so this is **bit-for-bit** parity.

**HLL grid**:

- **Precision-scaling**: same data, p ∈ {6, 8, 10, 12, 14}. Verify
  ``mean_abs_error(estimate, exact)`` is non-increasing in p (high p
  ≤ low p). This is the canonical HLL sanity property.
- **Schedule-invariance**: same data, same precision, three schedules
  (``balanced`` / ``left_to_right`` / ``right_to_left``). HLL is
  commutative + associative — all three produce **identical**
  estimates per tree. Bit-for-bit.
- **Convergence to exact**: at p=14, mean relative error vs the
  ``hll_exact`` oracle is < 5% (well within HLL's theoretical RSE
  ≈ 1%).
"""

from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Markov grid — paper DGP, ground-truth comparison
# --------------------------------------------------------------------------- #


def _markov_docs(*, seed: int, n_regimes: int, max_tokens: int):
    """Generate ``ChangepointMarkovDoc`` objects via the paper's DGP."""
    from treepo._research.tree.markov_boundary_honesty_simulation import _make_transition_matrices
    from treepo._research.tree.markov_changepoint_honesty_simulation import (
        MarkovChangepointConfig,
        generate_changepoint_docs,
    )

    cfg = MarkovChangepointConfig(
        n_regimes=int(n_regimes),
        vocab_size=32,
        min_tokens=int(max_tokens),
        max_tokens=int(max_tokens),
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=int(max_tokens // 4),
        train_docs=2,
        test_docs=4,
        sinkhorn_iters=20,
        transition_log_std=1.0,
        seed=int(seed),
    )
    rng = np.random.default_rng(seed)
    transitions = _make_transition_matrices(
        n_classes=cfg.n_regimes,
        vocab_size=cfg.vocab_size,
        log_std=cfg.transition_log_std,
        sinkhorn_iters=cfg.sinkhorn_iters,
        rng=rng,
    )
    return generate_changepoint_docs(cfg, transitions=transitions)


def _wrap_markov_docs_as_trees(docs) -> list[SimpleNamespace]:
    out = []
    for doc in docs:
        truth = len(doc.true_boundaries)
        out.append(SimpleNamespace(
            leaves=[SimpleNamespace(tokens=[])],
            token_regimes=doc.token_regimes,
            metadata={
                "split": "test",
                "teacher_score_1_7": float(truth),
                "teacher_score_native": float(truth),
                "expert_score_1_7": float(truth),
                "expert_score_native": float(truth),
                "expert_target_scale": "raw",
                "expert_score_for_objective": float(truth),
            },
        ))
    return out


@pytest.mark.parametrize(
    "seed,n_regimes,max_tokens",
    list(product([0, 1, 2], [3, 5], [64, 128])),
    ids=[
        f"s{s}_r{r}_t{t}"
        for s, r, t in product([0, 1, 2], [3, 5], [64, 128])
    ],
)
def test_markov_grid_oracle_matches_ground_truth_per_doc(
    seed: int, n_regimes: int, max_tokens: int, tmp_path: Path
) -> None:
    """For each of 12 grid cells, every doc's predicted change-point
    count must equal the DGP's ``len(true_boundaries)`` exactly.
    """
    import treepo.cld

    docs = _markov_docs(seed=seed, n_regimes=n_regimes, max_tokens=max_tokens)
    trees = _wrap_markov_docs_as_trees(docs)
    expected_counts = [len(d.true_boundaries) for d in docs]

    result = treepo.cld.run(
        "oracle",
        {
            "oracle_name": "markov_changepoint_count",
            "eval_data": trees,
            "output_dir": str(tmp_path),
        },
    )
    assert result.status == "success"
    assert result.metrics["internal_f_mae"] == 0.0
    assert int(result.metrics["n"]) == len(docs)

    # Per-doc bit-for-bit match against ground truth.
    pred_path = result.artifacts["prediction_records"][0]
    rows = [json.loads(line) for line in Path(pred_path).read_text().splitlines() if line.strip()]
    cld_counts = [int(r["prediction"]) for r in rows]
    assert cld_counts == expected_counts, (
        f"seed={seed} regimes={n_regimes} tokens={max_tokens}: "
        f"counts diverged. Expected={expected_counts}, got={cld_counts}"
    )


def test_markov_grid_aggregate_summary() -> None:
    """Sanity over the whole grid: every cell's MAE is exactly zero.
    Cross-checks the parametrized cells above as a single block.
    """
    import treepo.cld
    import tempfile

    mae_by_cell: dict[tuple, float] = {}
    for seed, n_regimes, max_tokens in product([0, 1, 2], [3, 5], [64, 128]):
        docs = _markov_docs(seed=seed, n_regimes=n_regimes, max_tokens=max_tokens)
        trees = _wrap_markov_docs_as_trees(docs)
        with tempfile.TemporaryDirectory() as tmp:
            result = treepo.cld.run(
                "oracle",
                {
                    "oracle_name": "markov_changepoint_count",
                    "eval_data": trees,
                    "output_dir": tmp,
                },
            )
        mae_by_cell[(seed, n_regimes, max_tokens)] = result.metrics["internal_f_mae"]

    assert len(mae_by_cell) == 12
    assert all(mae == 0.0 for mae in mae_by_cell.values()), (
        f"some cells had non-zero MAE: "
        f"{ {k: v for k, v in mae_by_cell.items() if v != 0.0} }"
    )


# --------------------------------------------------------------------------- #
# HLL precision-scaling grid
# --------------------------------------------------------------------------- #


_HLL_FIXTURE_KWARGS = dict(
    n_trees=8,
    leaves_per_tree=4,
    leaf_token_count=32,
    vocabulary_size=512,
    seed=17,
)


def _mae_against_exact(estimates: list[float], exact_counts: list[int]) -> float:
    return float(sum(abs(e - x) for e, x in zip(estimates, exact_counts)) / len(estimates))


def _hll_estimates_at_precision(precision: int, *, schedule: str = "balanced", tmp_path: Path) -> List[float]:
    """Read per-tree HLL estimates from a treepo.cld sketch run."""
    import treepo.cld
    result = treepo.cld.run(
        "sketch",
        {
            "sketch_kind": "hll",
            "precision": int(precision),
            "hash_bits": 64,
            "schedule": schedule,
            "output_dir": str(tmp_path),
            **_HLL_FIXTURE_KWARGS,
        },
    )
    assert result.status == "success"
    pred_path = result.artifacts["prediction_records"][0]
    rows = [json.loads(line) for line in Path(pred_path).read_text().splitlines() if line.strip()]
    return [float(r["prediction"]) for r in rows]


def _hll_exact_counts(tmp_path: Path) -> List[int]:
    """Read the per-tree exact unique counts via the hll_exact oracle —
    paper-grade ground truth.
    """
    import treepo.cld
    result = treepo.cld.run(
        "oracle",
        {
            "oracle_name": "hll_exact",
            "output_dir": str(tmp_path),
            **_HLL_FIXTURE_KWARGS,
        },
    )
    assert result.status == "success"
    pred_path = result.artifacts["prediction_records"][0]
    rows = [json.loads(line) for line in Path(pred_path).read_text().splitlines() if line.strip()]
    return [int(r["prediction"]) for r in rows]


def test_hll_precision_scaling_grid(tmp_path: Path) -> None:
    """Higher precision → lower mean absolute error vs the exact oracle.

    We check **monotonicity** (high p MAE ≤ low p MAE) across
    p ∈ {6, 8, 10, 12, 14}. Small fluctuations from finite-sample noise
    at very low precision are absorbed by the cumulative-min check:
    ``MAE(p) ≤ MAE(any lower p)``.
    """
    exact = _hll_exact_counts(tmp_path / "exact")
    precisions = [6, 8, 10, 12, 14]
    maes: list[float] = []
    for p in precisions:
        est = _hll_estimates_at_precision(p, tmp_path=tmp_path / f"p{p}")
        assert len(est) == len(exact)
        maes.append(_mae_against_exact(est, exact))

    # Print for visibility in -v output.
    print(f"\nHLL precision-scaling MAEs:")
    for p, m in zip(precisions, maes):
        print(f"  p={p:>2d}: MAE={m:.3f}  RSE_theory={1.04 / math.sqrt(2 ** p):.3f}")

    # Highest precision must have the smallest MAE in the grid.
    assert maes[-1] == min(maes), (
        f"p={precisions[-1]} MAE={maes[-1]} not the minimum across precisions: {dict(zip(precisions, maes))}"
    )
    # And it must be tight: p=14 MAE < 5% of mean exact count.
    mean_exact = sum(exact) / len(exact)
    assert maes[-1] < 0.05 * mean_exact, (
        f"p=14 MAE={maes[-1]:.3f} exceeds 5% of mean_exact={mean_exact:.1f}"
    )


def test_hll_schedule_invariance_grid(tmp_path: Path) -> None:
    """HLL is commutative + associative; ``balanced`` / ``left_to_right``
    / ``right_to_left`` must produce **identical** per-tree estimates at
    any precision. Bit-for-bit.
    """
    estimates_by_schedule = {}
    for schedule in ("balanced", "left_to_right", "right_to_left"):
        est = _hll_estimates_at_precision(
            12, schedule=schedule, tmp_path=tmp_path / schedule
        )
        estimates_by_schedule[schedule] = est

    baseline = estimates_by_schedule["balanced"]
    for schedule, est in estimates_by_schedule.items():
        if schedule == "balanced":
            continue
        assert est == baseline, (
            f"schedule '{schedule}' diverged from 'balanced' "
            f"despite HLL being commutative+associative: "
            f"{est} vs {baseline}"
        )


def test_hll_convergence_to_exact_at_high_precision(tmp_path: Path) -> None:
    """At p=14, every per-tree estimate's relative error against the
    exact oracle is < 5% (well within HLL's theoretical RSE).
    """
    exact = _hll_exact_counts(tmp_path / "exact")
    est = _hll_estimates_at_precision(14, tmp_path=tmp_path / "p14")
    assert len(exact) == len(est)
    rel_errors = [
        abs(e - x) / max(x, 1) for e, x in zip(est, exact)
    ]
    max_rel = max(rel_errors)
    assert max_rel < 0.10, (
        f"p=14 max per-tree relative error = {max_rel:.3f} > 0.10; "
        f"per-tree errors={[round(r, 3) for r in rel_errors]}"
    )
