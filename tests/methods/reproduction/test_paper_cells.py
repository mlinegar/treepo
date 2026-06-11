"""Paper-cell reproduction suite.

Each test takes a representative cell from the existing paper-code paths
and checks the package/research wrapper reaches the same number the
underlying function produces.

The existing parity wins (already locked elsewhere):

- Manifesto teacher metric (bit-for-bit on 23 trees) —
  ``tests/methods/test_manifesto_paper_parity.py``
- Live DSPy / Gemma manifesto cell (within 0.005 of paper Pearson) —
  ``tests/methods/integration/test_manifesto_dspy_live.py``
- LDA leaf-local-mixture oracle (bit-for-bit) —
  ``tests/methods/test_fit_real_lda.py``

This file adds four more cells across four different families:

1. HLL classical sketch — paper-native call vs research runtime wrapper
2. Markov change-point count — paper-native ``markov_changepoint_count``
   vs ``run("oracle", {"oracle_name": "markov_changepoint_count"})``
3. LDA tree recovery — paper script's subprocess output JSON vs
   direct call to ``run_lda_tree_recovery_experiment`` through
   ``treepo.methods.run("fit-raw", ...)``-equivalent invocation
4. Determinism — same spec, two runs, identical metrics
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TREEPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# Cell 1 — HLL classical sketch deterministic match
# --------------------------------------------------------------------------- #


def test_paper_cell_hll_classical_sketch_matches_native_call(tmp_path: Path) -> None:
    """The paper-native HLL pipeline calls
    ``make_hll_adapter(backend='native', precision=P)`` + ``treepo_reduce``
    on the same per-leaf token lists as the research runtime wrapper.
    Both must produce
    identical cardinality estimates — same seed, same precision,
    same adapter, same fold schedule.
    """
    from treepo._research.methods.sketch_family import ClassicalSketchFamilyRuntime
    from treepo.bench.sketches.adapters import make_hll_adapter
    from treepo.bench.sketches.tree_reducer import treepo_reduce

    from treepo.methods.fixtures import make_hll_token_trees

    fixture_kwargs = dict(
        n_trees=6, leaves_per_tree=4, leaf_token_count=16,
        vocabulary_size=128, seed=7,
    )
    trees = make_hll_token_trees(**fixture_kwargs)

    # Paper-native path: build adapter, fold each tree's leaf token lists.
    adapter = make_hll_adapter(backend="native", precision=12, hash_bits=64)
    native_estimates: list[float] = []
    for tree in trees:
        per_leaf = [list(leaf.tokens) for leaf in tree.leaves]
        root_state = treepo_reduce(per_leaf, adapter, schedule="balanced")
        native_estimates.append(float(adapter.query(root_state, None)))

    runtime = ClassicalSketchFamilyRuntime(adapter=adapter, schedule="balanced")
    runtime_estimates = [
        float(pred) for pred in runtime.score_roots_with_f(f=None, g=None, trees=trees)
    ]

    assert len(native_estimates) == len(runtime_estimates) == 6
    for i, (a, b) in enumerate(zip(native_estimates, runtime_estimates)):
        assert a == b, (
            f"tree {i}: paper-native HLL estimate {a} != research runtime {b}"
        )


# --------------------------------------------------------------------------- #
# Cell 2 — Markov change-point count oracle
# --------------------------------------------------------------------------- #


def test_paper_cell_markov_oracle_matches_native_call(tmp_path: Path) -> None:
    """The paper-native ``markov_changepoint_count`` counts transitions
    in a flat regime sequence. ``treepo.methods.run("oracle", {oracle_name:
    "markov_changepoint_count"})`` dispatches through the registered
    oracle's score_tree adapter, which calls the SAME function. Same
    input → identical output.
    """
    from treepo._research.ctreepo.oracles.markov import markov_changepoint_count

    import treepo.methods

    # Build a handful of synthetic regime sequences with known transition counts.
    rng = np.random.default_rng(13)
    n_trees = 8
    regimes_per_tree = [
        tuple(int(x) for x in rng.integers(0, 4, size=rng.integers(20, 50)))
        for _ in range(n_trees)
    ]
    native_counts = [markov_changepoint_count(seq) for seq in regimes_per_tree]

    # Wrap each as a tree with the attributes _score_tree_markov reads.
    trees = []
    for idx, seq in enumerate(regimes_per_tree):
        truth = native_counts[idx]
        trees.append(SimpleNamespace(
            leaves=[SimpleNamespace(tokens=[])],
            token_regimes=seq,
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

    result = treepo.methods.run(
        "oracle",
        {
            "oracle_name": "markov_changepoint_count",
            "eval_data": trees,
            "output_dir": str(tmp_path),
        },
    )
    assert result.status == "success"
    assert int(result.metrics["n"]) == n_trees
    # Oracle = ground truth by construction → MAE must be exactly zero.
    assert result.metrics["internal_f_mae"] == 0.0, (
        f"markov_changepoint_count via treepo.methods diverged from native; "
        f"MAE={result.metrics['internal_f_mae']}"
    )
    # And the per-tree predictions match the native counts bit-for-bit.
    pred_path = result.artifacts["prediction_records"][0]
    rows = [json.loads(line) for line in Path(pred_path).read_text().splitlines() if line.strip()]
    methods_preds = [int(r["prediction"]) for r in rows]
    assert methods_preds == native_counts


# --------------------------------------------------------------------------- #
# Cell 3 — LDA tree recovery: paper script output vs direct function call
# --------------------------------------------------------------------------- #


def _lda_recovery_config():
    """Tiny LDA recovery config — runs in ~3 seconds."""
    from treepo._research.ctreepo.sim.core.lda_tree_recovery import LDATreeRecoveryConfig

    return LDATreeRecoveryConfig(
        n_topics=4, vocab_size=64,
        min_tokens=64, max_tokens=64,
        anchor_words_per_topic=4,
        leaf_tokens=16,
        train_docs=4, test_docs=16,
        seed=0,
    )


def test_paper_cell_lda_recovery_subprocess_matches_direct_call(tmp_path: Path) -> None:
    """The paper script's subprocess output JSON vs the same function
    called in-process. ``treepo.methods`` doesn't yet have a registered
    family for LDA tree recovery (the script doesn't go through
    ``FamilyRuntime`` — it runs four baselines inline). But the
    underlying function ``run_lda_tree_recovery_experiment(config)`` is
    the same bit-for-bit invariant either way: subprocess or direct.

    This is the surgical reproduction: if the SAME function called the
    SAME way produces the SAME numbers, the dispatcher (or its absence)
    isn't perturbing results.
    """
    from treepo._research.ctreepo.sim.core.lda_tree_recovery import run_lda_tree_recovery_experiment

    cfg = _lda_recovery_config()

    # Subprocess run of the paper script with matching CLI args.
    json_out = tmp_path / "paper_lda.json"
    csv_out = tmp_path / "paper_lda.csv"
    cmd = [
        sys.executable,
        str(_TREEPO_ROOT / "scripts/research/run_lda_tree_recovery_simulation.py"),
        "--n-topics", str(cfg.n_topics),
        "--vocab-size", str(cfg.vocab_size),
        "--min-tokens", str(cfg.min_tokens),
        "--max-tokens", str(cfg.max_tokens),
        "--anchor-words-per-topic", str(cfg.anchor_words_per_topic),
        "--leaf-tokens", str(cfg.leaf_tokens),
        "--train-docs", str(cfg.train_docs),
        "--test-docs", str(cfg.test_docs),
        "--seed", str(cfg.seed),
        "--json-summary", str(json_out),
        "--csv-summary", str(csv_out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, f"paper script failed: {res.stderr[-500:]}"
    subprocess_summary = json.loads(json_out.read_text())

    # In-process call to the same function. Should produce identical numbers.
    direct_summary = run_lda_tree_recovery_experiment(cfg)
    direct_json = json.loads(direct_summary.to_json())

    # Both summaries carry an `exact_recovery` block with the headline
    # reproducible metrics. Compare bit-for-bit.
    for key in ("root_count_l1_mean", "root_pi_l1_mean",
                "root_utility_abs_mean", "root_loglik_abs_mean"):
        a = float(subprocess_summary["exact_recovery"][key])
        b = float(direct_json["exact_recovery"][key])
        assert a == b, (
            f"LDA recovery {key}: subprocess={a} != direct={b}"
        )

    # Method-level metrics (full_doc, exact_tree, leaf_average, leaf_utility_only)
    # also match bit-for-bit. NaN-equality is allowed when *both* are NaN
    # (some method/metric pairs are undefined at this tiny config).
    for method in ("full_doc", "exact_tree", "leaf_average", "leaf_utility_only"):
        sub_m = subprocess_summary["methods"][method]
        dir_m = direct_json["methods"][method]
        for key in ("pi_l1_to_true_mean", "utility_abs_to_true_mean"):
            if key in sub_m and key in dir_m:
                a = float(sub_m[key])
                b = float(dir_m[key])
                if math.isnan(a) and math.isnan(b):
                    continue
                assert a == b, (
                    f"LDA recovery method={method} {key}: subprocess={a} != direct={b}"
                )


# --------------------------------------------------------------------------- #
# Cell 4 — Determinism: same spec, two runs, identical metrics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "method,config",
    [
        (
            "oracle",
            {"oracle_name": "hll_exact", "seed": 42, "n_trees": 8,
             "vocabulary_size": 64},
        ),
        (
            "oracle",
            {"oracle_name": "leaf_local_mixture_target", "seed": 11, "_research_lda_fixture": True},
        ),
    ],
    ids=["oracle_hll_exact", "oracle_lda"],
)
def test_paper_cell_determinism_same_spec_two_runs(
    method: str, config: dict, tmp_path: Path,
) -> None:
    """Same call, same seed, twice. The metrics dict must be identical
    across the two runs. A reviewer asking 'is your pipeline
    deterministic?' should get a one-line proof here.

    Note: ``output_dir`` differs between the two runs (we deliberately
    use distinct temp dirs so neither call is reading a cached manifest).
    Metric identity must come from the computation, not from filesystem
    caching.
    """
    import treepo.methods

    # Cache might survive between tests via lru_cache on fixtures — that's
    # the intended behavior, but it doesn't affect the *metric* values.
    base_config = dict(config)
    if base_config.pop("_research_lda_fixture", False):
        from treepo._research.methods.lda_fixtures import make_leaf_local_mixture_trees

        trees, _cfg = make_leaf_local_mixture_trees(seed=int(base_config.get("seed", 0)))
        base_config["eval_data"] = list(trees)
    cfg_a = {**base_config, "output_dir": str(tmp_path / "run_a")}
    cfg_b = {**base_config, "output_dir": str(tmp_path / "run_b")}
    result_a = treepo.methods.run(method, cfg_a)
    result_b = treepo.methods.run(method, cfg_b)

    assert result_a.status == "success" and result_b.status == "success"
    # Every metric key must be present in both and equal bit-for-bit.
    assert set(result_a.metrics) == set(result_b.metrics)
    for key, value_a in result_a.metrics.items():
        value_b = result_b.metrics[key]
        # Allow NaN propagation as long as both are NaN.
        if isinstance(value_a, float) and math.isnan(value_a):
            assert isinstance(value_b, float) and math.isnan(value_b)
        else:
            assert value_a == value_b, (
                f"determinism: metric {key} drifted across runs "
                f"(a={value_a}, b={value_b})"
            )
