"""Phase 2 tests: family registry + oracle dispatch through fit().

DoD: ``resolve_family('oracle', backend_config)`` returns an
``OracleFamilyRuntime``; ``fit(spec)`` with ``spec.family='oracle'`` runs
end-to-end on a minimal tree fixture using the ``hll_exact`` oracle.
The heavy families (``"fno"`` / ``"dspy"`` / ``"trl"``) are exercised at
the *factory* level only — we verify they raise informative errors when
their required config is missing, without actually instantiating them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from treepo._research.ctreepo.contracts import CTreePOFitResult, CTreePOLearningSpec
from treepo._research.ctreepo.oracles.runtime import OracleFamilyRuntime
from treepo.cld import fit
from treepo.cld.families import list_families, resolve_family


def test_registry_has_all_v1_families() -> None:
    names = list_families()
    for required in ("oracle", "fno", "dspy", "trl", "sketch", "learnable_constant"):
        assert required in names, f"missing family {required!r} in registry"


def test_resolve_oracle_returns_oracle_family_runtime() -> None:
    family = resolve_family("oracle", {"oracle_name": "hll_exact"})
    assert isinstance(family, OracleFamilyRuntime)
    assert family.name == "oracle:hll_exact"


def test_resolve_oracle_missing_name_raises() -> None:
    with pytest.raises(ValueError, match="oracle_name"):
        resolve_family("oracle", {})


def test_resolve_unknown_family_raises() -> None:
    with pytest.raises(KeyError, match="not registered"):
        resolve_family("does_not_exist", {})


def test_resolve_fno_without_config_raises() -> None:
    # The registry validates required keys before importing torch, so this
    # raises ValueError on this machine regardless of whether torch is present.
    with pytest.raises((ValueError, ImportError)):
        resolve_family("fno", {})


def test_resolve_dspy_without_config_raises() -> None:
    with pytest.raises((ValueError, ImportError)):
        resolve_family("dspy", {})


def test_resolve_trl_without_config_raises() -> None:
    with pytest.raises((ValueError, ImportError)):
        resolve_family("trl", {})


def _make_hll_tree(tokens: List[int]) -> SimpleNamespace:
    """Two-leaf tree whose tokens split across leaves. Each leaf carries
    .tokens, so the ``hll_exact`` oracle's ``_score_tree_hll_exact`` adapter
    can recover the flat token list via ``tree.leaves``.
    """
    mid = len(tokens) // 2
    left = SimpleNamespace(tokens=tokens[:mid])
    right = SimpleNamespace(tokens=tokens[mid:])
    return SimpleNamespace(leaves=[left, right], tokens=tokens)


def test_fit_with_oracle_hll_exact_runs(tmp_path: Path) -> None:
    trees = [
        _make_hll_tree([1, 2, 3, 4, 5]),
        _make_hll_tree([2, 2, 3, 6, 7, 8]),
        _make_hll_tree([9, 10]),
    ]
    spec = CTreePOLearningSpec(
        space_kind="phase2",
        family="oracle",
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=trees,
        backend_config={
            "oracle_name": "hll_exact",
            "output_dir": str(tmp_path),
        },
        axis={"max_iterations": 0, "axis_value": 0},
    )
    result = fit(spec)
    assert isinstance(result, CTreePOFitResult)
    assert result.status == "success"
    assert result.summary["family"] == "oracle"


def test_fit_with_lda_oracle_resolves(tmp_path: Path) -> None:
    """Phase 5 minimum: the LDA oracle resolves and ``fit()`` runs the loop.

    We pass ``eval_data=[]`` so the oracle's tree-shape requirements
    (theta / W_base / lambda_multiplier) don't matter here. The full LDA
    exercise (exercise 4) needs real synthetic-LDA trees from
    ``scripts/run_lda_tree_recovery_simulation.py``; this test only proves
    the dispatch path.
    """
    spec = CTreePOLearningSpec(
        space_kind="phase5",
        family="oracle",
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=[],
        backend_config={
            "oracle_name": "leaf_local_mixture_target",
            "output_dir": str(tmp_path),
        },
        axis={"max_iterations": 0, "axis_value": 0},
    )
    result = fit(spec)
    assert result.status == "success"


def test_injected_family_runtime_still_works(tmp_path: Path) -> None:
    """The Phase 1 escape hatch (``backend_config['family_runtime']``) keeps
    working even after the registry exists. This is the way ad-hoc / custom
    FamilyRuntime implementations are tested.
    """

    class _NoopFamily:
        name = "noop"

        def train_f(self, *, f_init, g, traces, output_dir, iteration):
            return f_init

        def train_g(self, *, g_init, f, traces, output_dir, iteration):
            return g_init

        def score_roots_with_f(self, *, f, g, trees):
            return [None] * len(trees)

        def validate_artifact(self, *, kind, artifact):
            return None

    spec = CTreePOLearningSpec(
        space_kind="phase2",
        family="should_be_ignored",
        schedule="fg",
        initial_artifacts={"f": "f0", "g": "g0"},
        train_data=[],
        eval_data=[],
        backend_config={
            "family_runtime": _NoopFamily(),
            "output_dir": str(tmp_path),
        },
        axis={"max_iterations": 0, "axis_value": 0},
    )
    result = fit(spec)
    assert result.status == "success"
    assert result.summary["family"] == "should_be_ignored"  # spec value retained
