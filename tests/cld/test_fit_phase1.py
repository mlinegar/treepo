"""Phase 1 smoke test: fit() runs end-to-end with an injected no-op family.

DoD for Phase 1: ``from treepo.cld import fit; fit(spec)`` returns a
``CTreePOFitResult(status="success", ...)`` when given a stub FamilyRuntime.
No registry, no real backend, no laws — just the orchestration plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

from treepo._research.ctreepo.contracts import CTreePOFitResult, CTreePOLearningSpec
from treepo.cld import fit


class _NoopFamily:
    """Minimum viable FamilyRuntime: no-op train, no-op score."""

    name = "noop"

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        return f_init

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        return g_init

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> List[Optional[float]]:
        return [None] * len(trees)

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        return None


def _make_spec(tmp_path: Path, *, max_iterations: int) -> CTreePOLearningSpec:
    return CTreePOLearningSpec(
        space_kind="phase1",
        family="noop",
        schedule="fg",
        initial_artifacts={"f": "f0", "g": "g0"},
        train_data=[],
        eval_data=[],
        backend_config={
            "family_runtime": _NoopFamily(),
            "output_dir": str(tmp_path),
            "first_train_side": "f",
        },
        axis={"max_iterations": max_iterations, "axis_value": 0},
    )


def test_fit_returns_success_for_zero_iterations(tmp_path: Path) -> None:
    result = fit(_make_spec(tmp_path, max_iterations=0))

    assert isinstance(result, CTreePOFitResult)
    assert result.status == "success"
    assert result.summary["family"] == "noop"
    assert result.summary["n_iterations"] == 1  # k=0 always produces one record
    assert result.artifacts["f"] == "f0"
    assert result.artifacts["g"] == "g0"
    assert isinstance(result.history, list)
    assert len(result.history) == 1


def test_fit_runs_one_iteration(tmp_path: Path) -> None:
    result = fit(_make_spec(tmp_path, max_iterations=1))

    assert result.status == "success"
    # k=0 (no train) + k=1 (trains f under first_train_side='f') -> 2 records
    assert result.summary["n_iterations"] == 2
    assert result.artifacts["f"] == "f0"  # noop train returns f_init
    assert result.artifacts["g"] == "g0"


def test_fit_rejects_unresolvable_family(tmp_path: Path) -> None:
    """When neither ``family_runtime`` is injected nor ``spec.family`` is in the
    registry, ``fit()`` raises ``KeyError`` from the registry lookup. (Phase 1
    raised ``NotImplementedError``; Phase 2's registry makes the failure
    mode "unknown family name", which is the more useful error.)
    """
    spec = CTreePOLearningSpec(
        space_kind="phase1",
        family="not_a_registered_family",
        schedule="fg",
        initial_artifacts={"f": "f0", "g": "g0"},
        train_data=[],
        eval_data=[],
        backend_config={"output_dir": str(tmp_path)},
        axis={"max_iterations": 0},
    )
    try:
        fit(spec)
    except KeyError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("expected KeyError for unregistered family")


def test_fit_rejects_empty_family_and_no_injection(tmp_path: Path) -> None:
    """Empty ``spec.family`` with no injected runtime is an unrecoverable spec."""
    spec = CTreePOLearningSpec(
        space_kind="phase1",
        family="",
        schedule="fg",
        initial_artifacts={"f": "f0", "g": "g0"},
        train_data=[],
        eval_data=[],
        backend_config={"output_dir": str(tmp_path)},
        axis={"max_iterations": 0},
    )
    try:
        fit(spec)
    except ValueError as exc:
        assert "family" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for empty spec.family")
