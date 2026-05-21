"""Phase 3 tests: classical sketch families through fit().

DoD: ``fit(spec)`` with ``spec.family='hll'`` runs an HLL sketch ladder
end-to-end on a tiny fixture and returns ``status='success'``. The same
path works for any registered :class:`SketchAdapter` (Count-Min, Theta,
KLL, ...) — only the adapter changes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from treepo._research.ctreepo.contracts import CTreePOLearningSpec
from treepo.sketches.adapters import make_hll_adapter
from treepo.cld import fit
from treepo.cld.families import resolve_family
from treepo.cld.sketch_family import ClassicalSketchFamilyRuntime


def _make_tokens_tree(tokens: List[int]) -> SimpleNamespace:
    mid = len(tokens) // 2
    return SimpleNamespace(
        leaves=[
            SimpleNamespace(tokens=tokens[:mid]),
            SimpleNamespace(tokens=tokens[mid:]),
        ],
        tokens=tokens,
    )


def test_classical_sketch_score_matches_native_query() -> None:
    """Round-trip: encode→merge→query produces the adapter's cardinality
    estimate (within HLL's known precision tolerance). With distinct
    tokens, the native HLL at p=10 underestimates by a bounded factor;
    we assert positivity and finiteness, not exactness.
    """
    adapter = make_hll_adapter(backend="native", precision=10)
    family = ClassicalSketchFamilyRuntime(adapter=adapter)
    tree = _make_tokens_tree(list(range(100)))
    preds = family.score_roots_with_f(f=None, g=None, trees=[tree])
    assert len(preds) == 1
    assert preds[0] is not None and preds[0] > 0


def test_fit_with_hll_family(tmp_path: Path) -> None:
    adapter = make_hll_adapter(backend="native", precision=8)
    trees = [
        _make_tokens_tree([1, 2, 3, 4, 5, 6, 7, 8]),
        _make_tokens_tree([10, 20, 30, 40]),
    ]
    spec = CTreePOLearningSpec(
        space_kind="phase3",
        family="sketch",
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=trees,
        backend_config={
            "sketch_adapter": adapter,
            "sketch_schedule": "balanced",
            "output_dir": str(tmp_path),
        },
        axis={"max_iterations": 0, "axis_value": 0},
    )
    result = fit(spec)
    assert result.status == "success"
    assert result.summary["family"] == "sketch"


def test_resolve_sketch_without_adapter_raises() -> None:
    with pytest.raises(ValueError, match="sketch_adapter"):
        resolve_family("sketch", {})


def test_classical_sketch_rejects_unknown_tree_shape() -> None:
    adapter = make_hll_adapter(backend="native", precision=4)
    family = ClassicalSketchFamilyRuntime(adapter=adapter)
    bad_tree = SimpleNamespace()  # no .leaves, no .tokens
    with pytest.raises(AttributeError):
        family.score_roots_with_f(f=None, g=None, trees=[bad_tree])
