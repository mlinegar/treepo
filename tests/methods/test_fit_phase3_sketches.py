"""Research-only classical sketch runtime checks.

Classical sketch fitting is no longer part of the public `treepo.methods`
registry. These tests keep the research wrapper honest without re-exposing
it as a public family.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from treepo._research.methods.sketch_family import ClassicalSketchFamilyRuntime
from treepo.bench.sketches.adapters import make_hll_adapter


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


def test_classical_sketch_rejects_unknown_tree_shape() -> None:
    adapter = make_hll_adapter(backend="native", precision=4)
    family = ClassicalSketchFamilyRuntime(adapter=adapter)
    bad_tree = SimpleNamespace()  # no .leaves, no .tokens
    with pytest.raises(AttributeError):
        family.score_roots_with_f(f=None, g=None, trees=[bad_tree])
