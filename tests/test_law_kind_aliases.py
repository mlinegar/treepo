"""The law alias table accepts all three spellings and stays single-sourced.

Every law has the paper's C-condition name, the Lean L-numbering, and the
Lean ``LocalLawChannel`` constructor name. ``treepo.objective`` must resolve
component names through the same table as ``treepo.local_law``.
"""

from __future__ import annotations

import pytest

from treepo.local_law import LAW_KIND_ALIASES, LawKind
from treepo.objective import (
    CANONICAL_LAW_COMPONENTS,
    canonical_law_component_weights,
)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("c1", LawKind.C1_LEAF),
        ("l1", LawKind.C1_LEAF),
        ("c1_leaf", LawKind.C1_LEAF),
        ("c2", LawKind.C2_IDEMPOTENCE),
        ("l3", LawKind.C2_IDEMPOTENCE),
        ("c2_idempotence", LawKind.C2_IDEMPOTENCE),
        ("c3", LawKind.C3_MERGE),
        ("l2", LawKind.C3_MERGE),
        ("c3_merge", LawKind.C3_MERGE),
        ("leaf_preservation", LawKind.C1_LEAF),
        ("on_range_idempotence", LawKind.C2_IDEMPOTENCE),
        ("merge_preservation", LawKind.C3_MERGE),
    ],
)
def test_law_kind_accepts_paper_lean_and_channel_spellings(
    alias: str, expected: LawKind
) -> None:
    assert LawKind.from_value(alias) is expected
    assert LawKind.from_value(alias.upper()) is expected


def test_law_kind_rejects_unknown_spelling() -> None:
    with pytest.raises(ValueError):
        LawKind.from_value("c4_context")


def test_canonical_components_are_the_law_kind_values() -> None:
    assert CANONICAL_LAW_COMPONENTS == tuple(kind.value for kind in LawKind)


def test_objective_component_weights_route_through_law_kind_aliases() -> None:
    for alias, kind in LAW_KIND_ALIASES.items():
        weights = canonical_law_component_weights({alias: 0.25})
        assert weights[kind.value] == 0.25


def test_objective_component_weights_reject_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown local-law component weight"):
        canonical_law_component_weights({"c4_context": 1.0})
