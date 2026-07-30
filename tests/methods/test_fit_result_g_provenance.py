from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from treepo.methods._fit_result import _g_training_support

_TOPOLOGY = {
    "train_tree_count": 2,
    "training_composition_present": True,
}
_DSPY_SOURCE = "dspy_compiler_mixed_examples"


def _record(artifact: Any) -> SimpleNamespace:
    return SimpleNamespace(g_artifact=artifact)


def _dspy_artifact(
    *,
    roles: Any = ("leaf", "merge"),
    leaf_rows: Any = 3,
    recompression_rows: Any | None = None,
    merge_rows: Any = 2,
    source: Any = _DSPY_SOURCE,
) -> dict[str, Any]:
    artifact = {
        "g_training_call_roles": roles,
        "leaf_domain_training_row_count": leaf_rows,
        "merge_domain_training_row_count": merge_rows,
        "g_training_role_evidence_source": source,
    }
    if recompression_rows is not None:
        artifact["recompression_training_row_count"] = recompression_rows
    return artifact


def test_explicit_dspy_row_provenance_precedes_gradient_keys() -> None:
    artifact = _dspy_artifact(roles=("leaf",), leaf_rows=4, merge_rows=0)
    artifact.update(
        {
            "leaf_domain_gradient_path_present": True,
            "merge_domain_gradient_path_present": True,
        }
    )

    roles, merge_observed, source = _g_training_support(
        [_record(artifact)],
        topology=_TOPOLOGY,
    )

    assert roles == ["leaf"]
    assert merge_observed is False
    assert source == _DSPY_SOURCE


def test_explicit_dspy_row_roles_are_aggregated_across_train_g_calls() -> None:
    roles, merge_observed, source = _g_training_support(
        [
            _record(_dspy_artifact(roles=("leaf",), leaf_rows=3, merge_rows=0)),
            _record(_dspy_artifact(roles=("merge",), leaf_rows=0, merge_rows=2)),
        ],
        topology=_TOPOLOGY,
    )

    assert roles == ["leaf", "merge"]
    assert merge_observed is True
    assert source == _DSPY_SOURCE


def test_dspy_recompression_is_training_support_but_not_merge_domain_support() -> None:
    roles, merge_observed, source = _g_training_support(
        [
            _record(
                _dspy_artifact(
                    roles=("leaf", "recompression"),
                    leaf_rows=3,
                    recompression_rows=1,
                    merge_rows=0,
                )
            )
        ],
        topology=_TOPOLOGY,
    )

    assert roles == ["leaf", "recompression"]
    assert merge_observed is False
    assert source == _DSPY_SOURCE


def test_legacy_dspy_provenance_may_omit_recompression_row_count() -> None:
    roles, merge_observed, source = _g_training_support(
        [_record(_dspy_artifact())],
        topology=_TOPOLOGY,
    )

    assert roles == ["leaf", "merge"]
    assert merge_observed is True
    assert source == _DSPY_SOURCE


def test_explicit_dspy_schema_is_required_on_every_train_g_record() -> None:
    with pytest.raises(ValueError, match="must contain every field"):
        _g_training_support(
            [_record(_dspy_artifact()), _record({"kind": "legacy_g"})],
            topology=_TOPOLOGY,
        )


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"roles": "leaf"}, TypeError, "ordered list or tuple"),
        ({"roles": ("leaf", 3)}, TypeError, "entries must be strings"),
        ({"roles": ("root",)}, ValueError, "'recompression'"),
        ({"roles": ("merge", "leaf")}, ValueError, "unique and ordered"),
        ({"roles": ("leaf", "leaf")}, ValueError, "unique and ordered"),
        ({"leaf_rows": True}, TypeError, "must be an integer"),
        ({"recompression_rows": True}, TypeError, "must be an integer"),
        ({"merge_rows": 1.5}, TypeError, "must be an integer"),
        ({"leaf_rows": -1}, ValueError, "must be non-negative"),
        ({"recompression_rows": -1}, ValueError, "must be non-negative"),
        ({"roles": ("leaf",), "merge_rows": 2}, ValueError, "agree exactly"),
        (
            {"roles": ("leaf", "recompression"), "recompression_rows": None},
            ValueError,
            "agree exactly",
        ),
        ({"source": None}, TypeError, "must be a string"),
        ({"source": "   "}, ValueError, "must be non-empty"),
        ({"source": "dspy_gradient_path"}, ValueError, "must not be described"),
        ({"source": "dspy_c3_certificate"}, ValueError, "must not be described"),
    ],
)
def test_explicit_dspy_provenance_validation(
    overrides: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _g_training_support(
            [_record(_dspy_artifact(**overrides))],
            topology=_TOPOLOGY,
        )


def test_explicit_dspy_evidence_source_must_be_consistent() -> None:
    with pytest.raises(ValueError, match="must be consistent"):
        _g_training_support(
            [
                _record(_dspy_artifact()),
                _record(_dspy_artifact(source="different_dspy_training_rows")),
            ],
            topology=_TOPOLOGY,
        )


def test_fno_gradient_path_provenance_remains_separate() -> None:
    roles, merge_observed, source = _g_training_support(
        [
            _record(
                {
                    "leaf_domain_gradient_path_present": True,
                    "merge_domain_gradient_path_present": False,
                }
            ),
            _record(
                {
                    "leaf_domain_gradient_path_present": True,
                    "merge_domain_gradient_path_present": True,
                }
            ),
        ],
        topology=_TOPOLOGY,
    )

    assert roles == ["leaf", "merge"]
    assert merge_observed is True
    assert source == "g_artifact_gradient_path_presence"


def test_partial_fno_gradient_path_schema_fails_closed() -> None:
    with pytest.raises(ValueError, match="must contain every field"):
        _g_training_support(
            [
                _record(
                    {
                        "leaf_domain_gradient_path_present": True,
                        "merge_domain_gradient_path_present": False,
                    }
                ),
                _record({"leaf_domain_gradient_path_present": True}),
            ],
            topology=_TOPOLOGY,
        )


def test_legacy_topology_fallback_is_used_only_without_direct_fields() -> None:
    roles, merge_observed, source = _g_training_support(
        [_record({"kind": "legacy_g"})],
        topology=_TOPOLOGY,
    )

    assert roles == ["leaf", "merge"]
    assert merge_observed is True
    assert source == "inferred_from_train_g_and_topology"


def test_no_train_g_calls_have_no_training_support() -> None:
    assert _g_training_support([], topology=_TOPOLOGY) == (
        [],
        False,
        "no_train_g_calls",
    )
