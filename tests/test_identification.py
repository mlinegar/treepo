from __future__ import annotations

import math

import pytest

from treepo.identification import (
    additive_identification_metadata,
    additive_root_information_weight,
    additive_root_sensitivity,
    annotate_additive_identification_rows,
    identification_node_weight,
    normalize_identification_weight_profile,
    pairwise_trace_node_masses,
)
from treepo.local_law import LocalLawAuditRow


def test_additive_root_weights_cover_basic_cases() -> None:
    assert additive_root_sensitivity(3.0, 12.0) == pytest.approx(0.25)
    assert additive_root_information_weight(3.0, 12.0) == pytest.approx(0.0625)
    assert additive_root_sensitivity(12.0, 12.0) == pytest.approx(1.0)
    assert additive_root_information_weight(12.0, 12.0) == pytest.approx(1.0)
    assert additive_root_sensitivity(0.0, 12.0) == pytest.approx(0.0)
    assert additive_root_information_weight(0.0, 12.0) == pytest.approx(0.0)


def test_additive_root_information_weight_is_bounded_when_node_is_subtree() -> None:
    for node_mass in (0.0, 1.0, 4.0, 10.0):
        weight = additive_root_information_weight(node_mass, 10.0)
        assert weight >= 0.0
        assert weight <= 1.0


def test_additive_root_weights_reject_invalid_masses() -> None:
    with pytest.raises(ValueError, match="node_mass"):
        additive_root_sensitivity(-1.0, 10.0)
    with pytest.raises(ValueError, match="document_mass"):
        additive_root_sensitivity(1.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        additive_root_information_weight(math.nan, 10.0)


def test_pairwise_trace_node_masses_follow_even_and_odd_carry_schedule() -> None:
    assert pairwise_trace_node_masses([1.0, 1.0, 1.0, 1.0]) == [
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        4.0,
    ]
    assert pairwise_trace_node_masses([1.0, 1.0, 1.0, 1.0, 1.0]) == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        4.0,
        5.0,
    ]
    assert pairwise_trace_node_masses([2.0, 3.0, 5.0]) == [2.0, 3.0, 5.0, 5.0, 10.0]


def test_identification_metadata_and_node_weight_profiles() -> None:
    metadata = additive_identification_metadata(
        node_mass=2.0,
        document_mass=8.0,
        extra={"node": "x"},
    )
    assert metadata["node"] == "x"
    assert metadata["node_mass"] == pytest.approx(2.0)
    assert metadata["document_mass"] == pytest.approx(8.0)
    assert metadata["additive_root_sensitivity"] == pytest.approx(0.25)
    assert metadata["additive_root_information_weight"] == pytest.approx(0.0625)

    assert identification_node_weight(node_mass=2.0, document_mass=8.0, profile="none") == pytest.approx(1.0)
    assert identification_node_weight(node_mass=2.0, document_mass=8.0, profile="sensitivity") == pytest.approx(0.25)
    assert identification_node_weight(node_mass=2.0, document_mass=8.0, profile="information") == pytest.approx(0.0625)
    assert normalize_identification_weight_profile("SENSITIVITY") == "sensitivity"
    with pytest.raises(ValueError, match="identification_weight_profile"):
        normalize_identification_weight_profile("ipw")


def test_annotate_additive_identification_rows_is_model_agnostic() -> None:
    rows = (
        LocalLawAuditRow(
            row_id="n0",
            law_kind="c1",
            proxy_loss=0.1,
            observed=False,
            propensity=0.25,
            node_weight=7.0,
            metadata={"node_index": 0},
        ),
        LocalLawAuditRow(
            row_id="n1",
            law_kind="c3",
            proxy_loss=0.2,
            oracle_loss=0.2,
            observed=True,
            propensity=0.5,
            node_weight=8.0,
            metadata={"node_index": 1},
        ),
    )
    annotated = annotate_additive_identification_rows(
        rows,
        node_masses=[2.0, 8.0],
        document_mass=8.0,
        profile="none",
        mass_source="test",
    )
    assert [row.propensity for row in annotated] == [pytest.approx(0.25), pytest.approx(0.5)]
    assert [row.node_weight for row in annotated] == [pytest.approx(7.0), pytest.approx(8.0)]
    assert annotated[0].metadata["node_index"] == 0
    assert annotated[0].metadata["node_mass"] == pytest.approx(2.0)
    assert annotated[0].metadata["document_mass"] == pytest.approx(8.0)
    assert annotated[0].metadata["additive_root_information_weight"] == pytest.approx(0.0625)
    assert annotated[0].metadata["identification_weight_profile"] == "none"
    assert annotated[0].metadata["node_mass_source"] == "test"


def test_annotate_additive_identification_rows_can_set_information_weights() -> None:
    rows = (
        LocalLawAuditRow(row_id="n0", law_kind="c1", proxy_loss=0.1, propensity=0.25),
        LocalLawAuditRow(row_id="n1", law_kind="c3", proxy_loss=0.2, propensity=0.5),
        LocalLawAuditRow(row_id="root", law_kind="c3", proxy_loss=0.0, propensity=1.0),
    )
    annotated = annotate_additive_identification_rows(
        rows,
        node_masses=[2.0, 6.0, 8.0],
        profile="information",
    )
    assert [row.propensity for row in annotated] == [
        pytest.approx(0.25),
        pytest.approx(0.5),
        pytest.approx(1.0),
    ]
    assert [row.node_weight for row in annotated] == [
        pytest.approx(1.0 / 16.0),
        pytest.approx(9.0 / 16.0),
        pytest.approx(1.0),
    ]


def test_annotate_additive_identification_rows_validates_mass_alignment() -> None:
    rows = (LocalLawAuditRow(row_id="n0", law_kind="c1", proxy_loss=0.1),)
    with pytest.raises(ValueError, match="rows.*node masses"):
        annotate_additive_identification_rows(rows, node_masses=[1.0, 2.0])
