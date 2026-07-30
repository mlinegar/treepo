from __future__ import annotations

from pathlib import Path

import pytest

from treepo.methods._preference_traces import preference_training_rows
from treepo.methods.dspy import build_dspy_family
from treepo.methods.preference import Candidate, PreferenceDataset, PreferenceRecord


def _candidate() -> list[dict[str, object]]:
    return [{"id": "gold", "value": {"score": 0.5}, "preferred": True}]


def test_authoritative_joint_propensity_outranks_conflicting_generic_field() -> None:
    record = PreferenceRecord.from_mapping(
        {
            "unit_id": "joint",
            "unit_type": "root",
            "target": "f",
            "context": "score",
            "weight": 2.0,
            "propensity": "stale-display-value",
            "joint_propensity": 0.2,
            "candidates": _candidate(),
        }
    )
    dataset = PreferenceDataset.from_records((record,))
    (unit,) = dataset.units

    assert record.propensity == pytest.approx(0.2)
    assert unit["propensity"] == pytest.approx(0.2)
    assert unit["sample_weight"] == pytest.approx(10.0)
    assert unit["propensity_source"] == "metadata.joint_propensity"
    assert unit["metadata"]["propensity_source"] == "metadata.joint_propensity"


def test_flat_rows_resolve_nested_sampling_component_product() -> None:
    dataset = PreferenceDataset.from_flat_rows(
        (
            {
                "unit_id": "nested",
                "candidate_id": "gold",
                "unit_type": "root",
                "target": "f",
                "context": "score",
                "value": {"score": 0.5},
                "preferred": True,
                "propensity": 0.9,
                "metadata": {
                    "sampling": {
                        "document_propensity": 0.5,
                        "unit_propensity": 0.25,
                        "label_propensity": 0.4,
                    }
                },
            },
        )
    )
    (unit,) = dataset.units

    assert unit["propensity"] == pytest.approx(0.05)
    assert unit["sample_weight"] == pytest.approx(20.0)
    assert unit["propensity_source"] == (
        "metadata.sampling.component_product[document_propensity,unit_propensity,label_propensity]"
    )


def test_pairwise_mapping_uses_inclusion_probability_before_generic_propensity() -> None:
    record = PreferenceRecord.from_mapping(
        {
            "pair_id": "pair",
            "response_a": "A",
            "response_b": "B",
            "preferred": "A",
            "propensity": 0.9,
            "inclusion_probability": 0.4,
            "weight": 2.0,
        }
    )

    assert record.propensity == pytest.approx(0.4)
    assert record.sample_weight() == pytest.approx(5.0)
    assert record.propensity_source == "metadata.inclusion_probability"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "unit_id": "unsupported-record",
            "target": "f",
            "context": "score",
            "joint_propensity": 0.5,
            "supports_ipw_estimation": False,
            "candidates": _candidate(),
        },
        {
            "unit_id": "unsupported-flat",
            "candidate_id": "gold",
            "target": "f",
            "context": "score",
            "value": {"score": 0.5},
            "metadata": {
                "sampling": {
                    "joint_propensity": 0.5,
                    "supports_ipw_estimation": False,
                }
            },
        },
        {
            "pair_id": "unsupported-pair",
            "response_a": "A",
            "response_b": "B",
            "preferred": "A",
            "joint_propensity": 0.5,
            "supports_ipw_estimation": False,
        },
    ],
)
def test_unsupported_sampling_design_fails_closed(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="supports_ipw_estimation=False"):
        PreferenceDataset.from_value(payload)


def test_below_minimum_propensity_fails_instead_of_flooring() -> None:
    with pytest.raises(ValueError, match="below min_propensity.*flooring is not allowed"):
        PreferenceDataset.from_value(
            {
                "unit_id": "too-rare",
                "target": "f",
                "context": "score",
                "joint_propensity": 1.0e-9,
                "candidates": _candidate(),
            }
        )

    with pytest.raises(ValueError, match="below min_propensity"):
        PreferenceRecord(
            unit_id="too-rare-direct",
            unit_type="root",
            target="f",
            context="score",
            propensity=1.0e-9,
            candidates=(Candidate(id="gold", value={"score": 0.5}),),
        )


def test_direct_canonical_record_preserves_propensity_and_effective_weight() -> None:
    dataset = PreferenceDataset.from_records(
        (
            PreferenceRecord(
                unit_id="direct",
                unit_type="root",
                target="f",
                context="score",
                weight=2.0,
                propensity=0.25,
                candidates=(Candidate(id="gold", value={"score": 0.5}, preferred=True),),
            ),
        )
    )
    (unit,) = dataset.units

    assert unit["weight"] == pytest.approx(2.0)
    assert unit["propensity"] == pytest.approx(0.25)
    assert unit["sample_weight"] == pytest.approx(8.0)
    assert unit["propensity_source"] == "PreferenceRecord.propensity"


def test_precomputed_weight_round_trips_to_dspy_without_double_division(
    tmp_path: Path,
) -> None:
    dataset = PreferenceDataset.from_value(
        {
            "unit_id": "precomputed",
            "unit_type": "root",
            "target": "f",
            "context": "score this root",
            "weight": 99.0,
            "sample_weight": 8.0,
            "propensity": 0.9,
            "joint_propensity": 0.25,
            "candidates": _candidate(),
        }
    )
    # Exercise the serialized tables boundary before the optimizer adapter.
    round_tripped = PreferenceDataset.from_value(dataset.to_dict())
    (unit,) = round_tripped.units
    (trace,) = preference_training_rows(round_tripped, target="f")

    assert unit["weight"] == pytest.approx(99.0)
    assert unit["sample_weight"] == pytest.approx(8.0)
    assert unit["sample_weight_source"] == "precomputed_sample_weight"
    assert unit["propensity_source"] == "metadata.joint_propensity"
    assert trace.metadata["preference_input_weight"] == pytest.approx(99.0)
    assert trace.weight == pytest.approx(2.0)
    assert trace.propensity == pytest.approx(0.25)
    assert trace.sample_weight == pytest.approx(8.0)
    assert trace.metadata["propensity_source"] == "metadata.joint_propensity"

    family = build_dspy_family(
        {
            "optimizer": "none",
            "target_names": ("score",),
            "target_oracle_ids": ("fixture:score",),
            "target_dim": 1,
            "audit_laws": False,
        }
    )
    (example,) = family._dspy_f_examples((trace,), g=None)

    assert example.raw_weight == pytest.approx(2.0)
    assert example.propensity == pytest.approx(0.25)
    assert example.propensity_source == "metadata.joint_propensity"
    assert example.effective_weight == pytest.approx(8.0)
