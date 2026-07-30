from __future__ import annotations

import math

import pytest

from treepo.sampling import resolve_effective_propensity


def test_effective_propensity_has_highest_precedence() -> None:
    resolved = resolve_effective_propensity(
        {
            "effective_propensity": 0.8,
            "joint_propensity": 0.7,
            "inclusion_probability": 0.6,
            "sampling": {
                "joint_propensity": 0.5,
                "document_propensity": 0.4,
                "unit_propensity": 0.5,
                "label_propensity": 0.5,
            },
            "document_propensity": 0.3,
            "propensity": 0.2,
        }
    )
    assert resolved.propensity == pytest.approx(0.8)
    assert resolved.source == "metadata.effective_propensity"


def test_joint_then_inclusion_probability_outrank_nested_and_generic_fields() -> None:
    joint = resolve_effective_propensity(
        {
            "joint_propensity": 0.4,
            "inclusion_probability": 0.3,
            "sampling": {"joint_propensity": 0.2},
            "propensity": 0.1,
        }
    )
    inclusion = resolve_effective_propensity(
        {
            "inclusion_probability": 0.3,
            "sampling": {"joint_propensity": 0.2},
            "propensity": 0.1,
        }
    )
    assert (joint.propensity, joint.source) == (
        pytest.approx(0.4),
        "metadata.joint_propensity",
    )
    assert (inclusion.propensity, inclusion.source) == (
        pytest.approx(0.3),
        "metadata.inclusion_probability",
    )


def test_nested_joint_and_component_product_precede_top_level_components() -> None:
    nested_joint = resolve_effective_propensity(
        {
            "sampling": {
                "joint_propensity": 0.25,
                "document_propensity": 0.5,
                "unit_propensity": 0.5,
            },
            "document_propensity": 0.8,
            "unit_propensity": 0.5,
            "propensity": 0.9,
        }
    )
    nested_product = resolve_effective_propensity(
        {
            "sampling": {
                "document_propensity": 0.5,
                "unit_propensity": 0.25,
                "label_propensity": 0.4,
            },
            "document_propensity": 0.8,
            "unit_propensity": 0.5,
            "propensity": 0.9,
        }
    )
    assert (nested_joint.propensity, nested_joint.source) == (
        pytest.approx(0.25),
        "metadata.sampling.joint_propensity",
    )
    assert (nested_product.propensity, nested_product.source) == (
        pytest.approx(0.05),
        "metadata.sampling.component_product"
        "[document_propensity,unit_propensity,label_propensity]",
    )


def test_top_level_component_product_precedes_generic_propensity() -> None:
    resolved = resolve_effective_propensity(
        {
            "document_propensity": 0.5,
            "unit_propensity": 0.25,
            "propensity": 0.9,
        }
    )
    assert resolved.propensity == pytest.approx(0.125)
    assert resolved.source == (
        "metadata.component_product[document_propensity,unit_propensity]"
    )


def test_generic_propensity_is_last_and_default_is_one() -> None:
    generic = resolve_effective_propensity({"propensity": 0.2})
    nested = resolve_effective_propensity({"sampling": {"propensity": 0.3}})
    default = resolve_effective_propensity({})
    assert (generic.propensity, generic.source) == (
        pytest.approx(0.2),
        "metadata.propensity",
    )
    assert (nested.propensity, nested.source) == (
        pytest.approx(0.3),
        "metadata.sampling.propensity",
    )
    assert (default.propensity, default.source) == (pytest.approx(1.0), "default")


@pytest.mark.parametrize(
    "metadata",
    [
        {"supports_ipw_estimation": False, "joint_propensity": 0.5},
        {
            "joint_propensity": 0.5,
            "sampling": {"supports_ipw_estimation": False},
        },
    ],
)
def test_unsupported_ipw_design_fails_closed(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="supports_ipw_estimation=False"):
        resolve_effective_propensity(metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        {"effective_propensity": True},
        {"effective_propensity": 0.0},
        {"joint_propensity": -0.1},
        {"inclusion_probability": 1.1},
        {"sampling": {"joint_propensity": math.nan}},
        {"sampling": {"document_propensity": 0.5, "unit_propensity": math.inf}},
        {"document_propensity": "not-a-number"},
        {"propensity": None, "sampling": []},
    ],
)
def test_selected_or_structural_invalid_values_fail_closed(
    metadata: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError), match="sampling|propensity|finite"):
        resolve_effective_propensity(metadata)


def test_lower_precedence_invalid_field_does_not_override_valid_authority() -> None:
    resolved = resolve_effective_propensity(
        {
            "joint_propensity": 0.25,
            "propensity": "stale-display-value",
        }
    )
    assert (resolved.propensity, resolved.source) == (
        pytest.approx(0.25),
        "metadata.joint_propensity",
    )
