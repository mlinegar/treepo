from __future__ import annotations

import json

import pytest

from treepo import oracle_vector_l1
from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_targets import _node_target_value, _target_vector
from treepo.methods._runtime_evaluation import _split_metrics
from treepo.tasks.manifesto import (
    CMP_POLICY_CODES,
    MANIFESTO_RILE_NORMALIZED_SCHEMA_VERSION,
    RILE_CMP56_TARGET_NAMES,
    RILE_NORMALIZED_ORACLE_ID,
    RILE_NORMALIZED_TARGET_KEY,
    RILE_NORMALIZED_TARGET_NAME,
    RILE_POLARITY_TARGET_NAMES,
    attach_manifesto_rile_components,
    manifesto_rile_component_fit_fragment,
    manifesto_rile_component_output_schema,
    manifesto_rile_component_readout,
    manifesto_rile_component_report,
    manifesto_rile_component_targets,
    manifesto_rile_components_from_codes,
    manifesto_rile_components_from_counts,
    manifesto_rile_normalized_fit_fragment,
    normalize_cmp_code,
)
from treepo.tree import TreeNode, TreeRecord


def test_l1_is_one_metric_path_for_scalar_and_vector_oracles() -> None:
    assert oracle_vector_l1(0.4, 0.1) == pytest.approx(0.3)
    assert oracle_vector_l1([0.4], [0.1]) == pytest.approx(0.3)
    assert oracle_vector_l1(
        {"left": 0.2, "right": 0.8},
        {"left": 0.3, "right": 0.6},
        target_names=("left", "right"),
    ) == pytest.approx(0.3)

    scalar_tree = TreeRecord(tree_id="scalar", root_label=0.1)
    scalar_metrics = _split_metrics([0.4], [scalar_tree])
    assert scalar_metrics.joint_f_l1 == pytest.approx(0.3)
    assert scalar_metrics.joint_f_l1_n == 1

    vector_tree = TreeRecord(
        tree_id="vector",
        metadata={"targets": {"left": 0.3, "right": 0.6}},
    )
    vector_metrics = _split_metrics(
        [[0.2, 0.8]],
        [vector_tree],
        target_names=("left", "right"),
        target_vector_key="targets",
    )
    assert vector_metrics.joint_f_l1 == pytest.approx(0.3)
    assert vector_metrics.joint_f_l1_n == 1


def test_cmp_code_normalization_and_compact_polarity_readout() -> None:
    assert normalize_cmp_code("605.1") == "605"
    assert normalize_cmp_code(6051) == "605"
    assert normalize_cmp_code("header") == "H"
    assert normalize_cmp_code("0") == "000"

    components = manifesto_rile_components_from_codes(
        ["103", "104", "605.1", "000", "H"],
        granularity="polarity",
    )

    assert tuple(components) == RILE_POLARITY_TARGET_NAMES
    assert components == pytest.approx(
        {
            "rile_left_share": 0.25,
            "rile_other_share": 0.25,
            "rile_right_share": 0.5,
        }
    )
    assert manifesto_rile_component_readout(
        components,
        granularity="polarity",
    ) == pytest.approx(25.0)


def test_cmp56_is_dense_exact_and_keeps_residual_denominator_mass() -> None:
    components = manifesto_rile_components_from_counts(
        {
            "103": 2,
            "104": 1,
            "501": 3,
            "000": 4,
            "H": 2,
        },
        total_non_header_mass=10,
        granularity="cmp56",
    )

    assert len(CMP_POLICY_CODES) == 56
    assert tuple(components) == RILE_CMP56_TARGET_NAMES
    assert components["cmp_103_share"] == pytest.approx(0.2)
    assert components["cmp_104_share"] == pytest.approx(0.1)
    assert components["cmp_501_share"] == pytest.approx(0.3)
    assert components["cmp_residual_other_share"] == pytest.approx(0.4)
    assert sum(components.values()) == pytest.approx(1.0)
    assert manifesto_rile_component_readout(components) == pytest.approx(-10.0)

    with pytest.raises(ValueError, match="exceeds total"):
        manifesto_rile_components_from_counts(
            {"103": 2},
            total_non_header_mass=1,
        )
    with pytest.raises(ValueError, match="cannot be normalized"):
        manifesto_rile_components_from_counts({"not-a-code": 1})


def test_component_l1_detects_cancellation_hidden_by_scalar_rile() -> None:
    target = {
        "rile_left_share": 0.4,
        "rile_other_share": 0.2,
        "rile_right_share": 0.4,
    }
    prediction = {
        "rile_left_share": 0.2,
        "rile_other_share": 0.6,
        "rile_right_share": 0.2,
    }

    report = manifesto_rile_component_report(
        prediction,
        target,
        granularity="polarity",
    )

    assert report["predicted_rile"] == pytest.approx(0.0)
    assert report["target_rile"] == pytest.approx(0.0)
    assert report["rile_absolute_error"] == pytest.approx(0.0)
    assert report["component_l1"] == pytest.approx(0.8)
    assert report["rile_l1_bound_holds"] is True

    shifted = {
        "rile_left_share": 0.2,
        "rile_other_share": 0.2,
        "rile_right_share": 0.6,
    }
    shifted_report = manifesto_rile_component_report(
        shifted,
        target,
        granularity="polarity",
    )
    assert shifted_report["rile_normalized_absolute_error"] == pytest.approx(0.2)
    assert shifted_report["rile_normalized_l1_upper_bound"] == pytest.approx(0.2)
    assert shifted_report["rile_l1_bound_holds"] is True


def test_component_catalog_is_one_joint_oracle_with_frozen_readout_metadata() -> None:
    compact = manifesto_rile_component_targets("polarity")
    expanded = manifesto_rile_component_targets("cmp56")

    assert tuple(target.target_name for target in compact) == RILE_POLARITY_TARGET_NAMES
    assert tuple(target.target_name for target in expanded) == RILE_CMP56_TARGET_NAMES
    assert len(compact) == 3
    assert len(expanded) == 57
    assert len({target.oracle_id for target in expanded}) == 1
    assert (
        next(target for target in expanded if target.target_name == "cmp_103_share").metadata[
            "rile_readout_coefficient"
        ]
        == -100.0
    )
    assert (
        next(target for target in expanded if target.target_name == "cmp_104_share").metadata[
            "rile_readout_coefficient"
        ]
        == 100.0
    )
    assert (
        next(target for target in expanded if target.target_name == "cmp_501_share").metadata[
            "rile_readout_coefficient"
        ]
        == 0.0
    )


def test_k1_k3_k57_fragments_own_task_specific_f_and_g_instructions() -> None:
    fragments = {
        1: manifesto_rile_normalized_fit_fragment(),
        3: manifesto_rile_component_fit_fragment("polarity"),
        57: manifesto_rile_component_fit_fragment("cmp56"),
    }
    required_terms = {
        1: ("rile_normalized", "(rile + 100) / 200", "nonheader", "headers", "other"),
        3: ("rile", "left", "right", "other", "non-header"),
        57: ("rile", "cmp", "56", "residual", "non-header"),
    }

    for width, fragment in fragments.items():
        backend = fragment["backend_config"]
        f_instructions = str(backend["f_signature_instructions"]).strip()
        g_instructions = str(backend["g_signature_instructions"]).strip()
        combined = f"{f_instructions}\n{g_instructions}".lower()

        assert len(fragment["oracle_targets"]) == width
        assert f_instructions and g_instructions
        assert "strict json" in f_instructions.lower()
        assert all(role in g_instructions.lower() for role in ("leaf", "unary", "binary"))
        assert all(term in combined for term in required_terms[width])

    singleton = fragments[1]
    (target,) = singleton["oracle_targets"]
    assert target.target_name == RILE_NORMALIZED_TARGET_NAME
    assert target.oracle_id == RILE_NORMALIZED_ORACLE_ID
    assert target.metadata["schema_version"] == MANIFESTO_RILE_NORMALIZED_SCHEMA_VERSION
    assert singleton["backend_config"]["target_vector_key"] == RILE_NORMALIZED_TARGET_KEY
    assert "not an ideological-neutral label" in (
        singleton["backend_config"]["g_signature_instructions"]
    )


def test_full_document_component_schema_uses_the_same_frozen_target_order() -> None:
    schema = manifesto_rile_component_output_schema("cmp56")

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(RILE_CMP56_TARGET_NAMES)
    assert tuple(schema["properties"]) == RILE_CMP56_TARGET_NAMES
    assert all(
        field["minimum"] == 0.0 and field["maximum"] == 1.0
        for field in schema["properties"].values()
    )
    assert json.loads(json.dumps(schema, allow_nan=False)) == schema


def _component_tree() -> TreeRecord:
    return TreeRecord(
        tree_id="manifesto-1",
        root_label=7.5,
        nodes=(
            TreeNode(
                node_id="leaf-a",
                unit_type="leaf",
                level=0,
                parent_id="root",
                text="left text",
                label=-100.0,
                metadata={
                    "cmp_counts": {"103": 1},
                    "total_non_header_qsentences": 1,
                },
            ),
            TreeNode(
                node_id="leaf-b",
                unit_type="leaf",
                level=0,
                parent_id="root",
                text="right and other text",
                label=50.0,
                metadata={
                    "cmp_counts": {"104": 1, "000": 1},
                    "total_non_header_qsentences": 2,
                },
            ),
            TreeNode(
                node_id="root",
                unit_type="root",
                level=1,
                left_child_id="leaf-a",
                right_child_id="leaf-b",
                text="whole document",
                label=0.0,
                metadata={
                    "cmp_counts": {"103": 1, "104": 1, "000": 1},
                    "total_non_header_qsentences": 3,
                },
            ),
        ),
        metadata={"split": "train"},
    )


def test_tree_adapter_hydrates_the_existing_joint_vector_training_path() -> None:
    hydrated = attach_manifesto_rile_components(
        [_component_tree()],
        granularity="cmp56",
        law_state_key="rile_component_counts",
    )[0]
    fragment = manifesto_rile_component_fit_fragment("cmp56")
    targets = fragment["oracle_targets"]
    config = NeuralOperatorFamilyConfig(
        operator_kind="conv1d",
        target_names=tuple(target.target_name for target in targets),
        target_oracle_ids=tuple(target.oracle_id for target in targets),
        **fragment["backend_config"],
    )

    expected = hydrated.metadata["rile_cmp56_shares"]
    assert hydrated.root_label == pytest.approx(7.5)
    assert hydrated.metadata["rile_from_components"] == pytest.approx(0.0)
    assert _target_vector(hydrated, config) == pytest.approx(
        [expected[name] for name in RILE_CMP56_TARGET_NAMES]
    )
    first_leaf = hydrated.get_node("leaf-a")
    assert first_leaf is not None
    assert _node_target_value(first_leaf, config, width=57) == pytest.approx(
        [first_leaf.metadata["rile_cmp56_shares"][name] for name in RILE_CMP56_TARGET_NAMES]
    )
    assert len(first_leaf.metadata["rile_component_counts"]) == 57
    assert sum(first_leaf.metadata["rile_component_counts"]) == pytest.approx(1.0)
    assert config.node_target_exclusive is True
    assert config.root_readout == "root_state"

    component_only = attach_manifesto_rile_components(
        [_component_tree()],
        granularity="polarity",
        drop_scalar_root_label=True,
    )[0]
    assert component_only.root_label is None
    assert component_only.metadata["rile_from_components"] == pytest.approx(0.0)

    analytic = manifesto_rile_component_fit_fragment(
        "polarity",
        analytic_leaf_rollup=True,
    )
    assert analytic["backend_config"]["root_readout"] == "leaf_mean"
    assert analytic["backend_config"]["rollup_weight_key"] == "total_non_header_qsentences"
