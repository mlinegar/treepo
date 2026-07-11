from __future__ import annotations

import json
from pathlib import Path

import pytest

from treepo.certificate import (
    CommonMechanismEnvelopeEvidence,
    ConditionalAverageEnvelopeEvidence,
    TwoChannelResidual,
    UnifiedLearningComponentEvidence,
    build_error_certificate,
    build_two_channel_error_certificate,
)
from treepo.evidence import build_evidence
from treepo.local_law import (
    LawKind,
    LocalLawAuditRow,
    audit_local_laws,
    build_triangle_local_law_error_certificate,
    compute_influence_weighted_overlap,
    triangle_local_law_residual_from_audit,
)
from treepo.objective import (
    ObjectiveSpec,
    normalize_objective_spec,
    resolve_root_local_objective_weights,
)
from treepo.sampling import ObservationUnitKind, SamplingMetadata


def test_objective_rejects_additive_oracle_gap_terms() -> None:
    spec = ObjectiveSpec(
        objective_family="root_plus_laws",
        root_share=0.5,
        local_law_estimator="corrected",
        local_law_component_weights={"c1": 0.2, "c3": 0.3},
    )
    payload = spec.to_dict()
    assert set(payload["terms"]) == {"root", "local_law_corrected"}
    assert payload["local_law_component_weights"]["leaf_preservation"] == pytest.approx(0.2)
    assert payload["local_law_weight"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="oracle_gap"):
        normalize_objective_spec({"terms": {"oracle_gap": {"weight": 1.0}}})
    with pytest.raises(ValueError, match="unsupported objective fields"):
        normalize_objective_spec({"gap_weight": 1.0})


def test_objective_enforces_local_law_and_convex_weight_invariants() -> None:
    with pytest.raises(ValueError, match="positive law component"):
        ObjectiveSpec(
            root_share=1.0,
            local_law_estimator="corrected",
            local_law_component_weights={},
        )

    with pytest.raises(ValueError, match="convex combination"):
        ObjectiveSpec(
            root_share=0.8,
            local_law_estimator="corrected",
            local_law_weight=2.0,
            local_law_component_weights={"c1": 2.0},
        )

    spec = ObjectiveSpec(
        root_share=0.8,
        local_law_estimator="corrected",
        local_law_weight=2.0,
        local_law_component_weights={"c1": 2.0},
        allow_nonconvex_objective=True,
    )
    assert spec.to_dict()["allow_nonconvex_objective"] is True


def test_objective_accepts_oracle_state_and_external_passthrough_estimators() -> None:
    for estimator in ("oracle_state", "external_passthrough"):
        spec = ObjectiveSpec(
            objective_family="state_adapter",
            root_share=0.5,
            local_law_estimator=estimator,
            local_law_weight=0.5,
            local_law_component_weights={"c1": 0.5},
        )

        assert spec.to_dict()["local_law_estimator"] == estimator


def test_root_local_resolver_lambda_and_explicit_modes() -> None:
    resolved = resolve_root_local_objective_weights(
        local_law_weight=0.6,
        active_laws=("c1", "c2", "c3"),
    )
    assert resolved.root_share == pytest.approx(0.4)
    assert resolved.local_law_weight == pytest.approx(0.6)
    assert resolved.local_law_shares["leaf_preservation"] == pytest.approx(0.2)

    explicit = resolve_root_local_objective_weights(
        local_law_weight=None,
        active_laws=("c1", "c2"),
        explicit_root_weight=2.0,
        explicit_law_weights={"c1": 1.0, "c2": 1.0},
    )
    assert explicit.root_share == pytest.approx(0.5)
    assert explicit.local_law_weight == pytest.approx(0.5)

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_root_local_objective_weights(
            local_law_weight=0.5,
            active_laws=("c1",),
            explicit_root_weight=1.0,
        )


def test_audit_rows_compute_corrected_losses_and_overlap() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r0",
            law_kind=LawKind.C1_LEAF,
            proxy_loss=0.4,
            observed=False,
            propensity=0.25,
            node_weight=1.0,
        ),
        LocalLawAuditRow(
            row_id="r1",
            law_kind="c3",
            proxy_loss=0.4,
            oracle_loss=0.1,
            observed=True,
            propensity=0.5,
            node_weight=2.0,
        ),
    ]
    overlap = compute_influence_weighted_overlap(rows)
    assert overlap.D_lambda == pytest.approx(1.0 / 0.25 + 4.0 / 0.5)
    assert overlap.W_lambda == pytest.approx(4.0)
    assert overlap.effective_sample_size > 0.0


def test_triangle_local_law_residual_uses_audit_objective_as_transport_radius() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r0",
            law_kind=LawKind.C1_LEAF,
            proxy_loss=0.2,
            observed=False,
            propensity=0.5,
        ),
        LocalLawAuditRow(
            row_id="r1",
            law_kind=LawKind.C3_MERGE,
            proxy_loss=0.1,
            oracle_loss=0.4,
            observed=True,
            propensity=0.5,
        ),
    ]
    audit = audit_local_laws(rows)
    residual = triangle_local_law_residual_from_audit(
        audit=audit,
        radius_multiplier=2.0,
        root_down_radius=0.3,
        overidentification_radius=0.05,
        source="unit_test_audit",
        artifact_ids=("audit_summary",),
    )

    assert audit["local_law_objective"]["objective"] == pytest.approx(0.45)
    assert residual.leaf_up_radius == pytest.approx(0.9)
    assert residual.root_down_radius == pytest.approx(0.3)
    assert residual.overidentification_radius == pytest.approx(0.05)
    assert residual.total_radius == pytest.approx(1.25)
    assert residual.source == "unit_test_audit"
    assert residual.artifact_ids == ("audit_summary",)
    assert residual.metadata["transport_source"] == "merge_triangle_local_laws"
    assert residual.metadata["local_law_radius_source"] == "local_law_objective"
    assert residual.metadata["local_law_objective_mode"] == "corrected_local_law"
    assert residual.metadata["local_law_row_count"] == 2
    assert residual.metadata["local_law_observed_count"] == 1
    assert set(residual.metadata["by_law_kind"]) == {
        "leaf_preservation",
        "merge_preservation",
    }


def test_local_law_audit_records_depth_discount_weighting_metadata() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="root",
            law_kind=LawKind.C3_MERGE,
            proxy_loss=0.1,
            observed=False,
            propensity=0.25,
            node_weight=2.0,
            depth=0,
        ),
        LocalLawAuditRow(
            row_id="leaf",
            law_kind=LawKind.C1_LEAF,
            proxy_loss=0.2,
            observed=False,
            propensity=0.25,
            node_weight=3.0,
            depth=2,
        ),
    ]

    flat = audit_local_laws(rows, gamma_depth=1.0)
    discounted = audit_local_laws(rows, gamma_depth=0.9)

    assert [row.propensity for row in rows] == [pytest.approx(0.25), pytest.approx(0.25)]
    assert flat["local_law_weighting"]["effective_weight_sum"] == pytest.approx(5.0)
    assert discounted["local_law_weighting"]["gamma_depth"] == pytest.approx(0.9)
    assert discounted["local_law_weighting"]["effective_weight_formula"] == (
        "node_weight * gamma_depth ** depth"
    )
    assert discounted["local_law_weighting"]["by_depth"]["0"]["effective_weight_sum"] == pytest.approx(2.0)
    assert discounted["local_law_weighting"]["by_depth"]["2"]["effective_weight_sum"] == pytest.approx(
        3.0 * 0.9**2,
    )
    assert discounted["local_law_weighting"]["propensity_role"] == "sampling_probability_not_weight"


def test_triangle_local_law_certificate_routes_audit_into_two_channel_ledger() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r0",
            law_kind="c1",
            proxy_loss=0.05,
            observed=False,
            propensity=0.5,
        ),
        LocalLawAuditRow(
            row_id="r1",
            law_kind="c3",
            proxy_loss=0.05,
            oracle_loss=0.15,
            observed=True,
            propensity=0.5,
        ),
    ]
    cert = build_triangle_local_law_error_certificate(
        reported_estimate=0.2,
        rows=rows,
        leaf_up_radius=0.11,
        root_down_radius=0.12,
        overidentification_radius=0.02,
        common_mechanism_envelopes=[
            CommonMechanismEnvelopeEvidence(
                observed_root_radius=0.2,
                amplification=1.0,
                slack=0.01,
            ),
        ],
        confidence_delta=0.05,
    )

    assert cert.local_law_radius == pytest.approx(0.11)
    assert cert.calibration_radius == pytest.approx(0.12)
    assert cert.estimation_radius == pytest.approx(0.23)
    assert cert.radius_sum == pytest.approx(0.46)
    assert cert.confidence_delta == pytest.approx(0.05)
    assert cert.metadata["certificate_kind"] == "two_channel"
    assert cert.metadata["transport_certificate_kind"] == "triangle_local_law"
    assert cert.metadata["two_channel_residual"]["metadata"]["local_law_radius_source"] == (
        "explicit_leaf_up_radius"
    )


def test_triangle_local_law_certificate_preserves_certificate_gamma_metadata() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r0",
            law_kind="c1",
            proxy_loss=0.1,
            observed=False,
            propensity=1.0,
            depth=2,
        ),
    ]
    cert = build_triangle_local_law_error_certificate(
        reported_estimate=0.0,
        rows=rows,
        gamma_depth=0.9,
    )

    weighting = cert.metadata["two_channel_residual"]["metadata"]["local_law_weighting"]
    assert weighting["gamma_depth"] == pytest.approx(0.9)
    assert weighting["by_depth"]["2"]["effective_weight_sum"] == pytest.approx(0.9**2)


def test_triangle_local_law_residual_rejects_negative_derived_radius() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r0",
            law_kind="c3",
            proxy_loss=0.4,
            oracle_loss=0.1,
            observed=True,
            propensity=0.5,
        ),
    ]

    with pytest.raises(ValueError, match="derived leaf_up_radius is negative"):
        triangle_local_law_residual_from_audit(rows=rows)

    residual = triangle_local_law_residual_from_audit(rows=rows, leaf_up_radius=0.0)
    assert residual.leaf_up_radius == pytest.approx(0.0)


def test_certificate_keeps_component_radii_separate() -> None:
    cert = build_error_certificate(
        reported_estimate=0.2,
        component_evidence=[
            UnifiedLearningComponentEvidence(component="local_law", radius=0.1),
            UnifiedLearningComponentEvidence(component="calibration", radius=0.2),
            UnifiedLearningComponentEvidence(component="estimation", radius=0.3),
            UnifiedLearningComponentEvidence(component="clipping", radius=0.4),
        ],
    )
    assert cert.radius_sum == pytest.approx(1.0)
    assert cert.total_bound == pytest.approx(1.2)
    assert cert.local_law_radius == pytest.approx(0.1)


def test_two_channel_certificate_maps_to_existing_radius_ledger() -> None:
    cert = build_two_channel_error_certificate(
        reported_estimate=0.2,
        residual=TwoChannelResidual(
            leaf_up_radius=0.10,
            root_down_radius=0.20,
            overidentification_radius=0.03,
            source="audit",
        ),
        conditional_envelopes=[
            ConditionalAverageEnvelopeEvidence(
                degradation_radius=0.04,
                posterior_predictive_fit=True,
                psis_loo_stable=True,
                rank_calibrated=True,
                source="mrp",
            ),
        ],
    )
    assert cert.local_law_radius == pytest.approx(0.10)
    assert cert.calibration_radius == pytest.approx(0.20)
    assert cert.estimation_radius == pytest.approx(0.07)
    assert cert.radius_sum == pytest.approx(0.37)
    assert cert.total_bound == pytest.approx(0.57)
    assert cert.metadata["certificate_kind"] == "two_channel"
    semantic_components = {
        item.metadata["semantic_component"] for item in cert.component_evidence
    }
    assert semantic_components == {
        "leaf_up",
        "root_down",
        "overidentification",
        "conditional_average_envelope",
    }
    envelope_evidence = [
        item for item in cert.component_evidence
        if item.metadata["semantic_component"] == "conditional_average_envelope"
    ][0]
    assert envelope_evidence.metadata["envelope_source"] == "external_workflow"
    assert envelope_evidence.metadata["model_fitted_by_treepo"] is False


def test_common_mechanism_envelope_uses_observed_root_bound() -> None:
    cert = build_two_channel_error_certificate(
        reported_estimate=0.0,
        residual=TwoChannelResidual(
            leaf_up_radius=0.01,
            root_down_radius=0.20,
            overidentification_radius=0.03,
        ),
        common_mechanism_envelopes=[
            CommonMechanismEnvelopeEvidence(
                observed_root_radius=0.20,
                amplification=1.5,
                slack=0.01,
                source="heldout_roots",
            ),
        ],
    )
    assert cert.local_law_radius == pytest.approx(0.01)
    assert cert.calibration_radius == pytest.approx(0.20)
    assert cert.estimation_radius == pytest.approx(0.34)
    assert cert.radius_sum == pytest.approx(0.55)
    assert cert.metadata["common_mechanism_envelopes"][0]["degradation_radius"] == pytest.approx(
        0.31,
    )
    envelope_evidence = [
        item for item in cert.component_evidence
        if item.metadata["semantic_component"] == "common_mechanism_envelope"
    ][0]
    assert envelope_evidence.metadata["envelope_source"] == "observed_root_errors"
    assert envelope_evidence.metadata["transport_source"] == "merge_triangle_local_laws"
    assert envelope_evidence.metadata["root_control_source"] == "audit_bound"
    assert envelope_evidence.metadata["observed_root_radius"] == pytest.approx(0.20)
    assert envelope_evidence.metadata["assumptions"] == {
        "common_f": True,
        "common_g": True,
        "local_law_transport": True,
        "assumptions_satisfied": True,
    }


def test_common_mechanism_envelope_requires_transport_assumptions() -> None:
    envelope = CommonMechanismEnvelopeEvidence(
        observed_root_radius=0.2,
        common_f=True,
        common_g=True,
        local_law_transport=False,
    )
    with pytest.raises(ValueError, match="common-mechanism"):
        build_two_channel_error_certificate(
            reported_estimate=0.0,
            common_mechanism_envelopes=[envelope],
        )

    cert = build_two_channel_error_certificate(
        reported_estimate=0.0,
        common_mechanism_envelopes=[envelope],
        require_common_mechanism_assumptions=False,
    )
    assert cert.estimation_radius == pytest.approx(0.2)
    assert cert.metadata["common_mechanism_envelopes"][0]["assumptions_satisfied"] is False


def test_conditional_average_envelope_requires_explicit_diagnostics() -> None:
    envelope = ConditionalAverageEnvelopeEvidence(
        degradation_radius=0.04,
        posterior_predictive_fit=True,
        psis_loo_stable=False,
        rank_calibrated=True,
    )
    with pytest.raises(ValueError, match="diagnostics"):
        build_two_channel_error_certificate(
            reported_estimate=0.0,
            conditional_envelopes=[envelope],
        )

    cert = build_two_channel_error_certificate(
        reported_estimate=0.0,
        conditional_envelopes=[envelope],
        require_conditional_diagnostics=False,
    )
    assert cert.estimation_radius == pytest.approx(0.04)
    assert cert.metadata["conditional_envelopes"][0]["diagnostics_satisfied"] is False


def test_sampling_metadata_is_deterministic() -> None:
    sampling = SamplingMetadata(
        document_propensity=0.5,
        unit_propensity=0.25,
        label_propensity=1.0,
        unit_kind=ObservationUnitKind.LEAF,
    )
    assert sampling.effective_joint_propensity() == pytest.approx(0.125)
    assert sampling.ipw_weight() == pytest.approx(8.0)


def test_top_level_fit_learning_routes_through_methods_surface(tmp_path: Path) -> None:
    from treepo import fit

    class NoopFamily:
        name = "noop"

        def train_f(self, *, f_init, g, traces, output_dir, iteration):
            return f_init

        def train_g(self, *, g_init, f, traces, output_dir, iteration):
            return g_init

        def score_roots_with_f(self, *, f, g, trees):
            return [None] * len(trees)

        def validate_artifact(self, *, kind, artifact):
            return None

    result = fit(
        {
            "spec": {
                "space_kind": "top_level_fit",
                "family": "noop",
                "schedule": "fg",
                "initial_artifacts": {"f": "f0", "g": "g0"},
                "train_data": [],
                "eval_data": [],
                "backend_config": {"family_runtime": NoopFamily()},
                "axis": {"max_iterations": 0, "axis_value": 0},
            },
        },
        output_dir=tmp_path / "fit_learning",
    )

    assert result.status == "success"
    assert result.mode == "learning"
    assert result.summary["family"] == "noop"
    assert result.artifacts["f"] == "f0"


def test_fit_routes_preference_data_to_f_and_g(tmp_path: Path) -> None:
    import treepo
    from treepo import Candidate, PreferenceDataset, PreferenceRecord

    class RecordingFamily:
        name = "recording"

        def train_f(self, *, f_init, g, traces, output_dir, iteration):
            del f_init, g, output_dir
            return {"trained": "f", "iteration": iteration, "traces": list(traces)}

        def train_g(self, *, g_init, f, traces, output_dir, iteration):
            del g_init, f, output_dir
            return {"trained": "g", "iteration": iteration, "traces": list(traces)}

        def score_roots_with_f(self, *, f, g, trees):
            return [None] * len(trees)

        def validate_artifact(self, *, kind, artifact):
            assert artifact is None or artifact["trained"] == kind

    preferences = PreferenceDataset(
        [
            PreferenceRecord(
                unit_id="root_1",
                unit_type="root",
                target="f",
                context="root one",
                candidates=(Candidate(id="gold", value=1.0, score=1.0),),
            ),
            PreferenceRecord(
                unit_id="root_2",
                unit_type="root",
                target="f",
                context="root two",
                candidates=(Candidate(id="gold", value=2.0, score=1.0),),
            ),
            PreferenceRecord(
                unit_id="leaf_1",
                unit_type="leaf",
                target="g",
                context="leaf one",
                candidates=(Candidate(id="gold", value=3.0, score=1.0),),
            ),
        ]
    )

    result = treepo.fit(
        {
            "family": "recording",
            "train_data": ["shared"],
            "preference_data": preferences,
            "eval_data": [],
            "backend_config": {
                "family_runtime": RecordingFamily(),
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert [row.metadata["preference_unit_id"] for row in result.artifacts["f"]["traces"]] == [
        "root_1",
        "root_2",
    ]
    assert [row.metadata["preference_unit_id"] for row in result.artifacts["g"]["traces"]] == [
        "leaf_1"
    ]


def test_preference_dataset_exports_optimizer_records(tmp_path: Path) -> None:
    from treepo.methods.preference import (
        Candidate,
        PreferenceDataset,
        PreferenceRecord,
        export_preference_records,
    )

    dataset = PreferenceDataset.from_value(
        [
            PreferenceRecord(
                record_id="p1",
                unit_id="doc1",
                unit_type="qsentence",
                target="g",
                context="Summarize for RILE.",
                candidates=(
                    Candidate(id="left", value="left response", preferred=True),
                    Candidate(id="right", value="right response"),
                ),
                weight=2.0,
                propensity=0.5,
            ),
            {
                "record_id": "p2",
                "unit_id": "root1",
                "unit_type": "root",
                "target": "f",
                "context": "Score the root.",
                "candidates": [
                    {"id": "one", "value": "candidate one", "rank": 1},
                    {"id": "two", "value": "candidate two", "rank": 1},
                ],
            },
        ]
    )

    assert dataset.summary()["n_units"] == 2
    assert dataset.summary()["n_candidates"] == 4
    assert dataset.summary()["targets"] == ["f", "g"]
    assert len(dataset.filter_target("g")) == 1
    assert len(dataset.to_records("dpo")) == 1
    assert len(dataset.to_records("reward")) == 1
    assert len(dataset.to_records("grpo")) == 2
    assert len(dataset.to_records("supervised")) == 3

    hf_dataset = dataset.to_hf_dataset_dict()
    assert set(hf_dataset.keys()) == {"units", "candidates"}
    assert len(hf_dataset["units"]) == 2
    assert len(hf_dataset["candidates"]) == 4
    assert PreferenceDataset.from_value(hf_dataset).summary()["n_units"] == 2

    exported = export_preference_records(dataset, tmp_path)
    assert exported["counts"]["dataset"] == 2
    assert exported["counts"]["units"] == 2
    assert exported["counts"]["candidates"] == 4
    assert exported["counts"]["dpo"] == 1
    assert Path(exported["files"]["hf_dataset"]).exists()
    dpo_rows = (tmp_path / "preference_dpo.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(dpo_rows) == 1
    assert json.loads(dpo_rows[0])["chosen"] == "left response"
    reward_rows = (tmp_path / "preference_reward.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(reward_rows) == 1
    reward_record = json.loads(reward_rows[0])
    assert reward_record["chosen"] == "left response"
    assert reward_record["rejected"] == "right response"
    assert reward_record["chosen_score"] > reward_record["rejected_score"]


def test_preference_dataset_round_trips_task_state_values() -> None:
    from treepo import Candidate, PreferenceDataset, PreferenceRecord, TaskState

    state = TaskState(
        kind="lda_topics",
        counts={"topic_0": 2.0, "topic_1": 1.0},
        measures={"topic_proportions": [2.0 / 3.0, 1.0 / 3.0]},
        metadata={"source": "unit_test"},
    )
    dataset = PreferenceDataset(
        [
            PreferenceRecord(
                record_id="state1",
                unit_id="doc1:leaf0",
                unit_type="leaf",
                target="g",
                context="Infer topic state.",
                candidates=(Candidate(id="gold", value=state, score=1.0),),
                tree_id="doc1",
                node_id="leaf0",
                level=0,
            )
        ]
    )

    supervised = dataset.to_records("supervised")
    assert supervised[0]["value"]["kind"] == "lda_topics"
    assert supervised[0]["value"]["counts"]["topic_0"] == 2.0
    assert dataset.to_records("general")[0]["tree_id"] == "doc1"
    hf_dataset = dataset.to_hf_dataset_dict()
    loaded = PreferenceDataset.from_value(hf_dataset)
    assert loaded.to_records("supervised")[0]["value"]["kind"] == "lda_topics"


def test_tree_record_round_trips_and_exports_preference_units(tmp_path: Path) -> None:
    from treepo import TaskState, TreeNode, TreeRecord
    from treepo.artifacts import write_artifact_bundle
    from treepo.methods.preference import preference_units_from_trees
    from treepo.tree import (
        iter_tree_units,
        local_law_rows_from_tree_records,
        tree_summary,
        validate_tree_record,
        write_tree_records_jsonl,
    )

    leaf_state = TaskState(kind="manifesto_policy", counts={"qsentences": 1.0}, measures={"rile": 0.8})
    tree = TreeRecord(
        tree_id="doc1",
        doc_id="doc1",
        text="leaf A leaf B",
        root_label=0.5,
        nodes=(
            TreeNode(
                node_id="leaf0",
                unit_type="leaf",
                text="leaf A",
                level=0,
                position=0,
                parent_id="root",
                state=leaf_state,
                metadata={
                    "proxy_loss": 0.25,
                    "oracle_loss": 0.05,
                    "observed": True,
                    "propensity": 0.5,
                },
            ),
            TreeNode(
                node_id="leaf1",
                unit_type="leaf",
                text="leaf B",
                label=0.2,
                level=0,
                position=1,
                parent_id="root",
            ),
            TreeNode(
                node_id="root",
                unit_type="root",
                text="leaf A leaf B",
                label=0.5,
                level=1,
                position=0,
                left_child_id="leaf0",
                right_child_id="leaf1",
                metadata={"target": "f"},
            ),
        ),
    )

    assert tree.levels() == [["leaf0", "leaf1"], ["root"]]
    assert tree.root().node_id == "root"
    assert [node.node_id for node in tree.leaves()] == ["leaf0", "leaf1"]
    assert [node.node_id for node in iter_tree_units(tree, order="root_first")][:1] == ["root"]
    assert validate_tree_record(tree) == ()
    summary = tree_summary(tree)
    assert summary["n_nodes"] == 3
    assert summary["n_leaves"] == 2
    assert summary["n_state_nodes"] == 1
    rows = local_law_rows_from_tree_records([tree])
    assert len(rows) == 1
    assert rows[0].law_kind == LawKind.C1_LEAF
    assert rows[0].metadata["unit_id"] == "doc1:leaf0"
    assert rows[0].corrected_loss() == pytest.approx(-0.15)

    bad_tree = TreeRecord(tree_id="bad", nodes=(TreeNode(node_id="root", left_child_id="missing"),))
    assert "left_child_id does not exist" in validate_tree_record(bad_tree)[0]

    path = write_tree_records_jsonl(tmp_path / "trees.jsonl", [tree])
    loaded = [
        TreeRecord.from_value(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(loaded) == 1
    assert loaded[0].get_node("leaf0").state.kind == "manifesto_policy"

    preferences = preference_units_from_trees(loaded)
    general = preferences.to_records("general")
    assert [row["node_id"] for row in general] == ["leaf0", "leaf1", "root"]
    assert general[0]["unit_id"] == "doc1:leaf0"
    assert general[0]["parent_id"] == "root"
    assert general[2]["unit_type"] == "root"
    assert general[2]["target"] == "f"

    supervised = preferences.to_records("supervised")
    assert len(supervised) == 3
    assert supervised[0]["value"]["kind"] == "manifesto_policy"
    assert supervised[1]["value"] == 0.2
    assert supervised[2]["value"] == 0.5

    bundle = write_artifact_bundle(tmp_path / "bundle", trees=[tree], preference_data=preferences)
    assert Path(bundle["files"]["tree_records"]).exists()
    assert Path(bundle["files"]["manifest"]).exists()
    assert bundle["trees"]["n_nodes"] == 3
    assert bundle["preferences"]["counts"]["supervised"] == 3
    assert bundle["local_laws"]["local_law_objective"]["row_count"] == 1

    thinkingtrees_like = {
        "doc_id": "tt_doc",
        "document_text": "B then A",
        "document_score": 0.7,
        "levels": [["leaf_b", "leaf_a"], ["root"]],
        "nodes": {
            "root": {
                "node_id": "root",
                "level": 1,
                "text": "B then A",
                "score": 0.7,
                "left_child_id": "leaf_b",
                "right_child_id": "leaf_a",
            },
            "leaf_a": {"node_id": "leaf_a", "level": 0, "text": "A", "score": 0.1},
            "leaf_b": {"node_id": "leaf_b", "level": 0, "text": "B", "score": 0.2},
        },
    }
    converted = TreeRecord.from_value(thinkingtrees_like)
    assert [node.node_id for node in converted.nodes] == ["leaf_b", "leaf_a", "root"]
    assert [node.position for node in converted.nodes] == [0, 1, 0]
    assert converted.root_label == 0.7


def test_preference_dataset_exports_hf_and_optimizer_views() -> None:
    from treepo.methods.preference import Candidate, PreferenceDataset, PreferenceRecord

    dataset = PreferenceDataset()
    assert dataset.append(
        PreferenceRecord(
            record_id="score1",
            unit_id="node1",
            unit_type="qsentence",
            target="g",
            context="Summarize this node.",
            candidates=(
                Candidate(id="good", value="specific evidence", score=0.9),
                Candidate(id="weak", value="generic text", score=0.2),
            ),
        )
    ) is dataset
    dataset.extend(
        [
            {
                "record_id": "pair1",
                "unit_id": "root1",
                "unit_type": "root",
                "target": "f",
                "context": "Score this document.",
                "preferred": "chosen",
                "candidates": [
                    {"id": "chosen", "value": "RILE score: 2"},
                    {"id": "rejected", "value": "RILE score: -2"},
                ],
                "metadata": {"confidence": 0.75},
            },
            PreferenceRecord(
                record_id="rank1",
                unit_id="merge1",
                unit_type="merge",
                target="g",
                context="Merge these states.",
                candidates=(
                    Candidate(id="best", value="best merge", rank=1),
                    Candidate(id="tie_a", value="acceptable merge a", rank=2),
                    Candidate(id="tie_b", value="acceptable merge b", rank=2),
                ),
            ),
        ]
    )

    assert len(dataset) == 3
    assert len(dataset.sample(sample_size=2, seed=0)) == 2
    assert dataset.summary()["targets"] == ["f", "g"]

    hf_dataset = dataset.to_hf_dataset_dict()
    assert len(hf_dataset["units"]) == 3
    assert hf_dataset["units"][0]["unit_type"] == "qsentence"
    assert len(PreferenceDataset.from_value(hf_dataset)) == 3

    assert len(dataset.to_records("dpo")) == 3
    reward_rows = dataset.to_records("reward")
    assert len(reward_rows) == 3
    assert reward_rows[0]["chosen_score"] > reward_rows[0]["rejected_score"]
    grpo_rows = dataset.to_records("grpo")
    assert len(grpo_rows) == 3
    assert grpo_rows[-1]["ranks"] == [1, 2, 2]
    assert len(dataset.to_records("supervised")) == 3


def test_fit_exports_preference_data_without_training_runtime_coupling(tmp_path: Path) -> None:
    from treepo import fit

    class NoopFamily:
        name = "noop"

        def train_f(self, *, f_init, g, traces, output_dir, iteration):
            return f_init

        def train_g(self, *, g_init, f, traces, output_dir, iteration):
            return g_init

        def score_roots_with_f(self, *, f, g, trees):
            return [None] * len(trees)

        def validate_artifact(self, *, kind, artifact):
            return None

    result = fit(
        {
            "family": "noop",
            "train_data": [],
            "eval_data": [],
            "preference_data": [
                {
                    "pair_id": "pref1",
                    "target": "g",
                    "prompt": "Choose the better summary.",
                    "response_a": "summary a",
                    "response_b": "summary b",
                    "preferred": "B",
                    "confidence": 0.75,
                }
            ],
            "backend_config": {
                "family_runtime": NoopFamily(),
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.summary["preference_data"]["n_units"] == 1
    assert result.summary["preference_data"]["n_candidates"] == 2
    pref_artifacts = result.artifacts["preference_data"]
    assert pref_artifacts["counts"]["dpo"] == 1
    assert pref_artifacts["counts"]["units"] == 1
    assert Path(pref_artifacts["files"]["dpo"]).exists()
    assert Path(pref_artifacts["files"]["hf_dataset"]).exists()
    manifest = json.loads((tmp_path / "treepo_methods_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["spec"]["has_preference_data"] is True
    assert manifest["preference_data"]["counts"]["reward"] == 1


def test_fit_accepts_preference_dataset_as_preference_data(tmp_path: Path) -> None:
    from treepo import fit
    from treepo.methods.preference import Candidate, PreferenceDataset, PreferenceRecord

    class NoopFamily:
        name = "noop"

        def train_f(self, *, f_init, g, traces, output_dir, iteration):
            return f_init

        def train_g(self, *, g_init, f, traces, output_dir, iteration):
            return g_init

        def score_roots_with_f(self, *, f, g, trees):
            return [None] * len(trees)

        def validate_artifact(self, *, kind, artifact):
            return None

    preferences = PreferenceDataset(
        [
            PreferenceRecord(
                record_id="fb1",
                unit_id="node1",
                unit_type="node",
                target="g",
                context="Choose the better node summary.",
                candidates=(
                    Candidate(id="chosen", value="specific summary", score=1.0),
                    Candidate(id="rejected", value="generic summary", score=0.0),
                ),
            )
        ]
    )
    result = fit(
        {
            "family": "noop",
            "train_data": [],
            "eval_data": [],
            "preference_data": preferences,
            "backend_config": {
                "family_runtime": NoopFamily(),
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.summary["preference_data"]["n_units"] == 1
    assert result.artifacts["preference_data"]["counts"]["dpo"] == 1


def test_evidence_builder_has_stable_empty_sections() -> None:
    evidence = build_evidence(
        status="success",
        metrics={},
        summary={"family": "learnable_constant", "schedule": "fg", "n_iterations": 0},
        artifacts={},
    )

    assert evidence["version"] == "0.1"
    assert evidence["run"]["family"] == "learnable_constant"
    assert evidence["root"]["present"] is False
    assert evidence["preferences"]["present"] is False
    assert evidence["statistic"]["present"] is False
    assert evidence["local_laws"]["present"] is False
    assert evidence["predictions"]["present"] is False


def test_evidence_builder_summarizes_preferences_statistics_and_laws() -> None:
    rows = (
        LocalLawAuditRow(
            row_id="r0",
            law_kind="c1",
            proxy_loss=0.25,
            oracle_loss=0.10,
            observed=True,
            propensity=0.5,
        ),
    )
    evidence = build_evidence(
        status="success",
        metrics={"n": 1.0, "internal_f_mae": 0.0},
        summary={"family": "classical_sketch", "schedule": "fg", "n_iterations": 1},
        artifacts={
            "preference_data": {
                "summary": {"n_units": 1, "n_candidates": 2},
                "counts": {"dpo": 1},
                "files": {"dataset": "preference_dataset.json"},
            },
            "statistic": {
                "info": {"name": "hll", "state_kind": "hll", "exact": True},
                "local_law_summary": {"objective": 0.0, "row_count": 1},
                "local_law_row_count": 1,
            },
            "prediction_records": ["predictions.jsonl"],
        },
        local_law_rows=rows,
    )

    assert evidence["root"]["present"] is True
    assert evidence["preferences"]["present"] is True
    assert evidence["preferences"]["counts"]["dpo"] == 1
    assert evidence["statistic"]["present"] is True
    assert evidence["statistic"]["info"]["exact"] is True
    assert evidence["local_laws"]["present"] is True
    assert evidence["local_laws"]["source"] == "rows"
    assert "leaf_preservation" in evidence["local_laws"]["by_law_kind"]
    assert evidence["predictions"]["present"] is True
