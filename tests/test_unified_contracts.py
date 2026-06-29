from __future__ import annotations

import json
from pathlib import Path

import pytest

from treepo.local_law import (
    LawKind,
    LocalLawAuditRow,
    compute_influence_weighted_overlap,
    corrected_losses_from_rows,
)
from treepo.certificate import (
    UnifiedLearningComponentEvidence,
    build_error_certificate,
)
from treepo.honesty import (
    ThreeLayerHonestyConfig,
    assign_three_layer_roles,
    role_tuple_for_unit,
)
from treepo.manifest import (
    ArtifactLineage,
    ArtifactRef,
    ManifestRow,
    RunManifestContract,
    Span,
    TopLevelUnit,
)
from treepo.objective import (
    ObjectiveSpec,
    normalize_objective_spec,
    resolve_root_local_objective_weights,
)
from treepo.sampling import ObservationUnitKind, SamplingMetadata


def _lineage() -> ArtifactLineage:
    return ArtifactLineage(
        chunker="chunker:v1",
        g="g:v1",
        f="f:v1",
        oracle_online="oracle:online",
        oracle_eval="oracle:eval",
        query_policy="query:v1",
        proxy="proxy:v1",
    )


def test_manifest_round_trips_and_validates() -> None:
    manifest = RunManifestContract(
        run_id="run_1",
        top_level_units=(TopLevelUnit(unit_id="doc_1", length=100),),
        artifacts=(
            ArtifactRef("chunker:v1"),
            ArtifactRef("g:v1"),
            ArtifactRef("f:v1"),
            ArtifactRef("oracle:online"),
            ArtifactRef("oracle:eval"),
            ArtifactRef("query:v1"),
            ArtifactRef("proxy:v1"),
        ),
        rows=(
            ManifestRow(
                row_id="row_1",
                top_level_unit_id="doc_1",
                fold_id="fold_0",
                split_seed=7,
                roles=role_tuple_for_unit(
                    "doc_1",
                    ThreeLayerHonestyConfig(enabled=True, split_seed=7),
                ),
                artifacts=_lineage(),
                law_kind="c1_leaf",
                support=Span(0, 50, unit="char"),
                observed=True,
                propensity=0.5,
                truth_source="fixture_oracle",
                approx_source="fixture_proxy",
            ),
        ),
    )

    report = manifest.validate()
    assert report.ok, report.to_dict()
    restored = RunManifestContract.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest
    assert len(restored.digest) == 64


def test_manifest_rejects_missing_parent_and_invalid_row_propensity() -> None:
    with pytest.raises(ValueError, match="propensity"):
        ManifestRow(
            row_id="bad",
            top_level_unit_id="doc_1",
            support=Span(0, 1),
            observed=False,
            propensity=0.0,
        )

    manifest = RunManifestContract(
        run_id="run_1",
        top_level_units=(TopLevelUnit(unit_id="doc_1", length=10),),
        rows=(
            ManifestRow(
                row_id="row_1",
                top_level_unit_id="missing",
                support=Span(0, 1),
                observed=False,
                propensity=1.0,
            ),
        ),
    )
    report = manifest.validate(require_artifacts=False)
    assert not report.ok
    assert "missing top-level unit" in report.errors[0]


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
    assert corrected_losses_from_rows(rows) == pytest.approx([0.4, -0.2])
    overlap = compute_influence_weighted_overlap(rows)
    assert overlap.D_lambda == pytest.approx(1.0 / 0.25 + 4.0 / 0.5)
    assert overlap.W_lambda == pytest.approx(4.0)
    assert overlap.effective_sample_size > 0.0


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


def test_sampling_and_honesty_are_deterministic() -> None:
    sampling = SamplingMetadata(
        document_propensity=0.5,
        unit_propensity=0.25,
        label_propensity=1.0,
        unit_kind=ObservationUnitKind.LEAF,
    )
    assert sampling.effective_joint_propensity() == pytest.approx(0.125)
    assert sampling.ipw_weight() == pytest.approx(8.0)

    cfg = ThreeLayerHonestyConfig(enabled=True, split_seed=11)
    assert assign_three_layer_roles("doc_1", cfg) == assign_three_layer_roles("doc_1", cfg)


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
