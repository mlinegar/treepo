"""Public ``g_mode`` contract: identity, fixed, and learned execution.

These tests intentionally use a tiny injected ``FamilyRuntime``.  They cover
the package-level orchestration and persisted provenance without depending on
torch, neuralop, DSPy, or a model server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

import treepo
from treepo.methods import GTrainOutcome
from treepo.methods.contracts import CTreePOLearningSpec
from treepo.methods.preference import Candidate, PreferenceRecord
from treepo.objective import ObjectiveSpec
from treepo.tree import TreeNode, TreeRecord


class RecordingFamily:
    """Small family that makes every f/g orchestration call observable."""

    name = "recording_g_mode"

    def __init__(self) -> None:
        self.f_calls: list[dict[str, Any]] = []
        self.g_calls: list[dict[str, Any]] = []
        self.score_g_artifacts: list[Any] = []
        self.validation_calls: list[tuple[str, Any]] = []
        self.objectives: list[ObjectiveSpec] = []

    def configure_objective(self, objective: ObjectiveSpec) -> None:
        self.objectives.append(objective)

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> dict[str, Any]:
        self.f_calls.append(
            {
                "f_init": f_init,
                "g": g,
                "traces": list(traces),
                "output_dir": output_dir,
                "iteration": iteration,
            }
        )
        return {
            "kind": "recording_f",
            "trained": "f",
            "iteration": int(iteration),
            "value": 0.5,
        }

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> dict[str, Any]:
        self.g_calls.append(
            {
                "g_init": g_init,
                "f": f,
                "traces": list(traces),
                "output_dir": output_dir,
                "iteration": iteration,
            }
        )
        return {
            "kind": "recording_g",
            "trained": "g",
            "iteration": int(iteration),
            "same_g_across_node_roles": True,
            "reduce_g_is_derived": True,
        }

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[float]:
        del f
        self.score_g_artifacts.append(g)
        return [0.5] * len(trees)

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        assert kind in {"f", "g"}
        assert isinstance(artifact, dict)
        self.validation_calls.append((kind, artifact))
        if artifact.get("g_mode") == "fixed":
            assert kind == "g"
            assert artifact["trainable"] is False
            return
        assert artifact["trained"] == kind


class NoOpRecordingFamily(RecordingFamily):
    """A train_g call whose returned artifact is exactly unchanged."""

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        super().train_g(
            g_init=g_init,
            f=f,
            traces=traces,
            output_dir=output_dir,
            iteration=iteration,
        )
        return g_init


class ExplicitOutcomeFamily(RecordingFamily):
    """Exercise the public outcome contract independently of artifact identity."""

    def __init__(self, *, update_performed: bool, return_same_artifact: bool) -> None:
        super().__init__()
        self.update_performed = bool(update_performed)
        self.return_same_artifact = bool(return_same_artifact)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> GTrainOutcome:
        changed = super().train_g(
            g_init=g_init,
            f=f,
            traces=traces,
            output_dir=output_dir,
            iteration=iteration,
        )
        return GTrainOutcome(
            artifact=g_init if self.return_same_artifact else changed,
            update_performed=self.update_performed,
            reason="test_explicit_outcome",
        )


class GradientPathRecordingFamily(RecordingFamily):
    """Return direct shared-g call-domain gradient-path provenance."""

    def __init__(self, *, merge_domain: bool) -> None:
        super().__init__()
        self.merge_domain = bool(merge_domain)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> GTrainOutcome:
        artifact = super().train_g(
            g_init=g_init,
            f=f,
            traces=traces,
            output_dir=output_dir,
            iteration=iteration,
        )
        artifact.update(
            {
                "leaf_domain_gradient_path_present": True,
                "merge_domain_gradient_path_present": self.merge_domain,
            }
        )
        return GTrainOutcome(
            artifact=artifact,
            update_performed=True,
            reason="test_gradient_path_provenance",
        )


def _tree(tree_id: str, *, n_leaves: int, split: str) -> TreeRecord:
    if n_leaves == 1:
        nodes = (
            TreeNode(
                node_id=f"{tree_id}-root",
                unit_type="root",
                text=f"complete document {tree_id}",
                level=0,
                position=0,
            ),
        )
    else:
        nodes = tuple(
            TreeNode(
                node_id=f"{tree_id}-leaf-{index}",
                unit_type="leaf",
                text=f"document {tree_id} part {index}",
                level=0,
                position=index,
            )
            for index in range(n_leaves)
        )
    return TreeRecord(
        tree_id=tree_id,
        text=f"complete document {tree_id}",
        root_label=0.5,
        nodes=nodes,
        metadata={"split": split},
    )


def _fit_config(
    tmp_path: Path,
    family: RecordingFamily,
    *,
    g_mode: str | None,
    n_leaves: int = 1,
    max_iterations: int = 2,
    representation: str | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "space_kind": "test.g_mode.v1",
        "family": family.name,
        "train_data": [_tree("train", n_leaves=n_leaves, split="train")],
        "eval_data": [_tree("test", n_leaves=n_leaves, split="test")],
        "backend_config": {
            "family_runtime": family,
            "output_dir": str(tmp_path),
        },
        "axis": {
            "max_iterations": int(max_iterations),
            "leaf_count": int(n_leaves),
            "representation": representation or ("full_doc" if n_leaves == 1 else "ctree"),
        },
    }
    if g_mode is not None:
        config["g_mode"] = g_mode
    return config


def _fixed_g() -> dict[str, Any]:
    return {
        "kind": "fixture_mass_weighted_g",
        "g_mode": "fixed",
        "operator": "denominator_weighted_mean",
        "trainable": False,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
    }


def _preference(target: str) -> PreferenceRecord:
    return PreferenceRecord(
        unit_id=f"{target}-unit",
        unit_type="root",
        target=target,
        context="training context",
        candidates=(
            Candidate(
                id="gold",
                value=0.5,
                score=1.0,
                preferred=True,
            ),
        ),
    )


def test_learning_spec_defaults_to_learned_and_round_trips_g_mode() -> None:
    default = CTreePOLearningSpec.from_mapping(
        {
            "space_kind": "test",
            "family": "recording",
        }
    )
    assert default.g_mode == "learned"
    assert default.schedule == "fg"
    assert default.to_dict()["g_mode"] == "learned"

    for mode, schedule in (("identity", "f"), ("fixed", "f"), ("learned", "fg")):
        spec = CTreePOLearningSpec.from_mapping(
            {
                "space_kind": "test",
                "family": "recording",
                "g_mode": mode,
            }
        )
        restored = CTreePOLearningSpec.from_mapping(spec.to_dict())
        assert restored.g_mode == mode
        assert restored.schedule == schedule


def test_learning_spec_rejects_invalid_or_inconsistent_g_mode() -> None:
    with pytest.raises(ValueError, match="g_mode must be one of"):
        CTreePOLearningSpec.from_mapping(
            {
                "space_kind": "test",
                "family": "recording",
                "g_mode": "sometimes",
            }
        )

    with pytest.raises(ValueError, match="requires schedule='f'"):
        CTreePOLearningSpec.from_mapping(
            {
                "space_kind": "test",
                "family": "recording",
                "g_mode": "identity",
                "schedule": "fg",
            }
        )
    with pytest.raises(ValueError, match="requires schedule='fg'"):
        CTreePOLearningSpec.from_mapping(
            {
                "space_kind": "test",
                "family": "recording",
                "g_mode": "learned",
                "schedule": "f",
            }
        )


def test_identity_public_fit_is_f_only_and_persists_complete_provenance(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    result = treepo.fit(
        _fit_config(
            tmp_path,
            family,
            g_mode="identity",
            max_iterations=2,
        )
    )

    identity_g = {
        "kind": "treepo_identity_g",
        "g_mode": "identity",
        "operator": "identity",
        "trainable": False,
        "train_g_enabled": False,
        "merge_call_count": 0,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
    }
    assert result.status == "success"
    assert len(family.f_calls) == 1
    assert family.g_calls == []
    assert family.f_calls[0]["g"] == identity_g
    assert family.score_g_artifacts and all(
        artifact == identity_g for artifact in family.score_g_artifacts
    )
    assert result.artifacts["g"] == identity_g

    # k=2 is the alternating g slot. Identity mode elides it rather than
    # emitting a nominal no-op training/evaluation row.
    assert [row["iteration"] for row in result.history] == [0, 1]
    assert [row["trained"] for row in result.history] == ["none", "f"]
    assert [row["g_mode"] for row in result.history] == ["identity", "identity"]
    assert [row["g_degree"] for row in result.history] == [0, 0]
    assert [row["f_degree"] for row in result.history] == [1, 2]
    assert [row["stage_label"] for row in result.history] == [
        "f^1 g=identity",
        "f^2 g=identity",
    ]

    contract = result.summary["g_contract"]
    assert result.summary["schedule"] == "f"
    assert result.summary["g_mode"] == "identity"
    assert result.summary["n_iterations"] == 2
    assert result.summary["configured_max_iterations"] == 2
    assert result.summary["f_update_count"] == 1
    assert result.summary["g_update_count"] == 0
    assert contract == {
        "mode": "identity",
        "operator": "fixed_identity",
        "initial_g_artifact_present": False,
        "fit_status": "not_trainable",
        "learned_this_run": False,
        "trainable": False,
        "train_g_enabled": False,
        "configured_max_iterations": 2,
        "executed_record_count": 2,
        "executed_iteration_indices": [0, 1],
        "f_update_count": 1,
        "g_update_count": 0,
        "train_g_call_count": 0,
        "merge_call_count": 0,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
        "shared_g_evidence_source": "package_canonical_identity",
        "g_training_call_roles": [],
        "g_training_role_evidence_source": "no_train_g_calls",
        "merge_domain_training_observed": False,
        "shared_g_updated_with_merge_domain": False,
        "skipped_g_interpretation": "identity",
        "declared_representation": "full_doc",
        "topology_kind": "singleton",
        "declared_leaf_count": 1,
        "leaf_g_application_count_per_tree": 1,
        "merge_application_count_per_tree": 0,
        "leaf_g_application_materialization": "semantic_identity_elided",
        "leaf_g_materialized_application_count_per_tree": 0,
        "singleton": True,
        "direct": True,
        "summarized": False,
        "composition_present": False,
        "direct_readout_singleton": True,
        "summarized_singleton": False,
        "identity_single_leaf_contract": True,
        "canonical_full_document": True,
        "training_composition_present": False,
        "all_singleton_training": True,
        "train_tree_count": 1,
        "eval_tree_count": 1,
        "observed_leaf_count_min": 1,
        "observed_leaf_count_max": 1,
    }

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    results = json.loads(Path(result.artifacts["results_json"]).read_text(encoding="utf-8"))
    evidence = result.artifacts["evidence"]

    assert manifest["spec"]["g_mode"] == "identity"
    assert manifest["spec"]["schedule"] == "f"
    assert manifest["summary"]["g_contract"] == contract
    assert results["g_contract"] == contract
    assert results["cell"]["g_mode"] == "identity"
    assert results["cell"]["g_contract"] == contract
    assert results["cost"]["one_time_compute"]["configured_max_iterations"] == 2
    assert results["cost"]["one_time_compute"]["f_update_count"] == 1
    assert results["cost"]["one_time_compute"]["g_update_count"] == 0
    assert evidence["run"]["g_mode"] == "identity"
    assert evidence["run"]["g_contract"] == contract
    assert evidence["run"]["f_update_count"] == 1
    assert evidence["run"]["g_update_count"] == 0


@pytest.mark.parametrize(
    "representation",
    ["full_doc", "full_doc_direct", "ctree_base_summary", "ctree"],
)
@pytest.mark.parametrize("g_mode", ["identity", "fixed", "learned"])
def test_singleton_ctree_accepts_every_g_mode_and_alias(
    tmp_path: Path,
    representation: str,
    g_mode: str,
) -> None:
    family = RecordingFamily()
    config = _fit_config(
        tmp_path / representation / g_mode,
        family,
        g_mode=g_mode,
        representation=representation,
    )
    if g_mode == "fixed":
        config["initial_artifacts"] = {"g": _fixed_g()}

    result = treepo.fit(config)

    contract = result.summary["g_contract"]
    assert result.status == "success"
    assert contract["topology_kind"] == "singleton"
    assert contract["declared_leaf_count"] == 1
    assert contract["leaf_g_application_count_per_tree"] == 1
    assert contract["leaf_g_materialized_application_count_per_tree"] == (
        0 if g_mode == "identity" else 1
    )
    assert contract["merge_application_count_per_tree"] == 0
    assert contract["singleton"] is True
    assert contract["composition_present"] is False
    assert contract["direct"] is (g_mode == "identity")
    assert contract["summarized"] is (g_mode != "identity")
    assert contract["identity_single_leaf_contract"] is (g_mode == "identity")
    assert contract["direct_readout_singleton"] is (g_mode == "identity")
    assert contract["summarized_singleton"] is (g_mode != "identity")
    assert contract["same_g_across_node_roles"] is True
    assert contract["reduce_g_is_derived"] is True
    assert contract["g_training_call_roles"] == (["leaf"] if g_mode == "learned" else [])
    assert contract["g_training_role_evidence_source"] == (
        "inferred_from_train_g_and_topology" if g_mode == "learned" else "no_train_g_calls"
    )
    assert contract["merge_domain_training_observed"] is False
    assert contract["shared_g_updated_with_merge_domain"] is False
    assert len(family.g_calls) == (1 if g_mode == "learned" else 0)


def test_identity_g_is_rejected_on_a_recursive_ctree(tmp_path: Path) -> None:
    family = RecordingFamily()
    with pytest.raises(ValueError, match="identity.*singleton direct path"):
        treepo.fit(
            _fit_config(
                tmp_path,
                family,
                g_mode="identity",
                n_leaves=3,
                representation="ctree_recursive",
            )
        )
    assert family.f_calls == family.g_calls == []


def test_ctree_recursive_rejects_a_declared_singleton(tmp_path: Path) -> None:
    family = RecordingFamily()
    config = _fit_config(
        tmp_path,
        family,
        g_mode="learned",
        representation="ctree_recursive",
    )

    with pytest.raises(ValueError, match=r"ctree_recursive.*leaf_count >= 2"):
        treepo.fit(config)
    assert family.f_calls == []
    assert family.g_calls == []


def test_singleton_representation_rejects_known_multileaf_topology(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    config = _fit_config(
        tmp_path,
        family,
        g_mode="learned",
        n_leaves=2,
        representation="full_doc",
    )
    del config["axis"]["leaf_count"]

    with pytest.raises(ValueError, match="requires exactly 1 exposed leaves"):
        treepo.fit(config)
    assert family.f_calls == []
    assert family.g_calls == []


def test_declared_leaf_count_must_agree_with_known_topology(tmp_path: Path) -> None:
    family = RecordingFamily()
    config = _fit_config(
        tmp_path,
        family,
        g_mode="learned",
        representation="ctree",
    )
    config["axis"]["leaf_count"] = 2

    with pytest.raises(ValueError, match="axis.leaf_count=2 disagrees"):
        treepo.fit(config)
    assert family.f_calls == []
    assert family.g_calls == []


def test_singleton_leaf_may_store_a_summary_instead_of_raw_document(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    summarized_leaf = TreeRecord(
        tree_id="summary",
        text="the complete long document",
        root_label=0.5,
        nodes=(
            TreeNode(
                node_id="summary-root",
                unit_type="root",
                text="a g-produced state or summary",
                level=0,
                position=0,
            ),
        ),
        metadata={"split": "train"},
    )
    config = _fit_config(
        tmp_path,
        family,
        g_mode="learned",
        representation="ctree_base_summary",
    )
    config["train_data"] = [summarized_leaf]

    result = treepo.fit(config)

    assert result.status == "success"
    assert result.summary["g_contract"]["summarized"] is True
    assert result.summary["g_contract"]["merge_application_count_per_tree"] == 0


def test_uninspectable_nodes_are_not_misclassified_as_zero_leaf_trees(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    uninspectable = [
        TreeRecord(
            tree_id=f"uninspectable-{split}",
            text="complete document",
            root_label=0.5,
            nodes=(),
            metadata={"split": split},
        )
        for split in ("train", "test")
    ]
    config = _fit_config(tmp_path, family, g_mode="identity", representation="full_doc")
    config["train_data"] = [uninspectable[0]]
    config["eval_data"] = [uninspectable[1]]

    result = treepo.fit(config)

    contract = result.summary["g_contract"]
    assert result.status == "success"
    assert contract["topology_kind"] == "singleton"
    assert contract["observed_leaf_count_min"] is None
    assert contract["observed_leaf_count_max"] is None


def test_explicit_zero_leaf_declaration_is_rejected(tmp_path: Path) -> None:
    family = RecordingFamily()
    config = _fit_config(
        tmp_path,
        family,
        g_mode="identity",
        representation="ctree",
    )
    config["axis"]["leaf_count"] = 0

    with pytest.raises(ValueError, match="axis.leaf_count must be >= 1"):
        treepo.fit(config)

    explicit_family = RecordingFamily()
    explicit = TreeRecord(
        tree_id="known-empty",
        text="complete document",
        root_label=0.5,
        nodes=(),
        metadata={"split": "train", "leaf_count": 0},
    )
    explicit_config = _fit_config(
        tmp_path / "known-empty",
        explicit_family,
        g_mode="identity",
        representation="ctree",
    )
    del explicit_config["axis"]["leaf_count"]
    explicit_config["train_data"] = [explicit]
    with pytest.raises(ValueError, match="requires at least 1 exposed leaves"):
        treepo.fit(explicit_config)


def test_identity_rejects_nonidentity_g_warmstart(tmp_path: Path) -> None:
    family = RecordingFamily()
    config = _fit_config(tmp_path, family, g_mode="identity")
    config["initial_artifacts"] = {
        "g": {
            "kind": "previously_learned_g",
            "g_mode": "learned",
        }
    }

    with pytest.raises(ValueError, match=r"rejects initial_artifacts\['g'\]"):
        treepo.fit(config)
    assert family.f_calls == []
    assert family.g_calls == []


def test_identity_rejects_even_a_self_declared_identity_artifact(tmp_path: Path) -> None:
    family = RecordingFamily()
    config = _fit_config(tmp_path, family, g_mode="identity")
    config["initial_artifacts"] = {
        "g": {
            "kind": "spoofed_identity",
            "g_mode": "identity",
            "operator": "nonidentity_concat",
            "trainable": True,
        }
    }

    with pytest.raises(ValueError, match=r"rejects initial_artifacts\['g'\]"):
        treepo.fit(config)
    assert family.validation_calls == []


@pytest.mark.parametrize("g_mode", ["identity", "fixed", "learned"])
@pytest.mark.parametrize("supervision_kind", ["merge_weight", "c3_objective"])
def test_singleton_topology_rejects_merge_or_c3_supervision(
    tmp_path: Path,
    g_mode: str,
    supervision_kind: str,
) -> None:
    family = RecordingFamily()
    config = _fit_config(
        tmp_path / supervision_kind / g_mode,
        family,
        g_mode=g_mode,
    )
    if g_mode == "fixed":
        config["initial_artifacts"] = {"g": _fixed_g()}
    if supervision_kind == "merge_weight":
        config["merge_weight"] = 1.0
        match = "training topology has no merge applications"
    else:
        config["backend_config"]["objective"] = ObjectiveSpec(
            objective_family="root_plus_local_laws",
            local_law_estimator="oracle_state",
            local_law_weight=0.5,
            root_share=0.5,
            local_law_component_weights={"merge_preservation": 0.5},
        )
        match = "empty C3 population"

    with pytest.raises(ValueError, match=match):
        treepo.fit(config)
    assert family.f_calls == []
    assert family.g_calls == []


def test_nonidentity_singleton_allows_c1_without_inventing_c3(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    config = _fit_config(tmp_path, family, g_mode="learned")
    config["backend_config"]["objective"] = ObjectiveSpec(
        objective_family="root_plus_local_laws",
        local_law_estimator="oracle_state",
        local_law_weight=0.5,
        root_share=0.5,
        local_law_component_weights={"leaf_preservation": 0.5},
    )

    result = treepo.fit(config)

    contract = result.summary["g_contract"]
    assert result.status == "success"
    assert family.objectives
    assert contract["same_g_across_node_roles"] is True
    assert contract["reduce_g_is_derived"] is True
    assert contract["g_training_call_roles"] == ["leaf"]
    assert contract["g_training_role_evidence_source"] == ("inferred_from_train_g_and_topology")
    assert contract["merge_domain_training_observed"] is False
    assert contract["shared_g_updated_with_merge_domain"] is False
    assert contract["composition_present"] is False


@pytest.mark.parametrize("preference_target", ["g", "both"])
def test_identity_rejects_g_and_both_preferences(
    tmp_path: Path,
    preference_target: str,
) -> None:
    family = RecordingFamily()
    config = _fit_config(
        tmp_path / preference_target,
        family,
        g_mode="identity",
    )
    config["preference_data"] = [_preference(preference_target)]

    with pytest.raises(ValueError, match="preference records targeting g or both"):
        treepo.fit(config)
    assert family.f_calls == []
    assert family.g_calls == []


def test_fixed_mode_requires_a_concrete_operator_artifact(tmp_path: Path) -> None:
    family = RecordingFamily()
    config = _fit_config(
        tmp_path,
        family,
        g_mode="fixed",
        n_leaves=3,
    )
    with pytest.raises(ValueError, match="requires an explicit initial_artifacts"):
        treepo.fit(config)


@pytest.mark.parametrize(
    ("artifact", "error_type", "match"),
    [
        ("not-an-operator", TypeError, "must be a mapping"),
        (
            {"kind": "missing_contract"},
            ValueError,
            "missing required fixed-operator fields",
        ),
        (
            {
                "kind": "wrong_mode",
                "g_mode": "learned",
                "operator": "frozen",
                "trainable": False,
                "same_g_across_node_roles": True,
                "reduce_g_is_derived": True,
            },
            ValueError,
            "must declare g_mode='fixed'",
        ),
        (
            {
                "kind": "trainable",
                "g_mode": "fixed",
                "operator": "frozen",
                "trainable": True,
                "same_g_across_node_roles": True,
                "reduce_g_is_derived": True,
            },
            ValueError,
            "trainable=false",
        ),
    ],
)
def test_fixed_mode_rejects_malformed_operator_artifacts(
    tmp_path: Path,
    artifact: Any,
    error_type: type[Exception],
    match: str,
) -> None:
    family = RecordingFamily()
    config = _fit_config(tmp_path, family, g_mode="fixed", n_leaves=3)
    config["initial_artifacts"] = {"g": artifact}

    with pytest.raises(error_type, match=match):
        treepo.fit(config)
    assert family.validation_calls == []


def test_fixed_allows_multileaf_data_and_never_trains_g(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    fixed_g = _fixed_g()
    config = _fit_config(
        tmp_path,
        family,
        g_mode="fixed",
        n_leaves=3,
        max_iterations=2,
    )
    config["initial_artifacts"] = {"g": fixed_g}
    result = treepo.fit(config)

    assert result.status == "success"
    assert len(family.f_calls) == 1
    assert family.g_calls == []
    assert family.validation_calls[0] == ("g", fixed_g)
    assert family.f_calls[0]["g"] is fixed_g
    assert result.artifacts["g"] is fixed_g
    assert [row["iteration"] for row in result.history] == [0, 1]
    assert [row["g_degree"] for row in result.history] == [0, 0]
    assert result.summary["schedule"] == "f"
    assert result.summary["g_mode"] == "fixed"
    assert result.summary["g_contract"]["initial_g_artifact_present"] is True
    assert result.summary["g_contract"]["operator"] == ("fixed_nonidentity_or_family_owned")
    assert result.summary["g_contract"]["fit_status"] == "not_trainable"
    assert result.summary["g_contract"]["learned_this_run"] is False
    assert result.summary["g_contract"]["canonical_full_document"] is False
    contract = result.summary["g_contract"]
    assert contract["topology_kind"] == "recursive"
    assert contract["leaf_g_application_count_per_tree"] == 3
    assert contract["merge_application_count_per_tree"] == 2
    assert contract["summarized"] is True
    assert contract["composition_present"] is True
    assert contract["g_training_call_roles"] == []
    assert contract["g_training_role_evidence_source"] == "no_train_g_calls"
    assert contract["merge_domain_training_observed"] is False
    assert contract["shared_g_updated_with_merge_domain"] is False
    assert result.summary["f_update_count"] == 1
    assert result.summary["g_update_count"] == 0


def test_learned_default_preserves_alternating_f_then_g_behavior(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    result = treepo.fit(
        _fit_config(
            tmp_path,
            family,
            g_mode=None,
            n_leaves=2,
            max_iterations=2,
        )
    )

    assert result.status == "success"
    assert len(family.f_calls) == 1
    assert len(family.g_calls) == 1
    assert [row["iteration"] for row in result.history] == [0, 1, 2]
    assert [row["trained"] for row in result.history] == ["none", "f", "g"]
    assert [row["g_mode"] for row in result.history] == [
        "learned",
        "learned",
        "learned",
    ]
    assert [row["f_degree"] for row in result.history] == [1, 2, 2]
    assert [row["g_degree"] for row in result.history] == [1, 1, 2]
    assert result.artifacts["g"]["kind"] == "recording_g"
    assert result.summary["schedule"] == "fg"
    assert result.summary["g_mode"] == "learned"
    assert result.summary["f_update_count"] == 1
    assert result.summary["g_update_count"] == 1
    assert result.summary["g_contract"]["train_g_enabled"] is True
    assert result.summary["g_contract"]["operator"] == "learned_shared"
    assert result.summary["g_contract"]["fit_status"] == "fitted_this_run"
    assert result.summary["g_contract"]["learned_this_run"] is True
    assert result.summary["g_contract"]["executed_iteration_indices"] == [0, 1, 2]
    contract = result.summary["g_contract"]
    assert contract["topology_kind"] == "recursive"
    assert contract["leaf_g_application_count_per_tree"] == 2
    assert contract["merge_application_count_per_tree"] == 1
    assert contract["same_g_across_node_roles"] is True
    assert contract["reduce_g_is_derived"] is True
    assert contract["g_training_call_roles"] == ["leaf", "merge"]
    assert contract["g_training_role_evidence_source"] == ("inferred_from_train_g_and_topology")
    assert contract["merge_domain_training_observed"] is True
    assert contract["shared_g_updated_with_merge_domain"] is True
    assert contract["composition_present"] is True


@pytest.mark.parametrize("merge_domain", [False, True])
def test_direct_gradient_path_provenance_overrides_recursive_topology(
    tmp_path: Path,
    merge_domain: bool,
) -> None:
    family = GradientPathRecordingFamily(merge_domain=merge_domain)
    result = treepo.fit(
        _fit_config(
            tmp_path,
            family,
            g_mode="learned",
            n_leaves=2,
            max_iterations=2,
        )
    )

    contract = result.summary["g_contract"]
    assert contract["same_g_across_node_roles"] is True
    assert contract["reduce_g_is_derived"] is True
    assert contract["g_training_call_roles"] == (["leaf", "merge"] if merge_domain else ["leaf"])
    assert contract["g_training_role_evidence_source"] == ("g_artifact_gradient_path_presence")
    assert contract["merge_domain_training_observed"] is merge_domain
    assert contract["shared_g_updated_with_merge_domain"] is merge_domain
    assert contract["composition_present"] is True


def test_trainable_g_without_a_g_update_is_not_reported_as_learned(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    result = treepo.fit(
        _fit_config(
            tmp_path,
            family,
            g_mode="learned",
            n_leaves=2,
            max_iterations=1,
        )
    )

    contract = result.summary["g_contract"]
    assert contract["initial_g_artifact_present"] is False
    assert family.g_calls == []
    assert contract["mode"] == "learned"
    assert contract["operator"] == "trainable_not_updated_this_run"
    assert contract["fit_status"] == "not_updated_this_run"
    assert contract["learned_this_run"] is False
    assert contract["skipped_g_interpretation"] == ("trainable_not_updated_this_run")


def test_runtime_unwraps_g_train_outcome_warmstart_before_scoring(
    tmp_path: Path,
) -> None:
    family = RecordingFamily()
    seed_g = {"kind": "seed_g", "trained": "g", "iteration": 0}
    config = _fit_config(
        tmp_path,
        family,
        g_mode="learned",
        n_leaves=2,
        max_iterations=0,
    )
    config["initial_artifacts"] = {
        "g": GTrainOutcome(
            artifact=seed_g,
            update_performed=True,
            reason="previous_run",
        )
    }

    result = treepo.fit(config)

    assert result.status == "success"
    assert result.artifacts["g"] is seed_g
    assert family.score_g_artifacts == [seed_g]
    assert result.summary["g_contract"]["g_update_count"] == 0


def test_noop_train_g_call_is_not_reported_as_a_realized_update(tmp_path: Path) -> None:
    family = NoOpRecordingFamily()
    seed_g = {"kind": "seed_g", "trained": "g", "iteration": 0}
    config = _fit_config(
        tmp_path,
        family,
        g_mode="learned",
        n_leaves=2,
        max_iterations=2,
    )
    config["initial_artifacts"] = {"g": seed_g}

    result = treepo.fit(config)

    contract = result.summary["g_contract"]
    assert len(family.g_calls) == 1
    assert result.artifacts["g"] is seed_g
    assert result.history[-1]["extra"]["g_update_performed"] is False
    assert result.history[-1]["extra"]["g_update_signal"] == "legacy_unchanged_artifact"
    assert contract["train_g_call_count"] == 1
    assert contract["g_update_count"] == 0
    assert contract["g_training_call_roles"] == ["leaf", "merge"]
    assert contract["merge_domain_training_observed"] is True
    assert contract["shared_g_updated_with_merge_domain"] is False
    assert contract["operator"] == "trainable_not_updated_this_run"
    assert contract["learned_this_run"] is False
    assert contract["same_g_across_node_roles"] is False
    assert contract["reduce_g_is_derived"] is False
    assert contract["shared_g_evidence_source"] == "missing_g_artifact_contract"


@pytest.mark.parametrize(
    ("update_performed", "return_same_artifact", "expected_updates"),
    [(False, False, 0), (True, True, 1)],
)
def test_explicit_g_train_outcome_is_authoritative(
    tmp_path: Path,
    update_performed: bool,
    return_same_artifact: bool,
    expected_updates: int,
) -> None:
    family = ExplicitOutcomeFamily(
        update_performed=update_performed,
        return_same_artifact=return_same_artifact,
    )
    seed_g = {
        "kind": "seed_g",
        "trained": "g",
        "iteration": 0,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
    }
    config = _fit_config(
        tmp_path,
        family,
        g_mode="learned",
        n_leaves=2,
        max_iterations=2,
    )
    config["initial_artifacts"] = {"g": seed_g}

    result = treepo.fit(config)

    contract = result.summary["g_contract"]
    assert result.history[-1]["extra"]["g_update_performed"] is update_performed
    assert result.history[-1]["extra"]["g_update_signal"] == "test_explicit_outcome"
    assert contract["train_g_call_count"] == 1
    assert contract["g_update_count"] == expected_updates
    assert contract["learned_this_run"] is bool(expected_updates)
    assert contract["operator"] == (
        "learned_shared" if expected_updates else "learned_shared_reused"
    )


def test_g_update_without_one_operator_evidence_is_not_reported_as_shared(
    tmp_path: Path,
) -> None:
    family = ExplicitOutcomeFamily(
        update_performed=True,
        return_same_artifact=True,
    )
    seed_g = {"kind": "unverified_g", "trained": "g", "iteration": 0}
    config = _fit_config(
        tmp_path,
        family,
        g_mode="learned",
        n_leaves=2,
        max_iterations=2,
    )
    config["initial_artifacts"] = {"g": seed_g}

    result = treepo.fit(config)

    contract = result.summary["g_contract"]
    assert contract["g_update_count"] == 1
    assert contract["learned_this_run"] is True
    assert contract["operator"] == "learned_unverified_operator"
    assert contract["fit_status"] == "updated_without_shared_g_evidence"
    assert contract["same_g_across_node_roles"] is False
    assert contract["reduce_g_is_derived"] is False
