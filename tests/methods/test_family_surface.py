from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import treepo.methods.families as family_registry
from treepo import (
    Candidate,
    ComposableStatistic,
    PreferenceDataset,
    PreferenceRecord,
    TaskState,
    family_statistic,
    fit,
)
from treepo.methods.contracts import CTreePOLearningSpec
from treepo.methods.families import list_families, resolve_family
from treepo.methods.fixtures import make_markov_changepoint_trees
from treepo.tasks.manifesto import (
    MANIFESTO_POLICY_STATE_KIND,
    ManifestoPolicyStatistic,
    make_manifesto_preferences,
    make_manifesto_replication_trees,
    manifesto_document_unit_sampling_rows,
    manifesto_oracle_predict_fn,
    manifesto_prompt_template,
    sample_manifesto_replication_trees,
)


def _tiny_config(tmp_path: Path) -> dict[str, object]:
    return {
        "output_dir": str(tmp_path),
        "embedding_dim": 8,
        "hidden_channels": 4,
        "n_modes": 2,
        "n_layers": 1,
        "head_hidden_dim": 8,
        "epochs_per_iteration": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 7,
    }


def test_family_registry_exposes_builtin_routes() -> None:
    names = set(list_families())
    assert {"neural_operator", "fno", "llm", "dspy"} <= names

    family = resolve_family("llm", {"model": "teacher", "default_prediction": 3.5})
    assert family.name == "llm"
    assert family.config.model == "teacher"
    assert family.score_roots_with_f(f=None, g=None, trees=[SimpleNamespace(metadata={"text": "doc"})]) == [3.5]

    dspy = resolve_family("dspy", {"model": "teacher", "default_prediction": 2.25})
    assert dspy.name == "dspy"
    assert dspy.score_roots_with_f(f=None, g=None, trees=[SimpleNamespace(metadata={"text": "doc"})]) == [2.25]


def test_downstream_can_register_application_family(monkeypatch) -> None:
    monkeypatch.setattr(family_registry, "_REGISTRY", dict(family_registry._REGISTRY))

    def factory(config):
        return SimpleNamespace(name="custom_llm", config=dict(config))

    family_registry.register_family("custom_llm", factory)
    runtime = family_registry.resolve_family("custom_llm", {"model": "served-model"})

    assert runtime.name == "custom_llm"
    assert runtime.config == {"model": "served-model"}
    assert "custom_llm" in family_registry.list_families()


def test_builtin_non_state_families_expose_empty_statistic_hook() -> None:
    assert family_statistic(resolve_family("llm", {"default_prediction": 1.0})) is None
    assert family_statistic(resolve_family("dspy", {"default_prediction": 1.0})) is None
    assert family_statistic(resolve_family("oracle", {"oracle_name": "hll_exact"})) is None
    assert family_statistic(resolve_family("learnable_constant", {})) is None


def test_fit_uses_family_without_selector_indirection(tmp_path: Path) -> None:
    train = make_markov_changepoint_trees(
        n_trees=6,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=31,
        split="train",
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=32,
        split="test",
    )

    result = fit(
        {
            "family": "fno",
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": _tiny_config(tmp_path),
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.summary["family"] == "fno"
    assert result.artifacts["g"]["kind"] == "treepo_fno_g"
    manifest = json.loads((tmp_path / "treepo_methods_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["spec"]["family"] == "fno"


def test_spec_mapping_keeps_family_only_fit_surface() -> None:
    spec = CTreePOLearningSpec.from_mapping(
        {
            "space_kind": "fit",
            "family": "neural_operator",
            "schedule": "fg",
            "backend_config": {"operator_kind": "conv1d"},
            "axis": {"max_iterations": 0},
        }
    )
    assert spec.family == "neural_operator"
    assert spec.to_dict()["family"] == "neural_operator"
    assert "selector" not in spec.to_dict()


def test_fit_can_use_llm_family_with_injected_predict_fn(tmp_path: Path) -> None:
    trees = [
        SimpleNamespace(metadata={"split": "test", "text": "alpha", "teacher_score_native": 2.0}),
        SimpleNamespace(metadata={"split": "test", "text": "beta", "teacher_score_native": 4.0}),
    ]

    def predict_fn(*, tree, **kwargs):
        del kwargs
        return {"score": getattr(tree, "metadata", {})["teacher_score_native"]}

    result = fit(
        {
            "family": "llm",
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {
                "output_dir": str(tmp_path),
                "predict_fn": predict_fn,
                "prompt_template": "Score this: {text}",
            },
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.summary["family"] == "llm"
    assert result.artifacts["g"]["kind"] == "treepo_llm_g"
    assert result.metrics["internal_f_mae"] == 0.0


def test_manifesto_trees_are_plain_and_preferences_supply_root_and_node_views() -> None:
    trees = make_manifesto_replication_trees(split="test")
    assert len(trees) >= 3
    first = trees[0]
    assert first.metadata["root_label_name"] == "rile"
    assert isinstance(first.metadata["teacher_score_native"], float)
    removed_key = "training_" + "examples"
    assert removed_key not in first.metadata

    for mode in ("scores", "pairwise", "ranked"):
        preferences = make_manifesto_preferences(trees, mode=mode, sample_size=1, seed=0)
        assert preferences.summary()["unit_types"] == ["qsentence", "root"]
        assert preferences.summary()["targets"] == ["f", "g"]
        assert len(preferences) == len(trees) * 2
        pairwise_rows = len(trees) * (4 if mode == "ranked" else 2)
        assert len(preferences.to_records("dpo")) == pairwise_rows
        assert len(preferences.to_records("reward")) == pairwise_rows
        assert len(preferences.to_records("supervised")) == len(trees) * 2
        q_records = [row for row in preferences.to_records("supervised") if row["unit_type"] == "qsentence"]
        assert q_records
        assert q_records[0]["value"]["kind"] == MANIFESTO_POLICY_STATE_KIND
        assert q_records[0]["value"]["items"][0]["code"]
        grpo_rows = preferences.to_records("grpo")
        assert len(grpo_rows) == len(trees) * 2
        if mode == "ranked":
            assert max(len(row["responses"]) for row in grpo_rows) == 3


def test_manifesto_sampling_logs_document_and_qsentence_propensities() -> None:
    trees = make_manifesto_replication_trees(split="train")
    sampled, document_rows = sample_manifesto_replication_trees(trees, sample_size=2, seed=3)
    assert len(sampled) == 2
    assert len(document_rows) == 3
    assert sum(1 for row in document_rows if row["observed"]) == 2
    assert {row["inclusion_probability"] for row in document_rows} == {2 / 3}
    assert all(tree.metadata["document_propensity"] == 2 / 3 for tree in sampled)

    qsentence_rows = manifesto_document_unit_sampling_rows(sampled, sample_size=1, seed=5)
    assert len(qsentence_rows) == 4
    assert sum(1 for row in qsentence_rows if row["observed"]) == 2
    assert {row["document_propensity"] for row in qsentence_rows} == {2 / 3}
    assert {row["unit_propensity"] for row in qsentence_rows} == {0.5}
    assert {row["joint_propensity"] for row in qsentence_rows} == {1 / 3}

    preferences = make_manifesto_preferences(sampled, mode="scores", sample_size=1, seed=5)
    unit_rows = {row["unit_id"]: row for row in preferences.units}
    root_rows = [row for row in unit_rows.values() if row["unit_type"] == "root"]
    qsentence_units = [row for row in unit_rows.values() if row["unit_type"] == "qsentence"]
    assert {row["propensity"] for row in root_rows} == {2 / 3}
    assert {row["propensity"] for row in qsentence_units} == {1 / 3}
    assert all(row["metadata"]["document_propensity"] == 2 / 3 for row in qsentence_units)
    assert all(row["metadata"]["unit_propensity"] == 0.5 for row in qsentence_units)
    assert all(row["metadata"]["joint_propensity"] == 1 / 3 for row in qsentence_units)


def test_manifesto_preferences_can_scope_reward_units() -> None:
    trees = make_manifesto_replication_trees(split="train")
    root_preferences = make_manifesto_preferences(trees, mode="pairwise", scope="roots")
    qsentence_preferences = make_manifesto_preferences(trees, mode="pairwise", scope="qsentences")
    combined_preferences = make_manifesto_preferences(trees, mode="ranked", scope="both")

    assert len(root_preferences) == 3
    assert root_preferences.summary()["unit_types"] == ["root"]
    assert root_preferences.summary()["targets"] == ["f"]
    assert len(root_preferences.to_records("dpo")) == 3
    assert len(root_preferences.to_records("grpo")) == 3

    assert len(qsentence_preferences) == 6
    assert qsentence_preferences.summary()["unit_types"] == ["qsentence"]
    assert qsentence_preferences.summary()["targets"] == ["g"]
    assert len(qsentence_preferences.to_records("dpo")) == 6
    assert len(qsentence_preferences.to_records("grpo")) == 6

    assert len(combined_preferences) == 9
    assert combined_preferences.summary()["unit_types"] == ["qsentence", "root"]
    assert combined_preferences.summary()["targets"] == ["f", "g"]
    assert len(combined_preferences.to_records("dpo")) == 18
    assert len(combined_preferences.to_records("reward")) == 18
    assert len(combined_preferences.to_records("grpo")) == 9


def test_manifesto_root_only_can_group_document_units_into_leaves() -> None:
    trees = make_manifesto_replication_trees(split="test", leaf_unit_count=2)
    assert trees[0].metadata["leaf_unit_count"] == 2
    assert len(trees[0].leaves) == 1
    assert trees[0].leaves[0].metadata["grouped_leaf"] is True


def test_preference_dataset_supplies_target_specific_supervision() -> None:
    dataset = PreferenceDataset(
        [
            PreferenceRecord(
                record_id="f1",
                unit_id="root",
                unit_type="root",
                target="f",
                context="root calibration",
                candidates=(Candidate(id="gold", value=1.0, score=1.0),),
            ),
            PreferenceRecord(
                record_id="g1",
                unit_id="leaf",
                unit_type="leaf",
                target="g",
                context="leaf evidence",
                candidates=(Candidate(id="gold", value=2.0, score=1.0),),
            ),
            PreferenceRecord(
                record_id="both1",
                unit_id="shared",
                unit_type="unit",
                target="both",
                context="shared evidence",
                candidates=(Candidate(id="gold", value=3.0, score=1.0),),
            ),
        ]
    )

    assert len(dataset.filter_target("f")) == 2
    assert len(dataset.filter_target("g")) == 2
    assert [row["value"] for row in dataset.filter_target("f").to_records("supervised")] == [1.0, 3.0]
    assert [row["value"] for row in dataset.filter_target("g").to_records("supervised")] == [2.0, 3.0]


def test_dspy_prompt_uses_preference_supervision(tmp_path: Path) -> None:
    trees = make_manifesto_replication_trees(split="test")
    preferences = make_manifesto_preferences(trees, mode="scores", sample_size=1, seed=0)
    prompts: list[str] = []

    def predict_fn(*, prompt: str, tree, **kwargs):
        del kwargs
        prompts.append(prompt)
        return manifesto_oracle_predict_fn(tree=tree)

    result = fit(
        {
            "family": "dspy",
            "train_data": trees,
            "preference_data": preferences,
            "eval_data": trees,
            "backend_config": {
                "output_dir": str(tmp_path),
                "predict_fn": predict_fn,
                "prompt_template": manifesto_prompt_template(),
                "min_score": -100.0,
                "max_score": 100.0,
            },
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )
    assert result.status == "success"
    assert result.summary["family"] == "dspy"
    assert result.artifacts["g"]["kind"] == "treepo_dspy_g"
    assert result.metrics["internal_f_mae"] == 0.0
    assert any("Supervised examples:" in prompt for prompt in prompts)
    assert any("qsentence" in prompt and "target=" in prompt and "manifesto_policy" in prompt for prompt in prompts)


def test_manifesto_policy_statistic_matches_fixture_root_labels() -> None:
    trees = make_manifesto_replication_trees(split="test")
    statistic = ManifestoPolicyStatistic()
    assert isinstance(statistic, ComposableStatistic)
    root_state = statistic.encode_tree(trees[0])
    assert isinstance(root_state, TaskState)
    assert root_state.kind == MANIFESTO_POLICY_STATE_KIND
    assert statistic.readout(root_state) == trees[0].metadata["root_label"]
    rows = statistic.local_law_rows(trees)
    assert rows
    assert {row.metadata["state_kind"] for row in rows} == {MANIFESTO_POLICY_STATE_KIND}
    assert all(row.oracle_loss == 0.0 for row in rows)
