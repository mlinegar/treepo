from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from treepo.methods import run
from treepo.methods.contracts import CTreePOLearningSpec
from treepo.methods.families import resolve_family
from treepo.methods.fixtures import make_markov_changepoint_trees
from treepo.tasks.manifesto import (
    make_manifesto_replication_trees,
    manifesto_oracle_predict_fn,
    manifesto_prompt_template,
    qsentence_guidance_text,
)
from treepo.methods.estimators import (
    EstimatorDescriptor,
    EstimatorSpec,
    list_estimators,
    resolve_estimator,
)
from treepo.methods.g_estimators import resolve_g_estimator


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


def test_estimator_registry_exposes_neural_operator_and_llm_axes() -> None:
    names = set(list_estimators())
    assert {"neural_operator", "general_neural_operator", "fno", "conv1d", "llm", "prompted_llm", "dspy"} <= names

    descriptor = resolve_estimator({"name": "neural_operator", "operator_kind": "tfno"})
    assert isinstance(descriptor, EstimatorDescriptor)
    assert descriptor.family == "neural_operator"
    assert descriptor.category == "neural_operator"
    assert descriptor.supports_neural_operator is True
    assert descriptor.backend_config["operator_kind"] == "tfno"


def test_fno_estimator_specializes_family_and_backend_config() -> None:
    descriptor = resolve_estimator(EstimatorSpec(name="fno", backend_config={"operator_kind": "conv1d"}))
    assert descriptor.family == "fno"
    assert descriptor.backend_config["operator_kind"] == "fno"
    merged = descriptor.apply_to_backend_config({"output_dir": "out"})
    assert merged["operator_kind"] == "fno"
    assert merged["estimator"]["name"] == "fno"
    assert merged["estimator"]["target"] == "g"
    assert merged["g_estimator"]["name"] == "fno"


def test_llm_estimator_is_extension_descriptor_not_heavy_runtime() -> None:
    descriptor = resolve_estimator({"name": "prompted_llm", "family": "llm", "model": "teacher"})
    assert descriptor.family == "llm"
    assert descriptor.category == "llm"
    assert descriptor.supports_llm is True
    assert descriptor.extension_required is False
    assert descriptor.backend_config["model"] == "teacher"


def test_fit_can_use_estimator_without_repeating_family(tmp_path: Path) -> None:
    train = make_markov_changepoint_trees(
        n_trees=6,
        doc_tokens=32,
        leaf_token_count=8,
        vocabulary_size=64,
        seed=31,
        split="train",
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=32,
        leaf_token_count=8,
        vocabulary_size=64,
        seed=32,
        split="test",
    )

    result = run(
        "fit",
        {
            "estimator": {"name": "fno"},
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": _tiny_config(tmp_path),
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.summary["family"] == "fno"
    assert result.summary["estimator"]["name"] == "fno"
    assert result.summary["g_estimator"]["name"] == "fno"
    assert result.artifacts["g"]["kind"] == "treepo_fno_g"
    manifest = json.loads((tmp_path / "treepo_methods_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["estimator"]["name"] == "fno"
    assert manifest["g_estimator"]["name"] == "fno"


def test_spec_mapping_preserves_estimator_descriptor() -> None:
    spec = CTreePOLearningSpec.from_mapping(
        {
            "space_kind": "fit",
            "family": "",
            "schedule": "fg",
            "backend_config": {},
            "axis": {"max_iterations": 0},
            "estimator": {"name": "neural_operator", "operator_kind": "conv1d"},
        }
    )
    assert spec.estimator == {"name": "neural_operator", "operator_kind": "conv1d"}
    assert spec.to_dict()["estimator"]["name"] == "neural_operator"


def test_g_estimator_alias_remains_compatible() -> None:
    descriptor = resolve_g_estimator({"name": "fno"})
    assert descriptor.name == "fno"
    assert descriptor.target == "g"


def test_llm_family_is_registered_provider_neutral_runtime() -> None:
    family = resolve_family("llm", {"model": "teacher", "default_prediction": 3.5})
    assert family.name == "llm"
    assert family.config.model == "teacher"
    assert family.score_roots_with_f(f=None, g=None, trees=[SimpleNamespace(metadata={"text": "doc"})]) == [3.5]


def test_fit_can_use_prompted_llm_estimator_with_injected_predict_fn(tmp_path: Path) -> None:
    trees = [
        SimpleNamespace(metadata={"split": "test", "text": "alpha", "teacher_score_1_7": 2.0}),
        SimpleNamespace(metadata={"split": "test", "text": "beta", "teacher_score_1_7": 4.0}),
    ]

    def predict_fn(*, tree, **kwargs):
        del kwargs
        return {"score": getattr(tree, "metadata", {})["teacher_score_1_7"]}

    result = run(
        "fit",
        {
            "estimator": {"name": "prompted_llm", "model": "teacher"},
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
    assert result.summary["estimator"]["name"] == "prompted_llm"
    assert result.artifacts["g"]["kind"] == "treepo_llm_g"
    assert result.metrics["internal_f_mae"] == 0.0


def test_dspy_family_is_registered_provider_neutral_runtime() -> None:
    descriptor = resolve_estimator({"name": "dspy", "model": "teacher"})
    assert descriptor.family == "dspy"
    assert descriptor.extension_required is False
    family = resolve_family("dspy", {"model": "teacher", "default_prediction": 2.25})
    assert family.name == "dspy"
    assert family.score_roots_with_f(f=None, g=None, trees=[SimpleNamespace(metadata={"text": "doc"})]) == [2.25]


def test_manifesto_replication_trees_carry_root_labels_and_qsentence_guidance() -> None:
    trees = make_manifesto_replication_trees(split="test")
    assert len(trees) >= 3
    first = trees[0]
    assert first.metadata["root_label_name"] == "rile"
    assert isinstance(first.metadata["teacher_score_1_7"], float)
    assert first.metadata["g_guidance_qsentences"]
    assert "guidance_score" in qsentence_guidance_text(first)


def test_fit_can_use_dspy_manifesto_qsentence_guidance(tmp_path: Path) -> None:
    trees = make_manifesto_replication_trees(split="test")
    result = run(
        "fit",
        {
            "estimator": {"name": "dspy", "model": "teacher"},
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {
                "output_dir": str(tmp_path),
                "predict_fn": manifesto_oracle_predict_fn,
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
