"""Smoke-run the package-owned ``examples/methods`` walkthroughs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples" / "methods"

RUNNABLE_EXAMPLES = [
    "run_hll_sketch.py",
    "run_fno_markov.py",
    "run_neural_operator_markov_compare.py",
    "run_manifesto_end_to_end.py",
    "run_manifesto_replications.py",
    "run_manifesto_reward_mechanisms.py",
    "run_finetune_views.py",
    "run_manifesto_finetune_views.py",
    "run_preference_optimizer_views.py",
    "run_local_law_certificate.py",
]

def _run_example(name: str, tmp_path: Path, *extra: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name), "--output-dir", str(tmp_path), *extra],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"{name} exited {proc.returncode}\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    return proc.stdout


def _write_manifesto_config(path: Path, *, preference_mode: str) -> None:
    path.write_text(
        "\n".join(
            [
                'family = "dspy"',
                'model = "local-dspy-program"',
                'max_iterations = 2',
                'use_oracle_predictor = true',
                'sample_size = 1',
                'sample_seed = 0',
                'prompt_template = ""',
                f'preference_mode = "{preference_mode}"',
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("name", RUNNABLE_EXAMPLES)
def test_cpu_example_runs_with_default_config(name: str, tmp_path: Path) -> None:
    stdout = _run_example(name, tmp_path)
    if "status=" in stdout:
        assert "status=success" in stdout, stdout




def test_finetune_views_example_exports_all_views(tmp_path: Path) -> None:
    stdout = _run_example("run_finetune_views.py", tmp_path)
    assert "embedding_pairs=" in stdout
    payload = json.loads((tmp_path / "finetune_views_result.json").read_text(encoding="utf-8"))
    counts = payload["counts"]
    assert counts["embedding_pairs"] > 0
    assert counts["embedding_triplets"] > 0
    assert counts["embedding_ranked"] > 0
    assert counts["sft"] == counts["embedding_pairs"]
    assert Path(payload["artifacts"]["files"]["hf_dataset"]).exists()
    adapters = payload["adapters"]
    assert Path(adapters["embedding"]["files"]["embedding_pairs"]).exists()
    assert Path(adapters["trl_dpo"]["files"]["dpo"]).exists()
    assert Path(adapters["dspy_examples"]["files"]["sft"]).exists()


def test_manifesto_finetune_views_example_exports_root_and_qsentence_views(tmp_path: Path) -> None:
    stdout = _run_example("run_manifesto_finetune_views.py", tmp_path)
    assert "f_sft=" in stdout
    payload = json.loads((tmp_path / "manifesto_finetune_views_result.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["f_sft_rows"] == 3
    assert summary["g_sft_rows"] > summary["f_sft_rows"]
    assert summary["qsentence_triplets"] > 0
    assert summary["qsentence_ranked_groups"] > 0
    assert Path(payload["tree_records"]).exists()
    assert Path(payload["artifacts"]["files"]["dpo"]).exists()
    adapters = payload["adapters"]
    assert Path(adapters["trl_sft"]["files"]["sft"]).exists()
    assert Path(adapters["trl_grpo"]["files"]["grpo"]).exists()
    assert Path(adapters["dspy_examples"]["files"]["dpo"]).exists()


def test_methods_surface_keeps_fit_shortcut_small() -> None:
    from treepo import fit
    from treepo.methods.families import list_families

    assert callable(fit)
    assert "classical_sketch" in list_families()


def test_hll_example_uses_unified_fit_family(tmp_path: Path) -> None:
    stdout = _run_example("run_hll_sketch.py", tmp_path)
    assert "family=classical_sketch" in stdout
    payload = json.loads((tmp_path / "hll_sketch_result.json").read_text(encoding="utf-8"))
    result = payload["result"]
    assert result["summary"]["family"] == "classical_sketch"
    assert result["artifacts"]["f"]["kind"] == "treepo_classical_sketch_f"
    assert result["artifacts"]["g"]["kind"] == "treepo_classical_sketch_g"
    assert payload["statistic"]["info"]["exact"] is True
    assert payload["statistic"]["local_law_summary"]["objective"] == 0.0
    evidence = result["artifacts"]["evidence"]
    assert evidence["statistic"]["present"] is True
    assert evidence["local_laws"]["present"] is True


def test_manifesto_example_uses_unified_train_data(tmp_path: Path) -> None:
    stdout = _run_example("run_manifesto_replications.py", tmp_path)
    assert "train=3" in stdout
    payload = json.loads((tmp_path / "manifesto_replications_result.json").read_text(encoding="utf-8"))
    result = payload["result"]
    assert result["artifacts"]["f"]["n_train"] == 3
    assert result["artifacts"]["g"]["n_train"] == 3
    metadata = payload["replications"][0]["metadata"]
    removed_key = "training_" + "examples"
    assert removed_key not in metadata


@pytest.mark.parametrize("preference_mode", ["scores", "pairwise", "ranked"])
def test_manifesto_example_exports_preference_modes(tmp_path: Path, preference_mode: str) -> None:
    config = tmp_path / "manifesto.toml"
    _write_manifesto_config(config, preference_mode=preference_mode)
    stdout = _run_example("run_manifesto_replications.py", tmp_path, "--config", str(config))
    assert f"preferences={preference_mode}" in stdout
    payload = json.loads((tmp_path / "manifesto_replications_result.json").read_text(encoding="utf-8"))
    preferences = payload["preferences"]
    assert preferences["mode"] == preference_mode
    assert preferences["counts"]["dataset"] == 6
    expected_pairs = 12 if preference_mode == "ranked" else 6
    assert preferences["counts"]["dpo"] == expected_pairs
    assert preferences["counts"]["reward"] == expected_pairs
    assert preferences["counts"]["grpo"] == 6
    assert Path(preferences["files"]["dataset"]).exists()
    assert Path(preferences["files"]["hf_dataset"]).exists()
    assert payload["result"]["artifacts"]["preference_data"]["counts"]["dpo"] == expected_pairs


def test_manifesto_replication_example_can_scope_preferences(tmp_path: Path) -> None:
    config = tmp_path / "manifesto_roots.toml"
    config.write_text(
        "\n".join(
            [
                'family = "dspy"',
                'model = "local-dspy-program"',
                'max_iterations = 1',
                'use_oracle_predictor = true',
                'sample_size = 1',
                'sample_seed = 0',
                'prompt_template = ""',
                'preference_mode = "pairwise"',
                'preference_scope = "roots"',
            ]
        ),
        encoding="utf-8",
    )
    stdout = _run_example("run_manifesto_replications.py", tmp_path, "--config", str(config))
    assert "scope=roots" in stdout

    payload = json.loads((tmp_path / "manifesto_replications_result.json").read_text(encoding="utf-8"))
    preferences = payload["preferences"]
    assert preferences["scope"] == "roots"
    assert preferences["counts"]["dataset"] == 3
    assert preferences["counts"]["dpo"] == 3
    dataset = json.loads(Path(preferences["files"]["dataset"]).read_text(encoding="utf-8"))
    assert {row["unit_type"] for row in dataset["units"]} == {"root"}
    assert {row["target"] for row in dataset["units"]} == {"f"}


def test_manifesto_example_runs_root_and_document_unit_supervision_grid(tmp_path: Path) -> None:
    config = tmp_path / "manifesto_grid.toml"
    config.write_text(
        "\n".join(
            [
                'family = "dspy"',
                'model = "local-dspy-program"',
                'max_iterations = 1',
                'use_oracle_predictor = true',
                'sample_size = 1',
                'sample_seed = 0',
                'prompt_template = ""',
                'preference_mode = "none"',
                'doc_unit_kind = "qsentence"',
                'leaf_unit_counts = [1, 2]',
                'supervision_grid = ["none", "scores"]',
            ]
        ),
        encoding="utf-8",
    )
    stdout = _run_example("run_manifesto_replications.py", tmp_path, "--config", str(config))
    assert "grid_cells=3" in stdout
    payload = json.loads((tmp_path / "manifesto_replications_result.json").read_text(encoding="utf-8"))
    rows = payload["grid"]
    assert len(rows) == 3
    assert any(row["leaf_unit_count"] == 1 and row["preference_mode"] == "none" and row["supervision_unit"] == "root" and row["status"] == "success" for row in rows)
    assert any(row["leaf_unit_count"] == 2 and row["preference_mode"] == "none" and row["supervision_unit"] == "root" and row["status"] == "success" for row in rows)
    assert any(row["leaf_unit_count"] == 1 and row["preference_mode"] == "scores" and row["supervision_unit"] == "qsentence" and row["doc_unit_kind"] == "qsentence" and row["status"] == "success" for row in rows)
    assert not any(row["status"] == "skipped" for row in rows)


def test_manifesto_example_exports_doc_and_qsentence_sampling(tmp_path: Path) -> None:
    config = tmp_path / "manifesto_sampling.toml"
    config.write_text(
        "\n".join(
            [
                'family = "dspy"',
                'model = "local-dspy-program"',
                'max_iterations = 1',
                'use_oracle_predictor = true',
                'doc_sample_size = 2',
                'doc_sample_seed = 7',
                'qsentence_sample_size = 1',
                'qsentence_sample_seed = 5',
                'prompt_template = ""',
                'preference_mode = "scores"',
                'doc_unit_kind = "qsentence"',
                'leaf_unit_count = 1',
            ]
        ),
        encoding="utf-8",
    )
    stdout = _run_example("run_manifesto_replications.py", tmp_path, "--config", str(config))
    assert "train=2" in stdout
    assert "doc_sample=n=2" in stdout
    assert "qsentence_sample=n=1" in stdout
    payload = json.loads((tmp_path / "manifesto_replications_result.json").read_text(encoding="utf-8"))
    sampling = payload["sampling"]
    assert sampling["summary"]["documents"]["population_count"] == 3
    assert sampling["summary"]["documents"]["observed_count"] == 2
    assert sampling["summary"]["documents"]["propensities"] == [2 / 3]
    assert sampling["summary"]["qsentences"]["population_count"] == 4
    assert sampling["summary"]["qsentences"]["observed_count"] == 2
    assert sampling["summary"]["qsentences"]["propensities"] == [1 / 3]
    for path in sampling["files"].values():
        assert Path(path).exists()

    preferences = payload["preferences"]
    assert preferences["counts"]["dataset"] == 4
    dataset = json.loads(Path(preferences["files"]["dataset"]).read_text(encoding="utf-8"))
    root_units = [row for row in dataset["units"] if row["unit_type"] == "root"]
    qsentence_units = [row for row in dataset["units"] if row["unit_type"] == "qsentence"]
    assert {row["propensity"] for row in root_units} == {2 / 3}
    assert {row["propensity"] for row in qsentence_units} == {1 / 3}
    assert all(row["metadata"]["document_propensity"] == 2 / 3 for row in qsentence_units)
    assert all(row["metadata"]["unit_propensity"] == 0.5 for row in qsentence_units)
    assert all(row["metadata"]["joint_propensity"] == 1 / 3 for row in qsentence_units)


def test_manifesto_example_full_fixture_uses_all_docs_and_qsentences(tmp_path: Path) -> None:
    config = tmp_path / "manifesto_full_fixture.toml"
    config.write_text(
        "\n".join(
            [
                'family = "dspy"',
                'model = "local-dspy-program"',
                'max_iterations = 1',
                'use_oracle_predictor = true',
                'prompt_template = ""',
                'preference_mode = "scores"',
                'doc_unit_kind = "qsentence"',
                'leaf_unit_count = 1',
            ]
        ),
        encoding="utf-8",
    )
    stdout = _run_example("run_manifesto_replications.py", tmp_path, "--config", str(config))
    assert "train=3" in stdout
    assert "doc_sample=all" in stdout
    assert "qsentence_sample=all" in stdout
    payload = json.loads((tmp_path / "manifesto_replications_result.json").read_text(encoding="utf-8"))
    sampling = payload["sampling"]
    assert sampling["summary"]["documents"] == {
        "population_count": 3,
        "observed_count": 3,
        "propensities": [1.0],
    }
    assert sampling["summary"]["qsentences"] == {
        "population_count": 6,
        "observed_count": 6,
        "propensities": [1.0],
    }

    preferences = payload["preferences"]
    assert preferences["counts"]["dataset"] == 9
    assert preferences["counts"]["units"] == 9
    assert preferences["counts"]["candidates"] == 18
    dataset = json.loads(Path(preferences["files"]["dataset"]).read_text(encoding="utf-8"))
    root_units = [row for row in dataset["units"] if row["unit_type"] == "root"]
    qsentence_units = [row for row in dataset["units"] if row["unit_type"] == "qsentence"]
    assert len(root_units) == 3
    assert len(qsentence_units) == 6
    assert {row["propensity"] for row in root_units} == {1.0}
    assert {row["propensity"] for row in qsentence_units} == {1.0}
    assert all(row["metadata"]["document_propensity"] == 1.0 for row in qsentence_units)
    assert all(row["metadata"]["unit_propensity"] == 1.0 for row in qsentence_units)
    assert all(row["metadata"]["joint_propensity"] == 1.0 for row in qsentence_units)


def test_manifesto_end_to_end_example_wires_fit_evidence_and_reward_views(tmp_path: Path) -> None:
    stdout = _run_example("run_manifesto_end_to_end.py", tmp_path)
    assert "fit_preferences=scores/both" in stdout
    assert "reward_cells=6" in stdout

    payload = json.loads((tmp_path / "manifesto_end_to_end_result.json").read_text(encoding="utf-8"))
    assert payload["sampling"]["summary"]["documents"] == {
        "population_count": 3,
        "observed_count": 3,
        "propensities": [1.0],
    }
    assert payload["sampling"]["summary"]["qsentences"] == {
        "population_count": 6,
        "observed_count": 6,
        "propensities": [1.0],
    }
    for path in payload["sampling"]["files"].values():
        assert Path(path).exists()

    result = payload["result"]
    assert result["status"] == "success"
    assert result["artifacts"]["f"]["n_train"] == 3
    assert result["artifacts"]["g"]["n_train"] == 6
    evidence = payload["evidence"]
    assert evidence["root"]["present"] is True
    assert evidence["preferences"]["present"] is True
    assert evidence["predictions"]["present"] is True
    assert Path(payload["files"]["evidence"]).exists()
    assert payload["fit_preferences"]["counts"]["dataset"] == 9
    assert payload["fit_preferences"]["counts"]["supervised"] == 9
    assert payload["fit_optimizer_views"]["counts"] == {"supervised": 9, "dpo": 9, "reward": 9, "grpo": 9}
    assert payload["fit_optimizer_views"]["first_dpo"]["chosen"]
    assert payload["fit_optimizer_views"]["first_grpo"]["responses"]
    bundle = payload["artifact_bundle"]
    assert bundle["trees"]["n_trees"] == 3
    assert bundle["trees"]["n_nodes"] == 9
    assert bundle["trees"]["n_leaves"] == 6
    assert bundle["preferences"]["counts"]["dataset"] == 9
    assert Path(bundle["files"]["tree_records"]).exists()
    assert Path(bundle["files"]["manifest"]).exists()

    cells = {(row["scope"], row["mode"]): row for row in payload["reward_views"]}
    assert set(cells) == {
        ("roots", "pairwise"),
        ("roots", "ranked"),
        ("qsentences", "pairwise"),
        ("qsentences", "ranked"),
        ("both", "pairwise"),
        ("both", "ranked"),
    }
    assert cells[("roots", "pairwise")]["counts"]["dpo"] == 3
    assert cells[("roots", "pairwise")]["optimizer_views"]["counts"]["dpo"] == 3
    assert cells[("qsentences", "pairwise")]["counts"]["dpo"] == 6
    assert cells[("qsentences", "pairwise")]["optimizer_views"]["counts"]["reward"] == 6
    assert cells[("both", "ranked")]["counts"]["dpo"] == 18
    assert cells[("both", "ranked")]["counts"]["grpo"] == 9
    assert cells[("both", "ranked")]["optimizer_views"]["counts"] == {"supervised": 9, "dpo": 18, "reward": 18, "grpo": 9}
    assert cells[("both", "ranked")]["optimizer_views"]["first_grpo"]["responses"]
    for row in payload["reward_views"]:
        assert Path(row["files"]["dpo"]).exists()
        assert Path(row["files"]["reward"]).exists()
        assert Path(row["files"]["grpo"]).exists()


def test_manifesto_reward_mechanism_example_exports_scoped_trainer_views(tmp_path: Path) -> None:
    stdout = _run_example("run_manifesto_reward_mechanisms.py", tmp_path)
    assert "cells=6" in stdout
    payload = json.loads((tmp_path / "manifesto_reward_mechanisms_result.json").read_text(encoding="utf-8"))
    assert payload["sampling"]["summary"]["documents"] == {
        "population_count": 3,
        "observed_count": 3,
        "propensities": [1.0],
    }
    assert payload["sampling"]["summary"]["qsentences"] == {
        "population_count": 6,
        "observed_count": 6,
        "propensities": [1.0],
    }
    cells = {(row["scope"], row["mode"]): row for row in payload["cells"]}
    assert set(cells) == {
        ("roots", "pairwise"),
        ("roots", "ranked"),
        ("qsentences", "pairwise"),
        ("qsentences", "ranked"),
        ("both", "pairwise"),
        ("both", "ranked"),
    }
    assert cells[("roots", "pairwise")]["counts"] == {
        "candidates": 6,
        "dataset": 3,
        "dpo": 3,
        "general": 3,
        "grpo": 3,
        "reward": 3,
        "supervised": 3,
        "units": 3,
    }
    assert cells[("qsentences", "pairwise")]["counts"]["dataset"] == 6
    assert cells[("qsentences", "pairwise")]["counts"]["dpo"] == 6
    assert cells[("both", "ranked")]["counts"]["dataset"] == 9
    assert cells[("both", "ranked")]["counts"]["dpo"] == 18
    assert cells[("both", "ranked")]["counts"]["grpo"] == 9
    assert cells[("both", "ranked")]["optimizer_views"]["counts"] == {"supervised": 9, "dpo": 18, "reward": 18, "grpo": 9}
    assert cells[("roots", "pairwise")]["optimizer_views"]["first_dpo"]["chosen"]

    for row in payload["cells"]:
        files = row["files"]
        assert Path(files["dataset"]).exists()
        assert Path(files["dpo"]).exists()
        assert Path(files["reward"]).exists()
        assert Path(files["grpo"]).exists()

    root_dataset = json.loads(Path(cells[("roots", "pairwise")]["files"]["dataset"]).read_text(encoding="utf-8"))
    q_dataset = json.loads(Path(cells[("qsentences", "pairwise")]["files"]["dataset"]).read_text(encoding="utf-8"))
    assert {row["unit_type"] for row in root_dataset["units"]} == {"root"}
    assert {row["target"] for row in root_dataset["units"]} == {"f"}
    assert {row["unit_type"] for row in q_dataset["units"]} == {"qsentence"}
    assert {row["target"] for row in q_dataset["units"]} == {"g"}


def test_markov_example_uses_preference_data_for_f_and_g(tmp_path: Path) -> None:
    stdout = _run_example("run_fno_markov.py", tmp_path)
    assert "train=8" in stdout
    assert "preferences=scores" in stdout
    payload = json.loads((tmp_path / "fno_markov_result.json").read_text(encoding="utf-8"))
    result = payload["result"]
    assert result["artifacts"]["f"]["n_train"] == 8
    assert result["artifacts"]["g"]["n_train"] == 8
    assert payload["preferences"]["n_units"] == 16
    assert payload["preferences"]["n_candidates"] == 32
    assert result["artifacts"]["preference_data"]["counts"]["dpo"] == 16
    assert payload["statistic"]["info"]["state_kind"] == "fno"
    evidence = result["artifacts"]["evidence"]
    assert evidence["preferences"]["present"] is True
    assert evidence["statistic"]["present"] is True


def test_preference_optimizer_views_example_exports_trl_and_dspy_bones(tmp_path: Path) -> None:
    stdout = _run_example("run_preference_optimizer_views.py", tmp_path)
    assert "family=dspy" in stdout
    assert "dpo=3" in stdout
    assert "reward=3" in stdout
    assert "grpo=2" in stdout

    payload = json.loads((tmp_path / "preference_optimizer_views_result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    views = payload["optimizer_views"]
    assert views["counts"] == {"supervised": 6, "dpo": 3, "reward": 3, "grpo": 2}
    assert views["first_dpo"]["chosen"]
    assert views["first_reward"]["chosen_score"] > views["first_reward"]["rejected_score"]
    assert len(views["first_grpo"]["responses"]) == 2

    fit_result = payload["fit"]
    assert fit_result["artifacts"]["f"]["kind"] == "treepo_dspy_f"
    assert fit_result["artifacts"]["g"]["kind"] == "treepo_dspy_g"
    assert "f examples:" in fit_result["artifacts"]["f"]["config"]["prompt_template"]
    assert "root:doc_a:root" in fit_result["artifacts"]["f"]["supervised_examples"]
    assert "leaf:doc_a:leaf_a0" in fit_result["artifacts"]["g"]["supervised_examples"]
    assert payload["preference_artifacts"]["counts"]["dpo"] == 3
    assert Path(payload["preference_artifacts"]["files"]["hf_dataset"]).exists()
    assert Path(payload["bundle"]["files"]["tree_records"]).exists()
    assert Path(payload["bundle"]["files"]["manifest"]).exists()


def test_local_law_certificate_example_exports_unified_evidence(tmp_path: Path) -> None:
    stdout = _run_example("run_local_law_certificate.py", tmp_path)
    assert "family=example" in stdout
    payload = json.loads((tmp_path / "local_law_certificate_result.json").read_text(encoding="utf-8"))
    evidence = payload["evidence"]
    assert evidence["root"]["present"] is True
    assert evidence["preferences"]["present"] is True
    assert evidence["statistic"]["present"] is True
    assert evidence["local_laws"]["present"] is True
    assert evidence["local_laws"]["source"] == "sampled_rows"
    assert set(evidence["local_laws"]["by_law_kind"]) == {
        "leaf_preservation",
        "merge_preservation",
        "on_range_idempotence",
    }
    assert payload["audit"]["local_law_objective"]["row_count"] == 6
    assert payload["audit"]["local_law_objective"]["observed_count"] == 4
    certificate = payload["certificate"]
    assert certificate["local_law_radius"] == payload["audit"]["local_law_objective"]["objective"]
    assert certificate["estimation_radius"] == 0.05
    for path in payload["files"].values():
        assert Path(path).exists()


def test_lda_example_runs_with_tiny_vector_fg_config(tmp_path: Path) -> None:
    config = tmp_path / "lda.toml"
    config.write_text(
        "\n".join(
            [
                'operator_kinds = ["fno", "conv1d"]',
                'n_train = 12',
                'n_eval = 6',
                'n_topics = 4',
                'doc_tokens = 64',
                'doc_unit_kind = "token"',
                'leaf_unit_count = 16',
                'vocabulary_size = 40',
                'doc_topic_concentration = 0.7',
                'topic_word_concentration = 0.05',
                'target_topic = 0',
                'topic_seed = 0',
                'seed = 0',
                'max_iterations = 3',
                'embedding_dim = 8',
                'hidden_channels = 4',
                'n_modes = 2',
                'n_layers = 1',
                'head_hidden_dim = 8',
                'epochs_per_iteration = 1',
                'batch_size = 4',
                'learning_rate = 0.01',
                'device = "cpu"',
                'sklearn_max_iter = 20',
            ]
        ),
        encoding="utf-8",
    )
    stdout = _run_example("run_neural_operator_lda.py", tmp_path, "--config", str(config))
    assert "status=success" in stdout
    result = tmp_path / "neural_operator_lda_result.json"
    assert result.exists()



def test_lda_leaf_grid_example_runs_with_tiny_config(tmp_path: Path) -> None:
    config = tmp_path / "lda_leaf_grid.toml"
    config.write_text(
        "\n".join(
            [
                'operator_kinds = ["fno", "conv1d"]',
                'doc_unit_kind = "token"',
                'leaf_unit_counts = [12, 24]',
                'n_train = 8',
                'n_eval = 5',
                'n_topics = 4',
                'doc_tokens = 48',
                'vocabulary_size = 30',
                'doc_topic_concentration = 0.7',
                'topic_word_concentration = 0.05',
                'target_topic = 0',
                'topic_seed = 0',
                'seed = 0',
                'max_iterations = 3',
                'embedding_dim = 8',
                'hidden_channels = 4',
                'n_modes = 2',
                'n_layers = 1',
                'conv_kernel_size = 3',
                'head_hidden_dim = 8',
                'epochs_per_iteration = 1',
                'batch_size = 4',
                'learning_rate = 0.01',
                'device = "cpu"',
                'sklearn_max_iter = 20',
            ]
        ),
        encoding="utf-8",
    )
    stdout = _run_example("run_neural_operator_lda_leaf_grid.py", tmp_path, "--config", str(config))
    assert "status=success" in stdout
    assert "rows=4" in stdout
    assert (tmp_path / "neural_operator_lda_leaf_grid.json").exists()
    assert (tmp_path / "neural_operator_lda_leaf_grid.csv").exists()


def test_markov_leaf_grid_example_runs_with_tiny_config(tmp_path: Path) -> None:
    config = tmp_path / "markov_leaf_grid.toml"
    config.write_text(
        "\n".join(
            [
                'operator_kinds = ["fno", "conv1d"]',
                'doc_unit_kind = "token"',
                'leaf_unit_counts = [12, 24]',
                'n_train = 8',
                'n_eval = 5',
                'n_states = 3',
                'doc_tokens = 48',
                'transition_prob = 0.2',
                'vocabulary_size = 96',
                'seed = 0',
                'max_iterations = 3',
                'embedding_dim = 8',
                'hidden_channels = 4',
                'n_modes = 2',
                'n_layers = 1',
                'conv_kernel_size = 3',
                'head_hidden_dim = 8',
                'epochs_per_iteration = 1',
                'batch_size = 4',
                'learning_rate = 0.01',
                'device = "cpu"',
                'normalize_targets = true',
                'numeric_transition_state_weight = 0.05',
            ]
        ),
        encoding="utf-8",
    )
    stdout = _run_example("run_neural_operator_markov_leaf_grid.py", tmp_path, "--config", str(config))
    assert "status=success" in stdout
    assert "rows=4" in stdout
    assert (tmp_path / "neural_operator_markov_leaf_grid.json").exists()
    assert (tmp_path / "neural_operator_markov_leaf_grid.csv").exists()
