"""Smoke-run the package-owned ``examples/methods`` walkthroughs."""

from __future__ import annotations

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
    "run_manifesto_replications.py",
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


@pytest.mark.parametrize("name", RUNNABLE_EXAMPLES)
def test_cpu_example_runs_with_default_config(name: str, tmp_path: Path) -> None:
    stdout = _run_example(name, tmp_path)
    if "status=" in stdout:
        assert "status=success" in stdout, stdout



def test_methods_surface_keeps_fit_shortcut_small() -> None:
    from treepo.methods import allowed_config_keys, list_families

    assert "classical_sketch" in list_families()
    fit_keys = allowed_config_keys("fit")
    assert {"family", "estimator", "g_estimator", "train_data", "eval_data", "backend_config", "axis", "spec"} <= fit_keys
    assert "schedule" not in fit_keys
    assert "space_kind" not in fit_keys
    assert "initial_artifacts" not in fit_keys


def test_hll_example_uses_unified_fit_family(tmp_path: Path) -> None:
    stdout = _run_example("run_hll_sketch.py", tmp_path)
    assert "family=classical_sketch" in stdout
    payload = __import__("json").loads((tmp_path / "hll_sketch_result.json").read_text(encoding="utf-8"))
    result = payload["result"]
    assert result["summary"]["family"] == "classical_sketch"
    assert result["artifacts"]["f"]["kind"] == "treepo_classical_sketch_f"
    assert result["artifacts"]["g"]["kind"] == "treepo_classical_sketch_g"


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
                'leaf_token_count = 16',
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
                'leaf_token_counts = [12, 24]',
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
                'leaf_token_counts = [12, 24]',
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
