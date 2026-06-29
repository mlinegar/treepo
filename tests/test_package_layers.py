from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from treepo.core import (
    BenchmarkRef,
    ExperimentContext,
    MethodRef,
    ROLE_SCORER,
    SamplingPlan,
    role_ref,
    roles_metadata,
)


def test_treepo_import_keeps_heavy_optional_modules_lazy() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
import treepo
heavy = ["dspy", "openai", "pandas", "scipy", "sklearn", "torch", "transformers", "vllm"]
print(json.dumps({name: name in sys.modules for name in heavy}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(proc.stdout) == {
        "dspy": False,
        "openai": False,
        "pandas": False,
        "scipy": False,
        "sklearn": False,
        "torch": False,
        "transformers": False,
        "vllm": False,
    }


def test_treepo_methods_import_keeps_optional_modules_lazy() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
import treepo.methods
from treepo.methods import fit, run, list_methods
heavy = ["dspy", "openai", "pandas", "scipy", "sklearn", "torch", "transformers", "vllm"]
print(json.dumps({
    "heavy": {name: name in sys.modules for name in heavy},
    "fit_module": fit.__module__,
    "run_module": run.__module__,
    "methods": list_methods.__module__,
}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(proc.stdout)
    assert payload == {
        "heavy": {
            "dspy": False,
            "openai": False,
            "pandas": False,
            "scipy": False,
            "sklearn": False,
            "torch": False,
            "transformers": False,
            "vllm": False,
        },
        "fit_module": "treepo.methods.learning",
        "run_module": "treepo.methods.dispatch",
        "methods": "treepo.methods.dispatch",
    }


def test_public_method_adapter_imports_keep_optional_modules_lazy() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
import treepo.methods.dspy
import treepo.methods.fno
import treepo.methods.neural_operator
import treepo.methods.llm
import treepo.methods.estimators
import treepo.methods.g_estimators
import treepo.methods.diffusion
import treepo.methods.lda
import treepo.llm.diffusion
heavy = ["dspy", "openai", "pandas", "scipy", "sklearn", "torch", "transformers", "vllm"]
print(json.dumps({
    "heavy": {name: name in sys.modules for name in heavy},
}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(proc.stdout) == {
        "heavy": {
            "dspy": False,
            "openai": False,
            "pandas": False,
            "scipy": False,
            "sklearn": False,
            "torch": False,
            "transformers": False,
            "vllm": False,
        }
    }


def test_public_defaults_helpers_keep_optional_modules_lazy() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
from treepo import LmSection, build_lm_config_dict, load_dataclass
import treepo.methods.canonical_defaults as cd
_ = cd.LmSection()
_ = cd.DEFAULT_BATCH_SIZE
heavy = ["dspy", "openai", "pandas", "scipy", "sklearn", "torch", "transformers", "vllm"]
print(json.dumps({
    "heavy": {name: name in sys.modules for name in heavy},
    "lm_module": LmSection.__module__,
}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(proc.stdout) == {
        "heavy": {
            "dspy": False,
            "openai": False,
            "pandas": False,
            "scipy": False,
            "sklearn": False,
            "torch": False,
            "transformers": False,
            "vllm": False,
        },
        "lm_module": "treepo.methods.canonical_defaults",
    }


def test_top_level_normalized_output_export() -> None:
    from treepo import NormalizedOutput

    out = NormalizedOutput(metrics={"x": 1.0})
    assert out.metrics["x"] == 1.0


def test_benchmarks_are_not_top_level_exports() -> None:
    import importlib
    import treepo

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("treepo.runtime")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("treepo.sketches")


def test_experiment_context_records_sampling_roles_and_results(tmp_path: Path) -> None:
    scorer = role_ref(ROLE_SCORER, kind="mock", model="unit-test-model")
    method_ref = MethodRef(
        family="unit",
        method_id="full_context",
        metadata=roles_metadata({"scorer": scorer}),
    )
    ctx = ExperimentContext(
        experiment_id="exp_unit",
        output_root=tmp_path,
        benchmark_ref=BenchmarkRef(family="fixture", name="tiny"),
        method_ref=method_ref,
        sampling=SamplingPlan(seed=7, split="test", strategy="fixture", sample_budget=2),
    )

    out = ctx.record(
        {
            "metrics": {"accuracy": 0.5},
            "artifacts": {"predictions": "predictions.jsonl"},
            "metadata": {"n": 2},
        },
        phase="evaluate",
    )

    assert out.metrics["accuracy"] == 0.5
    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "exp_unit"
    assert manifest["metadata"]["sampling"]["seed"] == 7
    assert manifest["method_ref"]["metadata"]["roles"]["scorer"]["model"] == "unit-test-model"

    rows = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["metric_name"] == "accuracy"
    assert row["seed"] == 7
    assert row["split"] == "test"


def test_experiment_context_rejects_raw_module_train_shape(tmp_path: Path) -> None:
    class RawModuleLike:
        def state_dict(self):
            return {}

        def parameters(self):
            return []

        def train(self, mode: bool = True):
            return self

    ctx = ExperimentContext(
        experiment_id="exp_unit",
        output_root=tmp_path,
        benchmark_ref=BenchmarkRef(family="fixture"),
        method_ref=MethodRef(family="raw"),
    )

    with pytest.raises(TypeError, match="raw model modules"):
        ctx.train(RawModuleLike(), train_data=[])
