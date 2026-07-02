from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from treepo.core import (
    ROLE_SCORER,
    BenchmarkRef,
    ExperimentContext,
    MethodRef,
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
public = sorted(treepo.__all__)
heavy = ["datasets", "dspy", "openai", "pandas", "peft", "scipy", "sentence_transformers", "sklearn", "torch", "transformers", "trl", "vllm"]
print(json.dumps({
    "heavy": {name: name in sys.modules for name in heavy},
    "public": public,
    "has_run": hasattr(treepo, "run"),
    "has_list_methods": hasattr(treepo, "list_methods"),
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
            "datasets": False,
            "dspy": False,
            "openai": False,
            "pandas": False,
            "peft": False,
            "scipy": False,
            "sentence_transformers": False,
            "sklearn": False,
            "torch": False,
            "transformers": False,
            "trl": False,
            "vllm": False,
        },
        "public": [
            "Candidate",
            "ComposableStatistic",
            "FitConfig",
            "FitResult",
            "InfluenceWeightedAuditOverlap",
            "LawKind",
            "LocalLawAuditRow",
            "LocalLawObjectiveSummary",
            "PreferenceDataset",
            "PreferenceRecord",
            "StatisticInfo",
            "TaskState",
            "TreeNode",
            "TreeRecord",
            "TreeUnitRef",
            "__version__",
            "compute_influence_weighted_overlap",
            "corrected_local_law_loss",
            "family_statistic",
            "fit",
            "local_law_objective_summary",
            "state_from_value",
            "state_to_dict",
            "unit_ref_from",
        ],
        "has_run": False,
        "has_list_methods": False,
    }


def test_treepo_methods_import_keeps_optional_modules_lazy() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
import treepo.methods
from treepo.methods import fit
heavy = ["datasets", "dspy", "openai", "pandas", "peft", "scipy", "sentence_transformers", "sklearn", "torch", "transformers", "trl", "vllm"]
print(json.dumps({
    "heavy": {name: name in sys.modules for name in heavy},
    "fit_module": fit.__module__,
    "methods_exports_run": hasattr(treepo.methods, "run"),
    "methods_exports_list_methods": hasattr(treepo.methods, "list_methods"),
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
            "datasets": False,
            "dspy": False,
            "openai": False,
            "pandas": False,
            "peft": False,
            "scipy": False,
            "sentence_transformers": False,
            "sklearn": False,
            "torch": False,
            "transformers": False,
            "trl": False,
            "vllm": False,
        },
        "fit_module": "treepo.methods.learning",
        "methods_exports_run": False,
        "methods_exports_list_methods": False,
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
import treepo.methods.lda
heavy = ["datasets", "dspy", "openai", "pandas", "peft", "scipy", "sentence_transformers", "sklearn", "torch", "transformers", "trl", "vllm"]
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
            "datasets": False,
            "dspy": False,
            "openai": False,
            "pandas": False,
            "peft": False,
            "scipy": False,
            "sentence_transformers": False,
            "sklearn": False,
            "torch": False,
            "transformers": False,
            "trl": False,
            "vllm": False,
        }
    }


def test_public_defaults_helpers_keep_optional_modules_lazy() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
from treepo.methods import LmSection, build_lm_config_dict, load_dataclass
import treepo.methods.canonical_defaults as cd
_ = cd.LmSection()
_ = cd.DEFAULT_BATCH_SIZE
heavy = ["datasets", "dspy", "openai", "pandas", "peft", "scipy", "sentence_transformers", "sklearn", "torch", "transformers", "trl", "vllm"]
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
            "datasets": False,
            "dspy": False,
            "openai": False,
            "pandas": False,
            "peft": False,
            "scipy": False,
            "sentence_transformers": False,
            "sklearn": False,
            "torch": False,
            "transformers": False,
            "trl": False,
            "vllm": False,
        },
        "lm_module": "treepo.methods.canonical_defaults",
    }


def test_llm_openai_compatible_example_stays_import_light() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
from treepo.llm import OpenAICompatibleConfig, render_chat_payload
heavy = ["datasets", "dspy", "openai", "pandas", "peft", "scipy", "sentence_transformers", "sklearn", "torch", "transformers", "trl", "vllm"]
config = OpenAICompatibleConfig(
    base_url="http://localhost:8000/v1",
    model="served-chat-model",
    api_key="EMPTY",
)
payload = render_chat_payload(
    model=config.model,
    messages=[
        {"role": "system", "content": "Return a compact answer."},
        {"role": "user", "content": "Summarize the local-law result."},
    ],
    temperature=0.0,
    max_tokens=64,
)
print(json.dumps({
    "base_url": config.base_url,
    "heavy": {name: name in sys.modules for name in heavy},
    "payload": payload,
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
    assert payload["base_url"] == "http://localhost:8000/v1"
    assert payload["payload"]["model"] == "served-chat-model"
    assert payload["payload"]["messages"][1]["content"] == "Summarize the local-law result."
    assert payload["payload"]["max_tokens"] == 64
    assert payload["heavy"] == {
        "datasets": False,
        "dspy": False,
        "openai": False,
        "pandas": False,
        "peft": False,
        "scipy": False,
        "sentence_transformers": False,
        "sklearn": False,
        "torch": False,
        "transformers": False,
        "trl": False,
        "vllm": False,
    }


def test_top_level_exposes_fit_and_preference_dataset_only() -> None:
    import treepo

    assert callable(treepo.fit)
    assert treepo.ComposableStatistic.__name__ == "ComposableStatistic"
    assert treepo.StatisticInfo.__name__ == "StatisticInfo"
    assert treepo.TaskState.__name__ == "TaskState"
    assert treepo.TreeNode.__name__ == "TreeNode"
    assert treepo.TreeRecord.__name__ == "TreeRecord"
    assert treepo.TreeUnitRef.__name__ == "TreeUnitRef"
    assert callable(treepo.family_statistic)
    assert callable(treepo.state_from_value)
    assert callable(treepo.state_to_dict)
    assert callable(treepo.unit_ref_from)
    assert treepo.PreferenceDataset.__name__ == "PreferenceDataset"
    assert treepo.PreferenceRecord.__name__ == "PreferenceRecord"
    assert treepo.Candidate.__name__ == "Candidate"
    assert not hasattr(treepo, "run")
    assert not hasattr(treepo, "list_methods")
    assert not hasattr(treepo, "NormalizedOutput")


def test_benchmarks_are_not_top_level_exports() -> None:
    import importlib


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
