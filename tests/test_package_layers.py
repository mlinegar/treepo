from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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
            "PreferenceDataset",
            "PreferenceRecord",
            "TaskState",
            "TreeNode",
            "TreeRecord",
            "__version__",
            "family_statistic",
            "fit",
            "state_from_value",
            "state_to_dict",
            "write_tree_visualization_html",
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
from treepo.methods import load_dataclass
import treepo.methods.canonical_defaults as cd
heavy = ["datasets", "dspy", "openai", "pandas", "peft", "scipy", "sentence_transformers", "sklearn", "torch", "transformers", "trl", "vllm"]
print(json.dumps({
    "heavy": {name: name in sys.modules for name in heavy},
    "load_dataclass_module": load_dataclass.__module__,
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
        "load_dataclass_module": "treepo.methods.canonical_defaults",
    }


def test_llm_openai_compatible_example_stays_import_light() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
from treepo.llm import render_chat_payload
heavy = ["datasets", "dspy", "openai", "pandas", "peft", "scipy", "sentence_transformers", "sklearn", "torch", "transformers", "trl", "vllm"]
payload = render_chat_payload(
    model="served-chat-model",
    messages=[
        {"role": "system", "content": "Return a compact answer."},
        {"role": "user", "content": "Summarize the local-law result."},
    ],
    temperature=0.0,
    max_tokens=64,
)
print(json.dumps({
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
    assert treepo.TaskState.__name__ == "TaskState"
    assert treepo.TreeNode.__name__ == "TreeNode"
    assert treepo.TreeRecord.__name__ == "TreeRecord"
    assert callable(treepo.family_statistic)
    assert callable(treepo.state_from_value)
    assert callable(treepo.state_to_dict)
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
