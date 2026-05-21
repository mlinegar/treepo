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
from treepo.runtime.longbench import (
    load_longbench_jsonl,
    parse_choice,
    render_longbench_prompt,
    score_choice_accuracy,
)


def test_treepo_import_keeps_heavy_optional_modules_lazy() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import json
import sys
import treepo
heavy = ["dspy", "openai", "pandas", "torch", "transformers", "vllm"]
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
        "torch": False,
        "transformers": False,
        "vllm": False,
    }


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


def test_longbench_fixture_helpers(tmp_path: Path) -> None:
    path = tmp_path / "longbench.jsonl"
    path.write_text(
        json.dumps(
            {
                "_id": "lbv2-1",
                "domain": "law",
                "sub_domain": "contracts",
                "difficulty": "easy",
                "length": "short",
                "question": "Which option is supported?",
                "choice_A": "No evidence",
                "choice_B": "The contract was signed.",
                "choice_C": "The contract expired.",
                "choice_D": "The contract was void.",
                "answer": "B",
                "context": "The contract was signed on Monday.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_longbench_jsonl(path)
    assert len(rows) == 1
    prompt = render_longbench_prompt(rows[0])
    assert "Context:" in prompt
    assert "B. The contract was signed." in prompt
    assert parse_choice("Answer: B") == "B"
    assert score_choice_accuracy(rows, ["B"]) == 1.0
