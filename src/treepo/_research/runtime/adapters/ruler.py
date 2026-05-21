from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from treepo._research.runtime.adapters.base import BenchmarkAdapter
from treepo._research.runtime.contracts import AnswerSpec, NodeContract, ProblemSpec, RuntimeTaskView


def _string_match_part(preds: List[str], refs: List[List[str]]) -> float:
    score = (
        sum(
            max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)
            for pred, ref in zip(preds, refs)
        )
        / max(1, len(preds))
        * 100
    )
    return round(score, 2)


def _string_match_all(preds: List[str], refs: List[List[str]]) -> float:
    score = (
        sum(
            sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / max(1, len(ref))
            for pred, ref in zip(preds, refs)
        )
        / max(1, len(preds))
        * 100
    )
    return round(score, 2)


def _read_jsonl(path: Path, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


@dataclass(frozen=True)
class RulerDatasetSpec:
    task_id: str
    split: str
    max_seq_length: int
    num_samples: int
    seed: int


class RulerSyntheticAdapter(BenchmarkAdapter):
    """Adapter for the RULER synthetic benchmark.

    This adapter can (optionally) generate datasets using the upstream RULER
    scripts, but scores predictions internally (no NeMo dependency).
    """

    def __init__(
        self,
        *,
        ruler_dir: Path,
        dataset_root: Path,
        spec: RulerDatasetSpec,
        benchmark_name: str = "synthetic",
        tokenizer_type: str = "openai",
        tokenizer_path: str = "cl100k_base",
        model_template_type: str = "base",
        ensure_prepared: bool = True,
    ):
        self.ruler_dir = Path(ruler_dir)
        self.dataset_root = Path(dataset_root)
        self.spec = spec
        self.benchmark_name = benchmark_name
        self.tokenizer_type = tokenizer_type
        self.tokenizer_path = tokenizer_path
        self.model_template_type = model_template_type

        self._task_type = self._load_task_type(task_id=spec.task_id)
        self._metric_fn = self._select_metric(self._task_type)

        if ensure_prepared:
            self._prepare_if_needed()

    @staticmethod
    def _select_metric(task_type: str):
        # Mirrors `outside_data/RULER/scripts/eval/synthetic/constants.py`.
        if task_type == "qa":
            return _string_match_part
        return _string_match_all

    def primary_metric(self) -> str:
        return "ruler_score"

    def supports_tools(self) -> bool:
        return False

    def _synthetic_yaml_path(self) -> Path:
        return self.ruler_dir / "scripts" / "synthetic.yaml"

    def _load_task_type(self, *, task_id: str) -> str:
        cfg = yaml.safe_load(self._synthetic_yaml_path().read_text())
        if task_id not in cfg:
            raise ValueError(f"Unknown RULER task_id={task_id!r} (not in synthetic.yaml)")
        task_type = cfg[task_id].get("task")
        if not isinstance(task_type, str) or not task_type:
            raise ValueError(f"Malformed RULER config for task_id={task_id!r}: missing 'task'")
        return task_type

    def _dataset_dir(self) -> Path:
        # Include length+seed to keep datasets stable across unit grids.
        return (
            self.dataset_root
            / "ruler"
            / self.benchmark_name
            / f"len_{self.spec.max_seq_length}"
            / f"seed_{self.spec.seed}"
        )

    def _dataset_file(self) -> Path:
        return self._dataset_dir() / self.spec.task_id / f"{self.spec.split}.jsonl"

    def _prepare_if_needed(self) -> None:
        dataset_file = self._dataset_file()
        if dataset_file.exists():
            return

        if not self.ruler_dir.exists():
            raise FileNotFoundError(
                f"RULER repo not found at {self.ruler_dir}. Set --ruler-dir or clone it."
            )

        prepare_py = self.ruler_dir / "scripts" / "data" / "prepare.py"
        if not prepare_py.exists():
            raise FileNotFoundError(f"RULER prepare script not found: {prepare_py}")

        out_dir = self._dataset_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(sys.executable),
            str(prepare_py),
            "--save_dir",
            str(out_dir),
            "--benchmark",
            self.benchmark_name,
            "--task",
            self.spec.task_id,
            "--subset",
            self.spec.split,
            "--tokenizer_path",
            self.tokenizer_path,
            "--tokenizer_type",
            self.tokenizer_type,
            "--max_seq_length",
            str(self.spec.max_seq_length),
            "--model_template_type",
            self.model_template_type,
            "--num_samples",
            str(self.spec.num_samples),
            "--random_seed",
            str(self.spec.seed),
        ]

        env = dict(os.environ)
        # Ensure the RULER scripts can import their local modules.
        env["PYTHONPATH"] = str(self.ruler_dir / "scripts" / "data") + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        # Ensure `python` resolution inside upstream helper scripts prefers the
        # active interpreter's environment (venv dependencies like nltk).
        py_bin_dir = str(Path(sys.executable).parent)
        env["PATH"] = py_bin_dir + os.pathsep + env.get("PATH", "")

        subprocess.run(cmd, check=True, env=env)

        if not dataset_file.exists():
            raise RuntimeError(
                f"RULER dataset generation finished but output not found: {dataset_file}"
            )

    def load_split(self, split: str, limit: int | None = None) -> Iterable[ProblemSpec]:
        if split != self.spec.split:
            raise ValueError(f"Adapter initialized for split={self.spec.split!r}, got {split!r}")

        rows = _read_jsonl(self._dataset_file(), limit=limit)

        for row in rows:
            idx = row.get("index")
            input_text = row.get("input", "")
            answer_prefix = row.get("answer_prefix", "")
            outputs = row.get("outputs", [])

            # Stable id: task/split/index + seed/length.
            problem_id = f"ruler:{self.spec.task_id}:{split}:{self.spec.max_seq_length}:{self.spec.seed}:{idx}"
            metadata = dict(row)
            metadata.update(
                {
                    "ruler_task_id": self.spec.task_id,
                    "ruler_task_type": self._task_type,
                    "ruler_max_seq_length": self.spec.max_seq_length,
                    "ruler_seed": self.spec.seed,
                }
            )

            yield ProblemSpec(
                problem_id=problem_id,
                input_text=str(input_text),
                query=self._extract_query(str(input_text)),
                references=[str(x) for x in outputs],
                metadata=metadata,
                success_metric=self.primary_metric(),
                success_target=None,
                constraints={"max_seq_length": self.spec.max_seq_length},
                allowed_actions=[],
            )

    def _extract_query(self, input_text: str) -> str:
        # Best-effort extraction for logging/conditioning; the full question is already in input_text.
        if self._task_type in {"variable_tracking", "common_words_extraction", "freq_words_extraction"}:
            marker = "\nQuestion:"
            if marker in input_text:
                return input_text.split(marker, 1)[-1].strip()
        if self._task_type == "qa":
            marker = "\n\nQuestion:"
            if marker in input_text:
                return input_text.split(marker, 1)[-1].strip()
        if self._task_type == "niah":
            marker = "\nWhat "
            if marker in input_text:
                return input_text.split(marker, 1)[-1].strip()
        return ""

    def _extract_context_and_question(self, input_text: str) -> Tuple[str, str]:
        task_type = str(self._task_type or "")
        if task_type == "niah":
            marker = "\nWhat "
            if marker in input_text:
                before, after = input_text.split(marker, 1)
                return before, "What " + after.strip()
            return input_text, ""
        if task_type in {"variable_tracking", "common_words_extraction", "freq_words_extraction"}:
            marker = "\nQuestion:"
            if marker in input_text:
                before, after = input_text.split(marker, 1)
                return before, after.strip()
            return input_text, ""
        if task_type == "qa":
            marker = "\n\nQuestion:"
            if marker in input_text:
                before, after = input_text.split(marker, 1)
                return before, after.strip()
            marker2 = "\nQuestion:"
            if marker2 in input_text:
                before, after = input_text.split(marker2, 1)
                return before, after.strip()
        return input_text, ""

    def build_contract(self, problem: ProblemSpec) -> NodeContract:
        return NodeContract(
            objective="Answer the benchmark question correctly using bounded-context steps.",
            must_preserve=[],
            output_schema={"answer": "string"},
            acceptance_checks=["budget_compliance"],
            max_input_tokens=8192,
            max_output_tokens=256,
        )

    def task_view(self, problem: ProblemSpec) -> RuntimeTaskView:
        context, question = self._extract_context_and_question(str(problem.input_text or ""))
        answer_prefix = str(problem.metadata.get("answer_prefix", "") or "")
        return RuntimeTaskView(
            context=context,
            question=question or str(problem.query or ""),
            choices={},
            answer_instruction="Answer the benchmark question using the provided context.",
            official_prompt=str(problem.input_text or "") + answer_prefix,
            answer_prefix=answer_prefix,
            metadata={
                "benchmark": "ruler_synthetic",
                "task_type": self._task_type,
            },
        )

    def parse_prediction(self, problem: ProblemSpec, text: str) -> str:
        return str(text or "").strip()

    def build_answer_spec(self, problem: ProblemSpec) -> AnswerSpec:
        return AnswerSpec(
            kind="free_text",
            answer_prefix=str(problem.metadata.get("answer_prefix", "") or ""),
            instruction="Answer the benchmark question using the provided context.",
            metadata={"benchmark": "ruler_synthetic", "task_type": self._task_type},
        )

    def score(self, problem: ProblemSpec, runtime_output: dict) -> dict[str, float]:
        pred = self.parse_prediction(problem, str(runtime_output.get("prediction", "") or ""))
        refs = [list(problem.references)]
        score = self._metric_fn([pred], refs)
        return {self.primary_metric(): float(score)}
