from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from treepo._research.runtime.adapters.base import BenchmarkAdapter
from treepo._research.runtime.contracts import AnswerSpec, NodeContract, ProblemSpec, RuntimeTaskView


ANSWER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", re.IGNORECASE)


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        rows: List[Dict[str, Any]] = []
        for child in sorted(path.glob("*.jsonl")):
            rows.extend(_read_json_or_jsonl(child))
        for child in sorted(path.glob("*.json")):
            rows.extend(_read_json_or_jsonl(child))
        return rows
    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    out.append(payload)
        return out
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
        return [dict(payload)]
    raise ValueError(f"Unsupported LongBench local payload in {path}")


def _load_hf_rows(
    *,
    dataset_name: str,
    split: str,
    dataset_config: Optional[str] = None,
    streaming: bool = False,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in environments without datasets
        raise ImportError(
            "LongBench HF loading requires the `datasets` package. "
            "Install thinking-trees[trl] or provide benchmark.dataset_path."
        ) from exc

    args: List[Any] = [dataset_name]
    if dataset_config:
        args.append(dataset_config)
    ds = load_dataset(*args, split=split, streaming=bool(streaming))
    rows: List[Dict[str, Any]] = []
    for row in ds:
        rows.append(dict(row))
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


@dataclass(frozen=True)
class LongBenchV2Spec:
    task_id: str
    split: str
    max_seq_length: int
    num_samples: int
    seed: int


class LongBenchV2Adapter(BenchmarkAdapter):
    """Adapter for THUDM LongBench v2 multiple-choice records."""

    def __init__(
        self,
        *,
        spec: LongBenchV2Spec,
        dataset_path: Optional[Path] = None,
        hf_dataset: str = "THUDM/LongBench-v2",
        hf_config: Optional[str] = None,
        streaming: bool = False,
        domains: Optional[Sequence[str]] = None,
        sub_domains: Optional[Sequence[str]] = None,
        difficulties: Optional[Sequence[str]] = None,
        length_buckets: Optional[Sequence[str]] = None,
    ) -> None:
        self.spec = spec
        self.dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        self.hf_dataset = str(hf_dataset)
        self.hf_config = str(hf_config) if hf_config else None
        self.streaming = bool(streaming)
        self.domains = {_slug(x) for x in (domains or [])}
        self.sub_domains = {_slug(x) for x in (sub_domains or [])}
        self.difficulties = {_slug(x) for x in (difficulties or [])}
        self.length_buckets = {_slug(x) for x in (length_buckets or [])}
        self._rows_cache: Optional[List[Dict[str, Any]]] = None

    def primary_metric(self) -> str:
        return "longbench_v2_accuracy"

    def supports_tools(self) -> bool:
        return False

    def _rows(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if self._rows_cache is None:
            if self.dataset_path is not None:
                self._rows_cache = _read_json_or_jsonl(self.dataset_path)
            else:
                rows = _load_hf_rows(
                    dataset_name=self.hf_dataset,
                    dataset_config=self.hf_config,
                    split=self.spec.split,
                    streaming=self.streaming,
                    limit=limit,
                )
                if limit is None:
                    self._rows_cache = rows
                return rows
        return list(self._rows_cache)

    def _row_matches(self, row: Mapping[str, Any]) -> bool:
        task_id = str(self.spec.task_id or "all").strip()
        task_slug = _slug(task_id)
        domain = _slug(row.get("domain"))
        sub_domain = _slug(row.get("sub_domain"))
        difficulty = _slug(row.get("difficulty"))
        length = _slug(row.get("length"))
        row_id = _slug(row.get("_id"))

        if task_slug not in {"", "all", "longbench_v2", "longbench"}:
            if task_slug not in {domain, sub_domain, difficulty, length, row_id}:
                return False
        if self.domains and domain not in self.domains:
            return False
        if self.sub_domains and sub_domain not in self.sub_domains:
            return False
        if self.difficulties and difficulty not in self.difficulties:
            return False
        if self.length_buckets and length not in self.length_buckets:
            return False
        return True

    @staticmethod
    def _choices(row: Mapping[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for letter in ("A", "B", "C", "D"):
            value = row.get(f"choice_{letter}")
            if value is not None:
                out[letter] = str(value)
        return out

    @staticmethod
    def _official_prompt(row: Mapping[str, Any]) -> str:
        choices = LongBenchV2Adapter._choices(row)
        choice_block = "\n".join(f"{letter}. {choices.get(letter, '')}" for letter in ("A", "B", "C", "D"))
        return (
            "Read the following long context and answer the multiple-choice question.\n\n"
            f"Context:\n{str(row.get('context', '') or '')}\n\n"
            f"Question:\n{str(row.get('question', '') or '')}\n\n"
            f"Choices:\n{choice_block}\n\n"
            "Return only one letter: A, B, C, or D."
        )

    def load_split(self, split: str, limit: int | None = None) -> Iterable[ProblemSpec]:
        if split != self.spec.split:
            raise ValueError(f"Adapter initialized for split={self.spec.split!r}, got {split!r}")
        yielded = 0
        for idx, row in enumerate(self._rows(limit=limit)):
            if not self._row_matches(row):
                continue
            row_id = str(row.get("_id") or f"row_{idx}")
            answer = str(row.get("answer", "") or "").strip().upper()[:1]
            choices = self._choices(row)
            metadata = {
                "_id": row_id,
                "domain": row.get("domain"),
                "sub_domain": row.get("sub_domain"),
                "difficulty": row.get("difficulty"),
                "length": row.get("length"),
                "choices": choices,
                "benchmark": "longbench_v2",
            }
            yield ProblemSpec(
                problem_id=f"longbench_v2:{row_id}",
                input_text=self._official_prompt(row),
                query=str(row.get("question", "") or ""),
                references=[answer] if answer else [],
                metadata=metadata,
                success_metric=self.primary_metric(),
                success_target=answer,
                constraints={
                    "max_seq_length": int(self.spec.max_seq_length),
                    "length_bucket": row.get("length"),
                },
                allowed_actions=[],
            )
            yielded += 1
            if limit is not None and yielded >= int(limit):
                break

    def task_view(self, problem: ProblemSpec) -> RuntimeTaskView:
        # Recover context by stripping the official prompt structure when the
        # original row is not retained in metadata.
        prompt = str(problem.input_text or "")
        context = prompt
        if "Context:\n" in prompt and "\n\nQuestion:\n" in prompt:
            context = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:\n", 1)[0]
        choices = {
            letter: str(value)
            for letter, value in dict(problem.metadata.get("choices") or {}).items()
            if str(letter).strip()
        }
        return RuntimeTaskView(
            context=context,
            question=str(problem.query or ""),
            choices=choices,
            answer_instruction="Return only one letter: A, B, C, or D.",
            official_prompt=prompt,
            answer_prefix="Answer:",
            metadata={
                "benchmark": "longbench_v2",
                "domain": problem.metadata.get("domain"),
                "sub_domain": problem.metadata.get("sub_domain"),
                "difficulty": problem.metadata.get("difficulty"),
                "length": problem.metadata.get("length"),
            },
        )

    def parse_prediction(self, problem: ProblemSpec, text: str) -> str:
        rendered = str(text or "").strip()
        if not rendered:
            return ""
        answer_match = re.search(r"answer\s*[:：]\s*([ABCD])", rendered, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        match = ANSWER_RE.search(rendered)
        if match:
            return match.group(1).upper()
        first = rendered[:1].upper()
        return first if first in {"A", "B", "C", "D"} else rendered

    def build_answer_spec(self, problem: ProblemSpec) -> AnswerSpec:
        choices = {
            str(letter).strip().upper()[:1]: str(value)
            for letter, value in dict(problem.metadata.get("choices") or {}).items()
            if str(letter).strip()
        }
        return AnswerSpec(
            kind="multi_choice",
            choices={letter: choices[letter] for letter in ("A", "B", "C", "D") if letter in choices},
            answer_prefix="Answer:",
            instruction="Choose the single best option. Return only one letter: A, B, C, or D.",
            metadata={
                "benchmark": "longbench_v2",
                "domain": problem.metadata.get("domain"),
                "sub_domain": problem.metadata.get("sub_domain"),
                "difficulty": problem.metadata.get("difficulty"),
                "length": problem.metadata.get("length"),
            },
        )

    def build_contract(self, problem: ProblemSpec) -> NodeContract:
        return NodeContract(
            objective="Answer the LongBench v2 multiple-choice question from the provided context.",
            must_preserve=["question", "answer choices", "evidence needed for A-D selection"],
            output_schema={"answer": "one of A, B, C, D"},
            acceptance_checks=["budget_compliance", "multiple_choice_letter"],
            max_input_tokens=int(problem.constraints.get("max_seq_length", 8192) or 8192),
            max_output_tokens=16,
        )

    def score(self, problem: ProblemSpec, runtime_output: dict) -> dict[str, float]:
        pred = self.parse_prediction(problem, str(runtime_output.get("prediction", "") or ""))
        gold = str((problem.references or [""])[0] or "").strip().upper()
        correct = 1.0 if pred == gold and gold in {"A", "B", "C", "D"} else 0.0
        metrics: Dict[str, float] = {
            self.primary_metric(): correct,
            "exact_match": correct,
        }
        for key in ("domain", "difficulty", "length"):
            metrics[f"{self.primary_metric()}_{key}_{_slug(problem.metadata.get(key))}"] = correct
        return metrics
