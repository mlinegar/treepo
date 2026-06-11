from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class LongBenchRow:
    id: str
    question: str
    choices: Mapping[str, str]
    answer: str
    context: str
    domain: str = ""
    sub_domain: str = ""
    difficulty: str = ""
    length: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "LongBenchRow":
        return cls(
            id=str(row.get("_id") or row.get("id") or ""),
            question=str(row.get("question") or ""),
            choices={key: str(row.get(f"choice_{key}") or "") for key in CHOICES},
            answer=str(row.get("answer") or "").strip().upper()[:1],
            context=str(row.get("context") or ""),
            domain=str(row.get("domain") or ""),
            sub_domain=str(row.get("sub_domain") or ""),
            difficulty=str(row.get("difficulty") or ""),
            length=str(row.get("length") or ""),
        )


def load_longbench_jsonl(path: str | Path, *, limit: int | None = None) -> list[LongBenchRow]:
    rows: list[LongBenchRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(LongBenchRow.from_mapping(json.loads(line)))
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def render_longbench_prompt(row: LongBenchRow) -> str:
    choices = "\n".join(f"{key}. {row.choices.get(key, '')}" for key in CHOICES)
    return (
        "Read the context and answer the multiple-choice question.\n\n"
        f"Context:\n{row.context}\n\n"
        f"Question:\n{row.question}\n\n"
        f"Choices:\n{choices}\n\n"
        "Return only one letter: A, B, C, or D."
    )


def parse_choice(text: str) -> str:
    stripped = str(text or "").strip().upper()
    match = re.match(r"^\s*([ABCD])(?:[\.\):\s]|$)", stripped)
    if match:
        return match.group(1)
    match = re.search(r"\b(?:ANSWER|OPTION|CHOICE)?\s*[:\-]?\s*([ABCD])\b", stripped)
    return match.group(1) if match else ""


def score_choice_accuracy(rows: Sequence[LongBenchRow], predictions: Iterable[str]) -> float:
    total = 0
    correct = 0
    for row, pred in zip(rows, predictions):
        total += 1
        correct += int(parse_choice(pred) == row.answer)
    return float(correct / total) if total else 0.0


__all__ = [
    "LongBenchRow",
    "load_longbench_jsonl",
    "parse_choice",
    "render_longbench_prompt",
    "score_choice_accuracy",
]
