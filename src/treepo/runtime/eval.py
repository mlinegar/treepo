from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from treepo.runtime.longbench import (
    CHOICES,
    LongBenchRow,
    load_longbench_jsonl,
    parse_choice,
    render_longbench_prompt,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_METHODS = ("full_context", "retrieval", "summary_tree", "state_tree", "neural_operator")
RUNTIME_CONFIG_KEYS = {
    "experiment_id",
    "benchmark",
    "methods",
    "scorer",
    "summarizer",
    "embedder",
    "state_model",
    "oracle",
    "runtime_defaults",
}


@dataclass(frozen=True)
class RuntimeCall:
    experiment_id: str
    method_id: str
    problem_id: str
    role: str
    surface: str
    request_kind: str
    latency_s: float = 0.0
    cost: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(
            {
                "experiment_id": self.experiment_id,
                "method_id": self.method_id,
                "problem_id": self.problem_id,
                "role": self.role,
                "surface": self.surface,
                "request_kind": self.request_kind,
                "latency_s": round(float(self.latency_s), 6),
                "cost": dict(self.cost),
                "artifacts": dict(self.artifacts),
            }
        )


@dataclass(frozen=True)
class RuntimePrediction:
    experiment_id: str
    method_id: str
    problem_id: str
    prediction: str
    answer: str
    correct: bool
    domain: str = ""
    difficulty: str = ""
    length: str = ""
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(
            {
                "experiment_id": self.experiment_id,
                "method_id": self.method_id,
                "problem_id": self.problem_id,
                "prediction": self.prediction,
                "answer": self.answer,
                "correct": bool(self.correct),
                "domain": self.domain,
                "difficulty": self.difficulty,
                "length": self.length,
                "artifacts": dict(self.artifacts),
            }
        )


@dataclass(frozen=True)
class RuntimeEvalSummary:
    experiment_id: str
    config: Mapping[str, Any]
    metrics: Mapping[str, Any]
    method_metrics: Sequence[Mapping[str, Any]]
    predictions: Sequence[RuntimePrediction]
    calls: Sequence[RuntimeCall]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "method_metrics": [dict(row) for row in self.method_metrics],
            "predictions": [pred.to_dict() for pred in self.predictions],
            "calls": [call.to_dict() for call in self.calls],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def validate_runtime_config(config: Mapping[str, Any]) -> None:
    unknown = sorted(str(key) for key in config.keys() if str(key) not in RUNTIME_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unsupported longbench-runtime config keys: {unknown}")
    methods = tuple(str(item) for item in config.get("methods", ()) or ())
    if not methods:
        raise ValueError("longbench-runtime config requires at least one method")
    bad = sorted(method for method in methods if method not in RUNTIME_METHODS)
    if bad:
        raise ValueError(f"unsupported longbench-runtime methods: {bad}")
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ValueError("longbench-runtime config requires benchmark mapping")
    if not benchmark.get("dataset"):
        raise ValueError("longbench-runtime benchmark requires dataset")
    if not isinstance(config.get("scorer"), Mapping):
        raise ValueError("longbench-runtime config requires scorer mapping")


def run_runtime_eval(config: Mapping[str, Any]) -> RuntimeEvalSummary:
    validate_runtime_config(config)
    experiment_id = str(config.get("experiment_id") or "runtime_eval")
    defaults = dict(config.get("runtime_defaults") or {})
    rows = _load_rows(dict(config.get("benchmark") or {}), limit=_optional_int(defaults.get("limit")))
    methods = tuple(str(item) for item in config.get("methods", ()) or ())
    scorer = dict(config.get("scorer") or {})
    summarizer = dict(config.get("summarizer") or scorer)
    embedder = dict(config.get("embedder") or {})
    state_model = dict(config.get("state_model") or {})
    mock = bool(defaults.get("mock", True))

    predictions: list[RuntimePrediction] = []
    calls: list[RuntimeCall] = []

    for method_id in methods:
        for row in rows:
            pred, artifacts = _predict_one(
                row=row,
                method_id=method_id,
                experiment_id=experiment_id,
                defaults=defaults,
                scorer=scorer,
                summarizer=summarizer,
                embedder=embedder,
                state_model=state_model,
                mock=mock,
                calls=calls,
            )
            prediction = parse_choice(pred)
            predictions.append(
                RuntimePrediction(
                    experiment_id=experiment_id,
                    method_id=method_id,
                    problem_id=row.id,
                    prediction=prediction,
                    answer=row.answer,
                    correct=prediction == row.answer,
                    domain=row.domain,
                    difficulty=row.difficulty,
                    length=row.length,
                    artifacts=artifacts,
                )
            )

    method_metrics = []
    for method_id in methods:
        method_preds = [pred for pred in predictions if pred.method_id == method_id]
        total = len(method_preds)
        correct = sum(1 for pred in method_preds if pred.correct)
        method_metrics.append(
            {
                "method_id": method_id,
                "accuracy": float(correct / total) if total else 0.0,
                "n": total,
                "correct": correct,
            }
        )
    total = len(predictions)
    correct = sum(1 for pred in predictions if pred.correct)
    return RuntimeEvalSummary(
        experiment_id=experiment_id,
        config=dict(config),
        metrics={"accuracy": float(correct / total) if total else 0.0, "n": total, "correct": correct},
        method_metrics=method_metrics,
        predictions=predictions,
        calls=calls,
    )


def runtime_summary_to_csv_rows(summary: RuntimeEvalSummary) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_metrics = {str(row.get("method_id")): dict(row) for row in summary.method_metrics}
    for pred in summary.predictions:
        row = pred.to_dict()
        metric = method_metrics.get(pred.method_id, {})
        row["method_accuracy"] = metric.get("accuracy", "")
        row["experiment"] = "longbench_runtime"
        rows.append(row)
    return rows


def _predict_one(
    *,
    row: LongBenchRow,
    method_id: str,
    experiment_id: str,
    defaults: Mapping[str, Any],
    scorer: Mapping[str, Any],
    summarizer: Mapping[str, Any],
    embedder: Mapping[str, Any],
    state_model: Mapping[str, Any],
    mock: bool,
    calls: list[RuntimeCall],
) -> tuple[str, dict[str, Any]]:
    chunk_chars = int(defaults.get("chunk_chars") or 2000)
    top_k = int(defaults.get("top_k") or 4)
    chunks = _chunk_text(row.context, chunk_chars=chunk_chars)
    artifacts: dict[str, Any] = {"n_context_chars": len(row.context), "n_chunks": len(chunks)}

    if method_id == "full_context":
        evidence = row.context
        artifacts["evidence_source"] = "full_context"
    elif method_id == "retrieval":
        evidence_chunks = _select_evidence(
            experiment_id=experiment_id,
            method_id=method_id,
            row=row,
            chunks=chunks,
            top_k=top_k,
            role="embedder",
            surface="embedding",
            request_kind="retrieve",
            calls=calls,
            role_config=embedder,
        )
        evidence = "\n\n".join(evidence_chunks)
        artifacts["evidence_source"] = "embedder"
        artifacts["selected_chunks"] = len(evidence_chunks)
    elif method_id == "summary_tree":
        leaf_summaries = _summarize_chunks(
            experiment_id=experiment_id,
            method_id=method_id,
            row=row,
            chunks=chunks,
            summarizer=summarizer,
            calls=calls,
        )
        evidence = "\n".join(leaf_summaries)
        artifacts["evidence_source"] = "summarizer"
        artifacts["selected_chunks"] = len(leaf_summaries)
    elif method_id == "state_tree":
        evidence = _render_state(
            experiment_id=experiment_id,
            method_id=method_id,
            row=row,
            chunks=chunks,
            state_model=state_model,
            top_k=top_k,
            calls=calls,
        )
        artifacts["evidence_source"] = "state_model"
    elif method_id == "neural_operator":
        evidence_chunks = _select_evidence(
            experiment_id=experiment_id,
            method_id=method_id,
            row=row,
            chunks=chunks,
            top_k=top_k,
            role="state_model" if state_model else "embedder",
            surface="operator" if state_model else "embedding",
            request_kind="select_evidence",
            calls=calls,
            role_config=state_model or embedder,
        )
        evidence = "\n\n".join(evidence_chunks)
        artifacts["evidence_source"] = "state_model" if state_model else "embedder"
        artifacts["selected_chunks"] = len(evidence_chunks)
    else:  # pragma: no cover - validate_runtime_config prevents this
        raise ValueError(f"unsupported runtime method: {method_id}")

    answer = _score_answer(
        experiment_id=experiment_id,
        method_id=method_id,
        row=row,
        evidence=evidence,
        scorer=scorer,
        calls=calls,
        mock=mock,
        defaults=defaults,
    )
    return answer, artifacts


def _load_rows(benchmark: Mapping[str, Any], *, limit: int | None = None) -> list[LongBenchRow]:
    dataset = str(benchmark.get("dataset") or "")
    path = _resolve_path(dataset)
    if path.suffix.lower() == ".jsonl":
        return load_longbench_jsonl(path, limit=limit)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_rows = payload.get("rows")
    else:
        raw_rows = payload
    if not isinstance(raw_rows, Sequence):
        raise ValueError(f"LongBench dataset fixture must contain rows: {path}")
    rows = [LongBenchRow.from_mapping(row) for row in raw_rows if isinstance(row, Mapping)]
    return rows[:limit] if limit is not None else rows


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path, PACKAGE_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_text)


def _chunk_text(text: str, *, chunk_chars: int) -> list[str]:
    text = str(text or "")
    size = max(1, int(chunk_chars))
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _select_evidence(
    *,
    experiment_id: str,
    method_id: str,
    row: LongBenchRow,
    chunks: Sequence[str],
    top_k: int,
    role: str,
    surface: str,
    request_kind: str,
    calls: list[RuntimeCall],
    role_config: Mapping[str, Any],
) -> list[str]:
    start = time.perf_counter()
    scored = sorted(
        ((idx, _overlap_score(row.question, chunk, row.choices), chunk) for idx, chunk in enumerate(chunks)),
        key=lambda item: (-item[1], item[0]),
    )
    chosen = [chunk for _, _, chunk in scored[: max(1, min(int(top_k), len(scored)))]]
    calls.append(
        RuntimeCall(
            experiment_id=experiment_id,
            method_id=method_id,
            problem_id=row.id,
            role=role,
            surface=surface,
            request_kind=request_kind,
            latency_s=time.perf_counter() - start,
            artifacts={
                "backend": str(role_config.get("kind") or "deterministic_overlap"),
                "n_candidates": len(chunks),
                "top_k": len(chosen),
            },
        )
    )
    return chosen


def _summarize_chunks(
    *,
    experiment_id: str,
    method_id: str,
    row: LongBenchRow,
    chunks: Sequence[str],
    summarizer: Mapping[str, Any],
    calls: list[RuntimeCall],
) -> list[str]:
    start = time.perf_counter()
    summaries = [_best_sentence(chunk, row.question, row.choices) for chunk in chunks]
    calls.append(
        RuntimeCall(
            experiment_id=experiment_id,
            method_id=method_id,
            problem_id=row.id,
            role="summarizer",
            surface="chat_openai",
            request_kind="summarize_tree",
            latency_s=time.perf_counter() - start,
            artifacts={"backend": str(summarizer.get("kind") or "deterministic_sentence"), "n_chunks": len(chunks)},
        )
    )
    return summaries


def _render_state(
    *,
    experiment_id: str,
    method_id: str,
    row: LongBenchRow,
    chunks: Sequence[str],
    state_model: Mapping[str, Any],
    top_k: int,
    calls: list[RuntimeCall],
) -> str:
    chosen = _select_evidence(
        experiment_id=experiment_id,
        method_id=method_id,
        row=row,
        chunks=chunks,
        top_k=top_k,
        role="state_model",
        surface="operator",
        request_kind="render_state",
        calls=calls,
        role_config=state_model,
    )
    return "\n\n".join(chosen)


def _score_answer(
    *,
    experiment_id: str,
    method_id: str,
    row: LongBenchRow,
    evidence: str,
    scorer: Mapping[str, Any],
    calls: list[RuntimeCall],
    mock: bool,
    defaults: Mapping[str, Any],
) -> str:
    start = time.perf_counter()
    prompt = render_longbench_prompt(
        LongBenchRow(
            id=row.id,
            question=row.question,
            choices=row.choices,
            answer=row.answer,
            context=evidence,
            domain=row.domain,
            sub_domain=row.sub_domain,
            difficulty=row.difficulty,
            length=row.length,
        )
    )
    if mock:
        prediction = _heuristic_choice(evidence, row.question, row.choices)
        backend = "deterministic_overlap"
    else:
        prediction = _openai_chat_choice(scorer, prompt=prompt, defaults=defaults)
        backend = str(scorer.get("kind") or "openai_compatible")
    calls.append(
        RuntimeCall(
            experiment_id=experiment_id,
            method_id=method_id,
            problem_id=row.id,
            role="scorer",
            surface="chat_openai",
            request_kind="answer_choice",
            latency_s=time.perf_counter() - start,
            cost={"prompt_chars": len(prompt), "completion_chars": len(prediction)},
            artifacts={"backend": backend, "model": str(scorer.get("model") or "")},
        )
    )
    return prediction


def _openai_chat_choice(scorer: Mapping[str, Any], *, prompt: str, defaults: Mapping[str, Any]) -> str:
    base_url = str(scorer.get("endpoint") or scorer.get("base_url") or "").rstrip("/")
    model = str(scorer.get("model") or "")
    if not base_url or not model:
        raise ValueError("live scorer calls require scorer.endpoint and scorer.model")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(defaults.get("temperature") or 0.0),
        "max_tokens": int(defaults.get("max_output_tokens") or 4),
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {scorer.get('api_key') or 'EMPTY'}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(defaults.get("timeout_s") or 120.0)) as response:
        raw = json.loads(response.read().decode("utf-8"))
    text = str(raw["choices"][0]["message"]["content"])
    return parse_choice(text)


def _heuristic_choice(context: str, question: str, choices: Mapping[str, str]) -> str:
    context_words = _tokens(context)
    question_words = _tokens(question)
    best_key = ""
    best_score = -math.inf
    for key in CHOICES:
        choice_words = _tokens(choices.get(key, ""))
        score = (
            2.0 * float(len(choice_words & context_words))
            + 0.25 * float(len(choice_words & question_words))
            - 0.001 * float(len(choice_words))
        )
        if score > best_score:
            best_key = key
            best_score = score
    return best_key or "A"


def _overlap_score(question: str, text: str, choices: Mapping[str, str]) -> float:
    q_words = _tokens(question)
    text_words = _tokens(text)
    choice_words = set().union(*(_tokens(value) for value in choices.values())) if choices else set()
    return float(len((q_words | choice_words) & text_words)) + 0.05 * float(len(choice_words & text_words))


def _best_sentence(text: str, question: str, choices: Mapping[str, str]) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text)) if part.strip()]
    if not sentences:
        return str(text).strip()
    return max(sentences, key=lambda sentence: _overlap_score(question, sentence, choices))


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if len(token) > 1}


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value not in ("", None, {}, [])}


__all__ = [
    "RUNTIME_CONFIG_KEYS",
    "RUNTIME_METHODS",
    "RuntimeCall",
    "RuntimeEvalSummary",
    "RuntimePrediction",
    "run_runtime_eval",
    "runtime_summary_to_csv_rows",
    "validate_runtime_config",
]
