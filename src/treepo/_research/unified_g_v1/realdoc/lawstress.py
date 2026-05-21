from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import requests

from treepo._research.unified_g_v1.core.artifact import (
    TextUnifiedGProgram,
    UnifiedGArtifact,
    resolve_text_unified_g_program,
)
from treepo._research.tasks.manifesto.lawstress_eval import (
    LawStressEvalConfig,
    RILE_RUBRIC,
    build_eval_metrics,
    build_predictions,
    score_and_judge_predictions,
)
from treepo._research.tasks.manifesto.lawstress_generator import (
    LawStressRecord,
    load_lawstress_records_jsonl,
)


_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client for LawStress scoring."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 120.0,
        enable_thinking: bool = False,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = str(api_key)
        self.timeout_seconds = float(timeout_seconds)
        self.enable_thinking = bool(enable_thinking)

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)},
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        choices = list(payload.get("choices") or [])
        if not choices:
            return ""
        message = dict(choices[0].get("message") or {})
        return str(message.get("content") or "").strip()


def _parse_score(text: str) -> float:
    matches = _NUMERIC_RE.findall(str(text or ""))
    if not matches:
        raise ValueError(f"could not parse numeric score from response={text!r}")
    return max(-100.0, min(100.0, float(matches[-1])))


def build_numeric_score_fn(
    client: OpenAIChatClient,
    *,
    temperature: float = 0.0,
    max_tokens: int = 16,
) -> Callable[[str], float]:
    def _score(text: str) -> float:
        rendered = client.chat(
            system="Return exactly one numeric RILE score in [-100,100].",
            user=(
                "Score this text on a RILE-style directional scale and return only one "
                f"number.\n\nTEXT:\n{text}"
            ),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return _parse_score(rendered)

    return _score


def build_unified_merge_fn(
    artifact: UnifiedGArtifact | TextUnifiedGProgram,
) -> Callable[[str, str, str], str]:
    program = resolve_text_unified_g_program(artifact)

    def _merge(left: str, right: str, rubric: str) -> str:
        return str(program.render_merge(str(left), str(right), str(rubric)) or "")

    return _merge


def load_lawstress_records(path: str | Path) -> list[LawStressRecord]:
    return list(load_lawstress_records_jsonl(Path(path)))


def evaluate_lawstress_records(
    records: Sequence[LawStressRecord],
    *,
    artifact: UnifiedGArtifact | TextUnifiedGProgram,
    score_fn: Callable[[str], float],
    judge_fn: Callable[..., Any] | None = None,
    config: LawStressEvalConfig | None = None,
    num_workers: int = 2,
) -> dict[str, Any]:
    program = resolve_text_unified_g_program(artifact)
    eval_config = config or LawStressEvalConfig()
    worker_count = int(num_workers)
    if len(records) > 1:
        worker_count = max(2, worker_count)
    predictions = build_predictions(
        records,
        summarize_fn=lambda text, rubric: str(program.render_leaf(str(text), str(rubric)) or ""),
        merge_fn=build_unified_merge_fn(program),
        rubric=RILE_RUBRIC,
        resummary_hops=int(eval_config.resummary_hops),
        num_workers=1 if len(records) <= 1 else worker_count,
    )
    results = score_and_judge_predictions(
        records,
        predictions,
        score_fn=score_fn,
        judge_fn=judge_fn,
        config=eval_config,
        rubric=RILE_RUBRIC,
        num_workers=1 if len(predictions) <= 1 else worker_count,
    )
    metrics = build_eval_metrics(results, eval_config)
    return {
        "config": asdict(eval_config),
        "program_contract": program.contract.to_dict(),
        "record_count": len(records),
        "prediction_count": len(predictions),
        "result_count": len(results),
        "predictions": [prediction.to_dict() for prediction in predictions],
        "results": [result.to_dict() for result in results],
        "metrics": metrics,
    }
