"""Provider-neutral LLM family for :mod:`treepo.methods`.

This is deliberately small: it owns prompt/artifact plumbing and accepts an
optional injected ``predict_fn``. Concrete OpenAI/vLLM/DSPy clients can live in
application code and either pass that callable or replace the registered family.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from treepo.llm.openai_compatible import render_chat_payload

PredictFn = Callable[..., Any]


@dataclass(frozen=True)
class PromptedLLMFamilyConfig:
    model: str = "llm"
    system_prompt: str = "You estimate the tree root statistic from the supplied document."
    prompt_template: str = "Return only one numeric score for this document.\n\n{text}\n\nScore:"
    temperature: float = 0.0
    max_tokens: int = 16
    max_prompt_chars: int = 4000
    score_regex: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    default_prediction: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PromptedLLMFamily:
    """Minimal prompt-backed family runtime.

    Without ``predict_fn`` it records artifacts and returns ``default_prediction``
    when configured, otherwise ``None``. With ``predict_fn`` it renders a prompt
    per tree and parses a numeric response.
    """

    name = "llm"

    def __init__(
        self,
        config: PromptedLLMFamilyConfig | None = None,
        *,
        predict_fn: PredictFn | None = None,
    ) -> None:
        self.config = config or PromptedLLMFamilyConfig()
        self.predict_fn = predict_fn
        self._last_f: Mapping[str, Any] | None = None
        self._last_g: Mapping[str, Any] | None = None

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del f_init, g, output_dir
        artifact = self._artifact(kind="f", iteration=iteration, n_train=len(traces))
        self._last_f = artifact
        return artifact

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del g_init, f, output_dir
        artifact = self._artifact(kind="g", iteration=iteration, n_train=len(traces))
        self._last_g = artifact
        return artifact

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[float | None]:
        del f, g
        out: list[float | None] = []
        for tree in trees:
            prompt = self.render_prompt(tree)
            if self.predict_fn is None:
                out.append(self._clamp(self.config.default_prediction))
                continue
            raw = self._call_predict_fn(prompt=prompt, tree=tree)
            out.append(self._parse_prediction(raw))
        return out

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if artifact is None:
            return
        if not isinstance(artifact, Mapping):
            raise TypeError(f"llm {kind} artifact must be a mapping")

    def render_prompt(self, tree: Any) -> str:
        variables = _prompt_variables(tree)
        text = str(variables.get("text", ""))
        if len(text) > int(self.config.max_prompt_chars):
            variables["text"] = text[: int(self.config.max_prompt_chars)]
        return str(self.config.prompt_template).format(**variables)

    def render_chat_payload(self, tree: Any) -> Mapping[str, Any]:
        return render_chat_payload(
            model=str(self.config.model),
            messages=(
                {"role": "system", "content": str(self.config.system_prompt)},
                {"role": "user", "content": self.render_prompt(tree)},
            ),
            temperature=float(self.config.temperature),
            max_tokens=int(self.config.max_tokens),
        )

    def _artifact(self, *, kind: str, iteration: int, n_train: int) -> Mapping[str, Any]:
        return {
            "kind": f"treepo_llm_{kind}",
            "trained": str(kind),
            "iteration": int(iteration),
            "n_train": int(n_train),
            "model": str(self.config.model),
            "has_predict_fn": self.predict_fn is not None,
            "config": asdict(self.config),
        }

    def _call_predict_fn(self, *, prompt: str, tree: Any) -> Any:
        assert self.predict_fn is not None
        payload = self.render_chat_payload(tree)
        try:
            return self.predict_fn(prompt=prompt, tree=tree, messages=payload["messages"], config=self.config)
        except TypeError:
            try:
                return self.predict_fn(prompt, tree)
            except TypeError:
                return self.predict_fn(prompt)

    def _parse_prediction(self, value: Any) -> float | None:
        if isinstance(value, Mapping):
            for key in ("score", "prediction", "value", "text", "content"):
                if key in value:
                    parsed = self._parse_prediction(value[key])
                    if parsed is not None:
                        return parsed
            return None
        try:
            return self._clamp(float(value))
        except (TypeError, ValueError):
            pass
        match = re.search(str(self.config.score_regex), str(value))
        if match is None:
            return None
        try:
            return self._clamp(float(match.group(0)))
        except ValueError:
            return None

    def _clamp(self, value: float | None) -> float | None:
        if value is None:
            return None
        out = float(value)
        if self.config.min_score is not None:
            out = max(out, float(self.config.min_score))
        if self.config.max_score is not None:
            out = min(out, float(self.config.max_score))
        return out


def build_llm_family(backend_config: Mapping[str, Any]) -> PromptedLLMFamily:
    raw_config = dict(backend_config.get("llm_config") or backend_config.get("prompted_llm_config") or {})
    for key in (
        "model",
        "system_prompt",
        "prompt_template",
        "temperature",
        "max_tokens",
        "max_prompt_chars",
        "score_regex",
        "default_prediction",
        "min_score",
        "max_score",
        "metadata",
    ):
        if key in backend_config:
            raw_config[key] = backend_config[key]
    config = PromptedLLMFamilyConfig(**raw_config)
    predict_fn = (
        backend_config.get("predict_fn")
        or backend_config.get("llm_predict_fn")
        or backend_config.get("scorer")
    )
    if predict_fn is not None and not callable(predict_fn):
        raise TypeError("llm predict_fn must be callable")
    return PromptedLLMFamily(config=config, predict_fn=predict_fn)


def _prompt_variables(tree: Any) -> dict[str, Any]:
    meta = getattr(tree, "metadata", None)
    metadata = dict(meta) if isinstance(meta, Mapping) else {}
    text = _tree_text(tree)
    return {
        "text": text,
        "qsentence_guidance": _qsentence_guidance(metadata),
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
    }


def _qsentence_guidance(metadata: Mapping[str, Any]) -> str:
    rows = metadata.get("g_guidance_qsentences") or metadata.get("qsentence_guidance") or []
    rendered: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            score = row.get("guidance_score")
            score_text = "unknown" if score is None else str(score)
            rendered.append(
                f"- {row.get('qid', '')}: code={row.get('code', '')}, "
                f"guidance_score={score_text}, text={row.get('text', '')}"
            )
        else:
            rendered.append(f"- {row}")
    return "\n".join(rendered)


def _tree_text(tree: Any) -> str:
    meta = getattr(tree, "metadata", None)
    if isinstance(meta, Mapping):
        for key in ("text", "document", "content", "summary"):
            value = meta.get(key)
            if value is not None:
                return str(value)
    for attr in ("text", "document", "content", "summary"):
        value = getattr(tree, attr, None)
        if value is not None:
            return str(value)
    tokens = getattr(tree, "tokens", None)
    if tokens is not None:
        return " ".join(str(token) for token in list(tokens))
    leaves = getattr(tree, "leaves", None)
    if leaves:
        parts: list[str] = []
        for leaf in leaves:
            leaf_tokens = getattr(leaf, "tokens", None)
            if leaf_tokens is not None:
                parts.extend(str(token) for token in list(leaf_tokens))
        if parts:
            return " ".join(parts)
    return str(tree)


__all__ = [
    "PredictFn",
    "PromptedLLMFamily",
    "PromptedLLMFamilyConfig",
    "build_llm_family",
]
