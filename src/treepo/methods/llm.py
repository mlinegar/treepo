"""Provider-neutral LLM family for :mod:`treepo.methods`.

This is deliberately small: it owns prompt/artifact plumbing and accepts an
optional injected ``predict_fn``. Concrete OpenAI/vLLM/DSPy clients can live in
application code and either pass that callable or replace the registered family.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from treepo.llm.openai_compatible import render_chat_payload
from treepo.state import state_to_dict

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
        artifact = self._artifact(kind="f", iteration=iteration, traces=traces)
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
        artifact = self._artifact(kind="g", iteration=iteration, traces=traces)
        self._last_g = artifact
        return artifact

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[float | None]:
        out: list[float | None] = []
        for tree in trees:
            prompt = self.render_prompt(tree, f=f, g=g)
            if self.predict_fn is None:
                out.append(self._clamp(self.config.default_prediction))
                continue
            raw = self._call_predict_fn(prompt=prompt, tree=tree, f=f, g=g)
            out.append(self._parse_prediction(raw))
        return out

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if artifact is None:
            return
        if not isinstance(artifact, Mapping):
            raise TypeError(f"llm {kind} artifact must be a mapping")

    def as_statistic(self, *, f: Any = None, g: Any = None) -> None:
        del f, g
        return None

    def render_prompt(self, tree: Any, *, f: Any = None, g: Any = None) -> str:
        variables = _prompt_variables(
            tree,
            f=f if f is not None else self._last_f,
            g=g if g is not None else self._last_g,
        )
        text = str(variables.get("text", ""))
        if len(text) > int(self.config.max_prompt_chars):
            variables["text"] = text[: int(self.config.max_prompt_chars)]
        return str(self.config.prompt_template).format(**variables)

    def render_chat_payload(self, tree: Any, *, f: Any = None, g: Any = None) -> Mapping[str, Any]:
        return render_chat_payload(
            model=str(self.config.model),
            messages=(
                {"role": "system", "content": str(self.config.system_prompt)},
                {"role": "user", "content": self.render_prompt(tree, f=f, g=g)},
            ),
            temperature=float(self.config.temperature),
            max_tokens=int(self.config.max_tokens),
        )

    def _artifact(self, *, kind: str, iteration: int, traces: Sequence[Any]) -> Mapping[str, Any]:
        examples = _render_supervised_examples(traces)
        return {
            "kind": f"treepo_llm_{kind}",
            "trained": str(kind),
            "iteration": int(iteration),
            "n_train": int(len(traces)),
            "model": str(self.config.model),
            "has_predict_fn": self.predict_fn is not None,
            "supervised_examples": examples,
            "config": asdict(self.config),
        }

    def _call_predict_fn(self, *, prompt: str, tree: Any, f: Any = None, g: Any = None) -> Any:
        assert self.predict_fn is not None
        payload = self.render_chat_payload(tree, f=f, g=g)
        available = {
            "prompt": prompt,
            "tree": tree,
            "messages": payload["messages"],
            "config": self.config,
        }
        kwargs = _select_kwargs(self.predict_fn, available)
        return self.predict_fn(**kwargs)

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


def _select_kwargs(predict_fn: PredictFn, available: Mapping[str, Any]) -> dict[str, Any]:
    """Pick the ``available`` arguments a ``predict_fn`` actually accepts.

    Detection is signature-based rather than exception-based so a genuine
    ``TypeError`` raised *inside* the callable propagates instead of being
    swallowed by arity retries. A callable with ``**kwargs`` receives every
    available argument; otherwise only its named parameters (``prompt``,
    ``tree``, ``messages``, ``config``) are forwarded. If the signature cannot
    be introspected (e.g. a builtin/C callable), fall back to the historical
    ``prompt``-only contract.
    """
    try:
        params = inspect.signature(predict_fn).parameters
    except (TypeError, ValueError):
        return {"prompt": available["prompt"]}
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(available)
    named = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    selected = {name: value for name, value in available.items() if name in named}
    return selected or {"prompt": available["prompt"]}


def build_llm_family(backend_config: Mapping[str, Any]) -> PromptedLLMFamily:
    raw_config = dict(backend_config.get("llm_config") or {})
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


def _prompt_variables(tree: Any, *, f: Any = None, g: Any = None) -> dict[str, Any]:
    meta = getattr(tree, "metadata", None)
    metadata = dict(meta) if isinstance(meta, Mapping) else {}
    text = _tree_text(tree)
    f_examples = _artifact_examples(f)
    g_examples = _artifact_examples(g)
    supervised_examples = "\n".join(item for item in (f_examples, g_examples) if item).strip()
    return {
        "text": text,
        "supervised_examples": supervised_examples,
        "f_supervised_examples": f_examples,
        "g_supervised_examples": g_examples,
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
    }


def _artifact_examples(artifact: Any) -> str:
    if isinstance(artifact, Mapping):
        return str(artifact.get("supervised_examples") or "")
    return ""


def _render_supervised_examples(traces: Sequence[Any]) -> str:
    rendered: list[str] = []
    for idx, trace in enumerate(traces):
        metadata = dict(getattr(trace, "metadata", None) or {})
        label = metadata.get("oracle_target", metadata.get("teacher_score_native"))
        label_text = "unknown" if label is None else _compact_json(label)
        unit_id = metadata.get("preference_unit_id") or metadata.get("doc_id") or metadata.get("tree_id") or idx
        unit_type = metadata.get("preference_unit_type") or "unit"
        text = str(getattr(trace, "text", getattr(trace, "content", "")) or "")
        rendered.append(f"- {unit_type}:{unit_id}, target={label_text}, text={text}")
    return "\n".join(rendered)


def _compact_json(value: Any) -> str:
    value = state_to_dict(value)
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return str(value)


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
