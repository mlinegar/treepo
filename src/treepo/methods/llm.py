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
from treepo.local_law import LawKind, LocalLawAuditRow
from treepo.methods._family_config import coerce_family_config
from treepo.state import state_to_dict

PredictFn = Callable[..., Any]


@dataclass(frozen=True)
class PromptedLLMFamilyConfig:
    model: str = "llm"
    api_base: str | None = None
    api_key: str = "EMPTY"
    timeout_seconds: float = 120.0
    verify_model: bool = True
    system_prompt: str = "You estimate the tree root statistic from the supplied document."
    prompt_template: str = "Return only one numeric score for this document.\n\n{text}\n\nScore:"
    temperature: float = 0.0
    max_tokens: int = 16
    max_prompt_chars: int = 4000
    score_regex: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    default_prediction: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    # Law auditing is on by default: fit() performs the same basic operations
    # for every family given the same input. Model-backed checks (C1 gold-leaf
    # readouts, C3 composed-vs-direct) cost one model call per row — set
    # audit_laws=False to keep only the call-free C2 identity check when that
    # per-iteration cost is prohibitive.
    audit_laws: bool = True
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

    def as_statistic(self, *, f: Any = None, g: Any = None) -> Any:
        if self.predict_fn is None and self.config.default_prediction is None:
            return None
        return _PromptedTextStatistic(family=self, f=f, g=g)

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
        config_payload = asdict(self.config)
        if config_payload.get("api_key"):
            config_payload["api_key"] = "<redacted>"
        return {
            "kind": f"treepo_llm_{kind}",
            "trained": str(kind),
            "iteration": int(iteration),
            "n_train": int(len(traces)),
            "model": str(self.config.model),
            "has_predict_fn": self.predict_fn is not None,
            "supervised_examples": examples,
            "config": config_payload,
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


class _PromptedTextStatistic:
    """ComposableStatistic surface for prompt-backed families.

    The composable state is document text: ``encode_leaf`` extracts leaf
    text, ``merge`` concatenates, ``readout`` prompts the model on the
    composed text. Law rows cost one model call per check: C1 audits leaf
    readouts against gold leaf scores where they exist, C3 compares the
    composed-text readout against the family's direct whole-tree prediction,
    and C2 is the exact (call-free) identity check that merging with empty
    text preserves the state. Predictors that cannot score a unit return
    ``None`` and that row is skipped rather than fabricated.
    """

    def __init__(self, *, family: "PromptedLLMFamily", f: Any = None, g: Any = None) -> None:
        self.family = family
        self._f = f
        self._g = g
        from treepo.statistic import StatisticInfo

        self.info = StatisticInfo(
            name=str(family.name),
            state_kind="prompted_text",
            exact=False,
            supports_local_laws=True,
            metadata={
                "model": str(family.config.model),
                "has_predict_fn": family.predict_fn is not None,
            },
        )

    def encode_leaf(self, leaf: Any) -> str:
        return _tree_text(leaf)

    def merge(self, left: Any, right: Any) -> str:
        parts = [str(part) for part in (left, right) if str(part or "").strip()]
        return "\n".join(parts)

    def readout(self, state: Any, query: Any = None) -> float | None:
        del query
        proxy = _TextOnlyTree(str(state or ""))
        predictions = self.family.score_roots_with_f(f=self._f, g=self._g, trees=[proxy])
        return predictions[0] if predictions else None

    def predict_tree(self, tree: Any) -> float | None:
        predictions = self.family.score_roots_with_f(f=self._f, g=self._g, trees=[tree])
        return predictions[0] if predictions else None

    def local_law_rows(
        self,
        units: Sequence[Any],
        *,
        query: Any = None,
        oracle: Any = None,
    ) -> Sequence[LocalLawAuditRow]:
        del query, oracle
        from treepo.schedule import merge_depths
        from treepo.tree import tree_leaves, tree_row_id

        rows: list[LocalLawAuditRow] = []
        for idx, tree in enumerate(list(units or ())):
            tree_id = tree_row_id(tree, idx, fallback_prefix="tree")
            base_metadata = {"statistic": self.info.name, "state_kind": self.info.state_kind}
            leaves = list(tree_leaves(tree) or ())
            texts = [self.encode_leaf(leaf) for leaf in leaves]
            state = ""
            for text in texts:
                state = self.merge(state, text)
            identity_holds = self.merge(state, "") == state and self.merge("", state) == state
            rows.append(
                LocalLawAuditRow(
                    row_id=f"{tree_id}:idempotence",
                    law_kind=LawKind.C2_IDEMPOTENCE,
                    proxy_loss=0.0 if identity_holds else 1.0,
                    oracle_loss=0.0 if identity_holds else 1.0,
                    observed=True,
                    propensity=1.0,
                    metadata={**base_metadata, "check": "empty_merge_identity", "law_facet": "c2_idempotence"},
                )
            )
            if not bool(self.family.config.audit_laws):
                continue
            depths = merge_depths(len(leaves), schedule="left_to_right") if leaves else []
            for leaf_idx, leaf in enumerate(leaves):
                score = getattr(leaf, "score", None)
                if score is None:
                    continue
                # Score the leaf object itself so predict_fns that read leaf
                # attributes/metadata see the real unit, not a text proxy.
                leaf_predictions = self.family.score_roots_with_f(
                    f=self._f, g=self._g, trees=[leaf]
                )
                prediction = leaf_predictions[0] if leaf_predictions else None
                if prediction is None:
                    continue
                loss = float((float(prediction) - float(score)) ** 2)
                rows.append(
                    LocalLawAuditRow(
                        row_id=f"{tree_id}:leaf:{leaf_idx}",
                        law_kind=LawKind.C1_LEAF,
                        proxy_loss=loss,
                        oracle_loss=loss,
                        observed=True,
                        propensity=1.0,
                        depth=int(depths[leaf_idx]) if leaf_idx < len(depths) else 0,
                        metadata={**base_metadata, "check": "gold_leaf_readout", "law_facet": "c1_sufficiency"},
                    )
                )
            composed = self.readout(state) if str(state).strip() else None
            direct = self.predict_tree(tree)
            if composed is not None and direct is not None:
                loss = float((float(composed) - float(direct)) ** 2)
                rows.append(
                    LocalLawAuditRow(
                        row_id=f"{tree_id}:composition",
                        law_kind=LawKind.C3_MERGE,
                        proxy_loss=loss,
                        oracle_loss=loss,
                        observed=True,
                        propensity=1.0,
                        metadata={**base_metadata, "check": "composed_vs_direct_readout", "law_facet": "c3b_compositionality"},
                    )
                )
        return tuple(rows)


class _TextOnlyTree:
    """Present composed text as a tree-like value for prompt rendering."""

    def __init__(self, text: str) -> None:
        self.text = str(text)
        self.metadata: dict[str, Any] = {}


def _select_kwargs(predict_fn: PredictFn, available: Mapping[str, Any]) -> dict[str, Any]:
    """Pick the ``available`` arguments a ``predict_fn`` actually accepts.

    Detection is signature-based, so a ``TypeError`` raised inside the callable
    propagates to the caller. A callable with ``**kwargs`` receives every
    available argument; otherwise only its named parameters (``prompt``,
    ``tree``, ``messages``, ``config``) are forwarded. When the signature
    cannot be introspected (e.g. a builtin/C callable), the callable receives
    the ``prompt``-only contract.
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
    config = coerce_family_config(
        PromptedLLMFamilyConfig,
        backend_config,
        nested_key="llm_config",
        aliases={"base_url": "api_base"},
    )
    predict_fn = resolve_prompted_predict_fn(
        config,
        backend_config,
        family_name="llm",
    )
    return PromptedLLMFamily(config=config, predict_fn=predict_fn)


def resolve_prompted_predict_fn(
    config: PromptedLLMFamilyConfig,
    backend_config: Mapping[str, Any],
    *,
    family_name: str,
) -> PredictFn | None:
    """Resolve a prompt family's predictor the same way for every route.

    Precedence: explicit ``predict_fn``, then an injected ``chat_client``,
    then an OpenAI-compatible client auto-built from ``api_base``.
    """

    predict_fn = backend_config.get("predict_fn")
    if predict_fn is not None and not callable(predict_fn):
        raise TypeError(f"{family_name} predict_fn must be callable")
    chat_client = backend_config.get("chat_client")
    if predict_fn is None and chat_client is not None:
        predict_fn = _chat_client_predict_fn(chat_client)
    if predict_fn is None and config.api_base:
        from treepo.llm import build_chat_client

        client = build_chat_client(
            api_base=config.api_base,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            session=backend_config.get("session"),
            verify_model=config.verify_model,
            default_temperature=config.temperature,
            default_max_tokens=config.max_tokens,
        )
        predict_fn = client.predict_text
    return predict_fn


def _chat_client_predict_fn(chat_client: Any) -> PredictFn:
    if callable(chat_client):
        return chat_client
    method = getattr(chat_client, "predict_text", None)
    if callable(method):
        return method
    method = getattr(chat_client, "complete_chat", None)
    if callable(method):
        def predict_from_complete_chat(*, messages, config=None, **kwargs):
            del kwargs
            return method(
                messages,
                temperature=getattr(config, "temperature", None),
                max_tokens=getattr(config, "max_tokens", None),
            )

        return predict_from_complete_chat
    raise TypeError(
        "backend_config['chat_client'] must be callable or expose predict_text()/complete_chat()"
    )


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
