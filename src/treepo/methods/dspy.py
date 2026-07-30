"""Optimizer-backed DSPy joint-vector family.

A declared ``oracle_targets`` catalog is copied into the family config by
:func:`treepo.fit`. DSPy compiles one readout program ``f`` and one shared
state program ``g`` over tagged leaf and merge calls; provider endpoints stay
configurable through ``lm_config``. ``optimizer="none"`` is the explicit
inference-only compatibility mode.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.methods._dspy_optimizer import DSPyOptimizerMixin
from treepo.methods._family_config import coerce_family_config
from treepo.methods.llm import (
    PromptedLLMFamily,
    PromptedLLMFamilyConfig,
    resolve_prompted_predict_fn,
)
from treepo.state import state_to_dict


@dataclass(frozen=True)
class DSPyFamilyConfig:
    lm_config: Mapping[str, Any] = field(default_factory=dict)
    # Task-neutral defaults are assembled from the declared target catalog.
    # Tasks may replace the prose without replacing the one f / one shared-g
    # program architecture.
    f_signature_instructions: str | None = None
    g_signature_instructions: str | None = None
    node_target_exclusive: bool = False
    root_weight: float = 1.0
    leaf_weight: float = 1.0
    merge_weight: float = 1.0
    f_record_source: str = "gold_state"
    prompt_template: str = (
        "Estimate the document-level target. Return only the requested output.\n\n"
        "Document:\n{text}\n\nSupervised examples:\n{supervised_examples}\n\nOutput:"
    )
    system_prompt: str = (
        "You are a DSPy program estimating tree root targets from supervised examples."
    )
    score_regex: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    default_prediction: Any = None
    min_score: float | None = None
    max_score: float | None = None
    max_prompt_chars: int = 4000
    # Ordered coordinates of one joint vector f*. ``treepo.fit`` populates
    # these fields from oracle_targets; K=1 uses this same contract.
    target_names: tuple[str, ...] = ()
    target_oracle_ids: tuple[str, ...] = ()
    target_dim: int | None = None
    target_key: str | None = None
    target_vector_key: str | None = None
    node_target_key: str | None = None
    target_min: float | None = None
    target_max: float | None = None
    # ``none`` is the explicit inference-only compatibility mode. Every
    # other value executes a DSPy optimizer compile and persists its program.
    optimizer: str = "bootstrap"
    optimizer_budget: str = "light"
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    # Whole-program DSPy persistence uses cloudpickle. State-only JSON remains
    # the default for the exact package-created dspy.Predict architecture.
    allow_pickle_program_load: bool = False
    validation_fraction: float = 0.2
    split_seed: int = 0
    # DAgger-style exposure matching for shared-g training. At rate zero
    # parent prompts retain their gold/reference child states exactly. At a
    # positive effective rate, selected child inputs come from the current
    # learned g's bottom-up states.
    g_scheduled_sampling_rate: float = 0.0
    g_scheduled_sampling_rate_start: float = 0.0
    g_scheduled_sampling_ramp_per_iter: float = 0.0
    # Optional historical no-truncation contract. It is deliberately inert
    # unless all four values are configured, so existing provider-neutral
    # package calls do not acquire a model-specific context-window default.
    leaf_size_tokens: int | None = None
    lm_context_window_tokens: int | None = None
    max_completion_tokens: int | None = None
    prompt_template_overhead_tokens: int | None = None
    min_propensity: float = 1e-8
    importance_weight_cap: float | None = None
    target_range: float | None = 1.0
    allow_identity_g_targets: bool = False
    g_target_source: str | None = None
    audit_laws: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in (self.target_names or ()))
        oracle_ids = tuple(str(value).strip() for value in (self.target_oracle_ids or ()))
        if any(not value for value in names):
            raise ValueError("target_names must contain only non-empty identifiers")
        if len(names) != len(set(names)):
            raise ValueError("target_names must be unique and preserve declared order")
        if any(not value for value in oracle_ids):
            raise ValueError("target_oracle_ids must contain only non-empty identifiers")
        if oracle_ids and len(oracle_ids) != len(names):
            raise ValueError("target_oracle_ids must align one-for-one with target_names")
        if names and not oracle_ids:
            raise ValueError("named vector targets require target_oracle_ids provenance")
        if not names and oracle_ids:
            raise ValueError("target_oracle_ids requires target_names")
        if names and self.target_dim is not None and int(self.target_dim) != len(names):
            raise ValueError(
                f"target_dim={self.target_dim!r} conflicts with len(target_names)={len(names)}"
            )
        if int(self.max_prompt_chars) <= 0:
            raise ValueError("max_prompt_chars must be positive")
        for name in ("f_signature_instructions", "g_signature_instructions"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} must be non-empty when configured")
            object.__setattr__(self, name, None if value is None else str(value).strip())
        if not isinstance(self.allow_pickle_program_load, bool):
            raise ValueError("allow_pickle_program_load must be a boolean")
        optimizer = str(self.optimizer or "").strip().lower()
        aliases = {
            "bootstrap_fewshot": "bootstrap",
            "bootstrap_random_search": "bootstrap_random_search",
            "miprov2": "mipro",
        }
        optimizer = aliases.get(optimizer, optimizer)
        if optimizer not in {
            "none",
            "bootstrap",
            "bootstrap_random_search",
            "mipro",
            "gepa",
        }:
            raise ValueError(
                "optimizer must be one of 'none', 'bootstrap', "
                "'bootstrap_random_search', 'mipro', or 'gepa'"
            )
        validation_fraction = float(self.validation_fraction)
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        for name in (
            "g_scheduled_sampling_rate",
            "g_scheduled_sampling_rate_start",
            "g_scheduled_sampling_ramp_per_iter",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0 and name != "g_scheduled_sampling_ramp_per_iter":
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
            object.__setattr__(self, name, value)
        _validate_no_truncation_budget(self)
        min_propensity = float(self.min_propensity)
        if not math.isfinite(min_propensity) or not 0.0 < min_propensity <= 1.0:
            raise ValueError("min_propensity must be finite and in (0, 1]")
        f_record_source = str(self.f_record_source or "").strip().lower()
        if f_record_source not in {"gold_state", "generated_when_available"}:
            raise ValueError("f_record_source must be 'gold_state' or 'generated_when_available'")
        role_weights = {
            name: float(getattr(self, name))
            for name in ("root_weight", "leaf_weight", "merge_weight")
        }
        if any(not math.isfinite(value) or value < 0.0 for value in role_weights.values()):
            raise ValueError(
                "root_weight, leaf_weight, and merge_weight must be finite and non-negative"
            )
        target_range = self.target_range
        if target_range is not None and (
            not math.isfinite(float(target_range)) or float(target_range) <= 0.0
        ):
            raise ValueError("target_range must be positive and finite")
        cap = self.importance_weight_cap
        if cap is not None and (not math.isfinite(float(cap)) or float(cap) <= 0.0):
            raise ValueError("importance_weight_cap must be positive and finite")
        lower = None if self.target_min is None else float(self.target_min)
        upper = None if self.target_max is None else float(self.target_max)
        if lower is not None and not math.isfinite(lower):
            raise ValueError("target_min must be finite")
        if upper is not None and not math.isfinite(upper):
            raise ValueError("target_max must be finite")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("target_min must not exceed target_max")
        object.__setattr__(self, "target_names", names)
        object.__setattr__(self, "target_oracle_ids", oracle_ids)
        if names:
            object.__setattr__(self, "target_dim", len(names))

        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "f_record_source", f_record_source)
        for name, value in role_weights.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "validation_fraction", validation_fraction)
        object.__setattr__(self, "min_propensity", min_propensity)
        object.__setattr__(
            self,
            "target_range",
            None if target_range is None else float(target_range),
        )
        object.__setattr__(
            self,
            "importance_weight_cap",
            None if cap is None else float(cap),
        )


def _validate_no_truncation_budget(config: DSPyFamilyConfig) -> None:
    names = (
        "leaf_size_tokens",
        "lm_context_window_tokens",
        "max_completion_tokens",
        "prompt_template_overhead_tokens",
    )
    raw = {name: getattr(config, name) for name in names}
    configured = [name for name, value in raw.items() if value is not None]
    if not configured:
        return
    if len(configured) != len(names):
        missing = [name for name in names if raw[name] is None]
        raise ValueError(f"DSPy no-truncation budget is all-or-none; missing {missing!r}")
    leaf = int(raw["leaf_size_tokens"])
    context = int(raw["lm_context_window_tokens"])
    completion = int(raw["max_completion_tokens"])
    overhead = int(raw["prompt_template_overhead_tokens"])
    if leaf <= 0 or context <= 0 or completion <= 0 or overhead < 0:
        raise ValueError(
            "DSPy no-truncation budgets require positive leaf/context/completion "
            "tokens and non-negative prompt overhead"
        )
    required = 2 * leaf
    if completion < required:
        raise RuntimeError(
            "DSPy two-child output budget is too small: "
            f"max_completion_tokens={completion} < 2 * leaf_size_tokens={required}"
        )
    available = context - completion - overhead
    if available < required:
        raise RuntimeError(
            "DSPy two-child input budget exceeds the LM context: "
            f"2 * leaf_size_tokens={required}, available input budget={available} "
            f"(lm_context_window_tokens={context} - "
            f"max_completion_tokens={completion} - "
            f"prompt_template_overhead_tokens={overhead})"
        )
    for name, value in (
        ("leaf_size_tokens", leaf),
        ("lm_context_window_tokens", context),
        ("max_completion_tokens", completion),
        ("prompt_template_overhead_tokens", overhead),
    ):
        object.__setattr__(config, name, value)


@dataclass(frozen=True)
class _DSPyPromptedConfig(PromptedLLMFamilyConfig):
    """Prompt config with the package's named-vector reporting schema."""

    target_names: tuple[str, ...] = ()
    target_oracle_ids: tuple[str, ...] = ()
    target_dim: int | None = None
    target_key: str | None = None
    target_vector_key: str | None = None
    node_target_key: str | None = None
    node_target_exclusive: bool = False
    root_weight: float = 1.0
    leaf_weight: float = 1.0
    merge_weight: float = 1.0
    target_min: float | None = None

    target_max: float | None = None


class _DSPyTrainingExample:
    """Small DSPy-compatible example used by injected and native compilers."""

    def __init__(self, **values: Any) -> None:
        self._values = dict(values)
        self._input_names: tuple[str, ...] = ()
        for key, value in self._values.items():
            setattr(self, key, value)

    def with_inputs(self, *names: str) -> "_DSPyTrainingExample":
        self._input_names = tuple(str(name) for name in names)
        return self

    def inputs(self) -> dict[str, Any]:
        return {name: self._values[name] for name in self._input_names}

    def toDict(self) -> dict[str, Any]:
        return dict(self._values)


class DSPyFamily(DSPyOptimizerMixin, PromptedLLMFamily):
    name = "dspy"

    def __init__(
        self,
        config: DSPyFamilyConfig | None = None,
        *,
        program: Any = None,
        f_program: Any = None,
        g_program: Any = None,
        compiler: Any = None,
        program_loader: Any = None,
        program_saver: Any = None,
        dspy_module: Any = None,
        predict_fn: Any = None,
        token_count_fn: Any = None,
    ) -> None:
        self.dspy_config = config or DSPyFamilyConfig()
        self.program = program
        self._init_dspy_optimizer_runtime(
            f_program=f_program,
            g_program=g_program,
            compiler=compiler,
            program_loader=program_loader,
            program_saver=program_saver,
            dspy_module=dspy_module,
            token_count_fn=token_count_fn,
        )
        lm = dict(self.dspy_config.lm_config or {})
        target_names = tuple(self.dspy_config.target_names or ())
        budget_completion = self.dspy_config.max_completion_tokens
        default_max_tokens = (
            int(budget_completion)
            if budget_completion is not None
            else 4096
            if target_names
            else 16
        )
        max_tokens = int(default_max_tokens if lm.get("max_tokens") is None else lm["max_tokens"])
        if max_tokens <= 0:
            raise ValueError("lm_config.max_tokens must be positive")
        if budget_completion is not None and max_tokens > int(budget_completion):
            raise ValueError(
                "lm_config.max_tokens cannot exceed the configured "
                "max_completion_tokens no-truncation reservation"
            )
        llm_config = _DSPyPromptedConfig(
            model=str(lm.get("model", "dspy")),
            api_base=lm.get("api_base"),
            api_key=str(lm.get("api_key", "EMPTY")),
            timeout_seconds=float(lm.get("timeout_seconds", 120.0) or 120.0),
            verify_model=bool(lm.get("verify_model", True)),
            system_prompt=str(self.dspy_config.system_prompt),
            prompt_template=str(self.dspy_config.prompt_template),
            temperature=float(lm.get("temperature", 0.0) or 0.0),
            max_tokens=max_tokens,
            max_prompt_chars=int(self.dspy_config.max_prompt_chars),
            score_regex=str(self.dspy_config.score_regex),
            default_prediction=self.dspy_config.default_prediction,
            min_score=self.dspy_config.min_score,
            max_score=self.dspy_config.max_score,
            target_names=target_names,
            target_oracle_ids=tuple(self.dspy_config.target_oracle_ids or ()),
            target_dim=self.dspy_config.target_dim,
            target_key=self.dspy_config.target_key,
            target_vector_key=self.dspy_config.target_vector_key,
            node_target_exclusive=bool(self.dspy_config.node_target_exclusive),
            root_weight=float(self.dspy_config.root_weight),
            leaf_weight=float(self.dspy_config.leaf_weight),
            merge_weight=float(self.dspy_config.merge_weight),
            node_target_key=self.dspy_config.node_target_key,
            target_min=self.dspy_config.target_min,
            target_max=self.dspy_config.target_max,
            audit_laws=bool(self.dspy_config.audit_laws),
            metadata=dict(self.dspy_config.metadata or {}),
        )
        super().__init__(llm_config, predict_fn=predict_fn or _program_predict_fn(program))

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[Any | None]:
        """Score all widths through the same parser, including defaults."""
        learned_artifact = isinstance(f, Mapping) and bool(f.get("program_path"))
        if self.dspy_config.optimizer != "none" or learned_artifact:
            learned = self._score_roots_with_dspy(f=f, g=g, trees=trees)
            if learned is not None:
                return learned

        out: list[Any | None] = []
        for tree in trees:
            prompt = self.render_prompt(tree, f=f, g=g)
            if self.predict_fn is None:
                raw = self.config.default_prediction
            else:
                raw = self._call_predict_fn(prompt=prompt, tree=tree, f=f, g=g)
            out.append(self._parse_prediction(raw))
        return out

    def render_prompt(self, tree: Any, *, f: Any = None, g: Any = None) -> str:
        prompt = super().render_prompt(tree, f=f, g=g)
        names = self._target_names()
        if not names:
            return prompt
        properties: dict[str, Any] = {}
        for name in names:
            field_schema: dict[str, Any] = {"type": "number"}
            if self.config.target_min is not None:
                field_schema["minimum"] = float(self.config.target_min)
            if self.config.target_max is not None:
                field_schema["maximum"] = float(self.config.target_max)
            properties[name] = field_schema
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(names),
            "additionalProperties": False,
        }
        return (
            f"{prompt}\n\nReturn exactly one JSON object matching this schema; "
            "do not omit, rename, or add coordinates:\n"
            f"{json.dumps(schema, separators=(',', ':'))}"
        )

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        if self.dspy_config.optimizer != "none":
            return self._train_dspy_f(
                f_init=f_init,
                g=g,
                traces=traces,
                output_dir=output_dir,
                iteration=iteration,
            )

        artifact = super().train_f(
            f_init=f_init,
            g=g,
            traces=traces,
            output_dir=output_dir,
            iteration=iteration,
        )
        return self._dspy_artifact(artifact)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        if self.dspy_config.optimizer != "none":
            return self._train_dspy_g(
                g_init=g_init,
                f=f,
                traces=traces,
                output_dir=output_dir,
                iteration=iteration,
            )

        outcome = super().train_g(
            g_init=g_init,
            f=f,
            traces=traces,
            output_dir=output_dir,
            iteration=iteration,
        )
        from treepo.methods.runtime import GTrainOutcome

        return GTrainOutcome(
            artifact=self._dspy_artifact(outcome.artifact),
            update_performed=outcome.update_performed,
            reason=outcome.reason,
        )

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if self._validate_dspy_artifact(kind=kind, artifact=artifact):
            return
        super().validate_artifact(kind=kind, artifact=artifact)

    def _artifact(
        self,
        *,
        kind: str,
        iteration: int,
        traces: Sequence[Any],
    ) -> Mapping[str, Any]:
        artifact = dict(
            super()._artifact(
                kind=kind,
                iteration=iteration,
                traces=traces,
            )
        )
        if self._target_names():
            try:
                artifact["supervised_examples"] = self._render_vector_examples(traces)
            except ValueError:
                if self.dspy_config.optimizer != "none":
                    raise
                artifact["supervised_examples"] = ""
                artifact["offline_missing_target_rows"] = True
            artifact["supervision_output_contract"] = "strict_named_vector"
        return artifact

    def _render_vector_examples(self, traces: Sequence[Any]) -> str:
        names = self._target_names()
        rendered: list[str] = []
        for index, trace in enumerate(traces):
            metadata = dict(getattr(trace, "metadata", None) or {})
            raw_target = None
            if self.config.target_vector_key:
                raw_target = metadata.get(str(self.config.target_vector_key))
                if raw_target is None:
                    raw_target = getattr(
                        trace,
                        str(self.config.target_vector_key),
                        None,
                    )
            if raw_target is None:
                raw_target = metadata.get("oracle_target")
            if raw_target is None:
                raw_target = getattr(trace, "root_label", None)
            target = self._parse_named_prediction(raw_target, names)
            if target is None:
                unit_id = (
                    metadata.get("preference_unit_id")
                    or metadata.get("doc_id")
                    or metadata.get("tree_id")
                    or getattr(trace, "tree_id", None)
                    or index
                )
                raise ValueError(
                    f"DSPy named-vector training trace {unit_id!r} does not "
                    "provide the exact declared target mapping"
                )
            unit_id = (
                metadata.get("preference_unit_id")
                or metadata.get("doc_id")
                or metadata.get("tree_id")
                or getattr(trace, "tree_id", None)
                or index
            )
            unit_type = metadata.get("preference_unit_type") or "unit"
            text = str(
                getattr(
                    trace,
                    "text",
                    getattr(trace, "content", ""),
                )
                or ""
            )
            target_json = json.dumps(
                state_to_dict(target),
                separators=(",", ":"),
            )
            rendered.append(f"- {unit_type}:{unit_id}, target={target_json}, text={text}")
        return "\n".join(rendered)

    def as_statistic(self, *, f: Any = None, g: Any = None) -> Any:
        # The generic prompted-text statistic is scalar. A vector DSPy program
        # may expose a faithful state/law statistic downstream; the package
        # must not silently average coordinates or create a K=1-only law path.
        if self._target_names():
            return None
        return super().as_statistic(f=f, g=g)

    def _parse_prediction(self, value: Any) -> Any | None:
        names = self._target_names()
        if not names:
            pairs = _mapping_pairs(value)
            if pairs is not None:
                by_name = {str(key): component for key, component in pairs}
                if "prediction_json" in by_name:
                    return super()._parse_prediction(by_name["prediction_json"])
            if hasattr(value, "prediction_json"):
                return super()._parse_prediction(getattr(value, "prediction_json"))
            return super()._parse_prediction(value)
        return self._parse_named_prediction(value, names)

    def _parse_named_prediction(
        self,
        value: Any,
        names: Sequence[str],
    ) -> dict[str, float] | None:
        declared = tuple(str(name) for name in names)
        if value is None:
            return None

        pairs = _mapping_pairs(value)
        if pairs is not None:
            normalized = [(str(key), component) for key, component in pairs]
            if len({key for key, _component in normalized}) != len(normalized):
                return None
            by_name = dict(normalized)
            if set(by_name) == set(declared):
                parsed: dict[str, float] = {}
                for name in declared:
                    number = self._vector_number(by_name[name])
                    if number is None:
                        return None
                    parsed[name] = number
                return parsed
            for key in (
                "prediction_by_target",
                "prediction_json",
                "vector_json",
                "scores_json",
                "scores",
                "values",
                "output",
                "text",
                "content",
            ):
                if key in by_name:
                    parsed = self._parse_named_prediction(by_name[key], declared)
                    if parsed is not None:
                        return parsed
            return None

        # Some provider/DSPy result wrappers expose fields as attributes but
        # do not implement Mapping or items().
        for attr in (
            "prediction_by_target",
            "prediction_json",
            "vector_json",
            "scores_json",
            "scores",
            "output",
            "text",
            "content",
        ):
            if not hasattr(value, attr):
                continue
            nested = getattr(value, attr)
            if nested is value:
                continue
            parsed = self._parse_named_prediction(nested, declared)
            if parsed is not None:
                return parsed

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return None
        if isinstance(value, (str, bytes)):
            text = value.decode() if isinstance(value, bytes) else value
            try:
                decoded = json.loads(
                    text.strip(),
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_strict_json_object,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
            return self._parse_named_prediction(decoded, declared)

        # A declared named catalog always uses the vector transport. In
        # particular, K=1 accepts only {"name": value} or its JSON-object
        # encoding, never [value], a bare scalar, or a scalar response field.
        return None

    def _vector_number(self, value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        if self.config.target_min is not None and number < float(self.config.target_min):
            return None
        if self.config.target_max is not None and number > float(self.config.target_max):
            return None
        return number

    def _target_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in (self.config.target_names or ()))

    def _dspy_artifact(self, artifact: Mapping[str, Any]) -> Mapping[str, Any]:
        out = dict(artifact)
        trained = str(out.get("trained", "g"))
        out["kind"] = f"treepo_dspy_{trained}"
        config_payload = {
            item.name: getattr(self.dspy_config, item.name) for item in fields(self.dspy_config)
        }
        lm_config = dict(config_payload.get("lm_config") or {})
        for key in ("lm", "current_lm"):
            if lm_config.get(key) is not None:
                lm_config[key] = f"<{type(lm_config[key]).__name__}>"
        if lm_config.get("api_key"):
            lm_config["api_key"] = "<redacted>"
        config_payload["lm_config"] = lm_config
        out["lm_config"] = lm_config
        out["dspy_config"] = config_payload
        out["has_program"] = self.program is not None
        out["target_names"] = list(self._target_names())
        out["output_contract"] = "strict_named_json" if self._target_names() else "scalar_numeric"
        return out


def build_dspy_family(backend_config: Mapping[str, Any]) -> DSPyFamily:
    config = coerce_family_config(
        DSPyFamilyConfig,
        backend_config,
        nested_key="dspy_config",
    )
    # Convenience flat keys seed the LM config the same way the llm route
    # seeds its client: model plus api_base (or its base_url alias).
    lm = dict(config.lm_config or {})
    if backend_config.get("dspy_lm") is not None:
        lm.setdefault("lm", backend_config["dspy_lm"])
    if backend_config.get("model") is not None:
        lm.setdefault("model", backend_config["model"])
    api_base = backend_config.get("api_base") or backend_config.get("base_url")
    if api_base is not None:
        lm.setdefault("api_base", api_base)
    if lm != dict(config.lm_config or {}):
        config = replace(config, lm_config=lm)
    program = _first_not_none(
        backend_config.get("dspy_program"),
        backend_config.get("program"),
    )
    nested_config = backend_config.get("dspy_config")
    explicit_optimizer = "optimizer" in backend_config or (
        isinstance(nested_config, Mapping) and "optimizer" in nested_config
    )
    if not explicit_optimizer and (
        program is not None or backend_config.get("predict_fn") is not None
    ):
        config = replace(config, optimizer="none")
    f_program = _first_not_none(
        backend_config.get("dspy_f_program"),
        backend_config.get("f_program"),
        program,
    )
    g_program = _first_not_none(
        backend_config.get("dspy_g_program"),
        backend_config.get("g_program"),
    )
    family = DSPyFamily(
        config=config,
        program=program,
        f_program=f_program,
        g_program=g_program,
        compiler=_first_not_none(
            backend_config.get("dspy_compiler"),
            backend_config.get("compiler"),
        ),
        program_loader=_first_not_none(
            backend_config.get("dspy_program_loader"),
            backend_config.get("program_loader"),
        ),
        program_saver=_first_not_none(
            backend_config.get("dspy_program_saver"),
            backend_config.get("program_saver"),
        ),
        dspy_module=backend_config.get("dspy_module"),
        predict_fn=None,
        token_count_fn=_first_not_none(
            backend_config.get("dspy_token_count_fn"),
            backend_config.get("token_count_fn"),
        ),
    )
    predict_fn = backend_config.get("predict_fn")
    if predict_fn is not None and not callable(predict_fn):
        raise TypeError("dspy predict_fn must be callable")
    if predict_fn is None:
        predict_fn = _program_predict_fn(program)
    if predict_fn is None:
        # Parity with the llm route: auto-build an OpenAI-compatible client
        # from lm_config's api_base when neither program nor predict_fn given.
        predict_fn = resolve_prompted_predict_fn(
            family.config,
            backend_config,
            family_name="dspy",
        )
    family.predict_fn = predict_fn
    return family


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _mapping_pairs(value: Any) -> list[tuple[Any, Any]] | None:
    """Return mapping-like pairs without requiring ``collections.abc.Mapping``."""

    if isinstance(value, Mapping):
        return list(value.items())
    items = getattr(value, "items", None)
    if callable(items):
        try:
            pairs = list(items())
        except (TypeError, ValueError):
            pairs = []
        if all(
            isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2
            for pair in pairs
        ):
            return [(pair[0], pair[1]) for pair in pairs]
    to_dict = getattr(value, "toDict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except (TypeError, ValueError):
            return None
        if isinstance(payload, Mapping):
            return list(payload.items())
    return None


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is invalid: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [str(key) for key, _value in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON object keys are invalid")
    return {str(key): value for key, value in pairs}


def _program_predict_fn(program: Any) -> Any:
    if program is None:
        return None
    if callable(program):
        return program
    for attr in ("predict", "forward"):
        method = getattr(program, attr, None)
        if callable(method):
            return method
    raise TypeError("dspy_program must be callable or expose predict()/forward()")


__all__ = ["DSPyFamily", "DSPyFamilyConfig", "build_dspy_family"]
