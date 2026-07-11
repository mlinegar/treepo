"""Provider-neutral DSPy family wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.methods._family_config import coerce_family_config
from treepo.methods.llm import (
    PromptedLLMFamily,
    PromptedLLMFamilyConfig,
    resolve_prompted_predict_fn,
)


@dataclass(frozen=True)
class DSPyFamilyConfig:
    lm_config: Mapping[str, Any] = field(default_factory=dict)
    prompt_template: str = (
        "Estimate the document-level score. Return only one number.\n\n"
        "Document:\n{text}\n\nSupervised examples:\n{supervised_examples}\n\nScore:"
    )
    system_prompt: str = "You are a DSPy program estimating tree root labels from supervised examples."
    score_regex: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    default_prediction: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    audit_laws: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DSPyFamily(PromptedLLMFamily):
    name = "dspy"

    def __init__(
        self,
        config: DSPyFamilyConfig | None = None,
        *,
        program: Any = None,
        predict_fn: Any = None,
    ) -> None:
        self.dspy_config = config or DSPyFamilyConfig()
        self.program = program
        lm = dict(self.dspy_config.lm_config or {})
        llm_config = PromptedLLMFamilyConfig(
            model=str(lm.get("model", "dspy")),
            api_base=lm.get("api_base"),
            api_key=str(lm.get("api_key", "EMPTY")),
            timeout_seconds=float(lm.get("timeout_seconds", 120.0) or 120.0),
            verify_model=bool(lm.get("verify_model", True)),
            system_prompt=str(self.dspy_config.system_prompt),
            prompt_template=str(self.dspy_config.prompt_template),
            temperature=float(lm.get("temperature", 0.0) or 0.0),
            max_tokens=int(lm.get("max_tokens", 16) or 16),
            score_regex=str(self.dspy_config.score_regex),
            default_prediction=self.dspy_config.default_prediction,
            min_score=self.dspy_config.min_score,
            max_score=self.dspy_config.max_score,
            audit_laws=bool(self.dspy_config.audit_laws),
            metadata=dict(self.dspy_config.metadata or {}),
        )
        super().__init__(llm_config, predict_fn=predict_fn or _program_predict_fn(program))

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
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
    ) -> Mapping[str, Any]:
        artifact = super().train_g(
            g_init=g_init,
            f=f,
            traces=traces,
            output_dir=output_dir,
            iteration=iteration,
        )
        return self._dspy_artifact(artifact)

    def _dspy_artifact(self, artifact: Mapping[str, Any]) -> Mapping[str, Any]:
        out = dict(artifact)
        trained = str(out.get("trained", "g"))
        out["kind"] = f"treepo_dspy_{trained}"
        out["lm_config"] = dict(self.dspy_config.lm_config or {})
        out["dspy_config"] = asdict(self.dspy_config)
        out["has_program"] = self.program is not None
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
    if backend_config.get("model") is not None:
        lm.setdefault("model", backend_config["model"])
    api_base = backend_config.get("api_base") or backend_config.get("base_url")
    if api_base is not None:
        lm.setdefault("api_base", api_base)
    if lm != dict(config.lm_config or {}):
        from dataclasses import replace

        config = replace(config, lm_config=lm)
    program = backend_config.get("dspy_program") or backend_config.get("program")
    family = DSPyFamily(config=config, program=program, predict_fn=None)
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
