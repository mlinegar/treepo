"""Public ``fit`` entrypoint for treepo learning runs.

Wraps a ``FitConfig`` (or mapping) and routes it through the methods learning
backend, returning a normalized ``FitResult``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from treepo.methods.contracts import FitResult


@dataclass(frozen=True)
class FitConfig:
    spec: Mapping[str, Any] = field(default_factory=dict)
    output_dir: str | Path = "outputs/treepo_fit"
    train_data: Any = None
    eval_data: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FitConfig":
        data = dict(payload or {})
        nested = data.get("fit")
        if isinstance(nested, Mapping):
            merged = dict(data)
            merged.update(dict(nested))
            data = merged
        spec = data.get("spec")
        if spec is None:
            spec = {
                k: v
                for k, v in data.items()
                if k not in {"output_dir", "metadata", "fit", "train_data", "eval_data"}
            }
        return cls(
            output_dir=data.get("output_dir") or "outputs/treepo_fit",
            spec=dict(spec or {}),
            train_data=data.get("train_data"),
            eval_data=data.get("eval_data"),
            metadata=dict(data.get("metadata") or {}),
        )


def _as_fit_config(config: FitConfig | Mapping[str, Any] | None, **kwargs: Any) -> FitConfig:
    payload: dict[str, Any] = {}
    if isinstance(config, FitConfig):
        payload = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    elif config is not None:
        raise TypeError(f"fit config must be a mapping or FitConfig, got {type(config).__name__}")
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return FitConfig.from_mapping(payload)


def fit(
    config: FitConfig | Mapping[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
    train_data: Any = None,
    eval_data: Any = None,
    **kwargs: Any,
) -> FitResult:
    """Run one methods learning spec through the public package wrapper."""

    cfg = _as_fit_config(
        config,
        output_dir=output_dir,
        train_data=train_data,
        eval_data=eval_data,
        **kwargs,
    )
    return _fit_learning(cfg)


def _fit_learning(cfg: FitConfig) -> FitResult:
    spec = dict(cfg.spec or {})
    if cfg.train_data is not None:
        spec["train_data"] = cfg.train_data
    if cfg.eval_data is not None:
        spec["eval_data"] = cfg.eval_data
    backend_config = dict(spec.get("backend_config") or {})
    backend_config.setdefault("output_dir", str(cfg.output_dir))
    spec["backend_config"] = backend_config

    from treepo.methods.contracts import CTreePOLearningSpec
    from treepo.methods.learning import fit as methods_fit

    if isinstance(spec, CTreePOLearningSpec):
        learning_spec = spec
    elif hasattr(CTreePOLearningSpec, "from_mapping"):
        learning_spec = CTreePOLearningSpec.from_mapping(spec)
    else:
        learning_spec = CTreePOLearningSpec(**spec)
    result = methods_fit(learning_spec)
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return FitResult(
        status=str(payload.get("status", "ok")),
        metrics=dict(payload.get("metrics") or {}),
        artifacts=dict(payload.get("artifacts") or {}),
        history=tuple(dict(item) for item in list(payload.get("history") or [])),
        summary=dict(payload.get("summary") or {}),
        manifest_path=payload.get("manifest_path"),
        mode="learning",
    )


__all__ = ["FitConfig", "FitResult", "fit"]
