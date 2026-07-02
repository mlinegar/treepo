"""Public ``fit`` entrypoint for treepo learning runs.

Normalizes a mapping config and routes it through the methods learning
backend, returning a normalized ``FitResult``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from treepo.methods.contracts import FitResult

_NON_SPEC_KEYS = {"output_dir", "metadata", "train_data", "eval_data"}


def fit(
    config: Mapping[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
    train_data: Any = None,
    eval_data: Any = None,
    **kwargs: Any,
) -> FitResult:
    """Run one methods learning spec through the public package wrapper."""

    if config is None:
        payload: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        raise TypeError(f"fit config must be a mapping, got {type(config).__name__}")
    overrides = {"output_dir": output_dir, "train_data": train_data, "eval_data": eval_data}
    payload.update({k: v for k, v in overrides.items() if v is not None})
    payload.update({k: v for k, v in kwargs.items() if v is not None})

    spec = payload.get("spec")
    if spec is None:
        spec = {k: v for k, v in payload.items() if k not in _NON_SPEC_KEYS}
    spec = dict(spec or {})
    if payload.get("train_data") is not None:
        spec["train_data"] = payload["train_data"]
    if payload.get("eval_data") is not None:
        spec["eval_data"] = payload["eval_data"]
    backend_config = dict(spec.get("backend_config") or {})
    backend_config.setdefault("output_dir", str(payload.get("output_dir") or "outputs/treepo_fit"))
    spec["backend_config"] = backend_config

    from treepo.methods.contracts import CTreePOLearningSpec
    from treepo.methods.learning import fit as methods_fit

    return methods_fit(CTreePOLearningSpec.from_mapping(spec))


__all__ = ["FitResult", "fit"]
