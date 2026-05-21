"""Load a `TrainerConfig` from YAML / TOML / JSON.

Scalar fields (`n_epochs`, `learning_rate`, `model_name`, `mode`, ...) are
read straight off the mapping. Complex fields that can't round-trip through
text (`oracle`, `model`, `objective`, `trainer`, `feedback_fn`, `base_module`,
`run_spec`) may be specified as factory strings:

    oracle:
      factory: unified_g_v1.training.oracles.ManifestoRileTextOracle.from_path
      args:
        path: outputs/prep/manifesto_text

A factory string is `<module.path>:<attr>` or `<module.path>.<attr>`. The
loader resolves it with `importlib`, then calls it with `**args`. If a
caller wants to build objects in Python and then only load the scalar knobs,
they can pass those objects via `overrides` and they win over the file.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from treepo._research.unified_g_v1.training.tree_task import TrainerConfig


_COMPLEX_FIELDS = frozenset(
    {
        "oracle",
        "model",
        "objective",
        "trainer",
        "feedback_fn",
        "base_module",
        "run_spec",
        "supervision_dataset",
        "dspy_config",
        "trl_config",
        "optimizer_builder",
    }
)


def _resolve_factory(spec: str) -> Any:
    """Resolve `module.path:attr` or `module.path.attr` to an object."""
    if ":" in spec:
        module_path, attr_path = spec.split(":", 1)
    else:
        module_path, _, attr_path = spec.rpartition(".")
        if not module_path:
            raise ValueError(f"factory must include a module path: {spec!r}")
    obj = importlib.import_module(module_path)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _build_from_factory_spec(value: Any) -> Any:
    """If `value` is a `{factory: ..., args: {...}}` mapping, call it."""
    if isinstance(value, Mapping) and "factory" in value:
        factory = _resolve_factory(str(value["factory"]))
        args = dict(value.get("args") or {})
        return factory(**args)
    return value


def _read_mapping(path: str | Path) -> Mapping[str, Any]:
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in {".json"}:
        return json.loads(text)
    if suffix in {".toml"}:
        import tomllib

        return tomllib.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyyaml is required to load YAML configs; install `pyyaml`."
            ) from exc
        return yaml.safe_load(text)
    raise ValueError(f"unsupported config extension {suffix!r}; use .yaml/.toml/.json")


def load_trainer_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> TrainerConfig:
    """Load a `TrainerConfig` from disk.

    Scalars are read straight off the file. Complex fields may be given as
    factory specs (`{factory: "module:attr", args: {...}}`). Anything in
    `overrides` wins over the file.
    """
    raw = _read_mapping(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"config file must contain a mapping at the top level, got {type(raw).__name__}")

    known = {f.name for f in fields(TrainerConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            continue  # silently drop unknown keys so callers can add comments/metadata
        if key in _COMPLEX_FIELDS:
            kwargs[key] = _build_from_factory_spec(value)
        else:
            kwargs[key] = value
    if overrides:
        for key, value in overrides.items():
            if key in known:
                kwargs[key] = value
    return TrainerConfig(**kwargs)


__all__ = ["load_trainer_config"]
