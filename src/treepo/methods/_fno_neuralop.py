"""Torch/neuralop discovery for neural-operator families.

Machinery for locating the optional ``torch`` and ``neuralop`` dependencies and
turning an operator-kind string into a concrete model class plus constructor
kwargs. Kept separate so the rest of the FNO stack stays import-light and the
"which operator kinds are available" logic lives in one place.
"""

from __future__ import annotations

from typing import Any

from treepo.methods._fno_config import (
    _LOCAL_OPERATOR_KINDS,
    NeuralOperatorFamilyConfig,
    _normalize_operator_kind,
)

# Kinds that work with the leaf-sequence adapter, plus the explicitly rejected
# geometry/query kinds (kept so their rejection message names the real class).
_NEURALOP_KIND_ALIASES = {
    "fno": "FNO",
    "tfno": "TFNO",
    "uno": "UNO",
    "codano": "CODANO",
    "fno_gno": "FNOGNO",
    "fnogno": "FNOGNO",
    "gino": "GINO",
}
_SEQUENCE_INCOMPATIBLE_NEURALOP_KINDS = frozenset({"codano", "fno_gno", "fnogno", "gino"})
_SEQUENCE_COMPATIBLE_NEURALOP_KINDS = frozenset({"fno", "tfno", "uno"})


def _validate_operator_kind(operator_kind: str, *, family_name: str) -> None:
    if operator_kind in _LOCAL_OPERATOR_KINDS:
        return
    if _neuralop_model_class(operator_kind, required=False) is not None:
        return
    if not _neuralop_importable():
        # A kind that only looks unavailable because the optional dependency
        # is absent must say so — not claim the kind is unsupported.
        raise ImportError(
            f"operator_kind={operator_kind!r} needs the 'neuraloperator' package, "
            "which is not installed. Install the torch extra of this package: "
            "uv sync --extra torch (treepo[torch])."
        )
    supported = ", ".join(sorted((*_LOCAL_OPERATOR_KINDS, *_available_neuralop_kinds())))
    raise ValueError(
        f"family={family_name!r} does not support operator_kind={operator_kind!r}; "
        f"supported operator_kind values: {supported}"
    )


def _neuralop_importable() -> bool:
    try:
        import neuralop.models  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when deps absent
        raise ImportError(
            "neural-operator families require PyTorch, which is not installed. "
            "Install the torch extra of this package: "
            "uv sync --extra torch (treepo[torch])."
        ) from exc
    return torch


def _available_neuralop_kinds() -> tuple[str, ...]:
    try:
        import neuralop.models as models
    except ImportError:
        return tuple()
    names: list[str] = []
    for name in dir(models):
        if name.startswith("_"):
            continue
        obj = getattr(models, name)
        if isinstance(obj, type) and str(getattr(obj, "__module__", "")).startswith("neuralop"):
            names.append(_normalize_operator_kind(name))
    return tuple(sorted(set(names)))


def _neuralop_model_class(operator_kind: str, *, required: bool) -> Any:
    normalized = _normalize_operator_kind(operator_kind)
    try:
        import neuralop.models as models
    except ImportError as exc:  # pragma: no cover - exercised when deps absent
        if not required:
            return None
        raise ImportError(
            f"operator_kind={operator_kind!r} requires the 'neuraloperator' package, "
            "which is not installed. Install the torch extra of this package: "
            "uv sync --extra torch (treepo[torch])."
        ) from exc
    exact_name = _NEURALOP_KIND_ALIASES.get(normalized)
    if exact_name is not None and hasattr(models, exact_name):
        return getattr(models, exact_name)
    for name in dir(models):
        if _normalize_operator_kind(name) == normalized:
            obj = getattr(models, name)
            if isinstance(obj, type):
                return obj
    if required:
        supported = ", ".join(sorted((*_LOCAL_OPERATOR_KINDS, *_available_neuralop_kinds())))
        raise ValueError(
            f"operator_kind={operator_kind!r} is not available from neuralop.models; "
            f"supported operator_kind values: {supported}"
        )
    return None


def _neuralop_constructor_kwargs(
    *,
    operator_kind: str,
    config: NeuralOperatorFamilyConfig,
    model_cls: Any,
) -> dict[str, Any]:
    import inspect

    raw_kwargs = dict(config.operator_kwargs or {})
    signature = inspect.signature(model_cls)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    normalized = _normalize_operator_kind(operator_kind)
    if normalized in _SEQUENCE_INCOMPATIBLE_NEURALOP_KINDS:
        supported = ", ".join(sorted((*_LOCAL_OPERATOR_KINDS, *_SEQUENCE_COMPATIBLE_NEURALOP_KINDS)))
        raise ValueError(
            f"operator_kind={operator_kind!r} is available from neuralop.models, "
            "but treepo's built-in neural_operator family accepts one embedded "
            f"leaf-sequence tensor. Use one of {supported}, or register a "
            "downstream family for geometry/query neural operators."
        )
    in_channels = max(1, int(config.embedding_dim))
    hidden_channels = max(1, int(config.hidden_channels))
    n_layers = max(1, int(config.n_layers))
    n_modes = (max(1, int(config.n_modes)),)
    dense_defaults = {
        "in_channels": in_channels,
        "out_channels": hidden_channels,
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
        "n_modes": n_modes,
    }
    extended_defaults = {
        **dense_defaults,
        "fno_in_channels": in_channels,
        "fno_hidden_channels": hidden_channels,
        "fno_n_layers": n_layers,
        "fno_n_modes": n_modes,
    }
    if normalized == "uno":
        extended_defaults.update(
            {
                "lifting_channels": hidden_channels,
                "projection_channels": hidden_channels,
                "uno_out_channels": [hidden_channels] * n_layers,
                "uno_n_modes": [n_modes] * n_layers,
                "uno_scalings": [[1.0] * len(n_modes)] * n_layers,
                "horizontal_skips_map": {},
            }
        )
    kwargs = dict(dense_defaults) if accepts_kwargs else {
        key: value for key, value in extended_defaults.items() if key in signature.parameters
    }
    kwargs.update(raw_kwargs)
    required_missing = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if required_missing:
        missing = ", ".join(required_missing)
        raise ValueError(
            f"operator_kind={operator_kind!r} needs operator_kwargs for required "
            f"constructor argument(s): {missing}"
        )
    return kwargs


__all__ = [
    "_available_neuralop_kinds",
    "_neuralop_constructor_kwargs",
    "_neuralop_model_class",
    "_require_torch",
    "_validate_operator_kind",
]
