"""Family-name registry for :func:`treepo.methods.fit`.

`treepo` keeps this registry small. Built-ins cover deterministic oracles,
a simple learnable baseline, classical sketches, generic neural operators,
and provider-neutral LLM/DSPy wrappers. TRL, diffusion/dgemma, and specialized
large-training families register from the package that owns their application
dependencies.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping


FamilyRuntime = Any
FamilyFactory = Callable[[Mapping[str, Any]], FamilyRuntime]

_REGISTRY: dict[str, FamilyFactory] = {}
_EXTENSION_FAMILIES = frozenset(
    {
        "trl",
        "diffusion",
        "dgemma",
        "diffusiongemma",
    }
)


def register_family(name: str, factory: FamilyFactory) -> None:
    """Register ``factory`` under ``name`` (case-insensitive)."""

    key = _normalize(name)
    if not key:
        raise ValueError("family name must be non-empty")
    if key in _REGISTRY:
        raise ValueError(f"family {name!r} already registered")
    _REGISTRY[key] = factory


def resolve_family(
    name: str,
    backend_config: Mapping[str, Any] | None = None,
) -> FamilyRuntime:
    """Construct a family runtime for ``name`` from ``backend_config``."""

    key = _normalize(name)
    if key not in _REGISTRY:
        if key in _EXTENSION_FAMILIES:
            raise ImportError(
                f"family {name!r} is optional application code. Register a "
                "family factory before resolving it."
            )
        raise KeyError(
            f"family {name!r} not registered; available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key](dict(backend_config or {}))


def list_families() -> tuple[str, ...]:
    """Return built-in registered family names, sorted."""

    return tuple(sorted(_REGISTRY))


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _make_oracle(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    oracle_name = backend_config.get("oracle_name") or backend_config.get("oracle")
    if not oracle_name:
        raise ValueError("family='oracle' requires backend_config['oracle_name']")
    from treepo.methods.oracles import OracleFamilyRuntime

    return OracleFamilyRuntime(str(oracle_name))


def _make_learnable_constant(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    from treepo.methods.learnable import LearnableConstantFamily

    return LearnableConstantFamily(
        gamma_depth=float(backend_config.get("law_gamma_depth", 1.0)),
    )


def _make_classical_sketch(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    from treepo.methods.sketch import build_classical_sketch_family

    return build_classical_sketch_family(backend_config)


def _make_neural_operator(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    from treepo.methods.fno import build_neural_operator_family

    return build_neural_operator_family(backend_config)


def _make_dspy(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    from treepo.methods.dspy import build_dspy_family

    return build_dspy_family(backend_config)


def _make_llm(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    from treepo.methods.llm import build_llm_family

    return build_llm_family(backend_config)


def _make_fno(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    from treepo.methods.fno import build_fno_family

    return build_fno_family(backend_config)


register_family("oracle", _make_oracle)
register_family("learnable_constant", _make_learnable_constant)
register_family("classical_sketch", _make_classical_sketch)
register_family("neural_operator", _make_neural_operator)
register_family("fno", _make_fno)
register_family("dspy", _make_dspy)
register_family("llm", _make_llm)
register_family("prompted_llm", _make_llm)
register_family("llm_prompt", _make_llm)
register_family("summary_llm", _make_llm)


__all__ = [
    "FamilyFactory",
    "FamilyRuntime",
    "list_families",
    "register_family",
    "resolve_family",
]
