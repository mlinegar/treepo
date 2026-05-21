"""Family-name registry for :func:`treepo.cld.fit`.

Maps short family names (``"oracle"`` / ``"fno"`` / ``"dspy"`` / ``"trl"``
/ ``"hll"`` / ``"count_min"``) to :class:`FamilyRuntime` instances built
from ``spec.backend_config``. Heavy deps (torch, dspy, trl) are imported
lazily inside the factory bodies; missing-dep failures surface as
``ImportError`` with an actionable install hint.

There is no inheritance, no plugin system, and no validator layer above
this dispatch. The registry is a plain dict and exists at a single site.
Adding a family means adding one factory function and one
``register_family`` call.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from treepo._research.ctreepo.alternating import FamilyRuntime


FamilyFactory = Callable[[Mapping[str, Any]], FamilyRuntime]

_REGISTRY: dict[str, FamilyFactory] = {}


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
    """Construct a :class:`FamilyRuntime` for ``name`` from ``backend_config``."""
    key = _normalize(name)
    if key not in _REGISTRY:
        raise KeyError(
            f"family {name!r} not registered; available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key](dict(backend_config or {}))


def list_families() -> tuple[str, ...]:
    """Return registered family names, sorted."""
    return tuple(sorted(_REGISTRY))


def _normalize(name: str) -> str:
    return str(name).strip().lower()


# --------------------------------------------------------------------------- #
# Built-in factories. Each one:
#   - validates the small backend_config slice it needs
#   - imports its heavy deps lazily (torch / dspy / trl)
#   - raises ImportError or ValueError with an install/usage hint
# --------------------------------------------------------------------------- #


def _make_oracle(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    oracle_name = backend_config.get("oracle_name") or backend_config.get("oracle")
    if not oracle_name:
        raise ValueError(
            "family='oracle' requires backend_config['oracle_name'] (e.g. "
            "'hll_exact', 'markov_changepoint_count', 'lda_leaf_local_mixture', "
            "'type_oracle')"
        )
    from treepo._research.ctreepo.oracles.runtime import OracleFamilyRuntime

    return OracleFamilyRuntime(str(oracle_name))


def _make_fno(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    try:
        from treepo._research.ctreepo.fno_family import FNOFamily, FNOFamilyConfig
    except ImportError as exc:
        raise ImportError(
            "family='fno' requires torch (and the FNO stack): "
            "install with `pip install treepo[torch]`"
        ) from exc
    config = backend_config.get("fno_config")
    if config is None:
        raise ValueError(
            "family='fno' requires backend_config['fno_config'] "
            "(FNOFamilyConfig instance or mapping)"
        )
    if not isinstance(config, FNOFamilyConfig):
        config = FNOFamilyConfig(**dict(config))
    embedding_client = backend_config.get("embedding_client")
    if embedding_client is None:
        raise ValueError("family='fno' requires backend_config['embedding_client']")
    device = backend_config.get("device")
    return FNOFamily(config=config, embedding_client=embedding_client, device=device)


def _make_dspy(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    try:
        from treepo._research.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig
    except ImportError as exc:
        raise ImportError(
            "family='dspy' requires dspy: install with `pip install dspy>=3.0.0`"
        ) from exc
    config = backend_config.get("dspy_config")
    if config is None:
        raise ValueError(
            "family='dspy' requires backend_config['dspy_config'] "
            "(DSPyFamilyConfig instance or mapping)"
        )
    if not isinstance(config, DSPyFamilyConfig):
        config = DSPyFamilyConfig(**dict(config))
    return DSPyFamily(config=config)


def _make_trl(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    try:
        from treepo._research.ctreepo.trl_family import TRLFamily, TRLFamilyConfig
    except ImportError as exc:
        raise ImportError(
            "family='trl' requires trl/transformers: install with "
            "`pip install trl transformers`"
        ) from exc
    config = backend_config.get("trl_config")
    if config is None:
        raise ValueError(
            "family='trl' requires backend_config['trl_config'] "
            "(TRLFamilyConfig instance or mapping)"
        )
    if not isinstance(config, TRLFamilyConfig):
        config = TRLFamilyConfig(**dict(config))
    return TRLFamily(config=config)


def _make_classical_sketch(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    """Factory for any classical sketch wrapped in ``ClassicalSketchFamilyRuntime``.

    Expects ``backend_config["sketch_adapter"]`` (a :class:`SketchAdapter`)
    and optionally ``backend_config["sketch_schedule"]`` (default ``"balanced"``)
    and ``backend_config["leaf_items_fn"]`` (callable mapping ``tree -> list[Iterable]``).
    """
    from treepo.cld.sketch_family import ClassicalSketchFamilyRuntime

    adapter = backend_config.get("sketch_adapter")
    if adapter is None:
        raise ValueError(
            "family='sketch' requires backend_config['sketch_adapter'] "
            "(a treepo.sketches.protocol.SketchAdapter instance)"
        )
    return ClassicalSketchFamilyRuntime(
        adapter=adapter,
        schedule=str(backend_config.get("sketch_schedule", "balanced")),
        leaf_items_fn=backend_config.get("leaf_items_fn"),
    )


def _make_learnable_constant(backend_config: Mapping[str, Any]) -> FamilyRuntime:
    from treepo.cld.learnable import LearnableConstantFamily

    return LearnableConstantFamily(
        gamma_depth=float(backend_config.get("law_gamma_depth", 1.0)),
    )


register_family("oracle", _make_oracle)
register_family("fno", _make_fno)
register_family("dspy", _make_dspy)
register_family("trl", _make_trl)
# One sketch family handles every classical sketch — sketch_kind varies
# via backend_config['sketch_adapter']. No per-kind aliases.
register_family("sketch", _make_classical_sketch)
# Demonstrative family that exercises the local-law arithmetic as an
# in-loop training signal (see treepo.cld/learnable.py).
register_family("learnable_constant", _make_learnable_constant)


__all__ = [
    "list_families",
    "register_family",
    "resolve_family",
]
