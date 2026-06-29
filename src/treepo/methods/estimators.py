"""Estimator registry for :mod:`treepo.methods`.

The public axis is an estimator axis, not a local-law axis. Today the default
estimator target is unified ``g`` because the alternating fit loop trains
``f`` and ``g`` artifacts. The registry is intentionally target-aware so the
same surface can later describe estimators for ``f`` or joint ``fg`` artifacts
without another naming pass.

Local-law supervision remains an objective/correction layer: an estimator may
expose leaf/merge/tree state surfaces, but C1/C2/C3 penalties are selected
through ``ObjectiveSpec`` or the training code that consumes local-law rows.

Built-ins are lightweight metadata/adapters:

* ``neural_operator``: generic neural-operator estimator, with
  ``operator_kind`` passed through to ``treepo.methods.neural_operator`` /
  ``neuralop``.
* ``fno``: FNO convenience specialization.
* ``conv1d``: tiny local sequence baseline using the neural-operator family.
* ``llm`` / ``prompted_llm``: bundled provider-neutral prompt estimators.
* ``dspy``: bundled provider-neutral route for injected DSPy programs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping


JsonDict = dict[str, Any]
EstimatorFactory = Callable[["EstimatorSpec", Mapping[str, Any]], "EstimatorDescriptor"]


@dataclass(frozen=True)
class EstimatorSpec:
    """User-facing selection of a method used to estimate a learned artifact."""

    name: str = "neural_operator"
    target: str = "g"
    backend_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "EstimatorSpec":
        if isinstance(value, EstimatorSpec):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, Mapping):
            data = dict(value)
            nested = _nested_estimator_payload(data)
            if nested is not None:
                return cls.from_value(nested)
            raw_name = (
                data.get("name")
                or data.get("estimator")
                or data.get("kind")
                or data.get("g_estimator")
                or "neural_operator"
            )
            if isinstance(raw_name, Mapping):
                raw_name = raw_name.get("name") or raw_name.get("kind") or "neural_operator"
            raw_target = (
                data.get("target")
                or data.get("role")
                or data.get("estimator_target")
                or ("g" if "g_estimator" in data else "g")
            )
            backend = dict(data.get("backend_config") or data.get("config") or {})
            reserved = {
                "name",
                "estimator",
                "kind",
                "target",
                "role",
                "estimator_target",
                "g_estimator",
                "backend_config",
                "config",
                "metadata",
            }
            for key, item in data.items():
                if str(key) not in reserved:
                    backend.setdefault(str(key), item)
            return cls(
                name=str(raw_name),
                target=_normalize_target(raw_target),
                backend_config=backend,
                metadata=dict(data.get("metadata") or {}),
            )
        raise TypeError(
            "estimator must be a string, mapping, or EstimatorSpec; "
            f"got {type(value).__name__}"
        )

    def to_dict(self) -> JsonDict:
        return {
            "name": _normalize(self.name),
            "target": _normalize_target(self.target),
            "backend_config": _jsonable(dict(self.backend_config or {})),
            "metadata": _jsonable(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class EstimatorDescriptor:
    """Resolved estimator contract.

    ``family`` is the public :mod:`treepo.methods` family that can train or run
    the estimator. Extension descriptors may point to families that downstream
    packages must register before use.
    """

    name: str
    family: str
    category: str
    artifact_kind: str
    target: str = "g"
    backend_config: Mapping[str, Any] = field(default_factory=dict)
    extension_required: bool = False
    supports_neural_operator: bool = False
    supports_llm: bool = False
    supports_tree_merge: bool = True
    supports_local_law_penalty: bool = True
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def apply_to_backend_config(self, backend_config: Mapping[str, Any] | None = None) -> JsonDict:
        """Return backend config with estimator-specific settings applied."""

        out = dict(backend_config or {})
        out.update(dict(self.backend_config or {}))
        descriptor = self.to_dict()
        out["estimator"] = descriptor
        if _normalize_target(self.target) == "g":
            out["g_estimator"] = descriptor
        return out

    def to_dict(self) -> JsonDict:
        return {
            "name": _normalize(self.name),
            "target": _normalize_target(self.target),
            "family": str(self.family),
            "category": str(self.category),
            "artifact_kind": str(self.artifact_kind),
            "backend_config": _jsonable(dict(self.backend_config or {})),
            "extension_required": bool(self.extension_required),
            "supports_neural_operator": bool(self.supports_neural_operator),
            "supports_llm": bool(self.supports_llm),
            "supports_tree_merge": bool(self.supports_tree_merge),
            "supports_local_law_penalty": bool(self.supports_local_law_penalty),
            "notes": str(self.notes),
            "metadata": _jsonable(dict(self.metadata or {})),
        }


_REGISTRY: dict[str, EstimatorFactory] = {}


def register_estimator(name: str, factory: EstimatorFactory) -> None:
    """Register an estimator descriptor factory under ``name``."""

    key = _normalize(name)
    if not key:
        raise ValueError("estimator name must be non-empty")
    if key in _REGISTRY:
        raise ValueError(f"estimator {name!r} already registered")
    _REGISTRY[key] = factory


def list_estimators() -> tuple[str, ...]:
    """Return registered estimator names, including aliases."""

    return tuple(sorted(_REGISTRY))


def resolve_estimator(
    value: str | Mapping[str, Any] | EstimatorSpec | None = None,
    backend_config: Mapping[str, Any] | None = None,
) -> EstimatorDescriptor:
    """Resolve a user estimator selection to a descriptor."""

    base = dict(backend_config or {})
    raw = value
    if raw is None:
        raw = base.get("estimator")
    if raw is None:
        raw = base.get("g_estimator")
    spec = EstimatorSpec.from_value(raw)
    key = _normalize(spec.name)
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown estimator {spec.name!r}; available: {', '.join(list_estimators())}"
        )
    return _REGISTRY[key](spec, base)


def _make_neural_operator(spec: EstimatorSpec, backend_config: Mapping[str, Any]) -> EstimatorDescriptor:
    backend = _merged_backend(spec, backend_config)
    backend.setdefault("operator_kind", backend.get("operator_kind") or "fno")
    operator_kind = _normalize(backend.get("operator_kind", "fno"))
    base_kind = "treepo_fno" if operator_kind == "fno" else "treepo_neural_operator"
    return EstimatorDescriptor(
        name=_normalize(spec.name),
        target=_normalize_target(spec.target),
        family="neural_operator",
        category="neural_operator",
        artifact_kind=_artifact_kind(base_kind, spec.target),
        backend_config=backend,
        supports_neural_operator=True,
        supports_llm=False,
        notes="Generic neural-operator estimator; operator_kind selects FNO/TFNO/UNO/etc.",
        metadata=dict(spec.metadata or {}),
    )


def _make_fno(spec: EstimatorSpec, backend_config: Mapping[str, Any]) -> EstimatorDescriptor:
    backend = _merged_backend(spec, backend_config)
    backend["operator_kind"] = "fno"
    return EstimatorDescriptor(
        name="fno",
        target=_normalize_target(spec.target),
        family="fno",
        category="neural_operator",
        artifact_kind=_artifact_kind("treepo_fno", spec.target),
        backend_config=backend,
        supports_neural_operator=True,
        supports_llm=False,
        notes="FNO specialization of the generic estimator axis.",
        metadata=dict(spec.metadata or {}),
    )


def _make_conv1d(spec: EstimatorSpec, backend_config: Mapping[str, Any]) -> EstimatorDescriptor:
    backend = _merged_backend(spec, backend_config)
    backend["operator_kind"] = "conv1d"
    return EstimatorDescriptor(
        name="conv1d",
        target=_normalize_target(spec.target),
        family="neural_operator",
        category="neural_operator",
        artifact_kind=_artifact_kind("treepo_neural_operator", spec.target),
        backend_config=backend,
        supports_neural_operator=True,
        supports_llm=False,
        notes="Small local sequence-operator baseline through the neural_operator family.",
        metadata=dict(spec.metadata or {}),
    )


def _make_llm(spec: EstimatorSpec, backend_config: Mapping[str, Any]) -> EstimatorDescriptor:
    backend = _merged_backend(spec, backend_config)
    family = str(backend.pop("family", "llm") or "llm")
    return EstimatorDescriptor(
        name=_normalize(spec.name),
        target=_normalize_target(spec.target),
        family=family,
        category="llm",
        artifact_kind=_artifact_kind("treepo_llm", spec.target),
        backend_config=backend,
        extension_required=False,
        supports_neural_operator=False,
        supports_llm=True,
        notes=(
            "Provider-neutral LLM/prompt estimator. Pass predict_fn for a concrete "
            "backend, or register a replacement family from application code."
        ),
        metadata=dict(spec.metadata or {}),
    )


def _make_dspy(spec: EstimatorSpec, backend_config: Mapping[str, Any]) -> EstimatorDescriptor:
    backend = _merged_backend(spec, backend_config)
    backend.setdefault("family", "dspy")
    descriptor = _make_llm(spec, backend)
    return EstimatorDescriptor(
        name="dspy",
        target=_normalize_target(spec.target),
        family="dspy",
        category=descriptor.category,
        artifact_kind=descriptor.artifact_kind,
        backend_config=dict(descriptor.backend_config),
        extension_required=False,
        supports_neural_operator=False,
        supports_llm=True,
        supports_tree_merge=descriptor.supports_tree_merge,
        supports_local_law_penalty=descriptor.supports_local_law_penalty,
        notes="Provider-neutral DSPy estimator route; pass dspy_program or predict_fn.",
        metadata=dict(spec.metadata or {}),
    )


def _nested_estimator_payload(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    has_direct_name = any(key in data for key in ("name", "kind"))
    if not has_direct_name and isinstance(data.get("estimator"), Mapping):
        nested = dict(data["estimator"])
        nested.setdefault("target", data.get("target") or data.get("role") or "g")
        return nested
    if not has_direct_name and "estimator" not in data and isinstance(data.get("g_estimator"), Mapping):
        nested = dict(data["g_estimator"])
        nested.setdefault("target", data.get("target") or data.get("role") or "g")
        return nested
    return None


def _merged_backend(spec: EstimatorSpec, backend_config: Mapping[str, Any]) -> JsonDict:
    out = dict(backend_config or {})
    out.pop("estimator", None)
    out.pop("g_estimator", None)
    out.update(dict(spec.backend_config or {}))
    return out


def _artifact_kind(base_kind: str, target: Any) -> str:
    normalized_target = _normalize_target(target)
    if not normalized_target:
        return base_kind
    return f"{base_kind}_{normalized_target}"


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _normalize_target(value: Any) -> str:
    target = _normalize(value or "g")
    if target in {"unified_g", "summary_g", "state_g"}:
        return "g"
    return target


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        try:
            return asdict(value)
        except Exception:
            return str(value)
    return value


register_estimator("neural_operator", _make_neural_operator)
register_estimator("general_neural_operator", _make_neural_operator)
register_estimator("operator", _make_neural_operator)
register_estimator("fno", _make_fno)
register_estimator("conv1d", _make_conv1d)
register_estimator("llm", _make_llm)
register_estimator("prompted_llm", _make_llm)
register_estimator("llm_prompt", _make_llm)
register_estimator("dspy", _make_dspy)


# Backwards-compatible unified-g aliases. Prefer the Estimator* names in new code.
GEstimatorDescriptor = EstimatorDescriptor
GEstimatorFactory = EstimatorFactory
GEstimatorSpec = EstimatorSpec
list_g_estimators = list_estimators
register_g_estimator = register_estimator
resolve_g_estimator = resolve_estimator


__all__ = [
    "EstimatorDescriptor",
    "EstimatorFactory",
    "EstimatorSpec",
    "GEstimatorDescriptor",
    "GEstimatorFactory",
    "GEstimatorSpec",
    "list_estimators",
    "list_g_estimators",
    "register_estimator",
    "register_g_estimator",
    "resolve_estimator",
    "resolve_g_estimator",
]
