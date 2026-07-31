"""Package-wide model-artifact identity and cross-view reuse."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from treepo.common import jsonable
from treepo.methods._runtime_loop import canonical_g_artifact
from treepo.methods.contracts import G_MODE_IDENTITY, normalize_g_mode

ALIGNMENT_INDEPENDENT = "independent"
ALIGNMENT_REUSE = "reuse"
ALIGNMENT_MODES = (ALIGNMENT_INDEPENDENT, ALIGNMENT_REUSE)

G_POLICY_SOURCE = "source"
G_POLICY_IDENTITY = "identity"
G_POLICIES = (G_POLICY_SOURCE, G_POLICY_IDENTITY)

_ALIGNMENT_FIELDS = frozenset(
    {
        "mode",
        "scope_id",
        "source_id",
        "source_pair_digest",
        "source_f_digest",
        "source_g_digest",
        "g_policy",
    }
)


def artifact_digest(artifact: Any) -> str:
    """Digest one model artifact's canonical JSON projection.

    Opaque extension objects contribute their qualified type and ``repr``.
    """

    payload = jsonable(artifact)
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_artifact_json_fallback,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def model_artifact_pair_digest(f_artifact: Any, g_artifact: Any) -> str:
    return artifact_digest({"f": f_artifact, "g": g_artifact})


def align_model_artifacts(
    config: Mapping[str, Any],
    source: Any,
    *,
    scope_id: str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Return an evaluation-only fit config aligned to a source model pair.

    The source ``f`` is always reused exactly. Nonidentity views reuse the
    source ``g`` exactly; identity views use the package's canonical identity
    artifact while retaining the same source ``f``.
    """

    payload = dict(config or {})
    if payload.get("initial_artifacts"):
        raise ValueError("aligned fits reject pre-existing initial_artifacts")
    if payload.get("artifact_alignment"):
        raise ValueError("aligned fits reject a pre-existing artifact_alignment")
    source_artifacts = _source_artifacts(source)
    f_artifact = source_artifacts.get("f")
    g_artifact = source_artifacts.get("g")
    if f_artifact is None or g_artifact is None:
        raise ValueError("artifact source must expose non-null f and g artifacts")

    axis = dict(payload.get("axis") or {})
    configured_iterations = axis.get("max_iterations")
    if configured_iterations is not None and int(configured_iterations) != 0:
        raise ValueError("aligned artifact reuse requires axis.max_iterations=0")
    axis["max_iterations"] = 0
    payload["axis"] = axis

    g_mode = normalize_g_mode(payload.get("g_mode"))
    g_policy = G_POLICY_IDENTITY if g_mode == G_MODE_IDENTITY else G_POLICY_SOURCE
    initial = {"f": f_artifact}
    if g_policy == G_POLICY_SOURCE:
        initial["g"] = g_artifact
    payload["initial_artifacts"] = initial

    pair_digest = model_artifact_pair_digest(f_artifact, g_artifact)
    source_contract = _source_contract(source)
    resolved_scope = str(scope_id or source_contract.get("scope_id") or f"model:{pair_digest}")
    _validate_source_contract(source_contract, f_artifact=f_artifact, g_artifact=g_artifact)
    resolved_source = str(source_id or source_contract.get("source_id") or resolved_scope)
    payload["artifact_alignment"] = {
        "mode": ALIGNMENT_REUSE,
        "scope_id": resolved_scope,
        "source_id": resolved_source,
        "source_pair_digest": pair_digest,
        "source_f_digest": artifact_digest(f_artifact),
        "source_g_digest": artifact_digest(g_artifact),
        "g_policy": g_policy,
    }
    return payload


def validate_artifact_alignment_before_fit(spec: Any) -> dict[str, Any]:
    """Normalize and fail closed on invalid reuse before any family trains."""

    return _normalized_alignment(spec, validate_result=None)


def build_model_artifact_contract(
    *,
    spec: Any,
    artifacts: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Validate final artifacts and return the universal alignment block."""

    final = {"f": artifacts.get("f"), "g": artifacts.get("g")}
    alignment = _normalized_alignment(spec, validate_result=final)
    pair_digest = model_artifact_pair_digest(final["f"], final["g"])
    f_digest = artifact_digest(final["f"])
    g_digest = artifact_digest(final["g"])
    mode = alignment["mode"]
    source_pair_digest = alignment.get("source_pair_digest") or pair_digest
    scope_id = str(alignment.get("scope_id") or f"model:{source_pair_digest}")
    source_id = str(alignment.get("source_id") or output_dir)
    g_policy = alignment["g_policy"]
    reused = mode == ALIGNMENT_REUSE
    return {
        "definition": "one_f_with_explicit_g_policy",
        "universal_execution": "f(reduce_g(T))",
        "mode": mode,
        "scope_id": scope_id,
        "source_id": source_id,
        "source_pair_digest": source_pair_digest,
        "source_f_digest": alignment.get("source_f_digest") or f_digest,
        "source_g_digest": alignment.get("source_g_digest") or g_digest,
        "pair_digest": pair_digest,
        "f_digest": f_digest,
        "g_digest": g_digest,
        "f_alignment": "exact_source_artifact" if reused else "fit_output",
        "g_alignment": (
            "canonical_identity"
            if g_policy == G_POLICY_IDENTITY
            else "exact_source_artifact"
            if reused
            else "fit_output"
        ),
        "g_policy": g_policy,
        "evaluation_only": reused,
    }


def _normalized_alignment(
    spec: Any,
    *,
    validate_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = dict(getattr(spec, "artifact_alignment", None) or {})
    unknown = sorted(set(raw) - _ALIGNMENT_FIELDS)
    if unknown:
        raise ValueError(f"artifact_alignment has unsupported fields {unknown!r}")
    initial = dict(getattr(spec, "initial_artifacts", None) or {})
    axis = dict(getattr(spec, "axis", None) or {})
    max_iterations = int(axis.get("max_iterations", 2))
    g_mode = normalize_g_mode(getattr(spec, "g_mode", None))

    if raw:
        mode = str(raw.get("mode") or ALIGNMENT_INDEPENDENT).strip().lower()
        g_policy = str(raw.get("g_policy") or G_POLICY_SOURCE).strip().lower()
        alignment = {**raw, "mode": mode, "g_policy": g_policy}
    elif max_iterations == 0 and initial.get("f") is not None:
        g_policy = G_POLICY_IDENTITY if g_mode == G_MODE_IDENTITY else G_POLICY_SOURCE
        source_g = (
            canonical_g_artifact(G_MODE_IDENTITY)
            if g_policy == G_POLICY_IDENTITY
            else initial.get("g")
        )
        source_pair_digest = model_artifact_pair_digest(initial.get("f"), source_g)
        alignment = {
            "mode": ALIGNMENT_REUSE,
            "scope_id": f"model:{source_pair_digest}",
            "source_id": "initial_artifacts",
            "source_pair_digest": source_pair_digest,
            "source_f_digest": artifact_digest(initial.get("f")),
            "source_g_digest": artifact_digest(source_g),
            "g_policy": g_policy,
        }
    else:
        alignment = {
            "mode": ALIGNMENT_INDEPENDENT,
            "g_policy": G_POLICY_IDENTITY if g_mode == G_MODE_IDENTITY else G_POLICY_SOURCE,
        }

    mode = alignment["mode"]
    g_policy = alignment["g_policy"]
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"artifact_alignment.mode must be one of {ALIGNMENT_MODES!r}")
    if g_policy not in G_POLICIES:
        raise ValueError(f"artifact_alignment.g_policy must be one of {G_POLICIES!r}")
    if mode != ALIGNMENT_REUSE:
        return alignment
    required = (
        "scope_id",
        "source_id",
        "source_pair_digest",
        "source_f_digest",
        "source_g_digest",
    )
    missing = [field for field in required if not str(alignment.get(field) or "").strip()]
    if missing:
        raise ValueError(f"artifact_alignment reuse is missing required fields {missing!r}")
    if max_iterations != 0:
        raise ValueError("artifact-aligned reuse requires axis.max_iterations=0")
    if initial.get("f") is None:
        raise ValueError("artifact-aligned reuse requires initial_artifacts['f']")
    if artifact_digest(initial["f"]) != alignment["source_f_digest"]:
        raise ValueError("initial f artifact does not match the aligned source digest")
    if g_policy == G_POLICY_IDENTITY:
        if g_mode != G_MODE_IDENTITY:
            raise ValueError("artifact g_policy='identity' requires g_mode='identity'")
        if initial.get("g") is not None:
            raise ValueError("identity-aligned reuse must omit initial_artifacts['g']")
    else:
        if g_mode == G_MODE_IDENTITY:
            raise ValueError("g_mode='identity' requires artifact g_policy='identity'")
        if initial.get("g") is None:
            raise ValueError("source-g reuse requires initial_artifacts['g']")
        if artifact_digest(initial["g"]) != alignment["source_g_digest"]:
            raise ValueError("initial g artifact does not match the aligned source digest")
        if (
            model_artifact_pair_digest(initial["f"], initial["g"])
            != alignment["source_pair_digest"]
        ):
            raise ValueError("initial f/g pair does not match the aligned source pair digest")

    if validate_result is not None:
        if not _artifacts_equivalent(validate_result.get("f"), initial.get("f")):
            raise ValueError("artifact-aligned view changed the exact source f artifact")
        if g_policy == G_POLICY_SOURCE:
            if not _artifacts_equivalent(validate_result.get("g"), initial.get("g")):
                raise ValueError("artifact-aligned view changed the exact source g artifact")
        elif not _artifacts_equivalent(
            validate_result.get("g"), canonical_g_artifact(G_MODE_IDENTITY)
        ):
            raise ValueError("identity-aligned view did not execute canonical identity g")
    return alignment


def _source_artifacts(source: Any) -> dict[str, Any]:
    artifacts = getattr(source, "artifacts", source)
    if not isinstance(artifacts, Mapping):
        raise TypeError("artifact source must be a FitResult or artifact mapping")
    return {"f": artifacts.get("f"), "g": artifacts.get("g")}


def _source_contract(source: Any) -> dict[str, Any]:
    summary = getattr(source, "summary", None)
    if not isinstance(summary, Mapping):
        return {}
    value = summary.get("model_artifact_contract")
    return dict(value) if isinstance(value, Mapping) else {}


def _validate_source_contract(
    contract: Mapping[str, Any],
    *,
    f_artifact: Any,
    g_artifact: Any,
) -> None:
    if not contract:
        return
    expected = {
        "pair_digest": model_artifact_pair_digest(f_artifact, g_artifact),
        "f_digest": artifact_digest(f_artifact),
        "g_digest": artifact_digest(g_artifact),
    }
    for field, digest in expected.items():
        declared = contract.get(field)
        if declared is not None and str(declared) != digest:
            raise ValueError(f"artifact source {field} disagrees with its current model artifacts")


def _artifacts_equivalent(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        comparison = left == right
    except Exception:
        return False
    if isinstance(comparison, bool):
        return comparison
    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return False


def _artifact_json_fallback(value: Any) -> dict[str, str]:
    value_type = type(value)
    return {
        "python_type": f"{value_type.__module__}.{value_type.__qualname__}",
        "repr": repr(value),
    }


__all__ = [
    "ALIGNMENT_INDEPENDENT",
    "ALIGNMENT_MODES",
    "ALIGNMENT_REUSE",
    "G_POLICIES",
    "G_POLICY_IDENTITY",
    "G_POLICY_SOURCE",
    "align_model_artifacts",
    "artifact_digest",
    "build_model_artifact_contract",
    "model_artifact_pair_digest",
    "validate_artifact_alignment_before_fit",
]
