"""Named supervision levels for per-node training (Phase 1 of the fit-grid plan).

A supervision level is a named cell in the supervision grid: one concrete
assignment of the relative ``root`` / ``leaf`` / ``merge`` node weights the
neural-operator families consume. The level names and weight values mirror the
ThinkingTrees ladder's ``--supervision`` vocabulary exactly, so a cell trained
here and a cell trained there carry the same name and mean the same loss:

* ``default`` — identity level: no overrides; family config (or its defaults)
  passes through unchanged.
* ``root`` — root-only supervision (1/0/0): fit the holistic document label at
  the root and leave nodes unsupervised.
* ``leaf`` — leaf supervision only (0/1/0): supervise every labeled leaf
  against its local target; no root or merge terms.
* ``node`` — full node-level supervision (0/1/1): leaves and intermediate
  merges, no root term. The densest local signal.
* ``mix`` — balanced root+node mix (3/1/1): root-dominant holistic supervision
  with node-level grounding.

``resolve_supervision`` turns a spec's ``supervision_level`` plus optional
explicit weight fields into one override mapping for ``backend_config``. A
non-default level and explicit weights are mutually exclusive: the named cell
must mean exactly its published weights.
"""

from __future__ import annotations

from typing import Any, Mapping

SUPERVISION_WEIGHT_FIELDS: tuple[str, ...] = ("root_weight", "leaf_weight", "merge_weight")

#: Level name -> weight overrides. ``default`` applies none.
SUPERVISION_LEVELS: Mapping[str, Mapping[str, float]] = {
    "default": {},
    "root": {"root_weight": 1.0, "leaf_weight": 0.0, "merge_weight": 0.0},
    "leaf": {"root_weight": 0.0, "leaf_weight": 1.0, "merge_weight": 0.0},
    "node": {"root_weight": 0.0, "leaf_weight": 1.0, "merge_weight": 1.0},
    "mix": {"root_weight": 3.0, "leaf_weight": 1.0, "merge_weight": 1.0},
}

DEFAULT_SUPERVISION_LEVEL = "default"

#: Registered family names whose config consumes the node-weight knobs.
NODE_SUPERVISION_FAMILIES: frozenset[str] = frozenset({"neural_operator", "fno"})


def normalize_supervision_level(value: Any) -> str:
    name = str(value or DEFAULT_SUPERVISION_LEVEL).strip().lower()
    if name not in SUPERVISION_LEVELS:
        raise ValueError(
            f"unknown supervision_level {name!r}; allowed: {sorted(SUPERVISION_LEVELS)}"
        )
    return name


def resolve_supervision(spec: Any) -> dict[str, float]:
    """Return the node-weight overrides one spec's supervision fields imply.

    Explicit per-weight spec fields are honored only at the ``default`` level;
    combining them with a named level would let the cell's name and its
    executed weights disagree, so that is an error.
    """

    level = normalize_supervision_level(getattr(spec, "supervision_level", None))
    explicit = {
        field: getattr(spec, field, None)
        for field in SUPERVISION_WEIGHT_FIELDS
        if getattr(spec, field, None) is not None
    }
    if level != DEFAULT_SUPERVISION_LEVEL and explicit:
        raise ValueError(
            f"supervision_level={level!r} is mutually exclusive with explicit "
            f"weight fields {sorted(explicit)}; a named level must mean exactly "
            "its published weights — use supervision_level='default' with "
            "explicit weights instead"
        )
    overrides = dict(SUPERVISION_LEVELS[level])
    for field, value in explicit.items():
        weight = float(value)
        if weight < 0.0:
            raise ValueError(f"{field} must be non-negative, got {value!r}")
        overrides[field] = weight
    return {field: float(value) for field, value in overrides.items()}


def supervision_provenance(*, level: str, overrides: Mapping[str, float]) -> dict[str, Any]:
    return {
        "level": normalize_supervision_level(level),
        "overrides": {str(k): float(v) for k, v in dict(overrides or {}).items()},
    }


__all__ = [
    "DEFAULT_SUPERVISION_LEVEL",
    "NODE_SUPERVISION_FAMILIES",
    "SUPERVISION_LEVELS",
    "SUPERVISION_WEIGHT_FIELDS",
    "normalize_supervision_level",
    "resolve_supervision",
    "supervision_provenance",
]
