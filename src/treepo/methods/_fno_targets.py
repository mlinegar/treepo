"""Scalar and vector target extraction for neural-operator families.

Data-prep: read the supervision target(s) off each training tree, following the
config's ``target_key`` / ``target_vector_key`` preferences with sensible
fallbacks. ``_target_rows`` also enforces a uniform target width
across the batch. No torch here — this is pure metadata shaping.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from treepo.methods._coerce import safe_float as _safe_float
from treepo.methods._fno_config import NeuralOperatorFamilyConfig


def _target_rows(
    traces: Sequence[Any],
    config: NeuralOperatorFamilyConfig,
) -> tuple[list[Any], list[list[float]]]:
    rows: list[tuple[Any, list[float]]] = []
    for tree in traces:
        target = _target_vector(tree, config)
        if target is not None:
            rows.append((tree, target))
    if not rows:
        raise ValueError(
            "neural-operator families need training trees with scalar or vector "
            "target metadata ('teacher_score_native', backend_config['target_key'], "
            "or backend_config['target_vector_key'])."
        )
    width = len(rows[0][1])
    if width <= 0:
        raise ValueError("target vectors must be non-empty")
    for _tree, target in rows:
        if len(target) != width:
            raise ValueError("all target vectors must have the same length")
    return [tree for tree, _target in rows], [target for _tree, target in rows]


def _target_vector(tree: Any, config: NeuralOperatorFamilyConfig) -> list[float] | None:
    if config.target_vector_key:
        values = _vector_by_key(tree, config.target_vector_key)
        if values is None:
            return None
        if config.target_dim is not None and len(values) != int(config.target_dim):
            raise ValueError(
                f"target_vector_key={config.target_vector_key!r} produced {len(values)} values; "
                f"expected target_dim={int(config.target_dim)}"
            )
        return values
    if config.target_dim and int(config.target_dim) > 1:
        values = _vector_by_key(tree, "topic_proportions")
        if values is not None and len(values) == int(config.target_dim):
            return values
    score = _target_score(tree, config.target_key)
    return None if score is None else [float(score)]


def _vector_by_key(tree: Any, key: str | None) -> list[float] | None:
    if not key:
        return None
    meta = getattr(tree, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    value = meta.get(key) if key in meta else getattr(tree, str(key), None)
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        out = [float(item) for item in value]
    except TypeError:
        return None
    return out if out else None


def _target_score(tree: Any, target_key: str | None) -> float | None:
    meta = getattr(tree, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    keys = [target_key] if target_key else []
    keys.extend(
        [
            "teacher_score_native",
            "teacher_score_1_7",
            "expert_score_for_objective",
            "expert_score_native",
            "expert_score_1_7",
        ]
    )
    for key in keys:
        if not key:
            continue
        score = _safe_float(meta.get(key))
        if score is not None:
            return score
    return _safe_float(getattr(tree, "document_score", None))


__all__ = [
    "_target_rows",
    "_target_score",
    "_target_vector",
    "_vector_by_key",
]
