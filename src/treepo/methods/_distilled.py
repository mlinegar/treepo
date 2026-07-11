"""Fit-side wiring for the ``llm_distilled`` label mix (fit-grid Phase 2).

``resolve_label_mix`` (in ``_grid_axes``) validates the axis and records the
source; this module makes the cell *train*: it loads the cached teacher rows
(or calls the configured node predictor), attaches scores to the training
trees, and points the per-node supervision path at the distilled metadata key
exclusively — an unmatched node stays unobserved instead of silently falling
back to its gold label, so a distilled cell is distilled-only by construction.

No LLM client is ever constructed here: fit-time distillation is pure cache
consumption (see ``treepo.distilled``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from treepo.distilled import (
    DISTILLED_NODE_KEY,
    attach_distilled_labels,
    attach_predictor_labels,
    load_teacher_node_rows,
)
from treepo.methods._grid_axes import GridAxes, _resolve_node_predictor

#: Nested backend_config mappings whose fields feed the family config; flat
#: backend_config keys override them, so defaults must respect both layers.
_NESTED_CONFIG_KEYS = ("fno_config", "neural_operator_config", "config")


def apply_distilled_node_labels(
    traces: Sequence[Any],
    *,
    axes: GridAxes,
    backend_config: dict[str, Any],
) -> dict[str, Any]:
    """Attach distilled node labels to ``traces`` and wire their consumption.

    Returns the attach report to fold into the ``local_label_mix`` provenance.
    Raises when no configured loss channel would consume the labels — silently
    training root-only while claiming a distilled mix would mislabel the cell.
    """

    _require_consuming_channel(backend_config)
    if axes.distilled_labels_path:
        labels = load_teacher_node_rows(
            axes.distilled_labels_path,
            dimension=_optional_str(backend_config.get("distilled_dimension")),
        )
        if labels.degenerate_scores:
            raise ValueError(
                f"teacher cache {labels.source_path} is degenerate: all "
                f"{labels.n_rows} rows carry the same {labels.score_key} value. "
                "Training on a constant teacher fits a constant — this is a "
                "placeholder-label cache (the known case is "
                "outputs/mpds_rile_llmseg_grid/leafq001), not a usable teacher."
            )
        report = attach_distilled_labels(traces, labels, node_key=DISTILLED_NODE_KEY)
    else:
        predictor = _resolve_node_predictor(backend_config)
        if predictor is None:  # resolve_label_mix already guarantees one source
            raise ValueError(
                "local_label_mix='llm_distilled' resolved without a source; "
                "set spec.distilled_labels_path or backend_config['node_oracle_predictor']"
            )
        report = attach_predictor_labels(traces, predictor, node_key=DISTILLED_NODE_KEY)

    _set_config_default(backend_config, "node_target_key", DISTILLED_NODE_KEY)
    _set_config_default(backend_config, "node_target_exclusive", True)
    return report


def _require_consuming_channel(backend_config: Mapping[str, Any]) -> None:
    if _config_value(backend_config, "leaf_weight") > 0.0:
        return
    if _config_value(backend_config, "merge_weight") > 0.0:
        return
    objective = backend_config.get("objective")
    law_weight = getattr(objective, "local_law_weight", None)
    if law_weight is None and isinstance(objective, Mapping):
        law_weight = objective.get("local_law_weight")
    if law_weight is not None and float(law_weight) > 0.0:
        return
    raise ValueError(
        "local_label_mix='llm_distilled' attaches per-node teacher labels, but "
        "no loss channel consumes them: leaf_weight and merge_weight are 0 and "
        "no law-bearing objective is configured. Set supervision_level to "
        "'leaf'/'node'/'mix' (or spec.local_law_weight > 0) so the distilled "
        "cell actually trains on its labels."
    )


def _config_value(backend_config: Mapping[str, Any], key: str) -> float:
    if key in backend_config:
        return _as_float(backend_config.get(key))
    for nested_key in _NESTED_CONFIG_KEYS:
        nested = backend_config.get(nested_key)
        if isinstance(nested, Mapping) and key in nested:
            return _as_float(nested.get(key))
        value = getattr(nested, key, None)
        if value is not None:
            return _as_float(value)
    return 0.0


def _set_config_default(backend_config: dict[str, Any], key: str, value: Any) -> None:
    """Set a flat default only when neither layer already configures the key.

    Flat backend_config keys override the nested family-config mappings, so a
    blind ``setdefault`` would clobber a nested user setting.
    """

    if key in backend_config:
        return
    for nested_key in _NESTED_CONFIG_KEYS:
        nested = backend_config.get(nested_key)
        if isinstance(nested, Mapping) and key in nested:
            return
        if getattr(nested, key, None) is not None:
            return
    backend_config[key] = value


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = ["apply_distilled_node_labels"]
