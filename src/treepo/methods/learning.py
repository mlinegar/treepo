"""``fit()`` — the single unified entry point for treepo.methods."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from treepo.methods._fit_inputs import (
    as_sequence,
    optional_int,
    resolve_objective,
    resolve_runtime_family,
)
from treepo.methods._fit_result import build_result
from treepo.methods._grid_axes import GridAxes, apply_grid_axes
from treepo.methods._preference_traces import preference_training_rows
from treepo.methods._supervision import (
    DEFAULT_SUPERVISION_LEVEL,
    normalize_supervision_level,
    resolve_supervision,
    supervision_provenance,
)
from treepo.methods.preference import PreferenceDataset
from treepo.methods.runtime import run_alternating_family
from treepo.objective import LOCAL_LAW_ESTIMATOR_ORACLE_STATE, ObjectiveSpec


def fit(spec: Any) -> Any:
    """Run one alternating f/g ladder described by ``spec``."""
    started = time.perf_counter()
    backend_config = dict(spec.backend_config or {})
    axis = dict(spec.axis or {})
    initial = spec.initial_artifacts or {}

    # First-class supervision-grid axes: pin the document subset and resolve the
    # node-label mix before training, and carry one seed per fit() call so every
    # cell at the same (seed, n) trains on the identical, persisted subset.
    grid_axes = GridAxes.from_spec(spec)
    backend_config.setdefault("seed", int(grid_axes.seed))

    # Named supervision level / explicit node weights (Phase 1). A non-default
    # level overrides any backend_config weights so the cell's name always
    # means exactly its published weights; the default level passes through.
    supervision_level = normalize_supervision_level(
        getattr(spec, "supervision_level", DEFAULT_SUPERVISION_LEVEL)
    )
    supervision_overrides = resolve_supervision(spec)
    backend_config.update(supervision_overrides)
    _apply_spec_local_law_weight(spec, backend_config)

    train_traces, grid_axes_provenance = apply_grid_axes(
        as_sequence(spec.train_data),
        axes=grid_axes,
        backend_config=backend_config,
    )
    # The gold_fraction axis pins which leaf units keep their gold labels;
    # hand the pinned selection to the family so the loss actually consumes it.
    # The llm_distilled axis attaches cached-teacher scores to the training
    # trees and routes node targets exclusively through the distilled key.
    mix_provenance = grid_axes_provenance.get("local_label_mix") or {}
    if mix_provenance.get("mix") == "gold_fraction" and "selected_node_units" in mix_provenance:
        backend_config.setdefault(
            "supervised_node_units", tuple(mix_provenance["selected_node_units"])
        )
    elif mix_provenance.get("mix") == "llm_distilled":
        from treepo.methods._distilled import apply_distilled_node_labels

        mix_provenance.update(
            apply_distilled_node_labels(
                train_traces, axes=grid_axes, backend_config=backend_config
            )
        )

    family = resolve_runtime_family(spec, backend_config)
    _require_node_supervision_support(
        family, level=supervision_level, overrides=supervision_overrides
    )
    objective = resolve_objective(backend_config)
    configure = getattr(family, "configure_objective", None)
    if callable(configure):
        configure(objective)
    elif objective is not None and float(objective.local_law_weight or 0.0) > 0.0:
        raise ValueError(
            f"objective declares local_law_weight > 0 but family "
            f"{getattr(family, 'name', type(family).__name__)!r} cannot execute "
            "a law-bearing objective (no configure_objective hook); a declared "
            "training objective must not be provenance-only"
        )
    # Single output_dir default across both fit entrypoints (public wrapper
    # and this backend): the same spec must land in the same place.
    output_dir = Path(backend_config.get("output_dir") or "outputs/treepo_fit")

    preference_dataset = PreferenceDataset.from_value(getattr(spec, "preference_data", None))
    f_preference_traces = preference_training_rows(preference_dataset, target="f")
    g_preference_traces = preference_training_rows(preference_dataset, target="g")

    records = run_alternating_family(
        family=family,
        f_init=initial.get("f"),
        g_init=initial.get("g"),
        traces=train_traces,
        f_traces=f_preference_traces or None,
        g_traces=g_preference_traces or None,
        eval_trees=as_sequence(spec.eval_data),
        # A function named fit must train by default: 2 = one f pass and one
        # g pass. Pass max_iterations=0 explicitly for an evaluation-only run.
        max_iterations=int(axis.get("max_iterations", 2)),
        axis_value=int(axis.get("axis_value", 0)),
        output_dir=output_dir,
        axis_kind=str(axis.get("axis_kind", "leaf_count")),
        leaf_count=optional_int(axis.get("leaf_count")),
        objective=objective,
    )

    return build_result(
        spec=spec,
        records=records,
        output_dir=output_dir,
        objective=objective,
        preference_dataset=preference_dataset,
        grid_axes=grid_axes_provenance,
        supervision=supervision_provenance(
            level=supervision_level, overrides=supervision_overrides
        ),
        wall_seconds=time.perf_counter() - started,
    )


def _apply_spec_local_law_weight(spec: Any, backend_config: dict[str, Any]) -> None:
    """Turn ``spec.local_law_weight`` into the canonical convex objective.

    The shorthand builds ObjectiveSpec v1 directly: ``(1 - λ) · root + λ ·
    corrected`` with the C1 (leaf) and C3 (merge) law channels split evenly and
    exact per-node targets (``oracle_state``). An explicit
    ``backend_config['objective']`` must stay the single weight source, so
    combining the two is an error.
    """

    raw = getattr(spec, "local_law_weight", None)
    if raw is None:
        return
    lam = float(raw)
    if lam < 0.0 or lam > 1.0:
        raise ValueError(f"spec.local_law_weight must be in [0, 1], got {raw!r}")
    if backend_config.get("objective") is not None:
        raise ValueError(
            "spec.local_law_weight and backend_config['objective'] are mutually "
            "exclusive; declare the law weight on the objective spec instead"
        )
    if lam <= 0.0:
        return
    backend_config["objective"] = ObjectiveSpec(
        objective_family="root_plus_local_laws",
        local_law_estimator=LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
        local_law_weight=lam,
        root_share=1.0 - lam,
        local_law_component_weights={
            "leaf_preservation": lam / 2.0,
            "merge_preservation": lam / 2.0,
        },
    )


def _require_node_supervision_support(
    family: Any, *, level: str, overrides: dict[str, float]
) -> None:
    """A named non-default level (or explicit node weights) needs a family
    whose config actually consumes the root/leaf/merge weight knobs —
    silently dropping a requested supervision cell would mislabel results.
    """

    wants_nodes = level != DEFAULT_SUPERVISION_LEVEL or any(
        float(overrides.get(key, 0.0)) > 0.0 for key in ("leaf_weight", "merge_weight")
    )
    if not wants_nodes:
        return
    family_config = getattr(family, "config", None)
    if hasattr(family_config, "leaf_weight") and hasattr(family_config, "merge_weight"):
        return
    raise ValueError(
        f"supervision_level={level!r} (or explicit leaf/merge weights) requires a "
        "family with per-node supervision support; family "
        f"{getattr(family, 'name', type(family).__name__)!r} has no "
        "leaf_weight/merge_weight config. Use supervision_level='default', or a "
        "neural_operator/fno family."
    )


__all__ = ["fit"]
