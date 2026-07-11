"""``fit()`` — the single unified entry point for treepo.methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from treepo.methods._fit_inputs import (
    as_sequence,
    optional_int,
    resolve_objective,
    resolve_runtime_family,
)
from treepo.methods._fit_result import build_result
from treepo.methods._preference_traces import preference_training_rows
from treepo.methods.preference import PreferenceDataset
from treepo.methods.runtime import run_alternating_family


def fit(spec: Any) -> Any:
    """Run one alternating f/g ladder described by ``spec``."""
    backend_config = dict(spec.backend_config or {})
    axis = dict(spec.axis or {})
    initial = spec.initial_artifacts or {}

    family = resolve_runtime_family(spec, backend_config)
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
        traces=as_sequence(spec.train_data),
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
    )


__all__ = ["fit"]
