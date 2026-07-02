"""``fit()`` — the single unified entry point for treepo.methods."""

from __future__ import annotations

import tempfile
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
    output_dir = (
        Path(backend_config["output_dir"])
        if backend_config.get("output_dir")
        else Path(tempfile.mkdtemp(prefix="treepo_methods_fit_"))
    )

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
        max_iterations=int(axis.get("max_iterations", 0)),
        axis_value=int(axis.get("axis_value", 0)),
        output_dir=output_dir,
        axis_kind=str(axis.get("axis_kind", "leaf_count")),
        leaf_count=optional_int(axis.get("leaf_count")),
    )

    return build_result(
        spec=spec,
        records=records,
        output_dir=output_dir,
        objective=resolve_objective(backend_config),
        preference_dataset=preference_dataset,
    )


__all__ = ["fit"]
