#!/usr/bin/env python3
"""Run two isomorphic tiny Manifesto Semantic-Forest grids.

The experiment itself is deliberately just a loop around the one public API::

    for family in ("dspy", "fno"):
        for target_width in (1, 3, 57):
            for representation_path in (
                "full_doc_direct",
                "ctree_base_summary",
                "ctree_recursive",
            ):
                fit_kwargs = {...}  # targets, tree records, and family backend
                result = treepo.fit(**fit_kwargs)

``run_family_grid(...)`` contains that literal two-axis loop for one family;
``run_complete_grid(...)`` calls it once for DSPy and once for FNO. Everything
else in this file constructs the tiny fixture or validates/reports a result.

K=1 uses the same exact named-vector I/O path as K=3 and K=57.  Its sole
coordinate is ``rile_normalized`` in [0,1], with the reporting-only readout
``raw_RILE = 200 * rile_normalized - 100``.  All widths use sum-L1 for the
joint vector endpoint; there is no K=1 scalar API or metric branch.

The path/backend pair declares one explicit topology and ``g_mode`` contract:

* ``full_doc_direct`` is a singleton direct readout, ``f(X)``, with one leaf,
  no ``g`` application, ``g_mode="identity"``, and no composition;
* ``ctree_base_summary`` is a singleton summarized readout, ``f(g(X))``,
  with one leaf, one leaf-``g`` application, and no merge composition;
* ``ctree_recursive`` is the four-leaf, three-merge compositional readout.

Optimizer-backed DSPy and FNO each learn one shared ``g`` on both C-Tree paths;
``reduce_g`` is only the derived recursive fold of that same operator. The
learned-grid default is the recovered ``f -> g -> f`` sequence, so the final
readout is fitted against states produced by the current ``g``. A
singleton fit supplies only the leaf-input training domain, while a recursive
fit can additionally supply the merge-input domain. Direct cells retain
``g_mode="identity"`` and learn only the vector readout ``f``.

The records are tiny synthetic fixtures with authoritative CMP counts. Learned
DSPy execution requires optimizer/LM configuration (or injected compiler and
program adapters) and uses the package's optimizer-backed ``f``/shared-``g``
path. For a local interface smoke only, callers may explicitly request
``dspy_execution="offline_fixture"``; only that mode injects
:class:`FixtureOracleDSPyProgram` and a fixed denominator-weighted analytic
``g``. The offline fixture makes no live LLM call, performs no DSPy
optimization, and is never the learned or empirical DSPy grid.

Every cell fails closed unless the final fit history contains exactly one row
with one finite, exact-key named prediction and target vector for every
expected evaluation ``tree_id``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import treepo
from treepo.common import stable_digest
from treepo.tasks.manifesto.components import (
    RILE_CMP56_GRANULARITY,
    RILE_NORMALIZED_ORACLE_ID,
    RILE_NORMALIZED_TARGET_KEY,
    RILE_NORMALIZED_TARGET_NAME,
    RILE_POLARITY_GRANULARITY,
    attach_manifesto_rile_components,
    manifesto_rile_component_fit_fragment,
    manifesto_rile_component_readout,
    manifesto_rile_component_target_names,
    manifesto_rile_components_from_counts,
    manifesto_rile_normalized_fit_fragment,
)
from treepo.tasks.manifesto.semantic_forest_reporting import (
    manifesto_semantic_forest_comparison_report_schema,
    validate_manifesto_semantic_forest_comparison_row,
)
from treepo.tree import TreeNode, TreeRecord

GRID_VERSION = "treepo.examples.manifesto_semantic_forest_grid.v8"
EVIDENCE_CLASS = "synthetic_package_fixture_not_polmeth_evidence"
TARGET_WIDTHS = (1, 3, 57)
REPRESENTATION_PATHS = (
    "full_doc_direct",
    "ctree_base_summary",
    "ctree_recursive",
)
# Kept as an import-compatible alias while its values now name the three
# execution paths rather than the two coarse representation families.
REPRESENTATIONS = REPRESENTATION_PATHS
FAMILIES = ("dspy", "fno")
DSPY_EXECUTIONS = ("learned", "offline_fixture")
DEFAULT_DSPY_EXECUTION = "learned"

RILE_SINGLETON_NAME = RILE_NORMALIZED_TARGET_NAME
RILE_SINGLETON_KEY = RILE_NORMALIZED_TARGET_KEY
RILE_SINGLETON_ORACLE_ID = RILE_NORMALIZED_ORACLE_ID

FIXTURE_BACKEND = "deterministic_python_oracle"
REPLACEMENT_REQUIREMENT = (
    "Replace FixtureOracleDSPyProgram with the downstream live/optimized "
    "DSPy component before treating the DSPy family grid as empirical "
    "LLM evidence."
)
PLAN_WARNING = (
    "The learned DSPy plan is optimizer-backed and requires runtime DSPy/LM "
    "configuration before execution. Planned expected_* fields are not "
    "realized update evidence. The data remain synthetic package fixtures, "
    "so results are not Polmeth evidence."
)

OFFLINE_FIXTURE_WARNING = (
    "Explicit offline DSPy interface fixture only: "
    "fixture_backend=deterministic_python_oracle, live_llm_call=false, and "
    "dspy_optimization_executed=false. Its analytic g is fixed, not learned; "
    "this is never the learned or empirical DSPy grid."
)

_COMMON_WARNING = (
    "Synthetic package-fixture comparison only. Learned family execution may "
    "exercise real optimizers, but these fixture scores are not Polmeth or "
    "publication evidence."
)


# Ten deterministic documents, each represented as four source spans. A code
# contributes one unit of non-header CMP mass; "000" is residual content.
_DOCUMENTS: tuple[
    tuple[str, str, tuple[tuple[str, ...], ...]],
    ...,
] = (
    (
        "train-left",
        "train",
        (("103", "103"), ("202", "000"), ("501", "501"), ("104", "000")),
    ),
    (
        "train-right",
        "train",
        (("104", "104"), ("201", "000"), ("401", "501"), ("103", "000")),
    ),
    (
        "train-other",
        "train",
        (("501", "501"), ("000", "000"), ("103", "104"), ("301", "701")),
    ),
    (
        "train-green-left",
        "train",
        (("416", "501"), ("506", "103"), ("202", "000"), ("104", "503")),
    ),
    (
        "train-security-right",
        "train",
        (("605", "606"), ("104", "401"), ("501", "000"), ("103", "302")),
    ),
    (
        "train-welfare-left",
        "train",
        (("504", "506"), ("701", "103"), ("501", "000"), ("104", "602")),
    ),
    (
        "test-mixed-left",
        "test",
        (("103", "202"), ("403", "501"), ("104", "000"), ("506", "302")),
    ),
    (
        "test-mixed-right",
        "test",
        (("104", "201"), ("401", "501"), ("103", "000"), ("605", "302")),
    ),
    (
        "test-balanced",
        "test",
        (("103", "104"), ("202", "201"), ("501", "000"), ("301", "701")),
    ),
    (
        "test-residual",
        "test",
        (("000", "000"), ("501", "000"), ("103", "104"), ("302", "602")),
    ),
)


@dataclass(frozen=True)
class GridCell:
    """One cell in either isomorphic family grid."""

    target_width: int
    representation_path: str
    family: str
    seed: int
    output_dir: Path
    dspy_execution: str = DEFAULT_DSPY_EXECUTION

    @property
    def cell_id(self) -> str:
        mode = f"__{_resolve_dspy_execution(self.dspy_execution)}" if self.family == "dspy" else ""
        return (
            f"k{self.target_width:02d}__{self.representation_path}__{self.family}"
            f"{mode}__seed{self.seed}"
        )

    def to_dict(self, *, requested_max_iterations: int = 3) -> dict[str, Any]:
        g_contract = _g_contract(
            self.family,
            self.representation_path,
            requested_max_iterations=requested_max_iterations,
            dspy_execution=self.dspy_execution,
        )
        payload = {
            "cell_id": self.cell_id,
            "target_width": self.target_width,
            "representation_path": self.representation_path,
            "representation": g_contract["representation"],
            "family": self.family,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "substrate": _substrate(self.family),
            "optimizer": _optimizer(self.family, dspy_execution=self.dspy_execution),
            "execution_path": _execution_path(
                self.family,
                self.representation_path,
                dspy_execution=self.dspy_execution,
            ),
            **_common_provenance(),
            "g_mode": g_contract["g_mode"],
            "schedule": g_contract["schedule"],
            "leaf_count": g_contract["leaf_count"],
            "topology_kind": g_contract["topology_kind"],
            "merge_application_count": g_contract["merge_application_count"],
            "readout_path": g_contract["readout_path"],
            "leaf_g_application_count": g_contract["leaf_g_application_count"],
            "composition_present": g_contract["composition_present"],
            "same_g_across_node_roles": g_contract["same_g_across_node_roles"],
            "reduce_g_is_derived": g_contract["reduce_g_is_derived"],
            "expected_g_training_call_roles": g_contract["expected_g_training_call_roles"],
            "expected_g_training_role_evidence_source": g_contract[
                "expected_g_training_role_evidence_source"
            ],
            "expected_merge_domain_training_observed": g_contract[
                "expected_merge_domain_training_observed"
            ],
            "expected_shared_g_updated_with_merge_domain": g_contract[
                "expected_shared_g_updated_with_merge_domain"
            ],
            "g_contract": g_contract,
            "target_api_contract": _target_api_contract(),
        }
        if self.family == "dspy":
            payload["dspy_execution"] = _resolve_dspy_execution(self.dspy_execution)
            payload.update(_dspy_execution_provenance(self.dspy_execution))
        return payload


class FixtureOracleDSPyProgram:
    """Deterministic DSPy-shaped fixture with path-aware execution.

    ``full_doc_direct`` reads the document target. Both C-Tree paths traverse
    leaves and apply the same denominator-mass-weighted rule for K=1, K=3,
    and K=57. For ``ctree_base_summary`` that traversal contains one leaf and
    therefore exercises leaf ``g`` without exercising merge composition.
    """

    def __init__(self, target_vector_key: str) -> None:
        self.target_vector_key = str(target_vector_key)

    def __call__(self, *, tree: Any, **_kwargs: Any) -> dict[str, float]:
        record = TreeRecord.from_value(tree)
        representation_path = str(
            dict(record.metadata or {}).get("representation_path") or "full_doc_direct"
        )
        if representation_path == "full_doc_direct":
            raw = dict(record.metadata or {}).get(self.target_vector_key)
            return self._mapping(raw, where=f"tree {record.tree_id!r} root")

        leaves = tuple(record.leaves())
        if not leaves:
            raise ValueError(f"{representation_path} tree {record.tree_id!r} has no leaves")

        declared: tuple[str, ...] | None = None
        weighted_sum: dict[str, float] = {}
        total_mass = 0.0
        for leaf in leaves:
            metadata = dict(leaf.metadata or {})
            values = self._mapping(
                metadata.get(self.target_vector_key),
                where=f"tree {record.tree_id!r} leaf {leaf.node_id!r}",
            )
            names = tuple(values)
            if declared is None:
                declared = names
            elif names != declared:
                raise ValueError(
                    f"{representation_path} tree {record.tree_id!r} leaf target orders disagree"
                )
            mass = _strict_finite_number(metadata.get("total_non_header_qsentences"))
            if mass is None or mass <= 0.0:
                raise ValueError(
                    f"{representation_path} tree {record.tree_id!r} "
                    f"leaf {leaf.node_id!r} has "
                    "non-positive denominator mass"
                )
            total_mass += mass
            for name, value in values.items():
                weighted_sum[name] = weighted_sum.get(name, 0.0) + mass * value

        if total_mass <= 0.0:
            raise ValueError(
                f"{representation_path} tree {record.tree_id!r} has zero denominator mass"
            )
        return {name: weighted_sum[name] / total_mass for name in (declared or ())}

    @staticmethod
    def _mapping(value: Any, *, where: str) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError(
                f"{where} is missing the named target mapping required by the fixture DSPy program"
            )
        result: dict[str, float] = {}
        for raw_name, raw_component in value.items():
            name = str(raw_name)
            if name in result:
                raise ValueError(f"{where} contains colliding target names")
            component = _strict_finite_number(raw_component)
            if component is None:
                raise ValueError(f"{where} target {name!r} is not finite numeric")
            result[name] = component
        return result


def build_grid_plan(
    output_dir: Path | str,
    *,
    seed: int = 7,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
) -> list[GridCell]:
    """Return the frozen K x representation-path x family product."""

    root = Path(output_dir)
    resolved_dspy_execution = _resolve_dspy_execution(dspy_execution)
    return [
        GridCell(
            target_width=target_width,
            representation_path=representation_path,
            family=family,
            seed=int(seed),
            output_dir=_cell_output_dir(
                root,
                target_width=target_width,
                representation_path=representation_path,
                family=family,
                seed=int(seed),
                dspy_execution=resolved_dspy_execution,
            ),
            dspy_execution=resolved_dspy_execution,
        )
        for target_width in TARGET_WIDTHS
        for representation_path in REPRESENTATION_PATHS
        for family in FAMILIES
    ]


def build_family_grid_plan(
    output_dir: Path | str,
    family: str,
    *,
    seed: int = 7,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
) -> list[GridCell]:
    """Return one nine-cell K x representation-path family grid."""

    resolved = _resolve_family(family)
    return [
        cell
        for cell in build_grid_plan(
            output_dir,
            seed=seed,
            dspy_execution=dspy_execution,
        )
        if cell.family == resolved
    ]


def make_cmp_count_records(
    representation_path: str,
) -> tuple[list[TreeRecord], list[TreeRecord]]:
    """Build paired train/test fixtures for one representation path.

    The document roster, source text, and root CMP counts are identical across
    paths. The direct and base-summary paths each have one root/leaf unit;
    ``ctree_recursive`` has four leaves, two internal merges, and a root.
    """

    resolved = str(representation_path)
    if resolved not in REPRESENTATIONS:
        raise ValueError(
            f"representation_path must be one of {REPRESENTATION_PATHS!r}, "
            f"got {representation_path!r}"
        )
    records = [
        _cmp_record(
            doc_id=doc_id,
            split=split,
            leaf_code_groups=leaf_code_groups,
            representation_path=resolved,
        )
        for doc_id, split, leaf_code_groups in _DOCUMENTS
    ]
    return (
        [record for record in records if record.metadata["split"] == "train"],
        [record for record in records if record.metadata["split"] == "test"],
    )


def prepare_target_records(
    records: Iterable[TreeRecord],
    target_width: int,
) -> tuple[list[TreeRecord], dict[str, Any]]:
    """Attach one exact named-vector contract to every tree and node."""

    width = int(target_width)
    materialized = list(records)
    if width == 1:
        hydrated = [_attach_singleton_rile(record) for record in materialized]
        return hydrated, manifesto_rile_normalized_fit_fragment()

    granularity = _granularity_for_width(width)
    fragment = manifesto_rile_component_fit_fragment(granularity)
    hydrated = attach_manifesto_rile_components(
        materialized,
        granularity=granularity,
        drop_scalar_root_label=True,
    )
    return hydrated, fragment


def fit_grid_cell(
    cell: GridCell,
    *,
    fno_epochs: int = 1,
    max_iterations: int = 3,
    dspy_backend_config: Mapping[str, Any] | None = None,
    fit_fn: Callable[..., Any] | None = None,
    initial_artifacts: Mapping[str, Any] | None = None,
    shared_model_source_cell_id: str | None = None,
) -> dict[str, Any]:
    """Run one cell through ``treepo.fit`` and enforce exact row coverage."""

    _validate_cell(cell)
    g_contract = _g_contract(
        cell.family,
        cell.representation_path,
        requested_max_iterations=max_iterations,
        dspy_execution=cell.dspy_execution,
    )
    if g_contract["reuses_model_artifacts"] and initial_artifacts is None:
        raise ValueError(
            f"{cell.representation_path} evaluates the shared model from "
            f"{g_contract['shared_model_source_path']}; initial_artifacts are required"
        )
    train_raw, test_raw = make_cmp_count_records(cell.representation_path)
    _require_exact_leaf_count(
        (*train_raw, *test_raw),
        expected=int(g_contract["leaf_count"]),
        representation_path=cell.representation_path,
    )
    train, fragment = prepare_target_records(train_raw, cell.target_width)
    test, test_fragment = prepare_target_records(test_raw, cell.target_width)
    target_names = _target_names(fragment)
    if target_names != _target_names(test_fragment):
        raise AssertionError("train/test target contracts diverged")

    if cell.family == "dspy" and cell.dspy_execution == "learned":
        backend_config = _validated_learned_dspy_backend_config(dspy_backend_config)
        # Preserve the recovered alternating learner: the first f pass may use
        # references because no learned g exists yet; the final f pass consumes
        # states generated by the current g. Callers can still opt explicitly
        # into gold_state for an ablation.
        backend_config.setdefault("f_record_source", "generated_when_available")
        if g_contract["g_mode"] == "learned":
            # These tiny records intentionally have no teacher-authored summaries.
            # Opt into visible reference-text targets explicitly; the core must
            # never infer this source merely from missing summary labels.
            backend_config["allow_identity_g_targets"] = True
            backend_config["g_target_source"] = "explicit_reference_text_fixture"
        # The frozen target catalog is owned by this grid, not by the injected
        # optimizer/LM config. It takes precedence over overlapping keys.
        backend_config.update(dict(fragment["backend_config"]))
    else:
        backend_config = dict(fragment["backend_config"])
    backend_metadata = {
        "example_grid_version": GRID_VERSION,
        "representation": g_contract["representation"],
        "representation_path": cell.representation_path,
        "topology_kind": g_contract["topology_kind"],
        "leaf_count": g_contract["leaf_count"],
        "merge_application_count": g_contract["merge_application_count"],
        "readout_path": g_contract["readout_path"],
        "leaf_g_application_count": g_contract["leaf_g_application_count"],
        "composition_present": g_contract["composition_present"],
        "target_width": cell.target_width,
        "family": cell.family,
        "substrate": _substrate(cell.family),
        "optimizer": _optimizer(
            cell.family,
            dspy_execution=cell.dspy_execution,
        ),
        "execution_path": _execution_path(
            cell.family,
            cell.representation_path,
            dspy_execution=cell.dspy_execution,
        ),
        "g_mode": g_contract["g_mode"],
        "g_contract": g_contract,
        "schedule": g_contract["schedule"],
        "requested_max_iterations": g_contract["requested_max_iterations"],
        "effective_max_iterations": g_contract["effective_max_iterations"],
        **_common_provenance(),
    }
    if cell.family == "dspy":
        backend_metadata["dspy_execution"] = _resolve_dspy_execution(cell.dspy_execution)
        backend_metadata.update(_dspy_execution_provenance(cell.dspy_execution))
    backend_config.update(
        {
            "output_dir": str(cell.output_dir),
            "seed": int(cell.seed),
            "metadata": backend_metadata,
        }
    )
    if cell.family == "dspy" and cell.dspy_execution == "offline_fixture":
        if dspy_backend_config:
            raise ValueError(
                "dspy_backend_config is not consumed by offline_fixture; remove it "
                "or select dspy_execution='learned'"
            )
        backend_config.update(
            {
                "optimizer": "none",
                "dspy_program": FixtureOracleDSPyProgram(str(backend_config["target_vector_key"])),
                "audit_laws": False,
                "lm_config": {
                    "model": "injected-fixture-program",
                    "verify_model": False,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
            }
        )
    elif cell.family == "fno":
        backend_config.update(
            {
                "embedding_dim": 8,
                "hidden_channels": 4,
                "n_modes": 2,
                "n_layers": 1,
                "head_hidden_dim": 8,
                "epochs_per_iteration": int(fno_epochs),
                "batch_size": len(train),
                "eval_batch_size": len(test),
                "learning_rate": 0.01,
                "device": "cpu",
            }
        )

    fit_kwargs = {
        "space_kind": fragment["space_kind"],
        "family": cell.family,
        "schedule": g_contract["schedule"],
        "g_mode": g_contract["g_mode"],
        "oracle_targets": fragment["oracle_targets"],
        "train_data": train,
        "eval_data": test,
        "backend_config": backend_config,
        "seed": int(cell.seed),
        "axis": {
            "max_iterations": int(g_contract["effective_max_iterations"]),
            "axis_kind": "target_width",
            "axis_value": int(cell.target_width),
            "leaf_count": int(g_contract["leaf_count"]),
            "representation": g_contract["representation"],
            "representation_path": cell.representation_path,
            "topology_kind": g_contract["topology_kind"],
            "merge_application_count": int(g_contract["merge_application_count"]),
            "readout_path": g_contract["readout_path"],
            "leaf_g_application_count": int(g_contract["leaf_g_application_count"]),
            "composition_present": bool(g_contract["composition_present"]),
            "same_g_across_node_roles": True,
            "reduce_g_is_derived": True,
        },
    }
    if initial_artifacts is not None:
        model_artifacts = dict(initial_artifacts)
        missing = [kind for kind in ("f", "g") if model_artifacts.get(kind) is None]
        if missing:
            raise ValueError(f"shared-model initial_artifacts are missing {missing!r}")
        fit_kwargs["initial_artifacts"] = model_artifacts
    elif g_contract["g_mode"] == "fixed":
        fit_kwargs["initial_artifacts"] = {
            "g": {
                "kind": "manifesto_fixture_fixed_g",
                "g_mode": "fixed",
                "operator": g_contract["implementation"],
                "trainable": False,
                "train_g_enabled": False,
                "same_g_across_node_roles": True,
                "reduce_g_is_derived": True,
                "learning_evidence": "analytic_wiring_only_not_learned_shared_g",
            }
        }
    measurement_started_at = _utc_now()
    fit_wall_start = time.perf_counter()
    if fit_fn is None:
        result = treepo.fit(**fit_kwargs)
    else:  # Test/integration hook; production always uses the public package API.
        result = fit_fn(fit_kwargs)
    fit_wall_seconds = float(time.perf_counter() - fit_wall_start)
    measurement_finished_at = _utc_now()
    result_artifacts = dict(getattr(result, "artifacts", {}) or {})
    model_artifacts = {kind: result_artifacts.get(kind) for kind in ("f", "g")}
    if any(model_artifacts[kind] is None for kind in ("f", "g")):
        raise ValueError("treepo.fit result must expose both f and g model artifacts")
    if initial_artifacts is not None and model_artifacts != dict(initial_artifacts):
        raise ValueError(
            "evaluation-only shared-model view changed its f/g artifacts; "
            "leaf-count views must reuse the exact trained pair"
        )
    fit_summary = dict(result.summary or {})
    executed_g_contract = _validated_executed_g_contract(
        fit_summary.get("g_contract"),
        configured=g_contract,
    )
    rows = _final_prediction_rows(result)
    authoritative_targets = _authoritative_targets_by_id(
        test,
        target_names=target_names,
        target_vector_key=str(backend_config["target_vector_key"]),
    )
    coverage = _prediction_coverage(
        rows,
        expected_ids=[tree.tree_id for tree in test],
        target_names=target_names,
        authoritative_targets_by_id=authoritative_targets,
    )
    status = str(result.status)
    error: str | None = None
    if status == "success" and not coverage["complete"]:
        status = "failed"
        error = (
            "incomplete final evaluation-row roster: "
            f"rows={coverage['all_row_n']}, "
            f"expected={coverage['expected_n']}, "
            f"incomplete={coverage['incomplete_row_tree_ids']!r}, "
            f"duplicates={coverage['all_row_duplicate_tree_ids']!r}, "
            f"missing={coverage['all_row_missing_tree_ids']!r}, "
            f"unexpected={coverage['all_row_unexpected_tree_ids']!r}, "
            f"target_mismatches={coverage['target_mismatch_tree_ids']!r}"
        )
    common_metrics = summarize_common_metrics(
        rows,
        target_width=cell.target_width,
        authoritative_targets_by_id=authoritative_targets,
    )
    comparison_row = _comparison_row(
        cell,
        rows=rows,
        target_names=target_names,
        coverage=coverage,
        common_metrics=common_metrics,
        status=status,
        fit_wall_seconds=fit_wall_seconds,
        measurement_started_at=measurement_started_at,
        measurement_finished_at=measurement_finished_at,
        train=train,
        evaluation=test,
        fno_epochs=fno_epochs,
        max_iterations=max_iterations,
        executed_g_contract=executed_g_contract,
    )
    family_telemetry = _family_telemetry(
        cell,
        fno_epochs=fno_epochs,
        max_iterations=max_iterations,
    )
    report = {
        **cell.to_dict(requested_max_iterations=max_iterations),
        "status": status,
        "common_metrics": common_metrics,
        "treepo_metrics": dict(result.metrics or {}),
        "fit_summary": fit_summary,
        "executed_g_contract": executed_g_contract,
        "manifest_path": result.manifest_path,
        "model_artifacts": model_artifacts,
        "reuses_model_artifacts": bool(g_contract["reuses_model_artifacts"]),
        "shared_model_source_cell_id": (shared_model_source_cell_id or cell.cell_id),
        "shared_g_updated_with_merge_domain": comparison_row["identity"][
            "shared_g_updated_with_merge_domain"
        ],
        "prediction_coverage": coverage,
        "comparison_row": comparison_row,
        "family_telemetry": family_telemetry,
        "fit_contract": {
            "public_api": "treepo.fit",
            "single_fit_call": True,
            "model_scope": g_contract["model_scope"],
            "reuses_model_artifacts": bool(g_contract["reuses_model_artifacts"]),
            "shared_model_source_cell_id": (shared_model_source_cell_id or cell.cell_id),
            **g_contract,
            **_target_api_contract(),
        },
    }
    if error is not None:
        report["error"] = error
    return report


def run_family_grid(
    family: str,
    output_dir: Path | str,
    *,
    seed: int = 7,
    fno_epochs: int = 1,
    max_iterations: int = 3,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
    dspy_backend_config: Mapping[str, Any] | None = None,
    fit_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute one isomorphic nine-cell family grid."""

    resolved = _resolve_family(family)
    resolved_dspy_execution = _resolve_dspy_execution(dspy_execution)
    root = Path(output_dir)
    cells: list[dict[str, Any]] = []

    def execute_cell(
        cell: GridCell,
        *,
        initial_artifacts: Mapping[str, Any] | None = None,
        source_cell_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return fit_grid_cell(
                cell,
                fno_epochs=fno_epochs,
                max_iterations=max_iterations,
                dspy_backend_config=dspy_backend_config,
                fit_fn=fit_fn,
                initial_artifacts=initial_artifacts,
                shared_model_source_cell_id=source_cell_id,
            )
        except Exception as exc:
            return _failed_cell_report(
                cell,
                error=f"{type(exc).__name__}: {exc}",
                fno_epochs=fno_epochs,
                max_iterations=max_iterations,
            )

    for target_width in TARGET_WIDTHS:

        def make_cell(representation_path: str) -> GridCell:
            return GridCell(
                target_width=target_width,
                representation_path=representation_path,
                family=resolved,
                seed=int(seed),
                output_dir=_cell_output_dir(
                    root,
                    target_width=target_width,
                    representation_path=representation_path,
                    family=resolved,
                    seed=int(seed),
                    dspy_execution=resolved_dspy_execution,
                ),
                dspy_execution=resolved_dspy_execution,
            )

        by_path: dict[str, dict[str, Any]] = {}
        direct_cell = make_cell("full_doc_direct")
        by_path["full_doc_direct"] = execute_cell(direct_cell)

        recursive_cell = make_cell("ctree_recursive")
        recursive_report = execute_cell(recursive_cell)
        by_path["ctree_recursive"] = recursive_report

        base_cell = make_cell("ctree_base_summary")
        source_artifacts = recursive_report.get("model_artifacts")
        if recursive_report.get("status") == "success" and isinstance(source_artifacts, Mapping):
            by_path["ctree_base_summary"] = execute_cell(
                base_cell,
                initial_artifacts=source_artifacts,
                source_cell_id=recursive_cell.cell_id,
            )
        else:
            by_path["ctree_base_summary"] = _failed_cell_report(
                base_cell,
                error="RuntimeError: shared recursive model artifacts are unavailable",
                fno_epochs=fno_epochs,
                max_iterations=max_iterations,
            )
        cells.extend(by_path[path] for path in REPRESENTATION_PATHS)

    schema = comparison_report_schema()
    payload = {
        "version": GRID_VERSION,
        "family": resolved,
        "substrate": _substrate(resolved),
        "optimizer": _optimizer(
            resolved,
            dspy_execution=resolved_dspy_execution,
        ),
        "warning": _grid_warning(
            families=(resolved,),
            dspy_execution=resolved_dspy_execution,
        ),
        **_common_provenance(),
        "grid": _grid_summary(
            families=(resolved,),
            cells=cells,
            dspy_execution=resolved_dspy_execution,
        ),
        "comparison_report_schema": schema,
        "comparison_rows": [cell["comparison_row"] for cell in cells],
        "family_telemetry_sidecars": [cell["family_telemetry"] for cell in cells],
        "g_contracts": {
            representation_path: _g_contract(
                resolved,
                representation_path,
                requested_max_iterations=max_iterations,
                dspy_execution=resolved_dspy_execution,
            )
            for representation_path in REPRESENTATION_PATHS
        },
        "target_api_contract": _target_api_contract(),
        "cells": cells,
    }
    if resolved == "dspy":
        payload["dspy_execution"] = resolved_dspy_execution
        payload.update(_dspy_execution_provenance(resolved_dspy_execution))
    _write_json(
        root
        / _family_grid_report_filename(
            resolved,
            dspy_execution=resolved_dspy_execution,
        ),
        payload,
    )
    return payload


def run_complete_grid(
    output_dir: Path | str,
    *,
    seed: int = 7,
    fno_epochs: int = 1,
    max_iterations: int = 3,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
    dspy_backend_config: Mapping[str, Any] | None = None,
    fit_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute both grids and return family and flat eighteen-cell views."""

    resolved_dspy_execution = _resolve_dspy_execution(dspy_execution)
    root = Path(output_dir)
    family_grids: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        family_grids[family] = run_family_grid(
            family,
            root,
            seed=seed,
            fno_epochs=fno_epochs,
            max_iterations=max_iterations,
            dspy_execution=resolved_dspy_execution,
            dspy_backend_config=dspy_backend_config,
            fit_fn=fit_fn,
        )
    cells = [cell for family in FAMILIES for cell in family_grids[family]["cells"]]
    schema = comparison_report_schema()
    dspy_provenance = _dspy_execution_provenance(resolved_dspy_execution)
    if resolved_dspy_execution == "learned":
        dspy_paths = {
            "full_doc_direct": "direct optimizer-backed vector f with identity g",
            "ctree_base_summary": "one-leaf learned shared DSPy g summary before f",
            "ctree_recursive": "recursive fold of the same learned shared DSPy g before f",
        }
        dspy_expected = True
        analytic_only = False
    else:
        dspy_paths = {
            "full_doc_direct": "direct offline fixture program",
            "ctree_base_summary": "one-leaf analytic fixture summary before f",
            "ctree_recursive": "recursive fold of the same analytic fixture g before f",
        }
        dspy_expected = False
        analytic_only = True
    payload = {
        "version": GRID_VERSION,
        "warning": _grid_warning(
            families=FAMILIES,
            dspy_execution=resolved_dspy_execution,
        ),
        **_common_provenance(),
        "dspy_execution": resolved_dspy_execution,
        **dspy_provenance,
        "grid": _grid_summary(
            families=FAMILIES,
            cells=cells,
            dspy_execution=resolved_dspy_execution,
        ),
        "comparison_report_schema": schema,
        "comparison_report_schema_identical_across_families": (
            family_grids["dspy"]["comparison_report_schema"]
            == family_grids["fno"]["comparison_report_schema"]
            == schema
        ),
        "comparison_rows": [cell["comparison_row"] for cell in cells],
        "family_telemetry_sidecars": [cell["family_telemetry"] for cell in cells],
        "target_api_contract": _target_api_contract(),
        "family_grids": family_grids,
        "cells": cells,
        "execution_contracts": {
            "dspy": {
                "family": "dspy",
                "substrate": "llm",
                "optimizer": _optimizer(
                    "dspy",
                    dspy_execution=resolved_dspy_execution,
                ),
                "dspy_execution": resolved_dspy_execution,
                **dspy_paths,
                # Configuration expectations are not realized evidence. The
                # per-cell executed_g_contract remains authoritative.
                "expected_learned_shared_g_updates_in_ctree_cells": dspy_expected,
                "learned_shared_g_evidence": None if dspy_expected else False,
                "analytic_wiring_only": analytic_only,
                "g_by_representation_path": {
                    representation_path: _g_contract(
                        "dspy",
                        representation_path,
                        requested_max_iterations=max_iterations,
                        dspy_execution=resolved_dspy_execution,
                    )
                    for representation_path in REPRESENTATION_PATHS
                },
                **dspy_provenance,
            },
            "fno": {
                "family": "fno",
                "substrate": "neural_operator_fno",
                "optimizer": "gradient",
                "full_doc_direct": "single-unit direct neural operator",
                "ctree_base_summary": "one-leaf trainable g summary before f",
                "ctree_recursive": "four-leaf fold of the same trainable FNO g before f",
                "g_by_representation_path": {
                    representation_path: _g_contract(
                        "fno",
                        representation_path,
                        requested_max_iterations=max_iterations,
                        dspy_execution=resolved_dspy_execution,
                    )
                    for representation_path in REPRESENTATION_PATHS
                },
                "fixture_data": True,
            },
        },
    }
    _write_json(
        root / _complete_grid_report_filename(dspy_execution=resolved_dspy_execution),
        payload,
    )
    return payload


def summarize_common_metrics(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    target_width: int,
    authoritative_targets_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute sum-L1 and raw-RILE endpoints for every grid cell.

    Executed cells pass the evaluation roster's authoritative targets. A row
    whose reported ``target_by_name`` differs from that source of truth is
    excluded rather than allowing backend-supplied gold to determine quality.
    Direct metric-only callers may omit the roster and provide trusted rows.
    """

    width = int(target_width)
    names = _target_names_for_width(width)
    joint_l1: list[float] = []
    raw_rile_errors: list[float] = []
    predicted_rile: list[float] = []
    target_rile: list[float] = []
    errors_by_target: dict[str, list[float]] = {name: [] for name in names}
    positive_errors_by_target: dict[str, list[float]] = {name: [] for name in names}
    positive_gold_by_target = {name: 0 for name in names}

    for row in prediction_rows:
        prediction = _exact_named_vector(
            row.get("prediction_by_target"),
            names,
        )
        reported_target = _exact_named_vector(row.get("target_by_name"), names)
        if authoritative_targets_by_id is None:
            target = reported_target
        else:
            tree_id = str(row.get("tree_id") or "")
            target = _exact_named_vector(
                authoritative_targets_by_id.get(tree_id),
                names,
            )
            if reported_target != target:
                target = None
        if prediction is None or target is None:
            continue
        joint_l1.append(sum(abs(prediction[name] - target[name]) for name in names))
        for name in names:
            error = abs(prediction[name] - target[name])
            errors_by_target[name].append(error)
            if target[name] > 0.0:
                positive_errors_by_target[name].append(error)
                positive_gold_by_target[name] += 1
        if width == 1:
            pred_raw = 200.0 * prediction[RILE_SINGLETON_NAME] - 100.0
            gold_raw = 200.0 * target[RILE_SINGLETON_NAME] - 100.0
        else:
            granularity = _granularity_for_width(width)
            pred_raw = manifesto_rile_component_readout(
                prediction,
                granularity=granularity,
            )
            gold_raw = manifesto_rile_component_readout(
                target,
                granularity=granularity,
            )
        predicted_rile.append(float(pred_raw))
        target_rile.append(float(gold_raw))
        raw_rile_errors.append(abs(float(pred_raw) - float(gold_raw)))

    return {
        "n": len(joint_l1),
        "joint_l1_definition": "sum_absolute_coordinate_error",
        "mean_joint_l1": _mean_or_none(joint_l1),
        "raw_rile_mae": _mean_or_none(raw_rile_errors),
        "mean_predicted_raw_rile": _mean_or_none(predicted_rile),
        "mean_target_raw_rile": _mean_or_none(target_rile),
        "per_target_metrics": {
            name: {
                "n": len(errors_by_target[name]),
                "mae": _mean_or_none(errors_by_target[name]),
                "conditional_positive_mae": _mean_or_none(positive_errors_by_target[name]),
                "gold_prevalence": (
                    None
                    if not errors_by_target[name]
                    else float(positive_gold_by_target[name] / len(errors_by_target[name]))
                ),
            }
            for name in names
        },
    }


def _cmp_record(
    *,
    doc_id: str,
    split: str,
    leaf_code_groups: Sequence[Sequence[str]],
    representation_path: str,
) -> TreeRecord:
    leaf_texts = [_codes_text(codes) for codes in leaf_code_groups]
    document_text = "\n".join(leaf_texts)
    all_codes = [code for group in leaf_code_groups for code in group]
    root_metadata = _cmp_metadata(all_codes)
    path_contract = _representation_path_contract(representation_path)
    common_metadata: dict[str, Any] = {
        "split": str(split),
        "representation": path_contract["representation"],
        "representation_path": path_contract["representation_path"],
        "topology_kind": path_contract["topology_kind"],
        "leaf_count": path_contract["leaf_count"],
        "merge_application_count": path_contract["merge_application_count"],
        "readout_path": path_contract["readout_path"],
        "leaf_g_application_count": path_contract["leaf_g_application_count"],
        "composition_present": path_contract["composition_present"],
        "fixture_kind": "tiny_authoritative_cmp_counts",
    }
    if path_contract["topology_kind"] == "singleton":
        root = TreeNode(
            node_id="root",
            unit_type="root",
            level=0,
            position=0,
            text=document_text,
            metadata=root_metadata,
        )
        return TreeRecord(
            tree_id=doc_id,
            text=document_text,
            root_label=None,
            nodes=(root,),
            metadata=common_metadata,
        )

    leaves = [
        TreeNode(
            node_id=f"leaf-{index}",
            unit_type="leaf",
            level=0,
            position=index,
            parent_id="merge-01" if index < 2 else "merge-23",
            text=leaf_texts[index],
            metadata=_cmp_metadata(codes),
        )
        for index, codes in enumerate(leaf_code_groups)
    ]
    left_codes = [code for group in leaf_code_groups[:2] for code in group]
    right_codes = [code for group in leaf_code_groups[2:] for code in group]
    merge_left = TreeNode(
        node_id="merge-01",
        unit_type="internal",
        level=1,
        position=0,
        parent_id="root",
        left_child_id="leaf-0",
        right_child_id="leaf-1",
        text="\n".join(leaf_texts[:2]),
        metadata=_cmp_metadata(left_codes),
    )
    merge_right = TreeNode(
        node_id="merge-23",
        unit_type="internal",
        level=1,
        position=1,
        parent_id="root",
        left_child_id="leaf-2",
        right_child_id="leaf-3",
        text="\n".join(leaf_texts[2:]),
        metadata=_cmp_metadata(right_codes),
    )
    root = TreeNode(
        node_id="root",
        unit_type="root",
        level=2,
        position=0,
        left_child_id="merge-01",
        right_child_id="merge-23",
        text=document_text,
        metadata=root_metadata,
    )
    return TreeRecord(
        tree_id=doc_id,
        text=document_text,
        root_label=None,
        nodes=(*leaves, merge_left, merge_right, root),
        metadata=common_metadata,
    )


def _attach_singleton_rile(record: TreeRecord) -> TreeRecord:
    """Attach the [0,1] RILE singleton to every node and the tree."""

    nodes: list[TreeNode] = []
    for node in record.nodes:
        metadata = dict(node.metadata or {})
        components = manifesto_rile_components_from_counts(
            metadata["cmp_counts"],
            total_non_header_mass=metadata["total_non_header_qsentences"],
            granularity=RILE_POLARITY_GRANULARITY,
        )
        normalized = 0.5 + 0.5 * (
            float(components["rile_right_share"]) - float(components["rile_left_share"])
        )
        metadata[RILE_SINGLETON_KEY] = {RILE_SINGLETON_NAME: float(normalized)}
        nodes.append(replace(node, metadata=metadata))

    hydrated = replace(record, nodes=tuple(nodes))
    root = hydrated.root()
    if root is None:
        raise ValueError(f"tree {record.tree_id!r} has no root")
    root_target = dict(root.metadata[RILE_SINGLETON_KEY])
    metadata = dict(hydrated.metadata or {})
    metadata[RILE_SINGLETON_KEY] = root_target
    metadata["rile_from_components"] = 200.0 * float(root_target[RILE_SINGLETON_NAME]) - 100.0
    return replace(hydrated, root_label=None, metadata=metadata)


def _prediction_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_ids: Sequence[str],
    target_names: tuple[str, ...],
    authoritative_targets_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = [str(tree_id) for tree_id in expected_ids]
    all_row_ids = [str(row.get("tree_id") or "") for row in rows]
    authoritative = {
        str(tree_id): _exact_named_vector(target, target_names)
        for tree_id, target in authoritative_targets_by_id.items()
    }
    invalid_authoritative_ids = sorted(
        tree_id for tree_id, target in authoritative.items() if target is None
    )
    if invalid_authoritative_ids:
        raise ValueError(
            "authoritative evaluation targets are not exact finite named vectors: "
            f"{invalid_authoritative_ids!r}"
        )
    if Counter(authoritative.keys()) != Counter(expected):
        raise ValueError("authoritative evaluation target roster disagrees with expected tree IDs")
    incomplete_row_ids = [
        str(row.get("tree_id") or "")
        for row in rows
        if not _complete_named_prediction(row, target_names)
    ]
    target_mismatch_ids = sorted(
        {
            str(row.get("tree_id") or "")
            for row in rows
            if str(row.get("tree_id") or "") in authoritative
            and _exact_named_vector(row.get("target_by_name"), target_names)
            != authoritative[str(row.get("tree_id") or "")]
        }
    )
    complete_ids = [
        str(row.get("tree_id") or "")
        for row in rows
        if _complete_named_prediction(row, target_names)
        and str(row.get("tree_id") or "") in authoritative
        and _exact_named_vector(row.get("target_by_name"), target_names)
        == authoritative[str(row.get("tree_id") or "")]
    ]

    expected_counts = Counter(expected)
    all_counts = Counter(all_row_ids)
    complete_counts = Counter(complete_ids)
    expected_duplicates = sorted(
        tree_id for tree_id, count in expected_counts.items() if count != 1
    )
    all_duplicates = sorted(tree_id for tree_id, count in all_counts.items() if count > 1)
    complete_duplicates = sorted(tree_id for tree_id, count in complete_counts.items() if count > 1)
    all_missing = sorted((expected_counts - all_counts).elements())
    all_unexpected = sorted((all_counts - expected_counts).elements())
    complete_missing = sorted((expected_counts - complete_counts).elements())
    complete_unexpected = sorted((complete_counts - expected_counts).elements())
    complete = bool(
        expected
        and len(rows) == len(expected)
        and not incomplete_row_ids
        and not expected_duplicates
        and not target_mismatch_ids
        and not all_duplicates
        and not all_missing
        and not all_unexpected
        and all_counts == expected_counts
    )
    return {
        "expected_n": len(expected),
        "observed_complete_vector_n": len(complete_ids),
        "expected_tree_ids_digest": stable_digest({"tree_ids": expected}),
        "observed_tree_ids_digest": stable_digest({"tree_ids": complete_ids}),
        "expected_tree_ids": expected,
        "observed_complete_tree_ids": complete_ids,
        "incomplete_row_tree_ids": incomplete_row_ids,
        "target_mismatch_tree_ids": target_mismatch_ids,
        "missing_tree_ids": complete_missing,
        "unexpected_tree_ids": complete_unexpected,
        "duplicate_tree_ids": complete_duplicates,
        "expected_duplicate_tree_ids": expected_duplicates,
        "all_row_n": len(rows),
        "all_row_tree_ids": all_row_ids,
        "all_row_count_matches_expected": len(rows) == len(expected),
        "all_row_duplicate_tree_ids": all_duplicates,
        "all_row_missing_tree_ids": all_missing,
        "all_row_unexpected_tree_ids": all_unexpected,
        "no_incomplete_rows": not incomplete_row_ids,
        "all_row_targets_match_authoritative": not target_mismatch_ids,
        "complete": complete,
        "criterion": (
            "exactly_one final row with one finite complete named prediction and "
            "the authoritative target vector per expected tree_id"
        ),
    }


def _authoritative_targets_by_id(
    records: Sequence[TreeRecord],
    *,
    target_names: tuple[str, ...],
    target_vector_key: str,
) -> dict[str, dict[str, float]]:
    """Read and validate the one trusted target vector for every eval tree."""

    targets: dict[str, dict[str, float]] = {}
    for raw_record in records:
        record = TreeRecord.from_value(raw_record)
        tree_id = str(record.tree_id)
        if not tree_id:
            raise ValueError("authoritative evaluation tree_id must be non-empty")
        if tree_id in targets:
            raise ValueError(f"duplicate authoritative evaluation tree_id {tree_id!r}")
        target = _exact_named_vector(
            dict(record.metadata or {}).get(str(target_vector_key)),
            target_names,
        )
        if target is None:
            raise ValueError(
                f"evaluation tree {tree_id!r} is missing an exact finite authoritative "
                f"target vector under {str(target_vector_key)!r}"
            )
        targets[tree_id] = target
    return targets


def comparison_report_schema() -> dict[str, Any]:
    """Return the package-owned common DSPy/FNO comparison schema."""

    return manifesto_semantic_forest_comparison_report_schema()


def _validated_executed_g_contract(
    value: Any,
    *,
    configured: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate and preserve the realized package ``g_contract`` payload."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("FitResult.summary.g_contract must be a mapping when present")
    payload = dict(value)
    required = (
        "mode",
        "operator",
        "fit_status",
        "g_update_count",
        "learned_this_run",
        "declared_leaf_count",
        "merge_application_count_per_tree",
        "leaf_g_materialized_application_count_per_tree",
        "topology_kind",
        "direct_readout_singleton",
        "summarized_singleton",
        "composition_present",
        "same_g_across_node_roles",
        "reduce_g_is_derived",
        "g_training_call_roles",
        "g_training_role_evidence_source",
        "merge_domain_training_observed",
        "shared_g_updated_with_merge_domain",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"FitResult.summary.g_contract is missing realized fields {missing!r}")

    mode = str(payload["mode"])
    configured_mode = str(configured["g_mode"])
    if mode != configured_mode:
        raise ValueError(
            "FitResult.summary.g_contract mode disagrees with configured g_mode: "
            f"{mode!r} != {configured_mode!r}"
        )
    if not isinstance(payload["operator"], str) or not payload["operator"].strip():
        raise ValueError("FitResult.summary.g_contract.operator must be non-empty")
    if not isinstance(payload["fit_status"], str) or not payload["fit_status"].strip():
        raise ValueError("FitResult.summary.g_contract.fit_status must be non-empty")

    update_count = _strict_finite_number(payload["g_update_count"])
    if update_count is None or int(update_count) != update_count or int(update_count) < 0:
        raise ValueError(
            "FitResult.summary.g_contract.g_update_count must be a non-negative integer"
        )
    update_count_int = int(update_count)
    expected_update_count = int(configured["expected_g_update_count"])
    if update_count_int != expected_update_count:
        raise ValueError(
            "FitResult.summary.g_contract g_update_count disagrees with the configured "
            f"schedule: {update_count_int} != {expected_update_count}"
        )

    learned_this_run = payload["learned_this_run"]
    if not isinstance(learned_this_run, bool):
        raise ValueError("FitResult.summary.g_contract.learned_this_run must be boolean")
    if learned_this_run != (update_count_int > 0):
        raise ValueError(
            "FitResult.summary.g_contract learned_this_run disagrees with g_update_count"
        )
    if mode != "learned" and learned_this_run:
        raise ValueError("a non-learned g_mode cannot report learned_this_run=true")
    if mode == "identity":
        expected_operator, expected_fit_status = "fixed_identity", "not_trainable"
    elif mode == "fixed":
        expected_operator = "fixed_nonidentity_or_family_owned"
        expected_fit_status = "not_trainable"
    elif configured.get("reuses_model_artifacts") is True:
        expected_operator = "learned_shared_reused"
        expected_fit_status = "reused_without_update"
    elif update_count_int > 0:
        expected_operator, expected_fit_status = "learned_shared", "fitted_this_run"
    else:
        expected_operator = "trainable_not_updated_this_run"
        expected_fit_status = "not_updated_this_run"
    if payload["operator"] != expected_operator:
        raise ValueError("FitResult.summary.g_contract operator disagrees with mode and updates")
    if payload["fit_status"] != expected_fit_status:
        raise ValueError("FitResult.summary.g_contract fit_status disagrees with mode and updates")
    for field in ("same_g_across_node_roles", "reduce_g_is_derived"):
        if payload[field] is not True:
            raise ValueError(f"FitResult.summary.g_contract.{field} must be true")
    expected_support_fields = {
        "g_training_call_roles": "expected_g_training_call_roles",
        "g_training_role_evidence_source": ("expected_g_training_role_evidence_source"),
        "merge_domain_training_observed": ("expected_merge_domain_training_observed"),
        "shared_g_updated_with_merge_domain": ("expected_shared_g_updated_with_merge_domain"),
    }
    for realized_field, expected_field in expected_support_fields.items():
        if payload[realized_field] != configured[expected_field]:
            raise ValueError(
                f"FitResult.summary.g_contract.{realized_field} disagrees with configured "
                "shared-g support contract"
            )
    _validate_executed_topology_contract(payload, configured=configured)
    return payload


def _comparison_row(
    cell: GridCell,
    *,
    rows: Sequence[Mapping[str, Any]],
    target_names: tuple[str, ...],
    coverage: Mapping[str, Any],
    common_metrics: Mapping[str, Any],
    status: str,
    fit_wall_seconds: float | None,
    measurement_started_at: str,
    measurement_finished_at: str,
    train: Sequence[TreeRecord],
    evaluation: Sequence[TreeRecord],
    fno_epochs: int,
    max_iterations: int,
    executed_g_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    g_contract = _g_contract(
        cell.family,
        cell.representation_path,
        requested_max_iterations=max_iterations,
        dspy_execution=cell.dspy_execution,
    )
    missing: dict[str, str] = {
        "compute_performance.inference_wall_seconds": (
            "treepo.fit does not expose an isolated inference clock"
        ),
        "compute_performance.docs_per_second": ("requires an isolated inference clock"),
        "compute_performance.peak_accelerator_memory_bytes": (
            "the package smoke runs on CPU and records no accelerator peak"
        ),
        "acquisition_resources.oracle_bundle_queries": (
            "fixture targets are derived from authoritative counts without a query-ledger event"
        ),
        "acquisition_resources.model_node_rows": (
            "the final result exposes root prediction rows, not a comparable "
            "family-neutral node-row ledger"
        ),
    }
    if executed_g_contract is None:
        reason = "execution produced no FitResult.summary.g_contract"
        for field in (
            "g_operator",
            "g_fit_status",
            "g_update_count",
            "learned_this_run",
            "g_training_call_roles",
            "g_training_role_evidence_source",
            "merge_domain_training_observed",
            "shared_g_updated_with_merge_domain",
        ):
            missing[f"identity.{field}"] = reason
    metrics_by_target = dict(common_metrics["per_target_metrics"])
    for name, metrics in metrics_by_target.items():
        for field in ("mae", "conditional_positive_mae", "gold_prevalence"):
            if metrics[field] is None:
                missing[f"quality.per_target_metrics.{name}.{field}"] = (
                    "no eligible complete evaluation rows"
                    if field != "conditional_positive_mae"
                    else "no positive gold rows for this target"
                )
    if common_metrics["mean_joint_l1"] is None:
        missing["quality.mean_joint_sum_l1"] = "no complete named prediction/target row"
    if common_metrics["raw_rile_mae"] is None:
        missing["quality.raw_published_rile_mae"] = "no complete named prediction/target row"
    if fit_wall_seconds is None:
        missing["compute_performance.fit_wall_seconds"] = (
            "fit call did not begin or its clock was unavailable"
        )

    materialized_nodes = sum(len(record.nodes) for record in (*train, *evaluation))
    underlying_annotations = sum(
        int(record.root().metadata["total_non_header_qsentences"])
        for record in (*train, *evaluation)
        if record.root() is not None
    )
    measurement_status = (
        "failed" if str(status) != "success" else ("partial" if missing else "observed")
    )
    row = {
        "schema_version": comparison_report_schema()["schema_version"],
        "cell_id": cell.cell_id,
        "identity": {
            "cell_id": cell.cell_id,
            "model_family": "dspy_llm" if cell.family == "dspy" else "fno",
            "representation_path": cell.representation_path,
            "topology_kind": g_contract["topology_kind"],
            "leaf_count": int(g_contract["leaf_count"]),
            "merge_application_count": int(g_contract["merge_application_count"]),
            "readout_path": g_contract["readout_path"],
            "leaf_g_application_count": int(g_contract["leaf_g_application_count"]),
            "composition_present": bool(g_contract["composition_present"]),
            "same_g_across_node_roles": True,
            "reduce_g_is_derived": True,
            "g_training_call_roles": (
                None
                if executed_g_contract is None
                else executed_g_contract["g_training_call_roles"]
            ),
            "g_training_role_evidence_source": (
                None
                if executed_g_contract is None
                else executed_g_contract["g_training_role_evidence_source"]
            ),
            "merge_domain_training_observed": (
                None
                if executed_g_contract is None
                else executed_g_contract["merge_domain_training_observed"]
            ),
            "shared_g_updated_with_merge_domain": (
                None
                if executed_g_contract is None
                else executed_g_contract["shared_g_updated_with_merge_domain"]
            ),
            "width": int(cell.target_width),
            "seed": int(cell.seed),
            "leaf_scale": f"leafs{int(g_contract['leaf_count']):03d}",
            "g_mode": g_contract["g_mode"],
            "reuses_model_artifacts": bool(g_contract["reuses_model_artifacts"]),
            "g_operator": (
                None if executed_g_contract is None else executed_g_contract["operator"]
            ),
            "g_fit_status": (
                None if executed_g_contract is None else executed_g_contract["fit_status"]
            ),
            "g_update_count": (
                None if executed_g_contract is None else executed_g_contract["g_update_count"]
            ),
            "learned_this_run": (
                None if executed_g_contract is None else executed_g_contract["learned_this_run"]
            ),
            "schedule": g_contract["schedule"],
            "requested_max_iterations": g_contract["requested_max_iterations"],
            "effective_max_iterations": g_contract["effective_max_iterations"],
            "eval_split": "test",
            "roster_digest": stable_digest({"tree_ids": list(coverage["expected_tree_ids"])}),
            "config_digest": stable_digest(
                {
                    "grid_version": GRID_VERSION,
                    "family": cell.family,
                    "representation": g_contract["representation"],
                    "representation_path": cell.representation_path,
                    "topology_kind": g_contract["topology_kind"],
                    "target_names": list(target_names),
                    "seed": int(cell.seed),
                    "fno_epochs": int(fno_epochs),
                    "g_mode": g_contract["g_mode"],
                    "schedule": g_contract["schedule"],
                    "requested_max_iterations": g_contract["requested_max_iterations"],
                    "effective_max_iterations": g_contract["effective_max_iterations"],
                    "dspy_execution": (cell.dspy_execution if cell.family == "dspy" else None),
                }
            ),
        },
        "quality": {
            "raw_published_rile_mae": common_metrics["raw_rile_mae"],
            "mean_joint_sum_l1": common_metrics["mean_joint_l1"],
            "per_target_metrics": metrics_by_target,
            "exact_prediction_coverage": {
                key: coverage[key]
                for key in (
                    "expected_n",
                    "observed_complete_vector_n",
                    "expected_tree_ids_digest",
                    "observed_tree_ids_digest",
                    "missing_tree_ids",
                    "unexpected_tree_ids",
                    "duplicate_tree_ids",
                    "incomplete_row_tree_ids",
                    "target_mismatch_tree_ids",
                    "complete",
                )
            },
        },
        "compute_performance": {
            "fit_wall_seconds": fit_wall_seconds,
            "inference_wall_seconds": None,
            "docs_per_second": None,
            "peak_accelerator_memory_bytes": None,
            "fit_wall_scope": "complete treepo.fit call including its evaluation",
        },
        "acquisition_resources": {
            "train_documents": len(train),
            "evaluation_documents": len(evaluation),
            "oracle_bundle_queries": None,
            "underlying_atomic_annotations": underlying_annotations,
            "returned_coordinate_values": (
                int(coverage["observed_complete_vector_n"]) * len(target_names)
            ),
            "materialized_node_coordinate_values": (materialized_nodes * len(target_names)),
            "model_root_rows": len(rows),
            "model_node_rows": None,
        },
        "measurement_status": {
            "status": measurement_status,
            "missing_field_reasons": missing,
            "measurement_started_at": measurement_started_at,
            "measurement_finished_at": measurement_finished_at,
        },
    }
    return validate_manifesto_semantic_forest_comparison_row(
        row,
        target_names=target_names,
    )


def _failed_cell_report(
    cell: GridCell,
    *,
    error: str,
    fno_epochs: int,
    max_iterations: int,
) -> dict[str, Any]:
    g_contract = _g_contract(
        cell.family,
        cell.representation_path,
        requested_max_iterations=max_iterations,
        dspy_execution=cell.dspy_execution,
    )
    target_names = _target_names_for_width(cell.target_width)
    expected = list(_test_document_ids())
    train_raw, test_raw = make_cmp_count_records(cell.representation_path)
    train, _fragment = prepare_target_records(train_raw, cell.target_width)
    evaluation, fragment = prepare_target_records(test_raw, cell.target_width)
    authoritative_targets = _authoritative_targets_by_id(
        evaluation,
        target_names=target_names,
        target_vector_key=str(fragment["backend_config"]["target_vector_key"]),
    )
    coverage = _prediction_coverage(
        (),
        expected_ids=expected,
        target_names=target_names,
        authoritative_targets_by_id=authoritative_targets,
    )
    empty_metrics = summarize_common_metrics(
        (),
        target_width=cell.target_width,
        authoritative_targets_by_id=authoritative_targets,
    )
    now = _utc_now()
    return {
        **cell.to_dict(requested_max_iterations=max_iterations),
        "status": "failed",
        "prediction_coverage": coverage,
        "common_metrics": empty_metrics,
        "treepo_metrics": {},
        "fit_summary": {},
        "shared_g_updated_with_merge_domain": None,
        "executed_g_contract": None,
        "manifest_path": None,
        "comparison_row": _comparison_row(
            cell,
            rows=(),
            target_names=target_names,
            coverage=coverage,
            common_metrics=empty_metrics,
            status="failed",
            fit_wall_seconds=None,
            measurement_started_at=now,
            measurement_finished_at=now,
            train=train,
            evaluation=evaluation,
            fno_epochs=fno_epochs,
            max_iterations=max_iterations,
            executed_g_contract=None,
        ),
        "family_telemetry": _family_telemetry(
            cell,
            fno_epochs=fno_epochs,
            max_iterations=max_iterations,
        ),
        "fit_contract": {
            "public_api": "treepo.fit",
            "single_fit_call": True,
            **g_contract,
            **_target_api_contract(),
        },
        "error": error,
    }


def _family_telemetry(
    cell: GridCell,
    *,
    fno_epochs: int,
    max_iterations: int,
) -> dict[str, Any]:
    g_contract = _g_contract(
        cell.family,
        cell.representation_path,
        requested_max_iterations=max_iterations,
        dspy_execution=cell.dspy_execution,
    )
    if cell.family == "dspy":
        base = {
            "cell_id": cell.cell_id,
            "model_family": "dspy_llm",
            "dspy_execution": _resolve_dspy_execution(cell.dspy_execution),
            **g_contract,
            "target_api_contract": _target_api_contract(),
        }
        if cell.dspy_execution == "offline_fixture":
            return {
                **base,
                "lm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "parse_retries": 0,
                "transport_retries": 0,
                "estimated_cost": 0.0,
                "cost_currency": None,
                "fixture_backend": FIXTURE_BACKEND,
            }
        return {
            **base,
            "lm_calls": None,
            "input_tokens": None,
            "output_tokens": None,
            "parse_retries": None,
            "transport_retries": None,
            "estimated_cost": None,
            "cost_currency": None,
            "missing_field_reasons": {
                "lm_calls": "treepo.fit does not expose DSPy call telemetry",
                "input_tokens": "treepo.fit does not expose DSPy token telemetry",
                "output_tokens": "treepo.fit does not expose DSPy token telemetry",
                "parse_retries": "treepo.fit does not expose DSPy retry telemetry",
                "transport_retries": "treepo.fit does not expose DSPy retry telemetry",
                "estimated_cost": "requires provider token and pricing telemetry",
            },
        }
    return {
        "cell_id": cell.cell_id,
        "model_family": "fno",
        **g_contract,
        "target_api_contract": _target_api_contract(),
        "parameter_count": None,
        "trainable_parameter_count": None,
        "epochs_completed": None,
        "optimizer_steps": None,
        "gradient_updates": None,
        "configured_epochs_per_iteration": int(fno_epochs),
        "configured_max_iterations": int(g_contract["effective_max_iterations"]),
        "missing_field_reasons": {
            "parameter_count": "fit summary does not expose a parameter ledger",
            "trainable_parameter_count": ("fit summary does not expose a parameter ledger"),
            "epochs_completed": "fit summary does not expose completed epochs",
            "optimizer_steps": "fit summary does not expose optimizer steps",
            "gradient_updates": "fit summary does not expose gradient updates",
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _complete_named_prediction(
    row: Mapping[str, Any],
    target_names: tuple[str, ...],
) -> bool:
    return bool(
        target_names
        and _exact_named_vector(
            row.get("prediction_by_target"),
            target_names,
        )
        is not None
        and _exact_named_vector(row.get("target_by_name"), target_names) is not None
    )


def _exact_named_vector(
    value: Any,
    target_names: tuple[str, ...],
) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    by_name: dict[str, Any] = {}
    for raw_name, component in value.items():
        name = str(raw_name)
        if name in by_name:
            return None
        by_name[name] = component
    if set(by_name) != set(target_names):
        return None
    result: dict[str, float] = {}
    for name in target_names:
        component = _strict_finite_number(by_name[name])
        if component is None:
            return None
        result[name] = component
    return result


def _strict_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _cmp_metadata(codes: Sequence[str]) -> dict[str, Any]:
    counts = Counter(str(code) for code in codes)
    return {
        "cmp_counts": dict(sorted(counts.items())),
        "total_non_header_qsentences": len(codes),
    }


def _codes_text(codes: Sequence[str]) -> str:
    return " ".join(f"manifesto statement cmp_{str(code)}" for code in codes)


def _target_names(fragment: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(target.target_name) for target in fragment["oracle_targets"])


def _target_names_for_width(target_width: int) -> tuple[str, ...]:
    width = int(target_width)
    if width == 1:
        return (RILE_SINGLETON_NAME,)
    return manifesto_rile_component_target_names(_granularity_for_width(width))


def _granularity_for_width(target_width: int) -> str:
    width = int(target_width)
    if width == 3:
        return RILE_POLARITY_GRANULARITY
    if width == 57:
        return RILE_CMP56_GRANULARITY
    raise ValueError(f"only K=1, K=3, and K=57 are defined, got K={width}")


def _final_prediction_rows(result: Any) -> list[dict[str, Any]]:
    history = list(getattr(result, "history", None) or ())
    if not history:
        return []
    last = history[-1]
    extra = last.get("extra") if isinstance(last, Mapping) else getattr(last, "extra", None)
    rows = dict(extra or {}).get("prediction_rows") or ()
    return [dict(row) for row in rows]


def _substrate(family: str) -> str:
    if str(family) == "dspy":
        return "llm"
    if str(family) == "fno":
        return "neural_operator_fno"
    raise ValueError(f"unknown family {family!r}")


def _optimizer(
    family: str,
    *,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
) -> str:
    if str(family) == "dspy":
        return (
            "dspy_optimizer"
            if _resolve_dspy_execution(dspy_execution) == "learned"
            else "none_offline_fixture"
        )
    if str(family) == "fno":
        return "gradient"
    raise ValueError(f"unknown family {family!r}")


def _representation_path_contract(representation_path: str) -> dict[str, Any]:
    specs = {
        "full_doc_direct": ("full_doc", "singleton", 1, 0, "f_direct", 0, False),
        "ctree_base_summary": ("ctree", "singleton", 1, 0, "f_after_g", 1, False),
        "ctree_recursive": ("ctree", "recursive", 4, 3, "f_after_reduce_g", 4, True),
    }
    resolved = str(representation_path)
    try:
        (
            representation,
            topology_kind,
            leaf_count,
            merge_count,
            readout,
            leaf_g_count,
            composition,
        ) = specs[resolved]
    except KeyError as exc:
        raise ValueError(f"unsupported representation_path {representation_path!r}") from exc
    return {
        "representation_path": resolved,
        "representation": representation,
        "topology_kind": topology_kind,
        "leaf_count": leaf_count,
        "merge_application_count": merge_count,
        "readout_path": readout,
        "leaf_g_application_count": leaf_g_count,
        "composition_present": composition,
    }


def _execution_path(
    family: str,
    representation_path: str,
    *,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
) -> str:
    resolved_family = _resolve_family(family)
    resolved_path = _representation_path_contract(representation_path)["representation_path"]
    if resolved_family == "dspy":
        if _resolve_dspy_execution(dspy_execution) == "learned":
            return {
                "full_doc_direct": "full_document_direct_optimizer_backed_dspy_f",
                "ctree_base_summary": "ctree_single_leaf_shared_learned_dspy_g",
                "ctree_recursive": "ctree_recursive_shared_learned_dspy_g",
            }[resolved_path]
        return {
            "full_doc_direct": "full_document_direct_offline_fixture_program",
            "ctree_base_summary": "ctree_single_leaf_weighted_analytic_fixture_g",
            "ctree_recursive": "ctree_recursive_weighted_analytic_fixture_g",
        }[resolved_path]
    return {
        "full_doc_direct": "full_document_single_unit_fno",
        "ctree_base_summary": "singleton_shared_fno_g_then_f",
        "ctree_recursive": "recursive_fold_shared_fno_g_then_f",
    }[resolved_path]


def _g_mode(
    family: str,
    representation_path: str,
    *,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
) -> str:
    resolved_family = _resolve_family(family)
    resolved_path = _representation_path_contract(representation_path)["representation_path"]
    if resolved_path == "full_doc_direct":
        return "identity"
    if resolved_family == "dspy" and _resolve_dspy_execution(dspy_execution) == "offline_fixture":
        return "fixed"
    return "learned"


def _g_contract(
    family: str,
    representation_path: str,
    *,
    requested_max_iterations: int,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
) -> dict[str, Any]:
    requested = int(requested_max_iterations)
    resolved_family = _resolve_family(family)
    resolved_dspy_execution = _resolve_dspy_execution(dspy_execution)
    path_contract = _representation_path_contract(representation_path)
    mode = _g_mode(
        resolved_family,
        representation_path,
        dspy_execution=resolved_dspy_execution,
    )
    _validate_grid_path_g_mode(
        resolved_family,
        representation_path,
        mode,
        dspy_execution=resolved_dspy_execution,
    )
    schedule = "fg" if mode == "learned" else "f"
    reuses_model_artifacts = representation_path == "ctree_base_summary"
    effective = (
        0
        if reuses_model_artifacts
        else max(0, requested)
        if mode == "learned"
        else int(requested > 0)
    )
    expected_g_update_count = effective // 2 if mode == "learned" else 0
    train_g_called = expected_g_update_count > 0
    roles = ["leaf"] if train_g_called else []
    merge_domain = bool(train_g_called and path_contract["composition_present"])
    if merge_domain:
        roles.append("merge")

    if mode == "identity":
        implementation = "identity_shared_g"
        learner = "none_identity"
        learning_evidence = "none_identity"
    elif mode == "fixed":
        implementation = "denominator_weighted_analytic_fixture_g"
        learner = "none_analytic_wiring"
        learning_evidence = "analytic_wiring_only_not_learned_shared_g"
    elif resolved_family == "dspy":
        # The same program and learner IDs apply to singleton and recursive
        # calls. Only the available training-call domains differ by topology.
        implementation = "treepo_dspy_shared_g"
        learner = "treepo_dspy_shared_fg_learner"
        learning_evidence = "realized_dspy_compiler_update_required"
    else:
        implementation = "treepo_fno_shared_g"
        learner = "treepo_fno_shared_g_learner"
        learning_evidence = "realized_fit_update_required"

    if not train_g_called:
        expected_evidence_source = "no_train_g_calls"
    elif resolved_family == "dspy":
        expected_evidence_source = "dspy_compiler_mixed_examples"
    else:
        expected_evidence_source = "g_artifact_gradient_path_presence"

    return {
        **path_contract,
        "g_mode": mode,
        "g_trainable": mode == "learned",
        "expected_g_update_count": expected_g_update_count,
        "learned_g_expected": expected_g_update_count > 0,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
        "model_scope": (
            "direct_control_separate_from_shared_ctree_pair"
            if representation_path == "full_doc_direct"
            else "one_f_g_pair_per_target_width_and_family"
        ),
        "reuses_model_artifacts": reuses_model_artifacts,
        "shared_model_source_path": (
            "ctree_recursive" if reuses_model_artifacts else representation_path
        ),
        "expected_g_training_call_roles": roles,
        "expected_g_training_role_evidence_source": expected_evidence_source,
        "expected_merge_domain_training_observed": merge_domain,
        "expected_shared_g_updated_with_merge_domain": bool(
            expected_g_update_count > 0 and merge_domain
        ),
        "schedule": schedule,
        "requested_max_iterations": requested,
        "effective_max_iterations": effective,
        "implementation": implementation,
        "learner": learner,
        "learning_evidence": learning_evidence,
    }


def _validate_grid_path_g_mode(
    family: str,
    representation_path: str,
    mode: str,
    *,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
) -> None:
    """Enforce modes chosen by this experiment, not by the neutral schema."""

    resolved_family = _resolve_family(family)
    resolved_path = _representation_path_contract(representation_path)["representation_path"]
    resolved_dspy_execution = _resolve_dspy_execution(dspy_execution)
    expected = (
        "identity"
        if resolved_path == "full_doc_direct"
        else "fixed"
        if resolved_family == "dspy" and resolved_dspy_execution == "offline_fixture"
        else "learned"
    )
    if mode != expected:
        raise ValueError(
            f"grid path {resolved_family}/{resolved_path} with "
            f"dspy_execution={resolved_dspy_execution!r} requires g_mode={expected!r}"
        )


def _validate_executed_topology_contract(
    payload: Mapping[str, Any],
    *,
    configured: Mapping[str, Any],
) -> None:
    count_fields = {
        "declared_leaf_count": "leaf_count",
        "merge_application_count_per_tree": "merge_application_count",
        "leaf_g_materialized_application_count_per_tree": "leaf_g_application_count",
    }
    for realized_field, configured_field in count_fields.items():
        value = payload[realized_field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"FitResult.summary.g_contract.{realized_field} must be a non-negative integer"
            )
        expected = int(configured[configured_field])
        if value != expected:
            raise ValueError(
                f"FitResult.summary.g_contract.{realized_field} disagrees with configured topology: {value} != {expected}"
            )

    topology_kind = str(payload["topology_kind"])
    if topology_kind != str(configured["topology_kind"]):
        raise ValueError(
            "FitResult.summary.g_contract.topology_kind disagrees with configured topology"
        )

    expected_flags = {
        "direct_readout_singleton": configured["representation_path"] == "full_doc_direct",
        "summarized_singleton": configured["representation_path"] == "ctree_base_summary",
        "composition_present": bool(configured["composition_present"]),
    }
    for field, expected in expected_flags.items():
        value = payload[field]
        if not isinstance(value, bool):
            raise ValueError(f"FitResult.summary.g_contract.{field} must be boolean")
        if value != expected:
            raise ValueError(
                f"FitResult.summary.g_contract.{field} disagrees with configured topology"
            )


def _target_api_contract() -> dict[str, Any]:
    return {
        "target_api": "ordered_named_vector",
        "target_widths": list(TARGET_WIDTHS),
        "k_changes_only": "ordered_target_catalog_and_output_width",
        "point_metric": "sum_absolute_coordinate_error",
        "k1_scalar_api_branch": False,
        "raw_rile_readout": "reporting_only",
    }


def _require_exact_leaf_count(
    records: Sequence[TreeRecord],
    *,
    expected: int,
    representation_path: str,
) -> None:
    for record in records:
        observed = len(record.leaves())
        if observed != int(expected):
            raise ValueError(
                f"representation_path {representation_path!r} requires exactly {int(expected)} leaves; "
                f"tree {record.tree_id!r} has {observed}"
            )


def _common_provenance() -> dict[str, Any]:
    return {
        "evidence_class": EVIDENCE_CLASS,
        "scientific_claim": False,
        "fixture_data": True,
        "fixture_data_kind": "tiny_authoritative_cmp_counts",
    }


def _dspy_execution_provenance(dspy_execution: str) -> dict[str, Any]:
    resolved = _resolve_dspy_execution(dspy_execution)
    if resolved == "offline_fixture":
        return {
            "execution_evidence_scope": "explicit_offline_family_interface_fixture",
            "fixture_backend": FIXTURE_BACKEND,
            "live_llm_call": False,
            "dspy_optimization_executed": False,
            "learned_shared_llm_g_evidence": False,
            "analytic_wiring_only": True,
            "satisfies_live_llm_empirical_requirement": False,
            "replacement_requirement": REPLACEMENT_REQUIREMENT,
        }
    return {
        "execution_evidence_scope": "optimizer_backed_dspy_package_fixture",
        # Plan/configuration fields are intentionally not promoted to realized
        # measurements. Per-cell FitResult contracts remain authoritative.
        "live_llm_call": None,
        "dspy_optimization_executed": None,
        "expected_dspy_optimization": True,
        "learned_shared_llm_g_evidence": None,
        "analytic_wiring_only": False,
        "satisfies_live_llm_empirical_requirement": None,
        "realized_evidence_location": "cells[*].executed_g_contract",
    }


def _grid_warning(
    *,
    families: Sequence[str],
    dspy_execution: str,
) -> str:
    if "dspy" not in {str(family) for family in families}:
        return _COMMON_WARNING
    if _resolve_dspy_execution(dspy_execution) == "offline_fixture":
        return f"{OFFLINE_FIXTURE_WARNING} {_COMMON_WARNING}"
    return PLAN_WARNING


def _grid_summary(
    *,
    families: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
) -> dict[str, Any]:
    completed = sum(cell.get("status") == "success" for cell in cells)
    dspy_ctree_mode = "learned" if _resolve_dspy_execution(dspy_execution) == "learned" else "fixed"
    return {
        "target_widths": list(TARGET_WIDTHS),
        "representations": ["full_doc", "ctree"],
        "representation_paths": list(REPRESENTATION_PATHS),
        "families": list(families),
        "expected_cells": 9 * len(families),
        "completed_cells": completed,
        "failed_cells": len(cells) - completed,
        "primary_metric": "raw_rile_mae",
        "joint_vector_metric": "mean_joint_l1",
        "target_api_contract": _target_api_contract(),
        "g_mode_contract": {
            "full_doc_direct": {"dspy": "identity", "fno": "identity"},
            "ctree_base_summary": {"dspy": dspy_ctree_mode, "fno": "learned"},
            "ctree_recursive": {"dspy": dspy_ctree_mode, "fno": "learned"},
        },
        "representation_path_contracts": {
            path: _representation_path_contract(path) for path in REPRESENTATION_PATHS
        },
        "singleton_contract": {
            "target_name": RILE_SINGLETON_NAME,
            "bounds": [0.0, 1.0],
            "raw_readout": "200*rile_normalized-100",
            "execution_path": "named_vector",
        },
    }


def _resolve_dspy_execution(value: str) -> str:
    resolved = str(value).strip().lower()
    if resolved not in DSPY_EXECUTIONS:
        raise ValueError(f"dspy_execution must be one of {DSPY_EXECUTIONS!r}, got {value!r}")
    return resolved


def _validated_learned_dspy_backend_config(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        raise ValueError(
            "dspy_execution='learned' requires dspy_backend_config with an "
            "optimizer plus usable lm_config or injected f/g program adapters; "
            "use dspy_execution='offline_fixture' only for the deterministic smoke"
        )
    if not isinstance(value, Mapping):
        raise TypeError("dspy_backend_config must be a mapping")
    payload = dict(value)
    optimizer = payload.get("optimizer")
    disabled_optimizer = (
        optimizer is None
        or optimizer is False
        or (
            isinstance(optimizer, str)
            and optimizer.strip().lower() in {"", "none", "disabled", "off"}
        )
        or (isinstance(optimizer, Mapping) and not optimizer)
    )
    if disabled_optimizer:
        raise ValueError("dspy_execution='learned' requires backend_config['optimizer'] != 'none'")

    lm_config = payload.get("lm_config")
    if lm_config is not None and not isinstance(lm_config, Mapping):
        raise TypeError("dspy_backend_config['lm_config'] must be a mapping")
    lm = dict(lm_config or {})
    has_model = bool(str(lm.get("model") or "").strip())
    has_endpoint_or_current_lm = bool(
        lm.get("api_base")
        or lm.get("base_url")
        or lm.get("lm")
        or lm.get("current_lm")
        or lm.get("use_current_dspy_lm") is True
        or payload.get("dspy_lm")
    )
    has_usable_lm = has_model and has_endpoint_or_current_lm
    has_f_program = payload.get("f_program") is not None or payload.get("dspy_program") is not None
    has_g_program = payload.get("g_program") is not None
    if not has_usable_lm and not (has_f_program and has_g_program):
        raise ValueError(
            "learned DSPy execution requires either lm_config with model plus "
            "endpoint/current-LM selection, or both f_program (legacy "
            "dspy_program) and g_program adapters"
        )
    return payload


def _cell_output_dir(
    root: Path,
    *,
    target_width: int,
    representation_path: str,
    family: str,
    seed: int,
    dspy_execution: str,
) -> Path:
    base = root / f"k{int(target_width):02d}" / str(representation_path) / str(family)
    if str(family) == "dspy":
        base = base / _resolve_dspy_execution(dspy_execution)
    return base / f"seed_{int(seed)}"


def _family_grid_report_filename(
    family: str,
    *,
    dspy_execution: str,
) -> str:
    resolved = _resolve_family(family)
    if resolved == "dspy" and _resolve_dspy_execution(dspy_execution) == "offline_fixture":
        return "semantic_forest_dspy_grid_offline_fixture_smoke.json"
    return f"semantic_forest_{resolved}_grid.json"


def _complete_grid_report_filename(*, dspy_execution: str) -> str:
    if _resolve_dspy_execution(dspy_execution) == "offline_fixture":
        return "semantic_forest_grid_offline_fixture_smoke.json"
    return "semantic_forest_grid.json"


def _resolve_family(family: str) -> str:
    resolved = str(family).strip().lower()
    if resolved not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES!r}, got {family!r}")
    return resolved


def _validate_cell(cell: GridCell) -> None:
    if int(cell.target_width) not in TARGET_WIDTHS:
        raise ValueError(f"unsupported target width {cell.target_width!r}")
    if cell.representation_path not in REPRESENTATION_PATHS:
        raise ValueError(f"unsupported representation_path {cell.representation_path!r}")
    _resolve_family(cell.family)
    _resolve_dspy_execution(cell.dspy_execution)


def _test_document_ids() -> tuple[str, ...]:
    return tuple(doc_id for doc_id, split, _groups in _DOCUMENTS if split == "test")


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_dspy_backend_config_file(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".toml":
        import tomllib

        payload = tomllib.loads(source.read_text(encoding="utf-8"))
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("--dspy-config must be a .json or .toml file")
    if not isinstance(payload, Mapping):
        raise TypeError("DSPy config file root must be a mapping")
    nested = payload.get("backend_config", payload)
    if not isinstance(nested, Mapping):
        raise TypeError("DSPy config backend_config must be a mapping")
    return dict(nested)


def _plan_payload(
    output_dir: Path,
    *,
    selected_family: str,
    seed: int,
    max_iterations: int = 3,
    dspy_execution: str = DEFAULT_DSPY_EXECUTION,
    dspy_config_supplied: bool = False,
) -> dict[str, Any]:
    families = FAMILIES if selected_family == "both" else (_resolve_family(selected_family),)
    resolved_dspy_execution = _resolve_dspy_execution(dspy_execution)
    family_grids: dict[str, dict[str, Any]] = {}
    schema = comparison_report_schema()
    for family in families:
        family_payload = {
            "family": family,
            "substrate": _substrate(family),
            "optimizer": _optimizer(
                family,
                dspy_execution=resolved_dspy_execution,
            ),
            "expected_cells": 9,
            "warning": _grid_warning(
                families=(family,),
                dspy_execution=resolved_dspy_execution,
            ),
            **_common_provenance(),
            "comparison_report_schema": schema,
            "comparison_rows_are_not_plan_measurements": True,
            "g_contracts": {
                representation_path: _g_contract(
                    family,
                    representation_path,
                    requested_max_iterations=max_iterations,
                    dspy_execution=resolved_dspy_execution,
                )
                for representation_path in REPRESENTATION_PATHS
            },
            "target_api_contract": _target_api_contract(),
            "cells": [
                cell.to_dict(requested_max_iterations=max_iterations)
                for cell in build_family_grid_plan(
                    output_dir,
                    family,
                    seed=seed,
                    dspy_execution=resolved_dspy_execution,
                )
            ],
        }
        if family == "dspy":
            family_payload.update(
                {
                    "dspy_execution": resolved_dspy_execution,
                    "dspy_runtime_config_required": resolved_dspy_execution == "learned",
                    "dspy_runtime_config_supplied": bool(dspy_config_supplied),
                    **_dspy_execution_provenance(resolved_dspy_execution),
                }
            )
        family_grids[family] = family_payload

    payload = {
        "version": GRID_VERSION,
        "selected_family": selected_family,
        "expected_cells": 9 * len(families),
        "warning": _grid_warning(
            families=families,
            dspy_execution=resolved_dspy_execution,
        ),
        **_common_provenance(),
        "comparison_report_schema": schema,
        "comparison_report_schema_identical_across_families": all(
            family_grids[family]["comparison_report_schema"] == schema for family in families
        ),
        "comparison_rows_are_not_plan_measurements": True,
        "target_api_contract": _target_api_contract(),
        "family_grids": family_grids,
        "cells": [cell for family in families for cell in family_grids[family]["cells"]],
    }
    if "dspy" in families:
        payload.update(
            {
                "dspy_execution": resolved_dspy_execution,
                "dspy_runtime_config_required": resolved_dspy_execution == "learned",
                "dspy_runtime_config_supplied": bool(dspy_config_supplied),
                **_dspy_execution_provenance(resolved_dspy_execution),
            }
        )
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manifesto_semantic_forest_grid"),
    )
    parser.add_argument(
        "--family",
        choices=("both", *FAMILIES),
        default="both",
    )
    parser.add_argument(
        "--dspy-execution",
        choices=DSPY_EXECUTIONS,
        default=DEFAULT_DSPY_EXECUTION,
        help=(
            "learned uses optimizer-backed f/shared-g programs; offline_fixture "
            "is an explicit deterministic interface smoke"
        ),
    )
    parser.add_argument(
        "--dspy-config",
        type=Path,
        default=None,
        help=(
            "JSON/TOML backend_config for learned DSPy execution; programmatic "
            "callers may inject optimizer/program objects directly"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fno-epochs", type=int, default=1)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help=(
            "inclusive alternating iteration index; 3 gives the learned "
            "f->g->f sequence (identity/fixed g paths elide g slots)"
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dspy_backend_config = (
        None if args.dspy_config is None else _load_dspy_backend_config_file(args.dspy_config)
    )
    if args.plan_only:
        payload = _plan_payload(
            args.output_dir,
            selected_family=args.family,
            seed=args.seed,
            max_iterations=args.max_iterations,
            dspy_execution=args.dspy_execution,
            dspy_config_supplied=dspy_backend_config is not None,
        )
        mode_suffix = f"_{args.dspy_execution}" if args.family in {"both", "dspy"} else ""
        path = args.output_dir / (
            f"semantic_forest_grid{mode_suffix}_plan.json"
            if args.family == "both"
            else f"semantic_forest_{args.family}_grid{mode_suffix}_plan.json"
        )
        _write_json(path, payload)
        # Compatibility alias for callers that predate the explicit execution
        # suffix. The payload itself still records dspy_execution, so the alias
        # cannot erase the learned-versus-offline distinction.
        compatibility_path = args.output_dir / (
            "semantic_forest_grid_plan.json"
            if args.family == "both"
            else f"semantic_forest_{args.family}_grid_plan.json"
        )
        if compatibility_path != path:
            _write_json(compatibility_path, payload)
        print(
            f"status=planned family={args.family} cells={payload['expected_cells']} output={path}"
        )
        return 0

    if args.family == "both":
        payload = run_complete_grid(
            args.output_dir,
            seed=args.seed,
            fno_epochs=args.fno_epochs,
            max_iterations=args.max_iterations,
            dspy_execution=args.dspy_execution,
            dspy_backend_config=dspy_backend_config,
        )
    else:
        payload = run_family_grid(
            args.family,
            args.output_dir,
            seed=args.seed,
            fno_epochs=args.fno_epochs,
            max_iterations=args.max_iterations,
            dspy_execution=args.dspy_execution,
            dspy_backend_config=dspy_backend_config,
        )
    failed = int(payload["grid"]["failed_cells"])
    print(
        "status="
        + ("success" if failed == 0 else "failed")
        + f" family={args.family}"
        + f" completed={payload['grid']['completed_cells']}"
        + f"/{payload['grid']['expected_cells']}"
        + f" output={args.output_dir}"
    )
    return 0 if failed == 0 else 1


__all__ = [
    "DEFAULT_DSPY_EXECUTION",
    "DSPY_EXECUTIONS",
    "EVIDENCE_CLASS",
    "FAMILIES",
    "FIXTURE_BACKEND",
    "FixtureOracleDSPyProgram",
    "GRID_VERSION",
    "GridCell",
    "OFFLINE_FIXTURE_WARNING",
    "PLAN_WARNING",
    "REPLACEMENT_REQUIREMENT",
    "REPRESENTATIONS",
    "RILE_SINGLETON_KEY",
    "RILE_SINGLETON_NAME",
    "RILE_SINGLETON_ORACLE_ID",
    "TARGET_WIDTHS",
    "build_family_grid_plan",
    "build_grid_plan",
    "comparison_report_schema",
    "fit_grid_cell",
    "main",
    "make_cmp_count_records",
    "prepare_target_records",
    "run_complete_grid",
    "run_family_grid",
    "summarize_common_metrics",
]


if __name__ == "__main__":
    raise SystemExit(main())
