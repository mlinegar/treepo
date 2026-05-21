from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Protocol, Sequence

import numpy as np

from treepo._research.ctreepo.sim.core.full_tree_ipw_grid import grid_rows_from_payload
from treepo._research.ctreepo.sim.manifest import read_manifest_jsonl
from treepo._research.ctreepo.sim.local_law_backfill import load_or_backfill_local_law_payload
from treepo._research.ctreepo.sim.local_law_learnability import (
    LocalLawRunSummary,
    PolicyRole,
    split_metric_views,
    selected_policy_role,
)


ExpectationStatus = Literal["pass", "warn", "fail", "not_applicable"]
TrendDirection = Literal["decreasing", "increasing", "flat"]

FAMILY_MARKOV = "markov_ops_count"
FAMILY_SEGMENT_LDA = "segment_lda_ops_weight_recovery"
FAMILY_CTREE = "segmented_lda_ctreepo"
FAMILY_MERGEABLE = "mergeable_ablation"
FAMILY_LOCAL_LAW = "local_law_learnability"
VALID_FAMILIES: tuple[str, ...] = (
    FAMILY_MARKOV,
    FAMILY_SEGMENT_LDA,
    FAMILY_CTREE,
    FAMILY_MERGEABLE,
    FAMILY_LOCAL_LAW,
)

STATUS_PRIORITY: Dict[str, int] = {
    "fail": 0,
    "warn": 1,
    "pass": 2,
    "not_applicable": 3,
}

FULL_DOC_MARKOV_DIAGNOSTIC_SIMULATION = "markov_full_doc_anchor_diagnostics"
FULL_DOC_MARKOV_LADDER_SIMULATION = "markov_full_doc_anchor_ladder"
FULL_TREE_IPW_MARKOV_SIMULATION = "markov_full_tree_ipw_grid"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_text(path: Path) -> str:
    return str(path).lower()


def _path_parts_lower(path: Path) -> tuple[str, ...]:
    return tuple(str(part).lower() for part in path.parts)


def _path_contains_any(path: Path, needles: Sequence[str]) -> bool:
    text = _path_text(path)
    return any(str(needle).lower() in text for needle in needles)


def _finite(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _safe_list(values: Iterable[object]) -> List[float]:
    return [float(v) for v in (_finite(x) for x in values) if v is not None]


def _safe_median(values: Iterable[object]) -> float:
    xs = np.asarray(_safe_list(values), dtype=np.float64)
    return float(np.median(xs)) if xs.size else float("nan")


def _safe_mean(values: Iterable[object]) -> float:
    xs = _safe_list(values)
    return float(sum(xs) / float(len(xs))) if xs else float("nan")


def _safe_percentile(values: Iterable[object], q: float) -> float:
    xs = np.asarray(_safe_list(values), dtype=np.float64)
    return float(np.percentile(xs, float(q))) if xs.size else float("nan")


def _metric_scale(values: Sequence[float]) -> float:
    xs = [abs(float(x)) for x in values if math.isfinite(float(x))]
    return float(max([1.0, *xs])) if xs else 1.0


def _stringify(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def _role_value(role: object) -> str:
    value = getattr(role, "value", role)
    return str(value)


def _slug(parts: Dict[str, object]) -> str:
    keys = sorted(parts.keys())
    return "|".join(f"{k}={_stringify(parts[k])}" for k in keys)


def _objective_component_weights(values: object) -> Dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    out: Dict[str, float] = {}
    for name, value in values.items():
        v = _finite(value)
        if v is not None:
            out[str(name)] = float(v)
    return out


def _objective_metadata(payload: Mapping[str, Any]) -> Dict[str, object]:
    objective = payload.get("objective")
    if not isinstance(objective, Mapping):
        return {}
    out: Dict[str, object] = {}
    for key in (
        "name",
        "kind",
        "optimized_against",
        "weighting_scheme",
        "selection_metric_name",
        "interprets_lambda_as",
    ):
        value = objective.get(key)
        if value is not None and value != "":
            out[f"objective_{key}"] = str(value)
    component_weights = _objective_component_weights(objective.get("component_weights"))
    if component_weights:
        out["objective_component_weights"] = component_weights
    objective_metadata = objective.get("metadata")
    if isinstance(objective_metadata, Mapping) and objective_metadata:
        out["objective_metadata"] = dict(objective_metadata)
    return out


def _markov_objective_metadata(payload: Mapping[str, Any]) -> Dict[str, object]:
    objective = payload.get("objective")
    if not isinstance(objective, Mapping):
        return {}
    composite = objective.get("composite_objective")
    composite = composite if isinstance(composite, Mapping) else {}
    composite_meta = composite.get("metadata")
    composite_meta = composite_meta if isinstance(composite_meta, Mapping) else {}

    weighting_scheme = (
        objective.get("weighting_scheme")
        or composite.get("weighting_scheme")
        or ""
    )
    parameterization = (
        objective.get("parameterization")
        or composite_meta.get("parameterization")
        or ""
    )
    local_law_weight = _finite(objective.get("local_law_weight"))
    if local_law_weight is None:
        local_law_weight = _finite(composite_meta.get("local_law_weight"))
    local_law_active_raw = objective.get("local_law_active")
    if isinstance(local_law_active_raw, bool):
        local_law_active = local_law_active_raw
    else:
        local_law_active = local_law_weight is not None and float(local_law_weight) > 0.0
    theorem_relevant = bool(local_law_active) and str(parameterization) == "formal_local_law_weight"

    out: Dict[str, object] = {
        "objective_weighting_scheme": str(weighting_scheme or "unknown"),
        "objective_parameterization": str(parameterization or "unknown"),
        "theorem_relevant": bool(theorem_relevant),
        "objective_local_law_active": bool(local_law_active),
    }
    if local_law_weight is not None:
        out["objective_local_law_weight"] = float(local_law_weight)
    return out


def _full_doc_markov_run_rows(
    path: Path,
    payload: Mapping[str, Any],
) -> List["NormalizedRow"]:
    rows: List[NormalizedRow] = []
    for run in list(payload.get("runs") or []):
        benchmark = str(run.get("benchmark", "")).strip()
        cell_id = str(run.get("cell_id", "")).strip()
        baseline_family = str(run.get("baseline_family", "")).strip()
        scenario = _slug(
            {
                "surface": FULL_DOC_MARKOV_DIAGNOSTIC_SIMULATION,
                "benchmark": benchmark or "unknown",
                "cell_id": cell_id or benchmark or "default",
            }
        )
        train_docs = _finite(run.get("train_doc_count"))
        theorem_relevance = bool(run.get("theorem_relevance", False))
        test_identity = _slug(
            {
                "cell_id": cell_id or benchmark or "default",
                "bundle_source": str(run.get("bundle_source", "")).strip() or "none",
                "val_corpus_signature": str(run.get("val_corpus_signature", "")).strip()
                or "none",
                "test_corpus_signature": str(run.get("test_corpus_signature", "")).strip()
                or "none",
            }
        )
        metadata: Dict[str, object] = {
            "surface": FULL_DOC_MARKOV_DIAGNOSTIC_SIMULATION,
            "benchmark": benchmark,
            "cell_id": cell_id,
            "baseline_family": baseline_family,
            "bundle_source": str(run.get("bundle_source", "")),
            "train_corpus_signature": str(run.get("train_corpus_signature", "")),
            "val_corpus_signature": str(run.get("val_corpus_signature", "")),
            "test_corpus_signature": str(run.get("test_corpus_signature", "")),
            "parameterization": str(run.get("parameterization", "")),
            "optimization_root_weight": _finite(run.get("optimization_root_weight")),
            "local_law_c1_weight": _finite(run.get("local_law_c1_weight")),
            "local_law_c2_weight": _finite(run.get("local_law_c2_weight")),
            "local_law_c3_weight": _finite(run.get("local_law_c3_weight")),
            "task_objective_weight_source": str(
                run.get("task_objective_weight_source", "")
            ),
            "c2_metric_kind": str(run.get("c2_metric_kind", "")),
            "c2_proxy_metric_kind": str(run.get("c2_proxy_metric_kind", "")),
            "comparison_semantics": str(run.get("comparison_semantics", "")),
            "comparison_semantics_label": str(
                run.get("comparison_semantics_label", "")
            ),
            "legacy_semantics": bool(run.get("legacy_semantics", False)),
            "legacy_semantics_reason": str(run.get("legacy_semantics_reason", "")),
            "semantics_version": str(run.get("semantics_version", "")),
            "backend_name": str(run.get("backend_name", "")),
            "backend_package": str(run.get("backend_package", "")),
            "backend_version": str(run.get("backend_version", "")),
            "operator_class": str(run.get("operator_class", "")),
            "operator_evidence_status": str(run.get("operator_evidence_status", "")),
            "theorem_relevance": theorem_relevance,
            "objective_weights_active": bool(
                run.get("objective_weights_active", False)
            ),
            "n_regimes": _finite(run.get("n_regimes")),
        }
        rows.append(
            NormalizedRow(
                family=FAMILY_MARKOV,
                scenario=scenario,
                seed=int(run["seed"]) if run.get("seed") is not None else None,
                method=baseline_family,
                x_axis_name="train_docs",
                x_axis_value=float(train_docs if train_docs is not None else float("nan")),
                secondary_axis_name=None,
                secondary_axis_value=None,
                metric_name="root_mae",
                metric_value=float(run.get("test_root_mae", float("nan"))),
                doc_scale_tokens=None,
                leaf_tokens=None,
                leaves_per_doc=None,
                oracle_budget_fraction=None,
                train_docs=train_docs,
                evidence_status=str(run.get("operator_evidence_status", "APPROX_AUDITED")),
                source_path=str(path.resolve()),
                test_identity=test_identity,
                metadata=metadata,
            )
        )
    return rows


def _full_doc_markov_ladder_rows(
    path: Path,
    payload: Mapping[str, Any],
) -> List["NormalizedRow"]:
    rows: List[NormalizedRow] = []
    for stage in list(payload.get("stages") or []):
        scenario = _slug(
            {
                "surface": FULL_DOC_MARKOV_LADDER_SIMULATION,
                "observed_token_profile": str(
                    stage.get("observed_token_profile", "unknown")
                ),
            }
        )
        train_docs = _finite(stage.get("train_docs"))
        metadata: Dict[str, object] = {
            "surface": FULL_DOC_MARKOV_LADDER_SIMULATION,
            "stage_name": str(stage.get("stage_name", "")),
            "source": str(stage.get("source", "")),
            "description": str(stage.get("description", "")),
            "reference_only": bool(stage.get("reference_only", False)),
            "observed_token_profile": str(stage.get("observed_token_profile", "")),
            "bundle_source": str(stage.get("bundle_source", "")),
            "summary_json": str(stage.get("summary_json", "")),
            "train_docs": _finite(stage.get("train_docs")),
            "val_docs": _finite(stage.get("val_docs")),
            "test_docs": _finite(stage.get("test_docs")),
            "state_dim": _finite(stage.get("state_dim")),
            "hidden_dim": _finite(stage.get("hidden_dim")),
            "n_epochs": _finite(stage.get("n_epochs")),
            "batch_size": _finite(stage.get("batch_size")),
            "lr": _finite(stage.get("lr")),
            "weight_decay": _finite(stage.get("weight_decay")),
            "anchor_gap_to_ridge": _finite(stage.get("anchor_gap_to_ridge")),
            "backend_name": str(stage.get("doc_sequence_backend_name", "")),
            "backend_package": str(stage.get("doc_sequence_backend_package", "")),
            "backend_version": str(stage.get("doc_sequence_backend_version", "")),
            "operator_class": str(stage.get("doc_sequence_operator_class", "")),
            "operator_evidence_status": str(
                stage.get("doc_sequence_operator_evidence_status", "")
            ),
            "theorem_relevance": bool(
                stage.get("doc_sequence_theorem_relevance", False)
            ),
            "objective_weights_active": bool(
                stage.get("doc_sequence_objective_weights_active", False)
            ),
        }
        rows.append(
            NormalizedRow(
                family=FAMILY_MARKOV,
                scenario=scenario,
                seed=None,
                method=str(stage.get("stage_name", "")),
                x_axis_name="train_docs",
                x_axis_value=float(train_docs if train_docs is not None else float("nan")),
                secondary_axis_name=None,
                secondary_axis_value=None,
                metric_name="doc_sequence_test_root_mae",
                metric_value=float(stage.get("doc_sequence_test_root_mae", float("nan"))),
                doc_scale_tokens=None,
                leaf_tokens=None,
                leaves_per_doc=None,
                oracle_budget_fraction=None,
                train_docs=train_docs,
                evidence_status=str(
                    stage.get("doc_sequence_operator_evidence_status", "PROXY_ONLY")
                ),
                source_path=str(path.resolve()),
                test_identity=None,
                metadata=metadata,
            )
        )
    return rows


def _full_tree_ipw_grid_rows(
    path: Path,
    payload: Mapping[str, Any],
) -> List["NormalizedRow"]:
    rows: List[NormalizedRow] = []
    base_config = dict(payload.get("base_config") or {})
    bundle_metadata = dict(payload.get("bundle_metadata") or {})
    leaf_tokens = _finite(base_config.get("fixed_leaf_tokens"))
    train_docs = _finite(bundle_metadata.get("train_docs"))
    for cell in grid_rows_from_payload(payload):
        doc_sequence_train_fraction = _finite(cell.get("doc_sequence_train_fraction"))
        root_only_train_fraction = _finite(cell.get("root_only_train_fraction"))
        scenario = _slug(
            {
                "surface": FULL_TREE_IPW_MARKOV_SIMULATION,
                "doc_sequence_train_fraction": (
                    doc_sequence_train_fraction
                    if doc_sequence_train_fraction is not None
                    else "na"
                ),
                "root_only_train_fraction": (
                    root_only_train_fraction
                    if root_only_train_fraction is not None
                    else "na"
                ),
            }
        )
        metadata = {
            "surface": FULL_TREE_IPW_MARKOV_SIMULATION,
            "estimand_name": str(
                cell.get("estimand_name", "realized_full_tree_node_mean_loss")
            ),
            "population_kind": str(
                cell.get("population_kind", "realized_tree_nodes")
            ),
            "sampling_design": str(
                cell.get("sampling_design", "bernoulli_realized_node_sampling")
            ),
            "propensity_field": str(cell.get("propensity_field", "unit_propensity")),
            "document_channel": str(
                cell.get("document_channel", "always_observed_document_top_loss")
            ),
            "node_channel": str(
                cell.get("node_channel", "sampled_realized_tree_nodes")
            ),
            "estimator_families": list(
                cell.get("estimator_families") or ["naive", "ht", "hajek"]
            ),
            "ci_semantics": str(cell.get("ci_semantics", "point_estimation_only")),
            **dict(cell),
        }
        rows.append(
            NormalizedRow(
                family=FAMILY_MARKOV,
                scenario=scenario,
                seed=None,
                method=str(cell.get("regime", "full_tree_ipw_grid")),
                x_axis_name="p_leaf",
                x_axis_value=float(cell.get("p_leaf", float("nan"))),
                secondary_axis_name="p_internal",
                secondary_axis_value=_finite(cell.get("p_internal")),
                metric_name="root_mae",
                metric_value=float(cell.get("test_root_mae", float("nan"))),
                doc_scale_tokens=None,
                leaf_tokens=leaf_tokens,
                leaves_per_doc=None,
                oracle_budget_fraction=None,
                train_docs=train_docs,
                evidence_status="APPROX_AUDITED",
                source_path=str(path.resolve()),
                test_identity=_slug(
                    {
                        "train_corpus_signature": str(
                            bundle_metadata.get("train_corpus_signature", "")
                        ),
                        "val_corpus_signature": str(
                            bundle_metadata.get("val_corpus_signature", "")
                        ),
                        "test_corpus_signature": str(
                            bundle_metadata.get("test_corpus_signature", "")
                        ),
                    }
                ),
                metadata=metadata,
            )
        )
    return rows


def _finding(
    *,
    kind: str,
    title: str,
    status: ExpectationStatus,
    scenario: str,
    metric: str,
    method: str,
    observed_summary: Dict[str, object],
    thresholds: Dict[str, object],
    supporting_rows: Sequence[NormalizedRow],
) -> "ExpectationFinding":
    return ExpectationFinding(
        kind=kind,
        title=title,
        status=status,
        family=FAMILY_MARKOV,
        scenario=scenario,
        metric=metric,
        method=method,
        direction="flat",
        observed_summary=observed_summary,
        thresholds=thresholds,
        supporting_rows=_supporting_rows(list(supporting_rows)),
    )


@dataclass(frozen=True)
class ExpectationConfig:
    seed_aggregate: Literal["median", "mean"] = "median"
    min_effect_rel: float = 0.10
    min_effect_abs_scale: float = 0.02
    floor_flatness_rel: float = 0.05
    separation_rel: float = 0.10
    adjacent_tolerance: float = 0.01
    adjacent_success_rate_min: float = 0.70
    ceiling_epsilon_exact: float = 1e-8
    ceiling_epsilon_float: float = 1e-6
    calibrated_regression_rel: float = 0.05


@dataclass(frozen=True)
class NormalizedRow:
    family: str
    scenario: str
    seed: Optional[int]
    method: str
    x_axis_name: str
    x_axis_value: float
    secondary_axis_name: Optional[str]
    secondary_axis_value: Optional[float]
    metric_name: str
    metric_value: float
    doc_scale_tokens: Optional[float]
    leaf_tokens: Optional[float]
    leaves_per_doc: Optional[float]
    oracle_budget_fraction: Optional[float]
    train_docs: Optional[float]
    evidence_status: str
    source_path: str
    test_identity: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def value_for(self, name: str) -> Optional[float]:
        if name == "train_docs":
            return self.train_docs
        if name == "oracle_budget_fraction":
            return self.oracle_budget_fraction
        if name == "leaf_tokens":
            return self.leaf_tokens
        if name == "leaves_per_doc":
            return self.leaves_per_doc
        if name == "doc_scale_tokens":
            return self.doc_scale_tokens
        if name == self.x_axis_name:
            return self.x_axis_value
        if self.secondary_axis_name == name:
            return self.secondary_axis_value
        return _finite(self.metadata.get(name))

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "scenario": self.scenario,
            "seed": self.seed,
            "method": self.method,
            "x_axis_name": self.x_axis_name,
            "x_axis_value": self.x_axis_value,
            "secondary_axis_name": self.secondary_axis_name,
            "secondary_axis_value": self.secondary_axis_value,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "doc_scale_tokens": self.doc_scale_tokens,
            "leaf_tokens": self.leaf_tokens,
            "leaves_per_doc": self.leaves_per_doc,
            "oracle_budget_fraction": self.oracle_budget_fraction,
            "train_docs": self.train_docs,
            "evidence_status": self.evidence_status,
            "source_path": self.source_path,
            "test_identity": self.test_identity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "NormalizedRow":
        return cls(
            family=str(d.get("family", "")),
            scenario=str(d.get("scenario", "")),
            seed=int(d["seed"]) if d.get("seed") is not None else None,
            method=str(d.get("method", "")),
            x_axis_name=str(d.get("x_axis_name", "")),
            x_axis_value=float(d.get("x_axis_value", float("nan"))),
            secondary_axis_name=(
                str(d["secondary_axis_name"]) if d.get("secondary_axis_name") is not None else None
            ),
            secondary_axis_value=(
                float(d["secondary_axis_value"]) if d.get("secondary_axis_value") is not None else None
            ),
            metric_name=str(d.get("metric_name", "")),
            metric_value=float(d.get("metric_value", float("nan"))),
            doc_scale_tokens=_finite(d.get("doc_scale_tokens")),
            leaf_tokens=_finite(d.get("leaf_tokens")),
            leaves_per_doc=_finite(d.get("leaves_per_doc")),
            oracle_budget_fraction=_finite(d.get("oracle_budget_fraction")),
            train_docs=_finite(d.get("train_docs")),
            evidence_status=str(d.get("evidence_status", "")),
            source_path=str(d.get("source_path", "")),
            test_identity=str(d["test_identity"]) if d.get("test_identity") is not None else None,
            metadata=dict(d.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class TrendAssessment:
    status: ExpectationStatus
    direction: TrendDirection
    axis_name: str
    axis_values: List[float]
    central_values: List[float]
    p10_values: List[float]
    p90_values: List[float]
    endpoint_delta: float
    relative_endpoint_delta: float
    theil_sen_slope: float
    adjacent_success_rate: float
    metric_scale: float
    min_effect_abs: float
    test_identity_shared: Optional[bool]
    note: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "TrendAssessment":
        return cls(
            status=str(d.get("status", "not_applicable")),
            direction=str(d.get("direction", "flat")),
            axis_name=str(d.get("axis_name", "")),
            axis_values=[float(x) for x in (d.get("axis_values", []) or [])],
            central_values=[float(x) for x in (d.get("central_values", []) or [])],
            p10_values=[float(x) for x in (d.get("p10_values", []) or [])],
            p90_values=[float(x) for x in (d.get("p90_values", []) or [])],
            endpoint_delta=float(d.get("endpoint_delta", float("nan"))),
            relative_endpoint_delta=float(d.get("relative_endpoint_delta", float("nan"))),
            theil_sen_slope=float(d.get("theil_sen_slope", float("nan"))),
            adjacent_success_rate=float(d.get("adjacent_success_rate", float("nan"))),
            metric_scale=float(d.get("metric_scale", 1.0)),
            min_effect_abs=float(d.get("min_effect_abs", 0.0)),
            test_identity_shared=(
                bool(d["test_identity_shared"]) if d.get("test_identity_shared") is not None else None
            ),
            note=str(d.get("note", "")),
        )


@dataclass(frozen=True)
class ExpectationFinding:
    kind: str
    title: str
    status: ExpectationStatus
    family: str
    scenario: str
    metric: str
    method: str
    direction: TrendDirection
    observed_summary: Dict[str, object]
    thresholds: Dict[str, object]
    supporting_rows: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "family": self.family,
            "scenario": self.scenario,
            "metric": self.metric,
            "method": self.method,
            "direction": self.direction,
            "observed_summary": dict(self.observed_summary),
            "thresholds": dict(self.thresholds),
            "supporting_rows": list(self.supporting_rows),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "ExpectationFinding":
        return cls(
            kind=str(d.get("kind", "")),
            title=str(d.get("title", "")),
            status=str(d.get("status", "not_applicable")),
            family=str(d.get("family", "")),
            scenario=str(d.get("scenario", "")),
            metric=str(d.get("metric", "")),
            method=str(d.get("method", "")),
            direction=str(d.get("direction", "flat")),
            observed_summary=dict(d.get("observed_summary", {}) or {}),
            thresholds=dict(d.get("thresholds", {}) or {}),
            supporting_rows=list(d.get("supporting_rows", []) or []),
        )


@dataclass(frozen=True)
class ExpectationReport:
    input_root: Optional[str]
    manifest: Optional[str]
    families_scanned: List[str]
    rows_scanned: int
    expectations: List[ExpectationFinding]
    summary: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_root": self.input_root,
            "manifest": self.manifest,
            "families_scanned": list(self.families_scanned),
            "rows_scanned": int(self.rows_scanned),
            "expectations": [e.to_dict() for e in self.expectations],
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "ExpectationReport":
        return cls(
            input_root=str(d["input_root"]) if d.get("input_root") is not None else None,
            manifest=str(d["manifest"]) if d.get("manifest") is not None else None,
            families_scanned=[str(x) for x in (d.get("families_scanned", []) or [])],
            rows_scanned=int(d.get("rows_scanned", 0)),
            expectations=[
                ExpectationFinding.from_dict(x) for x in (d.get("expectations", []) or [])
            ],
            summary=dict(d.get("summary", {}) or {}),
        )


def _expectation_summary(expectations: Sequence[ExpectationFinding]) -> Dict[str, object]:
    return {
        "n_pass": int(sum(1 for e in expectations if e.status == "pass")),
        "n_warn": int(sum(1 for e in expectations if e.status == "warn")),
        "n_fail": int(sum(1 for e in expectations if e.status == "fail")),
        "n_not_applicable": int(sum(1 for e in expectations if e.status == "not_applicable")),
        "families_with_failures": sorted({e.family for e in expectations if e.status == "fail"}),
        "highest_priority_findings": [
            {
                "status": e.status,
                "family": e.family,
                "title": e.title,
                "scenario": e.scenario,
            }
            for e in list(expectations)[:10]
        ],
    }


def merge_expectation_reports(
    reports: Sequence[ExpectationReport],
    *,
    input_root: Optional[str] = None,
    manifest: Optional[str] = None,
) -> ExpectationReport:
    merged_expectations: List[ExpectationFinding] = []
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    for report in reports:
        for finding in report.expectations:
            key = (
                str(finding.family),
                str(finding.scenario),
                str(finding.title),
                str(finding.metric),
                str(finding.method),
                str(finding.kind),
                str(finding.status),
            )
            if key in seen:
                continue
            seen.add(key)
            merged_expectations.append(finding)
    merged_expectations.sort(key=lambda x: (STATUS_PRIORITY.get(x.status, 99), x.family, x.title))
    return ExpectationReport(
        input_root=input_root,
        manifest=manifest,
        families_scanned=sorted({fam for report in reports for fam in report.families_scanned}),
        rows_scanned=int(sum(int(report.rows_scanned) for report in reports)),
        expectations=merged_expectations,
        summary=_expectation_summary(merged_expectations),
    )


class FamilyAdapter(Protocol):
    family: str

    def can_load(self, path: Path) -> bool:
        ...

    def load_rows(self, path: Path) -> List[NormalizedRow]:
        ...

    def build_expectations(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        ...


def _test_identity_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    cfg = payload.get("config", {}) or {}
    topic_meta = payload.get("topic_meta", {}) or {}
    for value in (
        cfg.get("test_identity"),
        cfg.get("test_split_id"),
        cfg.get("test_seed"),
        payload.get("test_identity"),
        topic_meta.get("corpus_signature_test"),
    ):
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    if isinstance(cfg, dict) and cfg:
        # Fallback: many simulators generate the test split deterministically from
        # the seed and generation config but do not serialize an explicit test ID.
        # Strip training-budget knobs so train-doc trend checks can still verify
        # that the compared rows share the same underlying test split.
        excluded = {
            "train_docs",
            "n_books_train",
            "train_size",
            "n_train",
            "audit_fraction",
            "audit_fixed_nodes",
            "audit_scale",
            "topic_phi_docs",
            "calibration_leaf_query_rate",
            "eval_leaf_query_rate",
            "eval_internal_query_rate",
            "chunk_budget",
            "budget",
        }
        derived = {str(k): cfg.get(k) for k in sorted(cfg.keys()) if str(k) not in excluded}
        if derived:
            return "derived:" + json.dumps(derived, sort_keys=True, separators=(",", ":"))
    return None


def _estimate_doc_scale_from_segments(cfg: Dict[str, Any]) -> Optional[float]:
    min_segments = _finite(cfg.get("min_segments"))
    max_segments = _finite(cfg.get("max_segments"))
    min_seg_tokens = _finite(cfg.get("min_seg_tokens"))
    max_seg_tokens = _finite(cfg.get("max_seg_tokens"))
    if None in {min_segments, max_segments, min_seg_tokens, max_seg_tokens}:
        return None
    return float((min_segments + max_segments) * 0.5 * (min_seg_tokens + max_seg_tokens) * 0.5)


def _evidence_status_for_method(family: str, method: str) -> str:
    m = str(method)
    if m in {"exact", "oracle_tree", "one_pass_reference", "one_pass_oracle", "budget_one_pass_reference"}:
        return "THEOREM_BACKED"
    if m.startswith("naive_"):
        return "PROXY_ONLY"
    if m in {"estimated_uncalibrated", "estimated_calibrated", "estimated_calibrated_budgeted", "learned", "ridge"}:
        return "APPROX_AUDITED"
    if m.startswith("grid_") or m.startswith("full_model_") or m.startswith("budget_full_model_"):
        return "APPROX_AUDITED"
    if m.startswith("budget_wrong_chunker") or m == "right_rule_wrong_chunker":
        return "PROXY_ONLY"
    return "APPROX_AUDITED"


def _row_subset(rows: Sequence[NormalizedRow], **filters: object) -> List[NormalizedRow]:
    out: List[NormalizedRow] = []
    for row in rows:
        ok = True
        for key, expected in filters.items():
            actual: object
            if hasattr(row, key):
                actual = getattr(row, key)
            else:
                actual = row.metadata.get(key)
            if actual != expected:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def _supporting_rows(rows: Sequence[NormalizedRow]) -> List[Dict[str, object]]:
    return [r.to_dict() for r in rows]


def _aggregate_rows(
    rows: Sequence[NormalizedRow],
    *,
    axis_name: str,
    config: ExpectationConfig,
) -> tuple[List[float], List[float], List[float], List[float]]:
    buckets: Dict[float, List[float]] = {}
    for row in rows:
        axis_val = row.value_for(axis_name)
        if axis_val is None:
            continue
        metric_val = _finite(row.metric_value)
        if metric_val is None:
            continue
        buckets.setdefault(float(axis_val), []).append(float(metric_val))
    xs = sorted(buckets.keys())
    if not xs:
        return [], [], [], []
    central: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    for x in xs:
        vals = buckets[x]
        if config.seed_aggregate == "mean":
            c = _safe_mean(vals)
        else:
            c = _safe_median(vals)
        central.append(float(c))
        lows.append(float(_safe_percentile(vals, 10.0)))
        highs.append(float(_safe_percentile(vals, 90.0)))
    return xs, central, lows, highs


def _theil_sen_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    slopes: List[float] = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = float(xs[j]) - float(xs[i])
            if abs(dx) <= 1e-12:
                continue
            slopes.append((float(ys[j]) - float(ys[i])) / dx)
    return float(np.median(np.asarray(slopes, dtype=np.float64))) if slopes else float("nan")


def assess_trend(
    rows: Sequence[NormalizedRow],
    *,
    axis_name: str,
    direction: TrendDirection,
    config: ExpectationConfig,
) -> TrendAssessment:
    xs, central, lows, highs = _aggregate_rows(rows, axis_name=axis_name, config=config)
    if len(xs) < 2:
        return TrendAssessment(
            status="not_applicable",
            direction=direction,
            axis_name=axis_name,
            axis_values=xs,
            central_values=central,
            p10_values=lows,
            p90_values=highs,
            endpoint_delta=float("nan"),
            relative_endpoint_delta=float("nan"),
            theil_sen_slope=float("nan"),
            adjacent_success_rate=float("nan"),
            metric_scale=1.0,
            min_effect_abs=0.0,
            test_identity_shared=None,
            note="need at least two distinct axis values",
        )

    metric_scale = _metric_scale(central)
    start = float(central[0])
    end = float(central[-1])
    endpoint_delta = end - start
    rel_endpoint = abs(endpoint_delta) / max(1e-12, abs(start))
    min_effect_abs = max(
        float(config.min_effect_abs_scale) * float(metric_scale),
        float(config.min_effect_rel) * max(abs(start), 1e-12),
    )
    slope = _theil_sen_slope(xs, central)
    tol = float(config.adjacent_tolerance) * float(metric_scale)

    adjacent_good = 0
    total_adjacent = max(0, len(central) - 1)
    for i in range(total_adjacent):
        dy = float(central[i + 1]) - float(central[i])
        if direction == "decreasing":
            if dy <= tol:
                adjacent_good += 1
        elif direction == "increasing":
            if dy >= -tol:
                adjacent_good += 1
        else:
            if abs(dy) <= tol:
                adjacent_good += 1
    adjacent_rate = float(adjacent_good) / float(total_adjacent) if total_adjacent > 0 else 1.0

    if direction == "decreasing":
        endpoint_sign_ok = endpoint_delta < 0.0
        slope_ok = math.isfinite(slope) and slope < 0.0
        effect_ok = (start - end) >= min_effect_abs or rel_endpoint >= float(config.min_effect_rel)
    elif direction == "increasing":
        endpoint_sign_ok = endpoint_delta > 0.0
        slope_ok = math.isfinite(slope) and slope > 0.0
        effect_ok = (end - start) >= min_effect_abs or rel_endpoint >= float(config.min_effect_rel)
    else:
        endpoint_sign_ok = abs(endpoint_delta) <= min_effect_abs
        slope_ok = (not math.isfinite(slope)) or abs(slope) <= max(min_effect_abs, tol)
        effect_ok = abs(endpoint_delta) <= max(min_effect_abs, tol)

    test_ids = {str(r.test_identity) for r in rows if r.test_identity}
    test_identity_shared: Optional[bool]
    if axis_name == "train_docs":
        test_identity_shared = len(test_ids) == 1 if test_ids else None
    else:
        test_identity_shared = None

    status: ExpectationStatus
    note = ""
    if direction == "flat":
        if endpoint_sign_ok and slope_ok:
            status = "pass"
            note = "series is effectively flat"
        elif abs(endpoint_delta) <= 2.0 * min_effect_abs:
            status = "warn"
            note = "series drift is small but not clearly flat"
        else:
            status = "fail"
            note = "series shows a material trend when flatness was expected"
    else:
        if not endpoint_sign_ok or not slope_ok or not effect_ok:
            status = "fail"
            if not effect_ok and endpoint_sign_ok and slope_ok:
                note = "trend direction is correct but the improvement is too small"
            else:
                note = "trend direction does not match expectation"
        elif adjacent_rate < float(config.adjacent_success_rate_min):
            status = "warn"
            note = "global trend is correct but local reversals are frequent"
        else:
            status = "pass"
            note = "trend direction and effect size match expectation"

    if status == "pass" and axis_name == "train_docs" and test_identity_shared is not True:
        status = "warn"
        note = "trend is directionally correct but the test-set identity is missing or inconsistent"

    return TrendAssessment(
        status=status,
        direction=direction,
        axis_name=axis_name,
        axis_values=[float(x) for x in xs],
        central_values=[float(y) for y in central],
        p10_values=[float(y) for y in lows],
        p90_values=[float(y) for y in highs],
        endpoint_delta=float(endpoint_delta),
        relative_endpoint_delta=float(rel_endpoint),
        theil_sen_slope=float(slope),
        adjacent_success_rate=float(adjacent_rate),
        metric_scale=float(metric_scale),
        min_effect_abs=float(min_effect_abs),
        test_identity_shared=test_identity_shared,
        note=note,
    )


def _anchor_slice(
    rows: Sequence[NormalizedRow],
    *,
    primary_field: str,
    secondary_field: Optional[str] = None,
) -> List[NormalizedRow]:
    if not rows:
        return []
    primary_vals = [r.value_for(primary_field) for r in rows]
    finite_primary = [float(v) for v in primary_vals if v is not None]
    if not finite_primary:
        return []
    max_primary = max(finite_primary)
    subset = [r for r in rows if r.value_for(primary_field) is not None and abs(float(r.value_for(primary_field)) - max_primary) <= 1e-12]
    if secondary_field is None or not subset:
        return subset
    secondary_vals = [r.value_for(secondary_field) for r in subset]
    finite_secondary = [float(v) for v in secondary_vals if v is not None]
    if not finite_secondary:
        return subset
    max_secondary = max(finite_secondary)
    return [
        r
        for r in subset
        if r.value_for(secondary_field) is not None
        and abs(float(r.value_for(secondary_field)) - max_secondary) <= 1e-12
    ]


def _aggregate_metric(rows: Sequence[NormalizedRow], *, config: ExpectationConfig) -> float:
    vals = [float(r.metric_value) for r in rows if math.isfinite(float(r.metric_value))]
    if not vals:
        return float("nan")
    if config.seed_aggregate == "mean":
        return float(np.mean(np.asarray(vals, dtype=np.float64)))
    return float(np.median(np.asarray(vals, dtype=np.float64)))


@dataclass(frozen=True)
class CeilingExpectation:
    family: str
    scenario: str
    method: str
    metric: str
    epsilon: float
    title: str

    def evaluate(self, rows: Sequence[NormalizedRow]) -> ExpectationFinding:
        vals = [float(r.metric_value) for r in rows if math.isfinite(float(r.metric_value))]
        max_abs = float(max(abs(v) for v in vals)) if vals else float("nan")
        if not vals:
            status: ExpectationStatus = "not_applicable"
            note = "no matching rows"
        elif max_abs <= float(self.epsilon):
            status = "pass"
            note = "all exact/oracle values are within tolerance"
        else:
            status = "fail"
            note = "exact/oracle ceiling exceeds tolerance"
        return ExpectationFinding(
            kind="ceiling",
            title=self.title,
            status=status,
            family=self.family,
            scenario=self.scenario,
            metric=self.metric,
            method=self.method,
            direction="flat",
            observed_summary={
                "n_rows": int(len(vals)),
                "max_abs_value": float(max_abs),
                "note": note,
            },
            thresholds={"epsilon": float(self.epsilon)},
            supporting_rows=_supporting_rows(rows),
        )


@dataclass(frozen=True)
class BudgetTrendExpectation:
    family: str
    scenario: str
    method: str
    metric: str
    axis_name: str
    direction: TrendDirection
    title: str
    warn_only: bool = False

    def evaluate(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> ExpectationFinding:
        trend = assess_trend(rows, axis_name=self.axis_name, direction=self.direction, config=config)
        status = trend.status
        if self.warn_only and status == "fail":
            status = "warn"
        return ExpectationFinding(
            kind="budget_trend",
            title=self.title,
            status=status,
            family=self.family,
            scenario=self.scenario,
            metric=self.metric,
            method=self.method,
            direction=self.direction,
            observed_summary=trend.to_dict(),
            thresholds={
                "min_effect_rel": float(config.min_effect_rel),
                "min_effect_abs_scale": float(config.min_effect_abs_scale),
                "adjacent_tolerance": float(config.adjacent_tolerance),
                "adjacent_success_rate_min": float(config.adjacent_success_rate_min),
            },
            supporting_rows=_supporting_rows(rows),
        )


@dataclass(frozen=True)
class FailureModeExpectation:
    family: str
    scenario: str
    metric: str
    intended_method: str
    mismatch_method: str
    title: str
    separation_rel: float
    warn_only: bool = False

    def evaluate(
        self,
        intended_rows: Sequence[NormalizedRow],
        mismatch_rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> ExpectationFinding:
        scale = _metric_scale(
            [float(r.metric_value) for r in [*intended_rows, *mismatch_rows] if math.isfinite(float(r.metric_value))]
        )
        intended = _aggregate_metric(intended_rows, config=config)
        mismatch = _aggregate_metric(mismatch_rows, config=config)
        required_gap = max(scale * float(config.min_effect_abs_scale), abs(mismatch) * float(self.separation_rel))
        gap = mismatch - intended
        if not intended_rows or not mismatch_rows:
            status: ExpectationStatus = "not_applicable"
            note = "missing intended or mismatch rows"
        elif not math.isfinite(intended) or not math.isfinite(mismatch):
            status = "warn"
            note = "anchor aggregate is not finite"
        elif gap >= required_gap:
            status = "pass"
            note = "mismatch baseline remains materially worse than the intended method"
        elif gap >= 0.0:
            status = "warn" if self.warn_only else "fail"
            note = "mismatch baseline is only slightly worse than the intended method"
        else:
            status = "warn" if self.warn_only else "fail"
            note = "mismatch baseline is unexpectedly better than the intended method"
        return ExpectationFinding(
            kind="failure_mode",
            title=self.title,
            status=status,
            family=self.family,
            scenario=self.scenario,
            metric=self.metric,
            method=f"{self.intended_method} vs {self.mismatch_method}",
            direction="decreasing",
            observed_summary={
                "intended_value": float(intended),
                "mismatch_value": float(mismatch),
                "gap": float(gap),
                "note": note,
            },
            thresholds={"required_gap": float(required_gap), "separation_rel": float(self.separation_rel)},
            supporting_rows=_supporting_rows([*intended_rows, *mismatch_rows]),
        )


@dataclass(frozen=True)
class GranularitySensitivityExpectation:
    family: str
    scenario: str
    method: str
    metric: str
    support_axis: str
    title: str

    def evaluate(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> ExpectationFinding:
        leaf_values = sorted(
            {float(v) for v in (r.leaf_tokens for r in rows) if v is not None and math.isfinite(float(v))}
        )
        support_values = sorted(
            {
                float(v)
                for v in (r.value_for(self.support_axis) for r in rows)
                if v is not None and math.isfinite(float(v))
            }
        )
        if len(leaf_values) < 2 or len(support_values) < 2:
            return ExpectationFinding(
                kind="granularity",
                title=self.title,
                status="not_applicable",
                family=self.family,
                scenario=self.scenario,
                metric=self.metric,
                method=self.method,
                direction="decreasing",
                observed_summary={"note": "need at least two leaf granularities and two support levels"},
                thresholds={},
                supporting_rows=_supporting_rows(rows),
            )

        low_support = float(support_values[0])
        high_support = float(support_values[-1])
        low_rows = [
            r
            for r in rows
            if r.value_for(self.support_axis) is not None
            and abs(float(r.value_for(self.support_axis)) - low_support) <= 1e-12
        ]
        high_rows = [
            r
            for r in rows
            if r.value_for(self.support_axis) is not None
            and abs(float(r.value_for(self.support_axis)) - high_support) <= 1e-12
        ]
        low_by_leaf: Dict[float, List[float]] = {}
        high_by_leaf: Dict[float, List[float]] = {}
        for row in low_rows:
            if row.leaf_tokens is None:
                continue
            low_by_leaf.setdefault(float(row.leaf_tokens), []).append(float(row.metric_value))
        for row in high_rows:
            if row.leaf_tokens is None:
                continue
            high_by_leaf.setdefault(float(row.leaf_tokens), []).append(float(row.metric_value))
        if len(low_by_leaf) < 2 or len(high_by_leaf) < 2:
            return ExpectationFinding(
                kind="granularity",
                title=self.title,
                status="not_applicable",
                family=self.family,
                scenario=self.scenario,
                metric=self.metric,
                method=self.method,
                direction="decreasing",
                observed_summary={"note": "support slices do not contain enough leaf variations"},
                thresholds={},
                supporting_rows=_supporting_rows(rows),
            )

        low_aggs = {k: _safe_median(v) for k, v in low_by_leaf.items()}
        high_aggs = {k: _safe_median(v) for k, v in high_by_leaf.items()}
        low_range = float(max(low_aggs.values()) - min(low_aggs.values()))
        envelope_low = float(min(low_aggs.values()))
        envelope_high = float(min(high_aggs.values()))
        scale = _metric_scale([*low_aggs.values(), *high_aggs.values()])
        min_effect_abs = float(config.min_effect_abs_scale) * float(scale)
        variation_ok = low_range >= min_effect_abs
        envelope_ok = envelope_high <= (envelope_low - min_effect_abs)
        if variation_ok and envelope_ok:
            status: ExpectationStatus = "pass"
            note = "granularity changes the low-support regime, and the best envelope improves with support"
        elif variation_ok or envelope_ok:
            status = "warn"
            note = "only one of the expected granularity signals is clearly present"
        else:
            status = "fail"
            note = "granularity effects are missing or the best envelope does not improve"
        return ExpectationFinding(
            kind="granularity",
            title=self.title,
            status=status,
            family=self.family,
            scenario=self.scenario,
            metric=self.metric,
            method=self.method,
            direction="decreasing",
            observed_summary={
                "low_support": float(low_support),
                "high_support": float(high_support),
                "low_support_range": float(low_range),
                "best_low_support_value": float(envelope_low),
                "best_high_support_value": float(envelope_high),
                "note": note,
            },
            thresholds={"min_effect_abs": float(min_effect_abs)},
            supporting_rows=_supporting_rows([*low_rows, *high_rows]),
        )


class MarkovOPSAdapter:
    family = FAMILY_MARKOV

    def can_load(self, path: Path) -> bool:
        try:
            payload = _load_json(path)
        except Exception:
            return False
        simulation = str(payload.get("simulation", "")).strip()
        if simulation in {
            FULL_DOC_MARKOV_DIAGNOSTIC_SIMULATION,
            FULL_DOC_MARKOV_LADDER_SIMULATION,
            FULL_TREE_IPW_MARKOV_SIMULATION,
        }:
            return True
        parts = _path_parts_lower(path)
        if "markov" not in parts and not _path_contains_any(path, ("markov", "changepoint")):
            return False
        metrics = payload.get("metrics", {}) or {}
        return (
            isinstance(metrics, dict)
            and {"exact", "undersupported", "learned"} <= set(metrics.keys())
            and "training_geometry" in payload
            and "estimator_diagnostics" in payload
        )

    def load_rows(self, path: Path) -> List[NormalizedRow]:
        payload = _load_json(path)
        simulation = str(payload.get("simulation", "")).strip()
        if simulation == FULL_DOC_MARKOV_DIAGNOSTIC_SIMULATION:
            return _full_doc_markov_run_rows(path, payload)
        if simulation == FULL_DOC_MARKOV_LADDER_SIMULATION:
            return _full_doc_markov_ladder_rows(path, payload)
        if simulation == FULL_TREE_IPW_MARKOV_SIMULATION:
            return _full_tree_ipw_grid_rows(path, payload)
        cfg = payload.get("config", {}) or {}
        geom = payload.get("training_geometry", {}) or {}
        objective_meta = _objective_metadata(payload)
        markov_objective_meta = _markov_objective_metadata(payload)
        train_docs = _finite(cfg.get("train_docs"))
        audit_fraction = _finite(cfg.get("audit_fraction"))
        leaf_tokens = _finite(cfg.get("fixed_leaf_tokens"))
        transition_log_std = _finite(cfg.get("transition_log_std"))
        min_segments = _finite(cfg.get("min_segments"))
        max_segments = _finite(cfg.get("max_segments"))
        min_seg_len = _finite(cfg.get("min_seg_len"))
        max_seg_len = _finite(cfg.get("max_seg_len"))
        mean_tokens = _finite(geom.get("mean_tokens")) or _finite(cfg.get("max_tokens"))
        mean_leaves = _finite(geom.get("mean_leaves"))
        leaves_per_doc = mean_leaves
        if leaves_per_doc is None and mean_tokens is not None and leaf_tokens not in {None, 0.0}:
            leaves_per_doc = float(mean_tokens) / max(1.0, float(leaf_tokens))
        test_identity = _test_identity_from_payload(payload)
        theorem_relevant = bool(markov_objective_meta.get("theorem_relevant", False))
        local_law_weight = markov_objective_meta.get("objective_local_law_weight")
        weighting_scheme = markov_objective_meta.get("objective_weighting_scheme", "unknown")
        parameterization = markov_objective_meta.get("objective_parameterization", "unknown")
        scenario = _slug(
            {
                "model_family": cfg.get("model_family", "unknown"),
                "feature_mode": cfg.get("feature_mode", "unknown"),
                "fixed_leaf_tokens": leaf_tokens if leaf_tokens is not None else "na",
                "leaf_query_rate": cfg.get("leaf_query_rate", 1.0),
                "include_root_query": cfg.get("include_root_query", True),
                "test_docs": cfg.get("test_docs", "na"),
                "transition_log_std": transition_log_std if transition_log_std is not None else "na",
                "min_segments": min_segments if min_segments is not None else "na",
                "max_segments": max_segments if max_segments is not None else "na",
                "min_seg_len": min_seg_len if min_seg_len is not None else "na",
                "max_seg_len": max_seg_len if max_seg_len is not None else "na",
                "objective_weighting_scheme": weighting_scheme,
                "objective_parameterization": parameterization,
                "objective_local_law_weight": local_law_weight if local_law_weight is not None else "na",
                "theorem_relevant": theorem_relevant,
            }
        )
        granularity_group = _slug(
            {
                "model_family": cfg.get("model_family", "unknown"),
                "feature_mode": cfg.get("feature_mode", "unknown"),
                "leaf_query_rate": cfg.get("leaf_query_rate", 1.0),
                "include_root_query": cfg.get("include_root_query", True),
                "test_docs": cfg.get("test_docs", "na"),
                "transition_log_std": transition_log_std if transition_log_std is not None else "na",
                "min_segments": min_segments if min_segments is not None else "na",
                "max_segments": max_segments if max_segments is not None else "na",
                "min_seg_len": min_seg_len if min_seg_len is not None else "na",
                "max_seg_len": max_seg_len if max_seg_len is not None else "na",
                "objective_weighting_scheme": weighting_scheme,
                "objective_parameterization": parameterization,
                "objective_local_law_weight": local_law_weight if local_law_weight is not None else "na",
                "theorem_relevant": theorem_relevant,
            }
        )
        rows: List[NormalizedRow] = []
        metrics = payload.get("metrics", {}) or {}
        for method in ("exact", "undersupported", "learned"):
            block = metrics.get(method, {}) or {}
            for metric_name in ("root_mae", "merge_mae", "schedule_spread_mean"):
                metric_val = _finite(block.get(metric_name))
                if metric_val is None:
                    continue
                rows.append(
                    NormalizedRow(
                        family=self.family,
                        scenario=scenario,
                        seed=int(cfg["seed"]) if cfg.get("seed") is not None else None,
                        method=str(method),
                        x_axis_name="train_docs",
                        x_axis_value=float(train_docs if train_docs is not None else float("nan")),
                        secondary_axis_name="oracle_budget_fraction",
                        secondary_axis_value=audit_fraction,
                        metric_name=str(metric_name),
                        metric_value=float(metric_val),
                        doc_scale_tokens=mean_tokens,
                        leaf_tokens=leaf_tokens,
                        leaves_per_doc=leaves_per_doc,
                        oracle_budget_fraction=audit_fraction,
                        train_docs=train_docs,
                        evidence_status=_evidence_status_for_method(self.family, method),
                        source_path=str(path.resolve()),
                        test_identity=test_identity,
                        metadata={
                            "granularity_group": granularity_group,
                            "model_family": cfg.get("model_family"),
                            "transition_log_std": transition_log_std,
                            "min_segments": min_segments,
                            "max_segments": max_segments,
                            "min_seg_len": min_seg_len,
                            "max_seg_len": max_seg_len,
                            **objective_meta,
                            **markov_objective_meta,
                        },
                    )
                )
        return rows

    def build_expectations(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        findings: List[ExpectationFinding] = []
        diagnostics_rows = [
            row
            for row in rows
            if str(row.metadata.get("surface", "")) == FULL_DOC_MARKOV_DIAGNOSTIC_SIMULATION
        ]
        ladder_rows = [
            row
            for row in rows
            if str(row.metadata.get("surface", "")) == FULL_DOC_MARKOV_LADDER_SIMULATION
        ]
        full_tree_rows = [
            row
            for row in rows
            if str(row.metadata.get("surface", "")) == FULL_TREE_IPW_MARKOV_SIMULATION
        ]
        legacy_rows = [
            row
            for row in rows
            if str(row.metadata.get("surface", ""))
            not in {
                FULL_DOC_MARKOV_DIAGNOSTIC_SIMULATION,
                FULL_DOC_MARKOV_LADDER_SIMULATION,
                FULL_TREE_IPW_MARKOV_SIMULATION,
            }
        ]
        if diagnostics_rows:
            findings.extend(
                self._build_full_doc_anchor_diagnostic_expectations(
                    diagnostics_rows, config=config
                )
            )
        if ladder_rows:
            findings.extend(
                self._build_full_doc_anchor_ladder_expectations(
                    ladder_rows, config=config
                )
            )
        if full_tree_rows:
            findings.extend(
                self._build_full_tree_ipw_grid_expectations(
                    full_tree_rows, config=config
                )
            )
        if not legacy_rows:
            return findings
        by_scenario: Dict[str, List[NormalizedRow]] = {}
        for row in legacy_rows:
            by_scenario.setdefault(row.scenario, []).append(row)

        for scenario, srows in sorted(by_scenario.items()):
            theorem_relevant = any(bool(r.metadata.get("theorem_relevant")) for r in srows)
            for metric, epsilon in (
                ("root_mae", config.ceiling_epsilon_exact),
                ("merge_mae", config.ceiling_epsilon_exact),
                ("schedule_spread_mean", config.ceiling_epsilon_exact),
            ):
                exact_rows = _row_subset(srows, method="exact", metric_name=metric)
                findings.append(
                    CeilingExpectation(
                        family=self.family,
                        scenario=scenario,
                        method="exact",
                        metric=metric,
                        epsilon=float(epsilon),
                        title=f"Markov exact ceiling: {metric}",
                    ).evaluate(exact_rows)
                )
            for metric in ("root_mae", "merge_mae"):
                learned_rows = _row_subset(srows, method="learned", metric_name=metric)
                if learned_rows:
                    max_budget = max(
                        float(r.oracle_budget_fraction)
                        for r in learned_rows
                        if r.oracle_budget_fraction is not None
                    )
                    train_slice = [
                        r
                        for r in learned_rows
                        if r.oracle_budget_fraction is not None
                        and abs(float(r.oracle_budget_fraction) - float(max_budget)) <= 1e-12
                    ]
                    findings.append(
                        BudgetTrendExpectation(
                            family=self.family,
                            scenario=scenario,
                            method="learned",
                            metric=metric,
                            axis_name="train_docs",
                            direction="decreasing",
                            title=f"Markov learned {metric} improves with train_docs at max audit_fraction",
                            warn_only=True,
                        ).evaluate(train_slice, config=config)
                    )
                    max_train = max(float(r.train_docs) for r in learned_rows if r.train_docs is not None)
                    budget_slice = [
                        r
                        for r in learned_rows
                        if r.train_docs is not None and abs(float(r.train_docs) - float(max_train)) <= 1e-12
                    ]
                    findings.append(
                        BudgetTrendExpectation(
                            family=self.family,
                            scenario=scenario,
                            method="learned",
                            metric=metric,
                            axis_name="oracle_budget_fraction",
                            direction="decreasing",
                            title=f"Markov learned {metric} improves with audit_fraction at max train_docs",
                            warn_only=True,
                        ).evaluate(budget_slice, config=config)
                    )
                undersupported_rows = _row_subset(srows, method="undersupported", metric_name=metric)
                if undersupported_rows:
                    max_budget = max(
                        float(r.oracle_budget_fraction)
                        for r in undersupported_rows
                        if r.oracle_budget_fraction is not None
                    )
                    train_slice = [
                        r
                        for r in undersupported_rows
                        if r.oracle_budget_fraction is not None
                        and abs(float(r.oracle_budget_fraction) - float(max_budget)) <= 1e-12
                    ]
                    findings.append(
                        BudgetTrendExpectation(
                            family=self.family,
                            scenario=scenario,
                            method="undersupported",
                            metric=metric,
                            axis_name="train_docs",
                            direction="flat",
                            title=f"Markov undersupported {metric} remains flat with train_docs",
                        ).evaluate(train_slice, config=config)
                    )
                    max_train = max(float(r.train_docs) for r in undersupported_rows if r.train_docs is not None)
                    budget_slice = [
                        r
                        for r in undersupported_rows
                        if r.train_docs is not None and abs(float(r.train_docs) - float(max_train)) <= 1e-12
                    ]
                    findings.append(
                        BudgetTrendExpectation(
                            family=self.family,
                            scenario=scenario,
                            method="undersupported",
                            metric=metric,
                            axis_name="oracle_budget_fraction",
                            direction="flat",
                            title=f"Markov undersupported {metric} remains flat with audit_fraction",
                        ).evaluate(budget_slice, config=config)
                    )

            if theorem_relevant:
                learned_anchor = _anchor_slice(_row_subset(srows, method="learned", metric_name="root_mae"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
                unders_anchor = _anchor_slice(_row_subset(srows, method="undersupported", metric_name="root_mae"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
                findings.append(
                    FailureModeExpectation(
                        family=self.family,
                        scenario=scenario,
                        metric="root_mae",
                        intended_method="learned",
                        mismatch_method="undersupported",
                        title="Markov high-support anchor: learned root_mae beats undersupported",
                        separation_rel=config.separation_rel,
                    ).evaluate(learned_anchor, unders_anchor, config=config)
                )
                learned_anchor_merge = _anchor_slice(_row_subset(srows, method="learned", metric_name="merge_mae"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
                unders_anchor_merge = _anchor_slice(_row_subset(srows, method="undersupported", metric_name="merge_mae"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
                findings.append(
                    FailureModeExpectation(
                        family=self.family,
                        scenario=scenario,
                        metric="merge_mae",
                        intended_method="learned",
                        mismatch_method="undersupported",
                        title="Markov high-support anchor: learned merge_mae vs undersupported",
                        separation_rel=config.separation_rel,
                        warn_only=True,
                    ).evaluate(learned_anchor_merge, unders_anchor_merge, config=config)
                )

        by_granularity: Dict[str, List[NormalizedRow]] = {}
        for row in legacy_rows:
            if row.method == "learned" and row.metric_name == "root_mae":
                key = str(row.metadata.get("granularity_group", row.scenario))
                by_granularity.setdefault(key, []).append(row)
        for key, grows in sorted(by_granularity.items()):
            findings.append(
                GranularitySensitivityExpectation(
                    family=self.family,
                    scenario=key,
                    method="learned",
                    metric="root_mae",
                    support_axis="oracle_budget_fraction",
                    title="Markov granularity sensitivity: low-budget variation and best-envelope improvement",
                ).evaluate(grows, config=config)
            )
        return findings

    def _build_full_tree_ipw_grid_expectations(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        del config
        findings: List[ExpectationFinding] = []
        scenario = FULL_TREE_IPW_MARKOV_SIMULATION

        semantics_bad = [
            row
            for row in rows
            if str(row.metadata.get("estimand_name", "")) != "realized_full_tree_node_mean_loss"
            or str(row.metadata.get("population_kind", "")) != "realized_tree_nodes"
            or str(row.metadata.get("sampling_design", "")) != "bernoulli_realized_node_sampling"
            or str(row.metadata.get("propensity_field", "")) != "unit_propensity"
            or str(row.metadata.get("document_channel", "")) != "always_observed_document_top_loss"
            or str(row.metadata.get("node_channel", "")) != "sampled_realized_tree_nodes"
            or list(row.metadata.get("estimator_families") or []) != ["naive", "ht", "hajek"]
        ]
        findings.append(
            _finding(
                kind="full_tree_ipw_semantics",
                title="Full-tree IPW grid exposes the explicit realized-node estimand semantics",
                status="fail" if semantics_bad else "pass",
                scenario=scenario,
                metric="estimand_semantics",
                method="all_cells",
                observed_summary={
                    "n_rows": int(len(rows)),
                    "n_bad_rows": int(len(semantics_bad)),
                    "ci_semantics": sorted(
                        {
                            str(row.metadata.get("ci_semantics", ""))
                            for row in rows
                        }
                    ),
                },
                thresholds={
                    "estimand_name": "realized_full_tree_node_mean_loss",
                    "population_kind": "realized_tree_nodes",
                    "sampling_design": "bernoulli_realized_node_sampling",
                },
                supporting_rows=rows,
            )
        )

        full_tree_rows = [
            row
            for row in rows
            if abs(float(row.metadata.get("p_internal", float("nan"))) - 1.0) <= 1e-12
            and abs(float(row.metadata.get("p_leaf", float("nan"))) - 1.0) <= 1e-12
        ]
        endpoint_bad = [
            row
            for row in full_tree_rows
            if not math.isfinite(float(row.metadata.get("test_sampled_node_ht_abs_error", float("nan"))))
            or not math.isfinite(float(row.metadata.get("test_sampled_node_hajek_abs_error", float("nan"))))
            or float(row.metadata.get("test_sampled_node_ht_abs_error", float("inf"))) > 1e-6
            or float(row.metadata.get("test_sampled_node_hajek_abs_error", float("inf"))) > 1e-6
        ]
        findings.append(
            _finding(
                kind="full_tree_ipw_endpoint",
                title="Full-tree IPW endpoint (1,1) recovers the realized full-tree node estimand",
                status=(
                    "fail"
                    if full_tree_rows and endpoint_bad
                    else ("pass" if full_tree_rows else "not_applicable")
                ),
                scenario=scenario,
                metric="ht_hajek_endpoint_recovery",
                method="full_tree_endpoint",
                observed_summary={
                    "n_rows": int(len(full_tree_rows)),
                    "n_bad_rows": int(len(endpoint_bad)),
                    "max_ht_abs_error": max(
                        [
                            float(row.metadata.get("test_sampled_node_ht_abs_error", float("nan")))
                            for row in full_tree_rows
                        ]
                        or [float("nan")]
                    ),
                    "max_hajek_abs_error": max(
                        [
                            float(row.metadata.get("test_sampled_node_hajek_abs_error", float("nan")))
                            for row in full_tree_rows
                        ]
                        or [float("nan")]
                    ),
                },
                thresholds={"abs_error_tolerance": 1e-6},
                supporting_rows=full_tree_rows or rows,
            )
        )

        doc_only_rows = [
            row
            for row in rows
            if abs(float(row.metadata.get("p_internal", float("nan"))) - 0.0) <= 1e-12
            and abs(float(row.metadata.get("p_leaf", float("nan"))) - 0.0) <= 1e-12
        ]
        doc_only_bad = [
            row
            for row in doc_only_rows
            if abs(float(row.metadata.get("test_sample_fraction", float("nan")))) > 1e-12
            or int(row.metadata.get("test_sampled_nodes", 0)) != 0
        ]
        findings.append(
            _finding(
                kind="full_tree_ipw_endpoint",
                title="Full-tree IPW endpoint (0,0) remains a document-only reference",
                status=(
                    "fail"
                    if doc_only_rows and doc_only_bad
                    else ("pass" if doc_only_rows else "not_applicable")
                ),
                scenario=scenario,
                metric="document_only_reference",
                method="doc_only_endpoint",
                observed_summary={
                    "n_rows": int(len(doc_only_rows)),
                    "n_bad_rows": int(len(doc_only_bad)),
                },
                thresholds={"sample_fraction": 0.0, "sampled_nodes": 0},
                supporting_rows=doc_only_rows or rows,
            )
        )

        sampled_rows = [
            row
            for row in rows
            if float(row.metadata.get("test_sample_fraction", float("nan"))) > 0.0
        ]
        sampled_bad = [
            row
            for row in sampled_rows
            if not math.isfinite(float(row.metadata.get("test_effective_sample_size", float("nan"))))
            or not math.isfinite(float(row.metadata.get("test_max_weight", float("nan"))))
            or float(row.metadata.get("test_effective_sample_size", float("nan"))) <= 0.0
            or float(row.metadata.get("test_effective_sample_size", float("nan")))
            > float(row.metadata.get("test_sampled_nodes", float("inf"))) + 1e-9
            or float(row.metadata.get("test_max_weight", float("nan"))) <= 0.0
        ]
        findings.append(
            _finding(
                kind="full_tree_ipw_sampling",
                title="Full-tree IPW sampled cells expose finite, internally consistent ESS and max-weight diagnostics",
                status=(
                    "fail"
                    if sampled_rows and sampled_bad
                    else ("pass" if sampled_rows else "not_applicable")
                ),
                scenario=scenario,
                metric="sampling_diagnostics",
                method="sampled_cells",
                observed_summary={
                    "n_rows": int(len(sampled_rows)),
                    "n_bad_rows": int(len(sampled_bad)),
                },
                thresholds={"effective_sample_size>0": True, "max_weight>0": True},
                supporting_rows=sampled_rows or rows,
            )
        )

        plane_pairs = {
            (
                float(row.metadata.get("doc_sequence_train_fraction", float("nan"))),
                float(row.metadata.get("root_only_train_fraction", float("nan"))),
            )
            for row in rows
        }
        findings.append(
            _finding(
                kind="full_tree_ipw_planes",
                title="Full-tree IPW root-only and doc-sequence planes remain explicitly separated",
                status="pass" if plane_pairs else "not_applicable",
                scenario=scenario,
                metric="plane_separation",
                method="all_cells",
                observed_summary={
                    "plane_pairs": [
                        {
                            "doc_sequence_train_fraction": float(pair[0]),
                            "root_only_train_fraction": float(pair[1]),
                        }
                        for pair in sorted(plane_pairs)
                    ],
                    "n_plane_pairs": int(len(plane_pairs)),
                },
                thresholds={"plane_metadata_present": True},
                supporting_rows=rows,
            )
        )
        return findings

    def _build_full_doc_anchor_diagnostic_expectations(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        del config
        findings: List[ExpectationFinding] = []
        scenario = FULL_DOC_MARKOV_DIAGNOSTIC_SIMULATION

        grouped_signatures: Dict[tuple[str, str, float], set[tuple[str, str, str]]] = {}
        seed_counts: Dict[tuple[str, str, float], set[int | None]] = {}
        for row in rows:
            group_key = (
                str(row.metadata.get("benchmark", "")),
                str(row.metadata.get("cell_id", "")),
                float(row.train_docs if row.train_docs is not None else float("nan")),
            )
            grouped_signatures.setdefault(group_key, set()).add(
                (
                    str(row.metadata.get("bundle_source", "")),
                    str(row.metadata.get("val_corpus_signature", "")),
                    str(row.metadata.get("test_corpus_signature", "")),
                )
            )
            seed_counts.setdefault(group_key, set()).add(row.seed)
        inconsistent_groups = {
            key: len(value)
            for key, value in grouped_signatures.items()
            if len(value) > 1
        }
        max_seed_count = max((len(v) for v in seed_counts.values()), default=0)
        if inconsistent_groups:
            reproducibility_status: ExpectationStatus = "fail"
            reproducibility_note = "val/test bundle identity drifted across seeds"
        elif max_seed_count < 2:
            reproducibility_status = "warn"
            reproducibility_note = "only one seed is currently present, so cross-seed reproducibility is not fully exercised"
        else:
            reproducibility_status = "pass"
            reproducibility_note = "all cells reuse the same fixed val/test bundle across seeds"
        findings.append(
            _finding(
                kind="full_doc_reproducibility",
                title="Full-doc diagnostics fixed-bundle reproducibility stays locked across seeds",
                status=reproducibility_status,
                scenario=scenario,
                metric="bundle_identity",
                method="all_families",
                observed_summary={
                    "n_groups": int(len(grouped_signatures)),
                    "max_seed_count": int(max_seed_count),
                    "inconsistent_groups": {
                        "|".join(
                            [str(item[0]), str(item[1]), _stringify(item[2])]
                        ): int(count)
                        for item, count in inconsistent_groups.items()
                    },
                    "note": reproducibility_note,
                },
                thresholds={"required_unique_bundle_identities_per_group": 1},
                supporting_rows=rows,
            )
        )

        official_rows = [
            row
            for row in rows
            if row.method in {"official_fno", "official_fno_sumlen"}
        ]
        if official_rows:
            official_bad = [
                row
                for row in official_rows
                if str(row.metadata.get("backend_package", "")) != "neuraloperator"
                or str(row.metadata.get("operator_class", "")) != "neuralop.models.FNO"
                or not str(row.metadata.get("backend_version", "")).strip()
                or str(row.metadata.get("operator_evidence_status", "")) != "PROXY_ONLY"
            ]
            findings.append(
                _finding(
                    kind="full_doc_provenance",
                    title="Full-doc official FNO provenance matches the installed official backend",
                    status="fail" if official_bad else "pass",
                    scenario=scenario,
                    metric="backend_provenance",
                    method="official_fno",
                    observed_summary={
                        "n_rows": int(len(official_rows)),
                        "n_bad_rows": int(len(official_bad)),
                        "backend_packages": sorted(
                            {str(row.metadata.get("backend_package", "")) for row in official_rows}
                        ),
                        "backend_versions": sorted(
                            {str(row.metadata.get("backend_version", "")) for row in official_rows}
                        ),
                    },
                    thresholds={"backend_package": "neuraloperator", "operator_class": "neuralop.models.FNO"},
                    supporting_rows=official_rows,
                )
            )
        else:
            findings.append(
                _finding(
                    kind="full_doc_provenance",
                    title="Full-doc official FNO provenance matches the installed official backend",
                    status="not_applicable",
                    scenario=scenario,
                    metric="backend_provenance",
                    method="official_fno",
                    observed_summary={"note": "no official_fno rows were present"},
                    thresholds={"backend_package": "neuraloperator", "operator_class": "neuralop.models.FNO"},
                    supporting_rows=rows,
                )
            )

        tree_rows = [
            row
            for row in rows
            if str(row.method).startswith("tree_neural")
        ]
        current_tree_rows = [
            row
            for row in tree_rows
            if str(row.metadata.get("comparison_semantics", "")) == "current"
        ]
        if current_tree_rows:
            current_bad = [
                row
                for row in current_tree_rows
                if str(row.metadata.get("c2_metric_kind", "")) != "score_drift"
                or not str(row.metadata.get("semantics_version", "")).strip()
                or not str(row.metadata.get("parameterization", "")).strip()
                or str(row.metadata.get("operator_evidence_status", "")) != "APPROX_AUDITED"
                or not bool(row.metadata.get("objective_weights_active", False))
            ]
            findings.append(
                _finding(
                    kind="full_doc_tree_semantics",
                    title="Full-doc current tree-neural rows expose theorem-facing score-drift semantics",
                    status="fail" if current_bad else "pass",
                    scenario=scenario,
                    metric="tree_neural_semantics",
                    method="tree_neural",
                    observed_summary={
                        "n_current_rows": int(len(current_tree_rows)),
                        "n_bad_rows": int(len(current_bad)),
                        "c2_metric_kinds": sorted(
                            {str(row.metadata.get("c2_metric_kind", "")) for row in current_tree_rows}
                        ),
                        "semantics_versions": sorted(
                            {str(row.metadata.get("semantics_version", "")) for row in current_tree_rows}
                        ),
                    },
                    thresholds={"c2_metric_kind": "score_drift", "operator_evidence_status": "APPROX_AUDITED"},
                    supporting_rows=current_tree_rows,
                )
            )
        else:
            findings.append(
                _finding(
                    kind="full_doc_tree_semantics",
                    title="Full-doc current tree-neural rows expose theorem-facing score-drift semantics",
                    status="not_applicable",
                    scenario=scenario,
                    metric="tree_neural_semantics",
                    method="tree_neural",
                    observed_summary={"note": "no current tree-neural rows were present"},
                    thresholds={"c2_metric_kind": "score_drift"},
                    supporting_rows=rows,
                )
            )

        if tree_rows:
            labels_by_mode: Dict[str, set[str]] = {}
            separation_bad: List[NormalizedRow] = []
            for row in tree_rows:
                mode = str(row.metadata.get("comparison_semantics", ""))
                label = str(row.metadata.get("comparison_semantics_label", ""))
                labels_by_mode.setdefault(mode, set()).add(label)
                if mode == "legacy" and (
                    not bool(row.metadata.get("legacy_semantics", False))
                    or not label.strip()
                    or label == str(row.metadata.get("semantics_version", "")).strip()
                ):
                    separation_bad.append(row)
            shared_labels = sorted(
                labels_by_mode.get("legacy", set()) & labels_by_mode.get("current", set())
            )
            if shared_labels:
                separation_bad.extend(tree_rows)
            findings.append(
                _finding(
                    kind="full_doc_tree_semantics",
                    title="Full-doc tree-neural legacy rows stay explicitly separated from current semantics",
                    status="fail" if separation_bad else "pass",
                    scenario=scenario,
                    metric="legacy_current_separation",
                    method="tree_neural",
                    observed_summary={
                        "modes_present": sorted(labels_by_mode.keys()),
                        "labels_by_mode": {
                            key: sorted(values) for key, values in labels_by_mode.items()
                        },
                        "shared_labels": shared_labels,
                    },
                    thresholds={"legacy_labels_must_not_overlap_current": True},
                    supporting_rows=tree_rows,
                )
            )
        return findings

    def _build_full_doc_anchor_ladder_expectations(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        del config
        findings: List[ExpectationFinding] = []
        scenario = FULL_DOC_MARKOV_LADDER_SIMULATION

        provenance_bad = [
            row
            for row in rows
            if str(row.metadata.get("backend_package", "")) != "neuraloperator"
            or str(row.metadata.get("operator_class", "")) != "neuralop.models.FNO"
            or not str(row.metadata.get("backend_version", "")).strip()
        ]
        findings.append(
            _finding(
                kind="full_doc_ladder_provenance",
                title="Full-doc ladder doc-sequence stages record official neuraloperator provenance",
                status="fail" if provenance_bad else "pass",
                scenario=scenario,
                metric="backend_provenance",
                method="doc_sequence",
                observed_summary={
                    "n_rows": int(len(rows)),
                    "n_bad_rows": int(len(provenance_bad)),
                    "backend_versions": sorted(
                        {str(row.metadata.get("backend_version", "")) for row in rows}
                    ),
                },
                thresholds={"backend_package": "neuraloperator", "operator_class": "neuralop.models.FNO"},
                supporting_rows=rows,
            )
        )

        stage_by_name = {
            str(row.metadata.get("stage_name", "")): row for row in rows
        }
        paired_rows: List[NormalizedRow] = []
        mismatches: Dict[str, Dict[str, object]] = {}
        for stage_name, reproduction_row in stage_by_name.items():
            if not stage_name.endswith("_reproduction"):
                continue
            reference_name = stage_name[: -len("_reproduction")] + "_reference"
            reference_row = stage_by_name.get(reference_name)
            if reference_row is None:
                continue
            paired_rows.extend([reproduction_row, reference_row])
            compared_fields = (
                "observed_token_profile",
                "bundle_source",
                "train_docs",
                "val_docs",
                "test_docs",
                "state_dim",
                "hidden_dim",
                "n_epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "backend_package",
                "operator_class",
            )
            field_mismatches: Dict[str, object] = {}
            for field_name in compared_fields:
                left = reproduction_row.metadata.get(field_name)
                right = reference_row.metadata.get(field_name)
                if left != right:
                    field_mismatches[field_name] = {
                        "reproduction": left,
                        "reference": right,
                    }
            if field_mismatches:
                mismatches[stage_name] = field_mismatches
        if paired_rows:
            status: ExpectationStatus = "fail" if mismatches else "pass"
            note = (
                "reference/reproduction bundle and config semantics match"
                if not mismatches
                else "reference/reproduction semantics diverged"
            )
        else:
            status = "not_applicable"
            note = "no reference/reproduction pair was present"
        findings.append(
            _finding(
                kind="full_doc_ladder_pairing",
                title="Full-doc ladder reference and reproduction stages stay bundle/config matched",
                status=status,
                scenario=scenario,
                metric="reference_reproduction_match",
                method="doc_sequence",
                observed_summary={
                    "n_pairs": int(len(paired_rows) // 2),
                    "mismatches": mismatches,
                    "note": note,
                },
                thresholds={"compared_fields": ["bundle_source", "train_docs", "state_dim", "hidden_dim", "n_epochs", "batch_size", "lr", "weight_decay"]},
                supporting_rows=paired_rows or rows,
            )
        )
        return findings


class SegmentLDAOPSAdapter:
    family = FAMILY_SEGMENT_LDA

    def can_load(self, path: Path) -> bool:
        parts = _path_parts_lower(path)
        if (
            "segment" not in parts
            and "segment_lda_ops_weight_recovery" not in parts
            and "lda_tree_recovery_production" not in parts
            and not _path_contains_any(path, ("segment_lda", "lda_tree_recovery"))
        ):
            return False
        try:
            payload = _load_json(path)
        except Exception:
            return False
        metrics = payload.get("metrics", {}) or {}
        return (
            isinstance(metrics, dict)
            and {"exact", "undersupported", "ridge"} <= set(metrics.keys())
            and "training_geometry" in payload
            and "weight_truth" in payload
        )

    def load_rows(self, path: Path) -> List[NormalizedRow]:
        payload = _load_json(path)
        cfg = payload.get("config", {}) or {}
        geom = payload.get("training_geometry", {}) or {}
        objective_meta = _objective_metadata(payload)
        train_docs = _finite(cfg.get("train_docs"))
        audit_fraction = _finite(cfg.get("audit_fraction"))
        leaf_tokens = _finite(cfg.get("leaf_tokens"))
        mean_tokens = _finite(geom.get("mean_tokens")) or _estimate_doc_scale_from_segments(cfg)
        mean_leaves = _finite(geom.get("mean_leaves"))
        leaves_per_doc = mean_leaves
        if leaves_per_doc is None and mean_tokens is not None and leaf_tokens not in {None, 0.0}:
            leaves_per_doc = float(mean_tokens) / max(1.0, float(leaf_tokens))
        test_identity = _test_identity_from_payload(payload)
        topic_process = str(cfg.get("topic_process", "unknown"))
        lambda_multiplier = _finite(cfg.get("lambda_multiplier"))
        scenario = _slug(
            {
                "topic_process": topic_process,
                "lambda_multiplier": lambda_multiplier if lambda_multiplier is not None else "na",
                "leaf_tokens": leaf_tokens if leaf_tokens is not None else "na",
                "topic_phi_estimator": cfg.get("topic_phi_estimator", "unknown"),
            }
        )
        control_group = _slug(
            {
                "leaf_tokens": leaf_tokens if leaf_tokens is not None else "na",
                "topic_phi_estimator": cfg.get("topic_phi_estimator", "unknown"),
                "feature_inference": cfg.get("feature_inference", "na"),
            }
        )
        metric_names = (
            "root_mae",
            "merge_mae",
            "schedule_spread_mean",
            "theta_rmse",
            "theta_cosine",
            "beta_rmse",
            "beta_cosine",
            "bigram_rmse",
            "bigram_cosine",
            "lambda_hat",
            "lambda_abs_error",
            "W_direction_cosine",
            "leaf_accuracy_test",
        )
        rows: List[NormalizedRow] = []
        metrics = payload.get("metrics", {}) or {}
        for method, block in metrics.items():
            if not isinstance(block, dict):
                continue
            for metric_name in metric_names:
                metric_val = _finite(block.get(metric_name))
                if metric_val is None:
                    continue
                rows.append(
                    NormalizedRow(
                        family=self.family,
                        scenario=scenario,
                        seed=int(cfg["seed"]) if cfg.get("seed") is not None else None,
                        method=str(method),
                        x_axis_name="train_docs",
                        x_axis_value=float(train_docs if train_docs is not None else float("nan")),
                        secondary_axis_name="oracle_budget_fraction",
                        secondary_axis_value=audit_fraction,
                        metric_name=str(metric_name),
                        metric_value=float(metric_val),
                        doc_scale_tokens=mean_tokens,
                        leaf_tokens=leaf_tokens,
                        leaves_per_doc=leaves_per_doc,
                        oracle_budget_fraction=audit_fraction,
                        train_docs=train_docs,
                        evidence_status=_evidence_status_for_method(self.family, method),
                        source_path=str(path.resolve()),
                        test_identity=test_identity,
                        metadata={
                            "topic_process": topic_process,
                            "lambda_multiplier": lambda_multiplier,
                            "control_group": control_group,
                            # Boundary sensitivity is a property of the target functional here:
                            # the internal-node labels only become structurally necessary when the
                            # boundary/bigram term is active.
                            "boundary_sensitive": bool(
                                lambda_multiplier is not None and lambda_multiplier > 0.0
                            ),
                            **objective_meta,
                        },
                    )
                )
        return rows

    def build_expectations(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        findings: List[ExpectationFinding] = []
        by_scenario: Dict[str, List[NormalizedRow]] = {}
        for row in rows:
            by_scenario.setdefault(row.scenario, []).append(row)

        for scenario, srows in sorted(by_scenario.items()):
            for metric in ("root_mae", "merge_mae"):
                exact_rows = _row_subset(srows, method="exact", metric_name=metric)
                findings.append(
                    CeilingExpectation(
                        family=self.family,
                        scenario=scenario,
                        method="exact",
                        metric=metric,
                        epsilon=config.ceiling_epsilon_exact,
                        title=f"Segment-LDA exact ceiling: {metric}",
                    ).evaluate(exact_rows)
                )
            is_boundary_sensitive = any(bool(r.metadata.get("boundary_sensitive")) for r in srows)
            if is_boundary_sensitive:
                exact_anchor = _anchor_slice(_row_subset(srows, method="exact", metric_name="root_mae"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
                unders_anchor = _anchor_slice(_row_subset(srows, method="undersupported", metric_name="root_mae"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
                findings.append(
                    FailureModeExpectation(
                        family=self.family,
                        scenario=scenario,
                        metric="root_mae",
                        intended_method="exact",
                        mismatch_method="undersupported",
                        title="Segment-LDA boundary-sensitive regime: undersupported stays separated from exact",
                        separation_rel=config.separation_rel,
                    ).evaluate(exact_anchor, unders_anchor, config=config)
                )
            for metric in ("root_mae", "merge_mae"):
                ridge_rows = _row_subset(srows, method="ridge", metric_name=metric)
                if ridge_rows:
                    max_budget = max(float(r.oracle_budget_fraction) for r in ridge_rows if r.oracle_budget_fraction is not None)
                    train_slice = [
                        r
                        for r in ridge_rows
                        if r.oracle_budget_fraction is not None
                        and abs(float(r.oracle_budget_fraction) - float(max_budget)) <= 1e-12
                    ]
                    findings.append(
                        BudgetTrendExpectation(
                            family=self.family,
                            scenario=scenario,
                            method="ridge",
                            metric=metric,
                            axis_name="train_docs",
                            direction="decreasing",
                            title=f"Segment-LDA ridge {metric} improves with train_docs at max audit_fraction",
                        ).evaluate(train_slice, config=config)
                    )
                    max_train = max(float(r.train_docs) for r in ridge_rows if r.train_docs is not None)
                    budget_slice = [
                        r
                        for r in ridge_rows
                        if r.train_docs is not None and abs(float(r.train_docs) - float(max_train)) <= 1e-12
                    ]
                    findings.append(
                        BudgetTrendExpectation(
                            family=self.family,
                            scenario=scenario,
                            method="ridge",
                            metric=metric,
                            axis_name="oracle_budget_fraction",
                            direction="decreasing",
                            title=f"Segment-LDA ridge {metric} improves with audit_fraction at max train_docs",
                        ).evaluate(budget_slice, config=config)
                    )

                ridge_true_anchor = _anchor_slice(
                    _row_subset(srows, method="ridge_true_topics", metric_name="root_mae"),
                    primary_field="train_docs",
                    secondary_field="oracle_budget_fraction",
                )
                ridge_anchor = _anchor_slice(
                    _row_subset(srows, method="ridge", metric_name="root_mae"),
                    primary_field="train_docs",
                    secondary_field="oracle_budget_fraction",
                )
                if ridge_true_anchor and ridge_anchor:
                    ridge_true = _aggregate_metric(ridge_true_anchor, config=config)
                    ridge_base = _aggregate_metric(ridge_anchor, config=config)
                    tol = max(
                        config.min_effect_abs_scale * _metric_scale([ridge_true, ridge_base]),
                        config.calibrated_regression_rel * max(abs(ridge_base), 1e-12),
                    )
                    status: ExpectationStatus
                    if ridge_true <= ridge_base + tol:
                        status = "pass"
                        note = "true-topic upper bound is at least as good as ridge"
                    else:
                        status = "fail"
                        note = "true-topic upper bound regresses beyond tolerance"
                    findings.append(
                        ExpectationFinding(
                            kind="failure_mode",
                            title="Segment-LDA ridge_true_topics is at least as good as ridge at high support",
                            status=status,
                            family=self.family,
                            scenario=scenario,
                            metric="root_mae",
                            method="ridge_true_topics vs ridge",
                            direction="decreasing",
                            observed_summary={
                                "ridge_true_topics_root_mae": float(ridge_true),
                                "ridge_root_mae": float(ridge_base),
                                "note": note,
                            },
                            thresholds={"max_allowed_regression": float(tol)},
                            supporting_rows=_supporting_rows([*ridge_true_anchor, *ridge_anchor]),
                        )
                    )

            flip_rows = [r for r in srows if r.method in {"flip_R1", "flip_R2"} and r.metric_name in {"root_mae", "schedule_spread_mean"}]
            if flip_rows:
                flip_r1_root = _anchor_slice(
                    [r for r in flip_rows if r.method == "flip_R1" and r.metric_name == "root_mae"],
                    primary_field="train_docs",
                    secondary_field="oracle_budget_fraction",
                )
                flip_r2_root = _anchor_slice(
                    [r for r in flip_rows if r.method == "flip_R2" and r.metric_name == "root_mae"],
                    primary_field="train_docs",
                    secondary_field="oracle_budget_fraction",
                )
                flip_spread = _anchor_slice(
                    [r for r in flip_rows if r.metric_name == "schedule_spread_mean"],
                    primary_field="train_docs",
                    secondary_field="oracle_budget_fraction",
                )
                exact_root = _anchor_slice(_row_subset(srows, method="exact", metric_name="root_mae"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
                exact_spread = _anchor_slice(_row_subset(srows, method="exact", metric_name="schedule_spread_mean"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
                flip_r1_root_v = _aggregate_metric(flip_r1_root, config=config)
                flip_r2_root_v = _aggregate_metric(flip_r2_root, config=config)
                flip_spread_v = _aggregate_metric(flip_spread, config=config)
                exact_root_v = _aggregate_metric(exact_root, config=config)
                exact_spread_v = _aggregate_metric(exact_spread, config=config)
                scale = _metric_scale([flip_r1_root_v, flip_r2_root_v, flip_spread_v, exact_root_v, exact_spread_v])
                tol = config.min_effect_abs_scale * scale
                if exact_root and exact_spread:
                    one_pass_ok = bool(flip_r1_root) and flip_r1_root_v <= exact_root_v + tol
                    repeated_resummary_drift = bool(flip_r2_root) and flip_r2_root_v >= exact_root_v + tol
                    schedule_drift = bool(flip_spread) and flip_spread_v >= exact_spread_v + tol
                    if one_pass_ok and (repeated_resummary_drift or schedule_drift):
                        status = "pass"
                        if repeated_resummary_drift and schedule_drift:
                            note = "flip family preserves one-pass behavior and shows both repeated-resummary root drift and schedule/idempotence drift"
                        elif repeated_resummary_drift:
                            note = "flip family preserves one-pass behavior but fails under repeated resummary"
                        else:
                            note = "flip family preserves one-pass root behavior but shows elevated schedule/idempotence drift"
                    else:
                        status = "fail"
                        note = "flip family does not show the expected one-pass-vs-resummary pattern"
                else:
                    status = "not_applicable"
                    note = "missing flip or exact anchor rows"
                findings.append(
                    ExpectationFinding(
                        kind="failure_mode",
                        title="Segment-LDA flip family behaves like an L3-style counterexample",
                        status=status,
                        family=self.family,
                        scenario=scenario,
                        metric="schedule_spread_mean",
                        method="flip_R1/R2",
                        direction="increasing",
                        observed_summary={
                            "flip_r1_root_mae": float(flip_r1_root_v),
                            "flip_r2_root_mae": float(flip_r2_root_v),
                            "flip_schedule_spread_mean": float(flip_spread_v),
                            "exact_root_mae": float(exact_root_v),
                            "exact_schedule_spread_mean": float(exact_spread_v),
                            "note": note,
                        },
                        thresholds={"tolerance": float(tol)},
                        supporting_rows=_supporting_rows([*flip_r1_root, *flip_r2_root, *flip_spread, *exact_root, *exact_spread]),
                    )
                )
            else:
                findings.append(
                    ExpectationFinding(
                        kind="failure_mode",
                        title="Segment-LDA flip family behaves like an L3-style counterexample",
                        status="not_applicable",
                        family=self.family,
                        scenario=scenario,
                        metric="schedule_spread_mean",
                        method="flip_R1/R2",
                        direction="increasing",
                        observed_summary={"note": "flip outputs are not present in this scenario"},
                        thresholds={},
                        supporting_rows=[],
                    )
                )

        control_groups: Dict[str, List[NormalizedRow]] = {}
        for row in rows:
            if row.method == "ridge" and row.metric_name == "root_mae":
                control_groups.setdefault(str(row.metadata.get("control_group", row.scenario)), []).append(row)
        for group_key, grows in sorted(control_groups.items()):
            sensitive = [r for r in grows if bool(r.metadata.get("boundary_sensitive"))]
            control = [r for r in grows if not bool(r.metadata.get("boundary_sensitive"))]
            if not sensitive or not control:
                continue
            sens_anchor = _anchor_slice(sensitive, primary_field="train_docs")
            ctrl_anchor = _anchor_slice(control, primary_field="train_docs")
            sens_trend = assess_trend(sens_anchor, axis_name="oracle_budget_fraction", direction="decreasing", config=config)
            ctrl_trend = assess_trend(ctrl_anchor, axis_name="oracle_budget_fraction", direction="decreasing", config=config)
            sens_gain = -float(sens_trend.endpoint_delta) if math.isfinite(sens_trend.endpoint_delta) else float("nan")
            ctrl_gain = -float(ctrl_trend.endpoint_delta) if math.isfinite(ctrl_trend.endpoint_delta) else float("nan")
            scale = _metric_scale([sens_gain, ctrl_gain])
            min_gap = config.min_effect_abs_scale * scale
            if math.isfinite(sens_gain) and math.isfinite(ctrl_gain):
                if sens_gain >= ctrl_gain + min_gap:
                    status = "pass"
                    note = "internal-node labels matter more in the boundary-sensitive regime"
                elif sens_gain >= ctrl_gain:
                    status = "warn"
                    note = "boundary-sensitive regime improves more, but only slightly"
                else:
                    status = "fail"
                    note = "internal-node labels do not help more in the boundary-sensitive regime"
            else:
                status = "not_applicable"
                note = "missing paired sensitive/control curves"
            findings.append(
                ExpectationFinding(
                    kind="failure_mode",
                    title="Segment-LDA internal-node labels matter more when boundaries matter",
                    status=status,
                    family=self.family,
                    scenario=group_key,
                    metric="root_mae",
                    method="ridge",
                    direction="decreasing",
                    observed_summary={
                        "boundary_sensitive_gain": float(sens_gain),
                        "boundary_insensitive_gain": float(ctrl_gain),
                        "note": note,
                    },
                    thresholds={"min_gap": float(min_gap)},
                    supporting_rows=_supporting_rows([*sens_anchor, *ctrl_anchor]),
                )
            )
        return findings


class SegmentedLDACtreePOAdapter:
    family = FAMILY_CTREE

    def can_load(self, path: Path) -> bool:
        parts = _path_parts_lower(path)
        if (
            "ctree" not in parts
            and "segmented_lda_ctreepo" not in parts
            and "identifiable_zero_publication_ctreepo" not in parts
            and "identifiable_zero_dtm_lda" not in parts
            and not _path_contains_any(path, ("segmented_lda_ctreepo", "publication_ctreepo", "dtm_lda"))
        ):
            return False
        try:
            payload = _load_json(path)
        except Exception:
            return False
        metrics = payload.get("metrics", {}) or {}
        return (
            isinstance(metrics, dict)
            and {"oracle_tree", "estimated_uncalibrated", "estimated_calibrated", "estimated_calibrated_budgeted"}
            <= set(metrics.keys())
            and "decomposition" in payload
        )

    def load_rows(self, path: Path) -> List[NormalizedRow]:
        payload = _load_json(path)
        cfg = payload.get("config", {}) or {}
        topic_meta = payload.get("topic_meta", {}) or {}
        objective_meta = _objective_metadata(payload)
        train_docs = _finite(cfg.get("n_books_train"))
        leaf_tokens = _finite(cfg.get("fixed_leaf_tokens"))
        doc_scale_tokens = _estimate_doc_scale_from_segments(cfg)
        leaves_per_doc = None
        if doc_scale_tokens is not None and leaf_tokens not in {None, 0.0}:
            leaves_per_doc = float(doc_scale_tokens) / max(1.0, float(leaf_tokens))
        cal_rate = _finite(cfg.get("calibration_leaf_query_rate"))
        eval_leaf = _finite(cfg.get("eval_leaf_query_rate"))
        eval_internal = _finite(cfg.get("eval_internal_query_rate"))
        support_total = sum(x for x in (cal_rate, eval_leaf, eval_internal) if x is not None)
        test_identity = _test_identity_from_payload(payload)
        scenario = _slug(
            {
                "topic_process": cfg.get("topic_process", "unknown"),
                "leaf_theta_estimator": cfg.get("leaf_theta_estimator", "unknown"),
                "topic_phi_estimator": cfg.get("topic_phi_estimator", "unknown"),
                "fixed_leaf_tokens": leaf_tokens if leaf_tokens is not None else "na",
            }
        )
        granularity_group = _slug(
            {
                "topic_process": cfg.get("topic_process", "unknown"),
                "leaf_theta_estimator": cfg.get("leaf_theta_estimator", "unknown"),
                "topic_phi_estimator": cfg.get("topic_phi_estimator", "unknown"),
            }
        )
        rows: List[NormalizedRow] = []
        metric_names = ("root_l1_mean", "c1_violation_rate", "c3_violation_rate", "mean_leaf_queries", "mean_internal_queries")
        metrics = payload.get("metrics", {}) or {}
        for method, block in metrics.items():
            if not isinstance(block, dict):
                continue
            for metric_name in metric_names:
                metric_val = _finite(block.get(metric_name))
                if metric_val is None:
                    continue
                rows.append(
                    NormalizedRow(
                        family=self.family,
                        scenario=scenario,
                        seed=int(cfg["seed"]) if cfg.get("seed") is not None else None,
                        method=str(method),
                        x_axis_name="train_docs",
                        x_axis_value=float(train_docs if train_docs is not None else float("nan")),
                        secondary_axis_name="oracle_budget_fraction",
                        secondary_axis_value=float(max(x for x in (eval_leaf, eval_internal, cal_rate) if x is not None)) if any(x is not None for x in (eval_leaf, eval_internal, cal_rate)) else None,
                        metric_name=str(metric_name),
                        metric_value=float(metric_val),
                        doc_scale_tokens=doc_scale_tokens,
                        leaf_tokens=leaf_tokens,
                        leaves_per_doc=leaves_per_doc,
                        oracle_budget_fraction=float(max(x for x in (eval_leaf, eval_internal, cal_rate) if x is not None)) if any(x is not None for x in (eval_leaf, eval_internal, cal_rate)) else None,
                        train_docs=train_docs,
                        evidence_status=_evidence_status_for_method(self.family, method),
                        source_path=str(path.resolve()),
                        test_identity=(
                            test_identity
                            or (
                                str(topic_meta.get("corpus_signature_test"))
                                if topic_meta.get("corpus_signature_test") is not None
                                else None
                            )
                        ),
                        metadata={
                            "calibration_leaf_query_rate": cal_rate,
                            "eval_leaf_query_rate": eval_leaf,
                            "eval_internal_query_rate": eval_internal,
                            "support_total": float(support_total),
                            "granularity_group": granularity_group,
                            **objective_meta,
                        },
                    )
                )
        decomp = payload.get("decomposition", {}) or {}
        for metric_name in ("total_root_l1_mean", "upper_bound_mean", "slack_mean"):
            metric_val = _finite(decomp.get(metric_name))
            if metric_val is None:
                continue
            rows.append(
                NormalizedRow(
                    family=self.family,
                    scenario=scenario,
                    seed=int(cfg["seed"]) if cfg.get("seed") is not None else None,
                    method="decomposition",
                    x_axis_name="train_docs",
                    x_axis_value=float(train_docs if train_docs is not None else float("nan")),
                    secondary_axis_name="oracle_budget_fraction",
                    secondary_axis_value=float(max(x for x in (eval_leaf, eval_internal, cal_rate) if x is not None)) if any(x is not None for x in (eval_leaf, eval_internal, cal_rate)) else None,
                    metric_name=str(metric_name),
                    metric_value=float(metric_val),
                    doc_scale_tokens=doc_scale_tokens,
                    leaf_tokens=leaf_tokens,
                    leaves_per_doc=leaves_per_doc,
                    oracle_budget_fraction=float(max(x for x in (eval_leaf, eval_internal, cal_rate) if x is not None)) if any(x is not None for x in (eval_leaf, eval_internal, cal_rate)) else None,
                    train_docs=train_docs,
                    evidence_status="APPROX_AUDITED",
                    source_path=str(path.resolve()),
                    test_identity=(
                        test_identity
                        or (
                            str(topic_meta.get("corpus_signature_test"))
                            if topic_meta.get("corpus_signature_test") is not None
                            else None
                        )
                    ),
                    metadata={
                        "calibration_leaf_query_rate": cal_rate,
                        "eval_leaf_query_rate": eval_leaf,
                        "eval_internal_query_rate": eval_internal,
                        "support_total": float(support_total),
                        "granularity_group": granularity_group,
                        **objective_meta,
                    },
                )
            )
        return rows

    def build_expectations(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        findings: List[ExpectationFinding] = []
        by_scenario: Dict[str, List[NormalizedRow]] = {}
        for row in rows:
            by_scenario.setdefault(row.scenario, []).append(row)

        for scenario, srows in sorted(by_scenario.items()):
            oracle_rows = _row_subset(srows, method="oracle_tree", metric_name="root_l1_mean")
            findings.append(
                CeilingExpectation(
                    family=self.family,
                    scenario=scenario,
                    method="oracle_tree",
                    metric="root_l1_mean",
                    epsilon=config.ceiling_epsilon_float,
                    title="Segmented-LDA oracle_tree root_l1_mean ceiling",
                ).evaluate(oracle_rows)
            )

            for axis_name, title in (
                ("train_docs", "Segmented-LDA budgeted root_l1_mean improves with train_docs"),
                ("calibration_leaf_query_rate", "Segmented-LDA budgeted root_l1_mean improves with calibration labels"),
                ("eval_leaf_query_rate", "Segmented-LDA budgeted root_l1_mean improves with eval leaf queries"),
                ("eval_internal_query_rate", "Segmented-LDA budgeted root_l1_mean improves with eval internal queries"),
            ):
                budgeted_rows = _row_subset(srows, method="estimated_calibrated_budgeted", metric_name="root_l1_mean")
                if axis_name == "train_docs":
                    anchor_rows = [
                        r
                        for r in budgeted_rows
                        if r.value_for("calibration_leaf_query_rate") is not None
                        and r.value_for("eval_leaf_query_rate") is not None
                        and r.value_for("eval_internal_query_rate") is not None
                    ]
                    if anchor_rows:
                        max_cal = max(float(r.value_for("calibration_leaf_query_rate")) for r in anchor_rows if r.value_for("calibration_leaf_query_rate") is not None)
                        max_leaf = max(float(r.value_for("eval_leaf_query_rate")) for r in anchor_rows if r.value_for("eval_leaf_query_rate") is not None)
                        max_internal = max(float(r.value_for("eval_internal_query_rate")) for r in anchor_rows if r.value_for("eval_internal_query_rate") is not None)
                        anchor_rows = [
                            r
                            for r in anchor_rows
                            if r.value_for("calibration_leaf_query_rate") is not None
                            and r.value_for("eval_leaf_query_rate") is not None
                            and r.value_for("eval_internal_query_rate") is not None
                            and abs(float(r.value_for("calibration_leaf_query_rate")) - max_cal) <= 1e-12
                            and abs(float(r.value_for("eval_leaf_query_rate")) - max_leaf) <= 1e-12
                            and abs(float(r.value_for("eval_internal_query_rate")) - max_internal) <= 1e-12
                        ]
                else:
                    anchor_rows = _anchor_slice(budgeted_rows, primary_field="train_docs")
                    if anchor_rows:
                        max_cal = max(
                            float(r.value_for("calibration_leaf_query_rate"))
                            for r in anchor_rows
                            if r.value_for("calibration_leaf_query_rate") is not None
                        )
                        max_leaf = max(
                            float(r.value_for("eval_leaf_query_rate"))
                            for r in anchor_rows
                            if r.value_for("eval_leaf_query_rate") is not None
                        )
                        max_internal = max(
                            float(r.value_for("eval_internal_query_rate"))
                            for r in anchor_rows
                            if r.value_for("eval_internal_query_rate") is not None
                        )
                        filtered: List[NormalizedRow] = []
                        for row in anchor_rows:
                            cal_val = row.value_for("calibration_leaf_query_rate")
                            leaf_val = row.value_for("eval_leaf_query_rate")
                            internal_val = row.value_for("eval_internal_query_rate")
                            if cal_val is None or leaf_val is None or internal_val is None:
                                continue
                            if axis_name != "calibration_leaf_query_rate" and abs(float(cal_val) - max_cal) > 1e-12:
                                continue
                            if axis_name != "eval_leaf_query_rate" and abs(float(leaf_val) - max_leaf) > 1e-12:
                                continue
                            if axis_name != "eval_internal_query_rate" and abs(float(internal_val) - max_internal) > 1e-12:
                                continue
                            filtered.append(row)
                        anchor_rows = filtered
                findings.append(
                    BudgetTrendExpectation(
                        family=self.family,
                        scenario=scenario,
                        method="estimated_calibrated_budgeted",
                        metric="root_l1_mean",
                        axis_name=axis_name,
                        direction="decreasing",
                        title=title,
                    ).evaluate(anchor_rows, config=config)
                )

            unc_anchor = _anchor_slice(
                _row_subset(srows, method="estimated_uncalibrated", metric_name="root_l1_mean"),
                primary_field="train_docs",
                secondary_field="oracle_budget_fraction",
            )
            cal_anchor = _anchor_slice(
                _row_subset(srows, method="estimated_calibrated", metric_name="root_l1_mean"),
                primary_field="train_docs",
                secondary_field="oracle_budget_fraction",
            )
            unc_v = _aggregate_metric(unc_anchor, config=config)
            cal_v = _aggregate_metric(cal_anchor, config=config)
            scale = _metric_scale([unc_v, cal_v])
            tol = max(config.min_effect_abs_scale * scale, config.calibrated_regression_rel * max(abs(unc_v), 1e-12))
            if unc_anchor and cal_anchor:
                if cal_v <= unc_v + tol:
                    status: ExpectationStatus = "pass"
                    note = "calibration does not materially regress the uncalibrated estimator"
                else:
                    status = "fail"
                    note = "calibration regresses the uncalibrated estimator beyond tolerance"
            else:
                status = "not_applicable"
                note = "missing calibrated or uncalibrated anchor rows"
            findings.append(
                ExpectationFinding(
                    kind="failure_mode",
                    title="Segmented-LDA calibration does not materially regress the uncalibrated estimator",
                    status=status,
                    family=self.family,
                    scenario=scenario,
                    metric="root_l1_mean",
                    method="estimated_calibrated vs estimated_uncalibrated",
                    direction="decreasing",
                    observed_summary={
                        "estimated_calibrated_root_l1_mean": float(cal_v),
                        "estimated_uncalibrated_root_l1_mean": float(unc_v),
                        "note": note,
                    },
                    thresholds={"max_allowed_regression": float(tol)},
                    supporting_rows=_supporting_rows([*unc_anchor, *cal_anchor]),
                )
            )

            decomp_total = _anchor_slice(_row_subset(srows, method="decomposition", metric_name="total_root_l1_mean"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
            decomp_upper = _anchor_slice(_row_subset(srows, method="decomposition", metric_name="upper_bound_mean"), primary_field="train_docs", secondary_field="oracle_budget_fraction")
            total_v = _aggregate_metric(decomp_total, config=config)
            upper_v = _aggregate_metric(decomp_upper, config=config)
            if decomp_total and decomp_upper:
                if upper_v + 1e-9 >= total_v:
                    status = "pass"
                    note = "decomposition upper bound dominates total error"
                else:
                    status = "fail"
                    note = "decomposition upper bound is violated"
            else:
                status = "not_applicable"
                note = "decomposition rows are missing"
            findings.append(
                ExpectationFinding(
                    kind="ceiling",
                    title="Segmented-LDA decomposition upper bound dominates total error",
                    status=status,
                    family=self.family,
                    scenario=scenario,
                    metric="upper_bound_mean",
                    method="decomposition",
                    direction="flat",
                    observed_summary={
                        "total_root_l1_mean": float(total_v),
                        "upper_bound_mean": float(upper_v),
                        "note": note,
                    },
                    thresholds={"tolerance": 1e-9},
                    supporting_rows=_supporting_rows([*decomp_total, *decomp_upper]),
                )
            )

        by_granularity: Dict[str, List[NormalizedRow]] = {}
        for row in rows:
            if row.method == "estimated_calibrated_budgeted" and row.metric_name == "root_l1_mean":
                by_granularity.setdefault(str(row.metadata.get("granularity_group", row.scenario)), []).append(row)
        for key, grows in sorted(by_granularity.items()):
            findings.append(
                GranularitySensitivityExpectation(
                    family=self.family,
                    scenario=key,
                    method="estimated_calibrated_budgeted",
                    metric="root_l1_mean",
                    support_axis="support_total",
                    title="Segmented-LDA granularity sensitivity: high-support region dominates low-support region",
                ).evaluate(grows, config=config)
            )
        return findings


class MergeableAblationAdapter:
    family = FAMILY_MERGEABLE

    def can_load(self, path: Path) -> bool:
        parts = _path_parts_lower(path)
        if "mergeable" not in parts and not _path_contains_any(path, ("mergeable_",)):
            return False
        try:
            payload = _load_json(path)
        except Exception:
            return False
        if "stage_rows" in payload and "stage_metrics" in payload:
            return True
        if "rows" in payload and "budget_rows" in payload and "target_ks" in payload:
            return True
        if "rows" in payload and "chunk_sizes" in payload and "chunk_budgets" in payload:
            return True
        if "summaries" in payload and "distribution" in payload:
            return True
        return False

    def load_rows(self, path: Path) -> List[NormalizedRow]:
        payload = _load_json(path)
        if "stage_rows" in payload and "stage_metrics" in payload:
            return self._load_complexity_rows(path, payload)
        if "rows" in payload and "budget_rows" in payload and "target_ks" in payload:
            return self._load_k_phase_rows(path, payload)
        if "rows" in payload and "chunk_sizes" in payload and "chunk_budgets" in payload:
            return self._load_chunk_quality_rows(path, payload)
        if "summaries" in payload and "distribution" in payload:
            return self._load_param_recovery_rows(path, payload)
        return []

    def _load_chunk_quality_rows(self, path: Path, payload: Dict[str, Any]) -> List[NormalizedRow]:
        distribution = payload.get("distribution", {}) or {}
        objective_meta = _objective_metadata(payload)
        n_tokens = _finite(distribution.get("n_tokens"))
        scenario = _slug(
            {
                "variant": "chunk_quality",
                "chunker": payload.get("chunker", "unknown"),
                "selector": payload.get("selector", "unknown"),
                "target_k": payload.get("target_k", "na"),
                "sketch_order": payload.get("sketch_order", "na"),
            }
        )
        rows: List[NormalizedRow] = []
        for row in payload.get("rows", []) or []:
            metric_val = _finite(row.get("mean_abs_bias"))
            chunk_budget = _finite(row.get("chunk_budget"))
            leaf_tokens = _finite(row.get("fixed_chunk_size")) or _finite(row.get("max_chunk_size"))
            if metric_val is None or chunk_budget is None:
                continue
            leaves_per_doc = None
            if n_tokens is not None and leaf_tokens not in {None, 0.0}:
                leaves_per_doc = float(n_tokens) / max(1.0, float(leaf_tokens))
            rows.append(
                NormalizedRow(
                    family=self.family,
                    scenario=scenario,
                    seed=None,
                    method=str(row.get("method_name", "unknown")),
                    x_axis_name="chunk_budget",
                    x_axis_value=float(chunk_budget),
                    secondary_axis_name="leaf_tokens",
                    secondary_axis_value=leaf_tokens,
                    metric_name="mean_abs_bias",
                    metric_value=float(metric_val),
                    doc_scale_tokens=n_tokens,
                    leaf_tokens=leaf_tokens,
                    leaves_per_doc=leaves_per_doc,
                    oracle_budget_fraction=None,
                    train_docs=None,
                    evidence_status=_evidence_status_for_method(self.family, str(row.get("method_name", "unknown"))),
                    source_path=str(path.resolve()),
                    metadata={
                        "variant": "chunk_quality",
                        "supports_target": bool(row.get("supports_target", False)),
                        **objective_meta,
                    },
                )
            )
        return rows

    def _load_k_phase_rows(self, path: Path, payload: Dict[str, Any]) -> List[NormalizedRow]:
        distribution = payload.get("distribution", {}) or {}
        objective_meta = _objective_metadata(payload)
        n_tokens = _finite(distribution.get("n_tokens"))
        scenario = _slug({"variant": "k_phase"})
        rows: List[NormalizedRow] = []
        for row in payload.get("rows", []) or []:
            method = str(row.get("method_name", "unknown"))
            metric_val = _finite(row.get("mean_abs_bias"))
            if metric_val is None:
                continue
            rows.append(
                NormalizedRow(
                    family=self.family,
                    scenario=scenario,
                    seed=None,
                    method=method,
                    x_axis_name="target_k",
                    x_axis_value=float(row.get("target_k", float("nan"))),
                    secondary_axis_name="sketch_order",
                    secondary_axis_value=_finite(row.get("sketch_order")),
                    metric_name="mean_abs_bias",
                    metric_value=float(metric_val),
                    doc_scale_tokens=n_tokens,
                    leaf_tokens=None,
                    leaves_per_doc=None,
                    oracle_budget_fraction=None,
                    train_docs=None,
                    evidence_status=_evidence_status_for_method(self.family, method),
                    source_path=str(path.resolve()),
                    metadata={
                        "variant": "k_phase",
                        "supports_target": bool(row.get("supports_target", False)),
                        "target_k": _finite(row.get("target_k")),
                        "sketch_order": _finite(row.get("sketch_order")),
                        **objective_meta,
                    },
                )
            )
        for row in payload.get("budget_rows", []) or []:
            method = str(row.get("method_name", "unknown"))
            metric_val = _finite(row.get("mean_abs_bias"))
            budget = _finite(row.get("chunk_budget"))
            if metric_val is None or budget is None:
                continue
            rows.append(
                NormalizedRow(
                    family=self.family,
                    scenario=scenario,
                    seed=None,
                    method=method,
                    x_axis_name="chunk_budget",
                    x_axis_value=float(budget),
                    secondary_axis_name="target_k",
                    secondary_axis_value=_finite(row.get("target_k")),
                    metric_name="mean_abs_bias",
                    metric_value=float(metric_val),
                    doc_scale_tokens=n_tokens,
                    leaf_tokens=None,
                    leaves_per_doc=None,
                    oracle_budget_fraction=None,
                    train_docs=None,
                    evidence_status=_evidence_status_for_method(self.family, method),
                    source_path=str(path.resolve()),
                    metadata={
                        "variant": "k_phase_budget",
                        "target_k": _finite(row.get("target_k")),
                        **objective_meta,
                    },
                )
            )
        return rows

    def _load_complexity_rows(self, path: Path, payload: Dict[str, Any]) -> List[NormalizedRow]:
        cfg = payload.get("config", {}) or {}
        objective_meta = _objective_metadata(payload)
        n_tokens = _finite(cfg.get("n_tokens"))
        stage_order = list(payload.get("stage_order", []) or [])
        stage_index = {str(stage): idx for idx, stage in enumerate(stage_order)}
        rows: List[NormalizedRow] = []
        stage_metrics = payload.get("stage_metrics", {}) or {}
        for method, metric_map in stage_metrics.items():
            if not isinstance(metric_map, dict):
                continue
            for stage_key, val in metric_map.items():
                metric_val = _finite(val)
                if metric_val is None:
                    continue
                rows.append(
                    NormalizedRow(
                        family=self.family,
                        scenario="complexity_ladder",
                        seed=None,
                        method=str(method),
                        x_axis_name="stage_index",
                        x_axis_value=float(stage_index.get(str(stage_key), len(stage_index))),
                        secondary_axis_name="stage_key",
                        secondary_axis_value=None,
                        metric_name="aggregate_mean_abs_bias",
                        metric_value=float(metric_val),
                        doc_scale_tokens=n_tokens,
                        leaf_tokens=None,
                        leaves_per_doc=None,
                        oracle_budget_fraction=None,
                        train_docs=None,
                        evidence_status=_evidence_status_for_method(self.family, str(method)),
                        source_path=str(path.resolve()),
                        metadata={
                            "variant": "complexity",
                            "stage_key": str(stage_key),
                            **objective_meta,
                        },
                    )
                )
        stage_rows = payload.get("stage_rows", {}) or {}
        stage4 = stage_rows.get("stage4", {}) or {}
        for method, row in stage4.items():
            if not isinstance(row, dict):
                continue
            for metric_name in (
                "mean_abs_bias_p_spike",
                "mean_abs_bias_p_two_given_spike",
                "mean_abs_bias_p_three_given_spike",
                "mean_abs_bias_p_boundary_given_spike",
            ):
                metric_val = _finite(row.get(metric_name))
                if metric_val is None:
                    continue
                rows.append(
                    NormalizedRow(
                        family=self.family,
                        scenario="complexity_ladder",
                        seed=None,
                        method=str(method),
                        x_axis_name="stage_index",
                        x_axis_value=float(stage_index.get("stage4", 3)),
                        secondary_axis_name=None,
                        secondary_axis_value=None,
                        metric_name=str(metric_name),
                        metric_value=float(metric_val),
                        doc_scale_tokens=n_tokens,
                        leaf_tokens=None,
                        leaves_per_doc=None,
                        oracle_budget_fraction=None,
                        train_docs=None,
                        evidence_status=_evidence_status_for_method(self.family, str(method)),
                        source_path=str(path.resolve()),
                        metadata={"variant": "complexity_stage4", **objective_meta},
                    )
                )
        return rows

    def _load_param_recovery_rows(self, path: Path, payload: Dict[str, Any]) -> List[NormalizedRow]:
        distribution = payload.get("distribution", {}) or {}
        objective_meta = _objective_metadata(payload)
        n_tokens = _finite(distribution.get("n_tokens"))
        rows: List[NormalizedRow] = []
        for row in payload.get("summaries", []) or []:
            metric_val = _finite(row.get("mean_abs_bias"))
            if metric_val is None:
                continue
            method = str(row.get("method_name", "unknown"))
            rows.append(
                NormalizedRow(
                    family=self.family,
                    scenario="param_recovery",
                    seed=None,
                    method=method,
                    x_axis_name="scenario_index",
                    x_axis_value=0.0,
                    secondary_axis_name=None,
                    secondary_axis_value=None,
                    metric_name="mean_abs_bias",
                    metric_value=float(metric_val),
                    doc_scale_tokens=n_tokens,
                    leaf_tokens=None,
                    leaves_per_doc=None,
                    oracle_budget_fraction=None,
                    train_docs=None,
                    evidence_status=_evidence_status_for_method(self.family, method),
                    source_path=str(path.resolve()),
                    metadata={"variant": "param_recovery", **objective_meta},
                )
            )
        return rows

    def build_expectations(
        self,
        rows: Sequence[NormalizedRow],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        findings: List[ExpectationFinding] = []

        chunk_quality_rows = [r for r in rows if r.metadata.get("variant") == "chunk_quality"]
        if chunk_quality_rows:
            findings.append(
                GranularitySensitivityExpectation(
                    family=self.family,
                    scenario="chunk_quality",
                    method="grid_*",
                    metric="mean_abs_bias",
                    support_axis="chunk_budget",
                    title="Mergeable chunk-quality sweep captures non-monotone leaf-size effects without assuming smaller is always better",
                ).evaluate(chunk_quality_rows, config=config)
            )
            ref_rows = [r for r in chunk_quality_rows if r.method == "one_pass_reference"]
            grid_rows = [r for r in chunk_quality_rows if r.method.startswith("grid_")]
            if ref_rows and grid_rows:
                high_budget = max(float(r.x_axis_value) for r in grid_rows)
                grid_high = [r for r in grid_rows if abs(float(r.x_axis_value) - high_budget) <= 1e-12]
                if grid_high:
                    best_grid = min(float(r.metric_value) for r in grid_high)
                    ref_val = _aggregate_metric(ref_rows, config=config)
                    scale = _metric_scale([best_grid, ref_val])
                    tol = max(config.min_effect_abs_scale * scale, config.separation_rel * max(abs(ref_val), 1e-12))
                    if best_grid <= ref_val + tol:
                        status: ExpectationStatus = "pass"
                        note = "best high-budget grid approaches the one-pass reference"
                    else:
                        status = "fail"
                        note = "best high-budget grid remains too far from the one-pass reference"
                    findings.append(
                        ExpectationFinding(
                            kind="failure_mode",
                            title="Mergeable chunk-quality sweep: aligned high-budget regime approaches one-pass reference",
                            status=status,
                            family=self.family,
                            scenario="chunk_quality",
                            metric="mean_abs_bias",
                            method="grid_* vs one_pass_reference",
                            direction="decreasing",
                            observed_summary={
                                "best_high_budget_grid_abs_bias": float(best_grid),
                                "one_pass_reference_abs_bias": float(ref_val),
                                "note": note,
                            },
                            thresholds={"max_gap": float(tol)},
                            supporting_rows=_supporting_rows([*ref_rows, *grid_high]),
                        )
                    )

        k_phase_rows = [r for r in rows if r.metadata.get("variant") == "k_phase"]
        if k_phase_rows:
            by_target: Dict[float, List[NormalizedRow]] = {}
            for row in k_phase_rows:
                target_k = row.value_for("target_k") or row.x_axis_value
                by_target.setdefault(float(target_k), []).append(row)
            summaries: List[Dict[str, object]] = []
            comparable_targets = 0
            missing_targets = 0
            exact_pass = True
            for target_k, trows in sorted(by_target.items()):
                exact = _safe_mean(
                    r.metric_value
                    for r in trows
                    if r.method.startswith("full_model_m")
                    and r.metadata.get("sketch_order") is not None
                    and abs(float(r.metadata.get("sketch_order")) - float(target_k)) <= 1e-12
                )
                unsupported = _safe_mean(
                    r.metric_value
                    for r in trows
                    if r.method.startswith("full_model_m")
                    and r.metadata.get("sketch_order") is not None
                    and float(r.metadata.get("sketch_order")) < float(target_k)
                )
                oversup = _safe_mean(
                    r.metric_value
                    for r in trows
                    if r.method.startswith("full_model_m")
                    and r.metadata.get("sketch_order") is not None
                    and float(r.metadata.get("sketch_order")) > float(target_k)
                )
                scale = _metric_scale([exact, unsupported, oversup])
                min_gap = config.min_effect_abs_scale * scale
                has_exact = math.isfinite(exact)
                has_unsupported = math.isfinite(unsupported)
                if has_exact and has_unsupported:
                    comparable_targets += 1
                    if unsupported < exact + min_gap:
                        exact_pass = False
                else:
                    missing_targets += 1
                summaries.append(
                    {
                        "target_k": float(target_k),
                        "exact_mean_abs_bias": float(exact),
                        "unsupported_mean_abs_bias": float(unsupported),
                        "oversupported_mean_abs_bias": float(oversup),
                        "comparable": bool(has_exact and has_unsupported),
                    }
                )
            if comparable_targets == 0:
                status: ExpectationStatus = "not_applicable"
                note = "no target_k has both exact and unsupported comparators"
            elif exact_pass and missing_targets == 0:
                status = "pass"
                note = "every comparable target_k shows m<k materially worse than exact support"
            elif exact_pass:
                status = "warn"
                note = "comparable targets pass, but some target_k values are missing unsupported comparators"
            else:
                status = "fail"
                note = "at least one comparable target_k does not show a clear unsupported penalty"
            findings.append(
                ExpectationFinding(
                    kind="failure_mode",
                    title="Mergeable k-vs-m phase: insufficient sketch order (m<k) is materially worse than exact support",
                    status=status,
                    family=self.family,
                    scenario="k_phase",
                    metric="mean_abs_bias",
                    method="full_model_m*",
                    direction="increasing",
                    observed_summary={
                        "targets": summaries,
                        "comparable_targets": int(comparable_targets),
                        "missing_targets": int(missing_targets),
                        "note": note,
                    },
                    thresholds={"min_gap_abs_scale": float(config.min_effect_abs_scale)},
                    supporting_rows=_supporting_rows(k_phase_rows),
                )
            )

            naive_rows = [r for r in k_phase_rows if r.method in {"naive_majority", "naive_mean_of_means"}]
            full_exact = [
                r for r in k_phase_rows if r.method.startswith("full_model_m") and r.metadata.get("target_k") == r.metadata.get("sketch_order")
            ]
            if naive_rows and full_exact:
                findings.append(
                    FailureModeExpectation(
                        family=self.family,
                        scenario="k_phase",
                        metric="mean_abs_bias",
                        intended_method="full_model_exact",
                        mismatch_method="naive_baselines",
                        title="Mergeable k-vs-m phase: naive baselines remain worse than supported full-model runs",
                        separation_rel=config.separation_rel,
                    ).evaluate(full_exact, naive_rows, config=config)
                )

        k_phase_budget_rows = [r for r in rows if r.metadata.get("variant") == "k_phase_budget"]
        if k_phase_budget_rows:
            full_budget = [r for r in k_phase_budget_rows if r.method.startswith("budget_full_model_")]
            if full_budget:
                findings.append(
                    BudgetTrendExpectation(
                        family=self.family,
                        scenario="k_phase_budget",
                        method="budget_full_model_*",
                        metric="mean_abs_bias",
                        axis_name="chunk_budget",
                        direction="decreasing",
                        title="Mergeable budget sweep: aligned full-model improves with chunk budget",
                    ).evaluate(full_budget, config=config)
                )
            wrong_budget = [r for r in k_phase_budget_rows if r.method.startswith("budget_wrong_chunker_")]
            if full_budget and wrong_budget:
                findings.append(
                    FailureModeExpectation(
                        family=self.family,
                        scenario="k_phase_budget",
                        metric="mean_abs_bias",
                        intended_method="budget_full_model_*",
                        mismatch_method="budget_wrong_chunker_*",
                        title="Mergeable budget sweep: wrong chunker remains worse than aligned full-model at high budget",
                        separation_rel=config.separation_rel,
                    ).evaluate(
                        _anchor_slice(full_budget, primary_field="chunk_budget"),
                        _anchor_slice(wrong_budget, primary_field="chunk_budget"),
                        config=config,
                    )
                )

        complexity_rows = [r for r in rows if r.metadata.get("variant") == "complexity"]
        if complexity_rows:
            one_pass = [r for r in complexity_rows if r.method == "one_pass_oracle" and r.metric_name == "aggregate_mean_abs_bias"]
            full_model = [r for r in complexity_rows if r.method == "full_model_aligned" and r.metric_name == "aggregate_mean_abs_bias"]
            if one_pass and full_model:
                by_stage: Dict[str, Dict[str, float]] = {}
                for row in [*one_pass, *full_model]:
                    stage_key = str(row.metadata.get("stage_key", "unknown"))
                    by_stage.setdefault(stage_key, {})
                    by_stage[stage_key][row.method] = float(row.metric_value)
                common = [
                    {
                        "stage_key": stage_key,
                        "one_pass_oracle": vals.get("one_pass_oracle", float("nan")),
                        "full_model_aligned": vals.get("full_model_aligned", float("nan")),
                    }
                    for stage_key, vals in sorted(by_stage.items())
                    if "one_pass_oracle" in vals and "full_model_aligned" in vals
                ]
                if common:
                    gaps = [float(x["full_model_aligned"]) - float(x["one_pass_oracle"]) for x in common]
                    scale = _metric_scale(
                        [float(x["one_pass_oracle"]) for x in common] + [float(x["full_model_aligned"]) for x in common]
                    )
                    tol = max(config.min_effect_abs_scale * scale, config.separation_rel * max(abs(float(common[-1]["one_pass_oracle"])), 1e-12))
                    if max(gaps) <= tol:
                        status = "pass"
                        note = "full model tracks one-pass oracle across the complexity ladder"
                    else:
                        status = "fail"
                        note = "full model diverges too far from the one-pass oracle on the ladder"
                    findings.append(
                        ExpectationFinding(
                            kind="failure_mode",
                            title="Mergeable complexity ladder: full model tracks the one-pass oracle",
                            status=status,
                            family=self.family,
                            scenario="complexity_ladder",
                            metric="aggregate_mean_abs_bias",
                            method="full_model_aligned vs one_pass_oracle",
                            direction="decreasing",
                            observed_summary={"stages": common, "note": note},
                            thresholds={"max_allowed_gap": float(tol)},
                            supporting_rows=_supporting_rows([*one_pass, *full_model]),
                        )
                    )
            hardest_stage = _anchor_slice(complexity_rows, primary_field="stage_index")
            wrong = [r for r in hardest_stage if r.method == "right_rule_wrong_chunker" and r.metric_name == "aggregate_mean_abs_bias"]
            full = [r for r in hardest_stage if r.method == "full_model_aligned" and r.metric_name == "aggregate_mean_abs_bias"]
            if wrong and full:
                findings.append(
                    FailureModeExpectation(
                        family=self.family,
                        scenario="complexity_ladder",
                        metric="aggregate_mean_abs_bias",
                        intended_method="full_model_aligned",
                        mismatch_method="right_rule_wrong_chunker",
                        title="Mergeable complexity ladder: right rule + wrong chunker remains structurally worse",
                        separation_rel=config.separation_rel,
                    ).evaluate(full, wrong, config=config)
                )

        complexity_stage4_rows = [r for r in rows if r.metadata.get("variant") == "complexity_stage4"]
        if complexity_stage4_rows:
            for method, target_metric in (
                ("full_model_missing_boundary_stat", "mean_abs_bias_p_boundary_given_spike"),
                ("full_model_missing_three_stat", "mean_abs_bias_p_three_given_spike"),
            ):
                method_rows = [r for r in complexity_stage4_rows if r.method == method]
                if not method_rows:
                    continue
                target_rows = [r for r in method_rows if r.metric_name == target_metric]
                other_rows = [r for r in method_rows if r.metric_name != target_metric]
                target_val = _aggregate_metric(target_rows, config=config)
                other_val = _aggregate_metric(other_rows, config=config)
                scale = _metric_scale([target_val, other_val])
                min_gap = config.min_effect_abs_scale * scale
                if math.isfinite(target_val) and math.isfinite(other_val):
                    if target_val >= other_val + min_gap:
                        status = "pass"
                        note = "missing-stat method fails specifically on the target it underspecifies"
                    elif target_val >= other_val:
                        status = "warn"
                        note = "target-specific failure is present but weak"
                    else:
                        status = "fail"
                        note = "failure is not target-specific"
                else:
                    status = "not_applicable"
                    note = "missing target or comparison metrics"
                findings.append(
                    ExpectationFinding(
                        kind="failure_mode",
                        title=f"Mergeable stage-4 target-specific failure attribution: {method}",
                        status=status,
                        family=self.family,
                        scenario="complexity_ladder",
                        metric=target_metric,
                        method=method,
                        direction="increasing",
                        observed_summary={
                            "target_metric_value": float(target_val),
                            "other_metric_value": float(other_val),
                            "note": note,
                        },
                        thresholds={"min_gap": float(min_gap)},
                        supporting_rows=_supporting_rows(method_rows),
                    )
                )

        param_rows = [r for r in rows if r.metadata.get("variant") == "param_recovery"]
        if param_rows:
            one_pass = [r for r in param_rows if r.method == "one_pass_oracle"]
            full_model = [r for r in param_rows if r.method == "full_model_aligned"]
            if one_pass and full_model:
                findings.append(
                    FailureModeExpectation(
                        family=self.family,
                        scenario="param_recovery",
                        metric="mean_abs_bias",
                        intended_method="one_pass_oracle",
                        mismatch_method="full_model_aligned",
                        title="Mergeable param recovery: full model stays close to one-pass oracle",
                        separation_rel=0.0,
                    ).evaluate(one_pass, full_model, config=config)
                )
        return findings


DEFAULT_ADAPTERS: List[FamilyAdapter] = [
    MarkovOPSAdapter(),
    SegmentLDAOPSAdapter(),
    SegmentedLDACtreePOAdapter(),
    MergeableAblationAdapter(),
]


def _collect_paths_from_manifest(manifest_path: Path) -> List[Path]:
    runs = read_manifest_jsonl(manifest_path)
    out: List[Path] = []
    for run in runs:
        for value in dict(run.outputs).values():
            text = str(value).strip()
            if not text.endswith(".json"):
                continue
            p = Path(text)
            if p.exists():
                out.append(p.resolve())
    return sorted({p for p in out})


def _collect_paths(output_root: Optional[Path], manifest_path: Optional[Path]) -> List[Path]:
    out: List[Path] = []
    if output_root is not None:
        out.extend(sorted(output_root.resolve().rglob("*.json")))
    if manifest_path is not None and manifest_path.exists():
        out.extend(_collect_paths_from_manifest(manifest_path.resolve()))
    return sorted({p.resolve() for p in out})


def build_expectation_report(
    *,
    output_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    families: Optional[Sequence[str]] = None,
    config: Optional[ExpectationConfig] = None,
) -> ExpectationReport:
    cfg = config or ExpectationConfig()
    family_filter = {str(x) for x in (families or VALID_FAMILIES)}
    all_paths = _collect_paths(output_root, manifest_path)
    rows_by_family: Dict[str, List[NormalizedRow]] = {f: [] for f in VALID_FAMILIES}
    for path in all_paths:
        for adapter in DEFAULT_ADAPTERS:
            if adapter.family not in family_filter:
                continue
            if adapter.can_load(path):
                rows_by_family[adapter.family].extend(adapter.load_rows(path))
                break

    expectations: List[ExpectationFinding] = []
    for adapter in DEFAULT_ADAPTERS:
        if adapter.family not in family_filter:
            continue
        fam_rows = rows_by_family.get(adapter.family, [])
        if not fam_rows:
            continue
        expectations.extend(adapter.build_expectations(fam_rows, config=cfg))

    structured_local_law_loaded: List[tuple[Path, LocalLawRunSummary, Dict[str, Any]]] = []
    if FAMILY_LOCAL_LAW in family_filter:
        structured_local_law_adapter = StructuredLocalLawAdapter()
        structured_local_law_loaded = structured_local_law_adapter.load_summaries(all_paths)
        if structured_local_law_loaded:
            expectations.extend(
                structured_local_law_adapter.build_expectations(
                    structured_local_law_loaded,
                    config=cfg,
                )
            )

    families_scanned = sorted(
        {
            *[k for k, rows in rows_by_family.items() if rows],
            *([FAMILY_LOCAL_LAW] if structured_local_law_loaded else []),
        }
    )

    expectations.sort(key=lambda x: (STATUS_PRIORITY.get(x.status, 99), x.family, x.title))
    return ExpectationReport(
        input_root=str(output_root.resolve()) if output_root is not None else None,
        manifest=str(manifest_path.resolve()) if manifest_path is not None else None,
        families_scanned=families_scanned,
        rows_scanned=int(sum(len(v) for v in rows_by_family.values()) + len(structured_local_law_loaded)),
        expectations=expectations,
        summary=_expectation_summary(expectations),
    )


def render_expectation_markdown(report: ExpectationReport) -> str:
    lines: List[str] = []
    lines.append("# Simulation Expectation Report")
    lines.append("")
    if report.input_root is not None:
        lines.append(f"- Input root: `{report.input_root}`")
    if report.manifest is not None:
        lines.append(f"- Manifest: `{report.manifest}`")
    lines.append(f"- Families scanned: `{', '.join(report.families_scanned) if report.families_scanned else 'none'}`")
    lines.append(f"- Rows scanned: `{report.rows_scanned}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Pass: `{report.summary.get('n_pass', 0)}`")
    lines.append(f"- Warn: `{report.summary.get('n_warn', 0)}`")
    lines.append(f"- Fail: `{report.summary.get('n_fail', 0)}`")
    lines.append(f"- Not applicable: `{report.summary.get('n_not_applicable', 0)}`")
    lines.append("")

    by_family: Dict[str, List[ExpectationFinding]] = {}
    for finding in report.expectations:
        by_family.setdefault(finding.family, []).append(finding)
    for family in sorted(by_family.keys()):
        lines.append(f"## {family}")
        lines.append("")
        family_findings = sorted(
            by_family[family],
            key=lambda x: (STATUS_PRIORITY.get(x.status, 99), x.kind, x.title),
        )
        for finding in family_findings:
            lines.append(
                f"- `{finding.status.upper()}` `{finding.kind}` {finding.title}"
            )
            note = str((finding.observed_summary or {}).get("note", "")).strip()
            if note:
                lines.append(f"  Note: {note}")
            if "intended_value" in finding.observed_summary and "mismatch_value" in finding.observed_summary:
                lines.append(
                    "  "
                    f"Values: intended={finding.observed_summary['intended_value']:.6g}, "
                    f"mismatch={finding.observed_summary['mismatch_value']:.6g}"
                )
            trend = finding.observed_summary
            if "axis_name" in trend and "central_values" in trend:
                lines.append(
                    "  "
                    f"Trend: axis={trend.get('axis_name')} "
                    f"delta={trend.get('endpoint_delta'):.6g} "
                    f"slope={trend.get('theil_sen_slope'):.6g} "
                    f"adjacent_success={trend.get('adjacent_success_rate'):.3f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_expectation_report(
    report: ExpectationReport,
    *,
    output_json: Optional[Path],
    output_markdown: Optional[Path],
) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "output_json": None,
        "output_markdown": None,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out["output_json"] = str(output_json.resolve())
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_expectation_markdown(report), encoding="utf-8")
        out["output_markdown"] = str(output_markdown.resolve())
    return out


def _load_local_law_summary(path: Path) -> Optional[tuple[LocalLawRunSummary, Dict[str, Any]]]:
    try:
        payload = _load_json(path)
    except Exception:
        return None
    return load_or_backfill_local_law_payload(payload, source_path=str(path))


def _structured_split_metrics(
    summary: LocalLawRunSummary,
    policy_name: str,
    split: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    policies = dict(summary.policies)
    policy = policies.get(policy_name)
    if policy is None:
        return {}, {}, {}
    split_metrics = dict(policy.split_metrics).get(split, {})
    if not isinstance(split_metrics, dict):
        return {}, {}, {}
    return split_metric_views(split_metrics)


def _primary_objective_value(
    local: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> float:
    value = _finite(objective.get("full_objective_value", objective.get("value")))
    if value is not None:
        return float(value)
    combined = _finite(local.get("combined"))
    if combined is not None:
        return float(combined)
    return float("nan")


def _scenario_key_from_local_law(
    summary: LocalLawRunSummary,
    payload: Dict[str, Any],
    *,
    ignore_support: bool = True,
) -> str:
    metadata = dict(summary.metadata or {})
    cfg = dict(payload.get("config", {}) or {})
    parts: Dict[str, object] = {
        "family": summary.family,
        "dgp": summary.dgp,
        "suite_role": summary.suite_role,
    }
    for key in (
        "analysis_partition_mode",
        "lambda_multiplier",
        "law_leaf_query_design",
        "law_internal_query_design",
        "n_regimes",
        "fixed_leaf_tokens",
        "feature_mode",
        "model_family",
        "test_docs",
        "val_docs",
    ):
        value = metadata.get(key, cfg.get(key))
        if value is not None and value != "":
            parts[str(key)] = value
    if not ignore_support:
        parts["train_docs"] = int(summary.support_budget.train_docs)
        parts["total_queries_estimate"] = float(summary.support_budget.total_queries_estimate)
    return _slug(parts)


class StructuredLocalLawAdapter:
    family = FAMILY_LOCAL_LAW

    def can_load(self, path: Path) -> bool:
        if not _path_contains_any(
            path,
            (
                "learnability_summary",
                "law_stress_summary",
                "cross_dgp_law_stress",
                "local_law",
            ),
        ):
            return False
        return _load_local_law_summary(path) is not None

    def load_summaries(
        self,
        paths: Sequence[Path],
    ) -> List[tuple[Path, LocalLawRunSummary, Dict[str, Any]]]:
        out: List[tuple[Path, LocalLawRunSummary, Dict[str, Any]]] = []
        for path in paths:
            loaded = _load_local_law_summary(path)
            if loaded is None:
                continue
            summary, payload = loaded
            out.append((path, summary, payload))
        return out

    def build_expectations(
        self,
        loaded: Sequence[tuple[Path, LocalLawRunSummary, Dict[str, Any]]],
        *,
        config: ExpectationConfig,
    ) -> List[ExpectationFinding]:
        findings: List[ExpectationFinding] = []

        for path, summary, _payload in loaded:
            if str(summary.suite_role) != "positive_controls":
                continue
            oracle_policy_name = (
                "oracle_true_summary"
                if "oracle_true_summary" in summary.policies
                else "oracle_g"
                if "oracle_g" in summary.policies
                else str(summary.oracle_name)
            )
            local, _downstream, _objective = _structured_split_metrics(
                summary,
                oracle_policy_name,
                "test",
            )
            thresholds = dict(summary.thresholds or {})
            c1 = float(local.get("c1", float("nan")))
            c2 = float(local.get("c2", float("nan")))
            c3 = float(local.get("c3", float("nan")))
            ok = (
                c1 <= float(thresholds.get("c1", thresholds.get("c1_tau", config.ceiling_epsilon_float)))
                and c2 <= float(thresholds.get("c2", thresholds.get("c2_tau", thresholds.get("c2_proxy", 0.2))))
                and c3 <= float(thresholds.get("c3", thresholds.get("c3_tau", config.ceiling_epsilon_float)))
            )
            findings.append(
                ExpectationFinding(
                    kind="local_law_oracle_ceiling",
                    title=f"{summary.family} oracle_g respects local laws on positive controls",
                    status="pass" if ok else "fail",
                    family=self.family,
                    scenario=_scenario_key_from_local_law(summary, _payload),
                    metric="oracle_local_law",
                    method="oracle_g",
                    direction="decreasing",
                    observed_summary={
                        "note": "oracle_g should be exact or within configured thresholds on positive controls",
                        "c1": c1,
                        "c2": c2,
                        "c3": c3,
                        "source_path": str(path),
                    },
                    thresholds=thresholds,
                    supporting_rows=[{"source_path": str(path), "summary": summary.to_dict()}],
                )
            )

        for path, summary, _payload in loaded:
            if not summary.counterexamples:
                continue
            thresholds = dict(summary.thresholds or {})
            for counterexample in summary.counterexamples:
                test_metrics = dict(counterexample.metrics or {}).get("test", {})
                local = dict(
                    test_metrics.get("local_law", {})
                    or test_metrics.get("local_law_metrics", {})
                    or {}
                )
                law_values = {
                    "C1": float(local.get("c1", float("nan"))),
                    "C2": float(local.get("c2", float("nan"))),
                    "C3": float(local.get("c3", float("nan"))),
                }
                passed = True
                for law in counterexample.targeted_laws:
                    if law == "C1":
                        passed = passed and (law_values["C1"] > float(thresholds.get("c1", thresholds.get("c1_tau", 0.0))))
                    elif law == "C2":
                        passed = passed and (law_values["C2"] > float(thresholds.get("c2", thresholds.get("c2_tau", 0.0))))
                    elif law == "C3":
                        passed = passed and (law_values["C3"] > float(thresholds.get("c3", thresholds.get("c3_tau", 0.0))))
                findings.append(
                    ExpectationFinding(
                        kind="counterexample_breaks_target",
                        title=f"{summary.family} counterexample {counterexample.name} breaks its intended laws",
                        status="pass" if passed else "fail",
                        family=self.family,
                        scenario=_scenario_key_from_local_law(summary, _payload),
                        metric="counterexample_targeted_break",
                        method=str(counterexample.name),
                        direction="increasing",
                        observed_summary={
                            "note": "targeted local-law metric should exceed the configured threshold",
                            "targeted_laws": list(counterexample.targeted_laws),
                            "law_values": law_values,
                            "source_path": str(path),
                        },
                        thresholds=thresholds,
                        supporting_rows=[{"source_path": str(path), "summary": summary.to_dict()}],
                    )
                )

        selection_violations: List[Dict[str, Any]] = []
        for path, summary, _payload in loaded:
            selection = dict(summary.selection or {})
            uses_test = bool(
                selection.get("uses_test_metrics", False)
                or selection.get("test_metrics_used_for_selection", False)
            )
            selected_candidate = str(selection.get("selected_candidate", "") or "").strip()
            if not selected_candidate:
                continue
            selected_role = selected_policy_role(summary)
            requires_validation = selected_role in {PolicyRole.LEARNED_G, PolicyRole.CANDIDATE_G}
            if not requires_validation:
                continue
            if int(summary.support_budget.val_docs) > 0:
                if str(selection.get("selection_split", "")) != "val" or uses_test:
                    selection_violations.append(
                        {
                            "source_path": str(path),
                            "selection": selection,
                            "summary": summary.to_dict(),
                        }
                    )
            elif uses_test:
                selection_violations.append(
                    {
                        "source_path": str(path),
                        "selection": selection,
                        "summary": summary.to_dict(),
                    }
                )
        findings.append(
            ExpectationFinding(
                kind="validation_only_selection",
                title="Selection uses validation only and never test metrics",
                status="pass" if not selection_violations else "fail",
                family=self.family,
                scenario="global",
                metric="selection_protocol",
                method="all_selected_runs",
                direction="flat",
                observed_summary={
                    "note": "selected candidates must be chosen on validation only when val_docs > 0",
                    "n_violations": int(len(selection_violations)),
                },
                thresholds={},
                supporting_rows=selection_violations,
            )
        )

        support_groups: Dict[str, List[tuple[Path, LocalLawRunSummary, Dict[str, Any]]]] = {}
        for item in loaded:
            path, summary, payload = item
            if str(summary.suite_role) != "support_scaling":
                continue
            support_groups.setdefault(
                _scenario_key_from_local_law(summary, payload),
                [],
            ).append(item)
        for scenario, items in sorted(support_groups.items()):
            buckets: Dict[float, Dict[str, float]] = {}
            support_rows: List[Dict[str, Any]] = []
            for path, summary, payload in items:
                if str(summary.family) == "markov_ops_count":
                    support = float(summary.support_budget.train_docs)
                else:
                    support = float(summary.support_budget.total_queries_estimate)
                    if not math.isfinite(support) or support <= 0.0:
                        support = float(summary.support_budget.train_docs)
                bucket = buckets.setdefault(support, {})
                if "infer_identity" in summary.policies:
                    local, _downstream, objective = _structured_split_metrics(
                        summary,
                        "infer_identity",
                        "test",
                    )
                    if local:
                        bucket["baseline"] = _primary_objective_value(local, objective)
                        bucket["baseline_combined"] = float(local.get("combined", float("nan")))
                        bucket["baseline_task"] = float(
                            objective.get("task_objective_value", float("nan"))
                        )
                for name, policy in summary.policies.items():
                    if _role_value(policy.role) == "baseline_g":
                        local, _downstream, objective = _structured_split_metrics(
                            summary,
                            name,
                            "test",
                        )
                        if local:
                            bucket["baseline"] = _primary_objective_value(local, objective)
                            bucket["baseline_combined"] = float(local.get("combined", float("nan")))
                            bucket["baseline_task"] = float(
                                objective.get("task_objective_value", float("nan"))
                            )
                if "learned_g" in summary.policies:
                    local, _downstream, objective = _structured_split_metrics(
                        summary,
                        "learned_g",
                        "test",
                    )
                    if local:
                        current = _primary_objective_value(local, objective)
                        bucket["learned"] = min(current, bucket.get("learned", current))
                        bucket["learned_combined"] = float(local.get("combined", float("nan")))
                        bucket["learned_task"] = float(
                            objective.get("task_objective_value", float("nan"))
                        )
                for name, policy in summary.policies.items():
                    if _role_value(policy.role) != "learned_g":
                        continue
                    local, _downstream, objective = _structured_split_metrics(
                        summary,
                        name,
                        "test",
                    )
                    if not local:
                        continue
                    current = _primary_objective_value(local, objective)
                    bucket["learned"] = min(current, bucket.get("learned", current))
                    bucket["learned_combined"] = float(local.get("combined", float("nan")))
                    bucket["learned_task"] = float(
                        objective.get("task_objective_value", float("nan"))
                    )
                support_rows.append({"source_path": str(path), "summary": summary.to_dict()})
            xs = sorted(buckets.keys())
            gaps = [
                float(buckets[x]["learned"] - buckets[x]["baseline"])
                for x in xs
                if "learned" in buckets[x] and "baseline" in buckets[x]
            ]
            combined_gaps = [
                float(buckets[x]["learned_combined"] - buckets[x]["baseline_combined"])
                for x in xs
                if "learned_combined" in buckets[x] and "baseline_combined" in buckets[x]
            ]
            task_gaps = [
                float(buckets[x]["learned_task"] - buckets[x]["baseline_task"])
                for x in xs
                if "learned_task" in buckets[x] and "baseline_task" in buckets[x]
            ]
            if len(gaps) < 1:
                findings.append(
                    ExpectationFinding(
                        kind="support_scaling_improves_gap",
                        title=f"{scenario}: learned_g improves over baseline_g with more support",
                        status="not_applicable",
                        family=self.family,
                        scenario=scenario,
                        metric="learned_minus_baseline_objective",
                        method="learned_g vs baseline_g",
                        direction="decreasing",
                        observed_summary={"note": "missing paired learned/baseline support points"},
                        thresholds={},
                        supporting_rows=support_rows,
                    )
                )
                continue
            if len(gaps) < 2:
                findings.append(
                    ExpectationFinding(
                        kind="support_scaling_improves_gap",
                        title=f"{scenario}: learned_g improves over baseline_g with more support",
                        status="not_applicable",
                        family=self.family,
                        scenario=scenario,
                        metric="learned_minus_baseline_objective",
                        method="learned_g vs baseline_g",
                        direction="decreasing",
                        observed_summary={
                            "note": "need at least two paired support points to evaluate a support-scaling trend",
                            "support_values": xs,
                            "gap_values": gaps,
                            "combined_gap_values": combined_gaps,
                            "task_gap_values": task_gaps,
                        },
                        thresholds={},
                        supporting_rows=support_rows,
                    )
                )
                continue
            start_gap = float(gaps[0])
            end_gap = float(gaps[-1])
            scale = _metric_scale(gaps + [start_gap, end_gap])
            min_effect_abs = float(config.min_effect_abs_scale) * float(scale)
            if end_gap <= min(-min_effect_abs, start_gap):
                status = "pass"
                note = "best learned_g materially beats the matched baseline at high support"
            elif end_gap < 0.0:
                status = "warn"
                note = "learned_g beats baseline, but the support trend is modest"
            else:
                status = "fail"
                note = "learned_g does not beat the matched baseline at high support"
            findings.append(
                ExpectationFinding(
                    kind="support_scaling_improves_gap",
                    title=f"{scenario}: learned_g improves over baseline_g with more support",
                    status=status,
                    family=self.family,
                    scenario=scenario,
                    metric="learned_minus_baseline_objective",
                    method="learned_g vs baseline_g",
                    direction="decreasing",
                    observed_summary={
                        "note": note,
                        "support_values": xs,
                        "gap_values": gaps,
                        "combined_gap_values": combined_gaps,
                        "task_gap_values": task_gaps,
                        "start_gap": start_gap,
                        "end_gap": end_gap,
                    },
                    thresholds={"min_effect_abs": min_effect_abs},
                    supporting_rows=support_rows,
                )
            )

        lda_null_rows: List[Dict[str, Any]] = []
        pooled_deltas: List[float] = []
        null_law_gaps: List[float] = []
        null_objective_gaps: List[float] = []
        primary_gains: List[float] = []
        for path, summary, payload in loaded:
            if str(summary.family) != "tree_relevant_lda_local_law":
                continue
            metadata = dict(summary.metadata or {})
            lam = _finite(metadata.get("lambda_multiplier", dict(payload.get("config", {}) or {}).get("lambda_multiplier")))
            if lam is None or abs(float(lam)) > 1e-12:
                continue
            base_local, base_downstream, base_objective = _structured_split_metrics(
                summary,
                "infer_identity",
                "test",
            )
            learned_local, learned_downstream, learned_objective = _structured_split_metrics(
                summary,
                "learned_g",
                "test",
            )
            if not base_downstream:
                continue
            pooled_deltas.append(abs(float(base_downstream.get("oracle_target_delta", float("nan")))))
            if learned_downstream:
                pooled_deltas.append(abs(float(learned_downstream.get("oracle_target_delta", float("nan")))))
            base_primary = _finite(base_downstream.get("oracle_target_abs_error"))
            learned_primary = _finite(learned_downstream.get("oracle_target_abs_error"))
            if (
                base_primary is not None
                and learned_primary is not None
            ):
                if float(base_primary) <= 0.0:
                    primary_gains.append(0.0 if float(learned_primary) <= 0.0 else float("inf"))
                else:
                    primary_gains.append(
                        (float(base_primary) - float(learned_primary)) / float(base_primary)
                    )
            if base_local and learned_local:
                null_law_gaps.append(
                    abs(
                        float(learned_local.get("combined", float("nan")))
                        - float(base_local.get("combined", float("nan")))
                    )
                )
                objective_gap = abs(
                    _primary_objective_value(learned_local, learned_objective)
                    - _primary_objective_value(base_local, base_objective)
                )
                if math.isfinite(objective_gap):
                    null_objective_gaps.append(float(objective_gap))
            lda_null_rows.append({"source_path": str(path), "summary": summary.to_dict()})
        if lda_null_rows:
            max_abs_pooled_delta = max(
                [abs(x) for x in pooled_deltas if math.isfinite(float(x))],
                default=float("nan"),
            )
            max_law_gap = max(
                [abs(x) for x in null_law_gaps if math.isfinite(float(x))],
                default=float("nan"),
            )
            gain_tol = float(config.min_effect_rel)
            median_abs_gain = _safe_percentile((abs(x) for x in primary_gains), 50.0)
            p90_abs_gain = _safe_percentile((abs(x) for x in primary_gains), 90.0)
            mean_gain = _safe_mean(primary_gains)
            if (
                math.isfinite(median_abs_gain)
                and math.isfinite(p90_abs_gain)
                and median_abs_gain <= gain_tol
                and p90_abs_gain <= max(0.25, 2.5 * gain_tol)
            ):
                status = "pass"
                note = "lambda=0 keeps learned-vs-baseline primary gains modest; pooled-relative Delta remains a diagnostic"
            elif math.isfinite(median_abs_gain) and median_abs_gain <= 2.0 * gain_tol:
                status = "warn"
                note = "lambda=0 primary gains are not dominant, but some regimes still show noticeable benefit"
            else:
                status = "fail"
                note = "lambda=0 still shows material learned-vs-baseline primary gains"
            findings.append(
                ExpectationFinding(
                    kind="lambda_zero_null_control",
                    title="Tree-relevant LDA lambda=0 suppresses learned-vs-baseline relevance gains",
                    status=status,
                    family=self.family,
                    scenario="tree_relevant_lda_local_law|lambda=0",
                    metric="primary_gain_frac",
                    method="learned_g vs baseline_g",
                    direction="flat",
                    observed_summary={
                        "note": note,
                        "n_learned_runs": len(primary_gains),
                        "mean_primary_gain_frac": mean_gain,
                        "median_abs_primary_gain_frac": median_abs_gain,
                        "p90_abs_primary_gain_frac": p90_abs_gain,
                        "max_abs_pooled_delta": max_abs_pooled_delta,
                        "mean_objective_gap": _safe_mean(null_objective_gaps),
                        "max_law_gap": max_law_gap,
                    },
                    thresholds={"primary_gain_tolerance": gain_tol},
                    supporting_rows=lda_null_rows,
                )
            )

        findings.sort(key=lambda x: (STATUS_PRIORITY.get(x.status, 99), x.title))
        return findings


def build_local_law_expectation_report(
    *,
    output_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    config: Optional[ExpectationConfig] = None,
) -> ExpectationReport:
    cfg = config or ExpectationConfig()
    all_paths = _collect_paths(output_root, manifest_path)
    adapter = StructuredLocalLawAdapter()
    loaded = adapter.load_summaries(all_paths)
    expectations = adapter.build_expectations(loaded, config=cfg)
    expectations.sort(key=lambda x: (STATUS_PRIORITY.get(x.status, 99), x.family, x.title))
    summary = {
        **_expectation_summary(expectations),
    }
    return ExpectationReport(
        input_root=str(output_root.resolve()) if output_root is not None else None,
        manifest=str(manifest_path.resolve()) if manifest_path is not None else None,
        families_scanned=[adapter.family] if loaded else [],
        rows_scanned=int(len(loaded)),
        expectations=expectations,
        summary=summary,
    )


__all__ = [
    "BudgetTrendExpectation",
    "CeilingExpectation",
    "ExpectationConfig",
    "ExpectationFinding",
    "ExpectationReport",
    "ExpectationStatus",
    "FAMILY_LOCAL_LAW",
    "FailureModeExpectation",
    "FamilyAdapter",
    "GranularitySensitivityExpectation",
    "MarkovOPSAdapter",
    "MergeableAblationAdapter",
    "NormalizedRow",
    "SegmentLDAOPSAdapter",
    "SegmentedLDACtreePOAdapter",
    "StructuredLocalLawAdapter",
    "TrendAssessment",
    "TrendDirection",
    "VALID_FAMILIES",
    "assess_trend",
    "build_expectation_report",
    "build_local_law_expectation_report",
    "merge_expectation_reports",
    "render_expectation_markdown",
    "write_expectation_report",
]
