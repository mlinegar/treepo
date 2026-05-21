from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

import numpy as np

from treepo._research.ctreepo.sim.util import safe_float


class PolicyRole(str, Enum):
    ORACLE_G = "oracle_g"
    BASELINE_G = "baseline_g"
    LEARNED_G = "learned_g"
    CANDIDATE_G = "candidate_g"
    COUNTEREXAMPLE_G = "counterexample_g"


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_serialize(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class GArtifact:
    artifact_id: str
    name: str
    role: PolicyRole
    family: str
    dgp: str
    fmt: str
    manifest_path: str
    sidecar_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = _serialize(asdict(self))
        family = str(payload.pop("family", "") or "")
        dgp = str(payload.pop("dgp", "") or "")
        payload["problem_id"] = dgp
        payload["method_id"] = family or str(payload.get("name", ""))
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GArtifact":
        return cls(
            artifact_id=str(payload.get("artifact_id", "")),
            name=str(payload.get("name", "")),
            role=PolicyRole(str(payload.get("role", PolicyRole.CANDIDATE_G.value))),
            family=str(payload.get("method_id", payload.get("family", ""))),
            dgp=str(payload.get("problem_id", payload.get("dgp", ""))),
            fmt=str(payload.get("fmt", "")),
            manifest_path=str(payload.get("manifest_path", "")),
            sidecar_paths={
                str(k): str(v)
                for k, v in dict(payload.get("sidecar_paths", {}) or {}).items()
            },
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class LocalLawMetrics:
    c1: float
    c2: float
    c3: float
    combined: float
    root_error: float = float("nan")
    schedule_spread: float = float("nan")
    c1_violation_rate: float = float("nan")
    c2_violation_rate: float = float("nan")
    c3_violation_rate: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocalLawMetrics":
        return cls(
            c1=float(payload.get("c1", float("nan"))),
            c2=float(payload.get("c2", float("nan"))),
            c3=float(payload.get("c3", float("nan"))),
            combined=float(payload.get("combined", float("nan"))),
            root_error=float(payload.get("root_error", float("nan"))),
            schedule_spread=float(payload.get("schedule_spread", float("nan"))),
            c1_violation_rate=float(payload.get("c1_violation_rate", float("nan"))),
            c2_violation_rate=float(payload.get("c2_violation_rate", float("nan"))),
            c3_violation_rate=float(payload.get("c3_violation_rate", float("nan"))),
        )


@dataclass(frozen=True)
class DownstreamMetrics:
    oracle_target_abs_error: float = float("nan")
    oracle_target_delta: float = float("nan")
    root_error: float = float("nan")
    schedule_spread: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DownstreamMetrics":
        return cls(
            oracle_target_abs_error=float(
                payload.get("oracle_target_abs_error", float("nan"))
            ),
            oracle_target_delta=float(payload.get("oracle_target_delta", float("nan"))),
            root_error=float(payload.get("root_error", float("nan"))),
            schedule_spread=float(payload.get("schedule_spread", float("nan"))),
        )


@dataclass(frozen=True)
class SupportBudgetSummary:
    train_docs: int
    val_docs: int
    test_docs: int
    leaf_query_rate: float = 0.0
    internal_query_rate: float = 0.0
    root_query_rate: float = 0.0
    mean_leaf_labels_per_doc: float = 0.0
    mean_internal_labels_per_doc: float = 0.0
    mean_queries_per_doc: float = 0.0
    total_queries_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupportBudgetSummary":
        return cls(
            train_docs=int(payload.get("train_docs", 0)),
            val_docs=int(payload.get("val_docs", 0)),
            test_docs=int(payload.get("test_docs", 0)),
            leaf_query_rate=float(payload.get("leaf_query_rate", 0.0)),
            internal_query_rate=float(payload.get("internal_query_rate", 0.0)),
            root_query_rate=float(payload.get("root_query_rate", 0.0)),
            mean_leaf_labels_per_doc=float(
                payload.get("mean_leaf_labels_per_doc", 0.0)
            ),
            mean_internal_labels_per_doc=float(
                payload.get("mean_internal_labels_per_doc", 0.0)
            ),
            mean_queries_per_doc=float(payload.get("mean_queries_per_doc", 0.0)),
            total_queries_estimate=float(payload.get("total_queries_estimate", 0.0)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class LocalLawPolicyEvaluation:
    name: str
    role: PolicyRole
    artifact_id: Optional[str] = None
    selection_metric_value: float = float("nan")
    split_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocalLawPolicyEvaluation":
        return cls(
            name=str(payload.get("name", "")),
            role=PolicyRole(str(payload.get("role", PolicyRole.CANDIDATE_G.value))),
            artifact_id=(
                str(payload["artifact_id"])
                if payload.get("artifact_id") is not None
                else None
            ),
            selection_metric_value=float(
                payload.get("selection_metric_value", float("nan"))
            ),
            split_metrics=dict(payload.get("split_metrics", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class LocalLawCounterexampleEvaluation:
    name: str
    role: PolicyRole
    targeted_laws: Sequence[str]
    metrics: Dict[str, Any]
    artifact_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "LocalLawCounterexampleEvaluation":
        return cls(
            name=str(payload.get("name", "")),
            role=PolicyRole(str(payload.get("role", PolicyRole.COUNTEREXAMPLE_G.value))),
            targeted_laws=[str(x) for x in (payload.get("targeted_laws", []) or [])],
            metrics=dict(payload.get("metrics", {}) or {}),
            artifact_id=(
                str(payload["artifact_id"])
                if payload.get("artifact_id") is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class LocalLawRunSummary:
    family: str
    dgp: str
    oracle_name: str
    study_role: str
    split_ids: Dict[str, str]
    support_budget: SupportBudgetSummary
    selection: Dict[str, Any]
    policies: Dict[str, LocalLawPolicyEvaluation]
    counterexamples: Sequence[LocalLawCounterexampleEvaluation]
    thresholds: Dict[str, Any]
    suite_role: str = ""
    compositional_learning_problem: Dict[str, Any] = field(default_factory=dict)
    logged_observation_artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": str(self.dgp),
            "method_id": str(self.family),
            "oracle_name": str(self.oracle_name),
            "study_role": str(self.study_role),
            "split_ids": dict(self.split_ids),
            "support_budget": self.support_budget.to_dict(),
            "selection": _serialize(self.selection),
            "policies": {
                str(k): v.to_dict() for k, v in dict(self.policies).items()
            },
            "counterexamples": [x.to_dict() for x in self.counterexamples],
            "thresholds": _serialize(self.thresholds),
            "suite_role": str(self.suite_role),
            "compositional_learning_problem": _serialize(
                self.compositional_learning_problem
            ),
            "logged_observation_artifacts": _serialize(self.logged_observation_artifacts),
            "metadata": _serialize(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocalLawRunSummary":
        return cls(
            family=str(payload.get("method_id", payload.get("family", ""))),
            dgp=str(payload.get("problem_id", payload.get("dgp", ""))),
            oracle_name=str(payload.get("oracle_name", "")),
            study_role=str(payload.get("study_role", "")),
            split_ids={
                str(k): str(v) for k, v in dict(payload.get("split_ids", {}) or {}).items()
            },
            support_budget=SupportBudgetSummary.from_dict(
                dict(payload.get("support_budget", {}) or {})
            ),
            selection=dict(payload.get("selection", {}) or {}),
            policies={
                str(k): LocalLawPolicyEvaluation.from_dict(v)
                for k, v in dict(payload.get("policies", {}) or {}).items()
            },
            counterexamples=[
                LocalLawCounterexampleEvaluation.from_dict(x)
                for x in (payload.get("counterexamples", []) or [])
            ],
            thresholds=dict(payload.get("thresholds", {}) or {}),
            suite_role=str(payload.get("suite_role", "")),
            compositional_learning_problem=dict(
                payload.get("compositional_learning_problem", {}) or {}
            ),
            logged_observation_artifacts=dict(
                payload.get("logged_observation_artifacts", {}) or {}
            ),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


class LocalLawFamilyAdapter(Protocol):
    family: str

    def build_local_law_summary(self, payload: Mapping[str, Any]) -> LocalLawRunSummary:
        ...


def selected_policy_role(summary: LocalLawRunSummary) -> Optional[PolicyRole]:
    selected = str(dict(summary.selection or {}).get("selected_candidate", "") or "").strip()
    if not selected:
        return None
    for key, policy in dict(summary.policies).items():
        if str(key) == selected or str(policy.name) == selected:
            if isinstance(policy.role, PolicyRole):
                return policy.role
            return PolicyRole(str(policy.role))
    return None


_safe_float = safe_float


def split_metric_views(split_metrics: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    payload = dict(split_metrics or {})
    local = dict(payload.get("local_law", {}) or payload.get("local_law_metrics", {}) or {})
    downstream = dict(payload.get("downstream", {}) or payload.get("downstream_metrics", {}) or {})
    objective = dict(payload.get("objective", {}) or payload.get("objective_metrics", {}) or {})
    return local, downstream, objective


def primary_objective_value(
    split_metrics: Mapping[str, Any],
    *,
    default: float = float("nan"),
) -> float:
    local, _downstream, objective = split_metric_views(split_metrics)
    selection_metric_name = str(objective.get("selection_metric_name", "") or "").strip()
    if selection_metric_name:
        out = _safe_float(objective.get(selection_metric_name), float("nan"))
        if np.isfinite(out):
            return float(out)
    selection_metric_value = _safe_float(objective.get("selection_metric_value"), float("nan"))
    if np.isfinite(selection_metric_value):
        return float(selection_metric_value)
    for key in ("full_objective_value", "value"):
        out = _safe_float(objective.get(key), float("nan"))
        if np.isfinite(out):
            return float(out)
    combined = _safe_float(local.get("combined"), float("nan"))
    if np.isfinite(combined):
        return float(combined)
    return float(default)


def artifact_index(artifacts: Sequence[GArtifact]) -> Dict[str, Dict[str, Any]]:
    return {str(a.artifact_id): a.to_dict() for a in artifacts}


def write_json_g_artifact(
    *,
    output_dir: Path,
    artifact_id: str,
    name: str,
    role: PolicyRole,
    family: str,
    dgp: str,
    payload: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> GArtifact:
    manifest_path = output_dir / f"{artifact_id}.json"
    _write_json(
        manifest_path,
        {
            "artifact_id": str(artifact_id),
            "name": str(name),
            "role": role.value,
            "problem_id": str(dgp),
            "method_id": str(family),
            "fmt": "json",
            "payload": dict(payload),
            "metadata": dict(metadata or {}),
        },
    )
    return GArtifact(
        artifact_id=str(artifact_id),
        name=str(name),
        role=role,
        family=str(family),
        dgp=str(dgp),
        fmt="json",
        manifest_path=str(manifest_path),
        metadata=dict(metadata or {}),
    )


def write_npz_g_artifact(
    *,
    output_dir: Path,
    artifact_id: str,
    name: str,
    role: PolicyRole,
    family: str,
    dgp: str,
    manifest_payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Optional[Mapping[str, Any]] = None,
) -> GArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / f"{artifact_id}.npz"
    np.savez(sidecar_path, **{str(k): np.asarray(v) for k, v in arrays.items()})
    manifest_path = output_dir / f"{artifact_id}.json"
    _write_json(
        manifest_path,
        {
            "artifact_id": str(artifact_id),
            "name": str(name),
            "role": role.value,
            "problem_id": str(dgp),
            "method_id": str(family),
            "fmt": "json+npz",
            "payload": dict(manifest_payload),
            "sidecars": {"weights": str(sidecar_path)},
            "metadata": dict(metadata or {}),
        },
    )
    return GArtifact(
        artifact_id=str(artifact_id),
        name=str(name),
        role=role,
        family=str(family),
        dgp=str(dgp),
        fmt="json+npz",
        manifest_path=str(manifest_path),
        sidecar_paths={"weights": str(sidecar_path)},
        metadata=dict(metadata or {}),
    )


__all__ = [
    "DownstreamMetrics",
    "GArtifact",
    "LocalLawCounterexampleEvaluation",
    "LocalLawFamilyAdapter",
    "LocalLawMetrics",
    "LocalLawPolicyEvaluation",
    "LocalLawRunSummary",
    "PolicyRole",
    "SupportBudgetSummary",
    "artifact_index",
    "primary_objective_value",
    "split_metric_views",
    "write_json_g_artifact",
    "write_npz_g_artifact",
]
