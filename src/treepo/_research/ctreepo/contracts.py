"""Stable public contracts for the C-TreePO API layer.

These dataclasses are intentionally small and dependency-light.  Backend
families can keep their native configs and artifact formats internally, while
the public `src.ctreepo.learning` and `src.ctreepo.runtime` surfaces exchange
these records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

JsonDict = dict[str, Any]

TREE_BUNDLE_SCHEMA_VERSION = "ctreepo.tree_bundle.v1"
RUN_MANIFEST_SCHEMA_VERSION = "ctreepo.run_manifest.v1"
OBJECTIVE_SCHEMA_VERSION = "ctreepo.objective.v1"
RUN_AXIS_SCHEMA_VERSION = "ctreepo.run_axis.v1"
ORACLE_OBSERVATION_SCHEMA_VERSION = "ctreepo.oracle_observation.v1"
TREE_REPRESENTATION_PARTITION = "partition_tree"
REDUCER_CONTRACT_BOTTOM_UP = "bottom_up"

LEAF_UNIT_TEXT_TOKEN = "text_token"
LEAF_UNIT_STREAM_ITEM = "stream_item"
LEAF_UNIT_SYNTHETIC_ATOM = "synthetic_atom"
LEAF_UNIT_EMBEDDING_ROW = "embedding_row"

SOURCE_KIND_RAW_INPUT = "raw_input"
SOURCE_KIND_EXTERNAL_STATE = "external_state"
SOURCE_KIND_SYNTHETIC_ORACLE = "synthetic_oracle"
SOURCE_KIND_DERIVED_CACHE = "derived_cache"

STATE_CONTRACT_RAW_CONCAT = "raw_concat"
STATE_CONTRACT_BOTTOM_UP_G = "bottom_up_g"
STATE_CONTRACT_EXTERNAL_PASSTHROUGH = "external_passthrough"
STATE_CONTRACT_ORACLE_STATE = "oracle_state"

LEGACY_RAW_MANIFESTO_TOKEN_TREE = "raw_manifesto_token_tree"
LEGACY_EXTERNAL_SUMMARY_TOKEN_TREE = "external_summary_token_tree"

_ALLOWED_LEAF_UNITS = {
    LEAF_UNIT_TEXT_TOKEN,
    LEAF_UNIT_STREAM_ITEM,
    LEAF_UNIT_SYNTHETIC_ATOM,
    LEAF_UNIT_EMBEDDING_ROW,
}
_ALLOWED_SOURCE_KINDS = {
    SOURCE_KIND_RAW_INPUT,
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_SYNTHETIC_ORACLE,
    SOURCE_KIND_DERIVED_CACHE,
}
_ALLOWED_STATE_CONTRACTS = {
    STATE_CONTRACT_RAW_CONCAT,
    STATE_CONTRACT_BOTTOM_UP_G,
    STATE_CONTRACT_EXTERNAL_PASSTHROUGH,
    STATE_CONTRACT_ORACLE_STATE,
}
_ALLOWED_REDUCER_CONTRACTS = {REDUCER_CONTRACT_BOTTOM_UP}
_RUN_STATUSES = {
    "planned",
    "pending",
    "running",
    "completed",
    "partial",
    "failed",
    "skipped",
    "legacy",
    "quarantined",
    "unknown",
}
_BAD_QUARANTINE_CLASSES = {
    "missing_contract",
    "invalid_state_dim",
    "root_summary_shortcut_risk",
    "unknown",
}
_LEGACY_OBJECTIVE_TERM_ORACLE_GAP = "oracle_gap"
OBJECTIVE_TERM_ROOT = "root"
OBJECTIVE_TERM_LOCAL_LAW_CORRECTED = "local_law_corrected"

LAW_ID_LEAF_PRESERVATION = "leaf_preservation"
LAW_ID_MERGE_PRESERVATION = "merge_preservation"
LAW_ID_ON_RANGE_IDEMPOTENCE = "on_range_idempotence"
LAW_SET_ALL = "all"
LAW_SET_ROOT_ONLY = "root_only"
LAW_SET_LEAF_PRESERVATION_ONLY = "leaf_preservation_only"
LAW_SET_MERGE_PRESERVATION_ONLY = "merge_preservation_only"
LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY = "on_range_idempotence_only"
LAW_SET_LEAF_AND_MERGE_PRESERVATION = "leaf_and_merge_preservation"
LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE = "merge_and_on_range_idempotence"

RUN_ROLE_PRIMARY = "primary"
RUN_ROLE_REFERENCE = "reference"
RUN_ROLE_AUXILIARY = "auxiliary"

ORACLE_OBSERVATION_DESIGN_ROOT_ONLY = "root_only"
ORACLE_OBSERVATION_DESIGN_SAMPLED_NODES = "sampled_nodes"
ORACLE_OBSERVATION_DESIGN_SAMPLED_ROOT_NODES = "sampled_root_nodes"
ORACLE_OBSERVATION_DESIGN_DENSE_ORACLE = "dense_oracle"
ORACLE_OBSERVATION_DESIGN_BUDGETED_MASS = "budgeted_mass"

_ALLOWED_ORACLE_OBSERVATION_DESIGNS = {
    ORACLE_OBSERVATION_DESIGN_ROOT_ONLY,
    ORACLE_OBSERVATION_DESIGN_SAMPLED_NODES,
    ORACLE_OBSERVATION_DESIGN_SAMPLED_ROOT_NODES,
    ORACLE_OBSERVATION_DESIGN_DENSE_ORACLE,
    ORACLE_OBSERVATION_DESIGN_BUDGETED_MASS,
}

LOCAL_LAW_ESTIMATOR_NONE = "none"
LOCAL_LAW_ESTIMATOR_PROXY_ONLY = "proxy_only"
LOCAL_LAW_ESTIMATOR_CORRECTED = "corrected"
LOCAL_LAW_ESTIMATOR_ORACLE_STATE = "oracle_state"
LOCAL_LAW_ESTIMATOR_ORACLE_EXACT = "oracle_exact"
LOCAL_LAW_ESTIMATOR_EXTERNAL_PASSTHROUGH = "external_passthrough"

_ALLOWED_LOCAL_LAW_ESTIMATORS = {
    LOCAL_LAW_ESTIMATOR_NONE,
    LOCAL_LAW_ESTIMATOR_PROXY_ONLY,
    LOCAL_LAW_ESTIMATOR_CORRECTED,
    LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
    LOCAL_LAW_ESTIMATOR_ORACLE_EXACT,
    LOCAL_LAW_ESTIMATOR_EXTERNAL_PASSTHROUGH,
}

_LEGACY_GAP_OBJECTIVE_ALIASES = {
    _LEGACY_OBJECTIVE_TERM_ORACLE_GAP,
    "calibration",
    "score_calibration",
    "oracle_recovery",
    "oracle_recovery_loss",
    "oraclerecovery",
    "oracle_gap_loss",
    "proxy_oracle_gap",
    "proxy_gap",
    "gap",
}

_LEGACY_OBJECTIVE_TERM_ALIASES = {
    "task": OBJECTIVE_TERM_ROOT,
    "task_objective": OBJECTIVE_TERM_ROOT,
    "gold": OBJECTIVE_TERM_ROOT,
    "gold_standard": OBJECTIVE_TERM_ROOT,
    "root_loss": OBJECTIVE_TERM_ROOT,
    "local_law": OBJECTIVE_TERM_LOCAL_LAW_CORRECTED,
    "local_law_loss": OBJECTIVE_TERM_LOCAL_LAW_CORRECTED,
    "corrected_local_law": OBJECTIVE_TERM_LOCAL_LAW_CORRECTED,
    "law_corrected": OBJECTIVE_TERM_LOCAL_LAW_CORRECTED,
}

_LEGACY_OBJECTIVE_EFFECTIVE_WEIGHT_FIELDS = {
    "lambda_" + "eff",
    "lambda_" + "effective",
}
_LEGACY_OBJECTIVE_TRUST_FIELDS = {
    "relia" + "bility",
}

LEGACY_OBJECTIVE_PUBLIC_FIELDS = frozenset(
    {
        "bias_calibration",
        "bias_calibration_mode",
        "bias_excess",
        "bias_floor",
        "bias_gap",
        "c1_weight",
        "c2_weight",
        "c3_weight",
        "gap_weight",
        "gold_standard_lambda",
        "lambda",
        "lambda_local",
        "lambda_local_law",
        "lambda_nominal",
        "law_c1_weight",
        "law_c2_proxy_weight",
        "law_c2_weight",
        "law_c3_weight",
        "law_package",
        "law_package_name",
        "law_package_names",
        "law_task_objective_weight",
        "leaf_weight",
        "local_law_c1_weight",
        "local_law_c2_proxy_weight",
        "local_law_c2_weight",
        "local_law_c3_weight",
        "local_law_leaf_weight",
        "local_law_merge_weight",
        "local_law_weights",
        "oracle_gap_weight",
        "oracleGapWeight",
        "objective_local_law_c1_weight",
        "objective_local_law_c2_proxy_weight",
        "objective_local_law_c2_weight",
        "objective_local_law_c3_weight",
        "proxy_weights",
        "root_weight",
        "selected_lambda_local",
        "signal_scale",
        "task_weight",
        "task_objective_weight",
        "teacher_node_lambda",
        "tree_local_law_weight",
        "tree_task_objective_weight",
    }
    | _LEGACY_OBJECTIVE_EFFECTIVE_WEIGHT_FIELDS
    | _LEGACY_OBJECTIVE_TRUST_FIELDS
)

LEGACY_RUN_AXIS_PUBLIC_FIELDS = frozenset(
    {
        "baseline_family",
        "family",
        "families",
        "tree_families",
        "fno_families",
        "full_doc_anchor_families",
        "full_doc_anchor_family",
        "full_doc_anchor_mode",
        "full_doc_anchor_target",
        "law_package",
        "law_package_name",
        "law_package_names",
        "package",
        "supervision_recovery_tree_family",
        "oracle_budget_tree_families",
        "oracle_budget_reference_families",
    }
)

LEGACY_ORACLE_OBSERVATION_PUBLIC_FIELDS = frozenset(
    {
        "oracle_observation_mode",
    }
)

PUBLIC_CONTRACT_LEGACY_FIELDS = frozenset(
    set(LEGACY_OBJECTIVE_PUBLIC_FIELDS)
    | set(LEGACY_RUN_AXIS_PUBLIC_FIELDS)
    | set(LEGACY_ORACLE_OBSERVATION_PUBLIC_FIELDS)
)

PUBLIC_CONTRACT_LEGACY_FIELD_PREFIX_SUFFIXES = (
    ("tree_", "_weight"),
    ("law_c", "_weight"),
    ("local_law_c", "_weight"),
    ("objective_local_law_c", "_weight"),
)

ORACLE_OBSERVATION_DESIGN_PARAMETER_FIELDS = frozenset(
    {
        "sampled_node_rate",
        "root_label_share",
        "mass_target_per_doc",
        "local_label_pool",
        "local_label_allocation",
    }
)

LAW_ID_ALIASES = {
    "l1": LAW_ID_LEAF_PRESERVATION,
    "l1_leaf": LAW_ID_LEAF_PRESERVATION,
    "c1": LAW_ID_LEAF_PRESERVATION,
    "leaf": LAW_ID_LEAF_PRESERVATION,
    "leaf_loss": LAW_ID_LEAF_PRESERVATION,
    "sufficiency": LAW_ID_LEAF_PRESERVATION,
    "l2": LAW_ID_MERGE_PRESERVATION,
    "l2_merge": LAW_ID_MERGE_PRESERVATION,
    "c3": LAW_ID_MERGE_PRESERVATION,
    "merge": LAW_ID_MERGE_PRESERVATION,
    "merge_loss": LAW_ID_MERGE_PRESERVATION,
    "merge_consistency": LAW_ID_MERGE_PRESERVATION,
    "l3": LAW_ID_ON_RANGE_IDEMPOTENCE,
    "l3_idempotence": LAW_ID_ON_RANGE_IDEMPOTENCE,
    "c2": LAW_ID_ON_RANGE_IDEMPOTENCE,
    "c2_proxy": LAW_ID_ON_RANGE_IDEMPOTENCE,
    "idempotence": LAW_ID_ON_RANGE_IDEMPOTENCE,
    "idempotence_loss": LAW_ID_ON_RANGE_IDEMPOTENCE,
}

CANONICAL_LAW_IDS = frozenset(
    {
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_MERGE_PRESERVATION,
        LAW_ID_ON_RANGE_IDEMPOTENCE,
    }
)
CANONICAL_LAW_ID_ORDER = (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
)

CANONICAL_LAW_SET_IDS = frozenset(
    {
        LAW_SET_ALL,
        LAW_SET_ROOT_ONLY,
        LAW_SET_LEAF_PRESERVATION_ONLY,
        LAW_SET_LEAF_AND_MERGE_PRESERVATION,
        LAW_SET_MERGE_PRESERVATION_ONLY,
        LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
        LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE,
    }
)

LAW_SET_ID_ALIASES = {
    "tree_all_laws": LAW_SET_ALL,
    "all_laws": LAW_SET_ALL,
    "tree_root_only": LAW_SET_ROOT_ONLY,
    "root": LAW_SET_ROOT_ONLY,
    "root_only": LAW_SET_ROOT_ONLY,
    "tree_c1_only": LAW_SET_LEAF_PRESERVATION_ONLY,
    "c1_only": LAW_SET_LEAF_PRESERVATION_ONLY,
    "leaf_only": LAW_SET_LEAF_PRESERVATION_ONLY,
    "tree_c2_only": LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    "c2_only": LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    "idempotence_only": LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    "tree_c3_only": LAW_SET_MERGE_PRESERVATION_ONLY,
    "c3_only": LAW_SET_MERGE_PRESERVATION_ONLY,
    "merge_only": LAW_SET_MERGE_PRESERVATION_ONLY,
    "c1c3": LAW_SET_LEAF_AND_MERGE_PRESERVATION,
    "tree_c1c3": LAW_SET_LEAF_AND_MERGE_PRESERVATION,
    "leaf_and_merge": LAW_SET_LEAF_AND_MERGE_PRESERVATION,
    "tree_c2c3": LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE,
    "c2c3": LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE,
    "c2_c3": LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE,
}

METHOD_ID_LAW_SET_ALIASES = {
    "tree_neural_c2": ("tree_neural", LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY),
    "tree_neural_c2c3": ("tree_neural", LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE),
}


def jsonable(value: Any) -> Any:
    """Convert common Python objects into deterministic JSON-friendly values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ArtifactRef):
        return value.to_dict()
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return [jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _clean_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _legacy_source_kind(payload: Mapping[str, Any]) -> Optional[str]:
    kind = str(payload.get("tree_bundle_kind") or "").strip().lower()
    if kind == LEGACY_RAW_MANIFESTO_TOKEN_TREE:
        return SOURCE_KIND_RAW_INPUT
    if kind == LEGACY_EXTERNAL_SUMMARY_TOKEN_TREE:
        return SOURCE_KIND_EXTERNAL_STATE
    text_source = str(payload.get("tree_text_source") or "").strip().lower()
    if text_source == "aligned_text":
        return SOURCE_KIND_RAW_INPUT
    if text_source == "existing_summary":
        return SOURCE_KIND_EXTERNAL_STATE
    return None


def legacy_tree_bundle_kind_for_source_kind(source_kind: str) -> str:
    """Return the deprecated bundle-kind alias for Manifesto compatibility."""

    normalized = str(source_kind or "").strip().lower()
    if normalized == SOURCE_KIND_RAW_INPUT:
        return LEGACY_RAW_MANIFESTO_TOKEN_TREE
    if normalized == SOURCE_KIND_EXTERNAL_STATE:
        return LEGACY_EXTERNAL_SUMMARY_TOKEN_TREE
    return ""


def legacy_tree_text_source_for_source_kind(source_kind: str) -> str:
    """Return the deprecated text-source alias for Manifesto compatibility."""

    normalized = str(source_kind or "").strip().lower()
    if normalized == SOURCE_KIND_RAW_INPUT:
        return "aligned_text"
    if normalized == SOURCE_KIND_EXTERNAL_STATE:
        return "existing_summary"
    return ""


def default_state_contract_for_source_kind(source_kind: str) -> str:
    normalized = str(source_kind or "").strip().lower()
    if normalized == SOURCE_KIND_EXTERNAL_STATE:
        return STATE_CONTRACT_EXTERNAL_PASSTHROUGH
    if normalized == SOURCE_KIND_SYNTHETIC_ORACLE:
        return STATE_CONTRACT_ORACLE_STATE
    return STATE_CONTRACT_RAW_CONCAT


@dataclass(frozen=True)
class TreeBundleManifest:
    """Repo-wide contract for partition-tree training/evaluation bundles."""

    schema_version: str = TREE_BUNDLE_SCHEMA_VERSION
    tree_representation: str = TREE_REPRESENTATION_PARTITION
    leaf_unit: str = LEAF_UNIT_TEXT_TOKEN
    domain: str = ""
    source_kind: str = SOURCE_KIND_RAW_INPUT
    state_contract: str = STATE_CONTRACT_RAW_CONCAT
    reducer_contract: str = REDUCER_CONTRACT_BOTTOM_UP
    dimension: Optional[str] = None
    target_scale: Optional[str] = None
    leaf_policy: Mapping[str, Any] = field(default_factory=dict)
    state_dim: Optional[int] = None
    summary_dim: Optional[int] = None
    f_lineage: Mapping[str, Any] = field(default_factory=dict)
    g_lineage: Mapping[str, Any] = field(default_factory=dict)
    external_state_producer: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_kind = str(self.source_kind or "").strip()
        leaf_unit = str(self.leaf_unit or "").strip()
        state_contract = str(self.state_contract or "").strip()
        reducer_contract = str(self.reducer_contract or "").strip()
        if source_kind not in _ALLOWED_SOURCE_KINDS:
            raise ValueError(f"unsupported tree-bundle source_kind: {source_kind!r}")
        if leaf_unit not in _ALLOWED_LEAF_UNITS:
            raise ValueError(f"unsupported tree-bundle leaf_unit: {leaf_unit!r}")
        if state_contract not in _ALLOWED_STATE_CONTRACTS:
            raise ValueError(f"unsupported tree-bundle state_contract: {state_contract!r}")
        if reducer_contract not in _ALLOWED_REDUCER_CONTRACTS:
            raise ValueError(
                f"unsupported tree-bundle reducer_contract: {reducer_contract!r}"
            )
        if (
            state_contract != STATE_CONTRACT_ORACLE_STATE
            and self.summary_dim is not None
            and self.state_dim is not None
        ):
            summary_dim = int(self.summary_dim)
            state_dim = int(self.state_dim)
            if summary_dim > 0 and state_dim < 2 * summary_dim:
                raise ValueError(
                    "tree-bundle state_dim must be at least 2 * summary_dim: "
                    f"state_dim={state_dim}, summary_dim={summary_dim}"
                )

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": str(self.schema_version or TREE_BUNDLE_SCHEMA_VERSION),
            "tree_representation": str(
                self.tree_representation or TREE_REPRESENTATION_PARTITION
            ),
            "leaf_unit": str(self.leaf_unit),
            "domain": str(self.domain or ""),
            "source_kind": str(self.source_kind),
            "state_contract": str(self.state_contract),
            "reducer_contract": str(self.reducer_contract),
            "dimension": self.dimension,
            "target_scale": self.target_scale,
            "leaf_policy": jsonable(dict(self.leaf_policy or {})),
            "state_dim": self.state_dim,
            "summary_dim": self.summary_dim,
            "f_lineage": jsonable(dict(self.f_lineage or {})),
            "g_lineage": jsonable(dict(self.g_lineage or {})),
            "external_state_producer": self.external_state_producer,
            "metadata": jsonable(dict(self.metadata or {})),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TreeBundleManifest":
        nested = (
            payload.get("tree_bundle_manifest")
            or payload.get("tree_bundle_contract")
            or payload.get("tree_bundle")
            or payload.get("bundle_contract")
        )
        if isinstance(nested, Mapping):
            merged: dict[str, Any] = dict(payload)
            merged.update(dict(nested))
            payload = merged

        source_kind = _clean_optional_str(payload.get("source_kind"))
        if source_kind is None:
            source_kind = _legacy_source_kind(payload) or SOURCE_KIND_RAW_INPUT
        source_kind = str(source_kind).strip().lower()
        state_contract = _clean_optional_str(payload.get("state_contract"))
        if state_contract is None:
            state_contract = default_state_contract_for_source_kind(source_kind)
        state_contract = str(state_contract).strip().lower()
        external_state_producer = _clean_optional_str(
            payload.get("external_state_producer")
        )
        if source_kind == SOURCE_KIND_EXTERNAL_STATE:
            external_state_producer = external_state_producer or "g_benoit"
        return cls(
            schema_version=str(
                payload.get("schema_version") or TREE_BUNDLE_SCHEMA_VERSION
            ),
            tree_representation=str(
                payload.get("tree_representation") or TREE_REPRESENTATION_PARTITION
            ),
            leaf_unit=str(payload.get("leaf_unit") or LEAF_UNIT_TEXT_TOKEN).strip().lower(),
            domain=str(payload.get("domain") or "manifesto_rile"),
            source_kind=str(source_kind),
            state_contract=str(state_contract),
            reducer_contract=str(
                payload.get("reducer_contract") or REDUCER_CONTRACT_BOTTOM_UP
            ).strip().lower(),
            dimension=_clean_optional_str(payload.get("dimension")),
            target_scale=_clean_optional_str(
                payload.get("target_scale") or payload.get("expert_target_scale")
            ),
            leaf_policy=dict(payload.get("leaf_policy") or {}),
            state_dim=(
                int(payload["state_dim"]) if payload.get("state_dim") is not None else None
            ),
            summary_dim=(
                int(payload["summary_dim"])
                if payload.get("summary_dim") is not None
                else None
            ),
            f_lineage=dict(payload.get("f_lineage") or {}),
            g_lineage=dict(payload.get("g_lineage") or {}),
            external_state_producer=external_state_producer,
            metadata=dict(payload.get("metadata") or {}),
        )


def normalize_tree_bundle_manifest(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize new or legacy bundle metadata into TreeBundle v1 shape."""

    return TreeBundleManifest.from_mapping(payload).to_dict()


def tree_bundle_manifest_digest(payload: Mapping[str, Any]) -> str:
    """Stable digest for a normalized TreeBundle manifest."""

    normalized = normalize_tree_bundle_manifest(payload)
    encoded = json.dumps(
        jsonable(normalized),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tree_bundle_input_contract(payload: Mapping[str, Any]) -> JsonDict:
    """Return the canonical input-contract wrapper for a TreeBundle manifest."""

    normalized = normalize_tree_bundle_manifest(payload)
    return {
        "kind": "tree_bundle",
        "schema_version": TREE_BUNDLE_SCHEMA_VERSION,
        "digest": tree_bundle_manifest_digest(normalized),
        "manifest": normalized,
    }


def _normalize_objective_term_name(name: Any) -> tuple[str, Optional[str]]:
    raw = str(name or "").strip()
    normalized = raw.replace("-", "_").replace(" ", "_").strip("_").lower()
    if normalized in {
        OBJECTIVE_TERM_ROOT,
        OBJECTIVE_TERM_LOCAL_LAW_CORRECTED,
    }:
        return normalized, None
    if normalized in _LEGACY_GAP_OBJECTIVE_ALIASES:
        return "", raw or normalized
    alias = _LEGACY_OBJECTIVE_TERM_ALIASES.get(normalized)
    if alias:
        return alias, raw or normalized
    return normalized, None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _objective_weight(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return float(default if parsed is None else parsed)


@dataclass(frozen=True)
class LocalLawDescriptor:
    """Public descriptor for one local law in a problem adapter."""

    law_id: str
    display_name: str
    theorem_label: str = ""
    paper_label: str = ""
    metric_keys: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "law_id": str(self.law_id),
            "display_name": str(self.display_name),
            "theorem_label": str(self.theorem_label),
            "paper_label": str(self.paper_label),
            "metric_keys": [str(key) for key in tuple(self.metric_keys or ())],
            "metadata": jsonable(dict(self.metadata or {})),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LocalLawDescriptor":
        return cls(
            law_id=canonical_law_id(str(payload.get("law_id") or "")),
            display_name=str(payload.get("display_name") or payload.get("law_id") or ""),
            theorem_label=str(payload.get("theorem_label") or ""),
            paper_label=str(payload.get("paper_label") or ""),
            metric_keys=tuple(str(key) for key in list(payload.get("metric_keys") or ())),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LawSetSpec:
    """Named set of public law IDs selected by a problem adapter."""

    law_set_id: str
    law_ids: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "law_set_id": str(self.law_set_id),
            "law_ids": [canonical_law_id(str(law_id)) for law_id in tuple(self.law_ids or ())],
            "metadata": jsonable(dict(self.metadata or {})),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LawSetSpec":
        return cls(
            law_set_id=str(payload.get("law_set_id") or LAW_SET_ALL),
            law_ids=tuple(
                canonical_law_id(str(law_id))
                for law_id in list(payload.get("law_ids") or ())
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


DEFAULT_LOCAL_LAW_DESCRIPTORS = (
    LocalLawDescriptor(
        law_id=LAW_ID_LEAF_PRESERVATION,
        display_name="Leaf preservation",
        theorem_label="L1",
        paper_label="C1",
        metric_keys=("leaf_preservation",),
    ),
    LocalLawDescriptor(
        law_id=LAW_ID_MERGE_PRESERVATION,
        display_name="Merge preservation",
        theorem_label="L2",
        paper_label="C3",
        metric_keys=("merge_preservation",),
    ),
    LocalLawDescriptor(
        law_id=LAW_ID_ON_RANGE_IDEMPOTENCE,
        display_name="On-range idempotence",
        theorem_label="L3",
        paper_label="C2",
        metric_keys=("on_range_idempotence",),
    ),
)


def canonical_law_id(value: str, *, allow_aliases: bool = False) -> str:
    """Return a canonical public law ID, optionally accepting legacy aliases."""

    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError("law_id must be non-empty")
    if normalized in CANONICAL_LAW_IDS:
        return normalized
    alias = LAW_ID_ALIASES.get(normalized)
    if alias and allow_aliases:
        return alias
    if alias:
        raise ValueError(
            f"legacy local-law id {value!r} is not public; use {alias!r}"
        )
    raise ValueError(
        f"unknown local-law id {value!r}; expected one of {sorted(CANONICAL_LAW_IDS)}"
    )


def canonical_law_component_weights(
    values: Mapping[str, Any],
    *,
    allow_aliases: bool = False,
) -> dict[str, float]:
    """Normalize a mapping keyed by public law IDs."""

    out: dict[str, float] = {}
    for raw_key, raw_value in dict(values or {}).items():
        law_id = canonical_law_id(str(raw_key), allow_aliases=allow_aliases)
        out[law_id] = out.get(law_id, 0.0) + float(raw_value)
    return out


def default_law_set_specs(
    law_ids: Sequence[str] | None = None,
) -> tuple[LawSetSpec, ...]:
    ids = tuple(canonical_law_id(str(law_id)) for law_id in (law_ids or CANONICAL_LAW_ID_ORDER))
    specs: list[LawSetSpec] = [
        LawSetSpec(law_set_id=LAW_SET_ALL, law_ids=ids),
        LawSetSpec(law_set_id=LAW_SET_ROOT_ONLY, law_ids=()),
    ]
    if LAW_ID_LEAF_PRESERVATION in ids:
        specs.append(
            LawSetSpec(
                law_set_id=LAW_SET_LEAF_PRESERVATION_ONLY,
                law_ids=(LAW_ID_LEAF_PRESERVATION,),
            )
        )
    if LAW_ID_MERGE_PRESERVATION in ids:
        specs.append(
            LawSetSpec(
                law_set_id=LAW_SET_MERGE_PRESERVATION_ONLY,
                law_ids=(LAW_ID_MERGE_PRESERVATION,),
            )
        )
    if LAW_ID_ON_RANGE_IDEMPOTENCE in ids:
        specs.append(
            LawSetSpec(
                law_set_id=LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
                law_ids=(LAW_ID_ON_RANGE_IDEMPOTENCE,),
            )
        )
    if LAW_ID_LEAF_PRESERVATION in ids and LAW_ID_MERGE_PRESERVATION in ids:
        specs.append(
            LawSetSpec(
                law_set_id=LAW_SET_LEAF_AND_MERGE_PRESERVATION,
                law_ids=(LAW_ID_LEAF_PRESERVATION, LAW_ID_MERGE_PRESERVATION),
            )
        )
    if LAW_ID_MERGE_PRESERVATION in ids and LAW_ID_ON_RANGE_IDEMPOTENCE in ids:
        specs.append(
            LawSetSpec(
                law_set_id=LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE,
                law_ids=(LAW_ID_MERGE_PRESERVATION, LAW_ID_ON_RANGE_IDEMPOTENCE),
            )
        )
    return tuple(specs)


def canonical_law_set_id(value: str | None, *, allow_aliases: bool = False) -> str:
    normalized = str(value or LAW_SET_ALL).strip().lower() or LAW_SET_ALL
    if normalized in CANONICAL_LAW_SET_IDS:
        return normalized
    alias = LAW_SET_ID_ALIASES.get(normalized)
    if alias and allow_aliases:
        return alias
    if alias:
        raise ValueError(
            f"legacy law_set_id {value!r} is not public; use {alias!r}"
        )
    if normalized.startswith("tree_") or normalized.startswith("c"):
        raise ValueError(
            f"legacy law-set name {value!r} is not public; use a generic law_set_id"
        )
    return normalized


def resolve_law_set(
    law_set_id: str | None,
    *,
    registered_law_ids: Sequence[str] | None = None,
    law_sets: Sequence[LawSetSpec | Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Resolve a public law-set ID to canonical law IDs."""

    registered = tuple(
        canonical_law_id(str(law_id))
        for law_id in (registered_law_ids or CANONICAL_LAW_ID_ORDER)
    )
    requested = str(law_set_id or LAW_SET_ALL).strip() or LAW_SET_ALL
    requested = canonical_law_set_id(requested, allow_aliases=False)
    if requested == LAW_SET_ALL:
        return registered
    if requested == LAW_SET_ROOT_ONLY:
        return tuple()
    for raw_spec in tuple(law_sets or ()):
        spec = raw_spec if isinstance(raw_spec, LawSetSpec) else LawSetSpec.from_mapping(raw_spec)
        if str(spec.law_set_id) == requested:
            return tuple(canonical_law_id(str(law_id)) for law_id in tuple(spec.law_ids or ()))
    raise ValueError(f"unknown law_set_id {requested!r}")


@dataclass(frozen=True)
class ProblemAdapterSpec:
    """Problem-specific metadata behind a generic public contract."""

    problem_id: str
    document_type_name: str = "documents"
    theorem_domain_name: str = "summary_states"
    oracle_label_sources: Sequence[str] = field(default_factory=tuple)
    laws: Sequence[LocalLawDescriptor | Mapping[str, Any]] = field(
        default_factory=lambda: DEFAULT_LOCAL_LAW_DESCRIPTORS
    )
    law_sets: Sequence[LawSetSpec | Mapping[str, Any]] = field(default_factory=tuple)
    default_law_set_id: str = LAW_SET_ALL
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def registered_law_ids(self) -> tuple[str, ...]:
        return tuple(
            canonical_law_id(str((law if isinstance(law, Mapping) else law.to_dict()).get("law_id")))
            if isinstance(law, Mapping)
            else canonical_law_id(str(law.law_id))
            for law in tuple(self.laws or ())
        )

    def active_law_ids(self, law_set_id: str | None = None) -> tuple[str, ...]:
        return resolve_law_set(
            law_set_id or self.default_law_set_id,
            registered_law_ids=self.registered_law_ids(),
            law_sets=self.law_sets or default_law_set_specs(self.registered_law_ids()),
        )

    def to_dict(self) -> JsonDict:
        laws = [
            law.to_dict() if isinstance(law, LocalLawDescriptor) else LocalLawDescriptor.from_mapping(law).to_dict()
            for law in tuple(self.laws or ())
        ]
        law_sets = [
            law_set.to_dict() if isinstance(law_set, LawSetSpec) else LawSetSpec.from_mapping(law_set).to_dict()
            for law_set in tuple(self.law_sets or default_law_set_specs(self.registered_law_ids()))
        ]
        return {
            "problem_id": str(self.problem_id),
            "document_type_name": str(self.document_type_name),
            "theorem_domain_name": str(self.theorem_domain_name),
            "oracle_label_sources": [str(source) for source in tuple(self.oracle_label_sources or ())],
            "laws": laws,
            "law_sets": law_sets,
            "default_law_set_id": str(self.default_law_set_id),
            "metadata": jsonable(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class MethodDescriptor:
    """Method metadata used by public reports and runtime dispatch."""

    method_id: str
    method_family: str
    backend: str = ""
    runtime_hook: str = ""
    artifact_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "method_id": str(self.method_id),
            "method_family": str(self.method_family),
            "backend": str(self.backend),
            "runtime_hook": str(self.runtime_hook),
            "artifact_metadata": jsonable(dict(self.artifact_metadata or {})),
        }


def _clean_axis_float(value: Any) -> Optional[float]:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return float(parsed)


def _normalize_method_id(value: Any, *, allow_legacy_law_alias: bool = False) -> tuple[str, Optional[str]]:
    requested = str(value or "").strip()
    if not requested:
        raise ValueError("method_id must be non-empty")
    lowered = requested.lower()
    if lowered in METHOD_ID_LAW_SET_ALIASES:
        method_id, law_set_id = METHOD_ID_LAW_SET_ALIASES[lowered]
        if allow_legacy_law_alias:
            return method_id, law_set_id
        raise ValueError(
            f"legacy method_id {requested!r} encodes a law set; use "
            f"method_id={method_id!r} with law_set_id={law_set_id!r}"
        )
    return requested, None


@dataclass(frozen=True)
class OracleObservationDesignSpec:
    """Public description of how oracle rows were exposed."""

    design_id: str
    design_parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ORACLE_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        design_id = str(self.design_id or "").strip()
        if design_id not in _ALLOWED_ORACLE_OBSERVATION_DESIGNS:
            raise ValueError(
                f"unsupported oracle_observation_design={design_id!r}; expected "
                f"one of {sorted(_ALLOWED_ORACLE_OBSERVATION_DESIGNS)}"
            )
        params = dict(self.design_parameters or {})
        if design_id not in {
            ORACLE_OBSERVATION_DESIGN_SAMPLED_NODES,
            ORACLE_OBSERVATION_DESIGN_SAMPLED_ROOT_NODES,
        } and "sampled_node_rate" in params:
            raise ValueError("sampled_node_rate is active only for sampled-node designs")
        if design_id not in {
            ORACLE_OBSERVATION_DESIGN_BUDGETED_MASS,
            ORACLE_OBSERVATION_DESIGN_SAMPLED_ROOT_NODES,
        }:
            inactive = ORACLE_OBSERVATION_DESIGN_PARAMETER_FIELDS.intersection(params.keys()) - {
                "sampled_node_rate"
            }
            if inactive:
                raise ValueError(
                    "budgeted observation parameters are active only for budgeted_mass designs: "
                    + ", ".join(sorted(inactive))
                )
        if design_id == ORACLE_OBSERVATION_DESIGN_SAMPLED_ROOT_NODES:
            inactive = set(params).intersection(
                {"mass_target_per_doc", "local_label_pool", "local_label_allocation"}
            )
            if inactive:
                raise ValueError(
                    "budgeted mass-allocation parameters are inactive for sampled_root_nodes: "
                    + ", ".join(sorted(inactive))
                )

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": str(self.schema_version or ORACLE_OBSERVATION_SCHEMA_VERSION),
            "design_id": str(self.design_id),
            "design_parameters": jsonable(dict(self.design_parameters or {})),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OracleObservationDesignSpec":
        legacy_fields = sorted(
            str(key)
            for key in LEGACY_ORACLE_OBSERVATION_PUBLIC_FIELDS
            if key in payload
        )
        if legacy_fields:
            raise ValueError(
                "legacy oracle-observation fields are not supported: "
                + ", ".join(legacy_fields)
                + ". Use oracle_observation_design."
            )
        nested = payload.get("oracle_observation_design")
        if isinstance(nested, Mapping):
            payload = nested
        return cls(
            schema_version=str(
                payload.get("schema_version") or ORACLE_OBSERVATION_SCHEMA_VERSION
            ),
            design_id=str(payload.get("design_id") or ""),
            design_parameters=dict(payload.get("design_parameters") or {}),
        )


def oracle_observation_design_metadata(
    design_id: str,
    *,
    design_parameters: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Return the canonical public oracle-observation design payload."""

    return OracleObservationDesignSpec(
        design_id=str(design_id),
        design_parameters=dict(design_parameters or {}),
    ).to_dict()


@dataclass(frozen=True)
class RunAxisSpec:
    """Public run-axis contract shared by problem and method adapters."""

    problem_id: str
    method_id: str
    law_set_id: str = LAW_SET_ALL
    root_share: Optional[float] = None
    local_law_weight: Optional[float] = None
    local_law_component_weights: Mapping[str, float] = field(default_factory=dict)
    role: str = RUN_ROLE_PRIMARY
    display_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RUN_AXIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.problem_id or "").strip():
            raise ValueError("problem_id must be non-empty")
        _normalize_method_id(self.method_id, allow_legacy_law_alias=False)
        canonical_law_set_id(self.law_set_id, allow_aliases=False)
        canonical_law_component_weights(self.local_law_component_weights or {})
        for field_name in ("root_share", "local_law_weight"):
            value = getattr(self, field_name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        role = str(self.role or "").strip()
        if not role:
            raise ValueError("role must be non-empty")

    def resolved_root_share(self) -> Optional[float]:
        if self.root_share is not None:
            return float(self.root_share)
        if self.local_law_weight is not None:
            return max(0.0, 1.0 - float(self.local_law_weight))
        return None

    def resolved_local_law_weight(self) -> Optional[float]:
        if self.local_law_weight is not None:
            return float(self.local_law_weight)
        if self.root_share is not None:
            return max(0.0, 1.0 - float(self.root_share))
        if self.local_law_component_weights:
            return float(sum(float(v) for v in self.local_law_component_weights.values()))
        return None

    def to_dict(self) -> JsonDict:
        method_id, _law_alias = _normalize_method_id(self.method_id, allow_legacy_law_alias=False)
        return {
            "schema_version": str(self.schema_version or RUN_AXIS_SCHEMA_VERSION),
            "problem_id": str(self.problem_id),
            "method_id": str(method_id),
            "law_set_id": canonical_law_set_id(self.law_set_id, allow_aliases=False),
            "root_share": self.resolved_root_share(),
            "local_law_weight": self.resolved_local_law_weight(),
            "local_law_component_weights": canonical_law_component_weights(
                self.local_law_component_weights or {}
            ),
            "role": str(self.role or RUN_ROLE_PRIMARY),
            "display_metadata": jsonable(dict(self.display_metadata or {})),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RunAxisSpec":
        legacy_fields = sorted(
            str(key) for key in LEGACY_RUN_AXIS_PUBLIC_FIELDS if key in payload
        )
        if legacy_fields:
            raise ValueError(
                "legacy public run-axis fields are not supported: "
                + ", ".join(legacy_fields)
                + ". Use problem_id, method_id, law_set_id, root_share, "
                "local_law_weight, and local_law_component_weights."
            )
        method_id, method_law_set = _normalize_method_id(
            payload.get("method_id"), allow_legacy_law_alias=False
        )
        law_set_id = canonical_law_set_id(
            str(payload.get("law_set_id") or method_law_set or LAW_SET_ALL),
            allow_aliases=False,
        )
        return cls(
            schema_version=str(payload.get("schema_version") or RUN_AXIS_SCHEMA_VERSION),
            problem_id=str(payload.get("problem_id") or ""),
            method_id=method_id,
            law_set_id=law_set_id,
            root_share=_clean_axis_float(payload.get("root_share")),
            local_law_weight=_clean_axis_float(payload.get("local_law_weight")),
            local_law_component_weights=canonical_law_component_weights(
                dict(payload.get("local_law_component_weights") or {})
            ),
            role=str(payload.get("role") or RUN_ROLE_PRIMARY),
            display_metadata=dict(payload.get("display_metadata") or {}),
        )


def migrate_legacy_run_axis_mapping(
    payload: Mapping[str, Any],
    *,
    default_problem_id: str = "markov_ops_count",
    default_role: str = RUN_ROLE_PRIMARY,
) -> JsonDict:
    """Explicit migration path for legacy public run-axis payloads."""

    source = dict(payload or {})
    raw_method = source.get("method_id") or source.get("baseline_family")
    method_id, method_law_set = _normalize_method_id(
        raw_method,
        allow_legacy_law_alias=True,
    )
    raw_law_set = (
        source.get("law_set_id")
        or source.get("law_package")
        or source.get("law_package_name")
        or method_law_set
        or LAW_SET_ALL
    )
    migrated = RunAxisSpec(
        problem_id=str(source.get("problem_id") or default_problem_id),
        method_id=method_id,
        law_set_id=canonical_law_set_id(str(raw_law_set), allow_aliases=True),
        root_share=_clean_axis_float(source.get("root_share")),
        local_law_weight=_clean_axis_float(source.get("local_law_weight")),
        local_law_component_weights=canonical_law_component_weights(
            dict(source.get("local_law_component_weights") or {}),
            allow_aliases=True,
        ),
        role=str(source.get("role") or default_role),
        display_metadata={
            **dict(source.get("display_metadata") or {}),
            "legacy_run_axis_source": {
                key: jsonable(value)
                for key, value in source.items()
                if key in LEGACY_RUN_AXIS_PUBLIC_FIELDS
                or key in {"method_id", "law_set_id"}
            },
        },
    )
    return migrated.to_dict()


def _legacy_public_field_reason(key: str) -> str:
    if key in PUBLIC_CONTRACT_LEGACY_FIELDS:
        return "legacy public field"
    for prefix, suffix in PUBLIC_CONTRACT_LEGACY_FIELD_PREFIX_SUFFIXES:
        if key.startswith(prefix) and key.endswith(suffix):
            return f"legacy public field matching {prefix}*{suffix}"
    return ""


def _path_matches(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)


def assert_public_contract_clean(
    payload: Any,
    *,
    surface: str,
    allow_legacy_paths: Sequence[str] = (),
) -> None:
    """Fail fast when a public artifact leaks legacy run/objective vocabulary.

    The normal public path is hard-migrated: callers should emit
    `problem_id`, `method_id`, `law_set_id`, `root_share`,
    `local_law_weight`, and `local_law_component_weights`. Historical
    compatibility belongs in explicit migration helpers, not in report or
    manifest writers.
    """

    violations: list[str] = []

    def _inside_observation_parameters(current_path: str) -> bool:
        marker = "oracle_observation_design.design_parameters"
        return current_path == marker or f".{marker}." in current_path or current_path.endswith(
            f".{marker}"
        )

    def walk(value: Any, path: str) -> None:
        if _path_matches(path, allow_legacy_paths):
            return
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if _path_matches(child_path, allow_legacy_paths):
                    continue
                reason = _legacy_public_field_reason(key)
                if reason:
                    violations.append(f"{child_path}: {reason}")
                if (
                    key in ORACLE_OBSERVATION_DESIGN_PARAMETER_FIELDS
                    and not _inside_observation_parameters(path)
                ):
                    violations.append(
                        f"{child_path}: oracle-observation design parameter must "
                        "be nested under oracle_observation_design.design_parameters"
                    )
                if key == "oracle_observation_design":
                    if not isinstance(child, Mapping):
                        violations.append(
                            f"{child_path}: oracle_observation_design must be a record"
                        )
                    else:
                        try:
                            OracleObservationDesignSpec.from_mapping(child)
                        except Exception as exc:
                            violations.append(f"{child_path}: {exc}")
                if key in {"method_id", "reference_method_id", "supervision_recovery_method_id"}:
                    requested = str(child or "").strip().lower()
                    if requested in METHOD_ID_LAW_SET_ALIASES:
                        method_id, law_set_id = METHOD_ID_LAW_SET_ALIASES[requested]
                        violations.append(
                            f"{child_path}: legacy method_id {child!r} encodes "
                            f"a law set; use method_id={method_id!r} and "
                            f"law_set_id={law_set_id!r}"
                        )
                is_method_run_list = (
                    key in {"method_runs", "reference_method_runs"}
                    or key.endswith("_method_runs")
                    or key.endswith("_reference_method_runs")
                )
                if is_method_run_list and isinstance(child, str):
                    violations.append(
                        f"{child_path}: public method runs must be RunAxisSpec "
                        "records, not an encoded string"
                    )
                if is_method_run_list and isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
                    for index, item in enumerate(child):
                        if isinstance(item, str):
                            violations.append(
                                f"{child_path}[{index}]: public method runs must be "
                                "RunAxisSpec records, not encoded strings"
                            )
                if key in {"law_set_id", "law_set_ids"}:
                    law_values = child if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)) else (child,)
                    for index, raw_law_set in enumerate(law_values):
                        try:
                            canonical_law_set_id(str(raw_law_set), allow_aliases=False)
                        except Exception:
                            suffix = f"[{index}]" if key == "law_set_ids" else ""
                            violations.append(
                                f"{child_path}{suffix}: non-canonical law_set_id "
                                f"{raw_law_set!r}"
                            )
                walk(child, child_path)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "")
    if violations:
        formatted = "; ".join(violations[:20])
        extra = "" if len(violations) <= 20 else f"; ... {len(violations) - 20} more"
        raise ValueError(
            f"{surface} violates the public run/objective contract "
            f"(legacy public run-axis keys/objective keys): {formatted}{extra}"
        )


@dataclass(frozen=True)
class ObjectiveSpec:
    """Repo-wide objective contract for C-TreePO executions."""

    schema_version: str = OBJECTIVE_SCHEMA_VERSION
    objective_family: str = "root_only"
    local_law_estimator: str = LOCAL_LAW_ESTIMATOR_NONE
    local_law_weight: Optional[float] = None
    root_share: float = 1.0
    local_law_component_weights: Mapping[str, float] = field(default_factory=dict)
    terms: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        estimator = str(self.local_law_estimator or "").strip().lower()
        if estimator not in _ALLOWED_LOCAL_LAW_ESTIMATORS:
            raise ValueError(f"unsupported local_law_estimator: {estimator!r}")

    def _normalized_terms(self) -> tuple[JsonDict, list[str]]:
        terms: JsonDict = {}
        aliases: list[str] = []
        for raw_name, raw_payload in dict(self.terms or {}).items():
            name, alias = _normalize_objective_term_name(raw_name)
            if alias:
                raise ValueError(
                    f"legacy objective term {str(alias)!r} is not supported; "
                    "use canonical term names 'root' and 'local_law_corrected'"
                )
            if name not in {
                OBJECTIVE_TERM_ROOT,
                OBJECTIVE_TERM_LOCAL_LAW_CORRECTED,
            }:
                raise ValueError(
                    f"objective term {str(raw_name)!r} is not part of the public "
                    "objective contract; use 'root' and 'local_law_corrected'"
                )
            if not name:
                continue
            payload = dict(raw_payload or {}) if isinstance(raw_payload, Mapping) else {}
            if "weight" in payload:
                payload["weight"] = _objective_weight(payload.get("weight"), 0.0)
            existing = dict(terms.get(name, {}) or {})
            existing.update(payload)
            terms[name] = existing

        terms.setdefault(OBJECTIVE_TERM_ROOT, {})
        terms.setdefault(OBJECTIVE_TERM_LOCAL_LAW_CORRECTED, {})
        terms[OBJECTIVE_TERM_ROOT].setdefault("weight", float(self.root_share))
        terms[OBJECTIVE_TERM_ROOT].setdefault("metric", "root_loss")
        terms[OBJECTIVE_TERM_LOCAL_LAW_CORRECTED].setdefault(
            "weight", float(sum(float(v) for v in dict(self.local_law_component_weights or {}).values()))
        )
        terms[OBJECTIVE_TERM_LOCAL_LAW_CORRECTED].setdefault(
            "estimator", str(self.local_law_estimator)
        )
        terms[OBJECTIVE_TERM_LOCAL_LAW_CORRECTED].setdefault(
            "component_weights",
            canonical_law_component_weights(self.local_law_component_weights or {}),
        )
        return terms, aliases

    def to_dict(self) -> JsonDict:
        terms, aliases = self._normalized_terms()
        metadata = dict(self.metadata or {})
        local_law_weight = (
            float(self.local_law_weight)
            if self.local_law_weight is not None
            else float(sum(float(v) for v in dict(self.local_law_component_weights or {}).values()))
        )
        if aliases:
            metadata["legacy_objective_aliases"] = sorted(set(aliases))
        return {
            "schema_version": str(self.schema_version or OBJECTIVE_SCHEMA_VERSION),
            "objective_family": str(self.objective_family or "root_only"),
            "local_law_estimator": str(self.local_law_estimator or LOCAL_LAW_ESTIMATOR_NONE),
            "local_law_weight": float(local_law_weight),
            "root_share": float(self.root_share),
            "local_law_component_weights": canonical_law_component_weights(
                self.local_law_component_weights or {}
            ),
            "terms": jsonable(terms),
            "metadata": jsonable(metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObjectiveSpec":
        nested = payload.get("objective") or payload.get("objective_spec")
        if isinstance(nested, Mapping):
            merged: dict[str, Any] = dict(payload)
            merged.update(dict(nested))
            payload = merged

        legacy_fields = sorted(
            str(key) for key in LEGACY_OBJECTIVE_PUBLIC_FIELDS if key in payload
        )
        if legacy_fields:
            raise ValueError(
                "legacy public objective fields are not supported: "
                + ", ".join(legacy_fields)
                + ". Use root_share, local_law_weight, and "
                "local_law_component_weights."
            )

        raw_terms = dict(payload.get("terms") or {})
        metadata = dict(payload.get("metadata") or {})
        legacy_aliases: list[str] = []
        for raw_name in raw_terms.keys():
            name, alias = _normalize_objective_term_name(raw_name)
            if alias:
                legacy_aliases.append(str(alias))
            elif name not in {
                OBJECTIVE_TERM_ROOT,
                OBJECTIVE_TERM_LOCAL_LAW_CORRECTED,
            }:
                legacy_aliases.append(str(raw_name))
        if legacy_aliases:
            raise ValueError(
                "legacy or non-canonical objective terms are not supported: "
                + ", ".join(sorted(set(legacy_aliases)))
                + ". Use canonical term names 'root' and 'local_law_corrected'."
            )

        root_weight = _optional_float(
            _first_present(
                payload,
                "root_share",
            )
        )
        local_law_weights = payload.get("local_law_component_weights")
        local_law_component_weights = canonical_law_component_weights(
            dict(local_law_weights or {})
        )
        local_law_estimator = str(
            payload.get("local_law_estimator")
            or payload.get("local_law_estimation")
            or (
                LOCAL_LAW_ESTIMATOR_NONE
                if not local_law_component_weights
                else LOCAL_LAW_ESTIMATOR_CORRECTED
            )
        ).strip().lower()
        if local_law_estimator == "aipw":
            local_law_estimator = LOCAL_LAW_ESTIMATOR_CORRECTED
        if local_law_estimator == "oracle":
            local_law_estimator = LOCAL_LAW_ESTIMATOR_ORACLE_EXACT

        return cls(
            schema_version=str(payload.get("schema_version") or OBJECTIVE_SCHEMA_VERSION),
            objective_family=str(
                payload.get("objective_family")
                or payload.get("name")
                or payload.get("objective_name")
                or "root_only"
            ),
            local_law_estimator=local_law_estimator,
            local_law_weight=_optional_float(
                _first_present(
                    payload,
                    "local_law_weight",
                )
            ),
            root_share=float(1.0 if root_weight is None else root_weight),
            local_law_component_weights=local_law_component_weights,
            terms=raw_terms,
            metadata=metadata,
        )


def normalize_objective_spec(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize new or legacy objective metadata into ObjectiveSpec v1 shape."""

    return ObjectiveSpec.from_mapping(payload).to_dict()


def objective_spec_digest(payload: Mapping[str, Any]) -> str:
    """Stable digest for a normalized ObjectiveSpec."""

    return _stable_digest(normalize_objective_spec(payload))


def objective_metadata(
    *,
    objective_family: str,
    local_law_estimator: str = LOCAL_LAW_ESTIMATOR_NONE,
    local_law_weight: Optional[float] = None,
    root_share: float = 1.0,
    local_law_component_weights: Optional[Mapping[str, float]] = None,
    terms: Optional[Mapping[str, Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Build a normalized ObjectiveSpec v1 dictionary."""

    return ObjectiveSpec(
        objective_family=str(objective_family),
        local_law_estimator=str(local_law_estimator),
        local_law_weight=local_law_weight,
        root_share=float(root_share),
        local_law_component_weights=dict(local_law_component_weights or {}),
        terms=dict(terms or {}),
        metadata=dict(metadata or {}),
    ).to_dict()


def default_objective_for_run(*, role: str = "", backend: str = "") -> JsonDict:
    """Return a conservative default objective for legacy callers."""

    if str(backend).strip().lower() == "data_prep" or "split" in str(role).lower():
        return objective_metadata(
            objective_family="not_applicable",
            local_law_estimator=LOCAL_LAW_ESTIMATOR_NONE,
            root_share=0.0,
            local_law_component_weights={},
            metadata={"defaulted_by": "run_manifest_metadata"},
        )
    return objective_metadata(
        objective_family="root_only",
        local_law_estimator=LOCAL_LAW_ESTIMATOR_NONE,
        root_share=1.0,
        local_law_component_weights={},
        metadata={"defaulted_by": "run_manifest_metadata"},
    )


def validate_objective_spec(
    payload: Mapping[str, Any],
    *,
    require_canonical_public_names: bool = False,
) -> ObjectiveSpec:
    """Validate an ObjectiveSpec and return the parsed contract."""

    manifest = ObjectiveSpec.from_mapping(payload)
    normalized = manifest.to_dict()
    errors: list[str] = []
    if normalized["schema_version"] != OBJECTIVE_SCHEMA_VERSION:
        errors.append(
            "schema_version mismatch: "
            f"expected {OBJECTIVE_SCHEMA_VERSION}, found {normalized['schema_version']}"
        )
    terms = dict(normalized.get("terms") or {})
    for name in (
        OBJECTIVE_TERM_ROOT,
        OBJECTIVE_TERM_LOCAL_LAW_CORRECTED,
    ):
        term = terms.get(name)
        if not isinstance(term, Mapping):
            errors.append(f"ObjectiveSpec missing canonical term {name!r}")
        elif "weight" not in term:
            errors.append(f"ObjectiveSpec term {name!r} missing weight")
    metadata = dict(normalized.get("metadata") or {})
    if require_canonical_public_names and metadata.get("legacy_objective_aliases"):
        errors.append(
            "ObjectiveSpec uses legacy public objective aliases: "
            + ", ".join(str(x) for x in metadata["legacy_objective_aliases"])
        )
    if require_canonical_public_names and metadata.get("legacy_objective_field_aliases"):
        errors.append(
            "ObjectiveSpec uses legacy public objective field aliases: "
            + ", ".join(str(x) for x in metadata["legacy_objective_field_aliases"])
        )
    if errors:
        raise ValueError("; ".join(errors))
    return manifest


def fg_lineage_metadata(
    *,
    f_init: str,
    g_init: str,
    schedule: str = "",
    f_lineage: Optional[Mapping[str, Any]] = None,
    g_lineage: Optional[Mapping[str, Any]] = None,
    tree_bundle: Optional[Mapping[str, Any]] = None,
    reducer_contract: str = REDUCER_CONTRACT_BOTTOM_UP,
) -> JsonDict:
    """Common manifest fragment for f/g initialization, schedule, and bundle use."""

    out: JsonDict = {
        "f_init": str(f_init),
        "g_init": str(g_init),
        "schedule": str(schedule or ""),
        "schedule_steps": [str(ch) for ch in str(schedule or "")],
        "reducer_contract": str(reducer_contract or REDUCER_CONTRACT_BOTTOM_UP),
        "f_lineage": jsonable(dict(f_lineage or {})),
        "g_lineage": jsonable(dict(g_lineage or {})),
    }
    if tree_bundle is not None:
        normalized = normalize_tree_bundle_manifest(tree_bundle)
        out["tree_bundle_manifest"] = normalized
        out["tree_bundle_manifest_digest"] = tree_bundle_manifest_digest(normalized)
    return out


@dataclass(frozen=True)
class RunManifest:
    """Repo-wide execution envelope for C-TreePO runs.

    This is intentionally generic: paper runs, exploratory runs, long GPU jobs,
    and compatibility wrappers all use the same shape.  Publication readiness is
    a stricter validation state on this manifest, not a separate contract.
    """

    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION
    run_id: str = ""
    domain: str = ""
    role: str = ""
    backend: str = ""
    status: str = "planned"
    input_contracts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    f_init: str = ""
    g_init: str = ""
    f_lineage: Mapping[str, Any] = field(default_factory=dict)
    g_lineage: Mapping[str, Any] = field(default_factory=dict)
    reducer_contract: str = REDUCER_CONTRACT_BOTTOM_UP
    schedule: str = ""
    schedule_steps: Sequence[str] = field(default_factory=tuple)
    objective: Mapping[str, Any] = field(default_factory=dict)
    optimizer_config: Mapping[str, Any] = field(default_factory=dict)
    output_artifacts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    audit_results: Mapping[str, Any] = field(default_factory=dict)
    quarantine: Mapping[str, Any] = field(default_factory=dict)
    command: Sequence[str] = field(default_factory=tuple)
    allow_legacy: bool = False
    publication_ready: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status or "").strip().lower()
        if status not in _RUN_STATUSES:
            raise ValueError(f"unsupported run status: {self.status!r}")
        reducer = str(self.reducer_contract or REDUCER_CONTRACT_BOTTOM_UP).strip().lower()
        if reducer and reducer not in _ALLOWED_REDUCER_CONTRACTS:
            raise ValueError(f"unsupported run reducer_contract: {reducer!r}")
        for contract in self.input_contracts or ():
            if not isinstance(contract, Mapping):
                raise ValueError("run input_contracts must be mappings")
            kind = str(contract.get("kind") or "").strip()
            if kind == "tree_bundle":
                manifest = contract.get("manifest")
                if not isinstance(manifest, Mapping):
                    manifest = contract
                validate_tree_bundle_manifest(manifest)
        if self.objective:
            validate_objective_spec(self.objective)

    def to_dict(self) -> JsonDict:
        steps = list(self.schedule_steps or ())
        if not steps and self.schedule:
            steps = [str(ch) for ch in str(self.schedule)]
        return {
            "schema_version": str(self.schema_version or RUN_MANIFEST_SCHEMA_VERSION),
            "run_id": str(self.run_id or ""),
            "domain": str(self.domain or ""),
            "role": str(self.role or ""),
            "backend": str(self.backend or ""),
            "status": str(self.status or "planned"),
            "input_contracts": jsonable(list(self.input_contracts or ())),
            "f_init": str(self.f_init or ""),
            "g_init": str(self.g_init or ""),
            "f_lineage": jsonable(dict(self.f_lineage or {})),
            "g_lineage": jsonable(dict(self.g_lineage or {})),
            "reducer_contract": str(
                self.reducer_contract or REDUCER_CONTRACT_BOTTOM_UP
            ),
            "schedule": str(self.schedule or ""),
            "schedule_steps": jsonable(steps),
            "objective": jsonable(normalize_objective_spec(self.objective))
            if self.objective
            else {},
            "objective_digest": objective_spec_digest(self.objective)
            if self.objective
            else "",
            "optimizer_config": jsonable(dict(self.optimizer_config or {})),
            "output_artifacts": jsonable(list(self.output_artifacts or ())),
            "audit_results": jsonable(dict(self.audit_results or {})),
            "quarantine": jsonable(dict(self.quarantine or {})),
            "command": jsonable(list(self.command or ())),
            "allow_legacy": bool(self.allow_legacy),
            "publication_ready": bool(self.publication_ready),
            "metadata": jsonable(dict(self.metadata or {})),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RunManifest":
        nested = payload.get("run_manifest")
        if isinstance(nested, Mapping):
            merged: dict[str, Any] = dict(payload)
            merged.update(dict(nested))
            payload = merged

        input_contracts = list(payload.get("input_contracts") or ())
        if not input_contracts:
            bundle_payload: Any = None
            for key in (
                "tree_bundle_manifest",
                "tree_bundle_contract",
                "tree_bundle",
                "bundle_contract",
            ):
                value = payload.get(key)
                if isinstance(value, Mapping):
                    bundle_payload = value
                    break
            if bundle_payload is not None:
                input_contracts = [tree_bundle_input_contract(bundle_payload)]

        schedule = str(payload.get("schedule") or "")
        steps = payload.get("schedule_steps")
        if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
            schedule_steps = [str(x) for x in steps]
        else:
            schedule_steps = [str(ch) for ch in schedule] if schedule else []

        return cls(
            schema_version=str(
                payload.get("schema_version") or RUN_MANIFEST_SCHEMA_VERSION
            ),
            run_id=str(payload.get("run_id") or ""),
            domain=str(payload.get("domain") or ""),
            role=str(payload.get("role") or ""),
            backend=str(payload.get("backend") or ""),
            status=str(payload.get("status") or "planned").strip().lower(),
            input_contracts=tuple(
                jsonable(contract) for contract in input_contracts if isinstance(contract, Mapping)
            ),
            f_init=str(payload.get("f_init") or ""),
            g_init=str(payload.get("g_init") or ""),
            f_lineage=dict(payload.get("f_lineage") or {}),
            g_lineage=dict(payload.get("g_lineage") or {}),
            reducer_contract=str(
                payload.get("reducer_contract") or REDUCER_CONTRACT_BOTTOM_UP
            ).strip().lower(),
            schedule=schedule,
            schedule_steps=tuple(schedule_steps),
            objective=(
                normalize_objective_spec(payload.get("objective"))
                if isinstance(payload.get("objective"), Mapping)
                and bool(payload.get("objective"))
                else (
                    normalize_objective_spec(payload.get("objective_spec"))
                    if isinstance(payload.get("objective_spec"), Mapping)
                    and bool(payload.get("objective_spec"))
                    else {}
                )
            ),
            optimizer_config=dict(payload.get("optimizer_config") or {}),
            output_artifacts=tuple(
                item for item in (payload.get("output_artifacts") or ()) if isinstance(item, Mapping)
            ),
            audit_results=dict(payload.get("audit_results") or {}),
            quarantine=dict(payload.get("quarantine") or {}),
            command=tuple(str(x) for x in (payload.get("command") or ())),
            allow_legacy=bool(payload.get("allow_legacy", False)),
            publication_ready=bool(payload.get("publication_ready", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


def normalize_run_manifest(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize a mapping into RunManifest v1 shape."""

    return RunManifest.from_mapping(payload).to_dict()


def run_manifest_digest(payload: Mapping[str, Any]) -> str:
    """Stable digest for a normalized RunManifest."""

    return _stable_digest(normalize_run_manifest(payload))


def run_manifest_metadata(
    *,
    run_id: str,
    domain: str,
    role: str,
    backend: str,
    status: str = "planned",
    tree_bundle: Optional[Mapping[str, Any]] = None,
    input_contracts: Optional[Sequence[Mapping[str, Any]]] = None,
    f_init: str = "",
    g_init: str = "",
    f_lineage: Optional[Mapping[str, Any]] = None,
    g_lineage: Optional[Mapping[str, Any]] = None,
    reducer_contract: str = REDUCER_CONTRACT_BOTTOM_UP,
    schedule: str = "",
    objective: Optional[Mapping[str, Any]] = None,
    optimizer_config: Optional[Mapping[str, Any]] = None,
    output_artifacts: Optional[Sequence[Mapping[str, Any]]] = None,
    audit_results: Optional[Mapping[str, Any]] = None,
    quarantine: Optional[Mapping[str, Any]] = None,
    command: Optional[Sequence[Any]] = None,
    allow_legacy: bool = False,
    publication_ready: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Build a normalized RunManifest v1 dictionary."""

    contracts = list(input_contracts or ())
    if tree_bundle is not None:
        contracts.append(tree_bundle_input_contract(tree_bundle))
    manifest = RunManifest(
        run_id=str(run_id),
        domain=str(domain),
        role=str(role),
        backend=str(backend),
        status=str(status),
        input_contracts=tuple(contracts),
        f_init=str(f_init or ""),
        g_init=str(g_init or ""),
        f_lineage=dict(f_lineage or {}),
        g_lineage=dict(g_lineage or {}),
        reducer_contract=str(reducer_contract or REDUCER_CONTRACT_BOTTOM_UP),
        schedule=str(schedule or ""),
        objective=normalize_objective_spec(objective)
        if isinstance(objective, Mapping)
        else default_objective_for_run(role=role, backend=backend),
        optimizer_config=dict(optimizer_config or {}),
        output_artifacts=tuple(output_artifacts or ()),
        audit_results=dict(audit_results or {}),
        quarantine=dict(quarantine or {}),
        command=tuple(str(x) for x in (command or ())),
        allow_legacy=bool(allow_legacy),
        publication_ready=bool(publication_ready),
        metadata=dict(metadata or {}),
    ).to_dict()
    manifest["run_manifest_digest"] = run_manifest_digest(manifest)
    return manifest


def validate_run_manifest(
    payload: Mapping[str, Any],
    *,
    expected_domain: Optional[str] = None,
    expected_role: Optional[str] = None,
    expected_backend: Optional[str] = None,
    require_tree_bundle: bool = False,
    require_lineage: bool = False,
    require_objective: bool = False,
    require_publication_ready: bool = False,
    allow_legacy: bool = False,
) -> RunManifest:
    """Validate a RunManifest and return the parsed manifest."""

    manifest = RunManifest.from_mapping(payload)
    errors: list[str] = []
    if not allow_legacy and manifest.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
        errors.append(
            "schema_version mismatch: "
            f"expected {RUN_MANIFEST_SCHEMA_VERSION}, found {manifest.schema_version}"
        )
    if expected_domain and manifest.domain != expected_domain:
        errors.append(f"domain mismatch: expected {expected_domain}, found {manifest.domain}")
    if expected_role and manifest.role != expected_role:
        errors.append(f"role mismatch: expected {expected_role}, found {manifest.role}")
    if expected_backend and manifest.backend != expected_backend:
        errors.append(
            f"backend mismatch: expected {expected_backend}, found {manifest.backend}"
        )

    tree_contracts = [
        contract
        for contract in manifest.input_contracts
        if str(contract.get("kind") or "") == "tree_bundle"
    ]
    if require_tree_bundle and not tree_contracts:
        errors.append("RunManifest requires a TreeBundle input contract")
    for contract in tree_contracts:
        manifest_payload = contract.get("manifest")
        if not isinstance(manifest_payload, Mapping):
            manifest_payload = contract
        try:
            validate_tree_bundle_manifest(manifest_payload)
        except Exception as exc:
            errors.append(f"invalid TreeBundle input contract: {exc}")

    if require_objective or require_publication_ready:
        if not manifest.objective:
            errors.append("RunManifest requires ObjectiveSpec metadata")
        else:
            try:
                validate_objective_spec(
                    manifest.objective,
                    require_canonical_public_names=bool(require_publication_ready),
                )
            except Exception as exc:
                errors.append(f"invalid ObjectiveSpec metadata: {exc}")

    if require_lineage:
        if not manifest.f_init:
            errors.append("RunManifest requires f_init")
        if not manifest.g_init:
            errors.append("RunManifest requires g_init")
        if not manifest.f_lineage:
            errors.append("RunManifest requires f_lineage")
        if not manifest.g_lineage:
            errors.append("RunManifest requires g_lineage")

    if require_publication_ready:
        if not bool(manifest.publication_ready):
            errors.append("RunManifest is not publication_ready")
        if bool(manifest.allow_legacy):
            errors.append("publication_ready RunManifest cannot allow legacy mode")
        audit_ok = manifest.audit_results.get("ok")
        if audit_ok is not True:
            errors.append("publication_ready RunManifest requires audit_results.ok=true")
        quarantine_class = str(manifest.quarantine.get("classification") or "").strip()
        if quarantine_class in _BAD_QUARANTINE_CLASSES:
            errors.append(
                f"RunManifest quarantine classification is not publication safe: {quarantine_class}"
            )

    if errors:
        raise ValueError("; ".join(errors))
    return manifest


def sketch_tree_bundle_metadata(
    *,
    family: str = "",
    query: str = "",
    sketch: str = "",
    leaf_unit: str = LEAF_UNIT_STREAM_ITEM,
    source_kind: str = SOURCE_KIND_RAW_INPUT,
    state_contract: str = STATE_CONTRACT_RAW_CONCAT,
    summary_dim: Optional[int] = None,
    state_dim: Optional[int] = None,
    f_init: str = "official_oracle",
    g_init: str = STATE_CONTRACT_RAW_CONCAT,
    schedule: str = "",
    f_lineage: Optional[Mapping[str, Any]] = None,
    g_lineage: Optional[Mapping[str, Any]] = None,
    leaf_policy: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Build the canonical TreeBundle metadata for sketch/simulation rows."""

    f_meta = {
        "init": str(f_init),
        "artifact": "official_oracle",
        "backend": "datasketches_or_native",
        **dict(f_lineage or {}),
    }
    g_meta = {
        "init": str(g_init),
        "artifact": str(g_init),
        **dict(g_lineage or {}),
    }
    row_meta = {
        "family": str(family or ""),
        "query": str(query or ""),
        "sketch": str(sketch or ""),
        "f_init": str(f_init),
        "g_init": str(g_init),
        "schedule": str(schedule or ""),
        **dict(metadata or {}),
    }
    return tree_bundle_metadata(
        domain="classical_sketch",
        leaf_unit=str(leaf_unit),
        source_kind=str(source_kind),
        state_contract=str(state_contract),
        reducer_contract=REDUCER_CONTRACT_BOTTOM_UP,
        state_dim=state_dim,
        summary_dim=summary_dim,
        leaf_policy=dict(leaf_policy or {}),
        f_lineage=f_meta,
        g_lineage=g_meta,
        metadata=row_meta,
        include_legacy_manifesto_aliases=False,
    )


def markov_tree_bundle_metadata(
    *,
    leaf_policy: Optional[Mapping[str, Any]] = None,
    state_dim: Optional[int] = None,
    summary_dim: Optional[int] = None,
    f_init: str = "official_oracle",
    g_init: str = STATE_CONTRACT_RAW_CONCAT,
    schedule: str = "balanced",
    metadata: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Build the canonical TreeBundle metadata for Markov synthetic bundles."""

    return tree_bundle_metadata(
        domain="markov",
        leaf_unit=LEAF_UNIT_SYNTHETIC_ATOM,
        source_kind=SOURCE_KIND_SYNTHETIC_ORACLE,
        state_contract=STATE_CONTRACT_ORACLE_STATE,
        reducer_contract=REDUCER_CONTRACT_BOTTOM_UP,
        state_dim=state_dim,
        summary_dim=summary_dim,
        leaf_policy=dict(leaf_policy or {}),
        f_lineage={"init": str(f_init), "artifact": "synthetic_oracle"},
        g_lineage={"init": str(g_init), "artifact": str(g_init)},
        metadata={
            "f_init": str(f_init),
            "g_init": str(g_init),
            "schedule": str(schedule or ""),
            **dict(metadata or {}),
        },
        include_legacy_manifesto_aliases=False,
    )


def tree_bundle_metadata(
    *,
    domain: str,
    leaf_unit: str,
    source_kind: str,
    dimension: Optional[str] = None,
    target_scale: Optional[str] = None,
    leaf_policy: Optional[Mapping[str, Any]] = None,
    state_contract: Optional[str] = None,
    reducer_contract: str = REDUCER_CONTRACT_BOTTOM_UP,
    state_dim: Optional[int] = None,
    summary_dim: Optional[int] = None,
    f_lineage: Optional[Mapping[str, Any]] = None,
    g_lineage: Optional[Mapping[str, Any]] = None,
    external_state_producer: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    include_legacy_manifesto_aliases: bool = True,
) -> JsonDict:
    """Build flat metadata plus nested TreeBundle v1 manifest fields.

    The flat fields make artifacts easy to grep and preserve older code paths.
    The nested ``tree_bundle_manifest`` is the canonical contract.
    """

    manifest = TreeBundleManifest(
        domain=str(domain),
        leaf_unit=str(leaf_unit),
        source_kind=str(source_kind),
        state_contract=str(
            state_contract or default_state_contract_for_source_kind(source_kind)
        ),
        reducer_contract=str(reducer_contract),
        dimension=dimension,
        target_scale=target_scale,
        leaf_policy=dict(leaf_policy or {}),
        state_dim=state_dim,
        summary_dim=summary_dim,
        f_lineage=dict(f_lineage or {}),
        g_lineage=dict(g_lineage or {}),
        external_state_producer=external_state_producer,
        metadata=dict(metadata or {}),
    ).to_dict()
    out = dict(manifest)
    out["tree_bundle_manifest"] = dict(manifest)
    if include_legacy_manifesto_aliases:
        legacy_kind = legacy_tree_bundle_kind_for_source_kind(str(source_kind))
        legacy_text_source = legacy_tree_text_source_for_source_kind(str(source_kind))
        if legacy_kind:
            out["tree_bundle_kind"] = legacy_kind
        if legacy_text_source:
            out["tree_text_source"] = legacy_text_source
        if str(source_kind) == SOURCE_KIND_RAW_INPUT:
            out.setdefault("tree_state_source", "raw_input")
        elif str(source_kind) == SOURCE_KIND_EXTERNAL_STATE:
            out.setdefault("tree_state_source", "external_state")
    return out


def validate_tree_bundle_manifest(
    payload: Mapping[str, Any],
    *,
    expected_domain: Optional[str] = None,
    expected_leaf_unit: Optional[str] = None,
    expected_source_kind: Optional[str] = None,
    expected_dimension: Optional[str] = None,
    expected_target_scale: Optional[str] = None,
    require_bottom_up: bool = True,
) -> TreeBundleManifest:
    manifest = TreeBundleManifest.from_mapping(payload)
    errors: list[str] = []
    if expected_domain and manifest.domain != expected_domain:
        errors.append(f"domain mismatch: expected {expected_domain}, found {manifest.domain}")
    if expected_leaf_unit and manifest.leaf_unit != expected_leaf_unit:
        errors.append(
            f"leaf_unit mismatch: expected {expected_leaf_unit}, found {manifest.leaf_unit}"
        )
    if expected_source_kind and manifest.source_kind != expected_source_kind:
        errors.append(
            "source_kind mismatch: "
            f"expected {expected_source_kind}, found {manifest.source_kind}"
        )
    if expected_dimension and manifest.dimension and manifest.dimension != expected_dimension:
        errors.append(
            f"dimension mismatch: expected {expected_dimension}, found {manifest.dimension}"
        )
    if (
        expected_target_scale
        and manifest.target_scale
        and manifest.target_scale != expected_target_scale
    ):
        errors.append(
            "target_scale mismatch: "
            f"expected {expected_target_scale}, found {manifest.target_scale}"
        )
    if require_bottom_up and manifest.reducer_contract != REDUCER_CONTRACT_BOTTOM_UP:
        errors.append(f"reducer_contract must be bottom_up, found {manifest.reducer_contract}")
    if errors:
        raise ValueError("; ".join(errors))
    return manifest


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a persisted component artifact."""

    kind: str
    uri: str
    family: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "kind": str(self.kind),
            "uri": str(self.uri),
            "family": str(self.family),
            "metadata": jsonable(dict(self.metadata or {})),
        }

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        kind: str = "artifact",
        family: str = "",
    ) -> "ArtifactRef":
        if isinstance(value, ArtifactRef):
            return value
        if isinstance(value, Mapping) and "uri" in value:
            return cls(
                kind=str(value.get("kind") or kind),
                uri=str(value.get("uri") or ""),
                family=str(value.get("family") or family),
                metadata=dict(value.get("metadata") or {}),
            )
        return cls(kind=str(kind), uri=str(value), family=str(family))


@dataclass(frozen=True)
class CTreePOProgramSpec:
    """Runtime program description for f/g or classical-sketch execution."""

    space_kind: str
    family: str
    method_id: str = ""
    f_artifact: Any = None
    g_artifact: Any = None
    leaf_adapter_artifact: Any = None
    backend_config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "space_kind": str(self.space_kind),
            "method_id": str(self.method_id),
            "f_artifact": jsonable(self.f_artifact),
            "g_artifact": jsonable(self.g_artifact),
            "leaf_adapter_artifact": jsonable(self.leaf_adapter_artifact),
            "backend_config": jsonable(dict(self.backend_config or {})),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CTreePOProgramSpec":
        if "family" in payload and "method_id" not in payload:
            raise ValueError("public program specs require method_id; family is not a public fallback")
        return cls(
            space_kind=str(payload.get("space_kind") or ""),
            family=str(payload.get("family") or ""),
            method_id=str(payload.get("method_id") or ""),
            f_artifact=payload.get("f_artifact"),
            g_artifact=payload.get("g_artifact"),
            leaf_adapter_artifact=payload.get("leaf_adapter_artifact"),
            backend_config=dict(payload.get("backend_config") or {}),
        )


@dataclass(frozen=True)
class CTreePOLearningSpec:
    """Learning job description for a C-TreePO f/g ladder."""

    space_kind: str
    family: str
    schedule: str
    initial_artifacts: Mapping[str, Any] = field(default_factory=dict)
    train_data: Any = None
    eval_data: Any = None
    backend_config: Mapping[str, Any] = field(default_factory=dict)
    axis: Mapping[str, Any] = field(default_factory=dict)

    def with_schedule(self, schedule: str) -> "CTreePOLearningSpec":
        return replace(self, schedule=str(schedule))

    def with_initial_artifacts(
        self,
        artifacts: Mapping[str, Any],
    ) -> "CTreePOLearningSpec":
        return replace(self, initial_artifacts=dict(artifacts))

    def to_dict(self) -> JsonDict:
        return {
            "space_kind": str(self.space_kind),
            "family": str(self.family),
            "schedule": str(self.schedule),
            "initial_artifacts": jsonable(dict(self.initial_artifacts or {})),
            "backend_config": jsonable(dict(self.backend_config or {})),
            "axis": jsonable(dict(self.axis or {})),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CTreePOLearningSpec":
        return cls(
            space_kind=str(payload.get("space_kind") or ""),
            family=str(payload.get("family") or ""),
            schedule=str(payload.get("schedule") or ""),
            initial_artifacts=dict(payload.get("initial_artifacts") or {}),
            train_data=payload.get("train_data"),
            eval_data=payload.get("eval_data"),
            backend_config=dict(payload.get("backend_config") or {}),
            axis=dict(payload.get("axis") or {}),
        )


@dataclass(frozen=True)
class CTreePOFitResult:
    """Uniform result returned by `src.ctreepo.learning` entry points."""

    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    summary: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None

    def to_dict(self) -> JsonDict:
        return {
            "status": str(self.status),
            "metrics": jsonable(dict(self.metrics or {})),
            "artifacts": jsonable(dict(self.artifacts or {})),
            "history": jsonable(list(self.history or ())),
            "summary": jsonable(dict(self.summary or {})),
            "manifest_path": self.manifest_path,
        }


__all__ = [
    "ArtifactRef",
    "CANONICAL_LAW_ID_ORDER",
    "CANONICAL_LAW_IDS",
    "CANONICAL_LAW_SET_IDS",
    "CTreePOFitResult",
    "CTreePOLearningSpec",
    "CTreePOProgramSpec",
    "DEFAULT_LOCAL_LAW_DESCRIPTORS",
    "LEAF_UNIT_EMBEDDING_ROW",
    "LEAF_UNIT_STREAM_ITEM",
    "LEAF_UNIT_SYNTHETIC_ATOM",
    "LEAF_UNIT_TEXT_TOKEN",
    "LAW_ID_LEAF_PRESERVATION",
    "LAW_ID_MERGE_PRESERVATION",
    "LAW_ID_ON_RANGE_IDEMPOTENCE",
    "LAW_SET_ALL",
    "LAW_SET_LEAF_AND_MERGE_PRESERVATION",
    "LAW_SET_LEAF_PRESERVATION_ONLY",
    "LAW_SET_MERGE_AND_ON_RANGE_IDEMPOTENCE",
    "LAW_SET_MERGE_PRESERVATION_ONLY",
    "LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY",
    "LAW_SET_ROOT_ONLY",
    "LOCAL_LAW_ESTIMATOR_CORRECTED",
    "LOCAL_LAW_ESTIMATOR_EXTERNAL_PASSTHROUGH",
    "LOCAL_LAW_ESTIMATOR_NONE",
    "LOCAL_LAW_ESTIMATOR_ORACLE_EXACT",
    "LOCAL_LAW_ESTIMATOR_ORACLE_STATE",
    "LOCAL_LAW_ESTIMATOR_PROXY_ONLY",
    "LawSetSpec",
    "LocalLawDescriptor",
    "MethodDescriptor",
    "OBJECTIVE_SCHEMA_VERSION",
    "OBJECTIVE_TERM_LOCAL_LAW_CORRECTED",
    "OBJECTIVE_TERM_ROOT",
    "ORACLE_OBSERVATION_DESIGN_BUDGETED_MASS",
    "ORACLE_OBSERVATION_DESIGN_DENSE_ORACLE",
    "ORACLE_OBSERVATION_DESIGN_ROOT_ONLY",
    "ORACLE_OBSERVATION_DESIGN_SAMPLED_NODES",
    "ORACLE_OBSERVATION_DESIGN_SAMPLED_ROOT_NODES",
    "ORACLE_OBSERVATION_SCHEMA_VERSION",
    "REDUCER_CONTRACT_BOTTOM_UP",
    "RUN_AXIS_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RUN_ROLE_AUXILIARY",
    "RUN_ROLE_PRIMARY",
    "RUN_ROLE_REFERENCE",
    "SOURCE_KIND_DERIVED_CACHE",
    "SOURCE_KIND_EXTERNAL_STATE",
    "SOURCE_KIND_RAW_INPUT",
    "SOURCE_KIND_SYNTHETIC_ORACLE",
    "STATE_CONTRACT_BOTTOM_UP_G",
    "STATE_CONTRACT_EXTERNAL_PASSTHROUGH",
    "STATE_CONTRACT_ORACLE_STATE",
    "STATE_CONTRACT_RAW_CONCAT",
    "TREE_BUNDLE_SCHEMA_VERSION",
    "TREE_REPRESENTATION_PARTITION",
    "ObjectiveSpec",
    "OracleObservationDesignSpec",
    "ProblemAdapterSpec",
    "RunManifest",
    "RunAxisSpec",
    "assert_public_contract_clean",
    "TreeBundleManifest",
    "canonical_law_component_weights",
    "canonical_law_id",
    "canonical_law_set_id",
    "default_state_contract_for_source_kind",
    "default_law_set_specs",
    "default_objective_for_run",
    "jsonable",
    "legacy_tree_bundle_kind_for_source_kind",
    "legacy_tree_text_source_for_source_kind",
    "fg_lineage_metadata",
    "markov_tree_bundle_metadata",
    "migrate_legacy_run_axis_mapping",
    "normalize_objective_spec",
    "normalize_run_manifest",
    "normalize_tree_bundle_manifest",
    "objective_metadata",
    "objective_spec_digest",
    "oracle_observation_design_metadata",
    "resolve_law_set",
    "run_manifest_digest",
    "run_manifest_metadata",
    "sketch_tree_bundle_metadata",
    "tree_bundle_metadata",
    "tree_bundle_input_contract",
    "tree_bundle_manifest_digest",
    "validate_objective_spec",
    "validate_run_manifest",
    "validate_tree_bundle_manifest",
]
