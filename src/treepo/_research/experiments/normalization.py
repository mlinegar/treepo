from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from treepo._research.experiments.contracts import ControlRef, ResultRow, SupervisionRef


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in {None, ""}:
        return default
    try:
        return int(value)
    except Exception:
        return default


def derive_markov_coverage_label(
    *,
    root_rate: float | None = None,
    leaf_rate: float | None = None,
    internal_rate: float | None = None,
    package_name: str = "",
) -> str:
    package = str(package_name or "").strip()
    if package:
        return package
    if root_rate is None:
        return ""
    root_pct = int(round(100.0 * float(root_rate)))
    label = f"R{root_pct}"
    local_labels: List[str] = []
    if leaf_rate is not None and leaf_rate > 0.0:
        leaf_pct = int(round(100.0 * float(leaf_rate)))
        local_labels.append(f"LcIa{leaf_pct}" if internal_rate == leaf_rate else f"Lc{leaf_pct}")
    if internal_rate is not None and internal_rate > 0.0 and internal_rate != leaf_rate:
        internal_pct = int(round(100.0 * float(internal_rate)))
        local_labels.append(f"Ia{internal_pct}")
    if local_labels:
        label = label + "+" + "+".join(local_labels)
    return label


def _infer_rate_from_package_name(package_name: str, prefix: str) -> float | None:
    package = str(package_name or "").strip().lower()
    if not package:
        return None
    pattern = re.compile(rf"{re.escape(prefix)}(?:_)?(?:count|full)(\d+)")
    match = pattern.search(package)
    if match is None and prefix == "full":
        match = re.match(r"full(\d+)", package)
    if match is None:
        return None
    try:
        return float(int(match.group(1))) / 100.0
    except Exception:
        return None


def supervision_ref_from_markov_config(
    config: Mapping[str, Any] | None,
    *,
    package_name: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> SupervisionRef | None:
    data = dict(config or {})
    root_rate = _safe_float(
        data.get("root_label_rate", data.get("full_doc_label_rate", data.get("root_rate"))),
        None,
    )
    if root_rate is None:
        root_rate = _infer_rate_from_package_name(package_name, "full")
    leaf_kind = str(data.get("leaf_supervision_kind", "") or "")
    leaf_rate = _safe_float(data.get("leaf_label_rate"), None)
    if leaf_rate is None:
        leaf_rate = _infer_rate_from_package_name(package_name, "leaf")
    internal_kind = str(data.get("internal_supervision_kind", "") or "")
    internal_rate = _safe_float(data.get("internal_label_rate"), None)
    if internal_rate is None:
        internal_rate = _infer_rate_from_package_name(package_name, "internal")
    if root_rate is None and not leaf_kind and leaf_rate is None and not internal_kind and internal_rate is None:
        if not package_name:
            return None
    coverage_label = derive_markov_coverage_label(
        root_rate=root_rate,
        leaf_rate=leaf_rate,
        internal_rate=internal_rate,
        package_name=package_name,
    )
    merged_metadata = dict(metadata or {})
    if package_name:
        merged_metadata.setdefault("package_name", str(package_name))
    return SupervisionRef(
        root_rate=root_rate,
        leaf_kind=leaf_kind,
        leaf_rate=leaf_rate,
        internal_kind=internal_kind,
        internal_rate=internal_rate,
        topology_scope="tree",
        unit_selector="root+leaf+internal",
        supervision_kind="scalar",
        label_source="dataset_labels",
        labeler_kind="precomputed",
        coverage_label=coverage_label,
        metadata=merged_metadata,
    )


def supervision_ref_from_treepo_supervision_spec(
    spec: Mapping[str, Any] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SupervisionRef | None:
    data = dict(spec or {})
    if not data:
        return None
    unit_selector = str(data.get("unit_selector", "") or "")
    topology_scope = "tree" if unit_selector not in {"", "root"} else "document"
    return SupervisionRef(
        topology_scope=topology_scope,
        unit_selector=unit_selector,
        supervision_kind=str(data.get("supervision_kind", "") or ""),
        label_source=str(data.get("mode", "") or ""),
        labeler_kind=str(data.get("labeler_kind", "") or ""),
        doc_sample_probability=_safe_float(data.get("doc_sample_probability"), None),
        unit_sampling_probability=_safe_float(data.get("unit_sampling_probability"), None),
        sampling_strategy=str(data.get("sampling_strategy", "") or ""),
        max_units=_safe_int(data.get("max_units"), None),
        coverage_label=str(data.get("coverage_label", "") or ""),
        metadata=dict(metadata or {}),
    )


def control_ref_from_treepo_local_law_config(
    config: Mapping[str, Any] | None,
    *,
    source_kind: str = "verifier",
    metadata: Mapping[str, Any] | None = None,
) -> ControlRef | None:
    data = dict(config or {})
    if not data:
        return None
    law_ids: List[str] = []
    if bool(data.get("enable_l1", False)):
        law_ids.append("L1")
    if bool(data.get("enable_l2", False)):
        law_ids.append("L2")
    if bool(data.get("enable_l3", False)):
        law_ids.append("L3")
    enabled = bool(law_ids)
    return ControlRef(
        control_family="tree_local_law",
        law_ids=tuple(law_ids),
        applies_to="tree_nodes",
        enabled=enabled,
        source_kind=str(source_kind or ""),
        sample_budget=_safe_int(data.get("sample_budget"), None),
        sampling_probability=_safe_float(data.get("sampling_probability"), None),
        sampling_strategy=str(data.get("sampling_strategy", "") or ""),
        threshold=_safe_float(data.get("discrepancy_threshold"), None),
        metadata=dict(metadata or {}),
    )


def control_ref_from_treepo_audit_config(
    config: Mapping[str, Any] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ControlRef | None:
    data = dict(config or {})
    if not data:
        return None
    law_ids: List[str] = []
    if bool(data.get("audit_leaves", data.get("enable_l1", False))):
        law_ids.append("L1")
    if bool(data.get("audit_internal", data.get("enable_l2", False))):
        law_ids.append("L2")
    if bool(data.get("audit_idempotence", data.get("enable_l3", False))):
        law_ids.append("L3")
    return ControlRef(
        control_family="tree_audit",
        law_ids=tuple(law_ids),
        applies_to="tree_nodes",
        enabled=bool(law_ids),
        source_kind="audit",
        sample_budget=_safe_int(data.get("sample_budget"), None),
        sampling_probability=_safe_float(data.get("sampling_probability"), None),
        sampling_strategy=str(data.get("sampling_strategy", "") or ""),
        threshold=_safe_float(data.get("discrepancy_threshold"), None),
        metadata=dict(metadata or {}),
    )


def control_ref_from_ctreepo_local_law_config(
    config: Mapping[str, Any] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ControlRef | None:
    data = dict(config or {})
    if not data:
        return None
    law_ids: List[str] = []
    if (_safe_float(data.get("leaf_audit_weight"), 0.0) or 0.0) > 0.0:
        law_ids.append("L1")
    if (_safe_float(data.get("merge_audit_weight"), 0.0) or 0.0) > 0.0:
        law_ids.append("L2")
    source_kind = str(data.get("label_source_kind", data.get("source_kind", "")) or "")
    return ControlRef(
        control_family="ctreepo_local_law",
        law_ids=tuple(law_ids),
        applies_to="leaf+internal",
        enabled=bool(law_ids),
        source_kind=source_kind,
        threshold=_safe_float(data.get("violation_threshold"), None),
        metadata=dict(metadata or {}),
    )


def result_rows_from_scalar_metrics(
    *,
    base_row: ResultRow,
    metrics: Mapping[str, Any],
    allowed_keys: Iterable[str] | None = None,
    artifact_refs: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    allowed = set(str(item) for item in list(allowed_keys or ()))
    for key, value in dict(metrics or {}).items():
        if allowed and str(key) not in allowed:
            continue
        if isinstance(value, bool):
            metric_value: Any = bool(value)
        elif isinstance(value, (int, float, str)):
            metric_value = value
        else:
            continue
        rows.append(
            ResultRow(
                experiment_id=base_row.experiment_id,
                phase=base_row.phase,
                benchmark_ref=base_row.benchmark_ref,
                method_ref=base_row.method_ref,
                split=base_row.split,
                seed=base_row.seed,
                train_docs=base_row.train_docs,
                supervision_ref=base_row.supervision_ref,
                control_ref=base_row.control_ref,
                metric_name=str(key),
                metric_value=metric_value,
                artifact_refs=tuple(str(item) for item in artifact_refs) or base_row.artifact_refs,
                metadata={**dict(base_row.metadata), **dict(metadata or {})},
            )
        )
    return rows


def load_result_rows(path: str | Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    data_path = Path(path)
    if not data_path.exists():
        return rows
    for raw_line in data_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(ResultRow.from_dict(dict(json.loads(line))))
    return rows
