from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence

from treepo._research.ctreepo.sim.core.exact_utility_common import lean_theorem_refs


ExpectationStatus = Literal["pass", "warn", "fail", "not_applicable"]


@dataclass(frozen=True)
class UtilityTransportRow:
    lane: str
    oracle_profile: str
    slice_name: str
    objective_family: str
    structural_arm: str
    train_docs: int
    seed: int
    doc_scale_tokens: float
    fixed_leaf_tokens: int
    leaves_per_doc: float
    leaf_label_coverage: float
    internal_label_coverage: float
    root_query_rate: float
    pairwise_prefs_per_doc: float
    group_pref_groups_per_doc: float
    group_size: int
    ppo_rollouts_per_doc: float
    total_oracle_calls_estimate: float
    utility_regret: float
    exact_state_accuracy: float
    state_l1: float
    root_mae: float
    merge_mae: float
    local_oracle_coverage: float
    tree_relevance: str
    lean_theorems: List[str]
    source_path: str


@dataclass(frozen=True)
class UtilityTransportFinding:
    kind: str
    status: ExpectationStatus
    lane: str
    oracle_profile: str
    objective_family: str
    slice_name: str
    title: str
    observed: Dict[str, object]
    supporting_rows: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "status": self.status,
            "lane": self.lane,
            "oracle_profile": self.oracle_profile,
            "objective_family": self.objective_family,
            "slice_name": self.slice_name,
            "title": self.title,
            "observed": dict(self.observed),
            "supporting_rows": list(self.supporting_rows),
        }


@dataclass(frozen=True)
class UtilityTransportReport:
    rows: List[UtilityTransportRow]
    findings: List[UtilityTransportFinding]

    def to_dict(self) -> Dict[str, object]:
        return {
            "rows": [asdict(r) for r in self.rows],
            "findings": [f.to_dict() for f in self.findings],
            "summary": {
                "n_rows": len(self.rows),
                "n_pass": sum(1 for f in self.findings if f.status == "pass"),
                "n_warn": sum(1 for f in self.findings if f.status == "warn"),
                "n_fail": sum(1 for f in self.findings if f.status == "fail"),
                "n_not_applicable": sum(1 for f in self.findings if f.status == "not_applicable"),
            },
        }


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_utility_transport_rows(root: Path) -> List[UtilityTransportRow]:
    rows: List[UtilityTransportRow] = []
    known_slices = {
        "support_curves_local",
        "support_curves_preferences",
        "objective_family_high_support",
        "structural_matrix",
        "structural_controls_anchor",
    }
    for path in sorted(root.rglob("*.json")):
        payload = _load_json(path)
        if "lane" not in payload or "metrics" not in payload:
            continue
        metrics = dict(payload.get("metrics", {}) or {})
        root_metrics = dict(metrics.get("root", {}) or {})
        budget = dict(payload.get("budget", {}) or {})
        config = dict(payload.get("config", {}) or {})
        metadata = dict(payload.get("metadata", {}) or {})
        lane = str(payload.get("lane", ""))
        oracle_profile = str(payload.get("oracle_profile", ""))
        slice_name = next((part for part in path.parts if part in known_slices), "")
        theorem_refs = [str(x) for x in list(metadata.get("lean_theorems", []) or [])]
        if not theorem_refs and lane in {"markov", "nonseparable", "boundary_topic"}:
            theorem_refs = lean_theorem_refs(lane, oracle_profile)
        rows.append(
            UtilityTransportRow(
                lane=lane,
                oracle_profile=oracle_profile,
                slice_name=slice_name,
                objective_family=str(payload.get("objective_family", "")),
                structural_arm=str(payload.get("structural_arm", "")),
                train_docs=int(config.get("train_docs", budget.get("train_docs", 0))),
                seed=int(config.get("seed", 0)),
                doc_scale_tokens=float(budget.get("doc_scale_tokens", float("nan"))),
                fixed_leaf_tokens=int(budget.get("fixed_leaf_tokens", config.get("fixed_leaf_tokens", 0))),
                leaves_per_doc=float(budget.get("leaves_per_doc", float("nan"))),
                leaf_label_coverage=float(budget.get("leaf_label_coverage", float("nan"))),
                internal_label_coverage=float(budget.get("internal_label_coverage", float("nan"))),
                root_query_rate=float(budget.get("root_query_rate", float("nan"))),
                pairwise_prefs_per_doc=float(budget.get("pairwise_prefs_per_doc", float("nan"))),
                group_pref_groups_per_doc=float(budget.get("group_pref_groups_per_doc", float("nan"))),
                group_size=int(budget.get("group_size", 0)),
                ppo_rollouts_per_doc=float(budget.get("ppo_rollouts_per_doc", float("nan"))),
                total_oracle_calls_estimate=float(budget.get("total_oracle_calls_estimate", float("nan"))),
                utility_regret=float(metrics.get("utility_regret", root_metrics.get("utility_regret", float("nan")))),
                exact_state_accuracy=float(root_metrics.get("exact_state_accuracy", float("nan"))),
                state_l1=float(root_metrics.get("state_l1", float("nan"))),
                root_mae=float(metrics.get("root_mae", float("nan"))),
                merge_mae=float(metrics.get("merge_mae", float("nan"))),
                local_oracle_coverage=float(budget.get("local_oracle_coverage", float("nan"))),
                tree_relevance=str(metadata.get("tree_relevance", "")),
                lean_theorems=theorem_refs,
                source_path=str(path),
            )
        )
    return rows


def _median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    vals.sort()
    n = len(vals)
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def _group(rows: Iterable[UtilityTransportRow]) -> Dict[tuple[str, str, str, str, str], List[UtilityTransportRow]]:
    out: Dict[tuple[str, str, str, str, str], List[UtilityTransportRow]] = {}
    for row in rows:
        key = (row.lane, row.oracle_profile, row.objective_family, row.slice_name, row.structural_arm)
        out.setdefault(key, []).append(row)
    return out


def _add_finding(
    findings: List[UtilityTransportFinding],
    *,
    kind: str,
    status: ExpectationStatus,
    lane: str,
    oracle_profile: str,
    objective_family: str,
    slice_name: str,
    title: str,
    observed: Dict[str, object],
    rows: Sequence[UtilityTransportRow],
) -> None:
    findings.append(
        UtilityTransportFinding(
            kind=kind,
            status=status,
            lane=lane,
            oracle_profile=oracle_profile,
            objective_family=objective_family,
            slice_name=slice_name,
            title=title,
            observed=observed,
            supporting_rows=[asdict(r) for r in rows],
        )
    )


def _slice_is_structural(slice_name: str) -> bool:
    return slice_name in {"structural_matrix", "structural_controls_anchor", "objective_family_high_support"}


def _flat_comparison_is_fair(
    *,
    tree_rows: Sequence[UtilityTransportRow],
    flat_rows: Sequence[UtilityTransportRow],
    objective_family: str,
) -> bool:
    if not tree_rows or not flat_rows:
        return False
    flat_arm = str(flat_rows[0].structural_arm)
    if objective_family in {"dpo", "grpo", "ppo", "supervised_root"}:
        return True
    if objective_family.startswith("hybrid_supervised_plus_") or objective_family == "supervised_state":
        if flat_arm == "flat_equal_info":
            return False
        tree_local = _median([r.local_oracle_coverage for r in tree_rows])
        flat_local = _median([r.local_oracle_coverage for r in flat_rows])
        flat_root = _median([r.root_query_rate for r in flat_rows])
        return bool(tree_local > 1e-9 and flat_local > 1e-9) or bool(tree_local <= 1e-9 and (flat_local > 1e-9 or flat_root > 1e-9))
    return True


def _prefer_flat_rows(
    *,
    objective_family: str,
    tree_rows: Sequence[UtilityTransportRow],
    flat_rows: Sequence[UtilityTransportRow],
    flat_span_rows: Sequence[UtilityTransportRow],
) -> tuple[Sequence[UtilityTransportRow], str]:
    if objective_family.startswith("hybrid_supervised_plus_") or objective_family == "supervised_state":
        tree_local = _median([r.local_oracle_coverage for r in tree_rows])
        if tree_local > 1e-9 and flat_span_rows:
            return flat_span_rows, "flat_span_equal_info"
    if flat_rows:
        return flat_rows, "flat_equal_info"
    if flat_span_rows:
        return flat_span_rows, "flat_span_equal_info"
    return [], "none"


def build_utility_transport_report(root: Path) -> UtilityTransportReport:
    rows = load_utility_transport_rows(root)
    findings: List[UtilityTransportFinding] = []
    grouped = _group(rows)
    scenario_keys = sorted({(r.lane, r.oracle_profile, r.objective_family, r.slice_name) for r in rows})
    for lane, oracle_profile, objective_family, slice_name in scenario_keys:
        exact_rows = grouped.get((lane, oracle_profile, objective_family, slice_name, "oracle_exact"), [])
        exact_tree_rows = grouped.get((lane, oracle_profile, objective_family, slice_name, "tree_exact_supported"), [])
        tree_rows = grouped.get((lane, oracle_profile, objective_family, slice_name, "tree_neural_supported"), [])
        flat_rows = grouped.get((lane, oracle_profile, objective_family, slice_name, "flat_equal_info"), [])
        flat_span_rows = grouped.get((lane, oracle_profile, objective_family, slice_name, "flat_span_equal_info"), [])
        under_rows = grouped.get((lane, oracle_profile, objective_family, slice_name, "tree_undersupported"), [])
        one_leaf_rows = grouped.get((lane, oracle_profile, objective_family, slice_name, "one_leaf_control"), [])
        exact_regret = _median([r.utility_regret for r in exact_rows])
        if exact_rows:
            _add_finding(
                findings,
                kind="exact_zero_error",
                status="pass" if exact_regret <= 1e-8 else "fail",
                lane=lane,
                oracle_profile=oracle_profile,
                objective_family=objective_family,
                slice_name=slice_name,
                title="Oracle exact arm should have zero utility regret",
                observed={"oracle_exact_utility_regret": exact_regret},
                rows=exact_rows,
            )
        elif exact_tree_rows:
            exact_tree_regret = _median([r.utility_regret for r in exact_tree_rows])
            _add_finding(
                findings,
                kind="exact_zero_error",
                status="pass" if exact_tree_regret <= 1e-8 else "fail",
                lane=lane,
                oracle_profile=oracle_profile,
                objective_family=objective_family,
                slice_name=slice_name,
                title="Tree exact supported arm should have zero utility regret",
                observed={"tree_exact_supported_utility_regret": exact_tree_regret},
                rows=exact_tree_rows,
            )
        else:
            _add_finding(
                findings,
                kind="exact_zero_error",
                status="not_applicable",
                lane=lane,
                oracle_profile=oracle_profile,
                objective_family=objective_family,
                slice_name=slice_name,
                title="Oracle exact arm should have zero utility regret",
                observed={"oracle_exact_utility_regret": None},
                rows=[],
            )
        chosen_flat_rows, chosen_flat_arm = _prefer_flat_rows(
            objective_family=objective_family,
            tree_rows=tree_rows,
            flat_rows=flat_rows,
            flat_span_rows=flat_span_rows,
        )
        if _slice_is_structural(slice_name) and tree_rows and chosen_flat_rows:
            tree_regret = _median([r.utility_regret for r in tree_rows])
            flat_regret = _median([r.utility_regret for r in chosen_flat_rows])
            delta = float(flat_regret - tree_regret)
            tree_relevance = tree_rows[0].tree_relevance
            fair = _flat_comparison_is_fair(tree_rows=tree_rows, flat_rows=chosen_flat_rows, objective_family=objective_family)
            if not fair:
                status = "not_applicable"
                title = "Flat baseline comparison is not equal-information for this support type"
            elif tree_relevance == "tree_relevant":
                status = "pass" if delta >= 0.05 else ("warn" if delta >= 0.0 else "fail")
                title = "Tree-aware model should beat flat baseline in tree-relevant setting"
            else:
                status = "pass" if abs(delta) <= 0.05 else ("warn" if abs(delta) <= 0.10 else "fail")
                title = "Tree advantage should shrink in tree-irrelevant control"
            _add_finding(
                findings,
                kind="tree_relevance",
                status=status,
                lane=lane,
                oracle_profile=oracle_profile,
                objective_family=objective_family,
                slice_name=slice_name,
                title=title,
                observed={
                    "tree_regret": tree_regret,
                    "flat_regret": flat_regret,
                    "delta_flat_minus_tree": delta,
                    "fair_flat_comparison": fair,
                    "flat_arm": chosen_flat_arm,
                },
                rows=[*tree_rows, *chosen_flat_rows],
            )
        if _slice_is_structural(slice_name) and tree_rows and under_rows:
            tree_regret = _median([r.utility_regret for r in tree_rows])
            under_regret = _median([r.utility_regret for r in under_rows])
            delta = float(under_regret - tree_regret)
            tree_relevance = tree_rows[0].tree_relevance
            if tree_relevance == "tree_relevant":
                status = "pass" if delta >= 0.05 else ("warn" if delta >= 0.0 else "fail")
                title = "Supported tree model should beat undersupported variant"
            else:
                status = "warn" if abs(delta) <= 0.05 else "not_applicable"
                title = "Supported vs undersupported difference in tree-irrelevant control"
            _add_finding(
                findings,
                kind="supported_vs_undersupported",
                status=status,
                lane=lane,
                oracle_profile=oracle_profile,
                objective_family=objective_family,
                slice_name=slice_name,
                title=title,
                observed={"tree_regret": tree_regret, "undersupported_regret": under_regret, "delta_under_minus_tree": delta},
                rows=[*tree_rows, *under_rows],
            )
        if _slice_is_structural(slice_name) and tree_rows and one_leaf_rows:
            tree_regret = _median([r.utility_regret for r in tree_rows])
            one_leaf_regret = _median([r.utility_regret for r in one_leaf_rows])
            _add_finding(
                findings,
                kind="one_leaf_control",
                status="pass" if one_leaf_regret >= tree_regret else "warn",
                lane=lane,
                oracle_profile=oracle_profile,
                objective_family=objective_family,
                slice_name=slice_name,
                title="One-leaf control should not outperform the full tree by a large margin",
                observed={"tree_regret": tree_regret, "one_leaf_regret": one_leaf_regret},
                rows=[*tree_rows, *one_leaf_rows],
            )
    return UtilityTransportReport(rows=rows, findings=findings)


__all__ = [
    "ExpectationStatus",
    "UtilityTransportFinding",
    "UtilityTransportReport",
    "UtilityTransportRow",
    "build_utility_transport_report",
    "load_utility_transport_rows",
]
