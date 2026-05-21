"""
Optimizer audit schemas and classification helpers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from treepo._research.ctreepo.sim.util import safe_float, safe_int


COMPILE_STATUS_COMPLETED = "completed"
COMPILE_STATUS_SKIPPED = "skipped"
COMPILE_STATUS_FALLBACK = "fallback"
COMPILE_STATUS_NOOP = "noop"
COMPILE_STATUS_FAILED = "failed"

METRIC_DIRECTION_HIGHER_IS_BETTER = "higher_is_better"
METRIC_DIRECTION_LOWER_IS_BETTER = "lower_is_better"

CLASS_WORKS = "works"
CLASS_UNSTABLE_SEARCH = "unstable_search"
CLASS_DATA_LIMITED = "data_limited"
CLASS_IMPLEMENTATION_FALLBACK = "implementation_fallback"
CLASS_OBJECTIVE_MISMATCH = "objective_mismatch"
CLASS_RUNTIME_FAILURE = "runtime_failure"
CLASS_FORCED_CONTROL = "forced_control"


_safe_float = safe_float
_safe_int = safe_int


def _median(values: Sequence[float]) -> float:
    usable = sorted(float(v) for v in values if str(v) != "nan")
    usable = [v for v in usable if v == v]
    if not usable:
        return float("nan")
    n = len(usable)
    mid = n // 2
    if n % 2 == 1:
        return float(usable[mid])
    return float((usable[mid - 1] + usable[mid]) / 2.0)


def metric_gain(metric_before: Any, metric_after: Any, metric_direction: str) -> float:
    before = _safe_float(metric_before)
    after = _safe_float(metric_after)
    if before != before or after != after:
        return float("nan")
    if str(metric_direction).strip().lower() == METRIC_DIRECTION_LOWER_IS_BETTER:
        return float(before - after)
    return float(after - before)


def _config_value(config: Any, key: str, default: int) -> int:
    if config is None:
        return int(default)
    if isinstance(config, Mapping):
        return _safe_int(config.get(key, default), default)
    return _safe_int(getattr(config, key, default), default)


def dataset_regime_label(dataset_size: Any, config: Any) -> str:
    size = max(0, _safe_int(dataset_size, 0))
    bootstrap_threshold = _config_value(config, "bootstrap_threshold", 10)
    random_search_threshold = _config_value(config, "random_search_threshold", 120)
    mipro_threshold = _config_value(config, "mipro_threshold", 200)
    if size <= bootstrap_threshold:
        return f"<=bootstrap_threshold({bootstrap_threshold})"
    if size <= random_search_threshold:
        return (
            f"(bootstrap_threshold({bootstrap_threshold}),"
            f"random_search_threshold({random_search_threshold})]"
        )
    if size <= mipro_threshold:
        return (
            f"(random_search_threshold({random_search_threshold}),"
            f"mipro_threshold({mipro_threshold})]"
        )
    return f">mipro_threshold({mipro_threshold})"


def _active_mutation(flags: Any) -> bool:
    if flags is None:
        return False
    if isinstance(flags, Mapping):
        return any(_active_mutation(value) for value in flags.values())
    if isinstance(flags, (list, tuple, set)):
        return any(_active_mutation(value) for value in flags)
    if isinstance(flags, bool):
        return bool(flags)
    if isinstance(flags, (int, float)):
        return bool(flags)
    if isinstance(flags, str):
        text = flags.strip().lower()
        return text not in {"", "none", "false", "0", "inactive"}
    return True


@dataclass
class OptimizerRunRecord:
    optimizer_requested: str
    optimizer_used: str
    component: str
    dataset_size: int
    dataset_regime: str
    budget_mode: str
    seed: int
    iteration: int = 1
    compile_status: str = COMPILE_STATUS_COMPLETED
    skip_reason: str = "none"
    fallback_reason: str = "none"
    metric_direction: str = METRIC_DIRECTION_HIGHER_IS_BETTER
    metric_before: float = float("nan")
    metric_after: float = float("nan")
    heldout_gain: float = float("nan")
    train_gain: float = float("nan")
    input_mutation_flags: Dict[str, Any] = field(default_factory=dict)
    exception_summary: str | None = None
    comparison_control_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerCellSummary:
    optimizer_requested: str
    component: str
    dataset_regime: str
    budget_mode: str
    n_runs: int
    classification: str
    dominant_optimizer_used: str
    operational_success_rate: float
    gain_rate: float
    median_heldout_gain: float
    median_train_gain: float
    skipped_rate: float
    failed_rate: float
    fallback_rate: float
    noop_rate: float
    mutation_active_rate: float
    comparison_control_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_optimizer_runs(
    runs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        key = (
            str(row.get("optimizer_requested", "")),
            str(row.get("component", "")),
            str(row.get("dataset_regime", "")),
            str(row.get("budget_mode", "")),
        )
        grouped[key].append(row)

    summaries: List[OptimizerCellSummary] = []
    for (optimizer_requested, component, dataset_regime, budget_mode), group in sorted(grouped.items()):
        statuses = Counter(str(row.get("compile_status", "")) for row in group)
        used = Counter(str(row.get("optimizer_used", "")) for row in group)
        total = max(1, len(group))
        completed_or_fallback = statuses[COMPILE_STATUS_COMPLETED] + statuses[COMPILE_STATUS_FALLBACK]
        operational_success_rate = float(completed_or_fallback) / float(total)
        skipped_rate = float(statuses[COMPILE_STATUS_SKIPPED]) / float(total)
        failed_rate = float(statuses[COMPILE_STATUS_FAILED]) / float(total)
        fallback_rate = float(statuses[COMPILE_STATUS_FALLBACK]) / float(total)
        noop_rate = float(statuses[COMPILE_STATUS_NOOP]) / float(total)

        heldout_gains = [
            _safe_float(row.get("heldout_gain"))
            for row in group
            if _safe_float(row.get("heldout_gain")) == _safe_float(row.get("heldout_gain"))
        ]
        train_gains = [
            _safe_float(row.get("train_gain"))
            for row in group
            if _safe_float(row.get("train_gain")) == _safe_float(row.get("train_gain"))
        ]
        gain_rate = (
            float(sum(1 for value in heldout_gains if value > 0.0)) / float(len(heldout_gains))
            if heldout_gains
            else 0.0
        )
        median_heldout_gain = _median(heldout_gains)
        median_train_gain = _median(train_gains)
        mutation_active_rate = float(
            sum(1 for row in group if _active_mutation(row.get("input_mutation_flags")))
        ) / float(total)
        comparison_control_flag = any(bool(row.get("comparison_control_flag", False)) for row in group)

        classification = CLASS_UNSTABLE_SEARCH
        if comparison_control_flag:
            classification = CLASS_FORCED_CONTROL
        elif skipped_rate > 0.5:
            classification = CLASS_DATA_LIMITED
        elif (fallback_rate + noop_rate) > 0.1:
            classification = CLASS_IMPLEMENTATION_FALLBACK
        elif failed_rate > 0.1:
            classification = CLASS_RUNTIME_FAILURE
        elif (
            median_train_gain == median_train_gain
            and median_heldout_gain == median_heldout_gain
            and median_train_gain > 0.0
            and median_heldout_gain <= 0.0
            and mutation_active_rate > 0.0
        ):
            classification = CLASS_OBJECTIVE_MISMATCH
        elif (
            operational_success_rate >= 0.9
            and gain_rate >= 0.7
            and median_heldout_gain == median_heldout_gain
            and median_heldout_gain > 0.0
        ):
            classification = CLASS_WORKS
        elif operational_success_rate >= 0.9:
            classification = CLASS_UNSTABLE_SEARCH

        summaries.append(
            OptimizerCellSummary(
                optimizer_requested=optimizer_requested,
                component=component,
                dataset_regime=dataset_regime,
                budget_mode=budget_mode,
                n_runs=len(group),
                classification=classification,
                dominant_optimizer_used=used.most_common(1)[0][0] if used else "",
                operational_success_rate=float(operational_success_rate),
                gain_rate=float(gain_rate),
                median_heldout_gain=float(median_heldout_gain),
                median_train_gain=float(median_train_gain),
                skipped_rate=float(skipped_rate),
                failed_rate=float(failed_rate),
                fallback_rate=float(fallback_rate),
                noop_rate=float(noop_rate),
                mutation_active_rate=float(mutation_active_rate),
                comparison_control_flag=bool(comparison_control_flag),
            )
        )
    return [summary.to_dict() for summary in summaries]


__all__ = [
    "OptimizerRunRecord",
    "OptimizerCellSummary",
    "dataset_regime_label",
    "metric_gain",
    "summarize_optimizer_runs",
    "COMPILE_STATUS_COMPLETED",
    "COMPILE_STATUS_SKIPPED",
    "COMPILE_STATUS_FALLBACK",
    "COMPILE_STATUS_NOOP",
    "COMPILE_STATUS_FAILED",
    "METRIC_DIRECTION_HIGHER_IS_BETTER",
    "METRIC_DIRECTION_LOWER_IS_BETTER",
    "CLASS_WORKS",
    "CLASS_UNSTABLE_SEARCH",
    "CLASS_DATA_LIMITED",
    "CLASS_IMPLEMENTATION_FALLBACK",
    "CLASS_OBJECTIVE_MISMATCH",
    "CLASS_RUNTIME_FAILURE",
    "CLASS_FORCED_CONTROL",
]
