"""Manifest-driven performance harness for micro/meso/macro benchmark layers."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


SUPPORTED_OPS = {">=", ">", "<=", "<", "==", "!="}
SUPPORTED_SEVERITIES = {"error", "warn"}
SUPPORTED_LAYERS = {"micro", "meso", "macro"}
SUPPORTED_EXPECTED_OUTCOMES = {"pass", "fail"}
SUPPORTED_FAILURE_MODES = {"any", "command", "regression"}


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + str(key) + "}"


@dataclass(frozen=True)
class RegressionRule:
    metric: str
    op: str
    threshold: float
    severity: str = "error"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    layer: str
    command: str
    description: Optional[str] = None
    timeout_seconds: int = 1800
    workdir: str = "."
    env: Dict[str, str] = field(default_factory=dict)
    metrics_file: Optional[str] = None
    metrics: Dict[str, str] = field(default_factory=dict)
    regression_rules: List[RegressionRule] = field(default_factory=list)
    expected_outcome: str = "pass"
    expected_failure_modes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Profile:
    name: str
    include_layers: List[str]
    include_scenarios: List[str] = field(default_factory=list)
    exclude_scenarios: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PerfManifest:
    version: int
    defaults: Dict[str, Any]
    profiles: Dict[str, Profile]
    scenarios: List[Scenario]


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_layer(value: Any) -> str:
    layer = str(value or "").strip().lower()
    if layer not in SUPPORTED_LAYERS:
        raise ValueError(
            f"Unsupported layer {value!r}; expected one of {sorted(SUPPORTED_LAYERS)}"
        )
    return layer


def _safe_format(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(_SafeFormatDict(context))
    if isinstance(value, list):
        return [_safe_format(v, context) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_format(v, context) for k, v in value.items()}
    return value


def _matrix_values(axis_name: str, payload: Any) -> List[Any]:
    values = payload
    if isinstance(payload, Mapping):
        values = payload.get("values")
    if not isinstance(values, list):
        raise ValueError(f"Scenario matrix axis {axis_name!r} must be a list (or mapping with `values`).")
    if not values:
        raise ValueError(f"Scenario matrix axis {axis_name!r} has no values.")
    return list(values)


def _expand_matrix_contexts(matrix_raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    axes: List[tuple[str, List[Any]]] = []
    for axis_name, axis_payload in matrix_raw.items():
        name = str(axis_name).strip()
        if not name:
            raise ValueError("Scenario matrix axis name cannot be empty.")
        axes.append((name, _matrix_values(name, axis_payload)))

    contexts: List[Dict[str, Any]] = []
    axis_names = [axis[0] for axis in axes]
    axis_values = [axis[1] for axis in axes]
    for combo_idx, combo in enumerate(itertools.product(*axis_values), start=0):
        ctx: Dict[str, Any] = {"matrix_index": int(combo_idx)}
        for axis_i, axis_name in enumerate(axis_names):
            value = combo[axis_i]
            ctx[axis_name] = value
            value_index = 0
            for candidate_idx, candidate_value in enumerate(axis_values[axis_i]):
                if candidate_value == value:
                    value_index = int(candidate_idx)
                    break
            ctx[f"{axis_name}_index"] = value_index
            if isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    ctx[str(nested_key)] = nested_value
        contexts.append(ctx)
    return contexts


def _expand_scenarios_raw(scenarios_raw: List[Any]) -> List[Mapping[str, Any]]:
    expanded: List[Mapping[str, Any]] = []
    for raw in scenarios_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("Each scenario entry must be a mapping.")
        matrix_raw = raw.get("matrix")
        if matrix_raw is None:
            expanded.append(raw)
            continue
        if not isinstance(matrix_raw, Mapping):
            raise ValueError("Scenario `matrix` must be a mapping.")

        raw_without_matrix = {str(k): v for k, v in raw.items() if str(k) != "matrix"}
        base_id = str(raw_without_matrix.get("id", "")).strip()
        if not base_id:
            raise ValueError("Matrix scenario is missing required field 'id'.")

        for ctx in _expand_matrix_contexts(matrix_raw):
            ctx_with_base = dict(ctx)
            ctx_with_base["scenario_base_id"] = base_id
            rendered_id = str(_safe_format(base_id, ctx_with_base)).strip()
            if not rendered_id:
                raise ValueError("Expanded matrix scenario generated empty id.")
            final_ctx = dict(ctx_with_base)
            final_ctx["scenario_id"] = rendered_id
            expanded_payload = _safe_format(raw_without_matrix, final_ctx)
            if not isinstance(expanded_payload, Mapping):
                raise ValueError("Expanded matrix scenario payload is invalid.")
            expanded.append(dict(expanded_payload))
    return expanded


def _parse_regression_rules(raw: Any) -> List[RegressionRule]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Scenario regression must be a list of rules.")

    rules: List[RegressionRule] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Regression rule #{idx} must be a mapping.")
        metric = str(entry.get("metric", "")).strip()
        op = str(entry.get("op", "")).strip()
        severity = str(entry.get("severity", "error")).strip().lower()
        threshold = entry.get("threshold")

        if not metric:
            raise ValueError(f"Regression rule #{idx} is missing 'metric'.")
        if op not in SUPPORTED_OPS:
            raise ValueError(
                f"Regression rule #{idx} has unsupported op {op!r}; expected one of {sorted(SUPPORTED_OPS)}"
            )
        if severity not in SUPPORTED_SEVERITIES:
            raise ValueError(
                f"Regression rule #{idx} has unsupported severity {severity!r}; "
                f"expected one of {sorted(SUPPORTED_SEVERITIES)}"
            )
        try:
            threshold_f = float(threshold)
        except (TypeError, ValueError):
            raise ValueError(f"Regression rule #{idx} has non-numeric threshold {threshold!r}") from None

        rules.append(
            RegressionRule(
                metric=metric,
                op=op,
                threshold=threshold_f,
                severity=severity,
            )
        )
    return rules


def _parse_expected(raw: Any) -> tuple[str, List[str]]:
    if raw is None:
        return "pass", []

    outcome_raw: Any = raw
    modes_raw: Any = None
    if isinstance(raw, bool):
        outcome_raw = "fail" if raw else "pass"
    elif isinstance(raw, str):
        outcome_raw = raw
    elif isinstance(raw, Mapping):
        if "outcome" in raw:
            outcome_raw = raw.get("outcome")
        elif "status" in raw:
            outcome_raw = raw.get("status")
        elif isinstance(raw.get("fail"), bool):
            outcome_raw = "fail" if bool(raw.get("fail")) else "pass"
        else:
            outcome_raw = "pass"
        modes_raw = raw.get("failure_modes", raw.get("fail_modes"))
    else:
        raise ValueError("Scenario expected must be bool, str, or mapping.")

    outcome = str(outcome_raw or "pass").strip().lower()
    if outcome not in SUPPORTED_EXPECTED_OUTCOMES:
        raise ValueError(
            f"Scenario expected outcome {outcome!r} is unsupported; "
            f"expected one of {sorted(SUPPORTED_EXPECTED_OUTCOMES)}."
        )

    modes: List[str] = []
    if modes_raw is not None:
        if isinstance(modes_raw, str):
            modes_seq: Any = [modes_raw]
        else:
            modes_seq = modes_raw
        if not isinstance(modes_seq, list):
            raise ValueError("Scenario expected failure_modes must be a list (or string).")
        for item in modes_seq:
            mode = str(item).strip().lower()
            if mode not in SUPPORTED_FAILURE_MODES:
                raise ValueError(
                    f"Scenario expected failure mode {mode!r} is unsupported; "
                    f"expected one of {sorted(SUPPORTED_FAILURE_MODES)}."
                )
            if mode not in modes:
                modes.append(mode)

    if outcome == "pass" and modes:
        raise ValueError("Scenario expected failure_modes are only valid when expected outcome is 'fail'.")

    if outcome == "fail":
        if not modes:
            modes = ["any"]
        elif "any" in modes and len(modes) > 1:
            modes = ["any"]

    return outcome, modes


def _parse_profile(name: str, payload: Any) -> Profile:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Profile {name!r} must be a mapping.")
    include_layers = [_coerce_layer(v) for v in payload.get("include_layers", [])]
    include_scenarios = [str(v) for v in payload.get("include_scenarios", [])]
    exclude_scenarios = [str(v) for v in payload.get("exclude_scenarios", [])]
    return Profile(
        name=name,
        include_layers=include_layers,
        include_scenarios=include_scenarios,
        exclude_scenarios=exclude_scenarios,
    )


def _parse_scenario(raw: Any, defaults: Mapping[str, Any]) -> Scenario:
    if not isinstance(raw, Mapping):
        raise ValueError("Each scenario entry must be a mapping.")

    scenario_id = str(raw.get("id", "")).strip()
    if not scenario_id:
        raise ValueError("Scenario is missing required field 'id'.")
    layer = _coerce_layer(raw.get("layer"))
    command = str(raw.get("command", "")).strip()
    if not command:
        raise ValueError(f"Scenario {scenario_id!r} is missing required field 'command'.")

    timeout_seconds = _coerce_int(
        raw.get("timeout_seconds", defaults.get("timeout_seconds", 1800)),
        default=1800,
    )
    workdir = str(raw.get("workdir", defaults.get("workdir", ".")))

    env_raw = raw.get("env", {})
    if env_raw is None:
        env_raw = {}
    if not isinstance(env_raw, Mapping):
        raise ValueError(f"Scenario {scenario_id!r} has non-mapping env.")
    env = {str(k): str(v) for k, v in env_raw.items()}

    metrics_raw = raw.get("metrics", {})
    if metrics_raw is None:
        metrics_raw = {}
    if not isinstance(metrics_raw, Mapping):
        raise ValueError(f"Scenario {scenario_id!r} has non-mapping metrics.")
    metrics = {str(k): str(v) for k, v in metrics_raw.items()}
    expected_outcome, expected_failure_modes = _parse_expected(raw.get("expected"))

    return Scenario(
        scenario_id=scenario_id,
        layer=layer,
        command=command,
        description=str(raw.get("description")) if raw.get("description") else None,
        timeout_seconds=max(1, timeout_seconds),
        workdir=workdir,
        env=env,
        metrics_file=str(raw.get("metrics_file")) if raw.get("metrics_file") else None,
        metrics=metrics,
        regression_rules=_parse_regression_rules(raw.get("regression")),
        expected_outcome=expected_outcome,
        expected_failure_modes=expected_failure_modes,
    )


def load_manifest(path: Path) -> PerfManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Manifest root must be a mapping.")

    version = _coerce_int(payload.get("version", 1), default=1)
    defaults = payload.get("defaults", {}) or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("Manifest defaults must be a mapping.")

    profiles_raw = payload.get("profiles", {}) or {}
    if not isinstance(profiles_raw, Mapping):
        raise ValueError("Manifest profiles must be a mapping.")
    profiles: Dict[str, Profile] = {}
    for name, profile_payload in profiles_raw.items():
        profiles[str(name)] = _parse_profile(str(name), profile_payload)

    scenarios_raw = payload.get("scenarios", []) or []
    if not isinstance(scenarios_raw, list):
        raise ValueError("Manifest scenarios must be a list.")
    expanded_raw = _expand_scenarios_raw(scenarios_raw)
    scenarios = [_parse_scenario(raw, defaults) for raw in expanded_raw]

    seen_ids: set[str] = set()
    for scenario in scenarios:
        if scenario.scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario id after matrix expansion: {scenario.scenario_id!r}")
        seen_ids.add(scenario.scenario_id)

    return PerfManifest(
        version=version,
        defaults=dict(defaults),
        profiles=profiles,
        scenarios=scenarios,
    )


def select_scenarios(manifest: PerfManifest, profile: Optional[str]) -> List[Scenario]:
    if not profile:
        return list(manifest.scenarios)

    profile_key = str(profile)
    if profile_key not in manifest.profiles:
        raise ValueError(f"Profile {profile_key!r} was not found in manifest.")

    selected_profile = manifest.profiles[profile_key]
    include_layers = set(selected_profile.include_layers)
    include_ids = set(selected_profile.include_scenarios)
    exclude_ids = set(selected_profile.exclude_scenarios)

    selected: List[Scenario] = []
    for scenario in manifest.scenarios:
        if include_layers and scenario.layer not in include_layers:
            continue
        if include_ids and scenario.scenario_id not in include_ids:
            continue
        if scenario.scenario_id in exclude_ids:
            continue
        selected.append(scenario)

    return selected


def _op_eval(lhs: float, op: str, rhs: float) -> bool:
    if op == ">=":
        return lhs >= rhs
    if op == ">":
        return lhs > rhs
    if op == "<=":
        return lhs <= rhs
    if op == "<":
        return lhs < rhs
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    raise ValueError(f"Unsupported operator {op!r}")


def _resolve_path(payload: Any, dotted_path: str) -> Any:
    current = payload
    for piece in str(dotted_path).split("."):
        if isinstance(current, Mapping):
            if piece not in current:
                raise KeyError(piece)
            current = current[piece]
            continue
        if isinstance(current, list):
            idx = int(piece)
            current = current[idx]
            continue
        raise KeyError(piece)
    return current


def extract_metrics(metrics_file: Path, mapping: Mapping[str, str]) -> Dict[str, Any]:
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    extracted: Dict[str, Any] = {}
    for metric_name, path_expr in mapping.items():
        try:
            extracted[str(metric_name)] = _resolve_path(payload, str(path_expr))
        except Exception:
            extracted[str(metric_name)] = None
    return extracted


def evaluate_regressions(
    *,
    metrics: Mapping[str, Any],
    rules: Iterable[RegressionRule],
) -> List[Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    for rule in rules:
        raw = metrics.get(rule.metric)
        try:
            observed = float(raw)
            passed = _op_eval(observed, rule.op, rule.threshold)
        except Exception:
            observed = None
            passed = False

        outcomes.append(
            {
                "metric": rule.metric,
                "op": rule.op,
                "threshold": rule.threshold,
                "severity": rule.severity,
                "observed": observed,
                "passed": bool(passed),
            }
        )
    return outcomes


def evaluate_expectation(
    *,
    expected_outcome: str,
    expected_failure_modes: Iterable[str],
    command_ok: bool,
    regression_ok: bool,
) -> Dict[str, Any]:
    expected = str(expected_outcome or "pass").strip().lower()
    if expected not in SUPPORTED_EXPECTED_OUTCOMES:
        raise ValueError(
            f"Unsupported expected_outcome {expected_outcome!r}; "
            f"expected one of {sorted(SUPPORTED_EXPECTED_OUTCOMES)}"
        )

    failure_modes: List[str] = []
    if not bool(command_ok):
        failure_modes.append("command")
    if not bool(regression_ok):
        failure_modes.append("regression")
    actual_outcome = "pass" if not failure_modes else "fail"

    expected_modes = [str(mode).strip().lower() for mode in expected_failure_modes if str(mode).strip()]
    if expected == "pass":
        expectation_met = actual_outcome == "pass"
        expected_modes = []
    else:
        if not expected_modes:
            expected_modes = ["any"]
        if "any" in expected_modes and len(expected_modes) > 1:
            expected_modes = ["any"]
        if actual_outcome != "fail":
            expectation_met = False
        elif "any" in expected_modes:
            expectation_met = True
        else:
            expectation_met = any(mode in failure_modes for mode in expected_modes)

    return {
        "expected_outcome": expected,
        "expected_failure_modes": expected_modes,
        "actual_outcome": actual_outcome,
        "failure_modes": failure_modes,
        "expectation_met": bool(expectation_met),
    }


def run_scenario(
    *,
    scenario: Scenario,
    repo_root: Path,
    log_path: Path,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = (repo_root / scenario.workdir).resolve()
    env = dict(**os.environ)
    env.update(scenario.env)

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {scenario.command}\n")
        log_file.write(f"Workdir: {workdir}\n\n")
        try:
            proc = subprocess.run(
                ["bash", "-lc", scenario.command],
                cwd=str(workdir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
                timeout=int(scenario.timeout_seconds),
            )
            exit_code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            log_file.write(f"\nTimed out after {int(scenario.timeout_seconds)}s\n")
            exit_code = -124
    elapsed = float(time.time() - t0)
    finished = datetime.now(timezone.utc).isoformat()

    return {
        "started_utc": started,
        "finished_utc": finished,
        "wall_seconds": elapsed,
        "exit_code": exit_code,
        "log_path": str(log_path),
    }


def build_artifact(
    *,
    manifest_path: Path,
    profile: Optional[str],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for row in results if row.get("status") == "passed")
    failed = sum(1 for row in results if row.get("status") == "failed")
    skipped = sum(1 for row in results if row.get("status") == "skipped")
    executed = [row for row in results if row.get("status") != "skipped"]
    expected_failures = sum(1 for row in results if str(row.get("expected_outcome", "pass")).lower() == "fail")
    expected_failures_met = sum(
        1
        for row in executed
        if str(row.get("expected_outcome", "pass")).lower() == "fail" and bool(row.get("expectation_met"))
    )
    unexpected_passes = sum(
        1
        for row in executed
        if str(row.get("expected_outcome", "pass")).lower() == "fail" and not bool(row.get("expectation_met"))
    )
    unexpected_failures = sum(
        1
        for row in executed
        if str(row.get("expected_outcome", "pass")).lower() == "pass" and not bool(row.get("expectation_met", True))
    )
    def _is_expected_failure_met(row: Mapping[str, Any]) -> bool:
        return (
            str(row.get("expected_outcome", "pass")).lower() == "fail"
            and bool(row.get("expectation_met"))
        )

    regression_errors_total = sum(
        1
        for row in results
        for reg in row.get("regressions", [])
        if (not reg.get("passed")) and reg.get("severity") == "error"
    )
    regression_warnings_total = sum(
        1
        for row in results
        for reg in row.get("regressions", [])
        if (not reg.get("passed")) and reg.get("severity") == "warn"
    )
    regression_errors_unexpected = sum(
        1
        for row in results
        if not _is_expected_failure_met(row)
        for reg in row.get("regressions", [])
        if (not reg.get("passed")) and reg.get("severity") == "error"
    )
    regression_warnings_unexpected = sum(
        1
        for row in results
        if not _is_expected_failure_met(row)
        for reg in row.get("regressions", [])
        if (not reg.get("passed")) and reg.get("severity") == "warn"
    )

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "profile": profile,
        "host": socket.gethostname(),
        "summary": {
            "total_scenarios": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "expected_failures": expected_failures,
            "expected_failures_met": expected_failures_met,
            "unexpected_passes": unexpected_passes,
            "unexpected_failures": unexpected_failures,
            # Backward-compatible keys now represent unexpected regressions only.
            "regression_errors": regression_errors_unexpected,
            "regression_warnings": regression_warnings_unexpected,
            "regression_errors_unexpected": regression_errors_unexpected,
            "regression_warnings_unexpected": regression_warnings_unexpected,
            "regression_errors_total": regression_errors_total,
            "regression_warnings_total": regression_warnings_total,
        },
        "results": results,
    }


def has_regression_error(regressions: Iterable[Mapping[str, Any]]) -> bool:
    for row in regressions:
        if (not bool(row.get("passed"))) and str(row.get("severity", "")).lower() == "error":
            return True
    return False
