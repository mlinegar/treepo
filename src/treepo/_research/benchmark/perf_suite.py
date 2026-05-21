"""
Scenario-driven performance suite runner and comparator.

This module standardizes performance testing across:
- micro benchmarks
- component throughput benchmarks
- integration/e2e pipeline runs

A suite config is provided via YAML/JSON and executed case-by-case.
Each case can define artifact extractors and metric regression rules.
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import yaml
except Exception:  # pragma: no cover - guarded runtime dependency
    yaml = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + str(key) + "}"


def _safe_format(template: Any, context: Dict[str, Any]) -> Any:
    if isinstance(template, str):
        return template.format_map(_SafeFormatDict(context))
    if isinstance(template, list):
        return [_safe_format(item, context) for item in template]
    if isinstance(template, dict):
        return {str(k): _safe_format(v, context) for k, v in template.items()}
    return template


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    return text.strip("_.-") or "case"


def _deep_get(obj: Any, dotted_path: str) -> Any:
    cur = obj
    for part in str(dotted_path).split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur.get(part)
            continue
        if isinstance(cur, list):
            try:
                idx = int(part)
            except (TypeError, ValueError):
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        return None
    return cur


def _is_number(value: Any) -> bool:
    try:
        _ = float(value)
        return True
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> Optional[float]:
    if not _is_number(value):
        return None
    return float(value)


def _aggregate(values: Sequence[float], mode: str = "median") -> Optional[float]:
    if not values:
        return None
    mode_n = str(mode or "median").strip().lower()
    if mode_n == "mean":
        return float(statistics.fmean(values))
    if mode_n == "min":
        return float(min(values))
    if mode_n == "max":
        return float(max(values))
    return float(statistics.median(values))


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * float(q)
    lo = int(rank)
    hi = min(len(ordered) - 1, lo + 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def load_suite_config(path: Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Suite config not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for YAML suite files. Install `pyyaml` or use JSON."
            )
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        # Try YAML first, then JSON.
        payload = cfg_path.read_text(encoding="utf-8")
        if yaml is not None:
            try:
                data = yaml.safe_load(payload)
            except Exception:
                data = json.loads(payload)
        else:
            data = json.loads(payload)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid suite config (root must be object): {cfg_path}")

    data.setdefault("version", 1)
    data.setdefault("name", cfg_path.stem)
    data.setdefault("output_root", "outputs/performance_suite")
    data.setdefault("defaults", {})
    data.setdefault("cases", [])
    if not isinstance(data.get("cases"), list):
        raise ValueError(f"Invalid suite config: `cases` must be a list ({cfg_path})")
    return data


def _select_cases(
    cfg: Dict[str, Any],
    *,
    include_layers: Optional[Sequence[str]] = None,
    include_case_ids: Optional[Sequence[str]] = None,
    exclude_case_ids: Optional[Sequence[str]] = None,
    include_disabled: bool = False,
    max_cases: Optional[int] = None,
) -> List[Dict[str, Any]]:
    include_layer_set = {str(v).strip().lower() for v in (include_layers or []) if str(v).strip()}
    include_case_set = {str(v).strip() for v in (include_case_ids or []) if str(v).strip()}
    exclude_case_set = {str(v).strip() for v in (exclude_case_ids or []) if str(v).strip()}

    selected: List[Dict[str, Any]] = []
    for raw_case in cfg.get("cases", []):
        if not isinstance(raw_case, dict):
            continue
        case = dict(raw_case)
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            continue
        layer = str(case.get("layer", "")).strip().lower()
        enabled = bool(case.get("enabled", True))

        if not include_disabled and not enabled:
            continue
        if include_layer_set and layer not in include_layer_set:
            continue
        if include_case_set and case_id not in include_case_set:
            continue
        if case_id in exclude_case_set:
            continue
        selected.append(case)

    if max_cases is not None and max_cases >= 0:
        selected = selected[: int(max_cases)]
    return selected


def _resolve_cmd(
    command: Any,
    context: Dict[str, Any],
) -> Tuple[Any, bool]:
    rendered = _safe_format(command, context)
    if isinstance(rendered, str):
        return rendered, True
    if isinstance(rendered, list):
        return [str(part) for part in rendered], False
    raise ValueError(f"Unsupported command type: {type(command).__name__}")


def _resolve_path(value: Any, context: Dict[str, Any]) -> Path:
    rendered = str(_safe_format(value, context))
    p = Path(rendered)
    if p.is_absolute():
        return p
    return (Path(str(context["repo_root"])) / p).resolve()


def _parse_duration_seconds(started: Optional[str], completed: Optional[str]) -> Optional[float]:
    if not started or not completed:
        return None
    try:
        t0 = datetime.fromisoformat(str(started))
        t1 = datetime.fromisoformat(str(completed))
    except Exception:
        return None
    return float((t1 - t0).total_seconds())


_PIPELINE_PHASE_RE = re.compile(r"PHASE (\S+):")
_PIPELINE_TRANSITION_RE = re.compile(r"Transitioned to (\S+) mode in ([0-9.]+)s")
_PIPELINE_PROGRESS_RE = re.compile(
    r"Cascading progress:.*rate=([0-9.]+) items/s.*tokens=([0-9,]+).*tok/s=(\d+)"
)
_PIPELINE_CASCADING_DONE_RE = re.compile(r"Cascading build complete: (\d+)/(\d+) documents")
_PIPELINE_WARNING_RE = re.compile(r"\| WARNING \|")
_PIPELINE_ERROR_RE = re.compile(r"\| ERROR \|")


def _parse_pipeline_log(log_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "phases_seen": [],
        "warnings": 0,
        "errors": 0,
        "peak_items_per_sec": 0.0,
        "peak_tok_per_sec": 0.0,
        "total_tokens": 0,
        "cascade_docs_built": 0,
        "cascade_docs_total": 0,
        "gpu_transitions": [],
    }
    if not log_path.exists():
        return info

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return info

    for line in lines:
        m = _PIPELINE_PHASE_RE.search(line)
        if m:
            phase = str(m.group(1))
            if phase not in info["phases_seen"]:
                info["phases_seen"].append(phase)

        m = _PIPELINE_TRANSITION_RE.search(line)
        if m:
            info["gpu_transitions"].append(
                {
                    "mode": str(m.group(1)),
                    "seconds": float(m.group(2)),
                }
            )

        m = _PIPELINE_PROGRESS_RE.search(line)
        if m:
            items_s = float(m.group(1))
            tokens = int(m.group(2).replace(",", ""))
            tok_s = float(m.group(3))
            info["peak_items_per_sec"] = max(float(info["peak_items_per_sec"]), items_s)
            info["peak_tok_per_sec"] = max(float(info["peak_tok_per_sec"]), tok_s)
            info["total_tokens"] = max(int(info["total_tokens"]), tokens)

        m = _PIPELINE_CASCADING_DONE_RE.search(line)
        if m:
            info["cascade_docs_built"] += int(m.group(1))
            info["cascade_docs_total"] += int(m.group(2))

        if _PIPELINE_WARNING_RE.search(line):
            info["warnings"] += 1
        if _PIPELINE_ERROR_RE.search(line):
            info["errors"] += 1
    return info


def _extract_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_perf_baseline(path: Path) -> Dict[str, Any]:
    payload = _extract_json_file(path)
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    by_label = {
        str(run.get("label", "")): run
        for run in runs
        if isinstance(run, dict)
    }
    cold = by_label.get("cold", {})
    warm = by_label.get("warm", {})
    cold_agg = cold.get("aggregate", {}) if isinstance(cold, dict) else {}
    warm_agg = warm.get("aggregate", {}) if isinstance(warm, dict) else {}

    cold_docs_s = _to_float(cold_agg.get("docs_per_second"))
    warm_docs_s = _to_float(warm_agg.get("docs_per_second"))
    docs_speedup = None
    if cold_docs_s is not None and cold_docs_s > 0 and warm_docs_s is not None:
        docs_speedup = float(warm_docs_s / cold_docs_s)

    return {
        "summary": {
            "cold_docs_per_second": cold_docs_s,
            "warm_docs_per_second": warm_docs_s,
            "cold_tokens_per_second": _to_float(cold_agg.get("tokens_per_second")),
            "warm_tokens_per_second": _to_float(warm_agg.get("tokens_per_second")),
            "docs_speedup_warm_over_cold": docs_speedup,
            "cold_mae_mean": _to_float(cold_agg.get("mae_mean")),
            "warm_mae_mean": _to_float(warm_agg.get("mae_mean")),
            "cold_brittle_retention_mean": _to_float(cold_agg.get("brittle_retention_mean")),
            "warm_brittle_retention_mean": _to_float(warm_agg.get("brittle_retention_mean")),
        },
        "raw": payload,
    }


def _extract_throughput_suite(path: Path) -> Dict[str, Any]:
    payload = _extract_json_file(path)
    steps = payload.get("steps", {}) if isinstance(payload, dict) else {}
    summaries: Dict[str, Any] = {}
    req_s_vals: List[float] = []
    tok_s_vals: List[float] = []

    if isinstance(steps, dict):
        for step_name, step_block in steps.items():
            if not isinstance(step_block, dict):
                continue
            summary = step_block.get("summary", {})
            points = step_block.get("points", [])
            summary_block = dict(summary) if isinstance(summary, dict) else {}
            if isinstance(points, list) and points:
                peak_tok_s = max(
                    float(p.get("tokens_per_second", 0.0) or 0.0)
                    for p in points
                    if isinstance(p, dict)
                )
                summary_block["peak_tokens_per_second"] = peak_tok_s
            rec_req_s = _to_float(summary_block.get("recommended_req_per_s"))
            if rec_req_s is not None:
                req_s_vals.append(rec_req_s)
            peak_tok = _to_float(summary_block.get("peak_tokens_per_second"))
            if peak_tok is not None:
                tok_s_vals.append(peak_tok)
            summaries[str(step_name)] = summary_block

    return {
        "step_summaries": summaries,
        "overall": {
            "step_count": len(summaries),
            "mean_recommended_req_per_s": float(statistics.fmean(req_s_vals))
            if req_s_vals
            else None,
            "mean_peak_tokens_per_second": float(statistics.fmean(tok_s_vals))
            if tok_s_vals
            else None,
        },
        "raw": payload,
    }


def _extract_pipeline_run(path: Path) -> Dict[str, Any]:
    candidate = Path(path)
    final_stats_path: Optional[Path] = None
    run_dir: Optional[Path] = None

    if candidate.is_file():
        if candidate.name == "final_stats.json":
            final_stats_path = candidate
            run_dir = candidate.parent
    elif candidate.is_dir():
        fs = candidate / "final_stats.json"
        if fs.exists():
            final_stats_path = fs
            run_dir = candidate
        else:
            # Fallback: find latest nested final_stats.json.
            options = list(candidate.rglob("final_stats.json"))
            if options:
                options.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                final_stats_path = options[0]
                run_dir = final_stats_path.parent

    if final_stats_path is None or run_dir is None:
        raise FileNotFoundError(
            f"Could not resolve pipeline final_stats.json under: {candidate}"
        )

    stats = _extract_json_file(final_stats_path)
    log_info = _parse_pipeline_log(run_dir / "run.log")
    cfg = stats.get("config", {}) if isinstance(stats, dict) else {}
    train = stats.get("train", {}) if isinstance(stats, dict) else {}
    test = stats.get("test", {}) if isinstance(stats, dict) else {}
    cm = stats.get("conditional_memory", {}) if isinstance(stats, dict) else {}
    pred_dist = test.get("prediction_distribution", {}) if isinstance(test, dict) else {}
    transitions_raw = (
        log_info.get("gpu_transitions", [])
        if isinstance(log_info.get("gpu_transitions"), list)
        else []
    )
    transition_seconds: List[float] = []
    transition_by_mode: Dict[str, List[float]] = {}
    for t in transitions_raw:
        if not isinstance(t, dict):
            continue
        sec = _to_float(t.get("seconds"))
        if sec is None:
            continue
        mode = str(t.get("mode", "unknown")).strip() or "unknown"
        transition_seconds.append(sec)
        transition_by_mode.setdefault(mode, []).append(sec)

    by_mode_summary: Dict[str, Any] = {}
    for mode, vals in transition_by_mode.items():
        by_mode_summary[mode] = {
            "count": len(vals),
            "mean_seconds": float(statistics.fmean(vals)) if vals else None,
            "max_seconds": float(max(vals)) if vals else None,
            "min_seconds": float(min(vals)) if vals else None,
        }

    summary = {
        "run_dir": str(run_dir),
        "success": bool(stats.get("success")),
        "duration_seconds": _parse_duration_seconds(stats.get("started_at"), stats.get("completed_at")),
        "task": cfg.get("task"),
        "samples": {
            "train": cfg.get("train_samples"),
            "val": cfg.get("val_samples"),
            "test": cfg.get("test_samples"),
        },
        "train": {
            "mae": _to_float(train.get("mae")),
            "pearson_r": _to_float(train.get("pearson_r")),
            "spearman_r": _to_float(train.get("spearman_r")),
            "within_10pct": _to_float(train.get("within_10pct")),
            "n_evaluated": train.get("n_evaluated"),
        },
        "test": {
            "mae": _to_float(test.get("mae")),
            "pearson_r": _to_float(test.get("pearson_r")),
            "spearman_r": _to_float(test.get("spearman_r")),
            "within_10pct": _to_float(test.get("within_10pct")),
            "n_evaluated": test.get("n_evaluated"),
            "frac_neutral": _to_float(pred_dist.get("frac_neutral")),
            "n_unique_rounded_4dp": pred_dist.get("n_unique_rounded_4dp"),
        },
        "conditional_memory": {
            "mode": cm.get("mode"),
            "hit_rate": _to_float(cm.get("hit_rate")),
            "l1_hits": cm.get("l1_hits"),
            "l2_hits": cm.get("l2_hits"),
            "misses": cm.get("misses"),
            "writes": cm.get("writes"),
        }
        if isinstance(cm, dict) and cm
        else None,
        "throughput": {
            "peak_items_per_sec": _to_float(log_info.get("peak_items_per_sec")),
            "peak_tok_per_sec": _to_float(log_info.get("peak_tok_per_sec")),
            "total_tokens": log_info.get("total_tokens"),
            "cascade_docs_built": log_info.get("cascade_docs_built"),
            "cascade_docs_total": log_info.get("cascade_docs_total"),
        },
        "issues": {
            "warnings": log_info.get("warnings"),
            "errors": log_info.get("errors"),
        },
        "phases_seen": list(log_info.get("phases_seen", [])),
        "gpu_transitions": list(log_info.get("gpu_transitions", [])),
        "gpu_transition_stats": {
            "count": len(transition_seconds),
            "total_seconds": float(sum(transition_seconds)) if transition_seconds else 0.0,
            "mean_seconds": float(statistics.fmean(transition_seconds))
            if transition_seconds
            else None,
            "max_seconds": float(max(transition_seconds)) if transition_seconds else None,
            "min_seconds": float(min(transition_seconds)) if transition_seconds else None,
            "p95_seconds": _percentile(transition_seconds, 0.95),
            "by_mode": by_mode_summary,
        },
    }
    return {
        "summary": summary,
        "raw_final_stats": stats,
        "raw_log_summary": log_info,
    }


def _extract_gate_metrics(path: Path) -> Dict[str, Any]:
    payload = _extract_json_file(path)
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    return {
        "summary": {
            "manifesto_mae": _to_float(metrics.get("manifesto_mae")),
            "ruler_primary_mean": _to_float(metrics.get("ruler_primary_mean")),
            "status_manifesto": payload.get("status", {}).get("manifesto")
            if isinstance(payload.get("status"), dict)
            else None,
            "status_ruler": payload.get("status", {}).get("ruler")
            if isinstance(payload.get("status"), dict)
            else None,
        },
        "raw": payload,
    }


_EXTRACTOR_DISPATCH = {
    "json_file": _extract_json_file,
    "perf_baseline": _extract_perf_baseline,
    "throughput_suite": _extract_throughput_suite,
    "pipeline_run": _extract_pipeline_run,
    "gate_metrics": _extract_gate_metrics,
}


def _run_extractors(
    extractors: Sequence[Dict[str, Any]],
    *,
    context: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    out: Dict[str, Any] = {}
    errors: List[str] = []

    for i, raw_spec in enumerate(extractors):
        if not isinstance(raw_spec, dict):
            errors.append(f"extractor[{i}] is not an object")
            continue
        spec = dict(raw_spec)
        ext_type = str(spec.get("type", "")).strip()
        if ext_type not in _EXTRACTOR_DISPATCH:
            errors.append(f"extractor[{i}] unknown type: {ext_type}")
            continue
        name = str(spec.get("name") or ext_type)
        if not name:
            name = f"{ext_type}_{i+1}"

        path_value = spec.get("path")
        if path_value is None:
            errors.append(f"extractor[{i}] missing required field: path")
            continue
        try:
            path = _resolve_path(path_value, context)
            parsed = _EXTRACTOR_DISPATCH[ext_type](path)
            out[name] = parsed
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return out, errors


def run_performance_suite(
    cfg: Dict[str, Any],
    *,
    suite_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    include_layers: Optional[Sequence[str]] = None,
    include_case_ids: Optional[Sequence[str]] = None,
    exclude_case_ids: Optional[Sequence[str]] = None,
    include_disabled: bool = False,
    dry_run: bool = False,
    stop_on_failure: bool = False,
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {}
    selected_cases = _select_cases(
        cfg,
        include_layers=include_layers,
        include_case_ids=include_case_ids,
        exclude_case_ids=exclude_case_ids,
        include_disabled=include_disabled,
        max_cases=max_cases,
    )

    if output_dir is None:
        output_root = Path(str(cfg.get("output_root", "outputs/performance_suite")))
        if not output_root.is_absolute():
            output_root = (repo_root / output_root).resolve()
        run_dir = output_root / f"run_{_utc_stamp()}"
    else:
        run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now_iso()
    run_payload: Dict[str, Any] = {
        "suite_name": str(cfg.get("name", "performance_suite")),
        "suite_config_path": str(suite_config_path) if suite_config_path is not None else None,
        "generated_at": started_at,
        "run_dir": str(run_dir),
        "dry_run": bool(dry_run),
        "selected_case_count": len(selected_cases),
        "selected_case_ids": [str(c.get("id")) for c in selected_cases],
        "results": [],
    }
    progress_path = run_dir / "suite_progress.json"

    def _refresh_summary(payload: Dict[str, Any]) -> None:
        statuses = [str(r.get("status", "")) for r in payload.get("results", [])]
        payload["summary"] = {
            "cases_total": len(statuses),
            "cases_ok": sum(1 for s in statuses if s == "ok"),
            "cases_partial": sum(1 for s in statuses if s == "partial"),
            "cases_failed": sum(1 for s in statuses if s == "failed"),
            "cases_dry_run": sum(1 for s in statuses if s == "dry_run"),
            "cases_skipped": sum(1 for s in statuses if s == "skipped"),
        }

    def _write_progress(payload: Dict[str, Any]) -> None:
        try:
            progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Unable to write suite progress snapshot %s: %s", progress_path, exc)

    _refresh_summary(run_payload)
    _write_progress(run_payload)

    for case_idx, case in enumerate(selected_cases, start=1):
        case_id = str(case.get("id")).strip()
        layer = str(case.get("layer", "unspecified")).strip().lower()
        description = str(case.get("description", "")).strip()
        command = case.get("command")
        if command is None:
            run_payload["results"].append(
                {
                    "id": case_id,
                    "layer": layer,
                    "description": description,
                    "status": "skipped",
                    "error": "missing command",
                    "repeats": [],
                }
            )
            continue

        case_dir = run_dir / "cases" / _slugify(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        repeats = int(case.get("repeats", defaults.get("repeats", 1)) or 1)
        timeout_seconds = float(case.get("timeout_seconds", defaults.get("timeout_seconds", 0.0)) or 0.0)
        cwd_value = case.get("cwd", defaults.get("cwd", "."))
        env_overrides = case.get("env", {}) if isinstance(case.get("env"), dict) else {}
        extractors = case.get("extractors", []) if isinstance(case.get("extractors"), list) else []

        case_result: Dict[str, Any] = {
            "id": case_id,
            "layer": layer,
            "description": description,
            "repeats": [],
            "metric_rules": case.get("metric_rules", []),
        }

        logger.info(
            "Running case %d/%d: %s (layer=%s, repeats=%d)",
            case_idx,
            len(selected_cases),
            case_id,
            layer,
            repeats,
        )

        case_failed = False
        for rep in range(1, repeats + 1):
            rep_dir = case_dir / f"repeat_{rep:02d}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            log_path = rep_dir / "command.log"
            repeat_context = {
                "repo_root": str(repo_root),
                "run_dir": str(run_dir),
                "case_dir": str(rep_dir),
                "case_id": case_id,
                "repeat": rep,
                "timestamp": _utc_stamp(),
            }
            resolved_cwd = _resolve_path(cwd_value, repeat_context)

            rendered_env = _safe_format(env_overrides, repeat_context)
            cmd_error: Optional[str] = None
            shell_mode = False
            resolved_cmd: Any = None
            try:
                resolved_cmd, shell_mode = _resolve_cmd(command, repeat_context)
            except Exception as exc:
                cmd_error = str(exc)

            result_row: Dict[str, Any] = {
                "repeat_index": rep,
                "started_at": _utc_now_iso(),
                "command": resolved_cmd,
                "cwd": str(resolved_cwd),
                "timeout_seconds": timeout_seconds if timeout_seconds > 0 else None,
                "log_path": str(log_path),
                "shell": bool(shell_mode),
                "status": "ok",
                "returncode": 0,
                "duration_seconds": 0.0,
                "extracts": {},
                "errors": [],
            }

            t0 = time.perf_counter()
            if cmd_error is not None:
                result_row["status"] = "error"
                result_row["returncode"] = 2
                result_row["errors"].append(f"command_render: {cmd_error}")
            elif dry_run:
                log_path.write_text(
                    "DRY RUN\n"
                    f"cwd={resolved_cwd}\n"
                    f"shell={shell_mode}\n"
                    f"command={resolved_cmd}\n",
                    encoding="utf-8",
                )
                result_row["status"] = "dry_run"
                result_row["returncode"] = 0
            else:
                env = {k: str(v) for k, v in os.environ.items()}
                env.update({str(k): str(v) for k, v in rendered_env.items()})
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as handle:
                    handle.write(f"cwd={resolved_cwd}\n")
                    handle.write(f"shell={shell_mode}\n")
                    handle.write(f"command={resolved_cmd}\n\n")
                    try:
                        proc = subprocess.run(
                            resolved_cmd,
                            cwd=str(resolved_cwd),
                            env=env,
                            shell=shell_mode,
                            executable="/bin/bash" if shell_mode else None,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            timeout=timeout_seconds if timeout_seconds > 0 else None,
                            check=False,
                            text=True,
                        )
                        result_row["returncode"] = int(proc.returncode)
                        if proc.returncode != 0:
                            result_row["status"] = "failed"
                            result_row["errors"].append(
                                f"command returned non-zero exit code: {proc.returncode}"
                            )
                    except subprocess.TimeoutExpired:
                        result_row["status"] = "timeout"
                        result_row["returncode"] = 124
                        result_row["errors"].append("command timed out")
                    except Exception as exc:
                        result_row["status"] = "error"
                        result_row["returncode"] = 2
                        result_row["errors"].append(f"command execution error: {exc}")
            result_row["duration_seconds"] = float(time.perf_counter() - t0)

            should_extract = result_row["status"] == "ok"
            if should_extract and extractors:
                extracts, extract_errors = _run_extractors(extractors, context=repeat_context)
                result_row["extracts"] = extracts
                if extract_errors:
                    result_row["errors"].extend(extract_errors)
                    if result_row["status"] == "ok":
                        result_row["status"] = "partial"
            elif extractors and result_row["status"] in {"failed", "timeout", "error"}:
                result_row["errors"].append("extractors skipped due to command failure")

            case_result["repeats"].append(result_row)
            if result_row["status"] not in {"ok", "partial", "dry_run"}:
                case_failed = True
                if stop_on_failure:
                    break

        repeat_statuses = [str(r.get("status")) for r in case_result["repeats"]]
        if not repeat_statuses:
            case_result["status"] = "skipped"
        elif any(s in {"failed", "timeout", "error"} for s in repeat_statuses):
            case_result["status"] = "failed"
        elif any(s == "partial" for s in repeat_statuses):
            case_result["status"] = "partial"
        elif all(s == "dry_run" for s in repeat_statuses):
            case_result["status"] = "dry_run"
        else:
            case_result["status"] = "ok"
        case_result["successful_repeats"] = sum(1 for s in repeat_statuses if s in {"ok", "partial", "dry_run"})
        case_result["failed_repeats"] = sum(1 for s in repeat_statuses if s in {"failed", "timeout", "error"})

        run_payload["results"].append(case_result)
        _refresh_summary(run_payload)
        _write_progress(run_payload)
        if stop_on_failure and case_failed:
            break

    completed_at = _utc_now_iso()
    run_payload["completed_at"] = completed_at
    run_payload["duration_seconds"] = _parse_duration_seconds(started_at, completed_at)
    _refresh_summary(run_payload)
    _write_progress(run_payload)

    return run_payload


def save_suite_results(
    run_payload: Dict[str, Any],
    *,
    json_path: Optional[Path] = None,
    markdown_path: Optional[Path] = None,
) -> Dict[str, Path]:
    run_dir = Path(str(run_payload.get("run_dir", "."))).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = json_path or (run_dir / "suite_results.json")
    out_md = markdown_path or (run_dir / "suite_results.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    out_md.write_text(render_suite_markdown(run_payload), encoding="utf-8")
    return {"json": out_json, "markdown": out_md}


def render_suite_markdown(run_payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Performance Suite Run")
    lines.append("")
    lines.append(f"- Suite: `{run_payload.get('suite_name', 'unknown')}`")
    lines.append(f"- Generated: `{run_payload.get('generated_at', 'n/a')}`")
    lines.append(f"- Completed: `{run_payload.get('completed_at', 'n/a')}`")
    lines.append(f"- Duration (s): `{run_payload.get('duration_seconds', 'n/a')}`")
    lines.append(f"- Run dir: `{run_payload.get('run_dir', 'n/a')}`")
    lines.append("")
    lines.append("## Case Status")
    lines.append("")
    lines.append("| Case | Layer | Status | Repeats | Successful | Failed |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for case in run_payload.get("results", []):
        if not isinstance(case, dict):
            continue
        lines.append(
            "| `{id}` | `{layer}` | `{status}` | `{repeats}` | `{ok}` | `{failed}` |".format(
                id=case.get("id", "n/a"),
                layer=case.get("layer", "n/a"),
                status=case.get("status", "n/a"),
                repeats=len(case.get("repeats", [])),
                ok=case.get("successful_repeats", 0),
                failed=case.get("failed_repeats", 0),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _collect_case_metric_values(case_result: Dict[str, Any], path: str) -> List[float]:
    values: List[float] = []
    for rep in case_result.get("repeats", []):
        if not isinstance(rep, dict):
            continue
        value = _deep_get(rep, path)
        f = _to_float(value)
        if f is None:
            continue
        values.append(f)
    return values


def compare_suite_results(
    cfg: Dict[str, Any],
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_cases = {
        str(case.get("id", "")): case
        for case in baseline.get("results", [])
        if isinstance(case, dict)
    }
    candidate_cases = {
        str(case.get("id", "")): case
        for case in candidate.get("results", [])
        if isinstance(case, dict)
    }

    rows: List[Dict[str, Any]] = []
    for raw_case in cfg.get("cases", []):
        if not isinstance(raw_case, dict):
            continue
        case_id = str(raw_case.get("id", "")).strip()
        if not case_id:
            continue
        metric_rules = raw_case.get("metric_rules", [])
        if not isinstance(metric_rules, list):
            continue

        baseline_case = baseline_cases.get(case_id)
        candidate_case = candidate_cases.get(case_id)
        for rule_idx, raw_rule in enumerate(metric_rules):
            if not isinstance(raw_rule, dict):
                continue
            path = str(raw_rule.get("path", "")).strip()
            if not path:
                continue
            direction = str(raw_rule.get("direction", "higher")).strip().lower()
            if direction not in {"higher", "lower"}:
                direction = "higher"
            aggregate_mode = str(raw_rule.get("aggregate", "median") or "median")
            max_reg_pct = _to_float(raw_rule.get("max_regression_pct"))
            max_reg_abs = _to_float(raw_rule.get("max_regression_abs"))
            name = str(raw_rule.get("name") or path or f"metric_{rule_idx + 1}")

            base_values = _collect_case_metric_values(baseline_case or {}, path) if baseline_case else []
            cand_values = _collect_case_metric_values(candidate_case or {}, path) if candidate_case else []
            base_val = _aggregate(base_values, mode=aggregate_mode)
            cand_val = _aggregate(cand_values, mode=aggregate_mode)

            row: Dict[str, Any] = {
                "case_id": case_id,
                "metric": name,
                "path": path,
                "direction": direction,
                "aggregate": aggregate_mode,
                "baseline": base_val,
                "candidate": cand_val,
                "max_regression_pct": max_reg_pct,
                "max_regression_abs": max_reg_abs,
                "status": "pass",
                "message": "",
                "delta": None,
                "regression_abs": None,
                "regression_pct": None,
            }

            if base_val is None or cand_val is None:
                row["status"] = "missing"
                row["message"] = "missing baseline or candidate metric value"
                rows.append(row)
                continue

            delta = float(cand_val - base_val)
            row["delta"] = delta
            if direction == "higher":
                regression_abs = float(base_val - cand_val)
            else:
                regression_abs = float(cand_val - base_val)
            row["regression_abs"] = regression_abs

            if abs(base_val) > 1e-12:
                row["regression_pct"] = float((regression_abs / abs(base_val)) * 100.0)

            regression = regression_abs > 0.0
            if regression and max_reg_abs is not None:
                regression = regression and (regression_abs > max_reg_abs)
            if regression and max_reg_pct is not None:
                reg_pct = row.get("regression_pct")
                # If baseline is ~0, percent guard cannot be evaluated.
                if reg_pct is None:
                    regression = False
                else:
                    regression = regression and (float(reg_pct) > float(max_reg_pct))

            if regression:
                row["status"] = "regression"
                row["message"] = "candidate regressed beyond threshold"
            else:
                row["status"] = "pass"
                row["message"] = "within threshold"

            rows.append(row)

    summary = {
        "checks_total": len(rows),
        "checks_pass": sum(1 for r in rows if r.get("status") == "pass"),
        "checks_regression": sum(1 for r in rows if r.get("status") == "regression"),
        "checks_missing": sum(1 for r in rows if r.get("status") == "missing"),
    }
    return {
        "generated_at": _utc_now_iso(),
        "summary": summary,
        "rows": rows,
    }


def render_comparison_markdown(comparison: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Performance Comparison")
    lines.append("")
    summary = comparison.get("summary", {}) if isinstance(comparison, dict) else {}
    lines.append(
        "- Checks: total={t} pass={p} regression={r} missing={m}".format(
            t=summary.get("checks_total", 0),
            p=summary.get("checks_pass", 0),
            r=summary.get("checks_regression", 0),
            m=summary.get("checks_missing", 0),
        )
    )
    lines.append("")
    lines.append("| Case | Metric | Dir | Baseline | Candidate | Delta | Regr. % | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in comparison.get("rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{case}` | `{metric}` | `{dir}` | `{base}` | `{cand}` | `{delta}` | `{reg_pct}` | `{status}` |".format(
                case=row.get("case_id", "n/a"),
                metric=row.get("metric", "n/a"),
                dir=row.get("direction", "n/a"),
                base=_fmt_num(row.get("baseline")),
                cand=_fmt_num(row.get("candidate")),
                delta=_fmt_num(row.get("delta")),
                reg_pct=_fmt_num(row.get("regression_pct")),
                status=row.get("status", "n/a"),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _fmt_num(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
