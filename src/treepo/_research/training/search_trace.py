"""Helpers for structured training search specs and deterministic trial traces."""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class SearchDimension:
    flag: str
    values: tuple[Any, ...]
    false_flag: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchDimension":
        data = dict(payload or {})
        flag = str(data.get("flag", "") or "").strip()
        if not flag.startswith("--"):
            raise ValueError(f"Search dimension flag must start with '--', got {flag!r}")
        values = tuple(list(data.get("values") or ()))
        if not values:
            raise ValueError(f"Search dimension {flag!r} requires at least one value")
        return cls(
            flag=flag,
            values=values,
            false_flag=str(data.get("false_flag", "") or "").strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class SearchSpec:
    mode: str = "fixed"
    max_trials: Optional[int] = None
    selection_metric: str = "validation_mae"
    selection_metric_mode: str = "minimize"
    tie_breaker_metric: str = "training_time_seconds"
    tie_breaker_mode: str = "minimize"
    final_tie_breaker: str = "trial_index"
    seed_policy: str = "base_plus_trial_index"
    dimensions: tuple[SearchDimension, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchSpec":
        data = dict(payload or {})
        raw_dimensions = list(data.get("dimensions") or ())
        dimensions = tuple(SearchDimension.from_dict(item) for item in raw_dimensions)
        mode = str(data.get("mode", "fixed") or "fixed").strip().lower()
        if mode not in {"fixed", "grid", "random"}:
            raise ValueError(f"Unsupported search mode: {mode!r}")
        max_trials = data.get("max_trials")
        if max_trials is not None:
            max_trials = int(max_trials)
            if max_trials <= 0:
                raise ValueError("max_trials must be positive when provided")
        return cls(
            mode=mode if dimensions else "fixed",
            max_trials=max_trials,
            selection_metric=str(data.get("selection_metric", "validation_mae") or "validation_mae"),
            selection_metric_mode=str(data.get("selection_metric_mode", "minimize") or "minimize"),
            tie_breaker_metric=str(data.get("tie_breaker_metric", "training_time_seconds") or "training_time_seconds"),
            tie_breaker_mode=str(data.get("tie_breaker_mode", "minimize") or "minimize"),
            final_tie_breaker=str(data.get("final_tie_breaker", "trial_index") or "trial_index"),
            seed_policy=str(data.get("seed_policy", "base_plus_trial_index") or "base_plus_trial_index"),
            dimensions=dimensions,
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = [dimension.to_dict() for dimension in self.dimensions]
        return _json_safe(payload)


def load_search_spec(path: str | Path) -> SearchSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Search spec at {spec_path} must be a JSON object")
    spec = SearchSpec.from_dict(payload)
    return SearchSpec(
        mode=spec.mode,
        max_trials=spec.max_trials,
        selection_metric=spec.selection_metric,
        selection_metric_mode=spec.selection_metric_mode,
        tie_breaker_metric=spec.tie_breaker_metric,
        tie_breaker_mode=spec.tie_breaker_mode,
        final_tie_breaker=spec.final_tie_breaker,
        seed_policy=spec.seed_policy,
        dimensions=spec.dimensions,
        metadata={
            **dict(spec.metadata),
            "source_path": str(spec_path),
        },
    )


def fixed_search_spec(*, metadata: Optional[Mapping[str, Any]] = None) -> SearchSpec:
    return SearchSpec(
        mode="fixed",
        metadata=dict(metadata or {}),
    )


def _render_flag_tokens(flag: str, value: Any, *, false_flag: str = "") -> list[str]:
    if isinstance(value, bool):
        if value:
            return [str(flag)]
        if false_flag:
            return [str(false_flag)]
        return []
    if value is None:
        return []
    return [str(flag), str(value)]


def expand_search_trials(
    spec: SearchSpec,
    *,
    base_seed: int,
) -> list[Dict[str, Any]]:
    if spec.mode == "fixed" or not spec.dimensions:
        return [
            {
                "trial_id": "trial_000",
                "trial_index": 0,
                "seed": int(base_seed),
                "overrides": [],
                "arg_tokens": [],
            }
        ]

    value_grid = [list(dimension.values) for dimension in spec.dimensions]
    all_combinations = list(itertools.product(*value_grid))
    if spec.mode == "random":
        limit = min(len(all_combinations), int(spec.max_trials or len(all_combinations)))
        rng = random.Random(int(base_seed))
        selected_indices = sorted(rng.sample(range(len(all_combinations)), k=limit))
        combinations = [all_combinations[idx] for idx in selected_indices]
    else:
        combinations = all_combinations
        if spec.max_trials is not None:
            combinations = combinations[: int(spec.max_trials)]

    trials: list[Dict[str, Any]] = []
    for trial_index, combo in enumerate(combinations):
        overrides: list[Dict[str, Any]] = []
        arg_tokens: list[str] = []
        for dimension, value in zip(spec.dimensions, combo):
            override = {
                "flag": str(dimension.flag),
                "value": _json_safe(value),
            }
            if dimension.false_flag:
                override["false_flag"] = str(dimension.false_flag)
            overrides.append(override)
            arg_tokens.extend(
                _render_flag_tokens(
                    str(dimension.flag),
                    value,
                    false_flag=str(dimension.false_flag),
                )
            )
        trial_seed = int(base_seed) + int(trial_index) if spec.seed_policy == "base_plus_trial_index" else int(base_seed)
        trials.append(
            {
                "trial_id": f"trial_{trial_index:03d}",
                "trial_index": int(trial_index),
                "seed": int(trial_seed),
                "overrides": overrides,
                "arg_tokens": arg_tokens,
            }
        )
    return trials


def _score_key(value: Any, mode: str) -> float:
    if value is None:
        return math.inf
    try:
        numeric = float(value)
    except Exception:
        return math.inf
    if not math.isfinite(numeric):
        return math.inf
    return -numeric if str(mode).strip().lower() == "maximize" else numeric


def select_best_trial(
    trials: Sequence[Mapping[str, Any]],
    *,
    selection_metric: str,
    selection_metric_mode: str = "minimize",
    tie_breaker_metric: str = "training_time_seconds",
    tie_breaker_mode: str = "minimize",
) -> Optional[Dict[str, Any]]:
    successful_trials = [dict(trial) for trial in trials if bool(trial.get("success", False))]
    if not successful_trials:
        return None

    def _trial_key(trial: Mapping[str, Any]) -> tuple[float, float, int]:
        metrics = dict(trial.get("selection_metrics", {}) or {})
        return (
            _score_key(metrics.get(selection_metric), selection_metric_mode),
            _score_key(metrics.get(tie_breaker_metric), tie_breaker_mode),
            int(trial.get("trial_index", 0) or 0),
        )

    metricful_trials = [
        trial for trial in successful_trials
        if math.isfinite(_trial_key(trial)[0])
    ]
    pool = metricful_trials or successful_trials
    return min(pool, key=_trial_key) if pool else None


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(dict(payload)), indent=2) + "\n", encoding="utf-8")
    return out_path


__all__ = [
    "SearchDimension",
    "SearchSpec",
    "expand_search_trials",
    "fixed_search_spec",
    "load_search_spec",
    "select_best_trial",
    "write_json",
]
