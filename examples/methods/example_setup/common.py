"""Common helper utilities for method examples."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


EXAMPLES_DIR = Path(__file__).resolve().parents[1]


def parse_output_dir() -> Path:
    """Parse the shared ``--output-dir`` argument for examples without TOML."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_example_config(
    *,
    default_config: str,
    config_cls: type[Any],
    cli_overrides: Sequence[tuple[str, type[Any] | None]] = (),
) -> tuple[Path, Any]:
    """Parse the tiny common CLI and load the paired TOML into ``config_cls``."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=EXAMPLES_DIR / default_config)
    for name, value_type in cli_overrides:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=value_type,
            default=None,
        )
    args = parser.parse_args()

    from treepo.methods.canonical_defaults import load_dataclass

    overrides = {name: getattr(args, name) for name, _ in cli_overrides}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, load_dataclass(args.config, config_cls, overrides=overrides)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def config_dict(config: Any) -> dict[str, Any]:
    return asdict(config)


def grid_values(values: Sequence[Any], *, name: str) -> tuple[Any, ...]:
    out = tuple(values)
    if not out:
        raise ValueError(f"{name} must be non-empty")
    return out


def neural_operator_backend(config: Any, *, output_dir: Path, operator_kind: str) -> dict[str, Any]:
    backend = {
        "output_dir": str(output_dir),
        "operator_kind": str(operator_kind),
        "embedding_dim": int(config.embedding_dim),
        "hidden_channels": int(config.hidden_channels),
        "n_modes": int(config.n_modes),
        "n_layers": int(config.n_layers),
        "head_hidden_dim": int(config.head_hidden_dim),
        "epochs_per_iteration": int(config.epochs_per_iteration),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "device": str(config.device),
    }
    for key in (
        "conv_kernel_size",
        "normalize_targets",
        "numeric_transition_state_weight",
        "numeric_transition_count_scale",
    ):
        if hasattr(config, key):
            backend[key] = getattr(config, key)
    return backend


def result_metrics(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {kind: payload.get("metrics", {}) for kind, payload in results.items()}


def result_statistics(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        kind: dict((payload.get("artifacts") or {}).get("statistic") or {})
        for kind, payload in results.items()
    }


def metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def artifact_kind(result: Any) -> str | None:
    artifact = dict((result.artifacts or {}).get("f") or {})
    value = artifact.get("kind")
    return str(value) if value is not None else None

