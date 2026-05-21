"""Utilities for reproducible training runs and run-environment capture."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPRO_ARTIFACT = "reproducibility_manifest.json"
TRACKED_ENV_VARS: tuple[str, ...] = (
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "HF_HOME",
    "HF_DATASETS_CACHE",
)
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "dspy",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _tracked_env_snapshot() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in TRACKED_ENV_VARS:
        value = os.environ.get(key)
        if value is not None:
            out[str(key)] = str(value)
    return out


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in TRACKED_PACKAGES:
        try:
            versions[str(package_name)] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions


def _git_output(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip()


def git_snapshot() -> Dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    status_lines = [line for line in _git_output("status", "--short").splitlines() if line.strip()]
    return {
        "project_root": str(PROJECT_ROOT),
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status_lines),
        "status_short": status_lines[:200],
    }


def torch_runtime_snapshot() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"available": False}

    snapshot: Dict[str, Any] = {
        "available": True,
        "version": str(getattr(torch, "__version__", "")),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_version": str(getattr(torch.version, "cuda", "") or ""),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
    }

    try:
        snapshot["num_threads"] = int(torch.get_num_threads())
    except Exception:
        pass
    try:
        snapshot["num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:
        pass

    try:
        snapshot["cudnn_available"] = bool(torch.backends.cudnn.is_available())
    except Exception:
        snapshot["cudnn_available"] = False
    try:
        snapshot["cudnn_version"] = int(torch.backends.cudnn.version() or 0)
    except Exception:
        snapshot["cudnn_version"] = 0
    try:
        snapshot["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    except Exception:
        snapshot["cudnn_deterministic"] = False
    try:
        snapshot["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        snapshot["cudnn_benchmark"] = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            snapshot["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        snapshot["allow_tf32_matmul"] = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            snapshot["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        snapshot["allow_tf32_cudnn"] = None

    return snapshot


def configure_reproducibility(
    seed: int,
    *,
    deterministic_torch: bool = True,
    warn_only: bool = True,
) -> Dict[str, Any]:
    seed = int(seed)
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    numpy_applied = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_applied = True
    except Exception:
        numpy_applied = False

    torch_applied = False
    cuda_seeded = False
    deterministic_algorithms_enabled = False
    cudnn_deterministic = False
    cudnn_benchmark = None
    try:
        import torch

        torch.manual_seed(seed)
        torch_applied = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True

        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
            except TypeError:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        try:
            deterministic_algorithms_enabled = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:
            deterministic_algorithms_enabled = False

        try:
            torch.backends.cudnn.deterministic = bool(deterministic_torch)
            cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        except Exception:
            cudnn_deterministic = False

        try:
            torch.backends.cudnn.benchmark = False if bool(deterministic_torch) else bool(torch.backends.cudnn.benchmark)
            cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
        except Exception:
            cudnn_benchmark = None
    except Exception:
        pass

    return {
        "seed": seed,
        "pythonhashseed_env": str(os.environ.get("PYTHONHASHSEED", "")),
        "numpy_seed_applied": bool(numpy_applied),
        "torch_seed_applied": bool(torch_applied),
        "cuda_seed_applied": bool(cuda_seeded),
        "deterministic_torch_requested": bool(deterministic_torch),
        "deterministic_torch_warn_only": bool(warn_only),
        "deterministic_algorithms_enabled": bool(deterministic_algorithms_enabled),
        "cudnn_deterministic": bool(cudnn_deterministic),
        "cudnn_benchmark": cudnn_benchmark,
    }


def build_reproducibility_manifest(
    *,
    seed: Optional[int] = None,
    cli_args: Optional[Mapping[str, Any]] = None,
    config: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
    applied: Optional[Mapping[str, Any]] = None,
    command: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    rendered_command = [str(item) for item in (command or sys.argv)]
    payload: Dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "command": rendered_command,
        "command_str": " ".join(rendered_command),
        "cwd": str(Path.cwd().resolve()),
        "project_root": str(PROJECT_ROOT),
        "host": {
            "hostname": str(socket.gethostname()),
            "platform": str(platform.platform()),
            "python_version": str(sys.version),
            "python_executable": str(sys.executable),
        },
        "environment": _tracked_env_snapshot(),
        "packages": _package_versions(),
        "git": git_snapshot(),
        "torch_runtime": torch_runtime_snapshot(),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if cli_args is not None:
        payload["cli_args"] = _json_safe(dict(cli_args))
    if config is not None:
        payload["config"] = _json_safe(config)
    if applied is not None:
        payload["applied_reproducibility"] = _json_safe(dict(applied))
    if extra:
        payload["extra"] = _json_safe(dict(extra))
    return payload


def write_reproducibility_manifest(
    output_dir: str | Path,
    *,
    seed: Optional[int] = None,
    cli_args: Optional[Mapping[str, Any]] = None,
    config: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
    applied: Optional[Mapping[str, Any]] = None,
    filename: str = DEFAULT_REPRO_ARTIFACT,
    command: Optional[Sequence[str]] = None,
) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    payload = build_reproducibility_manifest(
        seed=seed,
        cli_args=cli_args,
        config=config,
        extra=extra,
        applied=applied,
        command=command,
    )
    path = output_path / str(filename)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_REPRO_ARTIFACT",
    "build_reproducibility_manifest",
    "configure_reproducibility",
    "git_snapshot",
    "torch_runtime_snapshot",
    "write_reproducibility_manifest",
]
