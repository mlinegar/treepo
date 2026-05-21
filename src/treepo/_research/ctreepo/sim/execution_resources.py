from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _parse_items(text: str) -> list[str]:
    items: list[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            items.append(item)
    return items


def normalize_device_mode(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if text == "gpu" or text.startswith("cuda"):
        return "cuda"
    if text == "auto":
        return "auto"
    return "cpu"


def parse_command_flags(command: str) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return flags
    idx = 0
    while idx < len(tokens):
        token = str(tokens[idx]).strip()
        if not token.startswith("--"):
            idx += 1
            continue
        key = token[2:].replace("-", "_")
        if idx + 1 < len(tokens) and not str(tokens[idx + 1]).startswith("--"):
            flags[key] = tokens[idx + 1]
            idx += 2
            continue
        flags[key] = True
        idx += 1
    return flags


def _script_name_from_command(command: str) -> str:
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return ""
    for token in tokens:
        if token.endswith(".py"):
            return Path(token).name
    for idx, token in enumerate(tokens[:-1]):
        if token == "-m":
            return str(tokens[idx + 1]).strip().split(".")[-1]
    return ""


def _coalesce_config(
    config: Mapping[str, Any] | None,
    command: str,
) -> Dict[str, Any]:
    merged = dict(config or {})
    cmd_flags = parse_command_flags(command)
    for key, value in cmd_flags.items():
        merged.setdefault(str(key), value)
    return merged


def _string_values(config: Mapping[str, Any], keys: Iterable[str]) -> list[str]:
    out: list[str] = []
    for key in keys:
        if key not in config:
            continue
        value = config.get(key)
        if isinstance(value, (list, tuple, set)):
            out.extend(str(x).strip().lower() for x in value if str(x).strip())
            continue
        out.extend(str(x).strip().lower() for x in _parse_items(str(value)))
    return out


def _infer_family_hint(family: str, script_name: str, config: Mapping[str, Any]) -> str:
    family_text = str(family or "").strip().lower()
    if family_text:
        return family_text
    if script_name == "run_markov_changepoint_ops_count_simulation.py":
        return "markov-ops-count"
    if script_name == "run_segment_lda_ops_weight_recovery_simulation.py":
        return "segment-lda-ops"
    if script_name == "run_segmented_lda_ctreepo_simulation.py":
        return "segmented-lda-ctreepo"
    if script_name == "run_lda_tree_recovery_learned_world_batch.py":
        return "lda-tree-recovery-learned"
    if script_name == "run_lda_tree_recovery_simulation.py":
        return "lda-tree-recovery-exact"
    return str(config.get("family", "")).strip().lower()


def _auto_gpu_capable(
    *,
    family: str,
    config: Mapping[str, Any],
    requires: Sequence[str],
    script_name: str,
) -> bool:
    reqs = {str(x).strip().lower() for x in requires}
    estimator_values = set(
        _string_values(
            config,
            [
                "topic_phi_estimator",
                "topic_phi_estimators",
                "leaf_theta_estimator",
                "leaf_theta_estimators",
                "model_family",
            ],
        )
    )
    if family == "markov-ops-count" or script_name == "run_markov_changepoint_ops_count_simulation.py":
        return True
    if family == "segment-lda-ops" or script_name == "run_segment_lda_ops_weight_recovery_simulation.py":
        return any(item.startswith("neural_") for item in estimator_values) or "torch" in reqs
    if family == "segmented-lda-ctreepo" or script_name == "run_segmented_lda_ctreepo_simulation.py":
        return (
            any(item.startswith("neural_") for item in estimator_values)
            or "mlp" in estimator_values
            or "torch" in reqs
        )
    if family == "lda-tree-recovery-learned" or script_name == "run_lda_tree_recovery_learned_world_batch.py":
        return True
    return "torch" in reqs


def _auto_gpu_preferred(
    *,
    family: str,
    config: Mapping[str, Any],
    requires: Sequence[str],
    script_name: str,
) -> bool:
    reqs = {str(x).strip().lower() for x in requires}
    estimator_values = set(
        _string_values(
            config,
            [
                "topic_phi_estimator",
                "topic_phi_estimators",
                "leaf_theta_estimator",
                "leaf_theta_estimators",
                "model_family",
            ],
        )
    )
    if family == "markov-ops-count" or script_name == "run_markov_changepoint_ops_count_simulation.py":
        # Markov sweeps are typically wide grids with many medium-cost jobs; in this repo they
        # saturate shared CPUs more effectively than scarce GPU lanes unless CUDA is requested
        # explicitly. Keep them auto-capable but CPU-preferred.
        return False
    if family == "segment-lda-ops" or script_name == "run_segment_lda_ops_weight_recovery_simulation.py":
        return any(item.startswith("neural_") for item in estimator_values)
    if family == "segmented-lda-ctreepo" or script_name == "run_segmented_lda_ctreepo_simulation.py":
        return any(item.startswith("neural_") for item in estimator_values) or "mlp" in estimator_values
    if family == "lda-tree-recovery-learned" or script_name == "run_lda_tree_recovery_learned_world_batch.py":
        return True
    return "torch" in reqs


def infer_run_resources(
    *,
    family: str = "",
    config: Mapping[str, Any] | None = None,
    requires: Sequence[str] | None = None,
    command: str = "",
) -> Dict[str, Any]:
    merged = _coalesce_config(config, command)
    reqs = [str(x) for x in (requires or [])]
    script_name = _script_name_from_command(command)
    family_hint = _infer_family_hint(str(family), script_name, merged)
    device_mode = normalize_device_mode(
        merged.get("device", ("cuda" if bool(merged.get("use_cuda", False)) else ""))
    )
    torch_threads_raw = merged.get("torch_threads", 0)
    try:
        torch_threads = int(torch_threads_raw)
    except Exception:
        torch_threads = 0
    cpu_threads = int(torch_threads) if int(torch_threads) > 0 else 1

    if device_mode == "cpu":
        accelerator = "cpu"
        gpu_eligible = False
        gpu_preferred = False
    elif device_mode == "cuda":
        accelerator = "gpu"
        gpu_eligible = True
        gpu_preferred = True
    else:
        gpu_eligible = _auto_gpu_capable(
            family=family_hint,
            config=merged,
            requires=reqs,
            script_name=script_name,
        )
        gpu_preferred = gpu_eligible and _auto_gpu_preferred(
            family=family_hint,
            config=merged,
            requires=reqs,
            script_name=script_name,
        )
        accelerator = "auto" if gpu_eligible else "cpu"

    return {
        "accelerator": str(accelerator),
        "device_mode": str(device_mode or ("auto" if accelerator == "auto" else "cpu")),
        "gpu_eligible": bool(gpu_eligible),
        "gpu_preferred": bool(gpu_preferred),
        "cpu_threads": int(max(1, cpu_threads)),
        "torch_threads": int(torch_threads),
    }
