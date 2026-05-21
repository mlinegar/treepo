from __future__ import annotations

from typing import Any, Dict, Mapping

from treepo._research.experiments.roles import metadata_with_roles, normalize_roles


def runtime_surface_metadata(surfaces: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact surface metadata for result rows and script payloads."""

    out: Dict[str, Any] = {}
    for key, raw_cfg in dict(surfaces or {}).items():
        if not isinstance(raw_cfg, Mapping):
            continue
        cfg = dict(raw_cfg)
        out[str(key)] = {
            "engine": str(cfg.get("engine", "") or ""),
            "model": str(cfg.get("model", "") or ""),
            "base_url": str(cfg.get("base_url", "") or ""),
            "execution_mode": str(cfg.get("execution_mode", "") or ""),
            "checkpoint_path": str(cfg.get("checkpoint_path", "") or ""),
        }
    return out


def runtime_role_metadata(roles: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact paper-role metadata for result rows and script payloads."""

    out: Dict[str, Any] = {}
    for role, raw_cfg in normalize_roles(roles).items():
        if not isinstance(raw_cfg, Mapping):
            continue
        cfg = dict(raw_cfg)
        out[str(role)] = {
            "surface": str(cfg.get("surface", "") or ""),
            "engine": str(cfg.get("engine", "") or ""),
            "model": str(cfg.get("model", "") or ""),
            "base_url": str(cfg.get("base_url", "") or ""),
            "execution_mode": str(cfg.get("execution_mode", "") or ""),
            "checkpoint_path": str(cfg.get("checkpoint_path", "") or ""),
            "defaulted_from": str(cfg.get("defaulted_from", "") or ""),
        }
    return out


def runtime_method_ref(
    *,
    method_id: str,
    runner_id: str = "",
    surfaces: Mapping[str, Any] | None = None,
    roles: Mapping[str, Any] | None = None,
    oracle: Mapping[str, Any] | None = None,
    adapter: str = "runtime_eval",
) -> Dict[str, Any]:
    surface_meta = runtime_surface_metadata(surfaces or {})
    role_meta = runtime_role_metadata(roles or {})
    scorer = dict(role_meta.get("scorer", {}) or {})
    if not scorer:
        scorer = dict(surface_meta.get("chat_openai", {}) or {})
    return {
        "method_id": str(method_id or "runtime_eval"),
        "variant": str(runner_id or method_id or ""),
        "engine": str(scorer.get("engine", "") or ""),
        "model": "",
        "adapter": str(adapter),
        "metadata": metadata_with_roles(
            {
                "runner_id": str(runner_id or ""),
                "surfaces": surface_meta,
            },
            roles=role_meta,
            oracle=dict(oracle or {}),
        ),
    }


__all__ = ["runtime_method_ref", "runtime_role_metadata", "runtime_surface_metadata"]
