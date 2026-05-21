"""
Shared persisted probe-cache primitives for tree-batch autotuning.

These helpers are intentionally backend-agnostic so neural-tree and other
batching paths can reuse the same telemetry and cache surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


AUTOTUNE_PROBE_CACHE_VERSION = 1
DEFAULT_AUTOTUNE_PROBE_CACHE_DIR = Path("outputs/tree_autotune_probe_cache")
AUTOTUNE_PROBE_CACHE_DIR_ENV = "TT_TREE_AUTOTUNE_PROBE_CACHE_DIR"


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8", errors="ignore")).hexdigest()


def resolve_autotune_probe_cache_dir(
    *,
    env_var: str = AUTOTUNE_PROBE_CACHE_DIR_ENV,
    default_dir: Path = DEFAULT_AUTOTUNE_PROBE_CACHE_DIR,
) -> Path:
    raw = str(os.getenv(env_var, "") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(default_dir)


def build_probe_cache_key(
    *,
    model_signature: Mapping[str, Any],
    pack_mode: str,
    topology_signature: str,
    probe_mode: str,
    device_class_signature: Mapping[str, Any],
) -> str:
    return _stable_digest(
        {
            "version": int(AUTOTUNE_PROBE_CACHE_VERSION),
            "model_signature": dict(model_signature),
            "pack_mode": str(pack_mode),
            "topology_signature": str(topology_signature),
            "probe_mode": str(probe_mode),
            "device_class_signature": dict(device_class_signature),
        }
    )


def classify_device_signature(
    *,
    device_name: str,
    total_memory_bytes: int,
    capability: Sequence[int] = (),
) -> Dict[str, Any]:
    rendered_name = str(device_name or "").strip()
    mig_match = re.search(r"\bMIG\s+([^\s]+)\b", rendered_name)
    mig_profile = str(mig_match.group(1)) if mig_match else ""
    return {
        "device_name": rendered_name,
        "total_memory_bytes": int(max(0, int(total_memory_bytes))),
        "compute_capability": tuple(int(v) for v in capability),
        "is_mig": bool(mig_profile),
        "mig_profile": str(mig_profile or "full_gpu"),
    }


@dataclass(frozen=True)
class ProbeCandidateProfile:
    candidate_docs: int
    pack_time_s: float
    forward_backward_time_s: float
    peak_reserved_gb: float
    peak_allocated_gb: float
    stop_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_docs": int(self.candidate_docs),
            "pack_time_s": float(self.pack_time_s),
            "forward_backward_time_s": float(self.forward_backward_time_s),
            "peak_reserved_gb": float(self.peak_reserved_gb),
            "peak_allocated_gb": float(self.peak_allocated_gb),
            "stop_reason": str(self.stop_reason),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProbeCandidateProfile":
        return cls(
            candidate_docs=int(payload.get("candidate_docs", 0) or 0),
            pack_time_s=float(payload.get("pack_time_s", 0.0) or 0.0),
            forward_backward_time_s=float(
                payload.get("forward_backward_time_s", 0.0) or 0.0
            ),
            peak_reserved_gb=float(payload.get("peak_reserved_gb", 0.0) or 0.0),
            peak_allocated_gb=float(payload.get("peak_allocated_gb", 0.0) or 0.0),
            stop_reason=str(payload.get("stop_reason", "") or ""),
        )


@dataclass(frozen=True)
class ProbeRunProfile:
    probe_mode: str
    topology_signature: str
    selected_docs_cap: int
    heuristic_docs_cap: int
    max_candidate_docs: int
    target_fraction: float
    cache_key: str
    cache_hit: bool
    total_wall_time_s: float
    stop_reason: str = ""
    cached_source_wall_time_s: float = 0.0
    candidate_profiles: Tuple[ProbeCandidateProfile, ...] = tuple()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "probe_mode": str(self.probe_mode),
            "topology_signature": str(self.topology_signature),
            "selected_docs_cap": int(self.selected_docs_cap),
            "heuristic_docs_cap": int(self.heuristic_docs_cap),
            "max_candidate_docs": int(self.max_candidate_docs),
            "target_fraction": float(self.target_fraction),
            "cache_key": str(self.cache_key),
            "cache_hit": bool(self.cache_hit),
            "total_wall_time_s": float(self.total_wall_time_s),
            "stop_reason": str(self.stop_reason),
            "cached_source_wall_time_s": float(self.cached_source_wall_time_s),
            "candidate_profiles": [
                item.as_dict() for item in self.candidate_profiles
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProbeRunProfile":
        return cls(
            probe_mode=str(payload.get("probe_mode", "") or ""),
            topology_signature=str(payload.get("topology_signature", "") or ""),
            selected_docs_cap=int(payload.get("selected_docs_cap", 0) or 0),
            heuristic_docs_cap=int(payload.get("heuristic_docs_cap", 0) or 0),
            max_candidate_docs=int(payload.get("max_candidate_docs", 0) or 0),
            target_fraction=float(payload.get("target_fraction", 0.0) or 0.0),
            cache_key=str(payload.get("cache_key", "") or ""),
            cache_hit=bool(payload.get("cache_hit", False)),
            total_wall_time_s=float(payload.get("total_wall_time_s", 0.0) or 0.0),
            stop_reason=str(payload.get("stop_reason", "") or ""),
            cached_source_wall_time_s=float(
                payload.get("cached_source_wall_time_s", 0.0) or 0.0
            ),
            candidate_profiles=tuple(
                ProbeCandidateProfile.from_dict(item)
                for item in list(payload.get("candidate_profiles", ()) or ())
            ),
        )


@dataclass(frozen=True)
class ProbeCacheEntry:
    cache_key: str
    cache_version: int
    created_at_utc: str
    model_signature: Dict[str, Any]
    pack_mode: str
    topology_signature: str
    probe_mode: str
    device_class_signature: Dict[str, Any]
    selected_docs_cap: int
    run_profile: ProbeRunProfile

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cache_key": str(self.cache_key),
            "cache_version": int(self.cache_version),
            "created_at_utc": str(self.created_at_utc),
            "model_signature": dict(self.model_signature),
            "pack_mode": str(self.pack_mode),
            "topology_signature": str(self.topology_signature),
            "probe_mode": str(self.probe_mode),
            "device_class_signature": dict(self.device_class_signature),
            "selected_docs_cap": int(self.selected_docs_cap),
            "run_profile": self.run_profile.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Optional["ProbeCacheEntry"]:
        version = int(payload.get("cache_version", 0) or 0)
        if version != int(AUTOTUNE_PROBE_CACHE_VERSION):
            return None
        return cls(
            cache_key=str(payload.get("cache_key", "") or ""),
            cache_version=version,
            created_at_utc=str(payload.get("created_at_utc", "") or ""),
            model_signature=dict(payload.get("model_signature", {}) or {}),
            pack_mode=str(payload.get("pack_mode", "") or ""),
            topology_signature=str(payload.get("topology_signature", "") or ""),
            probe_mode=str(payload.get("probe_mode", "") or ""),
            device_class_signature=dict(payload.get("device_class_signature", {}) or {}),
            selected_docs_cap=int(payload.get("selected_docs_cap", 0) or 0),
            run_profile=ProbeRunProfile.from_dict(
                dict(payload.get("run_profile", {}) or {})
            ),
        )


@dataclass
class ProbeCacheStore:
    root_dir: Path = field(default_factory=resolve_autotune_probe_cache_dir)

    def _entry_path(self, cache_key: str) -> Path:
        safe_key = str(cache_key or "").strip()
        if not safe_key:
            raise ValueError("cache_key must be non-empty")
        return Path(self.root_dir) / f"{safe_key}.json"

    def get(self, cache_key: str) -> Optional[ProbeCacheEntry]:
        path = self._entry_path(cache_key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception:
            return None
        return ProbeCacheEntry.from_dict(payload)

    def put(self, entry: ProbeCacheEntry) -> Path:
        path = self._entry_path(entry.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(entry.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return path
