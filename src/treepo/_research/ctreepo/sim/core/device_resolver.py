"""Shared GPU/MIG device discovery and environment utilities.

Consolidates device detection logic that was previously duplicated across
run_tree_neural_full_doc_mig.py and run_markov_optimization_tradeoff_pipeline.py.
"""
from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


# ---------------------------------------------------------------------------
# Thread environment defaults
# ---------------------------------------------------------------------------

THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def set_thread_env_defaults() -> None:
    """Set single-threaded defaults for common BLAS/OpenMP env vars."""
    for key in THREAD_ENV_KEYS:
        os.environ.setdefault(key, "1")


# ---------------------------------------------------------------------------
# MIG slice inventory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MigSliceInfo:
    """Complete MIG device inventory entry with memory stats."""
    uuid: str
    gpu_index: int
    mig_index: int
    total_mib: int = 0
    used_mib: int = 0
    free_mib: int = 0


def _parse_mib_text(value: str | None) -> int:
    """Extract integer MiB from strings like '40960 MiB'."""
    if not value:
        return 0
    try:
        return int(str(value).strip().split()[0])
    except Exception:
        return 0


def parse_mig_uuids(value: str) -> List[str]:
    """Parse a comma/space-separated string of MIG UUIDs."""
    return [
        token.strip()
        for token in str(value or "").replace(",", " ").split()
        if token.strip()
    ]


def discover_mig_uuids() -> List[str]:
    """Discover all MIG UUIDs via nvidia-smi -L."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    uuids: List[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "MIG" not in line or "UUID:" not in line:
            continue
        uuids.append(line.split("UUID: ", 1)[1].rstrip(")"))
    return uuids


def parse_mig_layout_from_nvidia_smi_listing(listing: str) -> List[Dict[str, Any]]:
    """Parse GPU/MIG layout from nvidia-smi -L output text."""
    entries: List[Dict[str, Any]] = []
    current_gpu_index: int | None = None
    current_gpu_uuid = ""
    for raw_line in str(listing or "").splitlines():
        line = str(raw_line).strip()
        if not line:
            continue
        if line.startswith("GPU ") and "UUID:" in line:
            try:
                prefix, uuid_part = line.split("UUID: ", 1)
                current_gpu_index = int(prefix.split("GPU ", 1)[1].split(":", 1)[0])
                current_gpu_uuid = uuid_part.rstrip(")")
            except Exception:
                current_gpu_index = None
                current_gpu_uuid = ""
            continue
        if "MIG" not in line or "UUID:" not in line:
            continue
        if current_gpu_index is None or not current_gpu_uuid:
            continue
        mig_uuid = line.split("UUID: ", 1)[1].rstrip(")")
        entries.append(
            {
                "gpu_index": int(current_gpu_index),
                "gpu_uuid": str(current_gpu_uuid),
                "mig_uuid": str(mig_uuid),
            }
        )
    return entries


def discover_mig_layout() -> List[Dict[str, Any]]:
    """Discover full GPU/MIG layout via nvidia-smi -L."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    return parse_mig_layout_from_nvidia_smi_listing(result.stdout)


def mig_layout_by_uuid(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index layout entries by MIG UUID for O(1) lookup."""
    out: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        mig_uuid = str(entry.get("mig_uuid", "")).strip()
        if not mig_uuid:
            continue
        out[mig_uuid] = {
            "mig_uuid": mig_uuid,
            "gpu_index": int(entry.get("gpu_index", -1)),
            "gpu_uuid": str(entry.get("gpu_uuid", "")),
        }
    return out


def detect_mig_inventory() -> List[MigSliceInfo]:
    """Full MIG discovery with memory stats via nvidia-smi -L + nvidia-smi -q -x."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    uuid_by_slot: Dict[tuple[int, int], str] = {}
    current_gpu_index = -1
    for raw_line in listing.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("GPU "):
            prefix = line.split(":", 1)[0]
            try:
                current_gpu_index = int(prefix.split()[1])
            except Exception:
                current_gpu_index = -1
            continue
        if (
            current_gpu_index >= 0
            and line.startswith("MIG ")
            and "Device" in line
            and "UUID:" in line
        ):
            try:
                device_fragment = line.split("Device", 1)[1]
                mig_index = int(device_fragment.split(":", 1)[0])
                mig_uuid = line.split("UUID:", 1)[1].rstrip(")").strip()
            except Exception:
                continue
            if mig_index >= 0 and mig_uuid:
                uuid_by_slot[(current_gpu_index, mig_index)] = mig_uuid

    inventory: List[MigSliceInfo] = []
    for gpu_index in sorted({key[0] for key in uuid_by_slot}):
        try:
            xml_result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_index), "-q", "-x"],
                check=True,
                capture_output=True,
                text=True,
            )
            root = ET.fromstring(xml_result.stdout)
        except Exception:
            root = None

        mig_stats: Dict[int, tuple[int, int, int]] = {}
        if root is not None:
            for gpu_elem in root.findall("gpu"):
                mig_devices = gpu_elem.find("mig_devices")
                if mig_devices is None:
                    continue
                for mig_elem in mig_devices.findall("mig_device"):
                    try:
                        mi = int(mig_elem.findtext("index") or "-1")
                    except Exception:
                        continue
                    if mi < 0:
                        continue
                    fb_usage = mig_elem.find("fb_memory_usage")
                    total_mib = _parse_mib_text(
                        fb_usage.findtext("total") if fb_usage is not None else None
                    )
                    used_mib = _parse_mib_text(
                        fb_usage.findtext("used") if fb_usage is not None else None
                    )
                    free_mib = _parse_mib_text(
                        fb_usage.findtext("free") if fb_usage is not None else None
                    )
                    mig_stats[mi] = (total_mib, used_mib, free_mib)

        for (slot_gpu_index, mig_index), mig_uuid in sorted(uuid_by_slot.items()):
            if slot_gpu_index != gpu_index:
                continue
            total_mib, used_mib, free_mib = mig_stats.get(mig_index, (0, 0, 0))
            inventory.append(
                MigSliceInfo(
                    uuid=str(mig_uuid),
                    gpu_index=int(gpu_index),
                    mig_index=int(mig_index),
                    total_mib=int(total_mib),
                    used_mib=int(used_mib),
                    free_mib=int(free_mib),
                )
            )
    return inventory


# ---------------------------------------------------------------------------
# Device filtering and ordering
# ---------------------------------------------------------------------------

def filter_available_devices(
    inventory: Sequence[MigSliceInfo],
    *,
    min_free_fraction: float = 0.8,
    max_used_mib: int = 1024,
) -> List[str]:
    """Filter MIG devices to those that are likely available (sufficient free memory)."""
    return [
        str(row.uuid)
        for row in inventory
        if str(row.uuid).strip()
        and (
            row.total_mib <= 0
            or (
                row.free_mib >= int(min_free_fraction * row.total_mib)
                and row.used_mib <= max_used_mib
            )
        )
    ]


def interleave_devices_by_physical_gpu(
    tokens: Sequence[str],
    *,
    layout_by_uuid: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    """Reorder MIG devices in round-robin across physical GPUs."""
    grouped: Dict[str, List[str]] = {}
    unknown: List[str] = []
    for token in [str(value) for value in tokens]:
        info = layout_by_uuid.get(str(token))
        if info is None:
            unknown.append(str(token))
            continue
        gpu_key = str(info.get("gpu_uuid", "") or f"gpu_index_{int(info.get('gpu_index', -1))}")
        grouped.setdefault(gpu_key, []).append(str(token))
    interleaved: List[str] = []
    while any(grouped.values()):
        for gpu_key in list(grouped.keys()):
            if not grouped[gpu_key]:
                continue
            interleaved.append(str(grouped[gpu_key].pop(0)))
    interleaved.extend(unknown)
    return interleaved


def limit_devices_per_physical_gpu(
    tokens: Sequence[str],
    *,
    layout_by_uuid: Mapping[str, Mapping[str, Any]],
    max_per_physical_gpu: int,
) -> List[str]:
    """Enforce a per-physical-GPU device limit."""
    limit = int(max_per_physical_gpu)
    if limit <= 0:
        return [str(token) for token in tokens]
    counts: Dict[str, int] = {}
    kept: List[str] = []
    for token in [str(value) for value in tokens]:
        info = layout_by_uuid.get(str(token))
        if info is None:
            kept.append(str(token))
            continue
        gpu_key = str(info.get("gpu_uuid", "") or f"gpu_index_{int(info.get('gpu_index', -1))}")
        current = int(counts.get(gpu_key, 0))
        if current >= limit:
            continue
        counts[gpu_key] = int(current + 1)
        kept.append(str(token))
    return kept


def group_devices_by_physical_gpu(
    tokens: Sequence[str],
    *,
    layout_by_uuid: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Group MIG slices by parent physical GPU."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for token in [str(value) for value in tokens]:
        info = dict(layout_by_uuid.get(str(token), {}))
        gpu_key = str(info.get("gpu_uuid", "") or f"unknown::{token}")
        entry = grouped.setdefault(
            gpu_key,
            {
                "gpu_uuid": str(info.get("gpu_uuid", "")),
                "gpu_index": int(info.get("gpu_index", -1)),
                "mig_uuids": [],
            },
        )
        entry["mig_uuids"].append(str(token))
    return list(grouped.values())


# ---------------------------------------------------------------------------
# Unified device resolution
# ---------------------------------------------------------------------------

def resolve_devices(
    *,
    explicit_migs: str = "",
    device_mode: str = "auto",
    max_workers: int = 0,
) -> List[str]:
    """Resolve available devices from explicit list, auto-detection, or CPU mode.

    Args:
        explicit_migs: Comma/space-separated MIG UUIDs (takes priority).
        device_mode: "auto" (detect available, fall back to all, then CPU),
                     "gpu" (detect, fail if none), or "cpu" (no GPU).
        max_workers: Limit returned devices to this count (0 = no limit).

    Returns:
        List of device labels. Empty string means CPU.
    """
    mode = str(device_mode).strip().lower() or "auto"
    if mode == "cpu":
        return [""]

    explicit = parse_mig_uuids(explicit_migs)
    if explicit:
        devices = explicit
    else:
        inventory = detect_mig_inventory()
        devices = filter_available_devices(inventory)
        if not devices:
            devices = [str(row.uuid) for row in inventory if str(row.uuid).strip()]

    if not devices:
        if mode == "auto":
            return [""]
        return []

    if max_workers > 0:
        devices = devices[:max_workers]
    return devices


# ---------------------------------------------------------------------------
# Worker environment construction
# ---------------------------------------------------------------------------

def build_worker_env(
    device_label: str,
    *,
    use_cuda: bool = True,
) -> Dict[str, str]:
    """Build subprocess environment with CUDA_VISIBLE_DEVICES and thread defaults."""
    env = dict(os.environ)
    for key in THREAD_ENV_KEYS:
        env.setdefault(key, "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if use_cuda and device_label:
        env["CUDA_VISIBLE_DEVICES"] = str(device_label)
    elif not use_cuda:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def worker_device_context() -> Dict[str, Any]:
    """Inspect the current process's CUDA_VISIBLE_DEVICES."""
    visible_devices = [
        token.strip()
        for token in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).replace(",", " ").split()
        if token.strip()
    ]
    return {
        "cuda_visible_devices": list(visible_devices),
        "primary_visible_device": str(visible_devices[0]) if visible_devices else "",
    }
