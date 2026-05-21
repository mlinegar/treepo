from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


VALID_DEVICE_MODES: Tuple[str, ...] = ("auto", "cpu", "cuda")


def validate_device_mode(device: str) -> str:
    mode = str(device).strip().lower()
    if mode not in VALID_DEVICE_MODES:
        raise ValueError(f"device must be one of {VALID_DEVICE_MODES}")
    return mode


def configure_torch_runtime(torch_module: Any, *, torch_threads: int = 0) -> None:
    if int(torch_threads) <= 0:
        return
    try:
        torch_module.set_num_threads(int(torch_threads))
    except RuntimeError:
        pass
    if hasattr(torch_module, "set_num_interop_threads"):
        try:
            torch_module.set_num_interop_threads(int(torch_threads))
        except RuntimeError:
            pass


def resolve_torch_device(
    *,
    torch_module: Any,
    device: str,
    cuda_device: Optional[int] = None,
) -> Tuple[Any, Dict[str, object]]:
    requested = validate_device_mode(str(device))
    resolved = requested
    if resolved == "auto":
        resolved = "cuda" if bool(torch_module.cuda.is_available()) else "cpu"

    if resolved == "cuda":
        if not bool(torch_module.cuda.is_available()):
            raise RuntimeError("CUDA requested but not available")
        n_cuda = int(torch_module.cuda.device_count())
        if cuda_device is not None:
            idx = int(cuda_device)
            if idx < 0 or idx >= n_cuda:
                raise ValueError(f"cuda_device={idx} out of range; available devices: 0..{n_cuda - 1}")
            torch_module.cuda.set_device(idx)
            device_obj = torch_module.device(f"cuda:{idx}")
        else:
            device_obj = torch_module.device("cuda")
    else:
        device_obj = torch_module.device("cpu")

    return device_obj, torch_runtime_metadata(
        torch_module=torch_module,
        device=device_obj,
        requested_mode=requested,
    )


def torch_runtime_metadata(
    *,
    torch_module: Any,
    device: Any,
    requested_mode: str,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "device_requested": str(validate_device_mode(str(requested_mode))),
        "device_used": str(device),
        "device_mode_resolved": str(getattr(device, "type", device)),
    }
    if getattr(device, "type", None) == "cuda":
        try:
            current = int(torch_module.cuda.current_device())
            meta["cuda_current_device"] = current
            meta["cuda_device_name"] = str(torch_module.cuda.get_device_name(current))
        except Exception:
            pass
    return meta


__all__ = [
    "VALID_DEVICE_MODES",
    "configure_torch_runtime",
    "resolve_torch_device",
    "torch_runtime_metadata",
    "validate_device_mode",
]
