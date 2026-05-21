from __future__ import annotations

import os


_CPU_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def apply_cpu_thread_limits(*, threads: int = 1) -> None:
    """Best-effort clamp common BLAS/OpenMP thread env vars to avoid oversubscription."""
    n = str(int(max(1, threads)))
    for key in _CPU_THREAD_ENV_VARS:
        os.environ.setdefault(key, n)

    # Torch has its own thread pools; clamp if present. (CPU-only stance.)
    try:  # pragma: no cover
        import torch

        torch.set_num_threads(int(max(1, threads)))
        torch.set_num_interop_threads(int(max(1, threads)))
    except Exception:
        pass
