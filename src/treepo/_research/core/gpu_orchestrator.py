"""
GPU Orchestrator for Dynamic Model Allocation.

Manages vLLM/SGLang servers to dynamically allocate GPUs across pipeline phases.

When sleep-mode endpoints are available (vLLM), servers are pre-loaded and
woken/slept for fast transitions. For backends without sleep support (SGLang),
the orchestrator uses explicit stop/start transitions.

Modes:
- "task_dp2": Task model on both GPU pairs (DP=2 for ~2x throughput)
- "dual_model": Task on GPUs 0,1 + GenRM on GPUs 2,3

Usage:
    orchestrator = GPUOrchestrator(config)
    await orchestrator.initialize()  # Start all servers, sleep secondary ones

    # Phase 1: Document processing with DP=2
    await orchestrator.enter_task_dp2_mode()
    ports = orchestrator.get_active_task_ports()  # [8000, 8002]

    # Phase 1.5: Need GenRM
    await orchestrator.enter_dual_model_mode()

    # Cleanup
    await orchestrator.shutdown()
"""

import asyncio
import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
from urllib.parse import urlparse

import aiohttp
import yaml

from treepo._research.core.vllm_runtime import resolve_vllm_runtime_flags

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend abstraction protocol (DualPath Opt 4.5)
# ---------------------------------------------------------------------------

@runtime_checkable
class ServerBackend(Protocol):
    """Protocol for inference server backends (vLLM, SGLang, etc.).

    Enables the GPU orchestrator to work with either vLLM or SGLang
    without rewriting orchestration logic.  ``ManagedServer`` implements
    this protocol for vLLM; a future ``SGLangManagedServer`` would
    provide the SGLang-specific implementation.
    """

    @property
    def port(self) -> int: ...

    @property
    def is_running(self) -> bool: ...

    @property
    def is_sleeping(self) -> bool: ...

    async def start(self) -> None: ...

    async def sleep(self, level: int = 1, timeout: float = 30.0) -> bool: ...

    async def wake(self, timeout: float = 60.0) -> bool: ...

    def stop(self) -> None: ...

    async def health_check(self) -> bool: ...

    async def is_server_sleeping(self) -> Optional[bool]: ...


def _looks_like_cuda_oom(message: str) -> bool:
    """Heuristic check for CUDA OOM startup failures in vLLM logs/errors."""
    text = (message or "").lower()
    markers = (
        "cuda out of memory",
        "torch.outofmemoryerror",
        "outofmemoryerror",
        "c10::outofmemoryerror",
        "tried to allocate",
    )
    return any(marker in text for marker in markers)


def _listener_pids_on_port(port: int) -> List[int]:
    """Return PIDs for LISTEN sockets bound to the given TCP port."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{int(port)}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids: List[int] = []
            for line in result.stdout.splitlines():
                try:
                    pids.append(int(line.strip()))
                except (TypeError, ValueError):
                    continue
            return sorted(set(pids))
        return []
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Error checking port %d listener via lsof: %s", int(port), exc)

    try:
        result = subprocess.run(
            ["ss", "-ltnp", f"sport = :{int(port)}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        pids: List[int] = []
        for line in result.stdout.splitlines():
            for token in line.split():
                if "pid=" not in token:
                    continue
                try:
                    pid_part = token.split("pid=", 1)[1]
                    pid_text = pid_part.split(",", 1)[0].rstrip(")")
                    pids.append(int(pid_text))
                except Exception:
                    continue
        return sorted(set(pids))
    except FileNotFoundError:
        logger.warning("Neither lsof nor ss available to check listener on port %d", int(port))
    except Exception as exc:
        logger.warning("Error checking port %d listener via ss: %s", int(port), exc)
    return []


def kill_process_on_port(port: int) -> bool:
    """Kill listener processes bound to the given TCP port.

    Returns True if a process was killed, False if no process was found.
    """
    pids = _listener_pids_on_port(int(port))
    if not pids:
        return False

    for pid_int in pids:
        try:
            logger.info("Killing existing process %d on port %d", int(pid_int), int(port))
            os.kill(int(pid_int), signal.SIGTERM)
        except (ValueError, ProcessLookupError):
            continue
        except Exception as exc:
            logger.warning("Failed to SIGTERM pid=%d on port %d: %s", int(pid_int), int(port), exc)

    # Wait a moment for cleanup
    time.sleep(1.0)
    return True


def _prepend_env_path(env: Dict[str, str], key: str, value: str) -> None:
    if not value:
        return
    current = env.get(key, "")
    parts = [part for part in current.split(":") if part]
    if value in parts:
        return
    env[key] = f"{value}:{current}" if current else value


def _nvcc_binary_works(nvcc_path: Optional[Path]) -> bool:
    if nvcc_path is None:
        return False
    candidate = Path(nvcc_path)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return False
    try:
        result = subprocess.run(
            [str(candidate), "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


@lru_cache(maxsize=8)
def _resolve_venv_site_packages(venv_path: str) -> Optional[Path]:
    root = Path(venv_path)
    candidates: List[Path] = []
    for base in (root / "lib", root / "local" / "lib"):
        if not base.is_dir():
            continue
        candidates.extend(base.glob("python*/site-packages"))
        candidates.extend(base.glob("python*/dist-packages"))
    for candidate in sorted(candidates, reverse=True):
        if candidate.is_dir():
            return candidate
    return None


def _stable_cache_slug(raw: str, *, fallback: str = "profile") -> str:
    rendered = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(raw or "").strip().lower())
    rendered = rendered.strip("-")
    if not rendered:
        rendered = fallback
    return rendered[:48]


def _configure_flashinfer_workspace_env(
    env: Dict[str, str],
    *,
    venv_path: str,
    profile: str,
) -> None:
    if env.get("FLASHINFER_WORKSPACE_BASE"):
        return

    key_parts = [
        str(Path(venv_path).expanduser()),
        str(env.get("CUDA_HOME", "")),
        str(env.get("FLASHINFER_NVCC", "")),
        str(profile),
    ]
    digest = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:12]
    profile_slug = _stable_cache_slug(Path(str(profile)).name or str(profile), fallback="model")
    workspace_base = Path(tempfile.gettempdir()) / "thinkingtrees" / "flashinfer" / f"{profile_slug}-{digest}"
    try:
        workspace_base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    env["FLASHINFER_WORKSPACE_BASE"] = str(workspace_base)


def _configure_nvfp4_runtime_env(env: Dict[str, str], venv_path: str, profile: str) -> None:
    """
    Configure the environment so NVFP4/FlashInfer JIT kernels can load.

    When vLLM is launched programmatically (vs. `scripts/start_vllm.sh`), we
    won't necessarily have CUDA_HOME/LD_LIBRARY_PATH pointing at the pip-
    installed CUDA toolchain inside `venv_path`. FlashInfer's TVM-loaded shared
    objects depend on e.g. `libcudart.so.13`, so we must add those directories
    to the dynamic linker path.
    """
    if "nvfp4" not in str(profile).lower():
        return

    # ModelOpt NVFP4 MoE checkpoints require FlashInfer CUTLASS kernels
    # (vLLM backend selector: "throughput"). Force sane defaults even if the
    # parent environment provided empty/invalid values.
    fp4_enabled = str(env.get("VLLM_USE_FLASHINFER_MOE_FP4", "")).strip().lower()
    if fp4_enabled not in {"1", "true", "yes", "on"}:
        env["VLLM_USE_FLASHINFER_MOE_FP4"] = "1"

    backend = str(env.get("VLLM_FLASHINFER_MOE_BACKEND", "")).strip().lower()
    if backend != "throughput":
        env["VLLM_FLASHINFER_MOE_BACKEND"] = "throughput"

    current_cuda_home = (
        Path(str(env.get("CUDA_HOME", ""))).expanduser()
        if env.get("CUDA_HOME")
        else None
    )
    current_nvcc = (current_cuda_home / "bin" / "nvcc") if current_cuda_home else None
    if current_cuda_home is not None and not _nvcc_binary_works(current_nvcc):
        env.pop("CUDA_HOME", None)
        env.pop("CUDA_PATH", None)

    site_packages = _resolve_venv_site_packages(venv_path)
    if site_packages is None:
        logger.warning(
            "Could not locate site-packages for vLLM venv (%s); NVFP4 runtime may fail",
            venv_path,
        )
        flashinfer_nvcc = Path(env["FLASHINFER_NVCC"]) if env.get("FLASHINFER_NVCC") else None
        if flashinfer_nvcc is not None and not _nvcc_binary_works(flashinfer_nvcc):
            env.pop("FLASHINFER_NVCC", None)
            env.pop("CUDACXX", None)
        if "FLASHINFER_NVCC" not in env:
            path_nvcc = shutil.which("nvcc", path=env.get("PATH"))
            if path_nvcc and _nvcc_binary_works(Path(path_nvcc)):
                env["FLASHINFER_NVCC"] = path_nvcc
                env.setdefault("CUDACXX", path_nvcc)
        _configure_flashinfer_workspace_env(env, venv_path=venv_path, profile=profile)
        return

    cu13_root = site_packages / "nvidia" / "cu13"
    if cu13_root.is_dir():
        cu13_nvcc = cu13_root / "bin" / "nvcc"
        current_cuda_home = (
            Path(str(env.get("CUDA_HOME", ""))).expanduser()
            if env.get("CUDA_HOME")
            else None
        )
        current_nvcc = (current_cuda_home / "bin" / "nvcc") if current_cuda_home else None
        if _nvcc_binary_works(cu13_nvcc) and not _nvcc_binary_works(current_nvcc):
            env["CUDA_HOME"] = str(cu13_root)
        elif current_cuda_home is not None and not _nvcc_binary_works(current_nvcc):
            env.pop("CUDA_HOME", None)

        cuda_home = env.get("CUDA_HOME")
        if cuda_home:
            cuda_home_path = Path(str(cuda_home)).expanduser()
            _prepend_env_path(env, "PATH", str(cuda_home_path / "bin"))
            env["CUDA_PATH"] = str(cuda_home_path)
            cuda_home_nvcc = cuda_home_path / "bin" / "nvcc"
            if _nvcc_binary_works(cuda_home_nvcc):
                env.setdefault("FLASHINFER_NVCC", str(cuda_home_nvcc))
                env.setdefault("CUDACXX", str(cuda_home_nvcc))
        else:
            env.pop("CUDA_PATH", None)

        for lib_dir in (cu13_root / "lib64", cu13_root / "lib"):
            if lib_dir.is_dir():
                _prepend_env_path(env, "LD_LIBRARY_PATH", str(lib_dir))

    flashinfer_nvcc = Path(env["FLASHINFER_NVCC"]) if env.get("FLASHINFER_NVCC") else None
    if flashinfer_nvcc is not None and not _nvcc_binary_works(flashinfer_nvcc):
        env.pop("FLASHINFER_NVCC", None)
        env.pop("CUDACXX", None)
    if "FLASHINFER_NVCC" not in env:
        path_nvcc = shutil.which("nvcc", path=env.get("PATH"))
        if path_nvcc and _nvcc_binary_works(Path(path_nvcc)):
            env["FLASHINFER_NVCC"] = path_nvcc
            env.setdefault("CUDACXX", path_nvcc)

    if Path("/lib/x86_64-linux-gnu").is_dir():
        _prepend_env_path(env, "LD_LIBRARY_PATH", "/lib/x86_64-linux-gnu")

    curand_include = site_packages / "nvidia" / "curand" / "include"
    if curand_include.is_dir():
        _prepend_env_path(env, "CPATH", str(curand_include))

    _configure_flashinfer_workspace_env(env, venv_path=venv_path, profile=profile)


@lru_cache(maxsize=64)
def _python_supports_vllm_with_cuda(python_path: str, cuda_devices: Optional[str]) -> bool:
    """Return True if interpreter can import vLLM and sees >=1 CUDA device."""
    candidate = Path(str(python_path)).expanduser()
    if not candidate.exists():
        return False

    probe = (
        "import importlib.util, sys\n"
        "spec = importlib.util.find_spec('vllm')\n"
        "if spec is None:\n"
        "    raise SystemExit(11)\n"
        "import torch\n"
        "ok = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0\n"
        "raise SystemExit(0 if ok else 12)\n"
    )
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        result = subprocess.run(
            [str(candidate), "-c", probe],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _resolve_vllm_python_interpreter(venv_path: str, cuda_devices: str) -> Tuple[str, bool]:
    """Pick interpreter and whether CUDA_VISIBLE_DEVICES isolation should be applied."""
    preferred = str(Path(venv_path) / "bin" / "python")
    repo_root = Path(__file__).resolve().parents[2]
    repo_venv_python3 = str(repo_root / "venv" / "bin" / "python3")
    repo_venv_python = str(repo_root / "venv" / "bin" / "python")
    override = str(
        os.getenv("TT_VLLM_PYTHON")
        or os.getenv("VLLM_PYTHON")
        or ""
    ).strip()
    candidates = []
    for raw in (
        override,
        preferred,
        repo_venv_python3,
        repo_venv_python,
        str(sys.executable or "").strip(),
        str(shutil.which("python3") or "").strip(),
        str(shutil.which("python") or "").strip(),
    ):
        if not raw:
            continue
        rendered = str(Path(raw).expanduser())
        if rendered not in candidates:
            candidates.append(rendered)

    preferred_path = str(Path(preferred).expanduser())
    first_existing: Optional[str] = preferred_path if Path(preferred_path).exists() else None
    for candidate in candidates:
        if not Path(candidate).exists():
            continue
        if first_existing is None:
            first_existing = candidate
        if _python_supports_vllm_with_cuda(candidate, str(cuda_devices)):
            return candidate, True

    for candidate in candidates:
        if not Path(candidate).exists():
            continue
        if _python_supports_vllm_with_cuda(candidate, None):
            logger.warning(
                "CUDA device isolation probe failed for all candidates (devices='%s'); "
                "using interpreter without CUDA_VISIBLE_DEVICES isolation: %s",
                str(cuda_devices),
                candidate,
            )
            return candidate, False

    if first_existing is not None:
        logger.warning(
            "No vLLM Python candidate passed CUDA probe for devices='%s'; "
            "falling back to first existing interpreter: %s",
            str(cuda_devices),
            first_existing,
        )
        return first_existing, True

    logger.warning(
        "No Python interpreter candidates found for vLLM; using preferred path: %s",
        preferred,
    )
    return preferred, True


class OrchestratorMode(Enum):
    """Current GPU allocation mode."""
    TASK_DP2 = "task_dp2"       # Task on both GPU pairs (DP=2)
    DUAL_MODEL = "dual_model"   # Task + GenRM on separate GPU pairs
    UNINITIALIZED = "uninitialized"


@dataclass
class ServerConfig:
    """Configuration for a single managed inference server."""
    profile: str
    port: int
    cuda_devices: str
    tensor_parallel: int
    backend: str = "vllm"
    max_model_len: int = 32768
    runtime_args: List[str] = field(default_factory=list)
    enable_prefix_caching: bool = False
    enable_sleep_mode: bool = True
    supports_sleep_mode: bool = True
    startup_timeout: float = 300.0
    gpu_memory_utilization: float = 0.90


@dataclass
class OrchestratorConfig:
    """Configuration for the GPU orchestrator."""
    # Server configs
    task_primary: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="nemotron-30b-nvfp4",
        port=8000,
        cuda_devices="0,1",
        tensor_parallel=2,
        max_model_len=32768,
    ))
    task_replica: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="nemotron-30b-nvfp4",
        port=8002,
        cuda_devices="2,3",
        tensor_parallel=2,
        max_model_len=32768,
    ))
    genrm: ServerConfig = field(default_factory=lambda: ServerConfig(
        profile="genrm-nvfp4",
        port=8001,
        cuda_devices="2,3",
        tensor_parallel=2,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
    ))
    embedding: Optional[ServerConfig] = None

    # Paths
    venv_path: str = "~/vllm-env"
    sglang_venv_path: str = "~/sglang-env"
    config_path: Optional[Path] = None

    # Timeouts
    sleep_timeout: float = 30.0
    wake_timeout: float = 60.0
    health_check_interval: float = 2.0
    post_stop_settle_seconds: float = 6.0
    # Stability-first toggle for servers that share GPUs (task_replica/genrm).
    # When enabled, mode transitions stop the peer process instead of relying
    # on vLLM sleep mode, which avoids frequent wake/start OOM failures.
    shared_gpu_hard_quiesce: bool = False

    # Overlapped phase transition (DualPath Opt 3)
    enable_prewarm: bool = True
    prewarm_threshold: float = 0.85  # Start prewarm at 85% completion

    # KV-cache persistence via LMCache (Phase 6.1)
    kv_persistence_enabled: bool = False
    kv_persistence_backend: str = "lmcache"  # "lmcache" or "native"
    kv_persistence_cpu_gb: float = 5.0
    kv_persistence_disk_path: str = "/tmp/thinkingtrees_kv_cache"
    kv_persistence_disk_gb: float = 50.0
    kv_persistence_chunk_size: int = 256

    # Optional: disable GenRM management entirely for runs that do not need it.
    # This avoids preloading large GenRM weights in dynamic GPU mode when
    # `--enable-genrm` is not set.
    enable_genrm: bool = True
    # Optional: manage embedding endpoint lifecycle under orchestrator control.
    # Useful to avoid keeping embedding model resident during long task/genrm-only phases.
    manage_embedding: bool = False
    quiesce_embedding_when_idle: bool = True

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        *,
        task_model_profile_override: Optional[str] = None,
    ) -> "OrchestratorConfig":
        """Load orchestrator config from settings.yaml."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        orch_cfg = cfg.get("orchestration", {})
        vllm_cfg = cfg.get("vllm", {})
        servers_cfg = cfg.get("servers", {}) if isinstance(cfg.get("servers", {}), dict) else {}
        inference_cfg = cfg.get("inference", {}) if isinstance(cfg.get("inference", {}), dict) else {}
        backend_cfg = inference_cfg.get("backend", {}) if isinstance(inference_cfg.get("backend", {}), dict) else {}
        models = vllm_cfg.get("models", {})

        # Get model profiles
        task_profile = (
            str(task_model_profile_override or "").strip()
            or str(os.getenv("TT_ORCH_TASK_MODEL_PROFILE", "") or "").strip()
            or str(orch_cfg.get("task_model_profile", vllm_cfg.get("default", "nemotron-30b-nvfp4")) or "nemotron-30b-nvfp4").strip()
        )
        genrm_profile = orch_cfg.get("genrm_profile", "genrm-nvfp4")
        embedding_profile = orch_cfg.get(
            "embedding_profile",
            servers_cfg.get("embedding_profile", "qwen3-embedding-8b"),
        )

        if task_profile not in models:
            fallback_profile = str(
                orch_cfg.get("task_model_profile", vllm_cfg.get("default", "nemotron-30b-nvfp4"))
                or "nemotron-30b-nvfp4"
            ).strip()
            logger.warning(
                "Dynamic task model profile '%s' not found in settings; falling back to '%s'.",
                task_profile,
                fallback_profile,
            )
            task_profile = fallback_profile

        # Get model configs
        task_model = models.get(task_profile, {})
        genrm_model = models.get(genrm_profile, {})
        task_runtime_args = resolve_vllm_runtime_flags(
            vllm_cfg=vllm_cfg,
            profile=str(task_profile),
        ).to_cli_args()
        genrm_runtime_args = resolve_vllm_runtime_flags(
            vllm_cfg=vllm_cfg,
            profile=str(genrm_profile),
        ).to_cli_args()
        prefix_cache = bool(vllm_cfg.get("enable_prefix_caching", False))
        task_primary_tp = int(orch_cfg.get("task_primary_tensor_parallel", task_model.get("tensor_parallel", 2)))
        task_replica_tp = int(orch_cfg.get("task_replica_tensor_parallel", task_model.get("tensor_parallel", 2)))
        genrm_tp = int(orch_cfg.get("genrm_tensor_parallel", genrm_model.get("tensor_parallel", 2)))
        embedding_model = models.get(str(embedding_profile), {})
        embedding_tp = int(orch_cfg.get("embedding_tensor_parallel", embedding_model.get("tensor_parallel", 1)))
        embedding_url = str(servers_cfg.get("embedding_url", "http://localhost:8003/v1") or "http://localhost:8003/v1")
        embedding_port = int(orch_cfg.get("embedding_port", urlparse(embedding_url).port or 8003))
        embedding_gpus = str(orch_cfg.get("embedding_gpus", "") or "").strip()
        if not embedding_gpus:
            replica_gpu_list = [
                chunk.strip()
                for chunk in str(orch_cfg.get("task_replica_gpus", "2,3")).split(",")
                if chunk.strip()
            ]
            embedding_gpus = replica_gpu_list[0] if replica_gpu_list else "2"
        manage_embedding = bool(orch_cfg.get("auto_manage_embedding_server", False))
        embedding_cfg: Optional[ServerConfig] = None
        if manage_embedding and embedding_model:
            embedding_runtime_args = resolve_vllm_runtime_flags(
                vllm_cfg=vllm_cfg,
                profile=str(embedding_profile),
            ).to_cli_args()
            embedding_cfg = ServerConfig(
                profile=str(embedding_profile),
                port=embedding_port,
                cuda_devices=embedding_gpus,
                tensor_parallel=max(1, embedding_tp),
                backend="vllm",
                max_model_len=embedding_model.get("max_model_len", 32768),
                runtime_args=list(embedding_runtime_args),
                enable_prefix_caching=prefix_cache,
                supports_sleep_mode=bool(orch_cfg.get("embedding_supports_sleep_mode", True)),
                gpu_memory_utilization=orch_cfg.get(
                    "embedding_gpu_memory_utilization",
                    min(float(vllm_cfg.get("gpu_memory_utilization", 0.90)), 0.90),
                ),
            )
        elif manage_embedding:
            logger.warning(
                "orchestration.auto_manage_embedding_server=true but embedding profile '%s' is missing in vllm.models; "
                "embedding lifecycle management disabled.",
                str(embedding_profile),
            )
            manage_embedding = False

        return cls(
            task_primary=ServerConfig(
                profile=task_profile,
                port=orch_cfg.get("task_primary_port", 8000),
                cuda_devices=orch_cfg.get("task_primary_gpus", "0,1"),
                tensor_parallel=max(1, task_primary_tp),
                backend="vllm",
                max_model_len=task_model.get("max_model_len", 32768),
                runtime_args=list(task_runtime_args),
                enable_prefix_caching=prefix_cache,
                supports_sleep_mode=bool(orch_cfg.get("task_primary_supports_sleep_mode", True)),
                gpu_memory_utilization=orch_cfg.get(
                    "task_primary_gpu_memory_utilization",
                    vllm_cfg.get("gpu_memory_utilization", 0.90),
                ),
            ),
            task_replica=ServerConfig(
                profile=task_profile,
                port=orch_cfg.get("task_replica_port", 8002),
                cuda_devices=orch_cfg.get("task_replica_gpus", "2,3"),
                tensor_parallel=max(1, task_replica_tp),
                backend="vllm",
                max_model_len=task_model.get("max_model_len", 32768),
                runtime_args=list(task_runtime_args),
                enable_prefix_caching=prefix_cache,
                supports_sleep_mode=bool(orch_cfg.get("task_replica_supports_sleep_mode", True)),
                gpu_memory_utilization=orch_cfg.get(
                    "task_replica_gpu_memory_utilization",
                    min(float(vllm_cfg.get("gpu_memory_utilization", 0.90)), 0.88),
                ),
            ),
            genrm=ServerConfig(
                profile=genrm_profile,
                port=orch_cfg.get("genrm_port", 8001),
                cuda_devices=orch_cfg.get("genrm_gpus", "2,3"),
                tensor_parallel=max(1, genrm_tp),
                backend="vllm",
                max_model_len=genrm_model.get("max_model_len", 32768),
                runtime_args=list(genrm_runtime_args),
                enable_prefix_caching=prefix_cache,
                supports_sleep_mode=bool(orch_cfg.get("genrm_supports_sleep_mode", True)),
                gpu_memory_utilization=orch_cfg.get("genrm_gpu_memory_utilization", 0.95),
            ),
            embedding=embedding_cfg,
            venv_path=(
                backend_cfg.get("vllm_venv_path")
                or orch_cfg.get("venv_path")
                or "~/vllm-env"
            ),
            sglang_venv_path=str(backend_cfg.get("sglang_venv_path") or "~/sglang-env"),
            config_path=config_path,
            sleep_timeout=orch_cfg.get("sleep_timeout", 30.0),
            wake_timeout=orch_cfg.get("wake_timeout", 60.0),
            post_stop_settle_seconds=orch_cfg.get("post_stop_settle_seconds", 6.0),
            shared_gpu_hard_quiesce=bool(orch_cfg.get("shared_gpu_hard_quiesce", False)),
            kv_persistence_enabled=bool(orch_cfg.get("kv_persistence", {}).get("enabled", False)),
            kv_persistence_backend=str(orch_cfg.get("kv_persistence", {}).get("backend", "lmcache")),
            kv_persistence_cpu_gb=float(orch_cfg.get("kv_persistence", {}).get("cpu_cache_gb", 5.0)),
            kv_persistence_disk_path=str(orch_cfg.get("kv_persistence", {}).get("disk_cache_path", "/tmp/thinkingtrees_kv_cache")),
            kv_persistence_disk_gb=float(orch_cfg.get("kv_persistence", {}).get("disk_cache_gb", 50.0)),
            kv_persistence_chunk_size=int(orch_cfg.get("kv_persistence", {}).get("chunk_size", 256)),
            manage_embedding=bool(manage_embedding and embedding_cfg is not None),
            quiesce_embedding_when_idle=bool(orch_cfg.get("quiesce_embedding_when_idle", True)),
        )


class ManagedServer:
    """Manages a single inference server process (vLLM or SGLang)."""

    def __init__(
        self,
        config: ServerConfig,
        venv_path: str,
        model_path: str,
        health_check_interval: float = 2.0,
        post_stop_settle_seconds: float = 6.0,
        kv_persistence: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.venv_path = venv_path
        self.model_path = model_path
        self.health_check_interval = health_check_interval
        self.post_stop_settle_seconds = max(0.0, float(post_stop_settle_seconds))
        self._kv_persistence = kv_persistence or {}

        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._is_sleeping = False
        self._attached_pids: List[int] = []

    @property
    def port(self) -> int:
        return self.config.port

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def is_running(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        return bool(_listener_pids_on_port(self.port))

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    async def start(self) -> None:
        """Start the managed server process and wait for readiness."""
        if self._process is not None and self._process.poll() is None:
            logger.warning("Server on port %d already running (owned process)", int(self.port))
            return

        # If something is already listening on the port, attempt to attach instead of killing.
        listener_pids = _listener_pids_on_port(self.port)
        if listener_pids:
            attached = await self._maybe_attach_existing(listener_pids)
            if attached:
                return

        # Kill any existing process on this port (from previous runs or incompatible servers)
        if kill_process_on_port(self.port):
            logger.info(f"Killed stale server on port {self.port}")
            # Wait a bit more for GPU memory to be released
            await asyncio.sleep(2)

        self._attached_pids = []
        backend_name = str(getattr(self.config, "backend", "vllm") or "vllm").strip().lower()
        logger.info("Starting %s server on port %d", backend_name, int(self.port))
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  CUDA devices: {self.config.cuda_devices}")
        logger.info(f"  Tensor parallel: {self.config.tensor_parallel}")
        logger.info(f"  Max model len: {self.config.max_model_len}")
        logger.info(f"  Prefix cache: {self.config.enable_prefix_caching}")
        logger.info(
            "  Sleep mode: %s (supports_sleep_mode=%s)",
            self.config.enable_sleep_mode,
            self.config.supports_sleep_mode,
        )
        if self.config.runtime_args:
            logger.info(f"  Runtime args: {' '.join(self.config.runtime_args)}")

        if backend_name == "sglang":
            script_path = Path(__file__).resolve().parents[2] / "scripts" / "start_sglang.sh"
            cmd = [
                "/bin/bash",
                str(script_path),
                str(self.config.profile),
                "--port", str(self.port),
                "--cuda-devices", str(self.config.cuda_devices),
                "--tensor-parallel", str(self.config.tensor_parallel),
                "--max-model-len", str(self.config.max_model_len),
                "--sglang-venv-path", str(self.venv_path),
            ]
            env = os.environ.copy()
            self._log_file = tempfile.NamedTemporaryFile(
                mode='w',
                prefix=f'sglang_port{self.port}_',
                suffix='.log',
                delete=False,
            )
            logger.info("  Log file: %s", self._log_file.name)

            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=env,
            )
            logger.info("Server process started (PID: %d)", int(self._process.pid))
            await self._wait_for_ready()
            self._is_sleeping = False
            return

        base_gmu = float(self.config.gpu_memory_utilization)
        effective_gmu = base_gmu
        attempted_oom_recovery = False

        while True:
            preferred_python_path = str(Path(self.venv_path) / "bin" / "python")
            python_path, use_cuda_device_isolation = _resolve_vllm_python_interpreter(
                self.venv_path,
                self.config.cuda_devices,
            )
            if python_path != preferred_python_path:
                logger.warning(
                    "vLLM interpreter fallback on port %d: %s (preferred %s)",
                    int(self.port),
                    python_path,
                    preferred_python_path,
                )
            cmd = [
                python_path,
                "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model_path,
                "--host", "0.0.0.0",
                "--port", str(self.port),
                "--tensor-parallel-size", str(self.config.tensor_parallel),
                "--max-model-len", str(self.config.max_model_len),
                "--gpu-memory-utilization", str(effective_gmu),
                "--trust-remote-code",
            ]

            if self.config.enable_prefix_caching:
                cmd.append("--enable-prefix-caching")

            if self.config.runtime_args:
                cmd.extend(self.config.runtime_args)

            cmd.append("--disable-log-requests")

            if self.config.enable_sleep_mode and self.config.supports_sleep_mode:
                cmd.append("--enable-sleep-mode")

            # KV-cache persistence via LMCache (Phase 6.1)
            if self._kv_persistence.get("enabled"):
                backend = str(self._kv_persistence.get("backend", "lmcache"))
                if backend == "lmcache":
                    import json as _json
                    cmd.extend([
                        "--kv-transfer-config",
                        _json.dumps({
                            "kv_connector": "LMCacheConnectorV1",
                            "kv_role": "kv_both",
                        }),
                    ])
                elif backend == "native":
                    import json as _json
                    cpu_blocks = int(self._kv_persistence.get("cpu_gb", 5.0) * 1000)
                    cmd.extend([
                        "--kv-transfer-config",
                        _json.dumps({
                            "kv_connector": "OffloadingConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {
                                "num_cpu_blocks": cpu_blocks,
                            },
                        }),
                    ])

            # Environment with CUDA device isolation and dev mode for sleep endpoints
            env = os.environ.copy()
            # Ensure vLLM venv binaries (e.g., `ninja`) are available even when this
            # orchestrator is invoked from a different Python environment.
            _prepend_env_path(env, "PATH", str(Path(self.venv_path) / "bin"))
            if use_cuda_device_isolation:
                env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
            else:
                env.pop("CUDA_VISIBLE_DEVICES", None)
                logger.warning(
                    "Launching vLLM on port %d without CUDA_VISIBLE_DEVICES isolation. "
                    "This mode is best-effort and may increase GPU contention.",
                    int(self.port),
                )
            if self.config.enable_sleep_mode and self.config.supports_sleep_mode:
                env["VLLM_SERVER_DEV_MODE"] = "1"
            _configure_nvfp4_runtime_env(env, venv_path=self.venv_path, profile=self.config.profile)
            if "nvfp4" in str(self.config.profile).lower():
                logger.info(
                    "  NVFP4 env: VLLM_USE_FLASHINFER_MOE_FP4=%s VLLM_FLASHINFER_MOE_BACKEND=%s",
                    str(env.get("VLLM_USE_FLASHINFER_MOE_FP4", "")),
                    str(env.get("VLLM_FLASHINFER_MOE_BACKEND", "")),
                )

            # LMCache environment variables (Phase 6.1)
            if self._kv_persistence.get("enabled") and self._kv_persistence.get("backend") == "lmcache":
                lmcache_cfg_path = self._kv_persistence.get("config_file")
                if lmcache_cfg_path:
                    env["LMCACHE_CONFIG_FILE"] = str(lmcache_cfg_path)
                env["LMCACHE_USE_EXPERIMENTAL"] = "True"

            # Create log file
            self._log_file = tempfile.NamedTemporaryFile(
                mode='w',
                prefix=f'vllm_port{self.port}_',
                suffix='.log',
                delete=False,
            )
            logger.info("  Log file: %s", self._log_file.name)
            logger.info("  Effective gpu_memory_utilization: %.2f", effective_gmu)

            # Start process
            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=env,
            )

            logger.info(f"Server process started (PID: {self._process.pid})")

            try:
                # Wait for ready
                await self._wait_for_ready()
                self._is_sleeping = False
                return
            except Exception as exc:
                err_text = str(exc)
                self.stop()
                if attempted_oom_recovery or not _looks_like_cuda_oom(err_text):
                    raise

                attempted_oom_recovery = True
                reduced_gmu = max(0.80, round(effective_gmu - 0.03, 2))
                logger.warning(
                    "Detected CUDA OOM while starting port %d. "
                    "Performing quick clear and retrying once%s.",
                    int(self.port),
                    f" (gmu {effective_gmu:.2f} -> {reduced_gmu:.2f})"
                    if reduced_gmu < effective_gmu
                    else "",
                )

                kill_process_on_port(self.port)
                await asyncio.sleep(max(2.0, self.post_stop_settle_seconds))
                effective_gmu = reduced_gmu

    def _model_ids_match(self, served_model_ids: List[str]) -> bool:
        expected = str(self.model_path).rstrip("/")
        expected_base = os.path.basename(expected)
        for raw in served_model_ids:
            mid = str(raw).rstrip("/")
            if mid == expected:
                return True
            if os.path.basename(mid) == expected_base:
                return True
        return False

    async def _fetch_served_model_ids(self, timeout: float = 5.0) -> Optional[List[str]]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=float(timeout)),
                ) as resp:
                    if resp.status != 200:
                        return None
                    payload = await resp.json()
        except Exception:
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None
        model_ids: List[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if mid is None:
                continue
            model_ids.append(str(mid))
        return model_ids or None

    async def _maybe_attach_existing(self, listener_pids: List[int]) -> bool:
        served_ids: Optional[List[str]] = None
        for attempt in range(1, 6):
            served_ids = await self._fetch_served_model_ids(timeout=2.0)
            if served_ids:
                break
            # If a vLLM server is still starting, the port can begin listening
            # before /v1/models is ready. Give it a short grace period so
            # sequential CV folds can attach without churn.
            await asyncio.sleep(min(1.0 * attempt, 2.0))

        served_ids = served_ids or None
        if not served_ids:
            logger.info(
                "Port %d has listener pids=%s but /v1/models is not ready; treating as stale",
                int(self.port),
                listener_pids,
            )
            return False

        if not self._model_ids_match(served_ids):
            logger.info(
                "Port %d already serving %s (expected %s); restarting",
                int(self.port),
                served_ids,
                str(self.model_path),
            )
            return False

        if self.config.enable_sleep_mode and self.config.supports_sleep_mode:
            sleep_state = await self.is_server_sleeping()
            if sleep_state is None:
                logger.info(
                    "Port %d serves expected model but sleep endpoints unavailable; restarting with sleep mode",
                    int(self.port),
                )
                return False

            self._is_sleeping = bool(sleep_state)
        else:
            self._is_sleeping = False

        self._attached_pids = list(listener_pids)
        logger.info(
            "Attached to existing %s server on port %d (pids=%s, sleeping=%s)",
            str(getattr(self.config, "backend", "vllm") or "vllm"),
            int(self.port),
            self._attached_pids,
            self._is_sleeping,
        )
        return True

    async def _wait_for_ready(self) -> None:
        """Wait for server to be ready."""
        start_time = time.time()
        timeout = self.config.startup_timeout

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                if self._process.poll() is not None:
                    # Process died
                    try:
                        self._log_file.flush()
                        with open(self._log_file.name, 'r') as f:
                            output = f.read()
                    except Exception:
                        output = "Could not read log file"
                    backend_name = str(getattr(self.config, "backend", "vllm") or "vllm")
                    raise RuntimeError(
                        f"{backend_name} server on port {self.port} exited with code {self._process.returncode}. "
                        f"Output:\n{output[-2000:]}"
                    )

                try:
                    async with session.get(
                        f"{self.url}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            elapsed = time.time() - start_time
                            logger.info(f"Server on port {self.port} ready in {elapsed:.1f}s")
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass

                await asyncio.sleep(self.health_check_interval)

        self.stop()
        raise TimeoutError(f"Server on port {self.port} did not start within {timeout}s")

    def stop(self) -> None:
        """Stop the server."""
        if self._process is None:
            killed = kill_process_on_port(self.port)
            if killed:
                logger.info("Stopped external server on port %d", int(self.port))
            self._attached_pids = []
            self._is_sleeping = False
            return

        logger.info(f"Stopping server on port {self.port} (PID: {self._process.pid})")

        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
                logger.info(f"Server on port {self.port} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Server on port {self.port} did not stop gracefully, forcing kill")
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"Error stopping server on port {self.port}: {e}")

        self._process = None
        self._is_sleeping = False

        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    async def sleep(self, level: int = 1, timeout: float = 30.0) -> bool:
        """Put server to sleep (offload weights to CPU RAM)."""
        if not self.config.supports_sleep_mode or not self.config.enable_sleep_mode:
            logger.debug(
                "Server on port %d does not support sleep mode; caller should stop/start instead.",
                int(self.port),
            )
            return False

        if not self.is_running:
            logger.warning(f"Cannot sleep server on port {self.port}: not running")
            return False

        if self._is_sleeping:
            # Avoid trusting local state when wake/sleep errors happened earlier.
            actual = await self.is_server_sleeping()
            if actual is True:
                logger.debug(f"Server on port {self.port} already sleeping")
                return True
            if actual is False:
                logger.warning(
                    "Server on port %d local sleep flag stale (server reports awake); retrying sleep",
                    self.port,
                )
            else:
                logger.warning(
                    "Server on port %d local sleep flag stale/unverifiable; retrying sleep",
                    self.port,
                )
            self._is_sleeping = False

        logger.info(f"Putting server on port {self.port} to sleep (level={level})...")
        start = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/sleep?level={level}",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start
                        logger.info(f"Server on port {self.port} sleeping in {elapsed:.1f}s")
                        self._is_sleeping = True
                        return True
                    else:
                        text = await resp.text()
                        logger.error(f"Sleep request failed: {resp.status} - {text}")
                        self._is_sleeping = False
                        return False
        except Exception as e:
            logger.error(f"Failed to sleep server on port {self.port}: {e}")
            self._is_sleeping = False
            return False

    async def wake(self, timeout: float = 60.0) -> bool:
        """Wake server from sleep (reload weights from CPU RAM)."""
        if not self.config.supports_sleep_mode or not self.config.enable_sleep_mode:
            return bool(self.is_running)

        if not self.is_running:
            logger.warning(f"Cannot wake server on port {self.port}: not running")
            return False

        if not self._is_sleeping:
            # Local state can become stale after wake/sleep endpoint errors.
            actual = await self.is_server_sleeping()
            if actual is True:
                logger.warning(
                    "Server on port %d local sleep flag stale (server reports sleeping); issuing wake request",
                    self.port,
                )
                self._is_sleeping = True
            else:
                logger.debug(f"Server on port {self.port} already awake")
                return True

        logger.info(f"Waking server on port {self.port}...")
        start = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/wake_up",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start
                        logger.info(f"Server on port {self.port} awake in {elapsed:.1f}s")
                        self._is_sleeping = False
                        return True
                    else:
                        text = await resp.text()
                        logger.error(f"Wake request failed: {resp.status} - {text}")
                        # Wake failure may still leave the server awake.
                        # Confirm sleep status before forcing expensive restart.
                        actual = await self.is_server_sleeping()
                        if actual is False:
                            logger.warning(
                                "Wake request failed on port %d but server reports awake; continuing without restart",
                                self.port,
                            )
                            self._is_sleeping = False
                            return True
                        # Keep conservative failure path when server still reports sleeping
                        # or state cannot be verified.
                        self._is_sleeping = False
                        return False
        except Exception as e:
            logger.error(f"Failed to wake server on port {self.port}: {e}")
            actual = await self.is_server_sleeping()
            if actual is False:
                logger.warning(
                    "Wake request errored on port %d but server reports awake; continuing without restart",
                    self.port,
                )
                self._is_sleeping = False
                return True
            self._is_sleeping = False
            return False

    async def is_server_sleeping(self) -> Optional[bool]:
        """Check if server is sleeping via API."""
        if not self.config.supports_sleep_mode or not self.config.enable_sleep_mode:
            return False if self.is_running else None

        if not self.is_running:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.url}/is_sleeping",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("is_sleeping", False)
        except Exception:
            pass
        return None


def load_model_path(profile: str, config_path: Optional[Path] = None) -> str:
    """Load model path from settings.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("vllm", {}).get("models", {})
    if profile not in models:
        raise ValueError(f"Profile '{profile}' not found. Available: {list(models.keys())}")

    return models[profile]["path"]


class GPUOrchestrator:
    """
    Manages dynamic GPU allocation across vLLM/SGLang backends.

    Uses vLLM sleep mode when available and falls back to explicit stop/start
    transitions for backends (or backend combinations) that do not support sleep.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration. If None, loads from settings.yaml.
        """
        if config is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
            config = OrchestratorConfig.from_yaml(config_path)

        self.config = config
        self._mode = OrchestratorMode.UNINITIALIZED
        self._recovery_lock = threading.Lock()

        # Build KV persistence config dict for ManagedServer
        _kv_cfg: Dict[str, Any] = {}
        if config.kv_persistence_enabled:
            _kv_cfg = {
                "enabled": True,
                "backend": config.kv_persistence_backend,
                "cpu_gb": config.kv_persistence_cpu_gb,
                "disk_path": config.kv_persistence_disk_path,
                "disk_gb": config.kv_persistence_disk_gb,
                "chunk_size": config.kv_persistence_chunk_size,
                "config_file": str(Path(config.config_path).parent / "lmcache_config.yaml")
                    if config.config_path else None,
            }

        def _server_venv_path(server_cfg: ServerConfig) -> str:
            backend = str(getattr(server_cfg, "backend", "vllm") or "vllm").strip().lower()
            if backend == "sglang":
                return str(getattr(config, "sglang_venv_path", "~/sglang-env"))
            return str(getattr(config, "venv_path", "~/vllm-env"))

        # Create managed servers
        self._task_primary = ManagedServer(
            config=config.task_primary,
            venv_path=_server_venv_path(config.task_primary),
            model_path=load_model_path(config.task_primary.profile, config.config_path),
            health_check_interval=config.health_check_interval,
            post_stop_settle_seconds=config.post_stop_settle_seconds,
            kv_persistence=_kv_cfg,
        )
        self._task_replica = ManagedServer(
            config=config.task_replica,
            venv_path=_server_venv_path(config.task_replica),
            model_path=load_model_path(config.task_replica.profile, config.config_path),
            health_check_interval=config.health_check_interval,
            post_stop_settle_seconds=config.post_stop_settle_seconds,
            kv_persistence=_kv_cfg,
        )
        self._genrm = ManagedServer(
            config=config.genrm,
            venv_path=_server_venv_path(config.genrm),
            model_path=load_model_path(config.genrm.profile, config.config_path),
            health_check_interval=config.health_check_interval,
            post_stop_settle_seconds=config.post_stop_settle_seconds,
            kv_persistence=_kv_cfg,
        )
        self._embedding: Optional[ManagedServer] = None
        if bool(getattr(config, "manage_embedding", False)) and config.embedding is not None:
            self._embedding = ManagedServer(
                config=config.embedding,
                venv_path=_server_venv_path(config.embedding),
                model_path=load_model_path(config.embedding.profile, config.config_path),
                health_check_interval=config.health_check_interval,
                post_stop_settle_seconds=config.post_stop_settle_seconds,
                kv_persistence={},
            )

    def _managed_server_for_port(self, port: int) -> Tuple[Optional[ManagedServer], str]:
        """Resolve an orchestrator-managed server for a TCP port."""
        port_int = int(port)
        if int(self._task_primary.port) == port_int:
            return self._task_primary, "task_primary"
        if int(self._task_replica.port) == port_int:
            return self._task_replica, "task_replica"
        if int(self._genrm.port) == port_int:
            return self._genrm, "genrm"
        if self._embedding is not None and int(self._embedding.port) == port_int:
            return self._embedding, "embedding"
        return None, "unknown"

    @staticmethod
    def _cuda_device_set(raw: str) -> set[int]:
        values: set[int] = set()
        for chunk in str(raw or "").split(","):
            token = chunk.strip()
            if not token:
                continue
            try:
                values.add(int(token))
            except ValueError:
                continue
        return values

    def _servers_share_gpus(
        self,
        left: Optional[ManagedServer],
        right: Optional[ManagedServer],
    ) -> bool:
        if left is None or right is None:
            return False
        left_set = self._cuda_device_set(getattr(left.config, "cuda_devices", ""))
        right_set = self._cuda_device_set(getattr(right.config, "cuda_devices", ""))
        if not left_set or not right_set:
            return False
        return bool(left_set & right_set)

    async def _ensure_server_quiesced(
        self,
        server: ManagedServer,
        *,
        role_label: str,
        reason: str,
        force_stop: Optional[bool] = None,
    ) -> None:
        """Ensure `server` is not actively occupying GPUs (sleep or stop)."""
        if not server.is_running:
            return

        if not server.config.supports_sleep_mode or not server.config.enable_sleep_mode:
            logger.info(
                "%s: %s on port %d has no sleep capability; stopping process for quiesce",
                reason,
                role_label,
                int(server.port),
            )
            server.stop()
            await asyncio.sleep(self.config.post_stop_settle_seconds)
            return

        hard_quiesce = self.config.shared_gpu_hard_quiesce if force_stop is None else bool(force_stop)
        if hard_quiesce:
            logger.info(
                "%s: hard-quiescing %s on port %d (stop to fully free shared GPUs)",
                reason,
                role_label,
                int(server.port),
            )
            server.stop()
            await asyncio.sleep(self.config.post_stop_settle_seconds)
            return

        logger.info("%s: quiescing %s on port %d", reason, role_label, int(server.port))
        slept = False
        try:
            slept = await server.sleep(timeout=self.config.sleep_timeout)
        except Exception as exc:
            logger.warning("%s: %s sleep raised: %s", reason, role_label, exc)
            slept = False

        if slept:
            actual = await server.is_server_sleeping()
            if actual is True:
                return
            if actual is False:
                logger.warning(
                    "%s: %s reports awake after sleep; stopping process to free GPUs.",
                    reason,
                    role_label,
                )
            else:
                logger.warning(
                    "%s: %s sleep state unverified; stopping process to free GPUs.",
                    reason,
                    role_label,
                )
        else:
            logger.warning(
                "%s: %s failed to sleep cleanly; stopping process to free GPUs.",
                reason,
                role_label,
            )

        server.stop()
        await asyncio.sleep(self.config.post_stop_settle_seconds)

    async def recover_port(self, port: int, *, reason: str = "") -> bool:
        """
        Force-restart a managed server for `port` and restore current mode.

        This is intended for runtime recovery after connection failures.
        """
        try:
            port_int = int(port)
        except (TypeError, ValueError):
            logger.warning("recover_port called with invalid port: %s", port)
            return False

        server, role = self._managed_server_for_port(port_int)
        if server is None:
            logger.warning("recover_port: port %d is not orchestrator-managed", port_int)
            return False

        with self._recovery_lock:
            logger.warning(
                "Recovering %s server on port %d%s",
                role,
                port_int,
                f" (reason={reason})" if reason else "",
            )

            # task_replica and genrm share GPUs 2,3. Ensure the peer is not awake
            # before restarting the target.
            if role == "task_replica":
                await self._ensure_server_quiesced(
                    self._genrm,
                    role_label="GenRM",
                    reason="Recovery",
                    force_stop=True,
                )

            if role == "genrm":
                await self._ensure_server_quiesced(
                    self._task_replica,
                    role_label="task replica",
                    reason="Recovery",
                    force_stop=True,
                )

            if role in {"task_primary", "task_replica", "genrm"} and self._embedding is not None:
                target = {
                    "task_primary": self._task_primary,
                    "task_replica": self._task_replica,
                    "genrm": self._genrm,
                }.get(role)
                if self._servers_share_gpus(self._embedding, target):
                    await self._ensure_server_quiesced(
                        self._embedding,
                        role_label="embedding",
                        reason="Recovery",
                        force_stop=True,
                    )

            if server.is_running:
                server.stop()
                await asyncio.sleep(2)

            try:
                await server.start()
            except Exception as exc:
                logger.error("Recovery: failed to start %s on port %d: %s", role, port_int, exc)
                return False

            # Re-assert current orchestrator mode so wake/sleep state is coherent.
            if role in {"task_primary", "task_replica"}:
                if self._mode == OrchestratorMode.TASK_DP2:
                    return await self.enter_task_dp2_mode()
                if self._mode == OrchestratorMode.DUAL_MODEL:
                    if role == "task_primary":
                        awake = await self._task_primary.wake(timeout=self.config.wake_timeout)
                        return bool(awake)
                    # Replica should generally remain sleeping in dual-model mode.
                    if self._task_replica.is_running:
                        try:
                            await self._task_replica.sleep(timeout=self.config.sleep_timeout)
                        except Exception:
                            pass
                    return True
                return True

            if role == "genrm":
                if self._mode == OrchestratorMode.DUAL_MODEL:
                    return await self.enter_dual_model_mode()
                return True

            return True

    @property
    def mode(self) -> OrchestratorMode:
        return self._mode

    async def initialize(self, initial_mode: OrchestratorMode = OrchestratorMode.TASK_DP2) -> None:
        """
        Start all servers and configure initial mode.

        Starts all three servers, then puts secondary ones to sleep.
        Initial mode determines which servers stay awake.

        Args:
            initial_mode: Initial GPU allocation mode.
        """
        logger.info("Initializing GPU orchestrator...")
        logger.info(f"  Task primary: port {self.config.task_primary.port}, GPUs {self.config.task_primary.cuda_devices}")
        logger.info(f"  Task replica: port {self.config.task_replica.port}, GPUs {self.config.task_replica.cuda_devices}")
        logger.info(f"  GenRM: port {self.config.genrm.port}, GPUs {self.config.genrm.cuda_devices}")
        if self._embedding is not None:
            logger.info(
                "  Embedding: port %s, GPUs %s (managed=%s)",
                int(self._embedding.port),
                str(self._embedding.config.cuda_devices),
                bool(getattr(self.config, "manage_embedding", False)),
            )
            if self._servers_share_gpus(self._embedding, self._task_primary):
                await self._ensure_server_quiesced(
                    self._embedding,
                    role_label="embedding",
                    reason="initialize:task_primary_conflict",
                    force_stop=True,
                )
            if (
                initial_mode == OrchestratorMode.TASK_DP2
                and self._servers_share_gpus(self._embedding, self._task_replica)
            ):
                await self._ensure_server_quiesced(
                    self._embedding,
                    role_label="embedding",
                    reason="initialize:task_replica_conflict",
                    force_stop=True,
                )
            if (
                initial_mode == OrchestratorMode.DUAL_MODEL
                and self._servers_share_gpus(self._embedding, self._genrm)
            ):
                await self._ensure_server_quiesced(
                    self._embedding,
                    role_label="embedding",
                    reason="initialize:genrm_conflict",
                    force_stop=True,
                )

        genrm_enabled = bool(getattr(self.config, "enable_genrm", True))
        if not genrm_enabled:
            if initial_mode == OrchestratorMode.DUAL_MODEL:
                logger.warning("GenRM disabled; forcing initial mode to task_dp2")
                initial_mode = OrchestratorMode.TASK_DP2

            await self._task_primary.start()

            # task_replica uses GPUs 2,3. If an external GenRM server is already
            # running on the shared GPUs, stop it to avoid startup OOM.
            await self._ensure_server_quiesced(
                self._genrm,
                role_label="GenRM",
                reason="initialize:disable_genrm",
                force_stop=True,
            )

            await self._task_replica.start()
            if self._task_replica.config.supports_sleep_mode and self._task_replica.config.enable_sleep_mode:
                awake = await self._task_replica.wake(timeout=self.config.wake_timeout)
                if not awake:
                    logger.warning(
                        "Initialization wake failed for task replica on port %d; cold restarting replica",
                        int(self._task_replica.port),
                    )
                    self._task_replica.stop()
                    await asyncio.sleep(self.config.post_stop_settle_seconds)
                    await self._task_replica.start()

            self._mode = OrchestratorMode.TASK_DP2
            logger.info("Orchestrator initialized in task_dp2 mode (GenRM disabled)")
            return

        # Start all servers (sequentially to avoid GPU memory contention during load).
        #
        # Important: task_replica and genrm share GPUs 2,3, so we must ensure the *other*
        # shared-GPU peer is quiesced (sleep/stop) before attempting to start one.
        #
        # This also handles the common "mixed static/dynamic" situation where users have
        # already started a GenRM server manually on port 8001 (no sleep endpoints), which
        # would otherwise leave only a few GiB free on GPUs 2,3 and cause task_replica
        # startup to fail.
        await self._task_primary.start()

        shared_supports_sleep = (
            self._task_replica.config.supports_sleep_mode
            and self._task_replica.config.enable_sleep_mode
            and self._genrm.config.supports_sleep_mode
            and self._genrm.config.enable_sleep_mode
        )

        # For TASK_DP2: task_replica active, genrm quiesced
        # For DUAL_MODEL: genrm active, task_replica quiesced
        if not shared_supports_sleep:
            logger.info(
                "Shared-GPU servers do not support sleep mode; using explicit stop/start transitions."
            )
            if initial_mode == OrchestratorMode.TASK_DP2:
                await self._ensure_server_quiesced(
                    self._genrm,
                    role_label="GenRM",
                    reason="initialize:task_dp2_no_sleep",
                    force_stop=True,
                )
                await self._task_replica.start()
            else:
                await self._ensure_server_quiesced(
                    self._task_replica,
                    role_label="task replica",
                    reason="initialize:dual_model_no_sleep",
                    force_stop=True,
                )
                await self._genrm.start()
        else:
            # Sleep-capable path: preload both shared-GPU peers and keep one sleeping.
            if initial_mode == OrchestratorMode.TASK_DP2:
                await self._ensure_server_quiesced(
                    self._genrm,
                    role_label="GenRM",
                    reason="initialize:prepare_task_replica",
                    force_stop=None,
                )
                await self._task_replica.start()
                await self._task_replica.sleep()
                await self._genrm.start()
                await self._genrm.sleep()
                awake = await self._task_replica.wake()
                if not awake:
                    logger.warning(
                        "Initialization wake failed for task replica on port %d; cold restarting replica",
                        int(self._task_replica.port),
                    )
                    self._task_replica.stop()
                    await asyncio.sleep(self.config.post_stop_settle_seconds)
                    await self._task_replica.start()
            else:  # DUAL_MODEL
                await self._ensure_server_quiesced(
                    self._task_replica,
                    role_label="task replica",
                    reason="initialize:prepare_genrm",
                    force_stop=None,
                )
                await self._genrm.start()
                await self._genrm.sleep()
                await self._task_replica.start()
                await self._task_replica.sleep()
                awake = await self._genrm.wake()
                if not awake:
                    logger.warning(
                        "Initialization wake failed for GenRM on port %d; cold restarting GenRM",
                        int(self._genrm.port),
                    )
                    self._genrm.stop()
                    await asyncio.sleep(self.config.post_stop_settle_seconds)
                    await self._genrm.start()

        self._mode = initial_mode
        logger.info(f"Orchestrator initialized in {initial_mode.value} mode")

    async def enter_task_dp2_mode(self) -> bool:
        """
        Switch to DP=2 mode (both task models active, GenRM sleeping).

        Returns:
            True if transition successful.
        """
        if self._mode == OrchestratorMode.TASK_DP2:
            logger.debug("Ensuring task_dp2 mode...")
        else:
            logger.info("Transitioning to task_dp2 mode...")
        start = time.time()

        if (
            self._embedding is not None
            and bool(getattr(self.config, "quiesce_embedding_when_idle", True))
            and (
                self._servers_share_gpus(self._embedding, self._task_primary)
                or self._servers_share_gpus(self._embedding, self._task_replica)
            )
        ):
            await self._ensure_server_quiesced(
                self._embedding,
                role_label="embedding",
                reason="task_dp2 transition",
                force_stop=None,
            )

        if self._genrm.is_running:
            await self._ensure_server_quiesced(
                self._genrm,
                role_label="GenRM",
                reason="task_dp2 transition",
                force_stop=None,
            )

        if not self._task_replica.is_running:
            logger.warning(
                "Task replica on port %d is not running; restarting.",
                int(self._task_replica.port),
            )
            try:
                await self._task_replica.start()
            except Exception as exc:
                logger.error("Failed to start task replica on port %d: %s", int(self._task_replica.port), exc)
                return False

        awake = await self._task_replica.wake(timeout=self.config.wake_timeout)
        if not awake:
            logger.error(
                "Failed to wake task replica on port %d; attempting cold restart",
                int(self._task_replica.port),
            )
            # Wake can fail with shared-memory/OOM issues on long runs; recover in-place.
            self._task_replica.stop()
            await asyncio.sleep(self.config.post_stop_settle_seconds)
            try:
                await self._task_replica.start()
            except Exception as exc:
                logger.error(
                    "Cold restart failed for task replica on port %d: %s",
                    int(self._task_replica.port),
                    exc,
                )
                return False

        elapsed = time.time() - start
        logger.info(f"Transitioned to task_dp2 mode in {elapsed:.1f}s")
        self._mode = OrchestratorMode.TASK_DP2
        return True

    async def enter_dual_model_mode(self) -> bool:
        """
        Switch to dual model mode (task + GenRM, replica sleeping).

        Returns:
            True if transition successful.
        """
        if not bool(getattr(self.config, "enable_genrm", True)):
            logger.warning("enter_dual_model_mode requested but GenRM is disabled")
            return False

        if self._mode == OrchestratorMode.DUAL_MODEL:
            logger.debug("Ensuring dual_model mode...")
        else:
            logger.info("Transitioning to dual_model mode...")
        start = time.time()

        if (
            self._embedding is not None
            and bool(getattr(self.config, "quiesce_embedding_when_idle", True))
            and (
                self._servers_share_gpus(self._embedding, self._task_primary)
                or self._servers_share_gpus(self._embedding, self._genrm)
            )
        ):
            await self._ensure_server_quiesced(
                self._embedding,
                role_label="embedding",
                reason="dual_model transition",
                force_stop=None,
            )

        if self._task_replica.is_running:
            await self._ensure_server_quiesced(
                self._task_replica,
                role_label="task replica",
                reason="dual_model transition",
                force_stop=None,
            )

        if not self._genrm.is_running:
            logger.warning("GenRM on port %d is not running; restarting.", int(self._genrm.port))
            try:
                await self._genrm.start()
            except Exception as exc:
                logger.error("Failed to start GenRM on port %d: %s", int(self._genrm.port), exc)
                return False

        awake = await self._genrm.wake(timeout=self.config.wake_timeout)
        if not awake:
            logger.error(
                "Failed to wake GenRM on port %d; attempting cold restart",
                int(self._genrm.port),
            )
            # Wake can fail with shared-memory/OOM issues on long runs; recover in-place.
            self._genrm.stop()
            await asyncio.sleep(self.config.post_stop_settle_seconds)
            try:
                await self._genrm.start()
            except Exception as exc:
                logger.error(
                    "Cold restart failed for GenRM on port %d: %s",
                    int(self._genrm.port),
                    exc,
                )
                return False

        elapsed = time.time() - start
        logger.info(f"Transitioned to dual_model mode in {elapsed:.1f}s")
        self._mode = OrchestratorMode.DUAL_MODEL
        return True

    # ------------------------------------------------------------------
    # Overlapped phase transitions (DualPath Opt 3)
    # ------------------------------------------------------------------

    async def begin_prewarm_genrm(self) -> bool:
        """Begin prewarming GenRM while tree building finishes.

        Call this when ``BatchTreeOrchestrator.completion_fraction`` exceeds
        the prewarm threshold (typically 0.85).  It sleeps the task replica
        and starts waking GenRM in the background so that by the time tree
        building completes, GenRM is already ready.

        Returns True if the prewarm was initiated successfully.
        """
        if not bool(getattr(self.config, "enable_genrm", True)):
            logger.debug("begin_prewarm_genrm: GenRM disabled, skipping")
            return False

        if self._mode != OrchestratorMode.TASK_DP2:
            logger.debug("begin_prewarm_genrm: not in TASK_DP2, skipping")
            return False

        logger.info("Begin prewarming GenRM (overlapped with tree building tail)")
        start = time.time()

        # Step 1: Quiesce the task replica to free GPUs 2,3
        if self._task_replica.is_running:
            await self._ensure_server_quiesced(
                self._task_replica,
                role_label="task replica",
                reason="genrm_prewarm",
                force_stop=None,
            )

        # Step 2: Begin waking GenRM (this may take 6-12s)
        if not self._genrm.is_running:
            try:
                await self._genrm.start()
            except Exception as exc:
                logger.error("Failed to start GenRM during prewarm: %s", exc)
                return False

        awake = await self._genrm.wake(timeout=self.config.wake_timeout)
        elapsed = time.time() - start

        if awake:
            logger.info("GenRM prewarm complete in %.1fs", elapsed)
        else:
            logger.warning("GenRM prewarm wake failed after %.1fs; will retry on finalize", elapsed)

        return awake

    async def finalize_prewarm_genrm(self) -> bool:
        """Finalize the prewarm transition after tree building completes.

        Call this after all tree building is done. If ``begin_prewarm_genrm``
        already succeeded, this is a no-op mode switch. Otherwise, it falls
        back to the full ``enter_dual_model_mode`` transition.

        Returns True if GenRM is ready.
        """
        # Check if GenRM is already awake from prewarm
        if self._genrm.is_running and not self._genrm.is_sleeping:
            sleeping = await self._genrm.is_server_sleeping()
            if sleeping is False:  # Explicitly awake
                self._mode = OrchestratorMode.DUAL_MODEL
                logger.info("Finalized prewarm: GenRM already awake, switched to dual_model mode")
                return True

        # Fall back to full transition
        logger.info("Prewarm incomplete, falling back to full dual_model transition")
        return await self.enter_dual_model_mode()

    async def begin_prewarm_task_replica(self) -> bool:
        """Begin prewarming task replica while GenRM scoring finishes.

        Mirror of ``begin_prewarm_genrm`` for the reverse transition.
        """
        if self._mode != OrchestratorMode.DUAL_MODEL:
            logger.debug("begin_prewarm_task_replica: not in DUAL_MODEL, skipping")
            return False

        logger.info("Begin prewarming task replica (overlapped with scoring tail)")
        start = time.time()

        if self._genrm.is_running:
            await self._ensure_server_quiesced(
                self._genrm,
                role_label="GenRM",
                reason="task_replica_prewarm",
                force_stop=None,
            )

        if not self._task_replica.is_running:
            try:
                await self._task_replica.start()
            except Exception as exc:
                logger.error("Failed to start task replica during prewarm: %s", exc)
                return False

        awake = await self._task_replica.wake(timeout=self.config.wake_timeout)
        elapsed = time.time() - start

        if awake:
            logger.info("Task replica prewarm complete in %.1fs", elapsed)
            self._mode = OrchestratorMode.TASK_DP2
        else:
            logger.warning("Task replica prewarm wake failed after %.1fs", elapsed)

        return awake

    async def ensure_embedding_ready(self, *, reason: str = "embedding") -> bool:
        """Ensure embedding endpoint is running/awake under orchestrator control."""
        if self._embedding is None:
            logger.debug("ensure_embedding_ready called but embedding management is disabled")
            return False

        if self._servers_share_gpus(self._embedding, self._task_primary):
            await self._ensure_server_quiesced(
                self._task_primary,
                role_label="task primary",
                reason=f"{reason}:embedding_conflict",
                force_stop=True,
            )
        if self._servers_share_gpus(self._embedding, self._task_replica):
            await self._ensure_server_quiesced(
                self._task_replica,
                role_label="task replica",
                reason=f"{reason}:embedding_conflict",
                force_stop=True,
            )
        if self._servers_share_gpus(self._embedding, self._genrm):
            await self._ensure_server_quiesced(
                self._genrm,
                role_label="GenRM",
                reason=f"{reason}:embedding_conflict",
                force_stop=True,
            )

        if not self._embedding.is_running:
            await self._embedding.start()

        awake = await self._embedding.wake(timeout=self.config.wake_timeout)
        if not awake:
            logger.warning(
                "%s: embedding wake failed on port %d; cold restarting embedding server",
                reason,
                int(self._embedding.port),
            )
            self._embedding.stop()
            await asyncio.sleep(self.config.post_stop_settle_seconds)
            await self._embedding.start()
            awake = await self._embedding.wake(timeout=self.config.wake_timeout)
        return bool(awake)

    async def quiesce_embedding(self, *, reason: str = "embedding") -> bool:
        """Sleep/stop embedding endpoint when it is no longer needed."""
        if self._embedding is None:
            return False
        if not self._embedding.is_running:
            return True
        await self._ensure_server_quiesced(
            self._embedding,
            role_label="embedding",
            reason=reason,
            force_stop=None,
        )
        return True

    def get_active_task_ports(self) -> List[int]:
        """Get list of active task model ports for current mode."""
        if self._mode == OrchestratorMode.TASK_DP2:
            return [self.config.task_primary.port, self.config.task_replica.port]
        else:
            return [self.config.task_primary.port]

    def get_genrm_port(self) -> int:
        """Get GenRM port."""
        return self.config.genrm.port

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        status = {
            "mode": self._mode.value,
            "task_primary": {
                "port": self._task_primary.port,
                "backend": self._task_primary.config.backend,
                "running": self._task_primary.is_running,
                "sleeping": self._task_primary.is_sleeping,
                "supports_sleep_mode": self._task_primary.config.supports_sleep_mode,
            },
            "task_replica": {
                "port": self._task_replica.port,
                "backend": self._task_replica.config.backend,
                "running": self._task_replica.is_running,
                "sleeping": self._task_replica.is_sleeping,
                "supports_sleep_mode": self._task_replica.config.supports_sleep_mode,
            },
            "genrm": {
                "port": self._genrm.port,
                "backend": self._genrm.config.backend,
                "running": self._genrm.is_running,
                "sleeping": self._genrm.is_sleeping,
                "supports_sleep_mode": self._genrm.config.supports_sleep_mode,
            },
        }
        if self._embedding is not None:
            status["embedding"] = {
                "port": self._embedding.port,
                "backend": self._embedding.config.backend,
                "running": self._embedding.is_running,
                "sleeping": self._embedding.is_sleeping,
                "supports_sleep_mode": self._embedding.config.supports_sleep_mode,
            }
        return status

    async def shutdown(self) -> None:
        """Stop all servers."""
        logger.info("Shutting down GPU orchestrator...")
        self._task_primary.stop()
        self._task_replica.stop()
        self._genrm.stop()
        if self._embedding is not None:
            self._embedding.stop()
        self._mode = OrchestratorMode.UNINITIALIZED
        logger.info("GPU orchestrator shutdown complete")

    async def __aenter__(self) -> "GPUOrchestrator":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
