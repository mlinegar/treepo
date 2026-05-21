from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Sequence

from treepo._research.ctreepo.sim.execution_resources import infer_run_resources, parse_command_flags
from treepo._research.ctreepo.sim.manifest import RunSpec, read_manifest_jsonl


_PRINT_LOCK = threading.Lock()


def _safe_print(text: str) -> None:
    with _PRINT_LOCK:
        print(text, flush=True)


def _short_id(text: str, *, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: int(max(8, min(64, n)))]


@dataclass(frozen=True)
class CommandJob:
    id: str
    family: str
    command: str
    outputs: Dict[str, str]
    resources: Dict[str, Any]


def _parse_items(text: str) -> List[str]:
    items: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            items.append(item)
    return items


def _script_name(command: str) -> str:
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return ""
    for token in tokens:
        if token.endswith(".py"):
            return Path(token).name
    return ""


def job_from_run_spec(run: RunSpec) -> CommandJob:
    return CommandJob(
        id=str(run.id),
        family=str(run.family),
        command=str(run.command),
        outputs={str(k): str(v) for k, v in dict(run.outputs).items()},
        resources=dict(run.resources or {}),
    )


def job_from_command(command: str, *, idx: int) -> CommandJob:
    cmd = str(command).strip()
    flags = parse_command_flags(cmd)
    outputs: Dict[str, str] = {}
    for key in ("json_summary", "csv_summary", "artifact_dir"):
        value = flags.get(key)
        if value is not None and str(value).strip():
            outputs[str(key)] = str(value)
    return CommandJob(
        id=f"cmd_{idx:05d}_{_short_id(cmd)}",
        family=_script_name(cmd) or "command",
        command=cmd,
        outputs=outputs,
        resources=infer_run_resources(command=cmd),
    )


def load_jobs(*, manifest_paths: Sequence[Path], cmd_files: Sequence[Path]) -> List[CommandJob]:
    jobs: List[CommandJob] = []
    for manifest_path in manifest_paths:
        for run in read_manifest_jsonl(Path(manifest_path)):
            jobs.append(job_from_run_spec(run))
    for cmd_file in cmd_files:
        idx_base = len(jobs)
        raw_lines = Path(cmd_file).read_text(encoding="utf-8").splitlines()
        for offset, raw in enumerate(raw_lines):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            jobs.append(job_from_command(line, idx=idx_base + offset))
    return jobs


def _query_gpu_layout() -> Dict[int, List[str]]:
    try:
        text = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except Exception:
        return {}
    gpu_to_tokens: Dict[int, List[str]] = {}
    current_gpu: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m_gpu = re.match(r"GPU\s+(\d+):.*\(UUID:\s*(GPU-[^)]+)\)", line)
        if m_gpu:
            current_gpu = int(m_gpu.group(1))
            gpu_to_tokens.setdefault(current_gpu, [])
            continue
        m_mig = re.match(r"\s*MIG\s+.*\(UUID:\s*(MIG-[^)]+)\)", line)
        if m_mig and current_gpu is not None:
            gpu_to_tokens.setdefault(current_gpu, []).append(m_mig.group(1))
    return gpu_to_tokens


def detect_gpu_tokens(spec: str) -> List[str]:
    spec_text = str(spec or "").strip()
    gpu_to_tokens = _query_gpu_layout()
    if not spec_text or spec_text.lower() in {"none", "cpu", "off"}:
        return []
    if spec_text.lower() == "auto":
        if gpu_to_tokens:
            out: List[str] = []
            for gpu_idx in sorted(gpu_to_tokens):
                tokens = gpu_to_tokens.get(gpu_idx, [])
                out.extend(tokens if tokens else [str(gpu_idx)])
            return out
        try:
            count = int(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    text=True,
                ).count("\n")
            )
        except Exception:
            count = 0
        return [str(i) for i in range(count)]
    out: List[str] = []
    seen = set()
    for item in _parse_items(spec_text):
        if item.startswith("MIG-") or item.startswith("GPU-"):
            if item not in seen:
                seen.add(item)
                out.append(item)
            continue
        try:
            idx = int(item)
        except ValueError:
            if item not in seen:
                seen.add(item)
                out.append(item)
            continue
        tokens = gpu_to_tokens.get(idx, [])
        expanded = tokens if tokens else [str(idx)]
        for token in expanded:
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def assign_job_lane(job: CommandJob, *, gpu_tokens: Sequence[str]) -> str:
    resources = dict(job.resources or {})
    accelerator = str(resources.get("accelerator", "cpu")).strip().lower()
    gpu_preferred = bool(resources.get("gpu_preferred", False))
    if accelerator == "gpu":
        return "gpu"
    if accelerator == "cpu":
        return "cpu"
    if gpu_tokens and gpu_preferred:
        return "gpu"
    return "cpu"


def split_jobs(jobs: Sequence[CommandJob], *, gpu_tokens: Sequence[str]) -> tuple[List[CommandJob], List[CommandJob]]:
    cpu_jobs: List[CommandJob] = []
    gpu_jobs: List[CommandJob] = []
    for job in jobs:
        lane = assign_job_lane(job, gpu_tokens=gpu_tokens)
        if lane == "gpu":
            gpu_jobs.append(job)
        else:
            cpu_jobs.append(job)
    return cpu_jobs, gpu_jobs


def rewrite_command_for_lane(command: str, *, lane: str) -> str:
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return str(command)

    rewritten: List[str] = []
    idx = 0
    saw_cuda_device = False
    saw_device = False
    while idx < len(tokens):
        token = str(tokens[idx])
        if token == "--device" and idx + 1 < len(tokens):
            saw_device = True
            current = str(tokens[idx + 1]).strip().lower()
            if lane == "cpu":
                rewritten.extend(["--device", "cpu"])
            elif lane == "gpu" and current == "auto":
                rewritten.extend(["--device", "cuda"])
            else:
                rewritten.extend(["--device", tokens[idx + 1]])
            idx += 2
            continue
        if token == "--cuda-device":
            saw_cuda_device = True
            if lane == "gpu":
                rewritten.extend(["--cuda-device", "0"])
            idx += 2
            continue
        rewritten.append(token)
        idx += 1

    # Only normalize device flags that the command already declared. Some publication
    # jobs (for example exact LDA sweeps and report/plot scripts) do not expose a
    # --device CLI at all, and the queue must not invent one.
    if lane == "gpu" and saw_device and not saw_cuda_device:
        rewritten.extend(["--cuda-device", "0"])
    return shlex.join(rewritten)


def _declared_output_paths(job: CommandJob) -> List[Path]:
    outputs = dict(job.outputs or {})
    paths: List[Path] = []
    for key in ("json_summary", "csv_summary"):
        value = outputs.get(str(key), "")
        if str(value).strip():
            paths.append(Path(str(value)))
    return paths


def _run_job(job: CommandJob, *, lane: str, log_dir: Path, gpu_token: str | None) -> Dict[str, Any]:
    declared_outputs = _declared_output_paths(job)
    log_path = log_dir / f"{job.id}.log"
    if declared_outputs and all(path.exists() for path in declared_outputs):
        return {
            "job_id": job.id,
            "ok": True,
            "returncode": 0,
            "seconds": 0.0,
            "lane": lane,
            "gpu_token": gpu_token,
            "log": str(log_path),
            "skipped_existing": True,
        }

    env = dict(os.environ)
    final_cmd = rewrite_command_for_lane(job.command, lane=lane)
    if gpu_token is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"lane={lane}\n")
        if gpu_token is not None:
            handle.write(f"gpu_token={gpu_token}\n")
        handle.write(f"cmd={final_cmd}\n")
        handle.flush()
        proc = subprocess.run(
            final_cmd,
            shell=True,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            executable="/bin/bash",
        )
    return {
        "job_id": job.id,
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - start, 1),
        "lane": lane,
        "gpu_token": gpu_token,
        "log": str(log_path),
        "skipped_existing": False,
    }


def _run_cpu_lane(jobs: Sequence[CommandJob], *, cpu_workers: int, log_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not jobs or int(cpu_workers) <= 0:
        return results
    with ThreadPoolExecutor(max_workers=int(cpu_workers)) as pool:
        future_map = {
            pool.submit(_run_job, job, lane="cpu", log_dir=log_dir, gpu_token=None): job
            for job in jobs
        }
        total = len(future_map)
        done_count = 0
        for future in as_completed(future_map):
            result = future.result()
            done_count += 1
            results.append(result)
            prefix = "skip" if result.get("skipped_existing") else ("ok" if result["ok"] else "fail")
            _safe_print(
                f"[cpu {done_count}/{total}] job={result['job_id']} {prefix} "
                f"{result['seconds']}s log={result['log']}"
            )
    return results


def _run_gpu_lane(jobs: Sequence[CommandJob], *, gpu_tokens: Sequence[str], log_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not jobs:
        return results
    if not gpu_tokens:
        raise RuntimeError("GPU jobs were queued, but no GPU tokens were supplied.")
    next_idx = 0
    inflight: Dict[Future, str] = {}
    with ThreadPoolExecutor(max_workers=len(gpu_tokens)) as pool:
        while next_idx < len(jobs) or inflight:
            while next_idx < len(jobs) and len(inflight) < len(gpu_tokens):
                gpu_token = str(gpu_tokens[len(inflight)])
                future = pool.submit(
                    _run_job,
                    jobs[next_idx],
                    lane="gpu",
                    log_dir=log_dir,
                    gpu_token=gpu_token,
                )
                inflight[future] = gpu_token
                next_idx += 1
            if not inflight:
                break
            done, _ = wait(inflight.keys(), timeout=5.0, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                gpu_token = inflight.pop(future)
                result = future.result()
                results.append(result)
                prefix = "skip" if result.get("skipped_existing") else ("ok" if result["ok"] else "fail")
                _safe_print(
                    f"[gpu {len(results)}/{len(jobs)}] job={result['job_id']} {prefix} "
                    f"{result['seconds']}s token={gpu_token} log={result['log']}"
                )
                if next_idx < len(jobs):
                    next_job = jobs[next_idx]
                    new_future = pool.submit(
                        _run_job,
                        next_job,
                        lane="gpu",
                        log_dir=log_dir,
                        gpu_token=gpu_token,
                    )
                    inflight[new_future] = gpu_token
                    next_idx += 1
    return results


def run_resource_queue(
    jobs: Sequence[CommandJob],
    *,
    cpu_workers: int,
    gpu_tokens: Sequence[str],
    log_dir: Path,
) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    cpu_jobs, gpu_jobs = split_jobs(jobs, gpu_tokens=gpu_tokens)
    _safe_print(
        f"resource_queue | total={len(jobs)} | cpu_jobs={len(cpu_jobs)} | gpu_jobs={len(gpu_jobs)} "
        f"| cpu_workers={int(cpu_workers)} | gpu_workers={len(gpu_tokens)}"
    )

    cpu_results: List[Dict[str, Any]] = []
    gpu_results: List[Dict[str, Any]] = []
    failures: List[str] = []

    def _cpu_runner() -> None:
        nonlocal cpu_results
        try:
            cpu_results = _run_cpu_lane(cpu_jobs, cpu_workers=int(cpu_workers), log_dir=log_dir)
        except Exception as exc:
            failures.append(f"cpu:{exc}")

    def _gpu_runner() -> None:
        nonlocal gpu_results
        try:
            gpu_results = _run_gpu_lane(gpu_jobs, gpu_tokens=list(gpu_tokens), log_dir=log_dir)
        except Exception as exc:
            failures.append(f"gpu:{exc}")

    threads: List[threading.Thread] = []
    if cpu_jobs and int(cpu_workers) > 0:
        threads.append(threading.Thread(target=_cpu_runner, name="cpu-lane", daemon=False))
    if gpu_jobs:
        threads.append(threading.Thread(target=_gpu_runner, name="gpu-lane", daemon=False))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    all_results = cpu_results + gpu_results
    n_fail = sum(1 for row in all_results if not bool(row.get("ok", False)))
    if failures:
        n_fail += len(failures)
    return {
        "total": int(len(jobs)),
        "cpu_jobs": int(len(cpu_jobs)),
        "gpu_jobs": int(len(gpu_jobs)),
        "cpu_workers": int(cpu_workers),
        "gpu_workers": int(len(gpu_tokens)),
        "n_fail": int(n_fail),
        "failures": list(failures),
    }
