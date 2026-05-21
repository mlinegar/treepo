from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def _slug_sha(text: str, *, n: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: int(max(6, min(64, n)))]


def read_cmds_file(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    duration_s: float
    log_path: Path


def run_commands(
    commands: Sequence[str],
    *,
    jobs: int,
    log_dir: Path,
    fail_fast: bool = False,
    env: Optional[dict[str, str]] = None,
) -> List[CommandResult]:
    """
    Execute shell commands with bounded parallelism, writing per-command logs.

    Commands are executed via `bash -lc <cmd>` to preserve quoting semantics.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = int(max(1, jobs))
    env_final = dict(os.environ)
    if env:
        env_final.update({str(k): str(v) for k, v in env.items()})

    pending = list(commands)
    running: List[tuple[subprocess.Popen[bytes], str, float, Path]] = []
    results: List[CommandResult] = []
    failed = False

    def _start(cmd: str, idx: int) -> None:
        slug = _slug_sha(cmd, n=10)
        log_path = log_dir / f"cmd_{idx:05d}_{slug}.log"
        f = open(log_path, "wb")
        start = time.time()
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env_final,
        )
        f.close()
        running.append((proc, cmd, start, log_path))

    try:
        idx = 0
        while pending or running:
            while pending and len(running) < n_jobs and not (fail_fast and failed):
                cmd = pending.pop(0)
                _start(cmd, idx)
                idx += 1

            if not running:
                break

            time.sleep(0.05)
            still: List[tuple[subprocess.Popen[bytes], str, float, Path]] = []
            for proc, cmd, start, log_path in running:
                rc = proc.poll()
                if rc is None:
                    still.append((proc, cmd, start, log_path))
                    continue
                dur = float(time.time() - start)
                results.append(
                    CommandResult(command=cmd, returncode=int(rc), duration_s=dur, log_path=Path(log_path))
                )
                if int(rc) != 0:
                    failed = True
            running = still

            if fail_fast and failed:
                for proc, _cmd, _start, _log_path in running:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                for proc, _cmd, _start, _log_path in running:
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                running = []
    except KeyboardInterrupt:
        for proc, _cmd, _start, _log_path in running:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc, _cmd, _start, _log_path in running:
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        raise

    return results
