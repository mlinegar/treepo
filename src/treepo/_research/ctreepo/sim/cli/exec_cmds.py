from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from treepo._research.ctreepo.sim.runner import read_cmds_file, run_commands


def _default_jobs() -> int:
    n = os.cpu_count() or 1
    return int(max(1, min(8, n)))


def _default_log_dir(cmds_path: Path) -> Path:
    return cmds_path.parent / f"{cmds_path.stem}_logs"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Execute a command list with bounded parallelism.")
    p.add_argument("--cmds", type=Path, required=True, help="Path to cmds.txt (one shell command per line).")
    p.add_argument("--jobs", type=int, default=_default_jobs(), help="Max concurrent commands.")
    p.add_argument("--log-dir", type=Path, default=None, help="Directory for per-command logs.")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    cmds_path = Path(args.cmds).expanduser().resolve()
    if not cmds_path.exists():
        raise SystemExit(f"cmds file not found: {cmds_path}")

    commands = read_cmds_file(cmds_path)
    log_dir = (Path(args.log_dir) if args.log_dir is not None else _default_log_dir(cmds_path)).resolve()
    if not commands:
        print(json.dumps({"cmds": str(cmds_path), "n_commands": 0, "log_dir": str(log_dir)}, indent=2))
        return 0

    try:
        results = run_commands(
            commands,
            jobs=int(args.jobs),
            log_dir=log_dir,
            fail_fast=bool(args.fail_fast),
        )
    except KeyboardInterrupt:
        return 130

    n_fail = sum(1 for r in results if int(r.returncode) != 0)
    payload = {
        "cmds": str(cmds_path),
        "n_commands": int(len(commands)),
        "n_completed": int(len(results)),
        "n_failed": int(n_fail),
        "log_dir": str(log_dir),
    }
    print(json.dumps(payload, indent=2))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

