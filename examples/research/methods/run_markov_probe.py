#!/usr/bin/env python3
"""Research example: run the standalone Markov FNO probe.

The TOML is a flat dict of probe argparse flag values. Missing flags fall
back to the probe's own argparse defaults.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/research/methods/markov_probe.toml")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs from TOML")
    ap.add_argument("--leaf-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    payload = tomllib.loads(args.config.read_text(encoding="utf-8"))
    payload["output_root"] = str(args.output_root)
    if args.epochs is not None:
        payload["epochs"] = int(args.epochs)
    if args.leaf_tokens is not None:
        payload["leaf_tokens"] = int(args.leaf_tokens)

    script = REPO_ROOT / "scripts/research/probe_clean_unified_no.py"
    cmd = [sys.executable, str(script)]
    for key, value in payload.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    result = subprocess.run(cmd, text=True, timeout=float(args.timeout))
    summary_path = args.output_root / "summary.json"
    print(f"returncode={result.returncode}")
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"test_root_mae={s.get('test_root_mae')}, "
              f"best_val_root_mae={s.get('best_val_root_mae')}, "
              f"best_val_epoch={s.get('best_val_epoch')}")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
