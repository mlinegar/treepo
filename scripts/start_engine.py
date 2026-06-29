#!/usr/bin/env python3
"""Generic engine launcher that delegates to the engine-specific wrapper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_SPECS = {
    "vllm": {
        "engine": "vllm",
        "openai_compatible": True,
        "launchable": True,
        "launch_script": str(SCRIPT_DIR / "start_vllm.sh"),
    },
    "sglang": {
        "engine": "sglang",
        "openai_compatible": True,
        "launchable": True,
        "launch_script": str(SCRIPT_DIR / "start_sglang.sh"),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a configured engine wrapper.")
    parser.add_argument(
        "--engine",
        required=True,
        choices=sorted(ENGINE_SPECS),
        help="Engine name.",
    )
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="Print the resolved engine spec and exit without launching.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the engine wrapper.",
    )
    parsed = parser.parse_args()

    spec = dict(ENGINE_SPECS[str(parsed.engine).strip().lower()])
    if parsed.print_spec:
        print(spec)
        return 0

    script = str(spec["launch_script"])
    if not os.path.exists(script):
        raise SystemExit(
            f"Engine {spec['engine']!r} launch script not found at {script}. "
            "Make sure scripts/ is present in the project root."
        )

    passthrough = list(parsed.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    env = os.environ.copy()
    env["TT_START_ENGINE_DIRECT"] = "1"
    return subprocess.call(["/bin/bash", script, *passthrough], env=env)


if __name__ == "__main__":
    sys.exit(main())
