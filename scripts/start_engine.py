#!/usr/bin/env python3
"""Generic engine launcher that delegates to the engine-specific wrapper.

Looks up the engine via ``treepo._research.core.engines.EngineRegistry`` and
invokes its ``launch_script`` with ``TT_START_ENGINE_DIRECT=1`` set so the
shell wrapper executes directly instead of re-entering this script.

Usage::

    ./scripts/start_engine.py --engine vllm   -- [vllm args...]
    ./scripts/start_engine.py --engine sglang -- [sglang args...]

In practice users invoke the per-engine wrappers (``start_vllm.sh`` /
``start_sglang.sh``) which delegate here when ``TT_START_ENGINE_DIRECT``
is unset.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from treepo._research.core.engines import EngineRegistry, EngineType


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a configured engine wrapper.")
    parser.add_argument(
        "--engine", required=True,
        help="Engine name (for example: vllm, sglang).",
    )
    parser.add_argument(
        "--print-spec", action="store_true",
        help="Print the resolved engine spec and exit without launching.",
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="Arguments forwarded to the engine wrapper.",
    )
    parsed = parser.parse_args()

    spec = EngineRegistry.resolve(EngineType.normalize(parsed.engine))
    if parsed.print_spec:
        print(spec.to_dict())
        return 0
    if not spec.launchable or not spec.launch_script:
        raise SystemExit(
            f"Engine '{spec.engine.value}' does not provide a launchable local wrapper."
        )
    if not os.path.exists(spec.launch_script):
        raise SystemExit(
            f"Engine '{spec.engine.value}' launch script not found at "
            f"{spec.launch_script}. Make sure scripts/ is present in the project root."
        )

    passthrough = list(parsed.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    env = os.environ.copy()
    env["TT_START_ENGINE_DIRECT"] = "1"
    cmd = ["/bin/bash", spec.launch_script, *passthrough]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
