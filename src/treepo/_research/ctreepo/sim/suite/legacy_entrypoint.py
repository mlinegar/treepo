from __future__ import annotations

import sys


def legacy_entrypoint_message(*, script_name: str, replacement: str) -> str:
    return (
        f"{script_name} was retired in the v2 simulation API.\n"
        f"Use the canonical suite command instead:\n"
        f"  {replacement}"
    )


def fail_legacy_entrypoint(*, script_name: str, replacement: str, exit_code: int = 2) -> int:
    print(legacy_entrypoint_message(script_name=script_name, replacement=replacement), file=sys.stderr)
    return int(exit_code)


__all__ = ["fail_legacy_entrypoint", "legacy_entrypoint_message"]
