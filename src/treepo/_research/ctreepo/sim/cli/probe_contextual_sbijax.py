"""Package entrypoint for the contextual ``sbijax`` probe.

The implementation currently lives in ``scripts/probe_contextual_sbijax.py`` so
existing repo workflows keep working.  This module makes the same command
available through installed package entrypoints and the ``ctreepo`` dispatcher.
"""

from __future__ import annotations

from importlib import import_module
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    module = import_module("scripts.probe_contextual_sbijax")
    return int(module.main(None if argv is None else list(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
