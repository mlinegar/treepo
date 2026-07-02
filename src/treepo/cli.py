"""Public ``treepo-bench`` command entrypoint.

Thin wrapper that defers to ``treepo.bench.cli.main`` so the console script has
a stable, import-light home.
"""

from __future__ import annotations

from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Public `treepo-bench` entrypoint.

    The implementation lives in `treepo.bench.cli`; this module is the public
    command entry point.
    """

    from treepo.bench.cli import main as bench_main

    return int(bench_main(argv))


__all__ = ["main"]
