from __future__ import annotations

from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Public `treepo-bench` entrypoint.

    The implementation remains in `treepo.bench.cli` for CLI compatibility while
    the package is reorganized around `treepo.cli` as the public command module.
    """

    from treepo.bench.cli import main as bench_main

    return int(bench_main(argv))


__all__ = ["main"]
