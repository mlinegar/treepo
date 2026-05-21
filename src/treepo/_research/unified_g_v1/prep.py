"""Unified data-preparation CLI.

Usage:
    python -m unified_g_v1.prep --kind embedding --output-dir ... [flags]
    python -m unified_g_v1.prep --kind text       --output-dir ... [flags]
    python -m unified_g_v1.prep --kind preference --output-dir ... [flags]

Each kind dispatches to the corresponding prep script in
`parallel/unified_g_v1/scripts/`. Remaining CLI args are forwarded verbatim
to that script's `main()`.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_DIR = REPO_ROOT / "parallel" / "unified_g_v1" / "scripts"

_KINDS = {
    "embedding": _SCRIPTS_DIR / "prep_embedding_dataset.py",
    "text": _SCRIPTS_DIR / "prep_text_dataset.py",
    "preference": _SCRIPTS_DIR / "prep_preference_dataset.py",
}


def _print_help() -> None:
    print(__doc__ or "")
    print("Valid --kind values:", ", ".join(sorted(_KINDS)))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    kind: str | None = None
    forwarded: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token in {"-h", "--help"}:
            _print_help()
            return 0
        if token == "--kind":
            if i + 1 >= len(args):
                print("error: --kind needs a value", file=sys.stderr)
                return 2
            kind = args[i + 1]
            i += 2
            continue
        if token.startswith("--kind="):
            kind = token.split("=", 1)[1]
            i += 1
            continue
        forwarded.append(token)
        i += 1
    if kind is None:
        print("error: --kind is required (one of: embedding, text, preference)", file=sys.stderr)
        return 2
    if kind not in _KINDS:
        print(f"error: unknown kind {kind!r}", file=sys.stderr)
        return 2
    script_path = _KINDS[kind]
    if not script_path.exists():
        print(f"error: prep script not found: {script_path}", file=sys.stderr)
        return 2
    # Dispatch: rewrite sys.argv so the underlying script sees its own args.
    saved_argv = sys.argv
    sys.argv = [str(script_path), *forwarded]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = saved_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
