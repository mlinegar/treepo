#!/usr/bin/env python3
"""Example: Markov FNO probe via `treepo.cld.run("probe", {...})`.

The probe is a subprocess dispatch — no Python dataclass mirror needed.
The TOML is a flat dict of probe argparse flag values; the dispatcher
validates keys against ``allowed_config_keys("probe")`` and forwards them
as ``--flag value`` to scripts/probe_clean_unified_no.py. Missing flags
fall back to the probe's own argparse defaults.
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "treepo" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "treepo.cld/configs/markov_probe.toml")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override [training].epochs from TOML")
    ap.add_argument("--leaf-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    import treepo.cld

    # TOML is a flat dict of probe argparse knobs. The dispatcher validates
    # against allowed_config_keys("probe") and forwards as CLI flags.
    payload = tomllib.loads(args.config.read_text(encoding="utf-8"))
    payload["output_root"] = str(args.output_root)
    payload["timeout"] = float(args.timeout)
    if args.epochs is not None:
        payload["epochs"] = int(args.epochs)
    if args.leaf_tokens is not None:
        payload["leaf_tokens"] = int(args.leaf_tokens)

    result = treepo.cld.run("probe", payload)
    print(f"status={result['status']}, returncode={result['returncode']}")
    if result.get("summary"):
        s = result["summary"]
        print(f"test_root_mae={s.get('test_root_mae')}, "
              f"best_val_root_mae={s.get('best_val_root_mae')}, "
              f"best_val_epoch={s.get('best_val_epoch')}")
    return 0 if result["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
