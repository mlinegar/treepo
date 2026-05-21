#!/usr/bin/env python3
"""Example: HLL classical sketch via `treepo.cld.run("sketch", ...)`.

Pattern: HllSketchConfig holds both adapter knobs and fixture knobs;
the dispatcher auto-builds the canonical token-tree fixture when
eval_data is omitted.
"""

from __future__ import annotations

import argparse, dataclasses, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "treepo" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "treepo.cld/configs/hll_sketch.toml")
    ap.add_argument("--precision", type=int, default=None)
    ap.add_argument("--schedule", default=None)
    args = ap.parse_args()

    import treepo.cld
    from treepo.cld.canonical_defaults import load_dataclass, HllSketchConfig

    cfg = load_dataclass(args.config, HllSketchConfig, overrides={
        "precision": args.precision, "schedule": args.schedule,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    payload.update({"sketch_kind": "hll", "output_dir": str(args.output_dir)})
    result = treepo.cld.run("sketch", payload)
    print(f"status={result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
