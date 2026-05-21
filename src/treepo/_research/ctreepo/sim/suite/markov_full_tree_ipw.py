from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from treepo._research.ctreepo.sim.core.markov_alignment_validation import (
    build_markov_alignment_audit_report,
    write_markov_alignment_audit_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Markov full-tree IPW reporting suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    report = sub.add_parser("report")
    report.add_argument("--output-root", type=Path, required=True)
    report.add_argument("--out-dir", type=Path, default=None)
    report.add_argument("--bundle-manifest", type=Path, default=None)
    report.add_argument(
        "--run-lean-build",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.cmd != "report":
        raise SystemExit("only the report subcommand is currently supported")
    output_root = args.output_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (output_root / "figures" / "markov_full_tree_ipw")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_markov_alignment_audit_report(
        diagnostics_root=None,
        full_tree_ipw_root=output_root,
        bundle_manifest_path=(
            args.bundle_manifest.resolve()
            if args.bundle_manifest is not None
            else None
        ),
        run_lean_build=bool(args.run_lean_build),
    )
    outputs = write_markov_alignment_audit_report(
        report,
        output_json=out_dir / "markov_full_tree_ipw_latest_diagnostics.json",
        output_markdown=out_dir / "markov_full_tree_ipw_latest.md",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "report_dir": str(out_dir),
                **outputs,
                "summary": dict(report.summary),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if int(report.summary.get("n_fail", 0)) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
