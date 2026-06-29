from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from treepo.bench.io import dump_json, load_yaml_or_json
from treepo.bench.runner import (
    VALID_EXPERIMENTS,
    run_single,
)
from treepo.release import check_hygiene, check_inventory, check_release


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="treepo-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------
    # run
    # ------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run a single experiment and write JSON/CSV.")
    p_run.add_argument("experiment", choices=list(VALID_EXPERIMENTS))
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--json-out", type=Path, required=True)
    p_run.add_argument("--csv-out", type=Path, required=True)
    p_run.add_argument("--print-json", action="store_true", default=False)

    # ------------------------------------------------------------
    # check
    # ------------------------------------------------------------
    p_check = sub.add_parser("check", help="Run package checks.")
    check_sub = p_check.add_subparsers(dest="check", required=True)
    p_check_inv = check_sub.add_parser("inventory", help="Check inventory.yaml.")
    p_check_inv.add_argument("--json", action="store_true", default=False)
    p_check_hygiene = check_sub.add_parser("hygiene", help="Check package hygiene rules.")
    p_check_hygiene.add_argument("--json", action="store_true", default=False)
    p_check_release = check_sub.add_parser("release", help="Run all release checks.")
    p_check_release.add_argument("--json", action="store_true", default=False)

    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.cmd == "run":
        payload = load_yaml_or_json(Path(args.config))
        if not isinstance(payload, dict):
            raise SystemExit("--config must contain a JSON/YAML mapping")
        res = run_single(
            experiment=str(args.experiment),
            config=payload,
            json_out=Path(args.json_out),
            csv_out=Path(args.csv_out),
            print_json=bool(args.print_json),
        )
        if not bool(args.print_json):
            print(dump_json(res))
        return 0

    if args.cmd == "check":
        if args.check == "inventory":
            report = check_inventory()
        elif args.check == "hygiene":
            report = check_hygiene()
        elif args.check == "release":
            report = check_release()
        else:  # pragma: no cover
            raise SystemExit(f"unknown check: {args.check}")
        if bool(args.json):
            print(dump_json(report))
        else:
            print("ok" if report.get("ok") else "FAILED")
            for failure in list(report.get("failures") or []):
                print(dump_json(failure))
        return 0 if bool(report.get("ok")) else 1

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
