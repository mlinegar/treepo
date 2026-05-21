from __future__ import annotations

import json
from pathlib import Path

from treepo.cli import main as cli_main
from treepo.release import audit_launch_gate, audit_migration_inventory, audit_package_hygiene


def test_migration_inventory_is_classified() -> None:
    report = audit_migration_inventory()
    assert report["ok"], report
    assert report["entry_count"] > 0
    assert report["required_family_count"] > 0


def test_package_hygiene_gate_passes() -> None:
    report = audit_package_hygiene()
    assert report["ok"], report
    assert report["checked_files"] > 0


def test_cli_check_inventory_json(capsys) -> None:
    assert cli_main(["check", "inventory", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_cli_check_hygiene_json(capsys) -> None:
    assert cli_main(["check", "hygiene", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_launch_gate_passes() -> None:
    report = audit_launch_gate()
    assert report["ok"], report
    names = {str(check["name"]) for check in report["checks"]}
    assert {"inventory", "hygiene", "public_imports", "examples", "paper_suites"} <= names


def test_cli_check_launch_json(capsys) -> None:
    assert cli_main(["check", "launch", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_cli_suite_commands_only_smoke(tmp_path: Path, capsys) -> None:
    assert (
        cli_main(
            [
                "suite",
                "cardinality-paper",
                "--out-root",
                str(tmp_path / "cardinality"),
                "--jobs",
                "1",
                "--commands-only",
                "--seeds",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "commands_only"
    assert payload["n_runs"] > 0


def test_cli_paper_smoke_commands_only(tmp_path: Path, capsys) -> None:
    assert (
        cli_main(
            [
                "suite",
                "paper-smoke",
                "--out-root",
                str(tmp_path / "paper_smoke"),
                "--jobs",
                "1",
                "--commands-only",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "commands_only"
    assert payload["n_runs"] >= 5


def test_cli_paper_grids_commands_only_with_filters(tmp_path: Path, capsys) -> None:
    commands = tmp_path / "commands.sh"
    assert (
        cli_main(
            [
                "suite",
                "paper-grids",
                "--out-root",
                str(tmp_path / "paper_grids"),
                "--jobs",
                "1",
                "--commands-only",
                "--emit-commands",
                str(commands),
                "--seeds",
                "0",
                "--capacities",
                "small",
                "--leaf-counts",
                "1",
                "--topic-phi-estimators",
                "tensor_lda",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "commands_only"
    assert payload["n_runs"] > 5
    assert commands.exists()
