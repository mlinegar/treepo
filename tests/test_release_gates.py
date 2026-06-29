from __future__ import annotations

import json

import pytest

from treepo.cli import main as cli_main
from treepo.release import check_hygiene, check_inventory, check_release


def test_package_inventory_is_classified() -> None:
    report = check_inventory()
    assert report["ok"], report
    assert report["entry_count"] > 0
    assert report["area_count"] > 0


def test_package_hygiene_gate_passes() -> None:
    report = check_hygiene()
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


def test_release_check_passes() -> None:
    report = check_release()
    assert report["ok"], report
    names = {str(check["name"]) for check in report["checks"]}
    assert {
        "inventory",
        "hygiene",
        "public_imports",
        "lazy_exports",
        "examples",
        "cli_surface",
    } <= names


def test_cli_check_release_json(capsys) -> None:
    assert cli_main(["check", "release", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_cli_help_exposes_only_run_and_check(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "run" in help_text
    assert "check" in help_text
    for removed in ("su" + "ite", "rep" + "ort", "sw" + "eep"):
        assert removed not in help_text
