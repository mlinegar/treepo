from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
        "single_surface",
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


def test_release_surface_has_single_preference_vocabulary() -> None:
    root = Path(__file__).resolve().parents[1]
    scanned_roots = ["README.md", "docs", "examples", "src", "tests"]
    suffixes = {".py", ".md", ".toml", ".yaml", ".yml"}
    banned = (
        "feed" + "back",
        "Feed" + "back",
        "treepo.methods." + "feed" + "back",
    )
    offenders: list[str] = []
    for rel_root in scanned_roots:
        path = root / rel_root
        files = [path] if path.is_file() else path.rglob("*")
        for file_path in files:
            if not file_path.is_file() or file_path.suffix not in suffixes:
                continue
            rel = file_path.relative_to(root)
            text = file_path.read_text(encoding="utf-8")
            for token in banned:
                if token in text:
                    offenders.append(f"{rel}:{token}")
                    break
    assert offenders == []


def test_release_surface_rejects_removed_fit_channels_and_old_call_style() -> None:
    root = Path(__file__).resolve().parents[1]
    scanned_roots = ["README.md", "docs", "examples", "src", "tests"]
    suffixes = {".py", ".md", ".toml", ".yaml", ".yml"}
    old_axis = "estim" + "ator"
    old_rows = "training_" + "exam" + "ples"
    old_fit_call_double = 'run("' + "fit"
    old_fit_call_single = "run('" + "fit"
    banned = (
        old_fit_call_double,
        old_fit_call_single,
        "g_" + old_axis,
        "f_" + "train_data",
        "g_" + "train_data",
        old_rows,
        "f_" + old_rows,
        "g_" + old_rows,
        "g_" + old_rows[:-1] + "_rows",
        "methods." + "supervision",
        "methods." + "dispatch",
    )
    offenders: list[str] = []
    for rel_root in scanned_roots:
        path = root / rel_root
        files = [path] if path.is_file() else path.rglob("*")
        for file_path in files:
            if not file_path.is_file() or file_path.suffix not in suffixes:
                continue
            rel = file_path.relative_to(root)
            text = file_path.read_text(encoding="utf-8")
            for token in banned:
                if token in text:
                    offenders.append(f"{rel}:{token}")
                    break
    assert offenders == []

    public_docs = [root / "README.md", *(root / "docs").rglob("*"), *(root / "examples").rglob("*")]
    call_tokens = (
        "treepo." + "run",
        old_axis + " =",
        '"' + old_axis + '"',
        "'" + old_axis + "'",
    )
    old_call_sites: list[str] = []
    for file_path in public_docs:
        if not file_path.is_file() or file_path.suffix not in suffixes:
            continue
        text = file_path.read_text(encoding="utf-8")
        if any(token in text for token in call_tokens):
            old_call_sites.append(str(file_path.relative_to(root)))
    assert old_call_sites == []


def test_release_surface_has_no_generated_python_caches() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "tests",
            "examples",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cache_paths = [
        line
        for line in proc.stdout.splitlines()
        if "__pycache__" in Path(line).parts or Path(line).suffix in {".pyc", ".pyo"}
    ]
    assert sorted(cache_paths) == []



def test_finetune_surface_stays_export_only() -> None:
    root = Path(__file__).resolve().parents[1]
    finetune_source = (root / "src/treepo/finetune.py").read_text(encoding="utf-8")
    assert "PreferenceDataset.from_value" in finetune_source
    assert "FineTuneAdapter" in finetune_source

    forbidden_tokens = (
        "sentence_transformers",
        "from trl",
        "import trl",
        "from peft",
        "import peft",
        "accelerate",
        "SFTTrainer",
        "DPOTrainer",
        "RewardTrainer",
        "GRPOTrainer",
        "def train_",
    )
    assert all(token not in finetune_source for token in forbidden_tokens)

    learning_sources = [
        root / "src/treepo/learning.py",
        root / "src/treepo/methods/learning.py",
    ]
    assert all("finetune" not in path.read_text(encoding="utf-8") for path in learning_sources)
