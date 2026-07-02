"""Release-gate checks for the treepo package.

Inspects the source tree and inventory to enforce packaging hygiene: no
forbidden/heavy imports on the public surface, no generated or local-path
artifacts, single import surface, and CLI/example conventions.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACKAGE_ROOT / "src" / "treepo"
INVENTORY_PATH = PACKAGE_ROOT / "inventory.yaml"

GENERATED_PARTS = (
    ".egg-info",
    ".eggs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".pyre",
    ".hypothesis",
    ".tox",
    ".nox",
    ".cache",
    "htmlcov",
    ".ipynb_checkpoints",
    "pip-wheel-metadata",
)
GENERATED_SUFFIXES = (".pyc", ".pyo", ".log", ".tmp", ".prof")
FORBIDDEN_IMPORT_ROOTS = ("src",)
HEAVY_IMPORT_ROOTS = ("datasets", "dspy", "openai", "vllm", "torch", "transformers", "pandas", "sentence_transformers", "trl", "peft")
LOCAL_ABSOLUTE_MARKERS = tuple(f"/{name}/" for name in ("home", "mnt", "Users"))
PIP_INSTALL_MARKERS = ("pip " "install", "python -m " "pip", "pip " "wheel")
TEXT_SUFFIXES = (".md", ".py", ".toml", ".yaml", ".yml")


def read_inventory(path: str | Path = INVENTORY_PATH) -> dict[str, Any]:
    """Load and return the inventory YAML as a dict (empty if malformed)."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(payload or {}) if isinstance(payload, Mapping) else {}


def check_inventory(path: str | Path = INVENTORY_PATH) -> dict[str, Any]:
    """Validate inventory entries/areas against policy; return an ok/failures report."""
    inventory = read_inventory(path)
    policy = dict(inventory.get("policy") or {})
    allowed = set(policy.get("classes") or ())
    entries = [dict(item) for item in list(inventory.get("entries") or [])]
    areas = [dict(item) for item in list(inventory.get("areas") or [])]
    failures: list[dict[str, Any]] = []
    for entry in entries:
        cls = str(entry.get("class") or "")
        if cls not in allowed:
            failures.append({"path": entry.get("path", ""), "reason": "unknown_class", "class": cls})
        if not str(entry.get("path") or ""):
            failures.append({"class": cls, "reason": "missing_path"})
        if cls not in {"outside", "extension"} and not str(entry.get("target") or ""):
            failures.append({"path": entry.get("path", ""), "reason": "missing_target"})
        if not str(entry.get("note") or ""):
            failures.append({"path": entry.get("path", ""), "reason": "missing_note"})
    for area in areas:
        if not str(area.get("status") or ""):
            failures.append({"area": area.get("id", ""), "reason": "missing_status"})
        if not str(area.get("note") or ""):
            failures.append({"area": area.get("id", ""), "reason": "missing_note"})
    return {
        "ok": not failures,
        "entry_count": len(entries),
        "area_count": len(areas),
        "failures": failures,
    }


def check_hygiene(package_root: str | Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Scan sources for forbidden/heavy imports and generated/local-path artifacts."""
    root = Path(package_root)
    py_files = sorted((root / "src" / "treepo").rglob("*.py"))
    candidate_paths = _candidate_paths(root)
    failures: list[dict[str, Any]] = []
    for path in py_files:
        rel = str(path.relative_to(root))
        for name in _import_roots(path):
            if name in FORBIDDEN_IMPORT_ROOTS:
                failures.append({"path": rel, "reason": "forbidden_root_import", "import": name})
            if name in HEAVY_IMPORT_ROOTS and _is_core_light_path(path):
                failures.append({"path": rel, "reason": "heavy_import_in_core", "import": name})
    for path in candidate_paths:
        rel = str(path.relative_to(root))
        if _is_generated_path(path) or rel.endswith(GENERATED_SUFFIXES):
            failures.append({"path": rel, "reason": "generated_artifact"})
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            marker = next((item for item in LOCAL_ABSOLUTE_MARKERS if item in text), "")
            if marker:
                failures.append({"path": rel, "reason": "local_absolute_path", "marker": marker})
            marker = next((item for item in PIP_INSTALL_MARKERS if item in text), "")
            if marker:
                failures.append({"path": rel, "reason": "pip_install_command", "marker": marker})
    return {"ok": not failures, "checked_files": len(py_files), "failures": failures}


def check_release(package_root: str | Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Run all release gates and return an aggregate ok/checks/failures report."""
    root = Path(package_root)
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for name, report in (
        ("inventory", check_inventory(root / "inventory.yaml")),
        ("hygiene", check_hygiene(root)),
        ("public_imports", _check_public_imports(root)),
        ("lazy_exports", _check_lazy_exports(root)),
        ("single_surface", _check_single_surface(root)),
        ("examples", _check_examples(root)),
        ("cli_surface", _check_cli_surface(root)),
    ):
        check = {"name": name, **dict(report)}
        checks.append(check)
        if not bool(report.get("ok")):
            failures.extend({"check": name, **dict(item)} for item in list(report.get("failures") or []))

    return {"ok": not failures, "checks": checks, "failures": failures}


def _is_core_light_path(path: Path) -> bool:
    rel = path.relative_to(SRC_ROOT)
    return rel.parts[0] in {
        "__init__.py",
        "certificate.py",
        "common.py",
        "local_law.py",
        "objective.py",
        "sampling.py",
        "statistic.py",
        "state.py",
    }


def _import_roots(path: Path) -> Iterable[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return ()
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.append(node.module.split(".")[0])
    return roots


def _is_generated_path(path: Path) -> bool:
    for part in path.parts:
        if part in GENERATED_PARTS or part.endswith(".egg-info"):
            return True
    return False


def _candidate_paths(root: Path) -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "--", "."],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return [path for path in root.rglob("*") if path.is_file()]
    out: list[Path] = []
    for line in proc.stdout.splitlines():
        if line.strip():
            path = root / line.strip()
            if path.is_file():
                out.append(path)
    return out


def _check_public_imports(root: Path) -> dict[str, Any]:
    code = """
import json
import sys
import treepo
heavy = ["datasets", "dspy", "openai", "pandas", "peft", "sentence_transformers", "torch", "transformers", "trl", "vllm"]
from treepo import Candidate, ComposableStatistic, PreferenceDataset, PreferenceRecord, TaskState, TreeNode, TreeRecord, family_statistic, fit
print(json.dumps({
    "heavy": {name: name in sys.modules for name in heavy},
    "has_run": hasattr(treepo, "run"),
    "has_list_methods": hasattr(treepo, "list_methods"),
    "dataset_module": PreferenceDataset.__module__,
    "candidate_module": Candidate.__module__,
    "record_module": PreferenceRecord.__module__,
    "statistic_module": ComposableStatistic.__module__,
    "task_state_module": TaskState.__module__,
    "tree_node_module": TreeNode.__module__,
    "tree_record_module": TreeRecord.__module__,
    "family_statistic_module": family_statistic.__module__,
    "fit_module": fit.__module__,
}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "failures": [{"reason": "import_failed", "stderr": proc.stderr.strip()}],
        }
    loaded = json.loads(proc.stdout)
    heavy_loaded = dict(loaded.get("heavy") or {})
    failures = [
        {"reason": "heavy_import_loaded_by_public_import", "module": name}
        for name, is_loaded in heavy_loaded.items()
        if bool(is_loaded)
    ]
    for name in ("has_run", "has_list_methods"):
        if bool(loaded.get(name)):
            failures.append({"reason": "removed_top_level_export_present", "name": name})
    return {"ok": not failures, "loaded": loaded, "failures": failures}


def _check_lazy_exports(root: Path) -> dict[str, Any]:
    code = """
import json
import traceback
import treepo
failures = []
for name in sorted(treepo._LAZY_EXPORTS):
    try:
        getattr(treepo, name)
    except Exception as exc:
        failures.append({
            "name": name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc()[-2000:],
        })
print(json.dumps({"checked_exports": len(treepo._LAZY_EXPORTS), "failures": failures}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "failures": [{"reason": "lazy_export_subprocess_failed", "stderr": proc.stderr.strip()}],
        }
    payload = json.loads(proc.stdout)
    failures = [
        {"reason": "lazy_export_failed", **dict(item)}
        for item in list(payload.get("failures") or [])
    ]
    return {
        "ok": not failures,
        "checked_exports": int(payload.get("checked_exports", 0)),
        "failures": failures,
    }


def _check_single_surface(root: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    text_suffixes = {".py", ".md", ".toml", ".yaml", ".yml"}
    active_roots = ("README.md", "docs", "examples", "src/treepo", "tests")
    old_axis = "estim" + "ator"
    old_rows = "training_" + "exam" + "ples"
    old_fit_call_double = 'run("' + "fit"
    old_fit_call_single = "run('" + "fit"
    banned_anywhere = (
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
        "Feed" + "backDataset",
        "feed" + "back_data",
    )
    docs_examples_banned = (
        "treepo." + "run",
        "treepo." + "list_methods",
        "treepo." + "list_families",
        "treepo." + "list_registered_oracles",
        old_axis + " =",
        '"' + old_axis + '"',
        "'" + old_axis + "'",
    )
    public_vocab_banned = (
        "Training" + "Example",
        "Sample" + "Config",
    )

    for file_path in _text_files(root, active_roots, text_suffixes):
        rel = str(file_path.relative_to(root))
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for token in banned_anywhere:
            if token in text:
                failures.append({"path": rel, "reason": "removed_surface_token", "token": token})
        if rel == "README.md" or rel.startswith(("docs/", "examples/")):
            for token in docs_examples_banned:
                if token in text:
                    failures.append({"path": rel, "reason": "old_public_call_style", "token": token})
            for token in public_vocab_banned:
                if token in text:
                    failures.append({"path": rel, "reason": "old_public_vocab", "token": token})
    return {"ok": not failures, "failures": failures}


def _text_files(root: Path, rel_roots: Iterable[str], suffixes: set[str]) -> list[Path]:
    out: list[Path] = []
    for rel_root in rel_roots:
        path = root / rel_root
        if path.is_file():
            files = [path]
        elif path.exists():
            files = [item for item in path.rglob("*") if item.is_file()]
        else:
            files = []
        out.extend(item for item in files if item.suffix in suffixes)
    return sorted(out)


def _check_examples(root: Path) -> dict[str, Any]:
    from treepo.bench.io import load_yaml_or_json
    from treepo.bench.runner import validate_config_dict

    examples = root / "examples"
    experiment_examples = {
        "bench/classical_sketches.yaml": "classical-sketches",
        "bench/markov.yaml": "markov",
    }
    failures: list[dict[str, Any]] = []
    for filename, experiment in experiment_examples.items():
        try:
            payload = load_yaml_or_json(examples / filename)
            if not isinstance(payload, Mapping):
                raise ValueError("example config must be a mapping")
            validate_config_dict(experiment, payload)
        except Exception as exc:
            failures.append({"path": f"examples/{filename}", "reason": "invalid_example", "error": str(exc)})
    return {"ok": not failures, "checked_examples": len(experiment_examples), "failures": failures}


def _check_cli_surface(root: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "treepo.bench.cli", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    failures: list[dict[str, Any]] = []
    if proc.returncode != 0:
        failures.append({"reason": "cli_help_failed", "stderr": proc.stderr.strip()})
        help_text = ""
    else:
        help_text = proc.stdout
    for name in ("run", "check"):
        if name not in help_text:
            failures.append({"reason": "missing_cli_command", "command": name})
    for name in ("suite", "report", "sweep"):
        if name in help_text:
            failures.append({"reason": "removed_cli_command_present", "command": name})
    return {"ok": not failures, "failures": failures}


__all__ = [
    "check_hygiene",
    "check_inventory",
    "check_release",
    "main",
    "read_inventory",
]


def main() -> int:
    report = check_release()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report.get("ok")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
