from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACKAGE_ROOT / "src" / "treepo"
INVENTORY_PATH = PACKAGE_ROOT / "migration_inventory.yaml"

GENERATED_PARTS = (".egg-info", "__pycache__", ".pytest_cache", ".ruff_cache")
GENERATED_SUFFIXES = (".pyc", ".pyo", ".log")
FORBIDDEN_IMPORT_ROOTS = ("src",)
HEAVY_IMPORT_ROOTS = ("dspy", "openai", "vllm", "torch", "transformers", "pandas")
CORE_LIGHT_PACKAGES = ("treepo", "numpy")
# Subpackages still in migration from the research scaffolding. They are
# allowed to import from ``src.*`` and may carry local absolute paths in
# test fixtures (vLLM endpoints, model files) until their dependencies
# are promoted into the canonical package.
MIGRATION_TIER_PREFIXES = (
    "src/treepo/_research/",  # vendored research scaffolding consumed by methods
    "tests/methods/",
    "configs/research/",
    "examples/research/",
    "scripts/",  # release utilities plus research-only scripts
)
RELEASE_EXCLUDED_PREFIXES = (
    "docs/prepush_review/",
)
LOCAL_ABSOLUTE_MARKERS = tuple(f"/{name}/" for name in ("home", "mnt", "Users"))
PIP_INSTALL_MARKERS = ("pip " "install", "python -m " "pip", "pip " "wheel")
TEXT_SUFFIXES = (".md", ".py", ".toml", ".yaml", ".yml")


def load_migration_inventory(path: str | Path = INVENTORY_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(payload or {}) if isinstance(payload, Mapping) else {}


def audit_migration_inventory(path: str | Path = INVENTORY_PATH) -> dict[str, Any]:
    inventory = load_migration_inventory(path)
    allowed = set(inventory.get("policy", {}).get("allowed_classes") or ())
    entries = [dict(item) for item in list(inventory.get("entries") or [])]
    required = [dict(item) for item in list(inventory.get("required_families") or [])]
    failures: list[dict[str, Any]] = []
    for entry in entries:
        cls = str(entry.get("class") or "")
        if cls not in allowed:
            failures.append({"source": entry.get("source", ""), "reason": "unknown_class", "class": cls})
        if cls != "exclude_legacy" and not str(entry.get("target") or ""):
            failures.append({"source": entry.get("source", ""), "reason": "missing_target"})
        if not str(entry.get("reason") or ""):
            failures.append({"source": entry.get("source", ""), "reason": "missing_reason"})
    for family in required:
        if not str(family.get("status") or ""):
            failures.append({"family": family.get("id", ""), "reason": "missing_status"})
    return {
        "ok": not failures,
        "entry_count": len(entries),
        "required_family_count": len(required),
        "failures": failures,
    }


def audit_package_hygiene(package_root: str | Path = PACKAGE_ROOT) -> dict[str, Any]:
    root = Path(package_root)
    py_files = sorted((root / "src" / "treepo").rglob("*.py"))
    candidate_paths = _candidate_paths(root)
    failures: list[dict[str, Any]] = []
    for path in py_files:
        rel = str(path.relative_to(root))
        in_migration_tier = any(rel.startswith(prefix) for prefix in MIGRATION_TIER_PREFIXES)
        for name in _import_roots(path):
            if name in FORBIDDEN_IMPORT_ROOTS and not in_migration_tier:
                failures.append({"path": rel, "reason": "forbidden_root_import", "import": name})
            if name in HEAVY_IMPORT_ROOTS and _is_core_light_path(path):
                failures.append({"path": rel, "reason": "heavy_import_in_core", "import": name})
    for path in candidate_paths:
        rel = str(path.relative_to(root))
        if any(rel.startswith(prefix) for prefix in RELEASE_EXCLUDED_PREFIXES):
            continue
        in_migration_tier = any(rel.startswith(prefix) for prefix in MIGRATION_TIER_PREFIXES)
        if _is_generated_path(path) or rel.endswith(GENERATED_SUFFIXES):
            failures.append({"path": rel, "reason": "generated_artifact"})
        if path.suffix in TEXT_SUFFIXES and not in_migration_tier:
            text = path.read_text(encoding="utf-8", errors="ignore")
            marker = next((item for item in LOCAL_ABSOLUTE_MARKERS if item in text), "")
            if marker:
                failures.append({"path": rel, "reason": "local_absolute_path", "marker": marker})
            marker = next((item for item in PIP_INSTALL_MARKERS if item in text), "")
            if marker:
                failures.append({"path": rel, "reason": "pip_install_command", "marker": marker})
    return {"ok": not failures, "checked_files": len(py_files), "failures": failures}


def audit_launch_gate(package_root: str | Path = PACKAGE_ROOT) -> dict[str, Any]:
    root = Path(package_root)
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for name, report in (
        ("inventory", audit_migration_inventory(root / "migration_inventory.yaml")),
        ("hygiene", audit_package_hygiene(root)),
        ("public_imports", _audit_public_imports(root)),
        ("lazy_exports", _audit_lazy_exports(root)),
        ("examples", _audit_examples(root)),
        ("paper_suites", _audit_paper_suites(root)),
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
        "core",
        "hll.py",
        "honesty.py",
        "local_law.py",
        "manifest.py",
        "objective.py",
        "paths.py",
        "sampling.py",
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


def _audit_public_imports(root: Path) -> dict[str, Any]:
    code = """
import json
import sys
import treepo
heavy = ["dspy", "openai", "pandas", "torch", "transformers", "vllm"]
print(json.dumps({name: name in sys.modules for name in heavy}, sort_keys=True))
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
    failures = [
        {"reason": "heavy_import_loaded_by_public_import", "module": name}
        for name, is_loaded in dict(loaded).items()
        if bool(is_loaded)
    ]
    return {"ok": not failures, "loaded": loaded, "failures": failures}


def _audit_lazy_exports(root: Path) -> dict[str, Any]:
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


def _audit_examples(root: Path) -> dict[str, Any]:
    from treepo.bench.io import load_yaml_or_json
    from treepo.bench.runner import validate_config_dict
    from treepo.bench.runtime import validate_runtime_config

    examples = root / "examples"
    experiment_examples = {
        "research/bench/cardinality_recovery.yaml": "cardinality-recovery",
        "research/bench/hll_merge_learning.yaml": "hll-merge-learning",
        "research/bench/classical_sketches.yaml": "classical-sketches",
        "research/runtime/runtime_llm_full_context.yaml": "longbench-runtime",
        "research/runtime/runtime_embedding_retrieval.yaml": "longbench-runtime",
        "research/runtime/runtime_summary_tree.yaml": "longbench-runtime",
        "research/runtime/runtime_fno_state_model.yaml": "longbench-runtime",
        "research/runtime/runtime_all_methods.yaml": "longbench-runtime",
    }
    failures: list[dict[str, Any]] = []
    for filename, experiment in experiment_examples.items():
        try:
            payload = load_yaml_or_json(examples / filename)
            if not isinstance(payload, Mapping):
                raise ValueError("example config must be a mapping")
            validate_config_dict(experiment, payload)
            if experiment == "longbench-runtime":
                validate_runtime_config(payload)
        except Exception as exc:
            failures.append({"path": f"examples/{filename}", "reason": "invalid_example", "error": str(exc)})
    try:
        fixture = "research/runtime/longbench_v2_tiny.yaml"
        data = load_yaml_or_json(examples / fixture)
        rows = data.get("rows") if isinstance(data, Mapping) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("expected non-empty rows list")
    except Exception as exc:
        failures.append({"path": "examples/research/runtime/longbench_v2_tiny.yaml", "reason": "invalid_fixture", "error": str(exc)})
    return {"ok": not failures, "checked_examples": len(experiment_examples) + 1, "failures": failures}


def _audit_paper_suites(root: Path) -> dict[str, Any]:
    from treepo.bench.suites.paper import build_paper_grids_suite, build_paper_smoke_suite

    tmp = Path(tempfile.gettempdir())
    failures: list[dict[str, Any]] = []
    try:
        smoke = build_paper_smoke_suite(out_root=tmp / "treepo_launch_gate_smoke", skip_existing=False)
        if len(smoke) < 4:
            failures.append({"reason": "paper_smoke_too_small", "n_runs": len(smoke)})
    except Exception as exc:
        failures.append({"reason": "paper_smoke_failed", "error": str(exc)})
        smoke = []
    try:
        grids = build_paper_grids_suite(
            out_root=tmp / "treepo_launch_gate_grids",
            skip_existing=False,
            seeds="0",
            capacities="small",
            leaf_counts="1",
        )
        if len(grids) <= len(smoke):
            failures.append({"reason": "paper_grids_too_small", "n_runs": len(grids)})
    except Exception as exc:
        failures.append({"reason": "paper_grids_failed", "error": str(exc)})
        grids = []
    return {
        "ok": not failures,
        "paper_smoke_runs": len(smoke),
        "filtered_paper_grid_runs": len(grids),
        "failures": failures,
    }


__all__ = [
    "audit_launch_gate",
    "audit_migration_inventory",
    "audit_package_hygiene",
    "load_migration_inventory",
]


def main() -> int:
    report = audit_launch_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report.get("ok")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
