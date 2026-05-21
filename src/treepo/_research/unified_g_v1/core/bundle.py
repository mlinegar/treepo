from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from treepo._research.unified_g_v1.core.manifest import now_iso, write_json


@dataclass(frozen=True)
class StoredRunBundle:
    root: Path
    approach: str
    inputs_dir: Path
    artifacts_dir: Path
    results_dir: Path
    exports_dir: Path
    figures_dir: Path
    training_dir: Path
    manifest_path: Path

    def write_manifest(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        result_paths: Mapping[str, Any] | None = None,
        program_contract: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        payload = {
            "generated_at": now_iso(),
            "approach": self.approach,
            "root": str(self.root),
            "layout": {
                "inputs_dir": str(self.inputs_dir),
                "artifacts_dir": str(self.artifacts_dir),
                "results_dir": str(self.results_dir),
                "exports_dir": str(self.exports_dir),
                "figures_dir": str(self.figures_dir),
                "training_dir": str(self.training_dir),
            },
            "config": dict(config or {}),
            "result_paths": dict(result_paths or {}),
            "program_contract": dict(program_contract or {}),
            "extra": dict(extra or {}),
        }
        return write_json(self.manifest_path, payload)


def create_stored_run_bundle(
    output_root: str | Path,
    *,
    approach: str,
) -> StoredRunBundle:
    root = Path(output_root).expanduser()
    inputs_dir = root / "inputs"
    artifacts_dir = root / "artifacts"
    results_dir = root / "results"
    exports_dir = root / "exports"
    figures_dir = root / "figures"
    training_dir = root / "training"
    for path in (
        root,
        inputs_dir,
        artifacts_dir,
        results_dir,
        exports_dir,
        figures_dir,
        training_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return StoredRunBundle(
        root=root,
        approach=str(approach),
        inputs_dir=inputs_dir,
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        exports_dir=exports_dir,
        figures_dir=figures_dir,
        training_dir=training_dir,
        manifest_path=root / "bundle_manifest.json",
    )
