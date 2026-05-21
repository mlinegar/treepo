from __future__ import annotations

from pathlib import Path
from typing import Any

from treepo._research.unified_g_v1.sketch.runner import SketchRunRecord, SketchRunSpec, run_sketch_spec


def run_sketch_synthetic(
    *,
    run_spec: SketchRunSpec,
    output_dir: Path,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Thin adapter: calls `run_sketch_spec` and returns a FitResult-shaped dict.

    Parallel to `run_markov_synthetic`. Synthetic sketch runs do not consume
    a PreparedDataset — their data is generated inside `run_sketch_spec`.
    """
    record: SketchRunRecord = run_sketch_spec(
        run_spec,
        output_root=output_dir,
        reuse_existing=bool(reuse_existing),
    )
    return {
        "backend": "sketch_synthetic",
        "run_key": record.spec.run_key,
        "summary_path": str(record.summary_path),
        "run_dir": str(record.run_dir),
        "final_train_mae": float(record.final_train_mae),
        "final_val_mae": float(record.final_val_mae),
        "merge_recon_mse": float(record.merge_recon_mse),
        "baselines": dict(record.baselines),
        "program_contract": dict(record.program_contract),
        "history": list(record.history),
        "record": record,
    }
