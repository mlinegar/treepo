from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Dict, List, Optional

from treepo.bench.cardinality_recovery import CardinalityRecoveryConfig
from treepo.bench.runner import EXPERIMENT_CARDINALITY_RECOVERY, RunSpec


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _visible_cuda_count() -> int:
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible:
        return len([item for item in visible.split(",") if item.strip()])
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def build_cardinality_paper_suite(
    *,
    out_root: Path,
    skip_existing: bool,
    seeds: Optional[str] = None,
) -> List[RunSpec]:
    seed_list = _parse_ints(seeds) if seeds is not None else [0, 1, 2]
    audit_policies = ["all", "sqrt", "log2", "fraction"]
    n_cuda = _visible_cuda_count()
    use_cuda = bool(n_cuda > 0)

    base: Dict[str, object] = asdict(CardinalityRecoveryConfig())
    base.update(
        {
            "state_dims": [32, 64, 96, 128],
            "train_docs_grid": [128, 256, 512, 1024],
            "train_sizes": None,
            "n_val": 128,
            "n_test": 256,
            "hidden_dim": 160,
            "n_epochs": 12,
            "batch_size": 24,
            "audit_fraction": 0.25,
            "simulation_mode": "latent_proxy_baseline",
            "use_cuda": use_cuda,
            "data_seed": 0,
        }
    )

    specs: List[RunSpec] = []
    output_root = Path(out_root) / "cardinality" / "paper"
    run_idx = 0
    for audit_policy in audit_policies:
        for seed in seed_list:
            cfg = dict(base)
            cfg["audit_policy"] = str(audit_policy)
            cfg["seed"] = int(seed)
            if use_cuda:
                cfg["cuda_device"] = int(run_idx % n_cuda)
            run_dir = output_root / f"audit_{audit_policy}" / f"seed_{seed}"
            json_out = run_dir / "summary.json"
            csv_out = run_dir / "summary.csv"
            cfg_out = run_dir / "config.yaml"
            if skip_existing and json_out.exists() and csv_out.exists():
                continue
            specs.append(
                RunSpec(
                    experiment=EXPERIMENT_CARDINALITY_RECOVERY,
                    config=cfg,
                    json_out=json_out,
                    csv_out=csv_out,
                    config_out=cfg_out,
                )
            )
            run_idx += 1
    return specs


__all__ = ["build_cardinality_paper_suite"]
