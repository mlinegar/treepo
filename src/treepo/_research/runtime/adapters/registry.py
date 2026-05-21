from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from treepo._research.runtime.adapters.base import BenchmarkAdapter
from treepo._research.runtime.contracts import RunUnit


def build_benchmark_adapter(
    *,
    spec: Mapping[str, Any],
    run_dir: Path,
    unit: RunUnit,
) -> BenchmarkAdapter:
    """Construct a benchmark adapter for one run unit."""

    bench_cfg: Dict[str, Any] = dict(spec.get("benchmark", {}) or {})
    bench_cfg.update(dict(unit.benchmark_overrides or {}))
    name = str(unit.benchmark or bench_cfg.get("name", "")).strip().lower()

    if name == "ruler_synthetic":
        from treepo._research.runtime.adapters.ruler import RulerDatasetSpec, RulerSyntheticAdapter

        ds_spec = RulerDatasetSpec(
            task_id=unit.task_id,
            split=unit.split,
            max_seq_length=unit.max_seq_length,
            num_samples=unit.num_samples,
            seed=unit.seed,
        )
        return RulerSyntheticAdapter(
            ruler_dir=Path(bench_cfg.get("ruler_dir", "outside_data/RULER")).resolve(),
            dataset_root=run_dir / "datasets",
            spec=ds_spec,
            benchmark_name=str(bench_cfg.get("benchmark_name", "synthetic")),
            tokenizer_type=str(bench_cfg.get("tokenizer_type", "openai")),
            tokenizer_path=str(bench_cfg.get("tokenizer_path", "cl100k_base")),
            model_template_type=str(bench_cfg.get("model_template_type", "base")),
            ensure_prepared=bool(bench_cfg.get("ensure_prepared", True)),
        )

    if name in {"longbench_v2", "longbench-v2", "longbench"}:
        from treepo._research.runtime.adapters.longbench import LongBenchV2Adapter, LongBenchV2Spec

        dataset_path = bench_cfg.get("dataset_path")
        return LongBenchV2Adapter(
            spec=LongBenchV2Spec(
                task_id=unit.task_id,
                split=unit.split,
                max_seq_length=unit.max_seq_length,
                num_samples=unit.num_samples,
                seed=unit.seed,
            ),
            dataset_path=Path(dataset_path) if dataset_path else None,
            hf_dataset=str(bench_cfg.get("hf_dataset", "THUDM/LongBench-v2")),
            hf_config=bench_cfg.get("hf_config"),
            streaming=bool(bench_cfg.get("streaming", False)),
            domains=bench_cfg.get("domains"),
            sub_domains=bench_cfg.get("sub_domains"),
            difficulties=bench_cfg.get("difficulties"),
            length_buckets=bench_cfg.get("length_buckets"),
        )

    raise ValueError(f"Unknown runtime benchmark adapter: {name!r}")


__all__ = ["build_benchmark_adapter"]

