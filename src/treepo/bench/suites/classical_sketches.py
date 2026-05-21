from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from treepo.bench.classical_sketches import ClassicalSketchComparisonConfig
from treepo.bench.runner import EXPERIMENT_CLASSICAL_SKETCHES, RunSpec


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).replace(",", " ").split() if x.strip()]


def _parse_labels(text: str) -> List[str]:
    return [x.strip().lower() for x in str(text).replace(",", " ").split() if x.strip()]


def _parse_floats(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).replace(",", " ").split() if x.strip()]
    if not values:
        raise ValueError("expected at least one float")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"learned supervision rates must be in [0, 1], got {value!r}")
    return values


def _rate_slug(rate: float) -> str:
    pct = float(rate) * 100.0
    if abs(pct - round(pct)) <= 1e-9:
        return f"R{int(round(pct))}"
    text = f"{pct:.3g}".replace(".", "p")
    return f"R{text}"


def _supervision_slug(leaf_rate: float, internal_rate: float) -> str:
    if abs(float(leaf_rate) - float(internal_rate)) <= 1e-12:
        return _rate_slug(leaf_rate)
    return f"leaf{_rate_slug(leaf_rate)}_internal{_rate_slug(internal_rate)}"


def _supervision_slug3(root_rate: float, leaf_rate: float, internal_rate: float) -> str:
    if (
        abs(float(root_rate) - float(leaf_rate)) <= 1e-12
        and abs(float(root_rate) - float(internal_rate)) <= 1e-12
    ):
        return _rate_slug(root_rate)
    return (
        f"root{_rate_slug(root_rate)}_"
        f"leaf{_rate_slug(leaf_rate)}_"
        f"internal{_rate_slug(internal_rate)}"
    )


CAPACITY_PRESETS = {
    "small": {
        "distinct_lg_k": 8,
        "theta_lg_k": 8,
        "cms_num_buckets": 128,
        "frequent_lg_max_map_size": 7,
        "kll_k": 64,
        "quantiles_k": 64,
        "req_k": 8,
        "tdigest_k": 50,
        "tuple_lg_k": 10,
        "varopt_k": 32,
    },
    "medium": {
        "distinct_lg_k": 10,
        "theta_lg_k": 10,
        "cms_num_buckets": 256,
        "frequent_lg_max_map_size": 8,
        "kll_k": 128,
        "quantiles_k": 128,
        "req_k": 12,
        "tdigest_k": 100,
        "tuple_lg_k": 11,
        "varopt_k": 64,
    },
    "large": {
        "distinct_lg_k": 12,
        "theta_lg_k": 12,
        "cms_num_buckets": 512,
        "frequent_lg_max_map_size": 9,
        "kll_k": 256,
        "quantiles_k": 256,
        "req_k": 16,
        "tdigest_k": 200,
        "tuple_lg_k": 12,
        "varopt_k": 128,
    },
}


def _visible_cuda_count() -> int:
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible:
        return len([part for part in visible.split(",") if part.strip()])
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def build_classical_sketches_suite(
    *,
    out_root: Path,
    skip_existing: bool,
    seeds: Optional[str] = None,
    leaf_counts: Optional[str] = None,
    leaf_sizes: Optional[str] = None,
    capacities: Optional[str] = None,
    execution_backend: str = "unified_g",
    include_learned: bool = False,
    learned_targets: Optional[str] = None,
    learned_variants: Optional[str] = None,
    learned_readout_archs: Optional[str] = None,
    learned_n_epochs: int = 150,
    learned_n_train: int = 128,
    learned_n_val: int = 48,
    learned_batch_size: int = 1024,
    learned_target_jobs: int | str = "auto",
    learned_gpu_ids: str = "auto",
    learned_batch_reference_leaf_size: int = 128,
    learned_max_batch_size: int = 8192,
    learned_eval_every_n_epochs: int = 25,
    learned_local_label_rates: Optional[str] = None,
    learned_root_query_rates: Optional[str] = None,
    learned_leaf_query_rates: Optional[str] = None,
    learned_internal_query_rates: Optional[str] = None,
    learned_supervision_sampling_policy: str = "separate_axes",
) -> List[RunSpec]:
    seed_list = _parse_ints(seeds) if seeds is not None else [0]
    if leaf_counts is not None and leaf_sizes is not None:
        raise ValueError("use either leaf_counts or leaf_sizes, not both")
    leaf_count_list = _parse_ints(leaf_counts) if leaf_counts is not None else []
    leaf_size_list = _parse_ints(leaf_sizes) if leaf_sizes is not None else []
    if not leaf_count_list and not leaf_size_list:
        leaf_size_list = [16, 32, 64, 128, 256]
    capacity_list = _parse_labels(capacities) if capacities is not None else ["small", "medium", "large"]
    learned_target_list = tuple(_parse_labels(learned_targets)) if learned_targets is not None else ("all",)
    learned_variant_list = tuple(_parse_labels(learned_variants)) if learned_variants is not None else ("fg",)
    learned_readout_arch_list = (
        tuple(_parse_labels(learned_readout_archs))
        if learned_readout_archs is not None
        else ("structured",)
    )
    local_rate_list = (
        _parse_floats(learned_local_label_rates)
        if learned_local_label_rates is not None
        else [1.0]
    )
    if learned_leaf_query_rates is None and learned_internal_query_rates is None:
        rate_pairs = [(float(rate), float(rate)) for rate in local_rate_list]
    else:
        leaf_rate_list = (
            _parse_floats(learned_leaf_query_rates)
            if learned_leaf_query_rates is not None
            else list(local_rate_list)
        )
        internal_rate_list = (
            _parse_floats(learned_internal_query_rates)
            if learned_internal_query_rates is not None
            else list(local_rate_list)
        )
        rate_pairs = [
            (float(leaf_rate), float(internal_rate))
            for leaf_rate in leaf_rate_list
            for internal_rate in internal_rate_list
        ]
    explicit_root_rates = (
        _parse_floats(learned_root_query_rates)
        if learned_root_query_rates is not None
        else None
    )
    unknown = [label for label in capacity_list if label not in CAPACITY_PRESETS]
    if unknown:
        valid = ", ".join(sorted(CAPACITY_PRESETS))
        raise ValueError(f"unknown classical-sketch capacity labels {unknown}; expected one of: {valid}")
    specs: List[RunSpec] = []
    n_cuda = _visible_cuda_count() if bool(include_learned) else 0
    run_idx = 0
    output_root = Path(out_root) / "classical_sketches" / "paper"
    base = asdict(
        ClassicalSketchComparisonConfig(
            n_docs=32,
            min_tokens=128,
            max_tokens=512,
            include_families=("distinct", "frequency", "quantile", "set", "sampling"),
        )
    )
    leaf_axis: list[tuple[str, int | None, int]] = []
    for n_leaves in leaf_count_list:
        leaf_axis.append((f"L_{n_leaves}", int(n_leaves), 64))
    for leaf_size in leaf_size_list:
        leaf_axis.append((f"leaf_{leaf_size}tok", None, int(leaf_size)))
    for capacity in capacity_list:
        preset = CAPACITY_PRESETS[capacity]
        for axis_slug, n_leaves, leaf_size in leaf_axis:
            for leaf_rate, internal_rate in rate_pairs:
                policy_slug = str(learned_supervision_sampling_policy)
                if explicit_root_rates is not None:
                    root_rates = list(explicit_root_rates)
                elif policy_slug == "uniform_all_nodes":
                    root_rates = [
                        float(leaf_rate)
                        if abs(float(leaf_rate) - float(internal_rate)) <= 1e-12
                        else max(float(leaf_rate), float(internal_rate))
                    ]
                else:
                    root_rates = [1.0]
                for root_rate in root_rates:
                    supervision_slug = _supervision_slug3(root_rate, leaf_rate, internal_rate)
                    add_supervision_axis = (
                        len(rate_pairs) > 1
                        or len(root_rates) > 1
                        or abs(float(root_rate) - 1.0) > 1e-12
                        or abs(float(leaf_rate) - 1.0) > 1e-12
                        or abs(float(internal_rate) - 1.0) > 1e-12
                        or policy_slug != "separate_axes"
                    )
                    for seed in seed_list:
                        cfg = dict(base)
                        cfg.update(preset)
                        cfg["seed"] = int(seed)
                        cfg["n_leaves"] = int(n_leaves) if n_leaves is not None else None
                        cfg["leaf_size"] = int(leaf_size)
                        cfg["capacity_label"] = str(capacity)
                        cfg["execution_backend"] = str(execution_backend)
                        cfg["include_learned"] = bool(include_learned)
                        cfg["learned_targets"] = learned_target_list
                        cfg["learned_variants"] = learned_variant_list
                        cfg["learned_readout_archs"] = learned_readout_arch_list
                        cfg["learned_n_epochs"] = int(learned_n_epochs)
                        cfg["learned_n_train"] = int(learned_n_train)
                        cfg["learned_n_val"] = int(learned_n_val)
                        cfg["learned_batch_size"] = int(learned_batch_size)
                        cfg["learned_target_jobs"] = str(learned_target_jobs)
                        cfg["learned_gpu_ids"] = str(learned_gpu_ids)
                        cfg["learned_batch_reference_leaf_size"] = int(learned_batch_reference_leaf_size)
                        cfg["learned_max_batch_size"] = int(learned_max_batch_size)
                        cfg["learned_eval_every_n_epochs"] = int(learned_eval_every_n_epochs)
                        cfg["learned_root_query_rate"] = float(root_rate)
                        cfg["learned_leaf_query_rate"] = float(leaf_rate)
                        cfg["learned_internal_query_rate"] = float(internal_rate)
                        cfg["learned_supervision_sampling_policy"] = str(
                            learned_supervision_sampling_policy
                        )
                        cfg["learned_evaluate_train_on_eval"] = False
                        cfg["learned_use_cuda"] = bool(n_cuda > 0)
                        cfg["learned_cuda_device"] = int(run_idx % n_cuda) if n_cuda > 0 else None
                        run_dir = output_root / f"capacity_{capacity}" / axis_slug
                        if add_supervision_axis:
                            if policy_slug != "separate_axes":
                                run_dir = run_dir / policy_slug
                            run_dir = run_dir / supervision_slug
                        run_dir = run_dir / f"seed_{seed}"
                        run_idx += 1
                        json_out = run_dir / "summary.json"
                        csv_out = run_dir / "summary.csv"
                        cfg_out = run_dir / "config.yaml"
                        if skip_existing and json_out.exists() and csv_out.exists():
                            continue
                        specs.append(
                            RunSpec(
                                experiment=EXPERIMENT_CLASSICAL_SKETCHES,
                                config=cfg,
                                json_out=json_out,
                                csv_out=csv_out,
                                config_out=cfg_out,
                            )
                        )
    return specs


__all__ = ["CAPACITY_PRESETS", "build_classical_sketches_suite"]
