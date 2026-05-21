from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import yaml

from treepo.bench.runner import (
    EXPERIMENT_CARDINALITY_RECOVERY,
    EXPERIMENT_CLASSICAL_SKETCHES,
    EXPERIMENT_HLL_MERGE_LEARNING,
    EXPERIMENT_LONGBENCH_RUNTIME,
    EXPERIMENT_SEGMENTED,
    RunSpec,
)
from treepo.bench.suites.cardinality import build_cardinality_paper_suite
from treepo.bench.suites.classical_sketches import build_classical_sketches_suite
from treepo.bench.suites.identifiable_zero import (
    build_identifiable_zero_dtm_lda,
    build_identifiable_zero_lda_leafnoise,
    build_identifiable_zero_publication_ctreepo,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[4]


def build_paper_smoke_suite(*, out_root: Path, skip_existing: bool) -> List[RunSpec]:
    root = Path(out_root) / "paper_smoke"
    return _filter_existing(
        [
            _example_spec(
                experiment=EXPERIMENT_CARDINALITY_RECOVERY,
                example="cardinality_recovery.yaml",
                run_dir=root / "cardinality_recovery",
            ),
            _example_spec(
                experiment=EXPERIMENT_HLL_MERGE_LEARNING,
                example="hll_merge_learning.yaml",
                run_dir=root / "hll_merge_learning",
            ),
            _example_spec(
                experiment=EXPERIMENT_CLASSICAL_SKETCHES,
                example="classical_sketches.yaml",
                run_dir=root / "classical_sketches",
            ),
            _example_spec(
                experiment=EXPERIMENT_SEGMENTED,
                example="lda_embedding_spectral.yaml",
                run_dir=root / "lda_embedding_spectral",
            ),
            _example_spec(
                experiment=EXPERIMENT_LONGBENCH_RUNTIME,
                example="runtime_all_methods.yaml",
                run_dir=root / "longbench_runtime",
            ),
        ],
        skip_existing=skip_existing,
    )


def build_paper_grids_suite(
    *,
    out_root: Path,
    skip_existing: bool,
    seeds: Optional[str] = None,
    topic_phi_estimators: Optional[str] = None,
    leaf_counts: Optional[str] = None,
    leaf_sizes: Optional[str] = None,
    capacities: Optional[str] = None,
    execution_backend: str = "treepo",
    include_learned: bool = False,
    learned_targets: Optional[str] = None,
    learned_variants: Optional[str] = None,
    learned_n_epochs: int = 150,
    learned_n_train: int = 128,
    learned_n_val: int = 48,
    include_runtime: bool = True,
) -> List[RunSpec]:
    root = Path(out_root) / "paper_grids"
    specs: List[RunSpec] = []
    specs.extend(build_cardinality_paper_suite(out_root=root, skip_existing=skip_existing, seeds=seeds))
    specs.extend(
        build_classical_sketches_suite(
            out_root=root,
            skip_existing=skip_existing,
            seeds=seeds,
            leaf_counts=leaf_counts,
            leaf_sizes=leaf_sizes,
            capacities=capacities,
            execution_backend=execution_backend,
            include_learned=include_learned,
            learned_targets=learned_targets,
            learned_variants=learned_variants,
            learned_n_epochs=learned_n_epochs,
            learned_n_train=learned_n_train,
            learned_n_val=learned_n_val,
        )
    )
    specs.extend(
        build_identifiable_zero_dtm_lda(
            out_root=root,
            skip_existing=skip_existing,
            seeds=seeds,
            topic_phi_estimators=topic_phi_estimators,
        )
    )
    specs.extend(build_identifiable_zero_lda_leafnoise(out_root=root, skip_existing=skip_existing, seeds=seeds))
    specs.extend(build_identifiable_zero_publication_ctreepo(out_root=root, skip_existing=skip_existing, seeds=seeds))
    if include_runtime:
        specs.extend(
            _filter_existing(
                [
                    _example_spec(
                        experiment=EXPERIMENT_LONGBENCH_RUNTIME,
                        example="runtime_all_methods.yaml",
                        run_dir=root / "longbench_runtime",
                    )
                ],
                skip_existing=skip_existing,
            )
        )
    return specs


def _example_spec(*, experiment: str, example: str, run_dir: Path) -> RunSpec:
    cfg_path = PACKAGE_ROOT / "examples" / example
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"example config must be a mapping: {cfg_path}")
    run_dir = Path(run_dir)
    return RunSpec(
        experiment=experiment,
        config=dict(config),
        json_out=run_dir / "summary.json",
        csv_out=run_dir / "summary.csv",
        config_out=run_dir / "config.yaml",
    )


def _filter_existing(specs: Iterable[RunSpec], *, skip_existing: bool) -> List[RunSpec]:
    out: List[RunSpec] = []
    for spec in specs:
        if skip_existing and Path(spec.json_out).exists() and Path(spec.csv_out).exists():
            continue
        out.append(spec)
    return out


__all__ = ["build_paper_grids_suite", "build_paper_smoke_suite"]
