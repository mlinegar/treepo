from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing as mp
import shlex
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from treepo.bench.cardinality_recovery import (
    CardinalityRecoveryConfig,
    run_cardinality_recovery_experiment,
)
from treepo.bench.classical_sketches import (
    ClassicalSketchComparisonConfig,
    run_classical_sketch_comparison,
)
from treepo.bench.env import apply_cpu_thread_limits
from treepo.bench.hll_merge_learning import (
    HLLMergeLearningConfig,
    HLLMergeLearningSummary,
    run_hll_merge_learning_experiment,
)
from treepo.bench.io import (
    add_runtime_meta,
    atomic_write_text,
    dump_json,
    summary_to_csv_row_learned_ops_g,
    summary_to_csv_row_learned_segmented_theta_g,
    summary_to_csv_row_segmented,
    summary_to_csv_rows_cardinality_recovery,
    summary_to_csv_rows_classical_sketches,
    summary_to_csv_rows_hll_merge_learning,
    summary_to_csv_rows_ops,
    write_csv_rows,
)
from treepo.bench.lda.learned_segment_lda_ops_g import (
    LearnedSegmentLDAOpsGConfig,
    run_learned_segment_lda_ops_g_experiment,
)
from treepo.bench.lda.learned_segmented_lda_theta_g import (
    LearnedSegmentedLDATopicThetaGConfig,
    run_learned_segmented_lda_theta_g_experiment,
)
from treepo.bench.lda.segment_lda_ops_weight_recovery import (
    SegmentLDAOpsWeightRecoveryConfig,
    run_segment_lda_ops_weight_recovery_experiment,
)
from treepo.bench.lda.segmented_lda_ctreepo import (
    SegmentedLDACtreePOConfig,
    run_segmented_lda_ctreepo_simulation,
)
from treepo.bench.sweep_spec import SweepSpec, load_sweep_spec
from treepo.runtime import (
    RUNTIME_CONFIG_KEYS,
    run_runtime_eval,
    runtime_summary_to_csv_rows,
)

ExperimentName = str
EXPERIMENT_SEGMENTED = "segmented-lda-ctreepo"
EXPERIMENT_OPS = "segment-lda-ops-weight-recovery"
EXPERIMENT_LEARNED_OPS_G = "learned-segment-lda-ops-g"
EXPERIMENT_LEARNED_SEGMENTED_THETA_G = "learned-segmented-lda-theta-g"
EXPERIMENT_CARDINALITY_RECOVERY = "cardinality-recovery"
EXPERIMENT_HLL_MERGE_LEARNING = "hll-merge-learning"
EXPERIMENT_CLASSICAL_SKETCHES = "classical-sketches"
EXPERIMENT_LONGBENCH_RUNTIME = "longbench-runtime"
VALID_EXPERIMENTS: Tuple[ExperimentName, ...] = (
    EXPERIMENT_SEGMENTED,
    EXPERIMENT_OPS,
    EXPERIMENT_LEARNED_OPS_G,
    EXPERIMENT_LEARNED_SEGMENTED_THETA_G,
    EXPERIMENT_CARDINALITY_RECOVERY,
    EXPERIMENT_HLL_MERGE_LEARNING,
    EXPERIMENT_CLASSICAL_SKETCHES,
    EXPERIMENT_LONGBENCH_RUNTIME,
)


def allowed_config_keys(experiment: ExperimentName) -> set[str]:
    from dataclasses import fields

    if experiment == EXPERIMENT_SEGMENTED:
        return {f.name for f in fields(SegmentedLDACtreePOConfig)}
    if experiment == EXPERIMENT_OPS:
        return {f.name for f in fields(SegmentLDAOpsWeightRecoveryConfig)}
    if experiment == EXPERIMENT_LEARNED_OPS_G:
        return {f.name for f in fields(LearnedSegmentLDAOpsGConfig)}
    if experiment == EXPERIMENT_LEARNED_SEGMENTED_THETA_G:
        return {f.name for f in fields(LearnedSegmentedLDATopicThetaGConfig)}
    if experiment == EXPERIMENT_CARDINALITY_RECOVERY:
        return {f.name for f in fields(CardinalityRecoveryConfig)}
    if experiment == EXPERIMENT_HLL_MERGE_LEARNING:
        return {f.name for f in fields(HLLMergeLearningConfig)}
    if experiment == EXPERIMENT_CLASSICAL_SKETCHES:
        return {f.name for f in fields(ClassicalSketchComparisonConfig)}
    if experiment == EXPERIMENT_LONGBENCH_RUNTIME:
        return set(RUNTIME_CONFIG_KEYS)
    raise ValueError(f"unknown experiment: {experiment!r}")


def validate_config_dict(experiment: ExperimentName, config: Mapping[str, object]) -> None:
    allowed = allowed_config_keys(experiment)
    unknown = sorted([str(k) for k in config.keys() if str(k) not in allowed])
    if unknown:
        raise ValueError(f"unknown config keys for {experiment}: {unknown}")


def _normalized_config_json(config: Mapping[str, object]) -> str:
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _run_id(config: Mapping[str, object]) -> str:
    h = hashlib.sha256(_normalized_config_json(config).encode("utf-8")).hexdigest()
    return h[:12]


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _ensure_unified_g_src_on_path() -> None:
    """Prefer the local unified_g lane when running from the monorepo."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "parallel" / "unified_g_v1" / "src"
        if candidate.exists():
            for path in (str(parent), str(candidate)):
                if path not in sys.path:
                    sys.path.insert(0, path)
            return


def _run_classical_sketches_via_unified_g(
    cfg: ClassicalSketchComparisonConfig,
    *,
    output_dir: Path,
):
    try:
        learning = importlib.import_module("src.ctreepo.learning")
        return learning.fit_classical_sketch_grid(cfg, output_dir=output_dir)
    except ModuleNotFoundError:
        # Standalone treepo installs may not include the monorepo `src.ctreepo`
        # facade. Keep the legacy prototype bridge as a fallback.
        pass

    _ensure_unified_g_src_on_path()
    from unified_g_v1.sketch.classical_sketch_grid import classical_sketch_grid_task
    from unified_g_v1.training.fit import fit

    trainer_cfg = classical_sketch_grid_task(config=cfg)
    return fit(trainer_config=trainer_cfg, output_dir=output_dir)


@dataclass(frozen=True)
class RunSpec:
    experiment: ExperimentName
    config: Dict[str, object]
    json_out: Path
    csv_out: Path
    config_out: Optional[Path] = None

    @property
    def key(self) -> str:
        return f"{self.experiment}:{self.json_out}"


def _write_config_for_command(path: Path, config: Mapping[str, object]) -> None:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyyaml is required to emit commands; install with: pip install pyyaml>=6.0") from e
    atomic_write_text(path, yaml.safe_dump(dict(config), sort_keys=True))


def _identifiable_zero_paths_segmented(
    *,
    output_root: Path,
    configs: Sequence[Mapping[str, object]],
) -> List[Tuple[Path, Path, Path]]:
    """
    Reproduce `scripts/build_segmented_lda_ctreepo_cmds.py` output layout.

    Returns list aligned with `configs`: (json_path, csv_path, config_path).
    """
    proc_values = {str(c.get("topic_process", "segments")).strip().lower() for c in configs}
    theta_values = {str(c.get("leaf_theta_estimator", "lstsq")).strip().lower() for c in configs}
    docs_values = {int(c.get("topic_phi_docs", 0) or 0) for c in configs}
    default_seed_frac = float(SegmentedLDACtreePOConfig().neural_topic_seed_fraction)
    seed_fracs_global = {
        float(c.get("neural_topic_seed_fraction", default_seed_frac))
        for c in configs
        if str(c.get("topic_phi_estimator", "")).strip().lower().startswith("neural_")
    }
    multiple_seed_fracs = len(seed_fracs_global) > 1

    out: List[Tuple[Path, Path, Path]] = []
    for c in configs:
        proc = str(c.get("topic_process", "segments")).strip().lower()
        theta = str(c.get("leaf_theta_estimator", "lstsq")).strip().lower()
        est = str(c.get("topic_phi_estimator", "")).strip()
        est_norm = str(est).strip().lower()
        is_neural = bool(est_norm.startswith("neural_"))

        phi_docs = int(c.get("topic_phi_docs", 0) or 0)
        td = int(c.get("n_books_train"))
        fixed_leaf_tokens = int(c.get("fixed_leaf_tokens"))
        cal = float(c.get("calibration_leaf_query_rate"))
        el = float(c.get("eval_leaf_query_rate"))
        ei = float(c.get("eval_internal_query_rate"))
        seed = int(c.get("seed"))

        proc_prefix = ""
        if len(proc_values) > 1 or proc != "segments":
            proc_prefix = f"tp_{proc}/"

        theta_prefix = ""
        if len(theta_values) > 1 or theta != "lstsq":
            theta_prefix = f"theta_{theta}/"

        docs_component = ""
        if len(docs_values) > 1 or phi_docs != 0:
            docs_component = f"/docs_{phi_docs}"

        seed_component = ""
        if is_neural:
            seed_frac = float(c.get("neural_topic_seed_fraction", default_seed_frac))
            if multiple_seed_fracs or seed_frac != default_seed_frac:
                seed_component = f"/seedfrac_{_fmt_float(seed_frac)}"

        sub = (
            f"{proc_prefix}{theta_prefix}phi_{est}{docs_component}{seed_component}"
            f"/train_{td}/lt_{fixed_leaf_tokens}"
            f"/cal_{_fmt_float(cal)}/leaf_{_fmt_float(el)}/int_{_fmt_float(ei)}"
        )
        base = Path(output_root) / sub / f"seed_{seed}"
        json_out = base.with_suffix(".json")
        csv_out = base.with_suffix(".csv")
        cfg_out = base.with_suffix(".config.yaml")
        out.append((json_out, csv_out, cfg_out))
    return out


def _hash_paths(
    *,
    out_root: Path,
    experiment: ExperimentName,
    config: Mapping[str, object],
) -> Tuple[Path, Path, Path]:
    rid = _run_id(config)
    run_dir = Path(out_root) / experiment / "runs" / rid
    return run_dir / "summary.json", run_dir / "summary.csv", run_dir / "config.yaml"


def build_runs_from_sweep_spec(
    *,
    experiment: ExperimentName,
    spec: SweepSpec,
    out_root: Path,
) -> List[RunSpec]:
    # Cartesian product of grid keys.
    keys = list(spec.grid.keys())
    values_lists = [spec.grid[k] for k in keys]

    configs: List[Dict[str, object]] = []
    if not keys:
        configs = [dict(spec.base)]
    else:
        from itertools import product

        for combo in product(*values_lists):
            c = dict(spec.base)
            for k, v in zip(keys, combo):
                c[str(k)] = v
            configs.append(c)

    for c in configs:
        validate_config_dict(experiment, c)

    runs: List[RunSpec] = []
    if spec.output.layout == "identifiable_zero":
        if experiment != EXPERIMENT_SEGMENTED:
            raise ValueError("output.layout='identifiable_zero' is only supported for segmented-lda-ctreepo sweeps")
        paths = _identifiable_zero_paths_segmented(output_root=Path(out_root), configs=configs)
        for c, (json_out, csv_out, cfg_out) in zip(configs, paths):
            runs.append(RunSpec(experiment=experiment, config=c, json_out=json_out, csv_out=csv_out, config_out=cfg_out))
        return runs

    for c in configs:
        json_out, csv_out, cfg_out = _hash_paths(out_root=Path(out_root), experiment=experiment, config=c)
        runs.append(RunSpec(experiment=experiment, config=c, json_out=json_out, csv_out=csv_out, config_out=cfg_out))
    return runs


def emit_commands(specs: Sequence[RunSpec], *, out_path: Path) -> None:
    lines: List[str] = []
    for spec in specs:
        cfg_path = spec.config_out or spec.json_out.with_suffix(".config.yaml")
        _write_config_for_command(cfg_path, spec.config)
        cmd = (
            f"treepo-bench run {spec.experiment} "
            f"--config {shlex.quote(str(cfg_path))} "
            f"--json-out {shlex.quote(str(spec.json_out))} "
            f"--csv-out {shlex.quote(str(spec.csv_out))}"
        )
        lines.append(cmd)
    atomic_write_text(Path(out_path), "\n".join(lines) + ("\n" if lines else ""))


def _run_one(spec: RunSpec, *, skip_existing: bool) -> Dict[str, object]:
    apply_cpu_thread_limits(threads=1)
    json_out = Path(spec.json_out)
    csv_out = Path(spec.csv_out)

    if skip_existing and json_out.exists() and csv_out.exists():
        return {"status": "skipped", "json_out": str(json_out), "csv_out": str(csv_out)}

    try:
        if spec.experiment == EXPERIMENT_SEGMENTED:
            cfg = SegmentedLDACtreePOConfig(**dict(spec.config))
            summary = run_segmented_lda_ctreepo_simulation(cfg)
            payload = add_runtime_meta(json.loads(summary.to_json()))
            atomic_write_text(json_out, dump_json(payload))
            row = summary_to_csv_row_segmented(summary)
            write_csv_rows(csv_out, [row])
        elif spec.experiment == EXPERIMENT_OPS:
            cfg = SegmentLDAOpsWeightRecoveryConfig(**dict(spec.config))
            summary = run_segment_lda_ops_weight_recovery_experiment(cfg)
            payload = add_runtime_meta(json.loads(summary.to_json()))
            atomic_write_text(json_out, dump_json(payload))
            rows = summary_to_csv_rows_ops(summary)
            write_csv_rows(csv_out, rows)
        elif spec.experiment == EXPERIMENT_LEARNED_OPS_G:
            cfg = LearnedSegmentLDAOpsGConfig(**dict(spec.config))
            summary = run_learned_segment_lda_ops_g_experiment(cfg)
            payload = add_runtime_meta(json.loads(summary.to_json()))
            atomic_write_text(json_out, dump_json(payload))
            row = summary_to_csv_row_learned_ops_g(summary)
            write_csv_rows(csv_out, [row])
        elif spec.experiment == EXPERIMENT_LEARNED_SEGMENTED_THETA_G:
            cfg = LearnedSegmentedLDATopicThetaGConfig(**dict(spec.config))
            summary = run_learned_segmented_lda_theta_g_experiment(cfg)
            payload = add_runtime_meta(json.loads(summary.to_json()))
            atomic_write_text(json_out, dump_json(payload))
            row = summary_to_csv_row_learned_segmented_theta_g(summary)
            write_csv_rows(csv_out, [row])
        elif spec.experiment == EXPERIMENT_CARDINALITY_RECOVERY:
            cfg = CardinalityRecoveryConfig(**dict(spec.config))
            summary = run_cardinality_recovery_experiment(cfg)
            payload = add_runtime_meta(json.loads(summary.to_json()))
            atomic_write_text(json_out, dump_json(payload))
            rows = summary_to_csv_rows_cardinality_recovery(summary)
            write_csv_rows(csv_out, rows)
        elif spec.experiment == EXPERIMENT_HLL_MERGE_LEARNING:
            cfg = HLLMergeLearningConfig(**dict(spec.config))
            runs = run_hll_merge_learning_experiment(cfg)
            summary = HLLMergeLearningSummary(config=asdict(cfg), results=tuple(runs))
            payload = add_runtime_meta(json.loads(summary.to_json()))
            atomic_write_text(json_out, dump_json(payload))
            rows = summary_to_csv_rows_hll_merge_learning(summary)
            write_csv_rows(csv_out, rows)
        elif spec.experiment == EXPERIMENT_CLASSICAL_SKETCHES:
            cfg = ClassicalSketchComparisonConfig(**dict(spec.config))
            execution_backend = str(cfg.execution_backend).strip().lower()
            if execution_backend == "unified_g":
                fit_result = _run_classical_sketches_via_unified_g(
                    cfg,
                    output_dir=json_out.parent / "unified_g_fit",
                )
                summary_payload = dict(fit_result.summary["summary"])
                summary_payload["unified_g"] = {
                    "backend": str(fit_result.backend),
                    "status": str(fit_result.status),
                    "metrics": dict(fit_result.metrics),
                    "artifacts": dict(fit_result.artifacts),
                }
                payload = add_runtime_meta(summary_payload)
                atomic_write_text(json_out, dump_json(payload))
                rows = summary_to_csv_rows_classical_sketches(summary_payload["rows"])
                write_csv_rows(csv_out, rows)
                return {"status": "ok", "json_out": str(json_out), "csv_out": str(csv_out)}
            if execution_backend != "treepo":
                raise ValueError(
                    "ClassicalSketchComparisonConfig.execution_backend must be "
                    f"'unified_g' or 'treepo', got {cfg.execution_backend!r}"
                )
            if bool(cfg.include_learned) and not bool(
                getattr(cfg, "allow_classical_only_learned_ignore", False)
            ):
                raise ValueError(
                    "classical-sketch include_learned=True requires "
                    "execution_backend='unified_g'. Set "
                    "allow_classical_only_learned_ignore=True only for explicit "
                    "legacy classical-only compatibility runs."
                )
            summary = run_classical_sketch_comparison(cfg)
            payload = add_runtime_meta(json.loads(summary.to_json()))
            atomic_write_text(json_out, dump_json(payload))
            rows = summary_to_csv_rows_classical_sketches(summary)
            write_csv_rows(csv_out, rows)
        elif spec.experiment == EXPERIMENT_LONGBENCH_RUNTIME:
            summary = run_runtime_eval(dict(spec.config))
            payload = add_runtime_meta(summary.to_dict())
            atomic_write_text(json_out, dump_json(payload))
            rows = runtime_summary_to_csv_rows(summary)
            write_csv_rows(csv_out, rows)
        else:
            raise ValueError(f"unknown experiment: {spec.experiment}")
    except Exception:
        tb = traceback.format_exc()
        # Hash layout: run_dir/error.txt; identifiable-zero layout: seed_*.error.txt
        if json_out.name == "summary.json":
            err_path = json_out.parent / "error.txt"
        else:
            err_path = json_out.with_suffix(".error.txt")
        atomic_write_text(err_path, tb)
        raise

    return {"status": "ok", "json_out": str(json_out), "csv_out": str(csv_out)}


def run_single(
    *,
    experiment: ExperimentName,
    config: Mapping[str, object],
    json_out: Path,
    csv_out: Path,
    print_json: bool = False,
) -> Dict[str, object]:
    validate_config_dict(experiment, config)
    spec = RunSpec(experiment=experiment, config=dict(config), json_out=Path(json_out), csv_out=Path(csv_out))
    res = _run_one(spec, skip_existing=False)
    if print_json:
        print(Path(json_out).read_text(encoding="utf-8"))
    return res


def run_specs(
    specs: Sequence[RunSpec],
    *,
    jobs: int,
    skip_existing: bool,
) -> List[Dict[str, object]]:
    n_jobs = int(max(1, jobs))
    results: List[Dict[str, object]] = []
    wants_cuda = any(
        bool(spec.config.get("use_cuda", False)) or bool(spec.config.get("learned_use_cuda", False))
        for spec in specs
    )
    mp_context = mp.get_context("spawn") if wants_cuda else None
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_context) as ex:
        futs = {ex.submit(_run_one, spec, skip_existing=bool(skip_existing)): spec for spec in specs}
        for fut in as_completed(futs):
            spec = futs[fut]
            try:
                results.append(dict(fut.result()))
            except Exception as e:
                results.append({"status": "error", "spec": spec.key, "error": str(e)})
    return results


def run_sweep(
    *,
    experiment: ExperimentName,
    spec_path: Path,
    out_root: Path,
    jobs: int,
    skip_existing: bool,
    emit_commands_path: Optional[Path] = None,
    commands_only: bool = False,
) -> List[Dict[str, object]]:
    spec = load_sweep_spec(Path(spec_path))
    runs = build_runs_from_sweep_spec(experiment=experiment, spec=spec, out_root=Path(out_root))
    if emit_commands_path is not None:
        emit_commands(runs, out_path=Path(emit_commands_path))
    if commands_only:
        return [{"status": "commands_only", "n_runs": len(runs)}]
    return run_specs(runs, jobs=jobs, skip_existing=skip_existing)
