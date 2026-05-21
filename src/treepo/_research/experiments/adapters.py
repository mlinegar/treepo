from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo._research.ctreepo.sim.manifest import read_manifest_jsonl
from treepo._research.experiments.contracts import (
    ExperimentSpec,
    benchmark_ref_from_parts,
    default_phase_specs,
    method_ref_from_parts,
)
from treepo._research.experiments.legacy import ctreepo_runs_to_experiment, runtime_run_spec_to_experiment
from treepo._research.experiments.markov_full_doc import method_ref_from_markov_full_doc_run
from treepo._research.experiments.registry import register_method_adapter
from treepo._research.experiments.roles import (
    ROLE_SCORER,
    ROLE_STATE_MODEL,
    chat_role_ref,
    embedder_role_ref,
    metadata_with_roles,
    oracle_ref,
    state_model_role_ref,
)
from treepo._research.experiments.sidecars import sidecar_root_for_output_file
from treepo._research.runtime.contracts import RunPhaseSpec, RunSpec


def _strip_python(command: Sequence[str]) -> tuple[str, list[str]]:
    parts = [str(item) for item in list(command)]
    if not parts:
        raise ValueError("empty command")
    if parts[0].endswith("python") or parts[0].endswith("python3") or parts[0].endswith("pytest") or "python" in Path(parts[0]).name:
        if len(parts) < 2:
            raise ValueError("python command missing script path")
        return parts[1], parts[2:]
    return parts[0], parts[1:]


def _flag_value(args: Sequence[str], flag: str) -> str:
    items = [str(item) for item in list(args)]
    for idx, token in enumerate(items):
        if token == str(flag) and idx + 1 < len(items):
            return str(items[idx + 1]).strip()
        prefix = f"{flag}="
        if token.startswith(prefix):
            return str(token[len(prefix):]).strip()
    return ""


@register_method_adapter
class MarkovTreeAdapter:
    adapter_id = "markov_tree"
    aliases = ("markov", "tree_neural", "publication_bundle", "tradeoff_pipeline")

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        script_name, argv = _strip_python(command)
        script_basename = Path(script_name).name
        output_root = _flag_value(argv, "--output-root")
        if not output_root:
            raise ValueError("markov_tree adapter requires --output-root")
        output_root_path = (cwd / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root).resolve()
        title = script_basename.replace(".py", "")
        phases = []
        packages: list[str] = []
        benchmarks: list[dict[str, Any]] = []
        if script_basename == "run_markov_optimization_tradeoff_pipeline.py":
            mod = importlib.import_module("scripts.run_markov_optimization_tradeoff_pipeline")
            args = mod._parse_args(argv)
            devices = mod._resolve_devices(args)
            plan = mod.build_run_plan(args, devices=devices)
            phase_counts = dict(plan.get("phase_task_counts", {}) or {})
            phases = [
                phase_name
                for phase_name in phase_counts.keys()
            ]
            recovery = dict(plan.get("resolved_selection", {}) or {})
            packages = [str(item) for item in list(recovery.get("supervision_recovery_packages") or ())]
            structural_cell = str(recovery.get("supervision_recovery_structural_cell", "") or "")
            benchmarks = [
                {"family": "markov_full_doc", "scope": "recoverable_v4", "name": "recoverable_v4"},
                {"family": "markov_full_doc", "scope": "structural_core_v1", "cell": structural_cell, "name": f"structural_core_v1::{structural_cell}" if structural_cell else "structural_core_v1"},
            ]
        elif script_basename == "run_markov_publication_bundle.py":
            mod = importlib.import_module("scripts.run_markov_publication_bundle")
            args = mod._parse_args(argv)
            migs = mod._resolved_mig_uuids(args)
            plan = mod.build_publication_run_plan(args, mig_uuids=migs, output_root=output_root_path)
            phases = list(plan.get("resolved_selection", {}).get("phases") or [])
            tradeoff = dict(plan.get("resolved_selection", {}).get("tradeoff") or {})
            structural_cell = str(tradeoff.get("supervision_recovery_structural_cell", "") or "")
            benchmarks = [
                {"family": "markov_full_doc", "scope": "recoverable_v4", "name": "recoverable_v4"},
                {"family": "markov_full_doc", "scope": "structural_core_v1", "cell": structural_cell, "name": f"structural_core_v1::{structural_cell}" if structural_cell else "structural_core_v1"},
            ]
        else:
            phases = ["screen", "locked", "report"]
        benchmark_refs = tuple(
            benchmark_ref_from_parts(
                family=str(item.get("family", "markov_full_doc")),
                scope=str(item.get("scope", "") or ""),
                cell=str(item.get("cell", "") or ""),
                name=str(item.get("name", "") or ""),
            )
            for item in benchmarks
        )
        method_refs = (
            method_ref_from_markov_full_doc_run(
                family="tree_neural",
                variant="family_default",
                adapter=self.adapter_id,
                metadata={"packages": packages},
            ),
            method_ref_from_markov_full_doc_run(
                family="official_fno",
                variant="family_default",
                adapter=self.adapter_id,
                metadata={"packages": packages},
            ),
        )
        return ExperimentSpec.create(
            adapter_id=self.adapter_id,
            output_root=str(output_root_path),
            title=title,
            benchmark_refs=benchmark_refs,
            method_refs=method_refs,
            phases=default_phase_specs(phases),
            report_profiles=("tradeoff", "publication_bundle", "supervision_recovery"),
            launch_command=command,
            resume_command=command,
            metadata={"legacy_script": script_basename},
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        candidates = {
            "pipeline_summary_json": output_root / "pipeline_summary.json",
            "tradeoff_report_summary_json": output_root / "tradeoff_report" / "summary.json",
            "tradeoff_report_pdf": output_root / "tradeoff_report" / "report.pdf",
            "supervision_recovery_summary_json": output_root / "supervision_recovery" / "summary.json",
        }
        return {key: str(path) for key, path in candidates.items() if path.exists()}


@register_method_adapter
class RuntimeEvalAdapter:
    adapter_id = "runtime_eval"
    aliases = ("runtime", "runtime_evaluation")

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        script_name, argv = _strip_python(command)
        if Path(script_name).name != "run_runtime_eval.py":
            raise ValueError("runtime_eval adapter expects run_runtime_eval.py")
        if not argv or argv[0] != "init":
            raise ValueError("runtime_eval adapter expects the init subcommand")
        config_path = _flag_value(argv[1:], "--config")
        output_dir = _flag_value(argv[1:], "--output-dir") or "outputs/evals"
        experiment_id = _flag_value(argv[1:], "--experiment-id") or ""
        mod = importlib.import_module("scripts.run_runtime_eval")
        spec = mod._load_run_spec(
            Path(config_path).resolve(),
            output_dir=(cwd / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve(),
            experiment_id=(experiment_id or None),
        )
        return runtime_run_spec_to_experiment(
            spec,
            launch_command=command,
            adapter_id=self.adapter_id,
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        candidates = {
            "metrics_json": output_root / "metrics.json",
            "merged_steps_jsonl": output_root / "steps.jsonl",
            "merged_predictions_jsonl": output_root / "predictions.jsonl",
            "merged_calls_jsonl": output_root / "calls.jsonl",
            "units_jsonl": output_root / "units.jsonl",
        }
        return {key: str(path) for key, path in candidates.items() if path.exists()}


@register_method_adapter
class CTreePOSimAdapter:
    adapter_id = "ctreepo_sim"
    aliases = ("ctreepo", "suite_manifest")

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        _script_name, argv = _strip_python(command)
        manifest_path = _flag_value(argv, "--manifest") or _flag_value(argv, "--runspec-manifest")
        if not manifest_path:
            raise ValueError("ctreepo_sim adapter requires --manifest or --runspec-manifest")
        manifest = Path(manifest_path).expanduser()
        if not manifest.is_absolute():
            manifest = (cwd / manifest).resolve()
        runs = read_manifest_jsonl(manifest)
        return ctreepo_runs_to_experiment(
            runs,
            output_root=str(manifest.parent),
            title=manifest.stem,
            adapter_id=self.adapter_id,
            launch_command=command,
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        return {}


@register_method_adapter
class TreePOTrainingAdapter:
    adapter_id = "treepo_training"
    aliases = ("treepo", "training_pipeline", "train_neural_operators", "train_ctreepo")

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        script_name, argv = _strip_python(command)
        script_basename = Path(script_name).name
        output_root = _flag_value(argv, "--output-dir")
        if not output_root:
            raise ValueError("treepo_training adapter requires --output-dir")
        output_root_path = (cwd / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root).resolve()
        task_name = _flag_value(argv, "--task") or "manifesto_rile"
        benchmark_ref = benchmark_ref_from_parts(
            family="treepo_task",
            scope=str(task_name),
            name=str(task_name),
        )
        model_name = _flag_value(argv, "--model") or _flag_value(argv, "--scorer-model")
        base_roles = {
            ROLE_SCORER: chat_role_ref(
                role=ROLE_SCORER,
                model=model_name,
                metadata={"task": task_name, "script": script_basename},
            )
        }
        method_refs = []
        if script_basename in {"train_neural_operators.py"}:
            method_refs.extend(
                [
                    method_ref_from_parts(
                        family="ctreepo",
                        variant="local_law_training",
                        adapter=self.adapter_id,
                        metadata=metadata_with_roles(
                            {"task": task_name},
                            roles={
                                **base_roles,
                                ROLE_STATE_MODEL: state_model_role_ref(
                                    engine="pytorch",
                                    model="ctreepo",
                                    execution_mode="training",
                                ),
                            },
                            oracle=oracle_ref(kind="training_labels", source=task_name),
                        ),
                    ),
                    method_ref_from_parts(
                        family="mergeable_sketch",
                        variant="embedding_sketch_training",
                        adapter=self.adapter_id,
                        metadata=metadata_with_roles(
                            {"task": task_name},
                            roles={
                                **base_roles,
                                "embedder": embedder_role_ref(
                                    engine="local",
                                    model="embedding_proxy",
                                ),
                                ROLE_STATE_MODEL: state_model_role_ref(
                                    engine="pytorch",
                                    model="mergeable_sketch",
                                    execution_mode="training",
                                ),
                            },
                            oracle=oracle_ref(kind="training_labels", source=task_name),
                        ),
                    ),
                ]
            )
        elif script_basename in {"train_ctreepo.py"}:
            method_refs.append(
                method_ref_from_parts(
                    family="ctreepo",
                    variant="local_law_training",
                    adapter=self.adapter_id,
                    metadata=metadata_with_roles(
                        {"task": task_name},
                        roles={
                            **base_roles,
                            ROLE_STATE_MODEL: state_model_role_ref(
                                engine="pytorch",
                                model="ctreepo",
                                execution_mode="training",
                            ),
                        },
                        oracle=oracle_ref(kind="training_labels", source=task_name),
                    ),
                )
            )
        else:
            method_refs.extend(
                [
                    method_ref_from_parts(
                        family="llm_prompt_optimization",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                        metadata=metadata_with_roles(
                            {"task": task_name},
                            roles={
                                **base_roles,
                                "summarizer": chat_role_ref(
                                    role="summarizer",
                                    model=model_name,
                                    defaulted_from="scorer",
                                ),
                            },
                            oracle=oracle_ref(kind="training_labels", source=task_name),
                        ),
                    ),
                    method_ref_from_parts(
                        family="embedding_proxy",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                        metadata=metadata_with_roles(
                            {"task": task_name},
                            roles={
                                **base_roles,
                                "embedder": embedder_role_ref(
                                    engine="local",
                                    model="embedding_proxy",
                                ),
                            },
                            oracle=oracle_ref(kind="training_labels", source=task_name),
                        ),
                    ),
                    method_ref_from_parts(
                        family="ctreepo",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                        metadata=metadata_with_roles(
                            {"task": task_name},
                            roles={
                                **base_roles,
                                ROLE_STATE_MODEL: state_model_role_ref(
                                    engine="pytorch",
                                    model="ctreepo",
                                    execution_mode="training",
                                ),
                            },
                            oracle=oracle_ref(kind="training_labels", source=task_name),
                        ),
                    ),
                    method_ref_from_parts(
                        family="mergeable_sketch",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                        metadata=metadata_with_roles(
                            {"task": task_name},
                            roles={
                                **base_roles,
                                "embedder": embedder_role_ref(
                                    engine="local",
                                    model="embedding_proxy",
                                ),
                                ROLE_STATE_MODEL: state_model_role_ref(
                                    engine="pytorch",
                                    model="mergeable_sketch",
                                    execution_mode="training",
                                ),
                            },
                            oracle=oracle_ref(kind="training_labels", source=task_name),
                        ),
                    ),
                    method_ref_from_parts(
                        family="generator_finetune",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                        metadata=metadata_with_roles(
                            {"task": task_name},
                            roles=base_roles,
                            oracle=oracle_ref(kind="training_labels", source=task_name),
                        ),
                    ),
                ]
            )
        return ExperimentSpec.create(
            adapter_id=self.adapter_id,
            output_root=str(output_root_path),
            title=script_basename.replace(".py", ""),
            benchmark_refs=(benchmark_ref,),
            method_refs=tuple(method_refs),
            phases=default_phase_specs(("train", "eval", "aggregate", "report")),
            report_profiles=("runtime_eval_summary",),
            launch_command=command,
            resume_command=command,
            metadata={"legacy_script": script_basename, "task": task_name},
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        candidates = {
            "summary_json": output_root / "summary.json",
            "final_stats_json": output_root / "final_stats.json",
            "score_report_pdf": output_root / "score_report.pdf",
            "optimizer_audit_manifest_json": output_root / "optimizer_audit_manifest.json",
            "ctreepo_training_result_json": output_root / "ctreepo" / "training_result.json",
            "ctreepo_best_model": output_root / "ctreepo" / "best.pt",
            "mergeable_metrics_json": output_root / "mergeable_sketch" / "metrics.json",
        }
        return {key: str(path) for key, path in candidates.items() if path.exists()}


def _path_from_flag(args: Sequence[str], flag: str, *, cwd: Path) -> Path | None:
    value = _flag_value(args, flag)
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (cwd / path).resolve()


def _script_family(script_basename: str) -> tuple[str, str, str]:
    if script_basename == "run_longbench_batched_example.py":
        return "longbench_v2", "longbench_batched_tree", "batched_doc_pipeline"
    if script_basename == "run_manifesto_batched_example.py":
        return "manifesto_rile", "manifesto_batched_tree", "batched_doc_pipeline"
    if script_basename == "build_manifesto_coverage_split.py":
        return "manifesto_rile", "coverage_split", "coverage_split_builder"
    if script_basename == "run_manifesto_full_doc_gemma4_benchmark.py":
        return "manifesto_rile", "full_doc_direct_scorer", "manifesto_full_doc_direct"
    if script_basename == "run_manifesto_full_doc_dspy_global_f.py":
        return "manifesto_rile", "full_doc_dspy_global_f", "manifesto_full_doc_dspy_global_f"
    if script_basename == "run_method_compare.py":
        return "method_compare", "method_compare", "method_compare"
    if script_basename == "run_method_compare_lbv2.py":
        return "longbench_v2", "method_compare_lbv2", "method_compare_lbv2"
    if script_basename in {"generate_manifesto_lawstress.py", "eval_manifesto_lawstress.py"}:
        return "manifesto_lawstress", script_basename.replace(".py", ""), "lawstress"
    if script_basename == "generate_manifesto_teacher_traces.py":
        return "manifesto_teacher_traces", "teacher_trace_generation", "teacher_trace_generation"
    if script_basename == "run_tree_batching_benchmark.py":
        return "tree_batching", "tree_batching_benchmark", "tree_batching"
    if script_basename == "run_classical_parity_benchmark.py":
        return "hll_parity", "classical_parity_benchmark", "classical_parity_benchmark"
    if script_basename == "report_runtime_v1_results.py":
        return "runtime_v1", "runtime_v1_report", "runtime_v1_report"
    if script_basename in {"run_treepo_stack_generate_demo.py", "run_treepo_stack_markov_demo.py"}:
        return "treepo_stack_demo", script_basename.replace(".py", ""), "treepo_stack_demo"
    return "runtime_umbrella_script", script_basename.replace(".py", ""), "runtime_umbrella_script"


def _inferred_role_refs(script_basename: str, argv: Sequence[str]) -> dict[str, Any]:
    model = _flag_value(argv, "--model") or _flag_value(argv, "--scorer-model")
    base_url = _flag_value(argv, "--base-url") or _flag_value(argv, "--scorer-base-url")
    teacher_model = _flag_value(argv, "--teacher-model")
    teacher_base_url = _flag_value(argv, "--teacher-base-url")
    summarizer_model = _flag_value(argv, "--summarizer-model") or teacher_model or model
    summarizer_base_url = _flag_value(argv, "--summarizer-base-url") or teacher_base_url or base_url
    roles: dict[str, Any] = {}
    if script_basename in {
        "run_longbench_batched_example.py",
        "run_manifesto_batched_example.py",
        "run_method_compare.py",
        "run_method_compare_lbv2.py",
        "eval_manifesto_lawstress.py",
        "run_manifesto_full_doc_gemma4_benchmark.py",
        "run_manifesto_full_doc_dspy_global_f.py",
    }:
        roles[ROLE_SCORER] = chat_role_ref(
            role=ROLE_SCORER,
            model=model,
            base_url=base_url,
        )
    if script_basename in {
        "run_longbench_batched_example.py",
        "run_manifesto_batched_example.py",
        "generate_manifesto_lawstress.py",
        "eval_manifesto_lawstress.py",
        "generate_manifesto_teacher_traces.py",
        "run_tree_batching_benchmark.py",
        "run_treepo_stack_generate_demo.py",
    }:
        roles["summarizer"] = chat_role_ref(
            role="summarizer",
            model=summarizer_model,
            base_url=summarizer_base_url,
            defaulted_from="scorer" if summarizer_model == model and summarizer_base_url == base_url else "",
        )
    if script_basename in {"run_treepo_stack_markov_demo.py", "run_treepo_stack_generate_demo.py"}:
        roles[ROLE_STATE_MODEL] = state_model_role_ref(
            engine="treepo_stack",
            model=script_basename.replace(".py", ""),
        )
    if script_basename == "run_classical_parity_benchmark.py":
        roles[ROLE_SCORER] = {
            "role": ROLE_SCORER,
            "surface": "native",
            "engine": "python",
            "model": "classical_or_learned_hll",
        }
        roles[ROLE_STATE_MODEL] = state_model_role_ref(
            engine="pytorch",
            model="learned_hll_state",
            execution_mode="fit",
        )
    if script_basename == "report_runtime_v1_results.py":
        roles[ROLE_SCORER] = {
            "role": ROLE_SCORER,
            "surface": "report",
            "engine": "python",
            "model": "runtime_v1_results",
        }
    return roles


@register_method_adapter
class RuntimeUmbrellaScriptAdapter:
    adapter_id = "runtime_umbrella_script"
    aliases = (
        "umbrella",
        "batched_doc_pipeline",
        "method_compare",
        "method_compare_lbv2",
        "lawstress",
        "teacher_trace_generation",
        "tree_batching",
        "treepo_stack_demo",
        "classical_parity_benchmark",
        "runtime_v1_report",
        "coverage_split_builder",
        "manifesto_full_doc_direct",
    )

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        script_name, argv = _strip_python(command)
        script_basename = Path(script_name).name
        explicit_root = (
            _path_from_flag(argv, "--output-root", cwd=cwd)
            or _path_from_flag(argv, "--output-dir", cwd=cwd)
            or _path_from_flag(argv, "--out", cwd=cwd)
        )
        experiment_dir = _path_from_flag(argv, "--experiment-dir", cwd=cwd)
        output_file = (
            _path_from_flag(argv, "--output", cwd=cwd)
            or _path_from_flag(argv, "--output-json", cwd=cwd)
            or _path_from_flag(argv, "--output-jsonl", cwd=cwd)
        )
        if explicit_root is not None:
            output_root = (
                explicit_root / "hll"
                if script_basename == "run_classical_parity_benchmark.py"
                else explicit_root
            )
        elif script_basename == "report_runtime_v1_results.py" and experiment_dir is not None:
            output_root = experiment_dir / "paper_summary"
        elif output_file is not None:
            output_root = sidecar_root_for_output_file(output_file)
        else:
            output_root = (cwd / "outputs" / f"{script_basename.replace('.py', '')}_experiment").resolve()

        benchmark_family, method_family, adapter_name = _script_family(script_basename)
        benchmark_ref = benchmark_ref_from_parts(
            family=benchmark_family,
            scope=script_basename.replace(".py", ""),
            dataset_id=(
                _flag_value(argv, "--dataset-path")
                or _flag_value(argv, "--records")
                or _flag_value(argv, "--dataset")
                or ""
            ),
            name=benchmark_family,
            metadata={"script": script_basename},
        )
        roles = _inferred_role_refs(script_basename, argv)
        method_ref = method_ref_from_parts(
            family=method_family,
            variant="dry_run" if "--dry-run" in set(str(item) for item in argv) else "run",
            adapter=adapter_name,
            metadata=metadata_with_roles(
                {"legacy_script": script_basename},
                roles=roles,
                oracle=oracle_ref(
                    kind=(
                        "benchmark_labels"
                        if benchmark_family in {"longbench_v2", "manifesto_rile", "manifesto_lawstress"}
                        else "task_provenance"
                    ),
                    source=benchmark_family,
                ),
            ),
        )
        phase = (
            "dry_run"
            if "--dry-run" in set(str(item) for item in argv)
            else "generate"
            if script_basename.startswith("generate_")
            else "eval"
            if script_basename.startswith("eval_")
            else "run"
        )
        return ExperimentSpec.create(
            adapter_id=self.adapter_id,
            output_root=str(output_root),
            title=script_basename.replace(".py", ""),
            benchmark_refs=(benchmark_ref,),
            method_refs=(method_ref,),
            phases=default_phase_specs((phase,)),
            report_profiles=("runtime_eval_summary",),
            launch_command=command,
            resume_command=command,
            metadata={
                "legacy_script": script_basename,
                "inferred_adapter": adapter_name,
            },
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        candidates = {
            "experiment_manifest_json": output_root / "experiment_manifest.json",
            "experiment_status_json": output_root / "experiment_status.json",
            "artifacts_json": output_root / "artifacts.json",
            "results_jsonl": output_root / "results.jsonl",
            "calls_jsonl": output_root / "calls.jsonl",
        }
        return {key: str(path) for key, path in candidates.items() if path.exists()}


@register_method_adapter
class ReportOnlyAdapter:
    adapter_id = "report_only"
    aliases = ("report",)

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        output_root = _flag_value(command, "--output-root")
        if not output_root:
            raise ValueError("report_only adapter requires --output-root")
        output_root_path = (cwd / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root).resolve()
        return ExperimentSpec.create(
            adapter_id=self.adapter_id,
            output_root=str(output_root_path),
            title="report_only",
            phases=default_phase_specs(("report",)),
            report_profiles=("tradeoff", "publication_bundle", "runtime_eval_summary", "supervision_recovery"),
            launch_command=command,
            resume_command=command,
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        return {}
