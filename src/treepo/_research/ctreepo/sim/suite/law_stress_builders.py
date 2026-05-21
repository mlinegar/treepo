from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from treepo._research.ctreepo.sim.cli.sweep_markov_changepoint_ops_count import _iter_runs as _iter_markov_runs
from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    LAW_SET_ALL,
    LAW_SET_LEAF_AND_MERGE_PRESERVATION,
    LAW_SET_LEAF_PRESERVATION_ONLY,
    LAW_SET_MERGE_PRESERVATION_ONLY,
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    assert_public_contract_clean,
    canonical_law_set_id,
)
from treepo._research.ctreepo.sim.composite_objective import resolve_root_local_objective_weights
from treepo._research.ctreepo.sim.manifest import RunSpec, write_manifest_jsonl
from treepo._research.ctreepo.sim.suite.common import write_text
from treepo._research.ctreepo.sim.suite.law_stress_policy import (
    LDAMechanismPolicy,
    LDALawStressPolicy,
    LDASanityPolicy,
    LDATransitionMapPolicy,
    MarkovCapacityAppendixPolicy,
    MarkovCrossDgpPolicy,
    MarkovLawStressPolicy,
    MarkovMechanismPolicy,
    MarkovSanityPolicy,
    MarkovTransitionMapPolicy,
    MarkovWeightAblationPolicy,
    resolve_lda_law_stress_policy,
    resolve_markov_law_stress_policy,
)

LDA_CLI_SCRIPT = "scripts/run_leaf_local_mixture_utility_simulation.py"

_LDA_LEGACY_LAW_SET_MAP = {
    "root_only": (LAW_SET_ALL, 0.0, {}),
    "c1_only": (LAW_SET_LEAF_PRESERVATION_ONLY, 0.5, {}),
    "c3_only": (LAW_SET_MERGE_PRESERVATION_ONLY, 0.5, {}),
    "c1c3": (LAW_SET_LEAF_AND_MERGE_PRESERVATION, 0.5, {}),
    "c2_only": (LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY, 0.5, {}),
    "all_laws": (LAW_SET_ALL, 0.5, {}),
    "all_laws_plus_sched": (
        LAW_SET_ALL,
        0.5,
        {"auxiliary_schedule_diagnostics": True},
    ),
}

_LDA_LAW_SET_ACTIVE_LAWS = {
    LAW_SET_ALL: (
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_ON_RANGE_IDEMPOTENCE,
        LAW_ID_MERGE_PRESERVATION,
    ),
    LAW_SET_LEAF_PRESERVATION_ONLY: (LAW_ID_LEAF_PRESERVATION,),
    LAW_SET_MERGE_PRESERVATION_ONLY: (LAW_ID_MERGE_PRESERVATION,),
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY: (LAW_ID_ON_RANGE_IDEMPOTENCE,),
    LAW_SET_LEAF_AND_MERGE_PRESERVATION: (
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_MERGE_PRESERVATION,
    ),
}


def _lda_canonical_axis_fields(
    *,
    law_set_id: str,
    local_law_weight: float,
) -> Dict[str, object]:
    canonical_law_set = canonical_law_set_id(str(law_set_id), allow_aliases=False)
    resolved = resolve_root_local_objective_weights(
        local_law_weight=float(local_law_weight),
        active_laws=_LDA_LAW_SET_ACTIVE_LAWS.get(
            canonical_law_set,
            _LDA_LAW_SET_ACTIVE_LAWS[LAW_SET_ALL],
        ),
        objective_context="LDA law-stress builder",
    )
    return {
        "problem_id": "leaf_local_mixture_utility",
        "method_id": "tree_relevant_lda_local_law",
        "law_set_id": canonical_law_set,
        "root_share": float(resolved.root_share),
        "local_law_weight": float(resolved.local_law_weight),
        "local_law_component_weights": {
            str(k): float(v) for k, v in resolved.local_law_shares.items()
        },
    }


def _resolve_lda_public_law_axis(raw_value: str) -> tuple[str, float, Dict[str, object]]:
    key = str(raw_value or LAW_SET_ALL).strip().lower()
    law_set_id, local_law_weight, aux = _LDA_LEGACY_LAW_SET_MAP.get(
        key,
        (key, 0.5, {}),
    )
    return str(law_set_id), float(local_law_weight), dict(aux)


@dataclass(frozen=True)
class LawStressBuildResult:
    manifest_path: Path
    manifest: Dict[str, object]
    runs: List[RunSpec]


@dataclass(frozen=True)
class LDALawStressCommand:
    command: str
    json_path: Path
    csv_path: Path
    artifact_dir: Path
    config: Dict[str, object]


def _write_cmd_file(path: Path, commands: Sequence[str]) -> None:
    write_text(path, "\n".join(commands) + ("\n" if commands else ""))


def _markov_base_runs(
    *,
    python_bin: str,
    output_root: Path,
    n_regimes: int,
    fixed_leaf_tokens: int,
    train_docs: Iterable[int],
    val_docs: int,
    test_docs: int,
    audit_fractions: Iterable[float],
    law_packages: Iterable[str],
    exact_families: Iterable[str],
    state_dims: Iterable[int],
    hidden_dims: Iterable[int],
    root_weights: Iterable[float],
    data_seeds: Iterable[int],
    model_seeds: Iterable[int],
    n_epochs: int,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    suite_role: str,
) -> List[RunSpec]:
    return _iter_markov_runs(
        python_bin=str(python_bin),
        n_regimes=int(n_regimes),
        vocab_size=96,
        min_tokens=384,
        max_tokens=384,
        min_segments=12,
        max_segments=24,
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        train_docs=list(train_docs),
        val_docs=int(val_docs),
        test_docs=int(test_docs),
        audit_fractions=list(audit_fractions),
        c3_audit_strategies=["uniform"],
        c3_include_root=True,
        leaf_query_rates=[1.0],
        include_root_queries=[True],
        local_law_weights=[1.0],
        task_objective_weights=[],
        c1_relative_weights=[1.0],
        c2_relative_weights=[0.0],
        c3_relative_weights=[4.0],
        c2_weights=[0.0],
        root_weights=list(root_weights),
        schedule_consistency_weights=[0.0],
        law_packages=list(law_packages),
        exact_families=list(exact_families),
        guidance_override_modes=["reset"],
        eval_guidance_qs=[],
        eval_guidance_trials=0,
        eval_guidance_seed_offset=100_000,
        eval_guidance_include_root=True,
        include_rf_root_baseline=False,
        include_doc_level_baseline=False,
        include_doc_level_ridge_baseline=False,
        include_leaf_ridge_tree_baseline=False,
        include_leaf_endpoint_table_tree_baseline=False,
        include_leaf_dt_tree_baseline=False,
        include_leaf_knn_tree_baseline=False,
        include_leaf_rf_tree_baseline=False,
        rf_n_estimators=200,
        rf_max_depth=16,
        rf_min_samples_leaf=5,
        doc_level_ridge_alpha=1.0,
        leaf_knn_neighbors=8,
        include_sampled_leaf_pool_ridge_baseline=False,
        include_sampled_leaf_pool_rf_baseline=False,
        sampled_leaf_pool_leaf_counts=[],
        sampled_leaf_pool_seed_offset=300_000,
        data_seeds=list(data_seeds),
        seeds=list(model_seeds),
        output_root=output_root,
        model_families=["neural"],
        feature_modes=["full"],
        state_dims=list(state_dims),
        hidden_dims=list(hidden_dims),
        hidden_dim_multiplier=None,
        hidden_dim_min=64,
        n_epochs=int(n_epochs),
        device=str(device),
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        violation_tau=0.0,
        torch_threads=int(torch_threads),
        skip_existing=True,
        suite_role=str(suite_role),
    )


def _build_markov_sanity_suite(
    *,
    python_bin: str,
    suite_root: Path,
    policy: MarkovSanityPolicy,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> Tuple[List[RunSpec], List[RunSpec]]:
    learned: List[RunSpec] = []
    exact: List[RunSpec] = []
    for n_regimes in policy.n_regimes:
        for leaf in policy.fixed_leaf_tokens:
            learned.extend(
                _markov_base_runs(
                    python_bin=python_bin,
                    output_root=suite_root / "learned" / f"nreg_{n_regimes}" / f"leaf_{leaf}",
                    n_regimes=n_regimes,
                    fixed_leaf_tokens=leaf,
                    train_docs=policy.train_docs,
                    val_docs=int(policy.val_docs),
                    test_docs=int(policy.test_docs),
                    audit_fractions=[1.0],
                    law_packages=policy.learned_law_packages,
                    exact_families=[],
                    state_dims=policy.state_dims,
                    hidden_dims=policy.hidden_dims,
                    root_weights=policy.root_weights,
                    data_seeds=policy.data_seeds,
                    model_seeds=policy.model_seeds,
                    n_epochs=int(policy.n_epochs),
                    device=device,
                    cuda_device=cuda_device,
                    torch_threads=torch_threads,
                    suite_role="positive_controls",
                )
            )
            exact.extend(
                _markov_base_runs(
                    python_bin=python_bin,
                    output_root=suite_root / "exact" / f"nreg_{n_regimes}" / f"leaf_{leaf}",
                    n_regimes=n_regimes,
                    fixed_leaf_tokens=leaf,
                    train_docs=policy.train_docs,
                    val_docs=int(policy.val_docs),
                    test_docs=int(policy.test_docs),
                    audit_fractions=[1.0],
                    law_packages=[],
                    exact_families=policy.exact_families,
                    state_dims=policy.state_dims,
                    hidden_dims=policy.hidden_dims,
                    root_weights=policy.root_weights,
                    data_seeds=policy.data_seeds,
                    model_seeds=[0],
                    n_epochs=1,
                    device=device,
                    cuda_device=cuda_device,
                    torch_threads=torch_threads,
                    suite_role="failure_modes",
                )
            )
    return learned, exact


def _build_markov_transition_map_suite(
    *,
    python_bin: str,
    suite_root: Path,
    policy: MarkovTransitionMapPolicy,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[RunSpec]:
    return _markov_base_runs(
        python_bin=python_bin,
        output_root=suite_root,
        n_regimes=int(policy.n_regimes),
        fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
        train_docs=policy.train_docs,
        val_docs=int(policy.val_docs),
        test_docs=int(policy.test_docs),
        audit_fractions=policy.audit_fractions,
        law_packages=policy.law_packages,
        exact_families=[],
        state_dims=policy.state_dims,
        hidden_dims=policy.hidden_dims,
        root_weights=policy.root_weights,
        data_seeds=policy.data_seeds,
        model_seeds=policy.model_seeds,
        n_epochs=int(policy.n_epochs),
        device=device,
        cuda_device=cuda_device,
        torch_threads=torch_threads,
        suite_role="support_scaling",
    )


def _choose_boundary_cells(rows: Sequence[dict], *, limit: int) -> List[dict]:
    main_rows = [
        row
        for row in rows
        if str(row.get("law_package", "")) in {"all_laws_plus_sched", "all_laws"}
    ]
    ranked = sorted(
        main_rows,
        key=lambda row: (
            abs(float(row.get("val_bundle_full_success_rate", row.get("bundle_full_success_rate", 0.0))) - 0.5),
            abs(float(row.get("val_bundle_margin_mean", row.get("bundle_margin_mean", 0.0)))),
            int(row.get("train_docs", 0)),
            float(row.get("audit_fraction", 0.0)),
        ),
    )
    return ranked[: int(limit)]


def _build_markov_mechanism_suite(
    *,
    python_bin: str,
    suite_root: Path,
    transition_summary: Path,
    policy: MarkovMechanismPolicy,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> Tuple[List[RunSpec], List[dict]]:
    payload = json.loads(transition_summary.read_text(encoding="utf-8"))
    rows = list(payload.get("aggregated_rows") or [])
    chosen = _choose_boundary_cells(rows, limit=int(policy.selection_limit))
    runs: List[RunSpec] = []
    selected: List[dict] = []
    for idx, row in enumerate(chosen):
        selected.append(
            {
                "index": int(idx),
                "n_regimes": int(row["n_regimes"]),
                "fixed_leaf_tokens": int(row["fixed_leaf_tokens"]),
                "train_docs": int(row["train_docs"]),
                "val_docs": int(row["val_docs"]),
                "test_docs": int(row["test_docs"]),
                "audit_fraction": float(row["audit_fraction"]),
                "state_dim": int(row["state_dim"]),
                "hidden_dim": int(row["hidden_dim"]),
                "n_epochs": int(row["n_epochs"]),
            }
        )
        runs.extend(
            _markov_base_runs(
                python_bin=python_bin,
                output_root=suite_root / f"cell_{idx}",
                n_regimes=int(row["n_regimes"]),
                fixed_leaf_tokens=int(row["fixed_leaf_tokens"]),
                train_docs=[int(row["train_docs"])],
                val_docs=int(row["val_docs"]),
                test_docs=int(row["test_docs"]),
                audit_fractions=[float(row["audit_fraction"])],
                law_packages=policy.law_packages,
                exact_families=[],
                state_dims=[64],
                hidden_dims=[256],
                root_weights=policy.root_weights,
                data_seeds=policy.data_seeds,
                model_seeds=policy.model_seeds,
                n_epochs=int(row["n_epochs"]),
                device=device,
                cuda_device=cuda_device,
                torch_threads=torch_threads,
                suite_role="relevance_mediation",
            )
        )
    return runs, selected


def _build_markov_capacity_appendix_suite(
    *,
    python_bin: str,
    suite_root: Path,
    policy: MarkovCapacityAppendixPolicy,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for state_dim, hidden_dim in policy.caps:
        runs.extend(
            _markov_base_runs(
                python_bin=python_bin,
                output_root=suite_root / f"sd_{state_dim}_hd_{hidden_dim}",
                n_regimes=int(policy.n_regimes),
                fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
                train_docs=policy.train_docs,
                val_docs=int(policy.val_docs),
                test_docs=int(policy.test_docs),
                audit_fractions=policy.audit_fractions,
                law_packages=policy.law_packages,
                exact_families=[],
                state_dims=[state_dim],
                hidden_dims=[hidden_dim],
                root_weights=policy.root_weights,
                data_seeds=policy.data_seeds,
                model_seeds=policy.model_seeds,
                n_epochs=int(policy.n_epochs),
                device=device,
                cuda_device=cuda_device,
                torch_threads=torch_threads,
                suite_role="hardness",
            )
        )
    return runs


def _build_markov_cross_dgp_suite(
    *,
    python_bin: str,
    suite_root: Path,
    policy: MarkovCrossDgpPolicy,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for state_dim, hidden_dim in policy.caps:
        runs.extend(
            _markov_base_runs(
                python_bin=python_bin,
                output_root=suite_root / f"sd_{state_dim}_hd_{hidden_dim}",
                n_regimes=int(policy.n_regimes),
                fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
                train_docs=policy.train_docs,
                val_docs=int(policy.val_docs),
                test_docs=int(policy.test_docs),
                audit_fractions=policy.audit_fractions,
                law_packages=policy.law_packages,
                exact_families=[],
                state_dims=[state_dim],
                hidden_dims=[hidden_dim],
                root_weights=policy.root_weights,
                data_seeds=policy.data_seeds,
                model_seeds=policy.model_seeds,
                n_epochs=int(policy.n_epochs),
                device=device,
                cuda_device=cuda_device,
                torch_threads=torch_threads,
                suite_role="cross_dgp_matched",
            )
        )
    return runs


def _build_markov_weight_ablation_suite(
    *,
    python_bin: str,
    suite_root: Path,
    policy: MarkovWeightAblationPolicy,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for state_dim, hidden_dim in policy.caps:
        runs.extend(
            _markov_base_runs(
                python_bin=python_bin,
                output_root=suite_root / f"sd_{state_dim}_hd_{hidden_dim}",
                n_regimes=int(policy.n_regimes),
                fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
                train_docs=policy.train_docs,
                val_docs=int(policy.val_docs),
                test_docs=int(policy.test_docs),
                audit_fractions=policy.audit_fractions,
                law_packages=policy.baseline_law_packages,
                exact_families=[],
                state_dims=[state_dim],
                hidden_dims=[hidden_dim],
                root_weights=[1.0],
                data_seeds=policy.data_seeds,
                model_seeds=policy.model_seeds,
                n_epochs=int(policy.n_epochs),
                device=device,
                cuda_device=cuda_device,
                torch_threads=torch_threads,
                suite_role="weight_ablation_baseline",
            )
        )
        for weight_profile in policy.weight_profiles:
            runs.extend(
                _iter_markov_runs(
                    python_bin=str(python_bin),
                    n_regimes=int(policy.n_regimes),
                    vocab_size=96,
                    min_tokens=384,
                    max_tokens=384,
                    min_segments=12,
                    max_segments=24,
                    fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
                    train_docs=policy.train_docs,
                    val_docs=int(policy.val_docs),
                    test_docs=int(policy.test_docs),
                    audit_fractions=policy.audit_fractions,
                    c3_audit_strategies=["uniform"],
                    c3_include_root=True,
                    leaf_query_rates=[1.0],
                    include_root_queries=[True],
                    local_law_weights=[1.0],
                    task_objective_weights=[],
                    c1_relative_weights=[float(weight_profile.c1_relative_weight)],
                    c2_relative_weights=[float(weight_profile.c2_relative_weight)],
                    c3_relative_weights=[float(weight_profile.c3_relative_weight)],
                    c2_weights=[0.0],
                    root_weights=[1.0],
                    schedule_consistency_weights=[0.0],
                    law_packages=[],
                    exact_families=[],
                    guidance_override_modes=["reset"],
                    eval_guidance_qs=[],
                    eval_guidance_trials=0,
                    eval_guidance_seed_offset=100_000,
                    eval_guidance_include_root=True,
                    include_rf_root_baseline=False,
                    include_doc_level_baseline=False,
                    rf_n_estimators=200,
                    rf_max_depth=16,
                    rf_min_samples_leaf=5,
                    data_seeds=policy.data_seeds,
                    seeds=policy.model_seeds,
                    output_root=suite_root / f"sd_{state_dim}_hd_{hidden_dim}",
                    model_families=["neural"],
                    feature_modes=["full"],
                    state_dims=[state_dim],
                    hidden_dims=[hidden_dim],
                    hidden_dim_multiplier=None,
                    hidden_dim_min=64,
                    n_epochs=int(policy.n_epochs),
                    device=str(device),
                    cuda_device=int(cuda_device) if cuda_device is not None else None,
                    violation_tau=0.0,
                    torch_threads=int(torch_threads),
                    skip_existing=True,
                    suite_role="weight_ablation",
                )
            )
    return runs


def build_markov_law_stress_suites(
    *,
    suite: str,
    output_root: Path,
    cmd_dir: Path,
    python_bin: str,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    transition_summary: Path | None,
    smoke: bool,
) -> LawStressBuildResult:
    output_root = output_root.resolve()
    cmd_dir = cmd_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cmd_dir.mkdir(parents=True, exist_ok=True)

    policy = resolve_markov_law_stress_policy(smoke=bool(smoke))
    manifest: Dict[str, object] = {
        "output_root": str(output_root),
        "suite": str(suite),
        "smoke": bool(smoke),
        "policy": policy.to_dict(),
    }
    all_runs: List[RunSpec] = []

    if suite in {"sanity_suite", "all"}:
        learned, exact = _build_markov_sanity_suite(
            python_bin=python_bin,
            suite_root=output_root / "sanity_suite" / "markov_changepoint_ops_count",
            policy=policy.sanity,
            device=device,
            cuda_device=cuda_device,
            torch_threads=torch_threads,
        )
        learned_cmd = cmd_dir / "sanity_suite_learned_cmds.txt"
        exact_cmd = cmd_dir / "sanity_suite_exact_cmds.txt"
        _write_cmd_file(learned_cmd, [run.command for run in learned])
        _write_cmd_file(exact_cmd, [run.command for run in exact])
        all_runs.extend(learned)
        all_runs.extend(exact)
        manifest["sanity_suite"] = {
            "learned_n_commands": len(learned),
            "learned_cmd_file": str(learned_cmd),
            "exact_n_commands": len(exact),
            "exact_cmd_file": str(exact_cmd),
        }

    if suite in {"transition_map_suite", "all"}:
        runs = _build_markov_transition_map_suite(
            python_bin=python_bin,
            suite_root=output_root / "transition_map_suite" / "markov_changepoint_ops_count",
            policy=policy.transition_map,
            device=device,
            cuda_device=cuda_device,
            torch_threads=torch_threads,
        )
        cmd_file = cmd_dir / "transition_map_suite_cmds.txt"
        _write_cmd_file(cmd_file, [run.command for run in runs])
        all_runs.extend(runs)
        manifest["transition_map_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_file)}

    if suite in {"mechanism_suite", "all"}:
        if transition_summary is None:
            raise SystemExit("--transition-summary is required for mechanism_suite")
        runs, selected = _build_markov_mechanism_suite(
            python_bin=python_bin,
            suite_root=output_root / "mechanism_suite" / "markov_changepoint_ops_count",
            transition_summary=transition_summary,
            policy=policy.mechanism,
            device=device,
            cuda_device=cuda_device,
            torch_threads=torch_threads,
        )
        cmd_file = cmd_dir / "mechanism_suite_cmds.txt"
        _write_cmd_file(cmd_file, [run.command for run in runs])
        all_runs.extend(runs)
        manifest["mechanism_suite"] = {
            "n_commands": len(runs),
            "cmd_file": str(cmd_file),
            "selected_cells": selected,
        }

    if suite in {"capacity_appendix_suite", "all"}:
        runs = _build_markov_capacity_appendix_suite(
            python_bin=python_bin,
            suite_root=output_root / "capacity_appendix_suite" / "markov_changepoint_ops_count",
            policy=policy.capacity_appendix,
            device=device,
            cuda_device=cuda_device,
            torch_threads=torch_threads,
        )
        cmd_file = cmd_dir / "capacity_appendix_suite_cmds.txt"
        _write_cmd_file(cmd_file, [run.command for run in runs])
        all_runs.extend(runs)
        manifest["capacity_appendix_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_file)}

    if suite in {"cross_dgp_suite"}:
        runs = _build_markov_cross_dgp_suite(
            python_bin=python_bin,
            suite_root=output_root / "cross_dgp_suite" / "markov_changepoint_ops_count",
            policy=policy.cross_dgp,
            device=device,
            cuda_device=cuda_device,
            torch_threads=torch_threads,
        )
        cmd_file = cmd_dir / "cross_dgp_suite_cmds.txt"
        _write_cmd_file(cmd_file, [run.command for run in runs])
        all_runs.extend(runs)
        manifest["cross_dgp_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_file)}

    if suite in {"weight_ablation_suite"}:
        runs = _build_markov_weight_ablation_suite(
            python_bin=python_bin,
            suite_root=output_root / "weight_ablation_suite" / "markov_changepoint_ops_count",
            policy=policy.weight_ablation,
            device=device,
            cuda_device=cuda_device,
            torch_threads=torch_threads,
        )
        cmd_file = cmd_dir / "weight_ablation_suite_cmds.txt"
        _write_cmd_file(cmd_file, [run.command for run in runs])
        all_runs.extend(runs)
        manifest["weight_ablation_suite"] = {"n_commands": len(runs), "cmd_file": str(cmd_file)}

    manifest_jsonl_path = cmd_dir / "markov_law_stress_suite_manifest.jsonl"
    write_manifest_jsonl(manifest_jsonl_path, all_runs)
    manifest["runspec_manifest"] = str(manifest_jsonl_path)
    manifest_path = cmd_dir / "markov_law_stress_suite_manifest.json"
    write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return LawStressBuildResult(manifest_path=manifest_path, manifest=manifest, runs=all_runs)


def _build_lda_cmd(
    *,
    python_bin: str,
    output_root: Path,
    tau: float,
    quadratic_utility_weight: float,
    train_docs: int,
    test_docs: int,
    law_set_id: str,
    exact_family: str,
    local_law_mode: str,
    law_leaf_query_rate: float,
    law_internal_query_rate: float,
    analysis_partition_mode: str,
    seed: int,
    suite_role: str,
) -> LDALawStressCommand:
    resolved_law_set_id, local_law_weight, auxiliary = _resolve_lda_public_law_axis(law_set_id)
    axis_fields = _lda_canonical_axis_fields(
        law_set_id=str(resolved_law_set_id),
        local_law_weight=float(local_law_weight),
    )
    slug = f"tau{tau:g}_qweight{quadratic_utility_weight:g}_law_{resolved_law_set_id}_llw_{local_law_weight:g}_mode_{analysis_partition_mode}_s{seed}"
    if exact_family:
        slug = f"fam_{exact_family}_{slug}"
    json_path = output_root / "results" / suite_role / f"{slug}.json"
    csv_path = output_root / "results" / suite_role / f"{slug}.csv"
    artifact_dir = output_root / "results" / suite_role / f"{slug}_artifacts"
    parts = [
        python_bin,
        LDA_CLI_SCRIPT,
        f"--local-mixture-concentration {tau}",
        f"--quadratic-utility-weight {quadratic_utility_weight}",
        f"--train-docs {train_docs}",
        f"--test-docs {test_docs}",
        f"--local-law-mode {local_law_mode}",
        f"--law-set-id {resolved_law_set_id}",
        f"--local-law-weight {local_law_weight}",
        f"--law-leaf-query-rate {law_leaf_query_rate}",
        f"--law-internal-query-rate {law_internal_query_rate}",
        f"--analysis-partition-mode {analysis_partition_mode}",
        f"--seed {seed}",
        f"--json-summary {json_path}",
        f"--csv-summary {csv_path}",
        f"--artifact-dir {artifact_dir}",
    ]
    if exact_family:
        parts.append(f"--exact-family {exact_family}")
    return LDALawStressCommand(
        command=" ".join(parts),
        json_path=json_path,
        csv_path=csv_path,
        artifact_dir=artifact_dir,
        config={
            **axis_fields,
            "tau": float(tau),
            "quadratic_utility_weight": float(quadratic_utility_weight),
            "train_docs": int(train_docs),
            "test_docs": int(test_docs),
            "auxiliary_diagnostics": dict(auxiliary),
            "exact_family": str(exact_family),
            "local_law_mode": str(local_law_mode),
            "law_leaf_query_rate": float(law_leaf_query_rate),
            "law_internal_query_rate": float(law_internal_query_rate),
            "analysis_partition_mode": str(analysis_partition_mode),
            "seed": int(seed),
            "suite_role": str(suite_role),
        },
    )


def _append_lda_if_needed(
    commands: List[LDALawStressCommand],
    item: LDALawStressCommand,
    *,
    skip_existing: bool,
) -> None:
    if bool(skip_existing) and item.json_path.exists() and item.csv_path.exists():
        return
    commands.append(item)


def _build_lda_sanity_suite(
    *,
    python_bin: str,
    suite_root: Path,
    policy: LDASanityPolicy,
    skip_existing: bool,
) -> List[LDALawStressCommand]:
    learned_cmds: List[LDALawStressCommand] = []
    for tau, qweight, pkg, seed in itertools.product(
        policy.taus,
        policy.quadratic_utility_weights,
        policy.learned_law_set_ids,
        policy.seeds,
    ):
        _append_lda_if_needed(
            learned_cmds,
            _build_lda_cmd(
                python_bin=python_bin,
                output_root=suite_root,
                tau=tau,
                quadratic_utility_weight=qweight,
                train_docs=int(policy.train_docs),
                test_docs=int(policy.test_docs),
                law_set_id=pkg,
                exact_family="",
                local_law_mode="diagnostics_and_learned",
                law_leaf_query_rate=float(policy.law_leaf_query_rate),
                law_internal_query_rate=float(policy.law_internal_query_rate),
                analysis_partition_mode=str(policy.analysis_partition_mode),
                seed=seed,
                suite_role="sanity_learned",
            ),
            skip_existing=skip_existing,
        )

    exact_cmds: List[LDALawStressCommand] = []
    for tau, qweight, fam, seed in itertools.product(
        policy.taus,
        policy.quadratic_utility_weights,
        policy.exact_families,
        policy.seeds,
    ):
        _append_lda_if_needed(
            exact_cmds,
            _build_lda_cmd(
                python_bin=python_bin,
                output_root=suite_root,
                tau=tau,
                quadratic_utility_weight=qweight,
                train_docs=int(policy.train_docs),
                test_docs=int(policy.test_docs),
                law_set_id=LAW_SET_ALL,
                exact_family=fam,
                local_law_mode="diagnostics",
                law_leaf_query_rate=float(policy.law_leaf_query_rate),
                law_internal_query_rate=float(policy.law_internal_query_rate),
                analysis_partition_mode=str(policy.analysis_partition_mode),
                seed=seed,
                suite_role="sanity_exact",
            ),
            skip_existing=skip_existing,
        )

    return learned_cmds + exact_cmds


def _build_lda_transition_map_suite(
    *,
    python_bin: str,
    suite_root: Path,
    policy: LDATransitionMapPolicy,
    skip_existing: bool,
) -> List[LDALawStressCommand]:
    commands: List[LDALawStressCommand] = []
    for tau, qweight, pkg, seed in itertools.product(
        policy.taus,
        policy.quadratic_utility_weights,
        policy.law_set_ids,
        policy.seeds,
    ):
        _append_lda_if_needed(
            commands,
            _build_lda_cmd(
                python_bin=python_bin,
                output_root=suite_root,
                tau=tau,
                quadratic_utility_weight=qweight,
                train_docs=int(policy.train_docs),
                test_docs=int(policy.test_docs),
                law_set_id=pkg,
                exact_family="",
                local_law_mode="diagnostics_and_learned",
                law_leaf_query_rate=float(policy.law_leaf_query_rate),
                law_internal_query_rate=float(policy.law_internal_query_rate),
                analysis_partition_mode=str(policy.analysis_partition_mode),
                seed=seed,
                suite_role="transition_map",
            ),
            skip_existing=skip_existing,
        )
    return commands


def _build_lda_mechanism_suite(
    *,
    python_bin: str,
    suite_root: Path,
    policy: LDAMechanismPolicy,
    skip_existing: bool,
) -> List[LDALawStressCommand]:
    commands: List[LDALawStressCommand] = []
    for tau, qweight, mode, pkg, seed in itertools.product(
        policy.taus,
        policy.quadratic_utility_weights,
        policy.analysis_partition_modes,
        policy.law_set_ids,
        policy.seeds,
    ):
        _append_lda_if_needed(
            commands,
            _build_lda_cmd(
                python_bin=python_bin,
                output_root=suite_root,
                tau=tau,
                quadratic_utility_weight=qweight,
                train_docs=int(policy.train_docs),
                test_docs=int(policy.test_docs),
                law_set_id=pkg,
                exact_family="",
                local_law_mode="diagnostics_and_learned",
                law_leaf_query_rate=float(policy.law_leaf_query_rate),
                law_internal_query_rate=float(policy.law_internal_query_rate),
                analysis_partition_mode=mode,
                seed=seed,
                suite_role="mechanism",
            ),
            skip_existing=skip_existing,
        )
    return commands


def _lda_runspecs(commands: Sequence[LDALawStressCommand]) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for item in commands:
        config = dict(item.config)
        assert_public_contract_clean(config, surface="LDA law-stress run config")
        runs.append(
            RunSpec.create(
                family="leaf_local_mixture_utility",
                config=config,
                outputs={
                    "json_summary": str(item.json_path),
                    "csv_summary": str(item.csv_path),
                    "artifact_dir": str(item.artifact_dir),
                },
                command=str(item.command),
            )
        )
    return runs


def build_lda_law_stress_suites(
    *,
    suite: str,
    output_root: Path,
    cmd_dir: Path,
    python_bin: str,
    skip_existing: bool,
    smoke: bool,
) -> LawStressBuildResult:
    output_root = output_root.resolve()
    cmd_dir = cmd_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cmd_dir.mkdir(parents=True, exist_ok=True)

    policy = resolve_lda_law_stress_policy(smoke=bool(smoke))
    suites_to_build = (
        ["sanity_suite", "transition_map_suite", "mechanism_suite"]
        if suite == "all"
        else [str(suite)]
    )
    meta: Dict[str, object] = {"suites": {}, "policy": policy.to_dict()}
    all_runs: List[RunSpec] = []

    for suite_name in suites_to_build:
        suite_root = output_root / suite_name
        if suite_name == "sanity_suite":
            commands = _build_lda_sanity_suite(
                python_bin=python_bin,
                suite_root=suite_root,
                policy=policy.sanity,
                skip_existing=bool(skip_existing),
            )
        elif suite_name == "transition_map_suite":
            commands = _build_lda_transition_map_suite(
                python_bin=python_bin,
                suite_root=suite_root,
                policy=policy.transition_map,
                skip_existing=bool(skip_existing),
            )
        elif suite_name == "mechanism_suite":
            commands = _build_lda_mechanism_suite(
                python_bin=python_bin,
                suite_root=suite_root,
                policy=policy.mechanism,
                skip_existing=bool(skip_existing),
            )
        else:
            raise ValueError(f"Unknown suite: {suite_name}")

        cmd_file = cmd_dir / f"lda_law_stress_{suite_name}_cmds.txt"
        _write_cmd_file(cmd_file, [item.command for item in commands])
        suite_runs = _lda_runspecs(commands)
        all_runs.extend(suite_runs)
        meta["suites"][suite_name] = {"n_commands": len(commands), "cmd_file": str(cmd_file)}  # type: ignore[index]

    manifest_path = cmd_dir / "lda_law_stress_meta.json"
    runspec_manifest = cmd_dir / "lda_law_stress_manifest.jsonl"
    write_manifest_jsonl(runspec_manifest, all_runs)
    meta["runspec_manifest"] = str(runspec_manifest)
    write_text(manifest_path, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return LawStressBuildResult(manifest_path=manifest_path, manifest=meta, runs=all_runs)


__all__ = [
    "LawStressBuildResult",
    "build_lda_law_stress_suites",
    "build_markov_law_stress_suites",
]
