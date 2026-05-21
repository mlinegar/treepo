#!/usr/bin/env python3
"""Build (and optionally execute) xargs-friendly sweeps for Markov changepoint OPS-count."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    VALID_GENERATOR_PROFILES,
)
from treepo._research.ctreepo.sim.manifest import RunSpec, write_manifest_jsonl
from treepo._research.ctreepo.sim.runner import run_commands


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if x:
            out.append(x)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _parse_bools(text: str) -> List[bool]:
    out: List[bool] = []
    for raw in _parse_items(text):
        x = raw.strip().lower()
        if x in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif x in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"could not parse boolean: {raw!r}")
    return out


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _cmd_prefix(python_bin: str) -> str:
    return f"{python_bin} -m src.ctreepo.cli sim run markov-ops-count"


def _iter_runs(
    *,
    python_bin: str,
    n_regimes: int,
    vocab_size: int,
    generator_profile: str = "piecewise_markov",
    min_tokens: int,
    max_tokens: int,
    min_segments: int,
    max_segments: int,
    fixed_leaf_tokens: int,
    train_docs: Iterable[int],
    val_docs: int,
    test_docs: int,
    audit_fractions: Iterable[float],
    c3_audit_strategies: Iterable[str],
    c3_include_root: bool,
    leaf_query_rates: Iterable[float],
    include_root_queries: Iterable[bool],
    local_law_weights: Iterable[float],
    task_objective_weights: Iterable[float],
    c1_relative_weights: Iterable[float],
    c3_relative_weights: Iterable[float],
    root_weights: Iterable[float],
    schedule_consistency_weights: Iterable[float],
    guidance_override_modes: Iterable[str],
    eval_guidance_qs: Iterable[float],
    eval_guidance_trials: int,
    eval_guidance_seed_offset: int,
    eval_guidance_include_root: bool,
    include_rf_root_baseline: bool,
    include_doc_level_baseline: bool,
    include_doc_level_ridge_baseline: bool,
    include_leaf_ridge_tree_baseline: bool,
    include_leaf_endpoint_table_tree_baseline: bool,
    include_leaf_dt_tree_baseline: bool,
    include_leaf_knn_tree_baseline: bool,
    include_leaf_rf_tree_baseline: bool,
    rf_n_estimators: int,
    rf_max_depth: int,
    rf_min_samples_leaf: int,
    doc_level_ridge_alpha: float,
    leaf_knn_neighbors: int,
    include_sampled_leaf_pool_ridge_baseline: bool,
    include_sampled_leaf_pool_rf_baseline: bool,
    sampled_leaf_pool_leaf_counts: Iterable[int],
    sampled_leaf_pool_seed_offset: int,
    data_seeds: Iterable[int],
    seeds: Iterable[int],
    output_root: Path,
    model_families: Iterable[str],
    feature_modes: Iterable[str],
    state_dims: Iterable[int],
    hidden_dims: Iterable[int],
    hidden_dim_multiplier: float | None,
    hidden_dim_min: int,
    n_epochs: int,
    device: str,
    cuda_device: int | None,
    violation_tau: float,
    torch_threads: int,
    skip_existing: bool,
    suite_role: str = "",
    law_packages: Iterable[str] = (),
    exact_families: Iterable[str] = (),
    c2_relative_weights: Iterable[float] = (),
    c2_weights: Iterable[float] = (),
) -> List[RunSpec]:
    runs: List[RunSpec] = []

    if int(n_regimes) <= 0:
        raise ValueError("n_regimes must be positive")
    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    generator_profile_value = str(generator_profile).strip().lower() or "piecewise_markov"
    if generator_profile_value not in VALID_GENERATOR_PROFILES:
        raise ValueError(
            f"generator_profile must be one of {VALID_GENERATOR_PROFILES}, got {generator_profile!r}"
        )
    if int(min_tokens) <= 0 or int(max_tokens) <= 0:
        raise ValueError("min_tokens/max_tokens must be positive")
    if int(min_segments) <= 0 or int(max_segments) <= 0:
        raise ValueError("min_segments/max_segments must be positive")
    if int(fixed_leaf_tokens) <= 0:
        raise ValueError("fixed_leaf_tokens must be positive")

    include_root_values = list(include_root_queries) or [True]
    local_law_values = [float(x) for x in local_law_weights] or [0.0]
    task_objective_values = [float(x) for x in task_objective_weights]
    c1_relative_values = [float(x) for x in c1_relative_weights] or [1.0]
    c2_relative_values = [float(x) for x in c2_relative_weights] or [1.0]
    c3_relative_values = [float(x) for x in c3_relative_weights] or [1.0]
    c2_weight_values = [float(x) for x in c2_weights] or [0.0]
    law_package_values = [str(x).strip() for x in law_packages if str(x).strip()] or [""]
    exact_family_values = [str(x).strip() for x in exact_families if str(x).strip()] or [""]
    family_values = [str(x).strip() for x in model_families if str(x).strip()] or ["neural"]
    guidance_override_values = [
        str(x).strip().lower() for x in guidance_override_modes if str(x).strip()
    ] or ["reset"]
    for mode in guidance_override_values:
        if mode not in {"reset", "adjust"}:
            raise ValueError("guidance_override_modes must be a subset of {'reset','adjust'}")
    guidance_qs_list = [float(q) for q in eval_guidance_qs]
    guidance_qs_text = ",".join(f"{float(q):.6g}" for q in guidance_qs_list)
    sampled_leaf_pool_counts = sorted(
        {
            int(x)
            for x in sampled_leaf_pool_leaf_counts
            if int(x) > 0
        }
    )
    data_seed_values_raw = [int(x) for x in data_seeds]
    data_seed_values: List[int | None] = data_seed_values_raw if data_seed_values_raw else [None]

    feature_mode_values = [str(x).strip() for x in feature_modes if str(x).strip()] or ["full"]
    for fm in feature_mode_values:
        if fm not in {"full", "no_endpoints", "token_full", "token_bow"}:
            raise ValueError(
                "feature_modes must be a subset of {'full','no_endpoints','token_full','token_bow'}"
            )

    state_dim_values = [int(x) for x in state_dims] or [32]
    hidden_dim_values = [int(x) for x in hidden_dims] or [128]
    hdm: float | None = None
    if hidden_dim_multiplier is not None and float(hidden_dim_multiplier) > 0.0:
        hdm = float(hidden_dim_multiplier)
    hd_min = int(hidden_dim_min)

    prefix = _cmd_prefix(str(python_bin))
    val_component = f"/val_{int(val_docs)}" if int(val_docs) > 0 else ""

    for (
        fam,
        td,
        frac,
        c3_strat,
        lqr,
        rootq,
        llw,
        task_weight,
        c1_rel,
        c2_rel,
        c3_rel,
        c2_w,
        package,
        exact_family,
        rw,
        scw,
        gov_mode,
        fm,
        sd,
        data_seed,
        seed,
    ) in product(
        family_values,
        train_docs,
        audit_fractions,
        c3_audit_strategies,
        leaf_query_rates,
        include_root_values,
        local_law_values,
        ([None] + task_objective_values),
        c1_relative_values,
        c2_relative_values,
        c3_relative_values,
        c2_weight_values,
        law_package_values,
        exact_family_values,
        root_weights,
        schedule_consistency_weights,
        guidance_override_values,
        feature_mode_values,
        state_dim_values,
        data_seed_values,
        seeds,
    ):
        gov_component = ""
        if len(guidance_override_values) > 1 or str(gov_mode) != "reset":
            gov_component = f"/gov_{str(gov_mode)}"
        rf_component = "/rfroot_1" if bool(include_rf_root_baseline) else ""
        doc_component = "/docbase_1" if bool(include_doc_level_baseline) else ""
        ridge_component = "/ridge_1" if bool(include_doc_level_ridge_baseline) else ""
        leaf_ridge_component = "/leafridge_1" if bool(include_leaf_ridge_tree_baseline) else ""
        leaf_endpoint_table_component = (
            "/leafendtable_1" if bool(include_leaf_endpoint_table_tree_baseline) else ""
        )
        leaf_dt_component = "/leafdt_1" if bool(include_leaf_dt_tree_baseline) else ""
        leaf_knn_component = (
            f"/leafknn_k{int(leaf_knn_neighbors)}" if bool(include_leaf_knn_tree_baseline) else ""
        )
        leaf_rf_component = "/leafrf_1" if bool(include_leaf_rf_tree_baseline) else ""
        sampled_pool_component = ""
        if bool(include_sampled_leaf_pool_ridge_baseline) or bool(
            include_sampled_leaf_pool_rf_baseline
        ):
            pool_bits: List[str] = []
            if bool(include_sampled_leaf_pool_ridge_baseline):
                pool_bits.append("ridge")
            if bool(include_sampled_leaf_pool_rf_baseline):
                pool_bits.append("rf")
            budget_text = "-".join(str(int(x)) for x in sampled_leaf_pool_counts) or "none"
            sampled_pool_component = f"/spl_{'-'.join(pool_bits)}_{budget_text}"
        fm_component = ""
        if len(feature_mode_values) > 1 or str(fm) != "full":
            fm_component = f"/fm_{str(fm)}"
        generator_component = ""
        if generator_profile_value != "piecewise_markov":
            generator_component = f"/gp_{generator_profile_value}"
        if int(sd) <= 0:
            raise ValueError("state_dims must be positive")
        hd_candidates = (
            [max(hd_min, int(round(float(hdm) * float(sd))))]
            if hdm is not None
            else list(hidden_dim_values)
        )
        for hd in hd_candidates:
            if int(hd) <= 0:
                raise ValueError("hidden_dim must be positive")
            law_mix_component = ""
            if (
                len(c1_relative_values) > 1
                or len(c2_relative_values) > 1
                or len(c3_relative_values) > 1
                or abs(float(c1_rel) - 1.0) > 1e-12
                or abs(float(c2_rel) - 1.0) > 1e-12
                or abs(float(c3_rel) - 1.0) > 1e-12
            ):
                law_mix_component = (
                    f"/c1r_{_fmt_float(float(c1_rel))}"
                    f"/c2r_{_fmt_float(float(c2_rel))}"
                    f"/c3r_{_fmt_float(float(c3_rel))}"
                )
            task_component = ""
            if task_weight is not None:
                task_component = f"/taskw_{_fmt_float(float(task_weight))}"
            package_component = f"/pkg_{str(package)}" if str(package) else ""
            exact_component = f"/exact_{str(exact_family)}" if str(exact_family) else ""
            c2_weight_component = ""
            if len(c2_weight_values) > 1 or abs(float(c2_w)) > 1e-12:
                c2_weight_component = f"/c2w_{_fmt_float(float(c2_w))}"
            data_seed_component = f"/dseed_{int(data_seed)}" if data_seed is not None else ""
            sub = (
                f"fam_{str(fam)}"
                f"/train_{int(td)}"
                f"{val_component}"
                f"/audit_{_fmt_float(float(frac))}"
                f"/c3_{str(c3_strat)}"
                f"/lqr_{_fmt_float(float(lqr))}"
                f"/llw_{_fmt_float(float(llw))}{task_component}{law_mix_component}{c2_weight_component}{package_component}{exact_component}"
                f"/rootq_{1 if bool(rootq) else 0}"
                f"/rw_{_fmt_float(float(rw))}"
                f"/scw_{_fmt_float(float(scw))}"
                f"{gov_component}{rf_component}{doc_component}{ridge_component}{leaf_ridge_component}{leaf_endpoint_table_component}{leaf_dt_component}{leaf_knn_component}{leaf_rf_component}{sampled_pool_component}{fm_component}{data_seed_component}"
                f"{generator_component}"
                f"/sd_{int(sd)}"
                f"/hd_{int(hd)}"
            )
            base = output_root / sub / f"seed_{int(seed)}"
            out_json = base.with_suffix(".json")
            out_csv = base.with_suffix(".csv")
            artifact_dir = base.parent / f"{base.name}_artifacts"
            if skip_existing and out_json.exists() and out_csv.exists():
                continue

            parts = [
                prefix,
                f"--n-regimes {int(n_regimes)}",
                f"--vocab-size {int(vocab_size)}",
                f"--generator-profile {generator_profile_value}",
                f"--min-tokens {int(min_tokens)}",
                f"--max-tokens {int(max_tokens)}",
                f"--min-segments {int(min_segments)}",
                f"--max-segments {int(max_segments)}",
                f"--fixed-leaf-tokens {int(fixed_leaf_tokens)}",
                f"--train-docs {int(td)}",
                f"--val-docs {int(val_docs)}",
                f"--test-docs {int(test_docs)}",
                f"--model-family {str(fam)}",
                "--audit-policy fraction",
                f"--audit-fraction {float(frac)}",
                f"--c3-audit-strategy {str(c3_strat)}",
                f"--leaf-query-rate {float(lqr)}",
                f"--local-law-weight {float(llw)}",
                f"--c1-relative-weight {float(c1_rel)}",
                f"--c2-relative-weight {float(c2_rel)}",
                f"--c3-relative-weight {float(c3_rel)}",
                f"--c2-weight {float(c2_w)}",
                f"--root-weight {float(rw)}",
                f"--schedule-consistency-weight {float(scw)}",
                f"--feature-mode {str(fm)}",
                f"--state-dim {int(sd)}",
                f"--hidden-dim {int(hd)}",
                f"--n-epochs {int(n_epochs)}",
                f"--device {device}",
            ]
            if str(package):
                parts.append(f"--law-package {str(package)}")
            if str(exact_family):
                parts.append(f"--exact-family {str(exact_family)}")
            if task_weight is not None:
                parts.append(f"--root-share {float(task_weight)}")
            if data_seed is not None:
                parts.append(f"--data-seed {int(data_seed)}")
            parts.append(f"--model-seed {int(seed)}")
            if str(suite_role).strip():
                parts.append(f"--suite-role {str(suite_role).strip()}")
            if bool(include_rf_root_baseline):
                parts.append("--include-rf-root-baseline")
                parts.append(f"--rf-n-estimators {int(rf_n_estimators)}")
                parts.append(f"--rf-max-depth {int(rf_max_depth)}")
                parts.append(f"--rf-min-samples-leaf {int(rf_min_samples_leaf)}")
            if bool(include_doc_level_baseline):
                parts.append("--include-doc-level-baseline")
            if bool(include_doc_level_ridge_baseline):
                parts.append("--include-doc-level-ridge-baseline")
                parts.append(f"--doc-level-ridge-alpha {float(doc_level_ridge_alpha)}")
            if bool(include_leaf_ridge_tree_baseline):
                parts.append("--include-leaf-ridge-tree-baseline")
            if bool(include_leaf_endpoint_table_tree_baseline):
                parts.append("--include-leaf-endpoint-table-tree-baseline")
            if bool(include_leaf_dt_tree_baseline):
                parts.append("--include-leaf-dt-tree-baseline")
            if bool(include_leaf_knn_tree_baseline):
                parts.append("--include-leaf-knn-tree-baseline")
                parts.append(f"--leaf-knn-neighbors {int(leaf_knn_neighbors)}")
            if bool(include_leaf_rf_tree_baseline):
                parts.append("--include-leaf-rf-tree-baseline")
            if bool(include_sampled_leaf_pool_ridge_baseline):
                parts.append("--include-sampled-leaf-pool-ridge-baseline")
            if bool(include_sampled_leaf_pool_rf_baseline):
                parts.append("--include-sampled-leaf-pool-rf-baseline")
            if sampled_leaf_pool_counts:
                parts.append(
                    "--sampled-leaf-pool-leaf-counts "
                    + ",".join(str(int(x)) for x in sampled_leaf_pool_counts)
                )
                parts.append(
                    f"--sampled-leaf-pool-seed-offset {int(sampled_leaf_pool_seed_offset)}"
                )
            if gov_component:
                parts.append(f"--guidance-override-mode {str(gov_mode)}")
            if not bool(rootq):
                parts.append("--no-root-query")
            if not c3_include_root:
                parts.append("--no-c3-include-root")
            if cuda_device is not None:
                parts.append(f"--cuda-device {int(cuda_device)}")
            if int(eval_guidance_trials) > 0 and guidance_qs_list:
                parts.append(f"--eval-guidance-qs {guidance_qs_text}")
                parts.append(f"--eval-guidance-trials {int(eval_guidance_trials)}")
                parts.append(f"--eval-guidance-seed-offset {int(eval_guidance_seed_offset)}")
                parts.append(
                    "--eval-guidance-include-root"
                    if bool(eval_guidance_include_root)
                    else "--no-eval-guidance-include-root"
                )
            parts.extend(
                [
                    f"--torch-threads {int(torch_threads)}",
                    f"--violation-tau {float(violation_tau)}",
                    f"--seed {int(seed)}",
                    f"--json-summary {out_json}",
                    f"--csv-summary {out_csv}",
                    f"--artifact-dir {artifact_dir}",
                ]
            )
            cmd = " ".join(parts)

            cfg = OPSCountConfig(
                n_regimes=int(n_regimes),
                vocab_size=int(vocab_size),
                generator_profile=str(generator_profile_value),
                min_tokens=int(min_tokens),
                max_tokens=int(max_tokens),
                min_segments=int(min_segments),
                max_segments=int(max_segments),
                fixed_leaf_tokens=int(fixed_leaf_tokens),
                train_docs=int(td),
                val_docs=int(val_docs),
                test_docs=int(test_docs),
                model_family=str(fam),
                feature_mode=str(fm),
                state_dim=int(sd),
                hidden_dim=int(hd),
                n_epochs=int(n_epochs),
                audit_policy="fraction",
                audit_fraction=float(frac),
                c3_audit_strategy=str(c3_strat),
                c3_include_root=bool(c3_include_root),
                leaf_query_rate=float(lqr),
                law_package=str(package),
                exact_family=str(exact_family),
                local_law_weight=float(llw),
                task_objective_weight=(float(task_weight) if task_weight is not None else None),
                c1_relative_weight=float(c1_rel),
                c2_relative_weight=float(c2_rel),
                c3_relative_weight=float(c3_rel),
                c2_weight=float(c2_w),
                include_root_query=bool(rootq),
                eval_guidance_qs=tuple(guidance_qs_list),
                eval_guidance_trials=int(eval_guidance_trials),
                eval_guidance_seed_offset=int(eval_guidance_seed_offset),
                eval_guidance_include_root=bool(eval_guidance_include_root),
                guidance_override_mode=str(gov_mode),
                include_rf_root_baseline=bool(include_rf_root_baseline),
                include_doc_level_baseline=bool(include_doc_level_baseline),
                include_doc_level_ridge_baseline=bool(include_doc_level_ridge_baseline),
                include_leaf_ridge_tree_baseline=bool(include_leaf_ridge_tree_baseline),
                include_leaf_endpoint_table_tree_baseline=bool(
                    include_leaf_endpoint_table_tree_baseline
                ),
                include_leaf_dt_tree_baseline=bool(include_leaf_dt_tree_baseline),
                include_leaf_knn_tree_baseline=bool(include_leaf_knn_tree_baseline),
                include_leaf_rf_tree_baseline=bool(include_leaf_rf_tree_baseline),
                rf_n_estimators=int(rf_n_estimators),
                rf_max_depth=int(rf_max_depth),
                rf_min_samples_leaf=int(rf_min_samples_leaf),
                doc_level_ridge_alpha=float(doc_level_ridge_alpha),
                leaf_knn_neighbors=int(leaf_knn_neighbors),
                include_sampled_leaf_pool_ridge_baseline=bool(
                    include_sampled_leaf_pool_ridge_baseline
                ),
                include_sampled_leaf_pool_rf_baseline=bool(
                    include_sampled_leaf_pool_rf_baseline
                ),
                sampled_leaf_pool_leaf_counts=tuple(int(x) for x in sampled_leaf_pool_counts),
                sampled_leaf_pool_seed_offset=int(sampled_leaf_pool_seed_offset),
                violation_tau=float(violation_tau),
                suite_role=str(suite_role),
                artifact_dir=str(artifact_dir),
                seed=int(seed),
                data_seed=(int(data_seed) if data_seed is not None else None),
                model_seed=int(seed),
                use_cuda=(str(device) != "cpu"),
                cuda_device=int(cuda_device) if cuda_device is not None else None,
                torch_threads=int(torch_threads),
            )

            requires = ["torch"]
            if bool(include_rf_root_baseline):
                requires.append("sklearn")

            runs.append(
                RunSpec.create(
                    family="markov-ops-count",
                    config=asdict(cfg),
                    outputs={
                        "json_summary": str(out_json),
                        "csv_summary": str(out_csv),
                        "artifact_dir": str(artifact_dir),
                    },
                    command=cmd,
                    requires=sorted(set(requires)),
                )
            )
    return runs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Markov OPS-count sweep command list.")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--out-cmds", type=str, default="logs/markov_changepoint_ops_count_cmds.txt")
    p.add_argument("--out-manifest", type=str, default="")
    p.add_argument("--output-root", type=str, default="outputs/markov_changepoint_ops_count")

    p.add_argument("--n-regimes", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=96)
    p.add_argument(
        "--generator-profile",
        choices=list(VALID_GENERATOR_PROFILES),
        default="piecewise_markov",
        help="Document generator family. Keeps the target fixed as changepoint count.",
    )
    p.add_argument("--min-tokens", type=int, default=384)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--min-segments", type=int, default=12)
    p.add_argument("--max-segments", type=int, default=24)
    p.add_argument("--fixed-leaf-tokens", type=int, default=16)

    p.add_argument("--train-docs", type=str, default="50 100 200 500 1000 2000")
    p.add_argument("--val-docs", type=int, default=0)
    p.add_argument("--test-docs", type=int, default=1000)
    p.add_argument("--audit-fractions", type=str, default="0.05 0.1 0.2 0.5 1.0")
    p.add_argument(
        "--model-family",
        type=str,
        default="neural",
        help="Space/comma list of model families (neural, additive).",
    )
    p.add_argument("--c3-audit-strategies", type=str, default="uniform")
    p.add_argument("--c3-include-root", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--leaf-query-rates", type=str, default="1.0")
    p.add_argument(
        "--include-root-query",
        type=str,
        default="true",
        help="Space/comma list of booleans for include_root_query in learned training.",
    )
    p.add_argument(
        "--local-law-weights",
        type=str,
        default="0 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.35 0.5 0.65 0.8 0.9 1.0",
        help="Space/comma list of theorem-facing local-law tradeoff weights λ.",
    )
    p.add_argument(
        "--root-shares",
        type=str,
        default="",
        help="Optional space/comma list of explicit root-share values. Empty keeps theorem-facing `(1-lambda)` defaults.",
    )
    p.add_argument(
        "--c1-relative-weights",
        type=str,
        default="1.0",
        help="Space/comma list of relative weights assigned to C1/L1 within λ_local_law.",
    )
    p.add_argument(
        "--c2-relative-weights",
        type=str,
        default="1.0",
        help="Space/comma list of relative weights assigned to C2/L3 within λ_local_law.",
    )
    p.add_argument(
        "--c3-relative-weights",
        type=str,
        default="1.0",
        help="Space/comma list of relative weights assigned to C3/L2 within λ_local_law.",
    )
    p.add_argument(
        "--c2-weights",
        type=str,
        default="0.0",
        help="Optional space/comma list of legacy direct C2 weights.",
    )
    p.add_argument(
        "--law-packages",
        type=str,
        default="",
        help="Optional space/comma list of discrete theorem-law packages.",
    )
    p.add_argument(
        "--exact-families",
        type=str,
        default="",
        help="Optional space/comma list of exact stress families.",
    )
    p.add_argument("--root-weights", type=str, default="1.0")
    p.add_argument("--schedule-consistency-weights", type=str, default="0.0")
    p.add_argument(
        "--guidance-override-modes",
        type=str,
        default="reset",
        help="Space/comma list of guidance override modes (reset, adjust).",
    )
    p.add_argument(
        "--eval-guidance-qs",
        type=str,
        default="",
        help="Optional comma/space list of inference-time oracle guidance q values.",
    )
    p.add_argument(
        "--eval-guidance-trials",
        type=int,
        default=0,
        help="Guidance trials per q for guided_eval_curve. 0 disables guidance evaluation.",
    )
    p.add_argument("--eval-guidance-seed-offset", type=int, default=100000)
    p.add_argument(
        "--eval-guidance-include-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether root node is eligible for inference-time guidance replacement.",
    )
    p.add_argument(
        "--data-seeds",
        type=str,
        default="",
        help="Optional space/comma list of fixed corpus seeds. Empty keeps legacy seed semantics.",
    )
    p.add_argument("--seeds", type=str, default="0 1 2 3 4 5 6 7")

    p.add_argument(
        "--include-rf-root-baseline", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--include-doc-level-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--include-doc-level-ridge-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--include-leaf-ridge-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--include-leaf-endpoint-table-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--include-leaf-dt-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--include-leaf-knn-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--include-leaf-rf-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=16)
    p.add_argument("--rf-min-samples-leaf", type=int, default=5)
    p.add_argument("--doc-level-ridge-alpha", type=float, default=1.0)
    p.add_argument("--leaf-knn-neighbors", type=int, default=32)
    p.add_argument(
        "--include-sampled-leaf-pool-ridge-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--include-sampled-leaf-pool-rf-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--sampled-leaf-pool-leaf-counts",
        type=str,
        default="",
        help="Optional space/comma list of pooled sampled-leaf budgets per document.",
    )
    p.add_argument("--sampled-leaf-pool-seed-offset", type=int, default=200000)

    p.add_argument(
        "--feature-mode",
        choices=["full", "no_endpoints", "token_full", "token_bow"],
        default="full",
    )
    p.add_argument(
        "--feature-modes",
        type=str,
        default="",
        help="Optional space/comma list of feature modes. If set, overrides --feature-mode.",
    )
    p.add_argument(
        "--state-dims",
        type=str,
        default="32",
        help="Space/comma list of learned sketch latent dimensions (state_dim).",
    )
    p.add_argument(
        "--hidden-dims",
        type=str,
        default="128",
        help="Space/comma list of MLP hidden dimensions (hidden_dim). Ignored if --hidden-dim-multiplier is set.",
    )
    p.add_argument(
        "--hidden-dim-multiplier",
        type=float,
        default=0.0,
        help="If >0, sets hidden_dim = max(hidden_dim_min, round(multiplier*state_dim)) per state_dim.",
    )
    p.add_argument("--hidden-dim-min", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device mode. 'auto' leaves CPU/GPU placement to the launcher.",
    )
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch thread count per process (sweep-friendly). <=0 keeps torch defaults.",
    )
    p.add_argument("--violation-tau", type=float, default=0.0)

    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--log-dir", type=str, default="")
    p.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    out_cmds = Path(args.out_cmds)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    out_manifest = (
        Path(args.out_manifest)
        if str(args.out_manifest).strip()
        else out_cmds.with_name(out_cmds.stem + "_manifest.jsonl")
    )

    feature_modes = (
        _parse_items(args.feature_modes)
        if str(args.feature_modes).strip()
        else [str(args.feature_mode)]
    )

    runs = _iter_runs(
        python_bin=str(args.python_bin),
        n_regimes=int(args.n_regimes),
        vocab_size=int(args.vocab_size),
        generator_profile=str(args.generator_profile),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        train_docs=_parse_ints(args.train_docs),
        val_docs=int(args.val_docs),
        test_docs=int(args.test_docs),
        audit_fractions=_parse_floats(args.audit_fractions),
        c3_audit_strategies=_parse_items(args.c3_audit_strategies),
        c3_include_root=bool(args.c3_include_root),
        leaf_query_rates=_parse_floats(args.leaf_query_rates),
        include_root_queries=_parse_bools(args.include_root_query),
        local_law_weights=_parse_floats(args.local_law_weights),
        task_objective_weights=(
            _parse_floats(args.root_shares)
            if str(args.root_shares).strip()
            else []
        ),
        c1_relative_weights=_parse_floats(args.c1_relative_weights),
        c2_relative_weights=_parse_floats(args.c2_relative_weights),
        c3_relative_weights=_parse_floats(args.c3_relative_weights),
        root_weights=_parse_floats(args.root_weights),
        schedule_consistency_weights=_parse_floats(args.schedule_consistency_weights),
        guidance_override_modes=_parse_items(args.guidance_override_modes),
        eval_guidance_qs=_parse_floats(args.eval_guidance_qs),
        eval_guidance_trials=int(args.eval_guidance_trials),
        eval_guidance_seed_offset=int(args.eval_guidance_seed_offset),
        eval_guidance_include_root=bool(args.eval_guidance_include_root),
        include_rf_root_baseline=bool(args.include_rf_root_baseline),
        include_doc_level_baseline=bool(args.include_doc_level_baseline),
        include_doc_level_ridge_baseline=bool(args.include_doc_level_ridge_baseline),
        include_leaf_ridge_tree_baseline=bool(args.include_leaf_ridge_tree_baseline),
        include_leaf_endpoint_table_tree_baseline=bool(
            args.include_leaf_endpoint_table_tree_baseline
        ),
        include_leaf_dt_tree_baseline=bool(args.include_leaf_dt_tree_baseline),
        include_leaf_knn_tree_baseline=bool(args.include_leaf_knn_tree_baseline),
        include_leaf_rf_tree_baseline=bool(args.include_leaf_rf_tree_baseline),
        rf_n_estimators=int(args.rf_n_estimators),
        rf_max_depth=int(args.rf_max_depth),
        rf_min_samples_leaf=int(args.rf_min_samples_leaf),
        doc_level_ridge_alpha=float(args.doc_level_ridge_alpha),
        leaf_knn_neighbors=int(args.leaf_knn_neighbors),
        include_sampled_leaf_pool_ridge_baseline=bool(
            args.include_sampled_leaf_pool_ridge_baseline
        ),
        include_sampled_leaf_pool_rf_baseline=bool(args.include_sampled_leaf_pool_rf_baseline),
        sampled_leaf_pool_leaf_counts=_parse_ints(args.sampled_leaf_pool_leaf_counts),
        sampled_leaf_pool_seed_offset=int(args.sampled_leaf_pool_seed_offset),
        data_seeds=_parse_ints(args.data_seeds),
        seeds=_parse_ints(args.seeds),
        output_root=Path(args.output_root),
        model_families=_parse_items(args.model_family),
        feature_modes=feature_modes,
        state_dims=_parse_ints(args.state_dims),
        hidden_dims=_parse_ints(args.hidden_dims),
        hidden_dim_multiplier=(
            float(args.hidden_dim_multiplier) if float(args.hidden_dim_multiplier) > 0.0 else None
        ),
        hidden_dim_min=int(args.hidden_dim_min),
        n_epochs=int(args.n_epochs),
        device=str(args.device),
        cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
        violation_tau=float(args.violation_tau),
        torch_threads=int(args.torch_threads),
        skip_existing=bool(args.skip_existing),
        law_packages=_parse_items(args.law_packages),
        exact_families=_parse_items(args.exact_families),
        c2_weights=_parse_floats(args.c2_weights),
    )

    cmds = [r.command for r in runs]
    out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    write_manifest_jsonl(out_manifest, runs)

    print(
        json.dumps(
            {
                "out_cmds": str(out_cmds),
                "out_manifest": str(out_manifest),
                "n_commands": int(len(cmds)),
            },
            indent=2,
        )
    )

    if not bool(args.execute) or not cmds:
        return 0

    log_dir = (
        Path(args.log_dir)
        if str(args.log_dir).strip()
        else (out_cmds.parent / f"{out_cmds.stem}_logs")
    )
    try:
        results = run_commands(
            cmds,
            jobs=int(args.jobs),
            log_dir=log_dir,
            fail_fast=bool(args.fail_fast),
        )
    except KeyboardInterrupt:
        return 130

    n_fail = sum(1 for r in results if int(r.returncode) != 0)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
