#!/usr/bin/env python3
"""Run OPS-style Markov changepoint-count simulation (oracle-preserving sketches + IPW/DSL demo)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence

import torch

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
    OPSCountSummary,
    VALID_AUDIT_POLICIES,
    VALID_C3_AUDIT_STRATEGIES,
    VALID_DOC_SEQUENCE_OBJECTIVES,
    VALID_DOC_TRANSFORMER_HEAD_FAMILIES,
    VALID_EXACT_FAMILIES,
    VALID_GENERATOR_PROFILES,
    VALID_LAW_PACKAGES,
    VALID_LOCAL_LAW_OBJECTIVE_MODES,
    VALID_MODEL_FAMILIES,
    build_markov_changepoint_ops_count_data_bundle,
    run_markov_changepoint_ops_count_experiment,
    valid_theorem_feature_adapters,
)


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OPS-style Markov changepoint-count simulation.")

    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=96)
    parser.add_argument(
        "--generator-profile",
        type=str,
        choices=list(VALID_GENERATOR_PROFILES),
        default="piecewise_markov",
        help="Document generator family. Keeps the target fixed as the changepoint count.",
    )
    parser.add_argument("--min-tokens", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--min-segments", type=int, default=12)
    parser.add_argument("--max-segments", type=int, default=24)
    parser.add_argument("--min-seg-len", type=int, default=8)
    parser.add_argument("--max-seg-len", type=int, default=32)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--transition-log-std", type=float, default=1.25)

    parser.add_argument("--fixed-leaf-tokens", type=int, default=16)

    parser.add_argument("--train-docs", type=int, default=1000)
    parser.add_argument("--val-docs", type=int, default=0)
    parser.add_argument("--test-docs", type=int, default=1000)

    parser.add_argument(
        "--model-family",
        type=str,
        choices=list(VALID_MODEL_FAMILIES),
        default="neural",
        help="Sketch family. 'neural' learns an unstructured merger; 'additive' uses a structured additive merger.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["full", "no_endpoints", "token_full", "token_bow"],
        default="full",
        help="Leaf feature family used by the learned sketch.",
    )
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--law-package",
        type=str,
        choices=[""] + list(VALID_LAW_PACKAGES),
        default="",
        help="Discrete theorem-law bundle used by the stress suites.",
    )
    parser.add_argument(
        "--exact-family",
        type=str,
        choices=[""] + list(VALID_EXACT_FAMILIES),
        default="",
        help="If set, skip learned training and report the requested exact stress family only.",
    )
    parser.add_argument(
        "--local-law-weight",
        type=float,
        default=None,
        help=(
            "Overall theorem-facing local-law tradeoff λ. If set, it overrides the legacy "
            "--leaf-weight/--c3-weight pair using the relative weights below."
        ),
    )
    parser.add_argument(
        "--root-share",
        dest="root_share",
        type=float,
        default=None,
        help=(
            "Optional explicit share for the task/root loss. When set, this overrides the "
            "default theorem-facing `(1 - lambda)` task mass and yields a free weighted "
            "composite objective."
        ),
    )
    parser.add_argument(
        "--c1-relative-weight",
        type=float,
        default=1.0,
        help="Relative share of λ_local_law assigned to C1/L1 leaf preservation. Defaults to an equal split.",
    )
    parser.add_argument(
        "--c2-relative-weight",
        type=float,
        default=1.0,
        help="Relative share of λ_local_law assigned to C2/L3 idempotence. Defaults to an equal split.",
    )
    parser.add_argument(
        "--c3-relative-weight",
        type=float,
        default=1.0,
        help="Relative share of λ_local_law assigned to C3/L2 merge preservation. Defaults to an equal split.",
    )
    parser.add_argument(
        "--c3-weight",
        type=float,
        default=0.0,
        help="Legacy direct C3 weight. Ignored when --local-law-weight is set.",
    )
    parser.add_argument(
        "--c2-weight",
        type=float,
        default=0.0,
        help="Legacy direct C2/L3 idempotence weight. Ignored when --local-law-weight or --law-package is set.",
    )
    parser.add_argument(
        "--leaf-weight",
        type=float,
        default=0.0,
        help="Legacy direct C1 weight. Ignored when --local-law-weight is set.",
    )
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument(
        "--schedule-consistency-weight",
        type=float,
        default=0.0,
        help="Proxy-only associativity regularizer; not part of the Lean local-law bundle.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--tree-c2-mode",
        type=str,
        choices=["reconstruction", "fiber"],
        default="reconstruction",
        help=(
            "C2 supervision route for theorem-feature tree modes. "
            "'reconstruction' replays decoded summaries; 'fiber' uses same/different "
            "feature-class supervision."
        ),
    )
    parser.add_argument(
        "--c2-learned-resummary",
        action="store_true",
        default=False,
        help=(
            "Enable learned re-summarization cycle for C2 (Lean L3). "
            "When set, decode_summary/encode_summary are learned MLPs so the "
            "round-trip is non-trivial and C2 tests a functional property."
        ),
    )
    parser.add_argument(
        "--theorem-feature-adapter",
        type=str,
        choices=list(valid_theorem_feature_adapters()),
        default="markov_count_sketch",
        help="Theorem-feature adapter used to define oracle labels and same/different pair semantics.",
    )
    parser.add_argument(
        "--theorem-pair-same-threshold",
        type=float,
        default=None,
        help="Optional adapter-level threshold for marking a pair as the same theorem-feature class.",
    )
    parser.add_argument(
        "--theorem-pair-diff-threshold",
        type=float,
        default=None,
        help="Optional adapter-level threshold for marking a pair as different theorem-feature classes.",
    )

    parser.add_argument(
        "--audit-policy",
        type=str,
        choices=list(VALID_AUDIT_POLICIES),
        default="fraction",
        help="Internal-node label sampling policy used for C3 supervision.",
    )
    parser.add_argument("--audit-fixed-nodes", type=int, default=0)
    parser.add_argument("--audit-fraction", type=float, default=0.2)
    parser.add_argument("--audit-scale", type=float, default=1.0)
    parser.add_argument(
        "--c3-audit-strategy",
        type=str,
        choices=list(VALID_C3_AUDIT_STRATEGIES),
        default="uniform",
        help="Sampling strategy for selecting internal nodes to label under the audit budget.",
    )
    parser.add_argument(
        "--c3-include-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always include the root merge in C3 supervision when internal-node budget is positive.",
    )
    parser.add_argument(
        "--leaf-query-rate",
        type=float,
        default=1.0,
        help="Fraction of leaves labeled per doc for leaf-level supervision (in [0,1]).",
    )
    parser.add_argument(
        "--no-root-query",
        action="store_true",
        help="Disable one oracle label per doc at the root during learned training.",
    )
    parser.add_argument(
        "--ipw-leaf-sample-rate",
        type=float,
        default=1.0,
        help="Leaf-node sampling rate used by the unified IPW path.",
    )
    parser.add_argument(
        "--ipw-internal-sample-rate",
        type=float,
        default=1.0,
        help="Internal-node sampling rate used by the unified IPW path; the root is not special-cased.",
    )
    parser.add_argument(
        "--local-law-objective-mode",
        type=str,
        choices=list(VALID_LOCAL_LAW_OBJECTIVE_MODES),
        default="corrected_local_law",
        help=(
            "Local-law objective for FNO/tree models. 'corrected_local_law' uses "
            "the retained proxy population plus sampled oracle residuals; "
            "'sampled_ipw' keeps the sampled-only Hajek/IPW objective."
        ),
    )
    parser.add_argument(
        "--use-residual-decomposition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When unified IPW is enabled, supervise merge nodes on residual corrections "
            "instead of direct node counts."
        ),
    )
    parser.add_argument(
        "--root-only-train-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of training/validation docs collapsed to a degenerate one-node "
            "full-document view for the shared-g FNO lane."
        ),
    )
    parser.add_argument(
        "--doc-sequence-train-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of training docs routed through the in-model full-document "
            "doc-sequence objective."
        ),
    )
    parser.add_argument(
        "--eval-guidance-qs",
        type=str,
        default="",
        help="Optional comma/space list of inference-time oracle visibility q values in [0,1].",
    )
    parser.add_argument(
        "--eval-guidance-trials",
        type=int,
        default=0,
        help="Number of stochastic guidance trials per q on test docs (0 disables guidance eval).",
    )
    parser.add_argument(
        "--eval-guidance-seed-offset",
        type=int,
        default=100_000,
        help="Seed offset used by guidance evaluation RNG.",
    )
    parser.add_argument(
        "--eval-guidance-include-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether root internal node is eligible for inference-time oracle replacement.",
    )
    parser.add_argument(
        "--guidance-override-mode",
        choices=["reset", "adjust"],
        default="reset",
        help=(
            "How inference-time oracle overrides modify a neural sketch state. "
            "'reset' replaces h with a pure readout-aligned vector; "
            "'adjust' shifts h only along the readout direction, preserving orthogonal components."
        ),
    )

    parser.add_argument(
        "--include-rf-root-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also train a simple doc-level RandomForest baseline that predicts the root count "
            "from aggregate leaf features (no inference-time oracle)."
        ),
    )
    parser.add_argument(
        "--include-doc-level-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also train the matched no-tree baseline by applying the same sketch family "
            "to a single full-document leaf per document."
        ),
    )
    parser.add_argument(
        "--include-doc-sequence-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also train a full-document token-sequence baseline that predicts "
            "changepoint count directly from the raw observed document."
        ),
    )
    parser.add_argument(
        "--include-doc-transformer-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also train a simple full-document transformer baseline on the raw "
            "token sequence with document-level changepoint labels only."
        ),
    )
    parser.add_argument(
        "--include-doc-level-ridge-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also fit a closed-form ridge regressor on the same single full-document "
            "feature view. This is the simplest pooled no-tree control."
        ),
    )
    parser.add_argument(
        "--include-leaf-ridge-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also fit ridge on sampled leaf-local labels and evaluate it through "
            "the exact additive tree merge law."
        ),
    )
    parser.add_argument(
        "--include-leaf-endpoint-table-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also fit a simple piecewise-constant local baseline keyed by "
            "leaf endpoints and leaf length, then evaluate it through the exact additive "
            "tree merge law."
        ),
    )
    parser.add_argument(
        "--include-leaf-dt-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also fit a local single decision-tree regressor on sampled leaf-local "
            "labels and evaluate it through the exact additive tree merge law."
        ),
    )
    parser.add_argument(
        "--include-leaf-knn-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also fit a local distance-weighted kNN regressor on sampled leaf-local "
            "labels and evaluate it through the exact additive tree merge law."
        ),
    )
    parser.add_argument(
        "--include-leaf-rf-tree-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also fit a local random-forest regressor on sampled leaf-local labels "
            "and evaluate it through the exact additive tree merge law."
        ),
    )
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=16)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=5)
    parser.add_argument("--doc-level-ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--doc-level-ridge-breakdown-orders",
        type=str,
        default="",
        help="Optional comma/space list of ngram orders for separate full-document ridge baselines.",
    )
    parser.add_argument(
        "--doc-sequence-objective",
        type=str,
        choices=list(VALID_DOC_SEQUENCE_OBJECTIVES),
        default="count_ce_only",
        help="Requested objective for the full-document neuraloperator baseline. The FNO path currently uses count-support classification.",
    )
    parser.add_argument(
        "--doc-transformer-head-family",
        type=str,
        choices=list(VALID_DOC_TRANSFORMER_HEAD_FAMILIES),
        default="boundary_sum_count_hybrid",
        help="Root-label-only head family used by the full-sequence transformer baseline.",
    )
    parser.add_argument(
        "--doc-transformer-layers",
        type=int,
        default=0,
        help="Optional explicit transformer depth for the full-sequence transformer baseline (0 uses the default heuristic).",
    )
    parser.add_argument("--leaf-knn-neighbors", type=int, default=32)
    parser.add_argument(
        "--include-sampled-leaf-pool-ridge-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fit a pooled no-tree ridge regressor on uniformly sampled realized leaves. "
            "This keeps the leaf-level information surface but removes tree composition."
        ),
    )
    parser.add_argument(
        "--include-sampled-leaf-pool-rf-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fit a pooled no-tree random forest on uniformly sampled realized leaves."
        ),
    )
    parser.add_argument(
        "--sampled-leaf-pool-leaf-counts",
        type=str,
        default="",
        help="Comma/space list of sampled leaf budgets per document for pooled no-tree fairness baselines.",
    )
    parser.add_argument("--sampled-leaf-pool-seed-offset", type=int, default=200_000)

    parser.add_argument("--violation-tau", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Optional corpus-generation seed. Defaults to --seed when unset.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Optional optimization / initialization seed. Defaults to --seed when unset.",
    )
    parser.add_argument(
        "--val-seed-offset",
        type=int,
        default=5_000,
        help="Seed offset used to generate the fixed validation corpus from --data-seed.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device mode. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Alias for --device cpu (kept for backward compatibility).",
    )
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Set torch intra-op/inter-op threads (<=0 keeps torch defaults).",
    )

    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/markov_changepoint_ops_count/train_1000_seed_0.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/markov_changepoint_ops_count/train_1000_seed_0.csv",
        help="CSV summary output path.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help="Optional directory for serialized g artifacts. Defaults to a sibling directory next to --json-summary.",
    )
    parser.add_argument(
        "--save-data-bundle",
        type=str,
        default="",
        help=(
            "Optional path for serializing the generated Markov train/val/test corpus. "
            "When set, the run reuses that in-memory bundle instead of regenerating docs."
        ),
    )
    parser.add_argument(
        "--load-data-bundle",
        type=str,
        default="",
        help=(
            "Optional path to a previously saved Markov train/val/test corpus bundle. "
            "Use this to keep train/val/test sets fixed across comparisons."
        ),
    )
    parser.add_argument(
        "--suite-role",
        type=str,
        default="",
        help="Optional normalized suite role (positive_controls, support_scaling, failure_modes, relevance_mediation, hardness).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (in addition to saving files).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _rows_from_summary(summary: OPSCountSummary) -> List[dict]:
    cfg = dict(summary.config)
    geom = dict(summary.training_geometry)
    objective = dict(summary.objective)
    objective_flat = {
        "objective_parameterization": objective.get("parameterization"),
        "objective_training_scheme": objective.get("training_scheme"),
        "objective_root_share": objective.get("root_share"),
        "objective_root_share_source": objective.get("root_share_source"),
        "objective_root_supervision_active": objective.get("root_supervision_active"),
        "objective_local_law_weight": objective.get("local_law_weight"),
        "objective_local_law_c1_weight": objective.get("local_law_c1_weight"),
        "objective_local_law_c2_weight": objective.get("local_law_c2_weight"),
        "objective_local_law_c3_weight": objective.get("local_law_c3_weight"),
        "objective_local_law_c1_share": objective.get("local_law_c1_share"),
        "objective_local_law_c2_share": objective.get("local_law_c2_share"),
        "objective_local_law_c3_share": objective.get("local_law_c3_share"),
        "objective_law_package": objective.get("law_package"),
        "objective_proxy_schedule_consistency_weight": objective.get(
            "proxy_schedule_consistency_weight"
        ),
    }
    rows: List[dict] = []
    for name, metrics in summary.metrics.items():
        if not isinstance(metrics, dict):
            continue
        row = {
            "sketch": str(name),
            **{k: cfg.get(k) for k in cfg.keys()},
            **objective_flat,
            **{f"train_{k}": v for k, v in geom.items()},
            **metrics,
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if len(rows) == 0:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cpu:
        args.device = "cpu"
    if str(args.save_data_bundle).strip() and str(args.load_data_bundle).strip():
        raise ValueError("at most one of --save-data-bundle or --load-data-bundle may be set")
    use_cuda = args.device in ("auto", "cuda")
    if args.device == "auto":
        use_cuda = torch.cuda.is_available()
    json_path = Path(args.json_summary)
    artifact_dir = Path(args.artifact_dir) if str(args.artifact_dir).strip() else (
        json_path.parent / f"{json_path.stem}_artifacts"
    )

    cfg = OPSCountConfig(
        n_regimes=int(args.n_regimes),
        vocab_size=int(args.vocab_size),
        generator_profile=str(args.generator_profile),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=int(args.min_seg_len),
        max_seg_len=int(args.max_seg_len),
        sinkhorn_iters=int(args.sinkhorn_iters),
        transition_log_std=float(args.transition_log_std),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        train_docs=int(args.train_docs),
        val_docs=int(args.val_docs),
        test_docs=int(args.test_docs),
        model_family=str(args.model_family),
        feature_mode=str(args.feature_mode),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        law_package=str(args.law_package),
        exact_family=str(args.exact_family),
        local_law_weight=(float(args.local_law_weight) if args.local_law_weight is not None else None),
        task_objective_weight=(
            float(args.root_share) if args.root_share is not None else None
        ),
        c1_relative_weight=float(args.c1_relative_weight),
        c2_relative_weight=float(args.c2_relative_weight),
        c3_relative_weight=float(args.c3_relative_weight),
        c3_weight=float(args.c3_weight),
        c2_weight=float(args.c2_weight),
        leaf_weight=float(args.leaf_weight),
        root_weight=float(args.root_weight),
        schedule_consistency_weight=float(args.schedule_consistency_weight),
        grad_clip_norm=float(args.grad_clip_norm),
        tree_c2_mode=str(args.tree_c2_mode),
        c2_learned_resummary=bool(args.c2_learned_resummary),
        theorem_feature_adapter=str(args.theorem_feature_adapter),
        theorem_pair_same_threshold=(
            float(args.theorem_pair_same_threshold)
            if args.theorem_pair_same_threshold is not None
            else None
        ),
        theorem_pair_diff_threshold=(
            float(args.theorem_pair_diff_threshold)
            if args.theorem_pair_diff_threshold is not None
            else None
        ),
        audit_policy=str(args.audit_policy),
        audit_fixed_nodes=int(args.audit_fixed_nodes),
        audit_fraction=float(args.audit_fraction),
        audit_scale=float(args.audit_scale),
        c3_audit_strategy=str(args.c3_audit_strategy),
        c3_include_root=bool(args.c3_include_root),
        leaf_query_rate=float(args.leaf_query_rate),
        include_root_query=not bool(args.no_root_query),
        ipw_leaf_sample_rate=float(args.ipw_leaf_sample_rate),
        ipw_internal_sample_rate=float(args.ipw_internal_sample_rate),
        local_law_objective_mode=str(args.local_law_objective_mode),
        use_residual_decomposition=bool(args.use_residual_decomposition),
        root_only_train_fraction=float(args.root_only_train_fraction),
        doc_sequence_train_fraction=float(args.doc_sequence_train_fraction),
        eval_guidance_qs=tuple(_parse_float_list(str(args.eval_guidance_qs))),
        eval_guidance_trials=int(args.eval_guidance_trials),
        eval_guidance_seed_offset=int(args.eval_guidance_seed_offset),
        eval_guidance_include_root=bool(args.eval_guidance_include_root),
        guidance_override_mode=str(args.guidance_override_mode),
        include_rf_root_baseline=bool(args.include_rf_root_baseline),
        include_doc_level_baseline=bool(args.include_doc_level_baseline),
        include_doc_sequence_baseline=bool(args.include_doc_sequence_baseline),
        include_doc_transformer_baseline=bool(args.include_doc_transformer_baseline),
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
        doc_level_ridge_breakdown_orders=tuple(
            _parse_int_list(str(args.doc_level_ridge_breakdown_orders))
        ),
        doc_sequence_objective=str(args.doc_sequence_objective),
        doc_transformer_head_family=str(args.doc_transformer_head_family),
        doc_transformer_layers=int(args.doc_transformer_layers),
        leaf_knn_neighbors=int(args.leaf_knn_neighbors),
        include_sampled_leaf_pool_ridge_baseline=bool(
            args.include_sampled_leaf_pool_ridge_baseline
        ),
        include_sampled_leaf_pool_rf_baseline=bool(args.include_sampled_leaf_pool_rf_baseline),
        sampled_leaf_pool_leaf_counts=tuple(
            _parse_int_list(str(args.sampled_leaf_pool_leaf_counts))
        ),
        sampled_leaf_pool_seed_offset=int(args.sampled_leaf_pool_seed_offset),
        violation_tau=float(args.violation_tau),
        suite_role=str(args.suite_role),
        artifact_dir=str(artifact_dir),
        seed=int(args.seed),
        data_seed=(int(args.data_seed) if args.data_seed is not None else None),
        model_seed=(int(args.model_seed) if args.model_seed is not None else None),
        val_seed_offset=int(args.val_seed_offset),
        use_cuda=bool(use_cuda),
        cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
        torch_threads=int(args.torch_threads),
    )

    data_bundle: MarkovOPSDataBundle | None = None
    if str(args.load_data_bundle).strip():
        data_bundle = MarkovOPSDataBundle.load(Path(str(args.load_data_bundle).strip()))
    elif str(args.save_data_bundle).strip():
        data_bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
        data_bundle.save(Path(str(args.save_data_bundle).strip()))

    summary = run_markov_changepoint_ops_count_experiment(cfg, data_bundle=data_bundle)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")

    rows = _rows_from_summary(summary)
    _write_csv(Path(args.csv_summary), rows)

    if args.json:
        print(summary.to_json())
    else:
        print(json.dumps({"json_summary": str(json_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
