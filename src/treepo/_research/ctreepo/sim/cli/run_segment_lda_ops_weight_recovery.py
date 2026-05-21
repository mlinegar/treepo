#!/usr/bin/env python3
"""Run Segment-LDA OPS weight-recovery simulation (topic unigram+bigram oracle, ridge recovery)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import List, Sequence

from treepo._research.ctreepo.sim.core.segment_lda_ops_weight_recovery import (
    SegmentLDAOpsWeightRecoveryConfig,
    SegmentLDAOpsWeightRecoverySummary,
    VALID_AUDIT_POLICIES,
    VALID_AUDIT_STRATEGIES,
    VALID_BOUNDARY_PROFILES,
    VALID_DEVICE_MODES,
    VALID_FEATURE_INFERENCE,
    VALID_TOPIC_PHI_ESTIMATORS,
    VALID_TOPIC_SOURCES,
    VALID_TOPIC_PROCESSES,
    run_segment_lda_ops_weight_recovery_experiment,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Segment-LDA OPS weight-recovery simulation.")

    # Generator.
    parser.add_argument("--n-topics", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--max-segments", type=int, default=6)
    parser.add_argument("--min-seg-len", type=int, default=48)
    parser.add_argument("--max-seg-len", type=int, default=256)
    parser.add_argument("--leaf-tokens", type=int, default=16)
    parser.add_argument(
        "--align-segments-to-leaves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, topic changes occur only at leaf boundaries.",
    )
    parser.add_argument("--doc-topic-concentration", type=float, default=0.6)
    parser.add_argument(
        "--topic-process",
        type=str,
        choices=list(VALID_TOPIC_PROCESSES),
        default="segments",
        help="Latent topic process within a document (segmented LDA vs bag-of-words LDA).",
    )
    parser.add_argument(
        "--boundary-profile",
        type=str,
        choices=list(VALID_BOUNDARY_PROFILES),
        default="uniform",
        help="Global segment-boundary location profile (only used when topic-process=segments).",
    )
    parser.add_argument(
        "--boundary-profile-strength",
        type=float,
        default=0.0,
        help="How strongly to bias boundaries toward the chosen profile (0=uniform).",
    )
    parser.add_argument(
        "--boundary-profile-seed",
        type=int,
        default=-1,
        help="Seed for a random boundary profile (negative derives from --seed).",
    )
    parser.add_argument(
        "--segment-length-power",
        type=float,
        default=1.0,
        help="Bias segment lengths (in leaves) toward longer values: weight ∝ len^p (p=0 uniform).",
    )

    # Topic-word distributions.
    parser.add_argument("--topic-concentration", type=float, default=0.2)
    parser.add_argument(
        "--emission-mode",
        type=str,
        choices=["anchored", "disjoint"],
        default="anchored",
    )
    parser.add_argument("--anchor-words-per-topic", type=int, default=20)
    parser.add_argument("--anchor-multiplier", type=float, default=25.0)

    # Oracle weights.
    parser.add_argument("--relevant-topics", type=int, default=2)
    parser.add_argument("--theta-scale", type=float, default=1.0)
    parser.add_argument(
        "--zero-diagonal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, W[i,i]=0 so the bigram term emphasizes boundaries/transitions.",
    )
    parser.add_argument("--lambda-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--oracle-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to *training* oracle labels only (evaluation remains noiseless).",
    )

    # Supervision / budgets.
    parser.add_argument(
        "--audit-policy",
        type=str,
        choices=list(VALID_AUDIT_POLICIES),
        default="fraction",
    )
    parser.add_argument("--audit-fixed-nodes", type=int, default=0)
    parser.add_argument("--audit-fraction", type=float, default=0.2)
    parser.add_argument("--audit-scale", type=float, default=1.0)
    parser.add_argument(
        "--audit-strategy",
        type=str,
        choices=list(VALID_AUDIT_STRATEGIES),
        default="random",
    )
    parser.add_argument("--oracle-cost-power", type=float, default=1.25)
    parser.add_argument("--oracle-cost-per-query", type=float, default=0.0)

    # Estimation.
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--topic-source",
        type=str,
        choices=list(VALID_TOPIC_SOURCES),
        default="infer",
        help="Whether regression features come from true topics (upper bound) or inferred from words.",
    )
    parser.add_argument(
        "--feature-inference",
        type=str,
        choices=list(VALID_FEATURE_INFERENCE),
        default="hard",
        help="How to convert word observations into topic features when topic-source=infer.",
    )

    # Topic-word estimation (Tensor-LDA-inspired).
    parser.add_argument(
        "--topic-phi-estimator",
        type=str,
        choices=list(VALID_TOPIC_PHI_ESTIMATORS),
        default="true",
        help="How to obtain topic-word distributions for inference (oracle, tensor, embedding, neural).",
    )
    parser.add_argument(
        "--topic-phi-docs",
        type=int,
        default=0,
        help="Effective number of docs used for topic estimation (<=0 defaults to --train-docs).",
    )
    parser.add_argument("--tlda-delta", type=float, default=0.10)
    parser.add_argument("--tlda-rate-constant", type=float, default=1.0)
    parser.add_argument("--tlda-sigmaK-floor", type=float, default=1e-6)
    parser.add_argument(
        "--online-tensor-lda-burn-in-docs",
        type=int,
        default=0,
        help="Burn-in docs used to estimate whitening and initialize factors (0=auto).",
    )
    parser.add_argument("--online-tensor-lda-batch-docs", type=int, default=32)
    parser.add_argument("--online-tensor-lda-passes", type=int, default=1)
    parser.add_argument("--online-tensor-lda-lr", type=float, default=0.1)
    parser.add_argument("--online-tensor-lda-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--embedding-topic-svd-dim-extra", type=int, default=4)
    parser.add_argument("--embedding-topic-kmeans-inits", type=int, default=8)
    parser.add_argument("--embedding-topic-kmeans-max-iter", type=int, default=80)
    parser.add_argument("--embedding-topic-assignment-temperature", type=float, default=0.35)
    parser.add_argument("--embedding-topic-ppmi-shift", type=float, default=1.0)
    parser.add_argument("--neural-topic-base-estimator", type=str, default="tensor_lda")
    parser.add_argument("--neural-topic-seed-fraction", type=float, default=0.35)
    parser.add_argument("--neural-topic-hidden-dim", type=int, default=48)
    parser.add_argument("--neural-topic-steps", type=int, default=60)
    parser.add_argument("--neural-topic-lr", type=float, default=3e-3)
    parser.add_argument("--neural-topic-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-topic-mix-samples", type=int, default=128)
    parser.add_argument("--neural-topic-mix-temperature", type=float, default=1.0)
    parser.add_argument("--neural-topic-operator-boost", type=float, default=1.4)
    parser.add_argument("--neural-topic-seed-llm-min-weight", type=float, default=0.2)
    parser.add_argument("--neural-topic-seed-llm-max-weight", type=float, default=0.55)
    parser.add_argument("--neural-topic-similarity-temperature", type=float, default=0.15)
    parser.add_argument("--neural-topic-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--topic-phi-permute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, randomly permute estimated topics (identifiability is up to permutation).",
    )
    parser.add_argument(
        "--run-all-feature-modes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, also report ridge metrics for (true topics) and (infer topics using true φ).",
    )
    parser.add_argument("--device", type=str, choices=list(VALID_DEVICE_MODES), default="auto")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=0)

    # Eval sizes / misc.
    parser.add_argument("--train-docs", type=int, default=1000)
    parser.add_argument("--test-docs", type=int, default=1000)
    parser.add_argument("--violation-tau", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--json-summary", type=str, required=True)
    parser.add_argument("--csv-summary", type=str, required=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout as well.")
    return parser.parse_args(list(argv) if argv is not None else None)


def _rows_from_summary(summary: SegmentLDAOpsWeightRecoverySummary) -> List[dict]:
    cfg = dict(summary.config)
    truth = dict(summary.weight_truth)
    geom = dict(summary.training_geometry)
    meta = dict(summary.topic_meta)
    rows: List[dict] = []
    for name, metrics in summary.metrics.items():
        if not isinstance(metrics, dict):
            continue
        row = {
            "sketch": str(name),
            **{k: cfg.get(k) for k in cfg.keys()},
            **{f"topic_{k}": v for k, v in meta.items()},
            **{f"truth_{k}": v for k, v in truth.items()},
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

    cfg = SegmentLDAOpsWeightRecoveryConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=int(args.min_seg_len),
        max_seg_len=int(args.max_seg_len),
        leaf_tokens=int(args.leaf_tokens),
        align_segments_to_leaves=bool(args.align_segments_to_leaves),
        doc_topic_concentration=float(args.doc_topic_concentration),
        topic_process=str(args.topic_process),
        boundary_profile=str(args.boundary_profile),
        boundary_profile_strength=float(args.boundary_profile_strength),
        boundary_profile_seed=int(args.boundary_profile_seed),
        segment_length_power=float(args.segment_length_power),
        topic_concentration=float(args.topic_concentration),
        emission_mode=str(args.emission_mode),
        anchor_words_per_topic=int(args.anchor_words_per_topic),
        anchor_multiplier=float(args.anchor_multiplier),
        relevant_topics=int(args.relevant_topics),
        theta_scale=float(args.theta_scale),
        zero_diagonal=bool(args.zero_diagonal),
        lambda_multiplier=float(args.lambda_multiplier),
        oracle_noise_std=float(args.oracle_noise_std),
        audit_policy=str(args.audit_policy),
        audit_fixed_nodes=int(args.audit_fixed_nodes),
        audit_fraction=float(args.audit_fraction),
        audit_scale=float(args.audit_scale),
        audit_strategy=str(args.audit_strategy),
        oracle_cost_power=float(args.oracle_cost_power),
        oracle_cost_per_query=float(args.oracle_cost_per_query),
        ridge_lambda=float(args.ridge_lambda),
        topic_source=str(args.topic_source),
        feature_inference=str(args.feature_inference),
        topic_phi_estimator=str(args.topic_phi_estimator),
        topic_phi_docs=int(args.topic_phi_docs),
        tlda_delta=float(args.tlda_delta),
        tlda_rate_constant=float(args.tlda_rate_constant),
        tlda_sigmaK_floor=float(args.tlda_sigmaK_floor),
        topic_phi_permute=bool(args.topic_phi_permute),
        online_tensor_lda_burn_in_docs=int(args.online_tensor_lda_burn_in_docs),
        online_tensor_lda_batch_docs=int(args.online_tensor_lda_batch_docs),
        online_tensor_lda_passes=int(args.online_tensor_lda_passes),
        online_tensor_lda_lr=float(args.online_tensor_lda_lr),
        online_tensor_lda_grad_clip_norm=float(args.online_tensor_lda_grad_clip_norm),
        embedding_topic_svd_dim_extra=int(args.embedding_topic_svd_dim_extra),
        embedding_topic_kmeans_inits=int(args.embedding_topic_kmeans_inits),
        embedding_topic_kmeans_max_iter=int(args.embedding_topic_kmeans_max_iter),
        embedding_topic_assignment_temperature=float(args.embedding_topic_assignment_temperature),
        embedding_topic_ppmi_shift=float(args.embedding_topic_ppmi_shift),
        neural_topic_base_estimator=str(args.neural_topic_base_estimator),
        neural_topic_seed_fraction=float(args.neural_topic_seed_fraction),
        neural_topic_hidden_dim=int(args.neural_topic_hidden_dim),
        neural_topic_steps=int(args.neural_topic_steps),
        neural_topic_lr=float(args.neural_topic_lr),
        neural_topic_weight_decay=float(args.neural_topic_weight_decay),
        neural_topic_mix_samples=int(args.neural_topic_mix_samples),
        neural_topic_mix_temperature=float(args.neural_topic_mix_temperature),
        neural_topic_operator_boost=float(args.neural_topic_operator_boost),
        neural_topic_seed_llm_min_weight=float(args.neural_topic_seed_llm_min_weight),
        neural_topic_seed_llm_max_weight=float(args.neural_topic_seed_llm_max_weight),
        neural_topic_similarity_temperature=float(args.neural_topic_similarity_temperature),
        neural_topic_ridge=float(args.neural_topic_ridge),
        run_all_feature_modes=bool(args.run_all_feature_modes),
        device=str(args.device),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
        violation_tau=float(args.violation_tau),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        seed=int(args.seed),
    )

    summary = run_segment_lda_ops_weight_recovery_experiment(cfg)

    json_path = Path(args.json_summary)
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
