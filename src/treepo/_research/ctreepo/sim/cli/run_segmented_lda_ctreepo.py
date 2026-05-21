#!/usr/bin/env python3
"""Run segmented-LDA end-to-end C-TreePO simulation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from treepo._research.ctreepo.sim.core.segmented_lda_ctreepo import (
    SegmentedLDACtreePOConfig,
    VALID_DEVICE_MODES,
    VALID_TOPIC_PHI_ESTIMATORS,
    run_segmented_lda_ctreepo_simulation,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run segmented-LDA end-to-end C-TreePO simulation.")

    p.add_argument("--n-topics", type=int, default=5)
    p.add_argument("--vocab-size", type=int, default=600)
    p.add_argument("--alpha-topic", type=float, default=0.20)
    p.add_argument("--beta-word", type=float, default=0.10)
    p.add_argument("--topic-process", choices=["segments", "bag_of_words"], default="segments")

    p.add_argument("--n-books-train", type=int, default=256)
    p.add_argument("--n-books-test", type=int, default=256)
    p.add_argument("--min-segments", type=int, default=8)
    p.add_argument("--max-segments", type=int, default=20)
    p.add_argument("--min-seg-tokens", type=int, default=24)
    p.add_argument("--max-seg-tokens", type=int, default=64)
    p.add_argument("--segment-concentration", type=float, default=80.0)
    p.add_argument("--segment-background", type=float, default=2.0)
    p.add_argument("--fixed-leaf-tokens", type=int, default=32)

    p.add_argument("--leaf-theta-estimator", choices=["lstsq", "rf", "mlp", "sklearn_lda"], default="lstsq")
    p.add_argument("--leaf-theta-rf-n-estimators", type=int, default=200)
    p.add_argument("--leaf-theta-rf-max-depth", type=int, default=16)
    p.add_argument("--leaf-theta-rf-min-samples-leaf", type=int, default=5)
    p.add_argument("--leaf-theta-mlp-hidden-dim", type=int, default=128)
    p.add_argument("--leaf-theta-mlp-epochs", type=int, default=10)
    p.add_argument("--leaf-theta-mlp-batch-size", type=int, default=256)
    p.add_argument("--leaf-theta-mlp-lr", type=float, default=1e-3)
    p.add_argument("--leaf-theta-mlp-weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--include-full-doc-theta-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, also fit the matched no-tree topic-mixture baseline on full-document counts "
            "for supervised theta estimators."
        ),
    )

    p.add_argument(
        "--topic-phi-estimator",
        choices=list(VALID_TOPIC_PHI_ESTIMATORS),
        default="noisy_theory",
    )
    p.add_argument("--topic-phi-docs", type=int, default=0)
    p.add_argument("--tlda-delta", type=float, default=0.10)
    p.add_argument("--tlda-rate-constant", type=float, default=1.0)
    p.add_argument("--tlda-sigmaK-floor", type=float, default=1e-6)
    p.add_argument(
        "--topic-phi-permute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, randomly permute estimated topics (identifiability is up to permutation).",
    )
    p.add_argument("--online-tensor-lda-burn-in-docs", type=int, default=0)
    p.add_argument("--online-tensor-lda-batch-docs", type=int, default=32)
    p.add_argument("--online-tensor-lda-passes", type=int, default=1)
    p.add_argument("--online-tensor-lda-lr", type=float, default=0.1)
    p.add_argument("--online-tensor-lda-grad-clip-norm", type=float, default=1.0)
    p.add_argument("--embedding-topic-svd-dim-extra", type=int, default=4)
    p.add_argument("--embedding-topic-kmeans-inits", type=int, default=8)
    p.add_argument("--embedding-topic-kmeans-max-iter", type=int, default=80)
    p.add_argument("--embedding-topic-assignment-temperature", type=float, default=0.35)
    p.add_argument("--embedding-topic-ppmi-shift", type=float, default=1.0)
    p.add_argument("--neural-topic-base-estimator", type=str, default="tensor_lda")
    p.add_argument("--neural-topic-seed-fraction", type=float, default=0.35)
    p.add_argument("--neural-topic-hidden-dim", type=int, default=48)
    p.add_argument("--neural-topic-steps", type=int, default=60)
    p.add_argument("--neural-topic-lr", type=float, default=3e-3)
    p.add_argument("--neural-topic-weight-decay", type=float, default=1e-4)
    p.add_argument("--neural-topic-mix-samples", type=int, default=128)
    p.add_argument("--neural-topic-mix-temperature", type=float, default=1.0)
    p.add_argument("--neural-topic-operator-boost", type=float, default=1.4)
    p.add_argument("--neural-topic-seed-llm-min-weight", type=float, default=0.2)
    p.add_argument("--neural-topic-seed-llm-max-weight", type=float, default=0.55)
    p.add_argument("--neural-topic-similarity-temperature", type=float, default=0.15)
    p.add_argument("--neural-topic-ridge", type=float, default=1e-3)
    p.add_argument("--spectral-svd-dim-extra", type=int, default=2)
    p.add_argument("--spectral-max-leaves", type=int, default=4000)
    p.add_argument("--spectral-kmeans-inits", type=int, default=6)
    p.add_argument("--spectral-kmeans-max-iter", type=int, default=60)

    p.add_argument("--calibration-leaf-query-rate", type=float, default=0.10)
    p.add_argument("--calibration-policy", choices=["uniform", "entropy"], default="uniform")
    p.add_argument("--calibration-ridge", type=float, default=1e-4)
    p.add_argument("--calibration-pi-min", type=float, default=0.01)

    p.add_argument("--eval-leaf-query-rate", type=float, default=0.00)
    p.add_argument("--eval-internal-query-rate", type=float, default=0.00)
    p.add_argument("--eval-internal-query-design", choices=["none", "uniform", "risk"], default="none")

    p.add_argument("--c1-threshold", type=float, default=0.20)
    p.add_argument("--c3-threshold", type=float, default=0.20)

    p.add_argument("--selection-audit-trials", type=int, default=0)
    p.add_argument("--selection-audit-sample-rate", type=float, default=0.10)
    p.add_argument("--selection-audit-pi-min", type=float, default=0.01)
    p.add_argument("--device", type=str, choices=list(VALID_DEVICE_MODES), default="auto")
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=0)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--json-summary",
        type=str,
        default="outputs/segmented_lda_ctreepo/seed_0.json",
    )
    p.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/segmented_lda_ctreepo/seed_0.csv",
    )
    p.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    return p.parse_args(list(argv) if argv is not None else None)


def _write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = SegmentedLDACtreePOConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        alpha_topic=float(args.alpha_topic),
        beta_word=float(args.beta_word),
        topic_process=str(args.topic_process),
        n_books_train=int(args.n_books_train),
        n_books_test=int(args.n_books_test),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_tokens=int(args.min_seg_tokens),
        max_seg_tokens=int(args.max_seg_tokens),
        segment_concentration=float(args.segment_concentration),
        segment_background=float(args.segment_background),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        leaf_theta_estimator=str(args.leaf_theta_estimator),
        leaf_theta_rf_n_estimators=int(args.leaf_theta_rf_n_estimators),
        leaf_theta_rf_max_depth=int(args.leaf_theta_rf_max_depth),
        leaf_theta_rf_min_samples_leaf=int(args.leaf_theta_rf_min_samples_leaf),
        leaf_theta_mlp_hidden_dim=int(args.leaf_theta_mlp_hidden_dim),
        leaf_theta_mlp_epochs=int(args.leaf_theta_mlp_epochs),
        leaf_theta_mlp_batch_size=int(args.leaf_theta_mlp_batch_size),
        leaf_theta_mlp_lr=float(args.leaf_theta_mlp_lr),
        leaf_theta_mlp_weight_decay=float(args.leaf_theta_mlp_weight_decay),
        include_full_doc_theta_baseline=bool(args.include_full_doc_theta_baseline),
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
        spectral_svd_dim_extra=int(args.spectral_svd_dim_extra),
        spectral_max_leaves=int(args.spectral_max_leaves),
        spectral_kmeans_inits=int(args.spectral_kmeans_inits),
        spectral_kmeans_max_iter=int(args.spectral_kmeans_max_iter),
        calibration_leaf_query_rate=float(args.calibration_leaf_query_rate),
        calibration_policy=str(args.calibration_policy),
        calibration_ridge=float(args.calibration_ridge),
        calibration_pi_min=float(args.calibration_pi_min),
        eval_leaf_query_rate=float(args.eval_leaf_query_rate),
        eval_internal_query_rate=float(args.eval_internal_query_rate),
        eval_internal_query_design=str(args.eval_internal_query_design),
        c1_threshold=float(args.c1_threshold),
        c3_threshold=float(args.c3_threshold),
        selection_audit_trials=int(args.selection_audit_trials),
        selection_audit_sample_rate=float(args.selection_audit_sample_rate),
        selection_audit_pi_min=float(args.selection_audit_pi_min),
        device=str(args.device),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
        seed=int(args.seed),
    )

    out = run_segmented_lda_ctreepo_simulation(cfg)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(out.to_json(), encoding="utf-8")

    row = {f"config_{k}": v for k, v in out.config.items()}
    row.update({f"topic_{k}": v for k, v in out.topic_meta.items()})
    row["calibration_samples"] = out.calibration_samples
    for policy, metrics in out.metrics.items():
        for k, v in asdict(metrics).items():
            row[f"{policy}_{k}"] = v
    for k, v in asdict(out.decomposition).items():
        row[f"decomposition_{k}"] = v
    if out.selection_audit is not None:
        row["selection_audit_trials"] = out.selection_audit.trials
        row["selection_audit_mean_sample_size"] = out.selection_audit.mean_sample_size
        row["selection_audit_mean_effective_sample_size"] = out.selection_audit.mean_effective_sample_size
        row["selection_audit_ipw_ci_coverage"] = out.selection_audit.ipw_violation_ci_coverage
        row["selection_audit_ipw_ci_mean_radius"] = out.selection_audit.ipw_violation_ci_mean_radius

    csv_path = Path(args.csv_summary)
    _write_csv(csv_path, row)

    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {csv_path}")
    topic_l2 = float(out.topic_meta.get("topic_phi_l2_error_mean", float("nan")))
    print(f"topic_phi_estimator={cfg.topic_phi_estimator} | topic_phi_l2_error_mean={topic_l2:.6f}")
    print(f"calibration_samples={out.calibration_samples}")
    print(
        "decomposition | total={:.4f} | topic={:.4f} | calib={:.4f} | guidance={:.4f} | oracle_proxy={:.4f} | upper={:.4f} | slack={:.4f}".format(
            out.decomposition.total_root_l1_mean,
            out.decomposition.topic_component_mean,
            out.decomposition.calibration_component_mean,
            out.decomposition.guidance_component_mean,
            out.decomposition.oracle_proxy_component_mean,
            out.decomposition.upper_bound_mean,
            out.decomposition.slack_mean,
        )
    )
    for name, m in out.metrics.items():
        print(
            "policy={} | root_l1={:.4f} | c1={:.4f} | c3={:.4f} | q_leaf={:.2f} | q_internal={:.2f}".format(
                name,
                m.root_l1_mean,
                m.c1_violation_rate,
                m.c3_violation_rate,
                m.mean_leaf_queries,
                m.mean_internal_queries,
            )
        )

    if bool(args.json):
        print(out.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
