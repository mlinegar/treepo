#!/usr/bin/env python3
"""Run Tensor-LDA DGP "books" benchmark for TLDA vs C-TreePO-style comparisons."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from treepo._research.ctreepo.sim.core.tensor_lda_book_benchmark import (
    TensorLDABookBenchmarkConfig,
    run_tensor_lda_book_weight_benchmark,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Tensor-LDA DGP benchmark for ThinkingTrees comparisons.")

    # LDA DGP.
    p.add_argument("--n-topics", type=int, default=5)
    p.add_argument("--vocab-size", type=int, default=600)
    p.add_argument("--chapters-per-book", type=int, default=16)
    p.add_argument("--tokens-per-chapter", type=int, default=128)
    p.add_argument("--alpha-topic", type=float, default=0.20)
    p.add_argument("--beta-word", type=float, default=0.10)
    p.add_argument("--chapter-concentration", type=float, default=40.0)

    # Dataset sizes.
    p.add_argument("--n-books-train", type=int, default=256)
    p.add_argument("--n-books-test", type=int, default=256)

    # Proxy settings.
    p.add_argument("--anchor-words-per-topic", type=int, default=20)
    p.add_argument("--proxy-temperature", type=float, default=0.50)
    p.add_argument("--proxy-noise-std", type=float, default=0.05)

    # Calibration settings.
    p.add_argument("--calibration-leaf-query-rate", type=float, default=0.10)
    p.add_argument("--calibration-policy", choices=["uniform", "entropy"], default="uniform")
    p.add_argument("--calibration-ridge", type=float, default=1e-4)
    p.add_argument("--calibration-pi-min", type=float, default=0.01)

    # Eval-time guidance settings.
    p.add_argument("--eval-leaf-query-rate", type=float, default=0.0)
    p.add_argument("--eval-internal-query-rate", type=float, default=0.0)
    p.add_argument("--eval-internal-query-design", choices=["none", "uniform", "risk"], default="none")

    # Thresholds and optional audit.
    p.add_argument("--c1-threshold", type=float, default=0.20)
    p.add_argument("--c3-threshold", type=float, default=0.20)
    p.add_argument("--selection-audit-trials", type=int, default=0)
    p.add_argument("--selection-audit-sample-rate", type=float, default=0.10)
    p.add_argument("--selection-audit-pi-min", type=float, default=0.01)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--json-summary",
        type=str,
        default="outputs/tensor_lda_book_weight_benchmark/seed_0.json",
    )
    p.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/tensor_lda_book_weight_benchmark/seed_0.csv",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout in addition to saving files.")
    return p.parse_args(list(argv) if argv is not None else None)


def _write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = TensorLDABookBenchmarkConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        chapters_per_book=int(args.chapters_per_book),
        tokens_per_chapter=int(args.tokens_per_chapter),
        alpha_topic=float(args.alpha_topic),
        beta_word=float(args.beta_word),
        chapter_concentration=float(args.chapter_concentration),
        n_books_train=int(args.n_books_train),
        n_books_test=int(args.n_books_test),
        anchor_words_per_topic=int(args.anchor_words_per_topic),
        proxy_temperature=float(args.proxy_temperature),
        proxy_noise_std=float(args.proxy_noise_std),
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
        seed=int(args.seed),
    )

    summary = run_tensor_lda_book_weight_benchmark(cfg)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")

    row = {f"config_{k}": v for k, v in summary.config.items()}
    row["calibration_samples"] = summary.calibration_samples
    for policy, metrics in summary.metrics.items():
        for k, v in asdict(metrics).items():
            row[f"{policy}_{k}"] = v
    if summary.selection_audit is not None:
        row["selection_audit_trials"] = summary.selection_audit.trials
        row["selection_audit_mean_sample_size"] = summary.selection_audit.mean_sample_size
        row["selection_audit_mean_effective_sample_size"] = summary.selection_audit.mean_effective_sample_size
        row["selection_audit_ipw_ci_coverage"] = summary.selection_audit.ipw_violation_ci_coverage
        row["selection_audit_ipw_ci_mean_radius"] = summary.selection_audit.ipw_violation_ci_mean_radius

    csv_path = Path(args.csv_summary)
    _write_csv(csv_path, row)

    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {csv_path}")
    print(f"calibration_samples={summary.calibration_samples}")
    for key, m in summary.metrics.items():
        print(
            "policy={} | root_l1={:.4f} | root_l2={:.4f} | c1={:.4f} | c3={:.4f} | q_leaf={:.2f} | q_internal={:.2f}".format(
                key,
                m.root_l1_mean,
                m.root_l2_mean,
                m.c1_violation_rate,
                m.c3_violation_rate,
                m.mean_leaf_queries,
                m.mean_internal_queries,
            )
        )

    if bool(args.json):
        print(summary.to_json())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

