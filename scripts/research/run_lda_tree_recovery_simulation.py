#!/usr/bin/env python3
"""Run the exact bag-of-words LDA tree-recovery simulation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from treepo._research.ctreepo.sim.core.lda_tree_recovery import (  # noqa: E402
    LDATreeRecoveryConfig,
    VALID_EMISSION_MODES,
    run_lda_tree_recovery_experiment,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run exact bag-of-words LDA tree-recovery simulation.")

    # LDA DGP.
    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=384)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--doc-topic-concentration", type=float, default=0.6)

    # Topic-word distributions.
    p.add_argument("--topic-concentration", type=float, default=0.2)
    p.add_argument("--emission-mode", type=str, choices=list(VALID_EMISSION_MODES), default="anchored")
    p.add_argument("--anchor-words-per-topic", type=int, default=20)
    p.add_argument("--anchor-multiplier", type=float, default=25.0)

    # Utility on inferred topic mixtures.
    p.add_argument("--relevant-topics", type=int, default=2)
    p.add_argument("--theta-scale", type=float, default=1.0)
    p.add_argument(
        "--zero-diagonal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, suppress diagonal terms in the quadratic topic utility.",
    )
    p.add_argument(
        "--quadratic-utility-weight",
        "--lambda-multiplier",
        dest="quadratic_utility_weight",
        type=float,
        default=1.0,
        help="Quadratic utility weight (legacy alias: --lambda-multiplier).",
    )

    # Tree geometry.
    p.add_argument("--leaf-tokens", type=int, default=16)

    # Fixed-world sizes.
    p.add_argument("--train-docs", type=int, default=0)
    p.add_argument("--test-docs", type=int, default=1024)

    # Known-topic document-mixture inference.
    p.add_argument("--inference-prior-mass", type=float, default=0.25)
    p.add_argument("--inference-max-iter", type=int, default=200)
    p.add_argument("--inference-tol", type=float, default=1e-9)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--json-summary", type=str, required=True)
    p.add_argument("--csv-summary", type=str, required=True)
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout as well.")
    return p.parse_args(list(argv) if argv is not None else None)


def _rows_from_summary(summary) -> List[dict]:
    cfg = dict(summary.config)
    world_stats = dict(summary.world_stats)
    exact = dict(summary.exact_recovery)
    rows: List[dict] = []
    methods = summary.methods if isinstance(summary.methods, dict) else {}
    for method, metrics in methods.items():
        if not isinstance(metrics, dict):
            continue
        row = {
            "method": str(method),
            **{f"cfg_{k}": v for k, v in cfg.items()},
            **{f"world_{k}": v for k, v in world_stats.items()},
            **{f"exact_{k}": v for k, v in exact.items()},
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = LDATreeRecoveryConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        doc_topic_concentration=float(args.doc_topic_concentration),
        topic_concentration=float(args.topic_concentration),
        emission_mode=str(args.emission_mode),
        anchor_words_per_topic=int(args.anchor_words_per_topic),
        anchor_multiplier=float(args.anchor_multiplier),
        relevant_topics=int(args.relevant_topics),
        theta_scale=float(args.theta_scale),
        zero_diagonal=bool(args.zero_diagonal),
        lambda_multiplier=float(args.quadratic_utility_weight),
        leaf_tokens=int(args.leaf_tokens),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        inference_prior_mass=float(args.inference_prior_mass),
        inference_max_iter=int(args.inference_max_iter),
        inference_tol=float(args.inference_tol),
        seed=int(args.seed),
    )

    summary = run_lda_tree_recovery_experiment(cfg)

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(summary.to_json(), encoding="utf-8")

    csv_path = Path(args.csv_summary)
    _write_csv(csv_path, _rows_from_summary(summary))

    exact = dict(summary.exact_recovery)
    methods = summary.methods
    full_doc = methods.get("full_doc", {}) if isinstance(methods, dict) else {}
    exact_tree = methods.get("exact_tree", {}) if isinstance(methods, dict) else {}
    leaf_average = methods.get("leaf_average", {}) if isinstance(methods, dict) else {}
    leaf_u = methods.get("leaf_utility_only", {}) if isinstance(methods, dict) else {}

    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {csv_path}")
    print(
        "exact_recovery | count_l1={:.3e} | pi_l1={:.3e} | utility_abs={:.3e} | loglik_abs={:.3e}".format(
            float(exact.get("root_count_l1_mean", float("nan"))),
            float(exact.get("root_pi_l1_mean", float("nan"))),
            float(exact.get("root_utility_abs_mean", float("nan"))),
            float(exact.get("root_loglik_abs_mean", float("nan"))),
        )
    )
    print(
        "full_doc | pi_l1_to_true={:.4f} | utility_abs_to_true={:.4f}".format(
            float(full_doc.get("pi_l1_to_true_mean", float("nan"))),
            float(full_doc.get("utility_abs_to_true_mean", float("nan"))),
        )
    )
    print(
        "exact_tree | pi_l1_to_full={:.3e} | utility_abs_to_full={:.3e}".format(
            float(exact_tree.get("pi_l1_to_full_mean", float("nan"))),
            float(exact_tree.get("utility_abs_to_full_mean", float("nan"))),
        )
    )
    print(
        "leaf_average | pi_l1_to_full={:.4f} | utility_abs_to_full={:.4f}".format(
            float(leaf_average.get("pi_l1_to_full_mean", float("nan"))),
            float(leaf_average.get("utility_abs_to_full_mean", float("nan"))),
        )
    )
    print(
        "leaf_utility_only | utility_abs_to_full={:.4f}".format(
            float(leaf_u.get("utility_abs_to_full_mean", float("nan"))),
        )
    )

    if bool(args.json):
        print(summary.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
