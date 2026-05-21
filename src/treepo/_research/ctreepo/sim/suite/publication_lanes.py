from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from treepo._research.ctreepo.sim.suite.policy_common import join_items
from treepo._research.ctreepo.sim.suite.publication_policy import PublicationCtreepoPolicy


@dataclass(frozen=True)
class PublicationLaneCall:
    key: str
    regime: str
    variant: str
    output_root: Path
    argv_base: Tuple[str, ...]
    fixed_leaf_tokens_grid: Tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "key": str(self.key),
            "regime": str(self.regime),
            "variant": str(self.variant),
            "output_root": str(self.output_root),
            "argv_base": list(self.argv_base),
            "fixed_leaf_tokens_grid": [int(x) for x in self.fixed_leaf_tokens_grid],
        }


def resolve_publication_lane_calls(
    *,
    output_root: Path,
    policy: PublicationCtreepoPolicy,
) -> Tuple[PublicationLaneCall, ...]:
    output_root = output_root.resolve()
    seeds = join_items(policy.seeds)
    q_rates = join_items(policy.q_rates)
    q_rates_upper = join_items(policy.q_rates_upper)
    train_docs_lda = join_items(policy.train_docs_lda)
    train_docs_hard = join_items(policy.train_docs_hard)
    train_docs_hard_upper = join_items(policy.train_docs_hard_upper)
    cal_rates_lda = join_items(policy.cal_rates_lda)
    cal_rates_hard = join_items(policy.cal_rates_hard)
    cal_rates_upper = join_items(policy.cal_rates_upper)
    n_books_test_lda = int(policy.n_books_test_lda)
    n_books_test_hard = int(policy.n_books_test_hard)
    doc_tokens_lda = int(policy.doc_tokens_lda)

    calls: list[PublicationLaneCall] = []

    for lane_key, variant, lane_args in [
        (
            "lda_lane_lda_direct",
            "lda_direct",
            (
                "--topic-process",
                "bag_of_words",
                "--leaf-theta-estimator",
                "sklearn_lda",
                "--topic-phi-estimators",
                "sklearn_lda",
            ),
        ),
        (
            "lda_lane_phi_base",
            "phi_base",
            (
                "--topic-process",
                "bag_of_words",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "tensor_lda",
            ),
        ),
        (
            "lda_lane_neural_weak",
            "neural_weak",
            (
                "--topic-process",
                "bag_of_words",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fraction",
                "0.125",
                "--neural-topic-operator-boost",
                "0.6",
                "--neural-topic-seed-llm-min-weight",
                "0.02",
                "--neural-topic-seed-llm-max-weight",
                "0.15",
                "--neural-topic-mix-samples",
                "64",
            ),
        ),
        (
            "lda_lane_neural_default",
            "neural_default",
            (
                "--topic-process",
                "bag_of_words",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fractions",
                "0.25 0.5",
                "--neural-topic-operator-boost",
                "1.0",
                "--neural-topic-seed-llm-min-weight",
                "0.10",
                "--neural-topic-seed-llm-max-weight",
                "0.35",
                "--neural-topic-mix-samples",
                "128",
            ),
        ),
    ]:
        calls.append(
            PublicationLaneCall(
                key=lane_key,
                regime="lda",
                variant=variant,
                output_root=(
                    output_root
                    / "segmented_lda_ctreepo"
                    / "equivalence"
                    / "lda"
                    / "k8_v512"
                    / f"lane_{variant}"
                ),
                argv_base=(
                    "--output-root",
                    str(
                        output_root
                        / "segmented_lda_ctreepo"
                        / "equivalence"
                        / "lda"
                        / "k8_v512"
                        / f"lane_{variant}"
                    ),
                    "--train-docs",
                    str(train_docs_lda),
                    "--n-books-test",
                    str(n_books_test_lda),
                    "--calibration-rates",
                    str(cal_rates_lda),
                    "--eval-leaf-rates",
                    str(q_rates),
                    "--eval-internal-rates",
                    str(q_rates),
                    "--topic-phi-docs",
                    "0",
                    "--n-topics",
                    "8",
                    "--vocab-size",
                    "512",
                    "--min-segments",
                    "1",
                    "--max-segments",
                    "1",
                    "--min-seg-tokens",
                    str(doc_tokens_lda),
                    "--max-seg-tokens",
                    str(doc_tokens_lda),
                    "--alpha-topic",
                    "0.20",
                    "--beta-word",
                    "0.10",
                    "--segment-concentration",
                    "80.0",
                    "--segment-background",
                    "2.0",
                    "--topic-phi-permute",
                    "--eval-internal-query-design",
                    "risk",
                    "--seeds",
                    str(seeds),
                    *lane_args,
                ),
                fixed_leaf_tokens_grid=tuple(int(x) for x in policy.leaf_tokens_lda),
            )
        )

    for lane_key, variant, lane_args, train_docs, cal_rates, q_grid in [
        (
            "hard_lane_phi_base",
            "phi_base",
            (
                "--topic-process",
                "segments",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "tensor_lda",
            ),
            train_docs_hard,
            cal_rates_hard,
            q_rates,
        ),
        (
            "hard_lane_neural_weak",
            "neural_weak",
            (
                "--topic-process",
                "segments",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fraction",
                "0.0833333333",
                "--neural-topic-operator-boost",
                "0.6",
                "--neural-topic-seed-llm-min-weight",
                "0.02",
                "--neural-topic-seed-llm-max-weight",
                "0.15",
                "--neural-topic-mix-samples",
                "64",
            ),
            train_docs_hard,
            cal_rates_hard,
            q_rates,
        ),
        (
            "hard_lane_neural_default",
            "neural_default",
            (
                "--topic-process",
                "segments",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fractions",
                "0.2 0.35",
                "--neural-topic-operator-boost",
                "1.0",
                "--neural-topic-seed-llm-min-weight",
                "0.10",
                "--neural-topic-seed-llm-max-weight",
                "0.35",
                "--neural-topic-mix-samples",
                "128",
            ),
            train_docs_hard,
            cal_rates_hard,
            q_rates,
        ),
        (
            "hard_lane_neural_upper",
            "neural_upper",
            (
                "--topic-process",
                "segments",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fraction",
                "1.0",
                "--neural-topic-operator-boost",
                "1.4",
                "--neural-topic-seed-llm-min-weight",
                "0.35",
                "--neural-topic-seed-llm-max-weight",
                "0.85",
                "--neural-topic-mix-samples",
                "128",
            ),
            train_docs_hard_upper,
            cal_rates_upper,
            q_rates_upper,
        ),
    ]:
        calls.append(
            PublicationLaneCall(
                key=lane_key,
                regime="hard",
                variant=variant,
                output_root=(
                    output_root
                    / "segmented_lda_ctreepo"
                    / "equivalence"
                    / "hard"
                    / "k12_v1024"
                    / f"lane_{variant}"
                ),
                argv_base=(
                    "--output-root",
                    str(
                        output_root
                        / "segmented_lda_ctreepo"
                        / "equivalence"
                        / "hard"
                        / "k12_v1024"
                        / f"lane_{variant}"
                    ),
                    "--train-docs",
                    str(train_docs),
                    "--n-books-test",
                    str(n_books_test_hard),
                    "--calibration-rates",
                    str(cal_rates),
                    "--eval-leaf-rates",
                    str(q_grid),
                    "--eval-internal-rates",
                    str(q_grid),
                    "--topic-phi-docs",
                    "0",
                    "--n-topics",
                    "12",
                    "--vocab-size",
                    "1024",
                    "--min-segments",
                    "10",
                    "--max-segments",
                    "12",
                    "--min-seg-tokens",
                    "16",
                    "--max-seg-tokens",
                    "32",
                    "--alpha-topic",
                    "0.35",
                    "--beta-word",
                    "0.40",
                    "--segment-concentration",
                    "18.0",
                    "--segment-background",
                    "6.0",
                    "--topic-phi-permute",
                    "--eval-internal-query-design",
                    "risk",
                    "--seeds",
                    str(seeds),
                    *lane_args,
                ),
                fixed_leaf_tokens_grid=tuple(int(x) for x in policy.leaf_tokens_hard),
            )
        )

    return tuple(calls)


__all__ = ["PublicationLaneCall", "resolve_publication_lane_calls"]
