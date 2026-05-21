from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class IdentifiableZeroPolicy:
    profile: str
    segment_train_docs: Tuple[int, ...]
    segment_audit_fractions: Tuple[float, ...]
    segment_lambda_multipliers: Tuple[float, ...]
    segment_seeds: Tuple[int, ...]
    ctree_train_docs: Tuple[int, ...]
    ctree_calibration_rates: Tuple[float, ...]
    ctree_eval_leaf_rates: Tuple[float, ...]
    ctree_eval_internal_rates: Tuple[float, ...]
    ctree_seeds: Tuple[int, ...]
    markov_train_docs: Tuple[int, ...]
    markov_audit_fractions: Tuple[float, ...]
    markov_seeds: Tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_identifiable_zero_policy(profile: str) -> IdentifiableZeroPolicy:
    profile_name = str(profile).strip().lower() or "walk_long"
    if profile_name == "smoke":
        return IdentifiableZeroPolicy(
            profile="smoke",
            segment_train_docs=(200, 500),
            segment_audit_fractions=(0.1, 0.2, 0.5, 1.0),
            segment_lambda_multipliers=(0.0, 1.0),
            segment_seeds=(0, 1),
            ctree_train_docs=(128, 256),
            ctree_calibration_rates=(0.0, 0.1, 1.0),
            ctree_eval_leaf_rates=(0.0, 1.0),
            ctree_eval_internal_rates=(0.0, 0.5, 1.0),
            ctree_seeds=(0, 1),
            markov_train_docs=(200, 500),
            markov_audit_fractions=(0.1, 0.2, 0.5, 1.0),
            markov_seeds=(0, 1),
        )
    if profile_name == "paper":
        return IdentifiableZeroPolicy(
            profile="paper",
            segment_train_docs=(100, 200, 500, 1000, 2000),
            segment_audit_fractions=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
            segment_lambda_multipliers=(0.0, 0.25, 1.0),
            segment_seeds=(0, 1, 2, 3, 4, 5, 6, 7),
            ctree_train_docs=(64, 128, 256, 512, 1024),
            ctree_calibration_rates=(0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
            ctree_eval_leaf_rates=(0.0, 0.5, 1.0),
            ctree_eval_internal_rates=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
            ctree_seeds=(0, 1, 2, 3, 4, 5, 6, 7),
            markov_train_docs=(100, 200, 500, 1000, 2000),
            markov_audit_fractions=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
            markov_seeds=(0, 1, 2, 3, 4, 5, 6, 7),
        )
    if profile_name == "walk_long":
        return IdentifiableZeroPolicy(
            profile="walk_long",
            segment_train_docs=(100, 200, 500, 1000, 2000, 4000),
            segment_audit_fractions=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
            segment_lambda_multipliers=(0.0, 0.25, 1.0),
            segment_seeds=tuple(range(16)),
            ctree_train_docs=(64, 128, 256, 512, 1024),
            ctree_calibration_rates=(0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
            ctree_eval_leaf_rates=(0.0, 0.25, 0.5, 1.0),
            ctree_eval_internal_rates=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
            ctree_seeds=tuple(range(16)),
            markov_train_docs=(100, 200, 500, 1000, 2000, 4000),
            markov_audit_fractions=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
            markov_seeds=tuple(range(16)),
        )
    raise ValueError(f"unknown identifiable-zero profile: {profile}")


__all__ = ["IdentifiableZeroPolicy", "resolve_identifiable_zero_policy"]
