from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

from treepo._research.ctreepo.sim.suite.policy_common import join_items, parse_float_list, parse_int_list


@dataclass(frozen=True)
class IdentifiableZeroLearnabilityPolicy:
    profile: str = "paper"
    train_docs_grid: Tuple[int, ...] = (500, 1000, 2000, 4000, 8000)
    label_rate_grid: Tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4)
    heldout_docs: int = 2000
    base_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    hero_seeds: Tuple[int, ...] = (6, 7, 8, 9, 10, 11)
    ctree_eval_guidance_rates: Tuple[float, ...] = (0.0,)
    markov_sampled_leaf_pool_leaf_counts: Tuple[int, ...] = tuple()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_shell_exports(self) -> dict[str, str]:
        return {
            "TRAIN_DOCS_GRID": join_items(self.train_docs_grid),
            "LABEL_RATE_GRID": join_items(self.label_rate_grid),
            "HELDOUT_DOCS": str(int(self.heldout_docs)),
            "BASE_SEEDS": join_items(self.base_seeds),
            "HERO_SEEDS": join_items(self.hero_seeds),
            "CTREE_EVAL_GUIDANCE_RATES": join_items(self.ctree_eval_guidance_rates),
            "MARKOV_SAMPLED_LEAF_POOL_LEAF_COUNTS": join_items(
                self.markov_sampled_leaf_pool_leaf_counts
            ),
        }


def resolve_identifiable_zero_learnability_policy(
    *,
    profile_name: str = "paper",
    train_docs_grid: str | None = None,
    label_rate_grid: str | None = None,
    heldout_docs: int | None = None,
    base_seeds: str | None = None,
    hero_seeds: str | None = None,
    ctree_eval_guidance_rates: str | None = None,
    markov_sampled_leaf_pool_leaf_counts: str | None = None,
) -> IdentifiableZeroLearnabilityPolicy:
    profile_key = str(profile_name or "paper").strip().lower() or "paper"
    if profile_key == "paper":
        defaults = IdentifiableZeroLearnabilityPolicy(profile="paper")
    elif profile_key == "smoke":
        defaults = IdentifiableZeroLearnabilityPolicy(
            profile="smoke",
            train_docs_grid=(16,),
            label_rate_grid=(0.1,),
            heldout_docs=16,
            base_seeds=(0,),
            markov_sampled_leaf_pool_leaf_counts=(1, 2, 4, 8),
        )
    else:
        raise ValueError(f"unknown learnability profile: {profile_name!r}")
    return IdentifiableZeroLearnabilityPolicy(
        profile=str(defaults.profile),
        train_docs_grid=parse_int_list(
            train_docs_grid,
            default=defaults.train_docs_grid,
        ),
        label_rate_grid=parse_float_list(
            label_rate_grid,
            default=defaults.label_rate_grid,
        ),
        heldout_docs=int(defaults.heldout_docs if heldout_docs is None else heldout_docs),
        base_seeds=parse_int_list(base_seeds, default=defaults.base_seeds),
        hero_seeds=parse_int_list(hero_seeds, default=defaults.hero_seeds),
        ctree_eval_guidance_rates=parse_float_list(
            ctree_eval_guidance_rates,
            default=defaults.ctree_eval_guidance_rates,
        ),
        markov_sampled_leaf_pool_leaf_counts=parse_int_list(
            markov_sampled_leaf_pool_leaf_counts,
            default=defaults.markov_sampled_leaf_pool_leaf_counts,
        ),
    )


__all__ = [
    "IdentifiableZeroLearnabilityPolicy",
    "resolve_identifiable_zero_learnability_policy",
]
