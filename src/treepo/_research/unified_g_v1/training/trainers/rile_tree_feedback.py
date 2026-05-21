"""Tree-structured per-node feedback for LLM RILE training.

The live DSPy-GEPA trainer and the TRL-GRPO reward wrappers consume the same
shape of per-rollout data:

* a tree scaffold (leaf char spans + per-span RILE targets from the
  Manifesto Project codings)
* per-node RILE predictions from the student model (applied at every
  leaf, every merge, and at the root; plus the student's prediction on a
  swapped-merge probe for C2 commutativity)

This module converts that data into:

* `ScoreWithFeedback(score, feedback)` — what DSPy-GEPA's reflection LM
  consumes (`gepa_metric` return value).
* `list[float]` — what TRL's `reward_funcs=[...]` expects (per-completion
  reward in [0, 1]).

The actual LLM calls that produce per-node predictions live in the trainer.
Keeping the metric pure here means it's unit-testable without a running vLLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from treepo._research.unified_g_v1.realdoc.rile_tree import (
    DEFAULT_LOCAL_LAW_WEIGHT,
    PerNodeRilePrediction,
    RILETreeScaffold,
    RileTreeReward,
    rile_tree_reward,
)


@dataclass(frozen=True)
class TreeRilePredictions:
    """What the student model produces on one rollout of one document."""

    scaffold: RILETreeScaffold
    root_prediction: float
    leaf_predictions: tuple[PerNodeRilePrediction, ...] = ()
    merge_predictions: tuple[PerNodeRilePrediction, ...] = ()
    commutativity_pairs: tuple[tuple[float, float], ...] = ()


def tree_rile_reward_for_rollout(
    rollout: TreeRilePredictions,
    *,
    local_law_weight: float = DEFAULT_LOCAL_LAW_WEIGHT,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
) -> RileTreeReward:
    """Thin wrapper: one rollout → compound reward + feedback string.

    `local_law_weight` (λ, default 0.3) balances root vs. local laws;
    `*_relative_weight` knobs balance the three local laws against each
    other (default equal). See `rile_tree_reward` for the full formula.
    """
    return rile_tree_reward(
        scaffold=rollout.scaffold,
        root_prediction=float(rollout.root_prediction),
        leaf_predictions=rollout.leaf_predictions,
        merge_predictions=rollout.merge_predictions,
        commutativity_pairs=rollout.commutativity_pairs,
        local_law_weight=local_law_weight,
        c1_relative_weight=c1_relative_weight,
        c2_relative_weight=c2_relative_weight,
        c3_relative_weight=c3_relative_weight,
    )


def dspy_gepa_metric_from_rollout(
    rollout: TreeRilePredictions,
    *,
    local_law_weight: float = DEFAULT_LOCAL_LAW_WEIGHT,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
) -> Any:
    """Return a `ScoreWithFeedback` for GEPA reflection.

    Imports `ScoreWithFeedback` lazily so consumers that don't use DSPy
    aren't forced to install it.
    """
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    reward = tree_rile_reward_for_rollout(
        rollout,
        local_law_weight=local_law_weight,
        c1_relative_weight=c1_relative_weight,
        c2_relative_weight=c2_relative_weight,
        c3_relative_weight=c3_relative_weight,
    )
    return ScoreWithFeedback(score=float(reward.score), feedback=str(reward.feedback))


def trl_grpo_rewards_from_rollouts(
    rollouts: Sequence[TreeRilePredictions],
    *,
    local_law_weight: float = DEFAULT_LOCAL_LAW_WEIGHT,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
) -> list[float]:
    """Return a per-rollout reward list suitable as a TRL GRPO reward function.

    TRL GRPO's `reward_funcs` is a list of callables taking `(prompts,
    completions, **kwargs)` and returning `list[float]`. The actual
    integration point (pairing completions to scaffolds + per-node
    predictions) lives in the trainer; this helper handles the
    rollout-to-reward conversion.
    """
    return [
        float(
            tree_rile_reward_for_rollout(
                rollout,
                local_law_weight=local_law_weight,
                c1_relative_weight=c1_relative_weight,
                c2_relative_weight=c2_relative_weight,
                c3_relative_weight=c3_relative_weight,
            ).score
        )
        for rollout in rollouts
    ]
