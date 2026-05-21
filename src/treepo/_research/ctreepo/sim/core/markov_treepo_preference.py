from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from treepo._research.ctreepo.sim.core.exact_utility_common import (
    ExactUtilityDGP,
    ExactUtilityMetrics,
    ExactUtilityRunConfig,
    ExactUtilitySummary,
    UtilityDoc,
    lean_theorem_refs,
    predict_action_flat,
    predict_action_tree,
    realized_budget_dict,
    set_global_seed,
    summarize_predictions,
    train_flat_policy,
    train_neural_tree_policy,
)
from treepo._research.ctreepo.sim.objective_semantics import preference_training_objective_semantics
from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    _ExactState,
    _count_only_from_span,
    _count_only_merge,
    _exact_from_span,
    _exact_merge,
    _leaf_spans,
    _span_features,
)
from treepo._research.tree.markov_boundary_honesty_simulation import _make_transition_matrices
from treepo._research.tree.markov_changepoint_honesty_simulation import (
    MarkovChangepointConfig as _GeneratorConfig,
    generate_changepoint_docs,
)


@dataclass(frozen=True, kw_only=True)
class MarkovExactUtilityConfig(ExactUtilityRunConfig):
    lane: str = "markov"
    oracle_profile: str = "markov_count_endpoints"
    structural_arm: str = "tree_neural_supported"
    n_regimes: int = 4
    vocab_size: int = 96
    min_tokens: int = 128
    max_tokens: int = 128
    min_segments: int = 8
    max_segments: int = 16
    min_seg_len: int = 4
    max_seg_len: int = 16
    transition_log_std: float = 1.25
    sinkhorn_iters: int = 20


class MarkovExactUtilityDGP(ExactUtilityDGP[np.ndarray]):
    lane = "markov"

    def __init__(self, *, oracle_profile: str, n_regimes: int, max_count: int) -> None:
        self.oracle_profile = str(oracle_profile)
        self.n_regimes = int(n_regimes)
        self.max_count = int(max_count)
        self._actions: List[Tuple[int, Optional[int], Optional[int]]] = []
        if self.oracle_profile == "markov_count_only":
            for count in range(self.max_count + 1):
                self._actions.append((count, None, None))
        elif self.oracle_profile == "markov_count_endpoints":
            for count in range(self.max_count + 1):
                for first in range(self.n_regimes):
                    for last in range(self.n_regimes):
                        self._actions.append((count, first, last))
        else:
            raise ValueError(f"unsupported oracle_profile: {oracle_profile!r}")
        self._action_to_idx = {a: i for i, a in enumerate(self._actions)}
        self._obs_dim = (2 * self.n_regimes) + (self.n_regimes * self.n_regimes) + 1

    @property
    def action_labels(self) -> Sequence[str]:
        out = []
        for count, first, last in self._actions:
            if first is None:
                out.append(f"count={count}")
            else:
                out.append(f"count={count}|first={first}|last={last}")
        return tuple(out)

    @property
    def observation_dim(self) -> int:
        return int(self._obs_dim)

    def action_from_state_tuple(self, state: Tuple[int, Optional[int], Optional[int]]) -> int:
        return int(self._action_to_idx[state])

    def utility(self, action_idx: int, true_action_idx: int) -> float:
        return float(max(0.0, 1.0 - self.state_distance(action_idx, true_action_idx)))

    def state_distance(self, action_idx: int, true_action_idx: int) -> float:
        a_count, a_first, a_last = self._actions[int(action_idx)]
        t_count, t_first, t_last = self._actions[int(true_action_idx)]
        count_err = abs(float(a_count) - float(t_count)) / float(max(1, self.max_count))
        if self.oracle_profile == "markov_count_only":
            return float(count_err)
        first_err = 0.0 if int(a_first) == int(t_first) else 1.0
        last_err = 0.0 if int(a_last) == int(t_last) else 1.0
        return float((count_err + first_err + last_err) / 3.0)

    def pair_generator(self, true_action_idx: int, *, rng: random.Random) -> Tuple[int, int]:
        winner = int(true_action_idx)
        losers = [i for i in range(len(self._actions)) if i != winner]
        loser = int(rng.choice(losers))
        if self.utility(loser, winner) >= self.utility(winner, winner):
            loser = int((winner + 1) % len(self._actions))
        return (winner, loser)

    def group_generator(self, true_action_idx: int, *, rng: random.Random, k: int) -> List[int]:
        winner = int(true_action_idx)
        out = [winner]
        while len(out) < max(1, int(k)):
            cand = int(rng.randrange(len(self._actions)))
            if cand not in out:
                out.append(cand)
        rng.shuffle(out)
        return out

    def action_metrics(self, action_idx: int, true_action_idx: int) -> Dict[str, float]:
        a_count, a_first, a_last = self._actions[int(action_idx)]
        t_count, t_first, t_last = self._actions[int(true_action_idx)]
        out = {
            "exact_state_accuracy": 1.0 if int(action_idx) == int(true_action_idx) else 0.0,
            "utility": self.utility(action_idx, true_action_idx),
            "count_mae": abs(float(a_count) - float(t_count)),
        }
        if self.oracle_profile == "markov_count_endpoints":
            out["first_accuracy"] = 1.0 if int(a_first) == int(t_first) else 0.0
            out["last_accuracy"] = 1.0 if int(a_last) == int(t_last) else 0.0
        return out

    def tree_relevance_tag(self) -> str:
        return "tree_relevant" if self.oracle_profile == "markov_count_endpoints" else "tree_irrelevant_control"


def _state_to_action_idx(
    dgp: MarkovExactUtilityDGP,
    *,
    count: int,
    first: Optional[int],
    last: Optional[int],
) -> int:
    return dgp.action_from_state_tuple((int(count), first if first is None else int(first), last if last is None else int(last)))


def _prepare_markov_docs(
    docs,
    *,
    dgp: MarkovExactUtilityDGP,
    fixed_leaf_tokens: int,
    n_regimes: int,
) -> Tuple[UtilityDoc[np.ndarray], ...]:
    out: List[UtilityDoc[np.ndarray]] = []
    for doc in docs:
        n_tok = int(len(doc.token_regimes))
        spans = _leaf_spans(n_tok, leaf_tokens=int(fixed_leaf_tokens))
        leaf_obs = [
            np.asarray(_span_features(doc, sp, n_regimes=int(n_regimes), mode="full").numpy(), dtype=np.float32)
            for sp in spans
        ]
        if dgp.oracle_profile == "markov_count_only":
            leaf_states = [_count_only_from_span(doc, sp) for sp in spans]
            leaf_idx = tuple(_state_to_action_idx(dgp, count=st.count, first=None, last=None) for st in leaf_states)
            cur_states = list(leaf_states)
            merge_idx: List[int] = []
            while len(cur_states) > 1:
                nxt = []
                i = 0
                while i < len(cur_states):
                    if i + 1 >= len(cur_states):
                        nxt.append(cur_states[i])
                        i += 1
                        continue
                    merged = _count_only_merge(cur_states[i], cur_states[i + 1])
                    merge_idx.append(_state_to_action_idx(dgp, count=merged.count, first=None, last=None))
                    nxt.append(merged)
                    i += 2
                cur_states = nxt
            root_idx = _state_to_action_idx(dgp, count=int(cur_states[0].count), first=None, last=None)
        else:
            leaf_states = [_exact_from_span(doc, sp) for sp in spans]
            leaf_idx = tuple(_state_to_action_idx(dgp, count=st.count, first=st.first, last=st.last) for st in leaf_states)
            cur_states = list(leaf_states)
            merge_idx = []
            while len(cur_states) > 1:
                nxt = []
                i = 0
                while i < len(cur_states):
                    if i + 1 >= len(cur_states):
                        nxt.append(cur_states[i])
                        i += 1
                        continue
                    merged = _exact_merge(cur_states[i], cur_states[i + 1])
                    merge_idx.append(_state_to_action_idx(dgp, count=merged.count, first=merged.first, last=merged.last))
                    nxt.append(merged)
                    i += 2
                cur_states = nxt
            root_state: _ExactState = cur_states[0]
            root_idx = _state_to_action_idx(dgp, count=root_state.count, first=root_state.first, last=root_state.last)
        out.append(
            UtilityDoc(
                leaf_observations=tuple(leaf_obs),
                leaf_action_indices=tuple(leaf_idx),
                merge_action_indices_balanced=tuple(merge_idx),
                root_action_idx=int(root_idx),
                n_tokens=n_tok,
                metadata={"tree_relevance": dgp.tree_relevance_tag()},
            )
        )
    return tuple(out)


def _drop_endpoints(obs: np.ndarray, *, n_regimes: int) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).copy()
    arr[: 2 * int(n_regimes)] = 0.0
    return arr


def _run_oracle_exact(
    docs: Sequence[UtilityDoc[np.ndarray]],
) -> Tuple[List[int], List[List[int]]]:
    roots = [int(doc.root_action_idx) for doc in docs]
    merges = [list(map(int, doc.merge_action_indices_balanced)) for doc in docs]
    return roots, merges


def run_markov_exact_utility_experiment(config: MarkovExactUtilityConfig) -> ExactUtilitySummary:
    set_global_seed(int(config.seed), torch_threads=int(config.torch_threads))
    rng = np.random.default_rng(int(config.seed))
    transitions = _make_transition_matrices(
        n_classes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    gen_cfg = _GeneratorConfig(
        n_regimes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_len=int(config.min_seg_len),
        max_seg_len=int(config.max_seg_len),
        seed=int(config.seed),
    )
    raw_docs = generate_changepoint_docs(gen_cfg, transitions=transitions)
    raw_docs = raw_docs[: int(config.train_docs) + int(config.test_docs)]
    max_count = int(max(1, int(config.max_tokens) - 1))
    dgp = MarkovExactUtilityDGP(
        oracle_profile=str(config.oracle_profile),
        n_regimes=int(config.n_regimes),
        max_count=max_count,
    )
    docs = _prepare_markov_docs(
        raw_docs,
        dgp=dgp,
        fixed_leaf_tokens=int(config.fixed_leaf_tokens),
        n_regimes=int(config.n_regimes),
    )
    train_docs = docs[: int(config.train_docs)]
    test_docs = docs[int(config.train_docs) :]

    arm = str(config.structural_arm)
    if arm in {"oracle_exact", "tree_exact_supported"}:
        root_preds, merge_preds = _run_oracle_exact(test_docs)
    elif arm == "tree_neural_supported":
        model = train_neural_tree_policy(dgp, train_docs, config=config)
        preds = [predict_action_tree(model, doc) for doc in test_docs]
        root_preds = [p[0] for p in preds]
        merge_preds = [p[1] for p in preds]
    elif arm == "tree_undersupported":
        model = train_neural_tree_policy(
            dgp,
            train_docs,
            config=config,
            obs_transform=lambda x: _drop_endpoints(np.asarray(x, dtype=np.float32), n_regimes=int(config.n_regimes)),
        )
        preds = [
            predict_action_tree(
                model,
                doc,
                obs_transform=lambda x: _drop_endpoints(np.asarray(x, dtype=np.float32), n_regimes=int(config.n_regimes)),
            )
            for doc in test_docs
        ]
        root_preds = [p[0] for p in preds]
        merge_preds = [p[1] for p in preds]
    elif arm == "flat_equal_info":
        model = train_flat_policy(dgp, train_docs, config=config)
        root_preds = [predict_action_flat(model, doc) for doc in test_docs]
        merge_preds = [list(map(int, doc.merge_action_indices_balanced)) for doc in test_docs]
    elif arm == "one_leaf_control":
        model = train_neural_tree_policy(dgp, train_docs, config=config, one_leaf_override=True)
        preds = [predict_action_tree(model, doc, one_leaf_override=True) for doc in test_docs]
        root_preds = [p[0] for p in preds]
        merge_preds = [list(map(int, doc.merge_action_indices_balanced)) for doc in test_docs]
    else:
        raise ValueError(f"unsupported structural_arm for Markov lane: {arm!r}")

    metrics: ExactUtilityMetrics = summarize_predictions(
        dgp,
        test_docs,
        root_predictions=root_preds,
        merge_predictions=merge_preds,
    )
    budget = realized_budget_dict(docs, config=config, structural_arm=arm)
    root_metrics = asdict(metrics.root_recovery)
    merge_metrics = asdict(metrics.merge_recovery) if metrics.merge_recovery is not None else {}
    summary_metrics: Dict[str, object] = {
        "root": root_metrics,
        "merge": merge_metrics,
        "utility_regret": float(metrics.root_recovery.utility_regret),
        "dpo_gap_to_oracle": float(metrics.dpo_gap_to_oracle),
        "grpo_gap_to_oracle": float(metrics.grpo_gap_to_oracle),
        "ppo_reward_gap_to_oracle": float(metrics.ppo_reward_gap_to_oracle),
        "root_mae": float(metrics.root_mae),
        "merge_mae": float(metrics.merge_mae),
        "schedule_spread": float(metrics.schedule_spread),
        "gap_to_exact_ceiling": float(metrics.gap_to_exact_ceiling),
    }
    return ExactUtilitySummary(
        lane="markov",
        oracle_profile=str(config.oracle_profile),
        objective_family=str(config.objective_family),
        structural_arm=str(config.structural_arm),
        config=asdict(config),
        budget=budget,
        metrics=summary_metrics,
        metadata={
            "tree_relevance": dgp.tree_relevance_tag(),
            "lean_theorems": lean_theorem_refs("markov", str(config.oracle_profile)),
        },
        objective=preference_training_objective_semantics(
            objective_family=str(config.objective_family),
            root_query_rate=float(config.root_query_rate),
            leaf_label_rate=float(config.leaf_label_rate),
            internal_label_rate=float(config.internal_label_rate),
            hybrid_weight=float(config.hybrid_weight),
            dpo_beta=float(config.dpo_beta),
            grpo_beta=float(config.grpo_beta),
            ppo_kl_weight=float(config.ppo_kl_weight),
            entropy_weight=float(config.entropy_weight),
            pairwise_prefs_per_doc=float(config.pairwise_prefs_per_doc),
            group_pref_groups_per_doc=float(config.group_pref_groups_per_doc),
            ppo_rollouts_per_doc=float(config.ppo_rollouts_per_doc),
            metadata={"lane": "markov", "oracle_profile": str(config.oracle_profile)},
        ),
    )


__all__ = [
    "MarkovExactUtilityConfig",
    "MarkovExactUtilityDGP",
    "run_markov_exact_utility_experiment",
]
