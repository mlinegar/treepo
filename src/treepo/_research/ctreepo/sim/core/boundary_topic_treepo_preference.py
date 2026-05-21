from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

from treepo._research.ctreepo.sim.core.exact_utility_common import (
    ExactUtilityDGP,
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


def _count_vectors(total: int, k: int) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for cuts in itertools.combinations_with_replacement(range(k), total):
        counts = [0 for _ in range(k)]
        for idx in cuts:
            counts[int(idx)] += 1
        tup = tuple(counts)
        if tup not in out:
            out.append(tup)
    return out


@dataclass(frozen=True, kw_only=True)
class BoundaryTopicExactUtilityConfig(ExactUtilityRunConfig):
    lane: str = "boundary_topic"
    oracle_profile: str = "topic_plus_boundary"
    structural_arm: str = "tree_neural_supported"
    n_topics: int = 3
    n_leaves: int = 6


class BoundaryTopicExactUtilityDGP(ExactUtilityDGP[np.ndarray]):
    lane = "boundary_topic"

    def __init__(self, *, oracle_profile: str, n_topics: int, n_leaves: int) -> None:
        self.oracle_profile = str(oracle_profile)
        self.n_topics = int(n_topics)
        self.n_leaves = int(n_leaves)
        count_states: List[Tuple[int, ...]] = []
        for total in range(1, int(n_leaves) + 1):
            count_states.extend(_count_vectors(int(total), int(n_topics)))
        self._actions: List[Tuple[int, ...]] = []
        if self.oracle_profile == "topic_mass_only":
            self._actions = list(count_states)
        elif self.oracle_profile == "topic_plus_boundary":
            for counts in count_states:
                for first in range(self.n_topics):
                    for last in range(self.n_topics):
                        self._actions.append(tuple(counts) + (int(first), int(last)))
        else:
            raise ValueError(f"unsupported oracle_profile: {oracle_profile!r}")
        self._action_to_idx = {a: i for i, a in enumerate(self._actions)}
        self._obs_dim = int(self.n_topics)

    @property
    def action_labels(self) -> Sequence[str]:
        return tuple(str(a) for a in self._actions)

    @property
    def observation_dim(self) -> int:
        return int(self._obs_dim)

    def utility(self, action_idx: int, true_action_idx: int) -> float:
        return float(max(0.0, 1.0 - self.state_distance(action_idx, true_action_idx)))

    def state_distance(self, action_idx: int, true_action_idx: int) -> float:
        a = self._actions[int(action_idx)]
        t = self._actions[int(true_action_idx)]
        count_a = np.asarray(a[: self.n_topics], dtype=np.float64)
        count_t = np.asarray(t[: self.n_topics], dtype=np.float64)
        theta_l1 = float(np.sum(np.abs(count_a - count_t)) / float(max(1, 2 * self.n_leaves)))
        if self.oracle_profile == "topic_mass_only":
            return theta_l1
        endpoint_err = 0.0
        endpoint_err += 0.0 if int(a[-2]) == int(t[-2]) else 1.0
        endpoint_err += 0.0 if int(a[-1]) == int(t[-1]) else 1.0
        return float((theta_l1 + 0.5 * endpoint_err) / 2.0)

    def pair_generator(self, true_action_idx: int, *, rng: random.Random) -> Tuple[int, int]:
        winner = int(true_action_idx)
        loser = int(rng.randrange(len(self._actions)))
        while loser == winner:
            loser = int(rng.randrange(len(self._actions)))
        return winner, loser

    def group_generator(self, true_action_idx: int, *, rng: random.Random, k: int) -> List[int]:
        out = [int(true_action_idx)]
        while len(out) < max(1, int(k)):
            cand = int(rng.randrange(len(self._actions)))
            if cand not in out:
                out.append(cand)
        rng.shuffle(out)
        return out

    def action_metrics(self, action_idx: int, true_action_idx: int) -> Dict[str, float]:
        a = self._actions[int(action_idx)]
        t = self._actions[int(true_action_idx)]
        count_a = np.asarray(a[: self.n_topics], dtype=np.float64)
        count_t = np.asarray(t[: self.n_topics], dtype=np.float64)
        out: Dict[str, float] = {
            "exact_state_accuracy": 1.0 if int(action_idx) == int(true_action_idx) else 0.0,
            "utility": self.utility(action_idx, true_action_idx),
            "theta_l1": float(np.sum(np.abs(count_a - count_t)) / float(max(1, self.n_leaves))),
        }
        if self.oracle_profile == "topic_plus_boundary":
            out["first_accuracy"] = 1.0 if int(a[-2]) == int(t[-2]) else 0.0
            out["last_accuracy"] = 1.0 if int(a[-1]) == int(t[-1]) else 0.0
            out["boundary_l1"] = float((1.0 - out["first_accuracy"]) + (1.0 - out["last_accuracy"]))
        return out

    def tree_relevance_tag(self) -> str:
        return "tree_relevant" if self.oracle_profile == "topic_plus_boundary" else "tree_irrelevant_control"

    def action_idx(self, parts: Tuple[int, ...]) -> int:
        return int(self._action_to_idx[tuple(int(x) for x in parts)])


def _sample_docs(
    *,
    cfg: BoundaryTopicExactUtilityConfig,
    dgp: BoundaryTopicExactUtilityDGP,
    n_docs: int,
    rng: random.Random,
) -> Tuple[UtilityDoc[np.ndarray], ...]:
    out: List[UtilityDoc[np.ndarray]] = []
    for _ in range(int(n_docs)):
        if cfg.oracle_profile == "topic_mass_only":
            topics = [int(rng.randrange(int(cfg.n_topics))) for _ in range(int(cfg.n_leaves))]
            topics.sort()
        else:
            topics = [int(rng.randrange(int(cfg.n_topics))) for _ in range(int(cfg.n_leaves))]
        counts = [0 for _ in range(int(cfg.n_topics))]
        for t in topics:
            counts[int(t)] += 1
        leaf_obs = []
        leaf_idx = []
        for t in topics:
            obs = np.zeros((int(cfg.n_topics),), dtype=np.float32)
            obs[int(t)] = 1.0
            leaf_obs.append(obs)
            leaf_counts = [0 for _ in range(int(cfg.n_topics))]
            leaf_counts[int(t)] = 1
            if cfg.oracle_profile == "topic_mass_only":
                leaf_idx.append(dgp.action_idx(tuple(leaf_counts)))
            else:
                leaf_idx.append(dgp.action_idx(tuple(leaf_counts) + (int(t), int(t))))

        merge_idx: List[int] = []
        cur_counts = [[1 if idx == t else 0 for idx in range(int(cfg.n_topics))] for t in topics]
        cur_first = list(topics)
        cur_last = list(topics)
        while len(cur_counts) > 1:
            nxt_counts: List[List[int]] = []
            nxt_first: List[int] = []
            nxt_last: List[int] = []
            i = 0
            while i < len(cur_counts):
                if i + 1 >= len(cur_counts):
                    nxt_counts.append(cur_counts[i])
                    nxt_first.append(cur_first[i])
                    nxt_last.append(cur_last[i])
                    i += 1
                    continue
                merged_counts = [int(cur_counts[i][j]) + int(cur_counts[i + 1][j]) for j in range(int(cfg.n_topics))]
                if cfg.oracle_profile == "topic_mass_only":
                    merge_idx.append(dgp.action_idx(tuple(merged_counts)))
                else:
                    merge_idx.append(dgp.action_idx(tuple(merged_counts) + (int(cur_first[i]), int(cur_last[i + 1]))))
                nxt_counts.append(merged_counts)
                nxt_first.append(cur_first[i])
                nxt_last.append(cur_last[i + 1])
                i += 2
            cur_counts = nxt_counts
            cur_first = nxt_first
            cur_last = nxt_last

        if cfg.oracle_profile == "topic_mass_only":
            root_idx = dgp.action_idx(tuple(counts))
        else:
            root_idx = dgp.action_idx(tuple(counts) + (int(topics[0]), int(topics[-1])))
        out.append(
            UtilityDoc(
                leaf_observations=tuple(leaf_obs),
                leaf_action_indices=tuple(leaf_idx),
                merge_action_indices_balanced=tuple(merge_idx),
                root_action_idx=int(root_idx),
                n_tokens=int(cfg.n_leaves),
                metadata={"tree_relevance": dgp.tree_relevance_tag()},
            )
        )
    return tuple(out)


def _reverse_doc(doc: UtilityDoc[np.ndarray]) -> UtilityDoc[np.ndarray]:
    return UtilityDoc(
        leaf_observations=tuple(reversed(doc.leaf_observations)),
        leaf_action_indices=tuple(reversed(doc.leaf_action_indices)),
        merge_action_indices_balanced=tuple(doc.merge_action_indices_balanced),
        root_action_idx=int(doc.root_action_idx),
        n_tokens=int(doc.n_tokens),
        metadata=dict(doc.metadata),
    )


def run_boundary_topic_exact_utility_experiment(config: BoundaryTopicExactUtilityConfig) -> ExactUtilitySummary:
    set_global_seed(int(config.seed), torch_threads=int(config.torch_threads))
    rng = random.Random(int(config.seed))
    dgp = BoundaryTopicExactUtilityDGP(
        oracle_profile=str(config.oracle_profile),
        n_topics=int(config.n_topics),
        n_leaves=int(config.n_leaves),
    )
    docs = _sample_docs(cfg=config, dgp=dgp, n_docs=int(config.train_docs + config.test_docs), rng=rng)
    train_docs = docs[: int(config.train_docs)]
    test_docs = docs[int(config.train_docs) :]
    arm = str(config.structural_arm)

    if arm in {"oracle_exact", "tree_exact_supported"}:
        root_preds = [int(doc.root_action_idx) for doc in test_docs]
        merge_preds = [list(map(int, doc.merge_action_indices_balanced)) for doc in test_docs]
    elif arm == "tree_neural_supported":
        model = train_neural_tree_policy(dgp, train_docs, config=config)
        preds = [predict_action_tree(model, doc) for doc in test_docs]
        root_preds = [p[0] for p in preds]
        merge_preds = [p[1] for p in preds]
    elif arm == "tree_undersupported":
        def transform(x: np.ndarray) -> np.ndarray:
            arr = np.zeros((int(config.n_topics),), dtype=np.float32)
            arr[0] = float(np.asarray(x, dtype=np.float32)[0])
            return arr
        model = train_neural_tree_policy(dgp, train_docs, config=config, obs_transform=transform)
        preds = [predict_action_tree(model, doc, obs_transform=transform) for doc in test_docs]
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
    elif arm == "right_rule_wrong_chunker":
        wrong_train = tuple(_reverse_doc(doc) for doc in train_docs)
        wrong_test = tuple(_reverse_doc(doc) for doc in test_docs)
        model = train_neural_tree_policy(dgp, wrong_train, config=config)
        preds = [predict_action_tree(model, doc) for doc in wrong_test]
        root_preds = [p[0] for p in preds]
        merge_preds = [p[1] for p in preds]
    else:
        raise ValueError(f"unsupported structural_arm for boundary-topic lane: {arm!r}")

    metrics = summarize_predictions(dgp, test_docs, root_predictions=root_preds, merge_predictions=merge_preds)
    budget = realized_budget_dict(docs, config=config, structural_arm=arm)
    summary_metrics: Dict[str, object] = {
        "root": asdict(metrics.root_recovery),
        "merge": asdict(metrics.merge_recovery) if metrics.merge_recovery is not None else {},
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
        lane="boundary_topic",
        oracle_profile=str(config.oracle_profile),
        objective_family=str(config.objective_family),
        structural_arm=str(config.structural_arm),
        config=asdict(config),
        budget=budget,
        metrics=summary_metrics,
        metadata={
            "tree_relevance": dgp.tree_relevance_tag(),
            "lean_theorems": lean_theorem_refs("boundary_topic", str(config.oracle_profile)),
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
            metadata={"lane": "boundary_topic", "oracle_profile": str(config.oracle_profile)},
        ),
    )


__all__ = [
    "BoundaryTopicExactUtilityConfig",
    "BoundaryTopicExactUtilityDGP",
    "run_boundary_topic_exact_utility_experiment",
]
