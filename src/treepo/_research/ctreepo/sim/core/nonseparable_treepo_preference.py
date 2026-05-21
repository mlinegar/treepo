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
    predict_action_flat_span,
    predict_action_tree,
    realized_budget_dict,
    set_global_seed,
    summarize_predictions,
    train_flat_policy,
    train_flat_span_policy,
    train_neural_tree_policy,
)
from treepo._research.ctreepo.sim.objective_semantics import preference_training_objective_semantics


@dataclass(frozen=True, kw_only=True)
class NonseparableExactUtilityConfig(ExactUtilityRunConfig):
    lane: str = "nonseparable"
    oracle_profile: str = "dgp1_complementarity_and"
    structural_arm: str = "tree_neural_supported"
    count_max: int = 5
    n_binary_leaves: int = 4


class NonseparableExactUtilityDGP(ExactUtilityDGP[np.ndarray]):
    lane = "nonseparable"

    def __init__(self, *, oracle_profile: str, count_max: int, n_binary_leaves: int) -> None:
        self.oracle_profile = str(oracle_profile)
        self.count_max = int(count_max)
        self.n_binary_leaves = int(n_binary_leaves)
        if self.oracle_profile.startswith("dgp1_"):
            self._actions = [(l, r) for l in range(self.count_max + 1) for r in range(self.count_max + 1)]
            self._obs_dim = 1
        elif self.oracle_profile.startswith("dgp2_"):
            self._actions = [
                (ones, boundary)
                for ones in range(self.n_binary_leaves + 1)
                for boundary in range(self.n_binary_leaves)
            ]
            self._obs_dim = 2
        else:
            raise ValueError(f"unsupported oracle_profile: {oracle_profile!r}")
        self._action_to_idx = {a: i for i, a in enumerate(self._actions)}

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
        if self.oracle_profile.startswith("dgp1_"):
            return float((abs(float(a[0]) - float(t[0])) + abs(float(a[1]) - float(t[1]))) / (2.0 * float(max(1, self.count_max))))
        ones_err = abs(float(a[0]) - float(t[0])) / float(max(1, self.n_binary_leaves))
        boundary_err = abs(float(a[1]) - float(t[1])) / float(max(1, self.n_binary_leaves - 1))
        return float(0.5 * (ones_err + boundary_err))

    def pair_generator(self, true_action_idx: int, *, rng: random.Random) -> Tuple[int, int]:
        winner = int(true_action_idx)
        losers = [i for i in range(len(self._actions)) if i != winner]
        loser = int(rng.choice(losers))
        return winner, loser

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
        a = self._actions[int(action_idx)]
        t = self._actions[int(true_action_idx)]
        out: Dict[str, float] = {
            "exact_state_accuracy": 1.0 if int(action_idx) == int(true_action_idx) else 0.0,
            "utility": self.utility(action_idx, true_action_idx),
        }
        if self.oracle_profile.startswith("dgp1_"):
            out["state_l1"] = abs(float(a[0]) - float(t[0])) + abs(float(a[1]) - float(t[1]))
            downstream_true = 1.0 if int(t[0]) >= 3 and int(t[1]) >= 3 else 0.0
            downstream_pred = 1.0 if int(a[0]) >= 3 and int(a[1]) >= 3 else 0.0
            out["downstream_regret"] = abs(float(downstream_true) - float(downstream_pred))
        else:
            out["state_l1"] = abs(float(a[0]) - float(t[0])) + abs(float(a[1]) - float(t[1]))
            out["boundary_l1"] = abs(float(a[1]) - float(t[1]))
            out["downstream_regret"] = abs(
                float((2 * t[0]) + t[1]) / float(max(1, 3 * self.n_binary_leaves))
                - float((2 * a[0]) + a[1]) / float(max(1, 3 * self.n_binary_leaves))
            )
        return out

    def tree_relevance_tag(self) -> str:
        if self.oracle_profile in {"dgp1_complementarity_and", "dgp2_boundary_interaction"}:
            return "tree_relevant"
        return "tree_irrelevant_control"

    def action_idx(self, *parts: int) -> int:
        return int(self._action_to_idx[tuple(int(x) for x in parts)])


def _sample_dgp1_docs(
    *,
    cfg: NonseparableExactUtilityConfig,
    dgp: NonseparableExactUtilityDGP,
    n_docs: int,
    rng: random.Random,
) -> Tuple[UtilityDoc[np.ndarray], ...]:
    out: List[UtilityDoc[np.ndarray]] = []
    for _ in range(int(n_docs)):
        left = int(rng.randint(0, int(cfg.count_max)))
        if cfg.oracle_profile == "dgp1_complementarity_control":
            right = 3
        else:
            right = int(rng.randint(0, int(cfg.count_max)))
        leaf_obs = (
            np.asarray([float(left) / float(max(1, cfg.count_max))], dtype=np.float32),
            np.asarray([float(right) / float(max(1, cfg.count_max))], dtype=np.float32),
        )
        leaf_idx = (
            dgp.action_idx(left, 0),
            dgp.action_idx(0, right),
        )
        root_idx = dgp.action_idx(left, right)
        out.append(
            UtilityDoc(
                leaf_observations=leaf_obs,
                leaf_action_indices=leaf_idx,
                merge_action_indices_balanced=(root_idx,),
                root_action_idx=root_idx,
                n_tokens=2,
                metadata={"tree_relevance": dgp.tree_relevance_tag()},
            )
        )
    return tuple(out)


def _sample_dgp2_docs(
    *,
    cfg: NonseparableExactUtilityConfig,
    dgp: NonseparableExactUtilityDGP,
    n_docs: int,
    rng: random.Random,
) -> Tuple[UtilityDoc[np.ndarray], ...]:
    out: List[UtilityDoc[np.ndarray]] = []
    for _ in range(int(n_docs)):
        if cfg.oracle_profile == "dgp2_boundary_zero":
            bits = [int(rng.choice((0, 1))) for _ in range(int(cfg.n_binary_leaves))]
            bits.sort()
        else:
            bits = [int(rng.choice((0, 1))) for _ in range(int(cfg.n_binary_leaves))]
        ones = int(sum(bits))
        boundary = int(sum(1 for a, b in zip(bits[:-1], bits[1:]) if int(a) != int(b)))
        leaf_obs = tuple(
            np.asarray([float(bit), float(idx) / float(max(1, cfg.n_binary_leaves - 1))], dtype=np.float32)
            for idx, bit in enumerate(bits)
        )
        leaf_idx = tuple(dgp.action_idx(int(bit), 0) for bit in bits)
        cur = list(bits)
        merge_idx: List[int] = []
        while len(cur) > 1:
            nxt: List[Tuple[int, ...]] = []
            i = 0
            while i < len(cur):
                if i + 1 >= len(cur):
                    nxt.append((cur[i],))
                    i += 1
                    continue
                pair = (int(cur[i]), int(cur[i + 1]))
                pair_ones = int(sum(pair))
                pair_boundary = 1 if int(pair[0]) != int(pair[1]) else 0
                merge_idx.append(dgp.action_idx(pair_ones, pair_boundary))
                nxt.append(pair)
                i += 2
            cur = [int(sum(seg)) for seg in nxt]
        root_idx = dgp.action_idx(ones, boundary)
        out.append(
            UtilityDoc(
                leaf_observations=leaf_obs,
                leaf_action_indices=leaf_idx,
                merge_action_indices_balanced=tuple(merge_idx),
                root_action_idx=root_idx,
                n_tokens=int(cfg.n_binary_leaves),
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


def run_nonseparable_exact_utility_experiment(config: NonseparableExactUtilityConfig) -> ExactUtilitySummary:
    set_global_seed(int(config.seed), torch_threads=int(config.torch_threads))
    rng = random.Random(int(config.seed))
    dgp = NonseparableExactUtilityDGP(
        oracle_profile=str(config.oracle_profile),
        count_max=int(config.count_max),
        n_binary_leaves=int(config.n_binary_leaves),
    )
    if str(config.oracle_profile).startswith("dgp1_"):
        docs = _sample_dgp1_docs(cfg=config, dgp=dgp, n_docs=int(config.train_docs + config.test_docs), rng=rng)
    else:
        docs = _sample_dgp2_docs(cfg=config, dgp=dgp, n_docs=int(config.train_docs + config.test_docs), rng=rng)
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
        if str(config.oracle_profile).startswith("dgp1_"):
            transform = lambda x: np.asarray([float(x[0])], dtype=np.float32)
        else:
            transform = lambda x: np.asarray([float(x[0]), 0.0], dtype=np.float32)
        model = train_neural_tree_policy(dgp, train_docs, config=config, obs_transform=transform)
        preds = [predict_action_tree(model, doc, obs_transform=transform) for doc in test_docs]
        root_preds = [p[0] for p in preds]
        merge_preds = [p[1] for p in preds]
    elif arm == "flat_equal_info":
        model = train_flat_policy(dgp, train_docs, config=config)
        root_preds = [predict_action_flat(model, doc) for doc in test_docs]
        merge_preds = [list(map(int, doc.merge_action_indices_balanced)) for doc in test_docs]
    elif arm == "flat_span_equal_info":
        model = train_flat_span_policy(dgp, train_docs, config=config)
        preds = [predict_action_flat_span(model, doc) for doc in test_docs]
        root_preds = [p[0] for p in preds]
        merge_preds = [p[1] for p in preds]
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
        raise ValueError(f"unsupported structural_arm for nonseparable lane: {arm!r}")

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
        lane="nonseparable",
        oracle_profile=str(config.oracle_profile),
        objective_family=str(config.objective_family),
        structural_arm=str(config.structural_arm),
        config=asdict(config),
        budget=budget,
        metrics=summary_metrics,
        metadata={
            "tree_relevance": dgp.tree_relevance_tag(),
            "lean_theorems": lean_theorem_refs("nonseparable", str(config.oracle_profile)),
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
            metadata={"lane": "nonseparable", "oracle_profile": str(config.oracle_profile)},
        ),
    )


__all__ = [
    "NonseparableExactUtilityConfig",
    "NonseparableExactUtilityDGP",
    "run_nonseparable_exact_utility_experiment",
]
