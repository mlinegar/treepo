"""
Learned sketch recovery simulation.

Demonstrates that a small neural network, trained only on local oracle queries
at sampled tree nodes (the probabilistic audit), can recover the same merge
performance as a hand-designed sketch.  The sufficiency boundary (m < k)
persists as a structural limit: the linear readout from m state dimensions
cannot independently recover k per-type counts when m < k.

Oracle: each indicator position has a TYPE (assigned round-robin mod k).
The oracle returns the k-vector of per-type spike counts.  The threshold
event is "all k types present" (at least one spike per type).

Requires: torch >= 2.0
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise ImportError(
        "PyTorch is required for learned sketch experiments. "
        "Install with: pip install torch>=2.0.0"
    )

# Import directly from module file to avoid triggering src/tree/__init__.py
# which eagerly imports the full LLM stack (dspy, etc.)
import importlib.util as _ilu
import sys as _sys

_mod_name = "src.tree.mergeable_ablation"
if _mod_name not in _sys.modules:
    _mod_path = str(Path(__file__).parent / "mergeable_ablation.py")
    _spec = _ilu.spec_from_file_location(_mod_name, _mod_path)
    _ma = _ilu.module_from_spec(_spec)
    _sys.modules[_mod_name] = _ma
    _spec.loader.exec_module(_ma)
else:
    _ma = _sys.modules[_mod_name]

SpikeCountMixtureDistributionSpec = _ma.SpikeCountMixtureDistributionSpec
ToyTokenDocument = _ma.ToyTokenDocument
sample_spike_count_mixture_documents = _ma.sample_spike_count_mixture_documents
true_spike_count = _ma.true_spike_count
from treepo._research.training.config_sections import OptimizerConfig, RunConfig, TrainConfig, ValidationConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPIKE_THRESHOLD = 0.90

DEFAULT_DISTRIBUTION = SpikeCountMixtureDistributionSpec(
    p_spike_doc=0.62,
    spike_count_support=(1, 2, 3, 4, 5),
    spike_count_probs_given_spike=(0.10, 0.20, 0.25, 0.25, 0.20),
    p_boundary_given_spike=0.35,
    n_tokens=32,
    proxy_noise=0.12,
    boundary_span_tokens=4,
)


@dataclass(frozen=True, kw_only=True)
class LearnedSketchModelConfig:
    state_dim: int = 4
    target_k: int = 4
    hidden_dim: int = 32


@dataclass(frozen=True, kw_only=True)
class LearnedSketchDataConfig:
    chunk_size: int = 4


@dataclass(frozen=True, kw_only=True)
class LearnedSketchObjectiveConfig:
    n_audit: int = 7
    law_tolerance: float = 0.5


@dataclass(frozen=True, kw_only=True)
class LearnedSketchEvaluationConfig:
    eval_docs: int = 256


@dataclass(frozen=True, kw_only=True)
class LearnedSketchTrainingConfig:
    """Sectioned configuration for learned sketch training."""

    model: LearnedSketchModelConfig = field(default_factory=LearnedSketchModelConfig)
    data: LearnedSketchDataConfig = field(default_factory=LearnedSketchDataConfig)
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(batch_size=64, steps=2000)
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=1e-3)
    )
    validation: ValidationConfig = field(
        default_factory=lambda: ValidationConfig(eval_every=50)
    )
    run: RunConfig = field(default_factory=RunConfig)
    objective: LearnedSketchObjectiveConfig = field(
        default_factory=LearnedSketchObjectiveConfig
    )
    evaluation: LearnedSketchEvaluationConfig = field(
        default_factory=LearnedSketchEvaluationConfig
    )


# ---------------------------------------------------------------------------
# Oracle: k-type spike counts (re-exported from treepo._research.ctreepo.oracles registry)
# ---------------------------------------------------------------------------

from treepo._research.ctreepo.oracles.sketches import type_oracle  # noqa: F401  (public re-export)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LearnedSketchModel(nn.Module):
    """Small sketch model with explicit decoded summary and re-summary path."""

    def __init__(
        self, n_indicators: int, state_dim: int, n_types: int, hidden_dim: int = 32
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_types = n_types

        # Leaf encoder: raw indicators -> state
        self.leaf_encoder = nn.Sequential(
            nn.Linear(n_indicators, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Merge function: (left_state || right_state) -> state
        self.merge_fn = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Readout: state -> predicted per-type counts (k-dimensional)
        self.readout = nn.Linear(state_dim, n_types)
        self.summary_encoder = nn.Sequential(
            nn.Linear(n_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def encode_leaf(self, indicators: torch.Tensor) -> torch.Tensor:
        return self.leaf_encoder(indicators)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.merge_fn(torch.cat([left, right], dim=-1))

    def predict_type_counts(self, state: torch.Tensor) -> torch.Tensor:
        """Predict per-type spike counts from state vector. Returns k-vector."""
        return self.readout(state)

    def decode_summary(self, state: torch.Tensor) -> torch.Tensor:
        """Decode theorem-domain summary statistics from the latent sketch."""
        return self.predict_type_counts(state)

    def encode_summary(self, summary_counts: torch.Tensor) -> torch.Tensor:
        """Re-encode decoded summary statistics for on-range idempotence checks."""
        return self.summary_encoder(summary_counts)


# ---------------------------------------------------------------------------
# Tree forward pass
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in the binary merge tree."""

    level: int                                    # 0 = leaf, increases upward
    raw_indicators: List[float]                   # all indicators under this node
    positions: List[int]                          # global positions of indicators
    state: Optional[torch.Tensor] = None          # learned state (set during forward pass)
    children: Optional[Tuple[int, int]] = None    # indices of children (None for leaves)


def build_tree_from_doc(
    doc: ToyTokenDocument,
    chunk_size: int,
) -> List[TreeNode]:
    """Build a binary merge tree from a document, returning nodes bottom-up."""
    scores = list(doc.token_scores)
    n = len(scores)

    # Create leaf nodes from fixed-size chunks
    leaves = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        indicators = scores[start:end]
        pos = list(range(start, end))
        # Pad if needed
        while len(indicators) < chunk_size:
            indicators.append(0.0)
            pos.append(pos[-1] + 1 if pos else start)
        leaves.append(TreeNode(level=0, raw_indicators=indicators, positions=pos))

    nodes = list(leaves)

    # Build binary tree bottom-up
    current_level_start = 0
    current_level_count = len(leaves)
    level = 1

    while current_level_count > 1:
        next_level_start = len(nodes)
        for i in range(0, current_level_count, 2):
            left_idx = current_level_start + i
            if i + 1 < current_level_count:
                right_idx = current_level_start + i + 1
                merged_indicators = (
                    nodes[left_idx].raw_indicators + nodes[right_idx].raw_indicators
                )
                merged_positions = (
                    nodes[left_idx].positions + nodes[right_idx].positions
                )
                nodes.append(
                    TreeNode(
                        level=level,
                        raw_indicators=merged_indicators,
                        positions=merged_positions,
                        children=(left_idx, right_idx),
                    )
                )
            else:
                # Odd node: promote directly
                nodes.append(
                    TreeNode(
                        level=level,
                        raw_indicators=list(nodes[left_idx].raw_indicators),
                        positions=list(nodes[left_idx].positions),
                        children=(left_idx, left_idx),
                    )
                )
        current_level_start = next_level_start
        current_level_count = len(nodes) - next_level_start
        level += 1

    return nodes


def forward_pass(
    model: LearnedSketchModel,
    nodes: List[TreeNode],
    chunk_size: int,
) -> None:
    """Run forward pass through tree, setting state on each node."""
    for node in nodes:
        if node.children is None:
            indicators = torch.tensor(node.raw_indicators, dtype=torch.float32)
            node.state = model.encode_leaf(indicators)
        else:
            left_idx, right_idx = node.children
            left_state = nodes[left_idx].state
            right_state = nodes[right_idx].state
            assert left_state is not None and right_state is not None
            node.state = model.merge(left_state, right_state)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_step(
    model: LearnedSketchModel,
    optimizer: optim.Optimizer,
    docs: List[ToyTokenDocument],
    config: LearnedSketchTrainingConfig,
    rng: random.Random,
) -> float:
    """One training step with explicit L1/L2/L3-style decoded-summary losses."""
    model.train()
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0)
    n_samples = 0

    for doc in docs:
        nodes = build_tree_from_doc(doc, config.data.chunk_size)
        forward_pass(model, nodes, config.data.chunk_size)

        # Sample nodes for audit
        n_to_sample = min(config.objective.n_audit, len(nodes))
        sampled_indices = rng.sample(range(len(nodes)), k=n_to_sample)

        for idx in sampled_indices:
            node = nodes[idx]
            assert node.state is not None
            predicted = model.decode_summary(node.state)
            target = type_oracle(
                node.raw_indicators, node.positions, config.model.target_k
            )
            target_t = torch.tensor(target, dtype=torch.float32)
            decode_loss = ((predicted - target_t) ** 2).sum()
            resummarized = model.decode_summary(model.encode_summary(predicted))
            idemp_loss = ((resummarized - predicted) ** 2).sum()

            total_loss = total_loss + decode_loss + idemp_loss
            n_samples += 2

    if n_samples > 0:
        loss = total_loss / n_samples
        loss.backward()
        optimizer.step()
        return loss.item()
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    """Evaluation metrics at a single checkpoint."""

    step: int
    root_oracle_mse: float           # mean of sum-of-squared-errors over k dims
    root_threshold_accuracy: float   # "all types present" accuracy
    l1_leaf_error: float
    l2_merge_error: float
    l3_idemp_error: float
    mean_node_oracle_mse: float


@torch.no_grad()
def evaluate(
    model: LearnedSketchModel,
    docs: List[ToyTokenDocument],
    config: LearnedSketchTrainingConfig,
) -> EvalMetrics:
    """Evaluate learned sketch with explicit decoded-summary local-law metrics."""
    model.eval()

    root_errors_sq = []
    threshold_correct = []
    leaf_errors_sq = []
    merge_errors_sq = []
    idemp_errors_sq = []
    all_node_errors_sq = []

    for doc in docs:
        nodes = build_tree_from_doc(doc, config.data.chunk_size)
        forward_pass(model, nodes, config.data.chunk_size)

        # Root evaluation
        root = nodes[-1]
        assert root.state is not None
        pred_vec = model.decode_summary(root.state)
        true_vec = type_oracle(
            root.raw_indicators, root.positions, config.model.target_k
        )
        true_t = torch.tensor(true_vec, dtype=torch.float32)

        # MSE over k dimensions
        root_errors_sq.append(((pred_vec - true_t) ** 2).sum().item())

        # Threshold: "all types present" — each type has ≥ 1 spike
        pred_present = all(round(p) >= 1.0 for p in pred_vec.tolist())
        true_present = all(c >= 1.0 for c in true_vec)
        threshold_correct.append(1.0 if pred_present == true_present else 0.0)

        for node in nodes:
            assert node.state is not None
            pred = model.decode_summary(node.state)
            true_v = type_oracle(
                node.raw_indicators, node.positions, config.model.target_k
            )
            true_vt = torch.tensor(true_v, dtype=torch.float32)
            error = ((pred - true_vt) ** 2).sum().item()
            all_node_errors_sq.append(error)
            if node.children is None:
                leaf_errors_sq.append(error)
            else:
                merge_errors_sq.append(error)

            resummarized = model.decode_summary(model.encode_summary(pred))
            idemp_errors_sq.append(((resummarized - pred) ** 2).sum().item())

    root_mse = sum(root_errors_sq) / max(len(root_errors_sq), 1)
    accuracy = sum(threshold_correct) / max(len(threshold_correct), 1)
    l1_error = sum(leaf_errors_sq) / max(len(leaf_errors_sq), 1)
    l2_error = sum(merge_errors_sq) / max(len(merge_errors_sq), 1)
    l3_error = sum(idemp_errors_sq) / max(len(idemp_errors_sq), 1)
    mean_mse = sum(all_node_errors_sq) / max(len(all_node_errors_sq), 1)

    return EvalMetrics(
        step=0,
        root_oracle_mse=root_mse,
        root_threshold_accuracy=accuracy,
        l1_leaf_error=l1_error,
        l2_merge_error=l2_error,
        l3_idemp_error=l3_error,
        mean_node_oracle_mse=mean_mse,
    )


# ---------------------------------------------------------------------------
# Hand-designed baseline: type-aware top-m sketch
# ---------------------------------------------------------------------------


def _hand_designed_type_sketch(
    scores: List[float],
    chunk_size: int,
    n_types: int,
    sketch_order: int,
) -> List[float]:
    """Run hand-designed type-aware top-m sketch, returning predicted root type counts.

    At each leaf: compute per-type spike counts, retain the m types with
    highest counts (ties broken arbitrarily).  At each merge: combine two
    dicts, add counts for shared types, keep top-m by count.  At root:
    return the full dict (missing types treated as 0).
    """
    n = len(scores)
    m = sketch_order

    # Leaf states: dicts of {type: count} with at most m entries
    leaf_states = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_vals = scores[start:end]
        positions = list(range(start, end))

        type_counts: Dict[int, float] = {}
        for val, pos in zip(chunk_vals, positions):
            if val >= SPIKE_THRESHOLD:
                t = pos % n_types
                type_counts[t] = type_counts.get(t, 0.0) + 1.0

        # Keep top-m types by count
        if len(type_counts) > m:
            sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
            type_counts = dict(sorted_types[:m])

        leaf_states.append(type_counts)

    # Tree merge
    current_level = leaf_states
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                left = current_level[i]
                right = current_level[i + 1]
                merged: Dict[int, float] = dict(left)
                for t, c in right.items():
                    merged[t] = merged.get(t, 0.0) + c
                # Keep top-m
                if len(merged) > m:
                    sorted_types = sorted(merged.items(), key=lambda x: -x[1])
                    merged = dict(sorted_types[:m])
                next_level.append(merged)
            else:
                next_level.append(current_level[i])
        current_level = next_level

    root_state = current_level[0] if current_level else {}

    # Reconstruct full k-vector (untracked types = 0)
    result = [0.0] * n_types
    for t, c in root_state.items():
        if 0 <= t < n_types:
            result[t] = c
    return result


def hand_designed_accuracy(
    docs: List[ToyTokenDocument],
    target_k: int,
    sketch_order: int,
    chunk_size: int,
) -> float:
    """Run hand-designed type-aware top-m sketch and return 'all types present' accuracy."""
    correct = 0
    for doc in docs:
        scores = list(doc.token_scores)

        # True type counts
        true_counts = type_oracle(
            scores, list(range(len(scores))), target_k
        )
        true_present = all(c >= 1.0 for c in true_counts)

        # Hand-designed prediction
        pred_counts = _hand_designed_type_sketch(
            scores, chunk_size, target_k, sketch_order
        )
        pred_present = all(c >= 1.0 for c in pred_counts)

        if pred_present == true_present:
            correct += 1

    return correct / max(len(docs), 1)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


@dataclass
class LearningCurveResult:
    """Result of one learning curve run (one m value)."""

    state_dim: int
    target_k: int
    metrics: List[Dict]
    hand_designed_accuracy: float
    final_root_oracle_mse: float
    final_threshold_accuracy: float
    final_l1_leaf_error: float
    final_l2_merge_error: float
    final_l3_idemp_error: float


def run_learning_curve(
    config: LearnedSketchTrainingConfig,
    distribution: SpikeCountMixtureDistributionSpec = DEFAULT_DISTRIBUTION,
) -> LearningCurveResult:
    """Train a learned sketch and record metrics over training."""
    rng = random.Random(config.run.seed)
    torch.manual_seed(config.run.seed)

    n_indicators = config.data.chunk_size
    model = LearnedSketchModel(
        n_indicators=n_indicators,
        state_dim=config.model.state_dim,
        n_types=config.model.target_k,
        hidden_dim=config.model.hidden_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)

    # Generate held-out evaluation docs
    eval_docs = sample_spike_count_mixture_documents(
        spec=distribution,
        n_docs=config.evaluation.eval_docs,
        seed=config.run.seed + 999_999,
    )

    # Hand-designed baseline on eval docs
    hd_acc = hand_designed_accuracy(
        eval_docs, config.model.target_k, config.model.state_dim, config.data.chunk_size
    )

    all_metrics = []

    for step in range(config.train.steps):
        # Generate fresh training batch
        batch_seed = config.run.seed + step * 1000
        train_docs = sample_spike_count_mixture_documents(
            spec=distribution,
            n_docs=config.train.batch_size,
            seed=batch_seed,
        )

        train_step(model, optimizer, train_docs, config, rng)

        # Evaluate periodically
        if step % config.validation.eval_every == 0 or step == config.train.steps - 1:
            metrics = evaluate(model, eval_docs, config)
            metrics = EvalMetrics(
                step=step,
                root_oracle_mse=metrics.root_oracle_mse,
                root_threshold_accuracy=metrics.root_threshold_accuracy,
                l1_leaf_error=metrics.l1_leaf_error,
                l2_merge_error=metrics.l2_merge_error,
                l3_idemp_error=metrics.l3_idemp_error,
                mean_node_oracle_mse=metrics.mean_node_oracle_mse,
            )
            all_metrics.append(asdict(metrics))

    final = all_metrics[-1] if all_metrics else {}

    return LearningCurveResult(
        state_dim=config.model.state_dim,
        target_k=config.model.target_k,
        metrics=all_metrics,
        hand_designed_accuracy=hd_acc,
        final_root_oracle_mse=final.get("root_oracle_mse", float("nan")),
        final_threshold_accuracy=final.get("root_threshold_accuracy", float("nan")),
        final_l1_leaf_error=final.get("l1_leaf_error", float("nan")),
        final_l2_merge_error=final.get("l2_merge_error", float("nan")),
        final_l3_idemp_error=final.get("l3_idemp_error", float("nan")),
    )


def run_learning_curve_experiment(
    target_k: int = 4,
    state_dims: Sequence[int] = (2, 3, 4, 5, 6),
    n_steps: int = 2000,
    seed: int = 42,
    distribution: SpikeCountMixtureDistributionSpec = DEFAULT_DISTRIBUTION,
) -> Dict:
    """Experiment 1: learning curves for different state dimensions."""
    results = []
    for m in state_dims:
        print(f"  Training learned sketch: m={m}, k={target_k}, steps={n_steps}")
        config = LearnedSketchTrainingConfig(
            model=LearnedSketchModelConfig(state_dim=m, target_k=target_k),
            train=TrainConfig(batch_size=64, steps=n_steps),
            run=RunConfig(seed=seed + m * 100),
        )
        result = run_learning_curve(config, distribution)
        results.append(asdict(result))
        print(
            f"    Final: root_oracle_mse={result.final_root_oracle_mse:.4f}, "
            f"accuracy={result.final_threshold_accuracy:.3f}, "
            f"l1={result.final_l1_leaf_error:.3f}, "
            f"l2={result.final_l2_merge_error:.3f}, "
            f"l3={result.final_l3_idemp_error:.3f}, "
            f"hand_designed={result.hand_designed_accuracy:.3f}"
        )

    return {
        "experiment": "learning_curves",
        "target_k": target_k,
        "state_dims": list(state_dims),
        "n_steps": n_steps,
        "seed": seed,
        "results": results,
    }


def run_convergence_comparison(
    target_k: int = 4,
    state_dim: int = 4,
    n_steps: int = 2000,
    seed: int = 42,
    distribution: SpikeCountMixtureDistributionSpec = DEFAULT_DISTRIBUTION,
) -> Dict:
    """Experiment 2: compare learned vs hand-designed vs naive at convergence."""
    print(f"  Training learned sketch: m={state_dim}, k={target_k}")
    config = LearnedSketchTrainingConfig(
        model=LearnedSketchModelConfig(state_dim=state_dim, target_k=target_k),
        train=TrainConfig(batch_size=64, steps=n_steps),
        run=RunConfig(seed=seed),
    )
    result = run_learning_curve(config, distribution)

    comparison = {
        "learned": {
            "root_oracle_mse": result.final_root_oracle_mse,
            "threshold_accuracy": result.final_threshold_accuracy,
            "l1_leaf_error": result.final_l1_leaf_error,
            "l2_merge_error": result.final_l2_merge_error,
            "l3_idemp_error": result.final_l3_idemp_error,
        },
        "hand_designed": {
            "threshold_accuracy": result.hand_designed_accuracy,
        },
    }

    return {
        "experiment": "convergence_comparison",
        "target_k": target_k,
        "state_dim": state_dim,
        "n_steps": n_steps,
        "comparison": comparison,
    }


def run_phase_diagram_experiment(
    target_ks: Sequence[int] = (2, 3, 4),
    state_dims: Sequence[int] = (1, 2, 3, 4, 5, 6),
    n_steps: int = 2000,
    seed: int = 42,
    distribution: SpikeCountMixtureDistributionSpec = DEFAULT_DISTRIBUTION,
) -> Dict:
    """Experiment 3: phase diagram — learned sketch accuracy across (m, k)."""
    rows = []
    for k in target_ks:
        for m in state_dims:
            print(f"  Phase diagram: m={m}, k={k}")
            config = LearnedSketchTrainingConfig(
                model=LearnedSketchModelConfig(state_dim=m, target_k=k),
                train=TrainConfig(batch_size=64, steps=n_steps),
                run=RunConfig(seed=seed + k * 1000 + m * 100),
            )
            result = run_learning_curve(config, distribution)
            rows.append({
                "target_k": k,
                "state_dim": m,
                "relation": (
                    "unsupported(m<k)" if m < k
                    else ("exact(m=k)" if m == k else "oversupported(m>k)")
                ),
                "supports_target": m >= k,
                "learned_accuracy": result.final_threshold_accuracy,
                "learned_root_oracle_mse": result.final_root_oracle_mse,
                "learned_l1_leaf_error": result.final_l1_leaf_error,
                "learned_l2_merge_error": result.final_l2_merge_error,
                "learned_l3_idemp_error": result.final_l3_idemp_error,
                "hand_designed_accuracy": result.hand_designed_accuracy,
            })

    return {
        "experiment": "phase_diagram",
        "target_ks": list(target_ks),
        "state_dims": list(state_dims),
        "n_steps": n_steps,
        "rows": rows,
    }


def run_audit_budget_experiment(
    target_k: int = 4,
    state_dim: int = 4,
    audit_budgets: Sequence[int] = (1, 3, 7, 15),
    n_steps: int = 2000,
    seed: int = 42,
    distribution: SpikeCountMixtureDistributionSpec = DEFAULT_DISTRIBUTION,
) -> Dict:
    """Experiment 4: convergence speed vs audit budget."""
    results = []
    for n_audit in audit_budgets:
        print(f"  Audit budget: n_audit={n_audit}, m={state_dim}, k={target_k}")
        config = LearnedSketchTrainingConfig(
            model=LearnedSketchModelConfig(state_dim=state_dim, target_k=target_k),
            train=TrainConfig(batch_size=64, steps=n_steps),
            run=RunConfig(seed=seed + n_audit * 100),
            objective=LearnedSketchObjectiveConfig(n_audit=n_audit),
        )
        result = run_learning_curve(config, distribution)
        results.append({
            "n_audit": n_audit,
            "metrics": result.metrics,
            "final_root_oracle_mse": result.final_root_oracle_mse,
            "final_threshold_accuracy": result.final_threshold_accuracy,
            "final_l1_leaf_error": result.final_l1_leaf_error,
            "final_l2_merge_error": result.final_l2_merge_error,
            "final_l3_idemp_error": result.final_l3_idemp_error,
        })

    return {
        "experiment": "audit_budget",
        "target_k": target_k,
        "state_dim": state_dim,
        "audit_budgets": list(audit_budgets),
        "n_steps": n_steps,
        "results": results,
    }
